"""
使用整数线性规划（ILP）进行 chiplet 布局优化。

主要特性
--------
1. **相邻约束**：有连边的 chiplet 必须水平或垂直相邻（紧靠），并且共享边长度不少于给定下界。
2. **旋转约束**：每个 chiplet 允许 0°/90° 旋转，由二进制变量 ``r_k`` 控制宽高交换。
3. **非重叠约束**：任意两块 chiplet 之间不能重叠。
4. **外接方框约束**：显式构造覆盖所有 chiplet 的外接矩形，并对其宽高建立线性约束。
5. **多目标优化**：目标函数为

   ``β1 * wirelength + β2 * t``

   其中 ``wirelength`` 是所有连边中心点间的曼哈顿距离之和，
   ``t`` 是通过 AM–GM 凸近似得到的“面积代理”变量，用来近似外接矩形面积。

实现依赖 PuLP（建模）与 GLPK / CBC（求解器）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pulp

try:
    from tool import ChipletNode, build_random_chiplet_graph, draw_chiplet_diagram
except ImportError:
    # 如果作为模块导入，使用相对导入
    from .tool import ChipletNode, build_random_chiplet_graph, draw_chiplet_diagram


@dataclass
class ILPPlacementResult:
    """ILP求解结果"""

    layout: Dict[str, Tuple[float, float]]  # name -> (x, y)
    rotations: Dict[str, bool]  # name -> 是否旋转
    objective_value: float
    status: str
    solve_time: float
    bounding_box: Tuple[float, float]  # (W, H) 边界框尺寸


@dataclass
class ILPModelContext:
    """
    ILP 模型上下文。

    - `prob`  : 已经构建好的 PuLP 模型（包含变量、约束和目标函数，但可以继续加约束）
    - `x, y`  : 每个 chiplet 左下角坐标变量（用于排除解等约束）
    - `r`     : 每个 chiplet 的旋转变量
    - `z1, z2`: 每对有连边的 chiplet 的"相邻方式"变量（水平/垂直）
    - `z1L, z1R, z2D, z2U`: 每对有连边的 chiplet 的相对方向变量（左、右、下、上）
    - `connected_pairs` : 有连边的 (i, j) 索引列表（i < j）
    - `bbox_w, bbox_h` : 外接方框宽和高对应的变量
    - `W, H`  : 外接边界框的上界尺寸（建模阶段确定）
    - `grid_size` : 网格大小（如果使用网格化，否则为None）
    - `fixed_chiplet_idx` : 固定位置的chiplet索引（如果使用固定中心约束，否则为None）
    """

    prob: pulp.LpProblem
    nodes: List[ChipletNode]
    edges: List[Tuple[str, str]]

    x: Dict[int, pulp.LpVariable]
    y: Dict[int, pulp.LpVariable]
    r: Dict[int, pulp.LpVariable]
    z1: Dict[Tuple[int, int], pulp.LpVariable]
    z2: Dict[Tuple[int, int], pulp.LpVariable]
    z1L: Dict[Tuple[int, int], pulp.LpVariable]
    z1R: Dict[Tuple[int, int], pulp.LpVariable]
    z2D: Dict[Tuple[int, int], pulp.LpVariable]
    z2U: Dict[Tuple[int, int], pulp.LpVariable]
    connected_pairs: List[Tuple[int, int]]

    bbox_w: pulp.LpVariable
    bbox_h: pulp.LpVariable

    W: float
    H: float
    grid_size: Optional[float] = None
    fixed_chiplet_idx: Optional[int] = None


def build_placement_ilp_model(
    nodes: List[ChipletNode],
    edges: List[Tuple[str, str]],
    W: Optional[float] = None,
    H: Optional[float] = None,
    time_limit: int = 300,
    verbose: bool = True,
    min_shared_length: float = 0.0,
    minimize_bbox_area: bool = True,
    distance_weight: float = 1.0,
    area_weight: float = 0.1,
) -> ILPModelContext:
    """
    使用 ILP 求解 chiplet 布局（相邻 + 非重叠 + 外接方框 + 多目标）。

    约束概要
    --------
    - **相邻约束**：每一条连边对应的两个 chiplet 必须相邻：
      - 要么水平相邻（左右紧靠），要么垂直相邻（上下紧靠），二选一；
      - 相邻方向上的共享边长度必须大于等于 ``min_shared_length``。
    - **旋转约束**：每个 chiplet 通过二进制变量 ``r_k`` 决定是否旋转 90°（宽高交换）。
    - **非重叠约束**：任意两个 chiplet 在平面上不能有重叠区域。
    - **外接方框约束**：显式构造一个覆盖所有 chiplet 的外接矩形，并引入宽高 ``bbox_w, bbox_h``。

    目标函数
    --------
    - 线长（wirelength）：所有连边对应中心点之间曼哈顿距离的总和。
    - 面积代理 ``t``：利用 AM–GM 不等式
      ``t ≤ (bbox_w + bbox_h)/2``, ``bbox_w ≤ K·t``, ``bbox_h ≤ K·t`` 构造的凸近似，
      在最小化 ``t`` 时同时推动 ``bbox_w`` 和 ``bbox_h`` 变小。
    - 最终目标：``distance_weight * wirelength + area_weight * t``。

    参数
    ----
    nodes:
        从 JSON 读入的 chiplet 节点列表。
    edges:
        节点之间的连边列表（这些对会被强制“相邻”）。
    W, H:
        外接芯片区域的最大宽高；若为 ``None`` 则根据 chiplet 总面积和尺寸自动估算。
    time_limit:
        求解时间上限（秒）。
    verbose:
        若为 True，则打印建模与求解过程中的关键信息。
    min_shared_length:
        相邻 chiplet 之间共享边长度的下界（在相邻方向上）。
    minimize_bbox_area:
        是否在目标函数中加入面积代理项 ``t``。
    distance_weight:
        线长项的权重（对应公式中的 β₁）。
    area_weight:
        面积代理 ``t`` 的权重（对应公式中的 β₂）。

    返回
    ----
    ILPModelContext
        已经构建好的 ILP 模型上下文（尚未求解），
        可在外部继续添加约束（例如排除解）后再调用求解函数。
    """
    n = len(nodes)
    name_to_idx = {node.name: i for i, node in enumerate(nodes)}
    
    # ============ 步骤1: 读取已知条件（固定不变的量）============
    # 获取原始矩形尺寸（未旋转时）
    w_orig = {}  # w_k^o: 模块 k 的原始宽度
    h_orig = {}  # h_k^o: 模块 k 的原始高度
    for i, node in enumerate(nodes):
        w_orig[i] = float(node.dimensions.get("x", 0.0))
        h_orig[i] = float(node.dimensions.get("y", 0.0))
    
    # 找到所有有边连接的模块对（这些对必须相邻）
    connected_pairs = []
    for src_name, dst_name in edges:
        if src_name in name_to_idx and dst_name in name_to_idx:
            i = name_to_idx[src_name]
            j = name_to_idx[dst_name]
            if i != j:
                # 确保 i < j（避免重复）
                if i > j:
                    i, j = j, i
                connected_pairs.append((i, j))
    connected_pairs = list(set(connected_pairs))  # 去重
    
    if verbose:
        print(f"问题规模: {n} 个模块, {len(connected_pairs)} 对有连接的模块对（必须相邻）")
        print(f"模块尺寸:")
        for i, node in enumerate(nodes):
            print(f"  模块 {i} ({node.name}): 宽度={w_orig[i]:.2f}, 高度={h_orig[i]:.2f}")
    
    # 估算芯片边界框尺寸 ChipW × ChipH（如果不指定）
    if W is None or H is None:
        max_w = max(w_orig.values())
        max_h = max(h_orig.values())
        total_area = sum(w_orig[i] * h_orig[i] for i in range(n))
        # 估算：假设紧凑排列，总面积 * 2 作为边界框面积
        estimated_side = math.ceil(math.sqrt(total_area * 2))
        if W is None:
            W = max(estimated_side, max_w * 10)
        if H is None:
            H = max(estimated_side, max_h * 10)
    
    # 大 M 常数：一个足够大的数（通常取 2 倍芯片最大尺寸）
    M = max(W, H) * 10
    
    if verbose:
        print(f"芯片边界框尺寸: ChipW = {W:.2f}, ChipH = {H:.2f}")
        print(f"大 M 常数: M = {M:.2f}")
        print(f"相邻chiplet共享边最小长度: min_shared_length = {min_shared_length:.2f}")
    
    # ============ 步骤2: 创建ILP问题 ============
    prob = pulp.LpProblem("ChipletPlacement_Adjacent", pulp.LpMinimize)
    
    # ============ 步骤3: 定义变量（需要求解的量）============
    
    # 3.1 连续变量：模块的位置坐标
    # x_k, y_k: 模块 k 的左下角 x、y 坐标
    x = {}
    y = {}
    for k in range(n):
        # 边界约束：模块必须在 [0, W] × [0, H] 范围内
        # 考虑旋转后的最大尺寸
        max_dim_k = max(w_orig[k], h_orig[k])
        x[k] = pulp.LpVariable(f"x_{k}", lowBound=0, upBound=W - min(w_orig[k], h_orig[k]))
        y[k] = pulp.LpVariable(f"y_{k}", lowBound=0, upBound=H - min(w_orig[k], h_orig[k]))
    
    # 3.2 二进制变量：旋转变量
    # r_k = 1: 模块 k 旋转90度；r_k = 0: 不旋转
    r = {}
    for k in range(n):
        r[k] = pulp.LpVariable(f"r_{k}", cat='Binary')
    
    # 3.3 连续变量：实际宽度和高度（取决于旋转状态）
    # w_k = r_k * h_k^o + (1 - r_k) * w_k^o
    # h_k = r_k * w_k^o + (1 - r_k) * h_k^o
    w = {}
    h = {}
    for k in range(n):
        w_min = min(w_orig[k], h_orig[k])
        w_max = max(w_orig[k], h_orig[k])
        w[k] = pulp.LpVariable(f"w_{k}", lowBound=w_min, upBound=w_max)
        h[k] = pulp.LpVariable(f"h_{k}", lowBound=w_min, upBound=w_max)
    
    # 3.4 辅助变量：中心坐标（用于计算距离）
    cx = {}
    cy = {}
    for k in range(n):
        cx[k] = pulp.LpVariable(f"cx_{k}", lowBound=0, upBound=W)
        cy[k] = pulp.LpVariable(f"cy_{k}", lowBound=0, upBound=H)
    
    # 3.5 二进制变量：控制相邻方式（对于每对有连接的模块对 (i, j)）
    # z1[i,j] = 1: 模块 i 和 j 水平相邻（左右靠）；z1[i,j] = 0: 不水平相邻
    # z2[i,j] = 1: 模块 i 和 j 垂直相邻（上下靠）；z2[i,j] = 0: 不垂直相邻
    # z1L[i,j] = 1: 模块 i 在 j 左边（水平相邻时）
    # z1R[i,j] = 1: 模块 i 在 j 右边（水平相邻时）
    # z2D[i,j] = 1: 模块 i 在 j 下边（垂直相邻时）
    # z2U[i,j] = 1: 模块 i 在 j 上边（垂直相邻时）
    z1 = {}
    z2 = {}
    z1L = {}
    z1R = {}
    z2D = {}
    z2U = {}
    
    # 3.6 辅助变量：共享边长度（用于约束共享边的最小长度）
    # shared_y[i,j]: 水平相邻时，垂直方向的共享边长度
    # shared_x[i,j]: 垂直相邻时，水平方向的共享边长度
    shared_y = {}
    shared_x = {}
    
    for i, j in connected_pairs:
        z1[(i, j)] = pulp.LpVariable(f"z1_{i}_{j}", cat="Binary")
        z2[(i, j)] = pulp.LpVariable(f"z2_{i}_{j}", cat="Binary")
        z1L[(i, j)] = pulp.LpVariable(f"z1L_{i}_{j}", cat="Binary")
        z1R[(i, j)] = pulp.LpVariable(f"z1R_{i}_{j}", cat="Binary")
        z2D[(i, j)] = pulp.LpVariable(f"z2D_{i}_{j}", cat="Binary")
        z2U[(i, j)] = pulp.LpVariable(f"z2U_{i}_{j}", cat="Binary")
        
        # 共享边长度变量（非负）
        max_shared_y = min(h_orig[i], h_orig[j], max(h_orig.values()))  # 最大可能的垂直共享长度
        max_shared_x = min(w_orig[i], w_orig[j], max(w_orig.values()))  # 最大可能的水平共享长度
        shared_y[(i, j)] = pulp.LpVariable(f"shared_y_{i}_{j}", lowBound=0, upBound=max_shared_y)
        shared_x[(i, j)] = pulp.LpVariable(f"shared_x_{i}_{j}", lowBound=0, upBound=max_shared_x)
    
    # ============ 步骤4: 定义约束（必须满足的条件）============
    
    # 4.1 旋转约束：实际宽度和高度取决于旋转状态
    # w_k = w_k^o + r_k * (h_k^o - w_k^o)
    # h_k = h_k^o + r_k * (w_k^o - h_k^o)
    for k in range(n):
        prob += w[k] == w_orig[k] + r[k] * (h_orig[k] - w_orig[k]), f"width_rotation_{k}"
        prob += h[k] == h_orig[k] + r[k] * (w_orig[k] - h_orig[k]), f"height_rotation_{k}"
    
    # 4.2 中心坐标定义
    for k in range(n):
        prob += cx[k] == x[k] + w[k] / 2.0, f"cx_def_{k}"
        prob += cy[k] == y[k] + h[k] / 2.0, f"cy_def_{k}"
    
    # 4.3 相邻约束：对于每对有连接的模块对 (i, j)，它们必须相邻
    for i, j in connected_pairs:
        # 规则1: 必须相邻，且只能选一种方式
        # z1[i,j] + z2[i,j] = 1（要么水平相邻，要么垂直相邻，不能不相邻，也不能两种都选）
        prob += z1[(i, j)] + z2[(i, j)] == 1, f"must_adjacent_{i}_{j}"
        
        # 规则2: 如果水平相邻，要么 i 在左，要么 i 在右
        # z1L[i,j] + z1R[i,j] = z1[i,j]
        prob += z1L[(i, j)] + z1R[(i, j)] == z1[(i, j)], f"horizontal_direction_{i}_{j}"
        
        # 规则3: 如果垂直相邻，要么 i 在下，要么 i 在上
        # z2D[i,j] + z2U[i,j] = z2[i,j]
        prob += z2D[(i, j)] + z2U[(i, j)] == z2[(i, j)], f"vertical_direction_{i}_{j}"
        
        # 规则4: 水平相邻的具体约束（z1[i,j] = 1 时才生效）
        # (4.4.1) 边界对齐（左右紧靠，无间隙）
        # 如果 i 在左（z1L[i,j] = 1）：x_i + w_i = x_j（i 的右边 = j 的左边）
        # 用大 M 实现：x_i + w_i - x_j ≤ M*(1 - z1L[i,j])
        prob += x[i] + w[i] - x[j] <= M * (1 - z1L[(i, j)]), f"horizontal_left_{i}_{j}"
        prob += x[i] + w[i] - x[j] >= -M * (1 - z1L[(i, j)]), f"horizontal_left_eq_{i}_{j}"
        
        # 如果 i 在右（z1R[i,j] = 1）：x_j + w_j = x_i（j 的右边 = i 的左边）
        # 用大 M 实现：x_j + w_j - x_i ≤ M*(1 - z1R[i,j])
        prob += x[j] + w[j] - x[i] <= M * (1 - z1R[(i, j)]), f"horizontal_right_{i}_{j}"
        prob += x[j] + w[j] - x[i] >= -M * (1 - z1R[(i, j)]), f"horizontal_right_eq_{i}_{j}"
        
        # (4.4.2) 垂直方向必须重叠，且共享边长度 >= min_shared_length
        # 共享边长度 = min(y_i + h_i, y_j + h_j) - max(y_i, y_j)
        # 约束：y_i ≤ y_j + h_j（i 的底部不能低于 j 的顶部）
        #      y_j ≤ y_i + h_i（j 的底部不能低于 i 的顶部）
        # 用大 M 实现：只在 z1[i,j] = 1 时生效
        prob += y[i] - (y[j] + h[j]) <= M * (1 - z1[(i, j)]), f"horizontal_overlap_y1_{i}_{j}"
        prob += y[j] - (y[i] + h[i]) <= M * (1 - z1[(i, j)]), f"horizontal_overlap_y2_{i}_{j}"
        
        # 共享边长度约束：shared_y[i,j] = min(y_i + h_i, y_j + h_j) - max(y_i, y_j)
        # 使用线性化：shared_y[i,j] <= min(y_i + h_i, y_j + h_j) - max(y_i, y_j)
        # 简化：shared_y[i,j] <= (y_i + h_i) - max(y_i, y_j) 且 shared_y[i,j] <= (y_j + h_j) - max(y_i, y_j)
        # 进一步简化：shared_y[i,j] <= min(h_i, h_j) + min(y_i - y_j, y_j - y_i)
        # 更简单的方法：shared_y[i,j] <= min(y_i + h_i, y_j + h_j) - y_i 且 shared_y[i,j] <= min(y_i + h_i, y_j + h_j) - y_j
        # 使用辅助变量来线性化 min/max
        # shared_y[i,j] <= (y_i + h_i) - y_j 且 shared_y[i,j] <= (y_j + h_j) - y_i
        # 同时 shared_y[i,j] >= 0
        # 当 z1[i,j] = 1 时，这些约束生效
        prob += shared_y[(i, j)] <= (y[i] + h[i]) - y[j] + M * (1 - z1[(i, j)]), \
               f"shared_y_ub1_{i}_{j}"
        prob += shared_y[(i, j)] <= (y[j] + h[j]) - y[i] + M * (1 - z1[(i, j)]), \
               f"shared_y_ub2_{i}_{j}"
        prob += shared_y[(i, j)] >= min_shared_length - M * (1 - z1[(i, j)]), \
               f"shared_y_min_{i}_{j}"
        # 当 z1[i,j] = 0 时，shared_y[i,j] = 0
        prob += shared_y[(i, j)] <= M * z1[(i, j)], f"shared_y_zero_{i}_{j}"
        
        # 规则5: 垂直相邻的具体约束（z2[i,j] = 1 时才生效）
        # (4.5.1) 边界对齐（上下紧靠，无间隙）
        # 如果 i 在下（z2D[i,j] = 1）：y_i + h_i = y_j（i 的顶部 = j 的底部）
        # 用大 M 实现：y_i + h_i - y_j ≤ M*(1 - z2D[i,j])
        prob += y[i] + h[i] - y[j] <= M * (1 - z2D[(i, j)]), f"vertical_down_{i}_{j}"
        prob += y[i] + h[i] - y[j] >= -M * (1 - z2D[(i, j)]), f"vertical_down_eq_{i}_{j}"
        
        # 如果 i 在上（z2U[i,j] = 1）：y_j + h_j = y_i（j 的顶部 = i 的底部）
        # 用大 M 实现：y_j + h_j - y_i ≤ M*(1 - z2U[i,j])
        prob += y[j] + h[j] - y[i] <= M * (1 - z2U[(i, j)]), f"vertical_up_{i}_{j}"
        prob += y[j] + h[j] - y[i] >= -M * (1 - z2U[(i, j)]), f"vertical_up_eq_{i}_{j}"
        
        # (4.5.2) 水平方向必须重叠，且共享边长度 >= min_shared_length
        # 共享边长度 = min(x_i + w_i, x_j + w_j) - max(x_i, x_j)
        # 约束：x_i ≤ x_j + w_j（i 的左边不能超过 j 的右边）
        #      x_j ≤ x_i + w_i（j 的左边不能超过 i 的右边）
        # 用大 M 实现：只在 z2[i,j] = 1 时生效
        prob += x[i] - (x[j] + w[j]) <= M * (1 - z2[(i, j)]), f"vertical_overlap_x1_{i}_{j}"
        prob += x[j] - (x[i] + w[i]) <= M * (1 - z2[(i, j)]), f"vertical_overlap_x2_{i}_{j}"
        
        # 共享边长度约束：shared_x[i,j] = min(x_i + w_i, x_j + w_j) - max(x_i, x_j)
        # 使用线性化：shared_x[i,j] <= min(x_i + w_i, x_j + w_j) - max(x_i, x_j)
        prob += shared_x[(i, j)] <= (x[i] + w[i]) - x[j] + M * (1 - z2[(i, j)]), \
               f"shared_x_ub1_{i}_{j}"
        prob += shared_x[(i, j)] <= (x[j] + w[j]) - x[i] + M * (1 - z2[(i, j)]), \
               f"shared_x_ub2_{i}_{j}"
        prob += shared_x[(i, j)] >= min_shared_length - M * (1 - z2[(i, j)]), \
               f"shared_x_min_{i}_{j}"
        # 当 z2[i,j] = 0 时，shared_x[i,j] = 0
        prob += shared_x[(i, j)] <= M * z2[(i, j)], f"shared_x_zero_{i}_{j}"
    
    # 4.6 边界约束：所有模块必须在芯片边界内
    for k in range(n):
        # x_k + w_k ≤ ChipW（模块 k 的右边不超过芯片右边界）
        prob += x[k] + w[k] <= W, f"boundary_x_{k}"
        # y_k + h_k ≤ ChipH（模块 k 的顶部不超过芯片上边界）
        prob += y[k] + h[k] <= H, f"boundary_y_{k}"
        # x_k ≥ 0, y_k ≥ 0（坐标不能为负，已在变量定义中设置 lowBound=0）
    
    # 4.7 非重叠约束：所有模块对都不能重叠（即使没有连接）
    # 对于每对模块 (i, j)，其中 i < j，它们必须满足以下条件之一：
    # 1. i 在 j 的左边：x_i + w_i <= x_j
    # 2. i 在 j 的右边：x_j + w_j <= x_i
    # 3. i 在 j 的下边：y_i + h_i <= y_j
    # 4. i 在 j 的上边：y_j + h_j <= y_i
    # 使用二进制变量 p_left, p_right, p_down, p_up 来表示这四种情况
    
    # 定义非重叠约束的二进制变量（对于所有模块对，不仅仅是连接的）
    p_left = {}
    p_right = {}
    p_down = {}
    p_up = {}
    
    all_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            all_pairs.append((i, j))
            p_left[(i, j)] = pulp.LpVariable(f"p_left_{i}_{j}", cat='Binary')
            p_right[(i, j)] = pulp.LpVariable(f"p_right_{i}_{j}", cat='Binary')
            p_down[(i, j)] = pulp.LpVariable(f"p_down_{i}_{j}", cat='Binary')
            p_up[(i, j)] = pulp.LpVariable(f"p_up_{i}_{j}", cat='Binary')
    
    # 对于每对模块 (i, j)，至少选择一种非重叠关系
    for i, j in all_pairs:
        # 至少选择一种非重叠关系
        prob += p_left[(i, j)] + p_right[(i, j)] + p_down[(i, j)] + p_up[(i, j)] == 1, \
               f"non_overlap_any_{i}_{j}"
        
        # 情况1: i 在 j 的左边（x_i + w_i <= x_j）
        # 用大 M 实现：x_i + w_i - x_j <= M * (1 - p_left[i,j])
        prob += x[i] + w[i] - x[j] <= M * (1 - p_left[(i, j)]), \
               f"non_overlap_left_{i}_{j}"
        
        # 情况2: i 在 j 的右边（x_j + w_j <= x_i）
        # 用大 M 实现：x_j + w_j - x_i <= M * (1 - p_right[i,j])
        prob += x[j] + w[j] - x[i] <= M * (1 - p_right[(i, j)]), \
               f"non_overlap_right_{i}_{j}"
        
        # 情况3: i 在 j 的下边（y_i + h_i <= y_j）
        # 用大 M 实现：y_i + h_i - y_j <= M * (1 - p_down[i,j])
        prob += y[i] + h[i] - y[j] <= M * (1 - p_down[(i, j)]), \
               f"non_overlap_down_{i}_{j}"
        
        # 情况4: i 在 j 的上边（y_j + h_j <= y_i）
        # 用大 M 实现：y_j + h_j - y_i <= M * (1 - p_up[i,j])
        prob += y[j] + h[j] - y[i] <= M * (1 - p_up[(i, j)]), \
               f"non_overlap_up_{i}_{j}"
    
    if verbose:
        print(f"非重叠约束: {len(all_pairs)} 对模块对（所有模块对）")
    
    # ============ 步骤4.8: 外接方框约束 ============
    # 定义所有chiplet的外接矩形（bounding box）
    # bbox_min_x, bbox_max_x: 外接矩形的最小和最大x坐标
    # bbox_min_y, bbox_max_y: 外接矩形的最小和最大y坐标
    # bbox_w, bbox_h: 外接矩形的宽度和高度
    bbox_min_x = pulp.LpVariable("bbox_min_x", lowBound=0, upBound=W)
    bbox_max_x = pulp.LpVariable("bbox_max_x", lowBound=0, upBound=W)
    bbox_min_y = pulp.LpVariable("bbox_min_y", lowBound=0, upBound=H)
    bbox_max_y = pulp.LpVariable("bbox_max_y", lowBound=0, upBound=H)
    bbox_w = pulp.LpVariable("bbox_w", lowBound=0, upBound=W)
    bbox_h = pulp.LpVariable("bbox_h", lowBound=0, upBound=H)
    
    # 约束：每个chiplet的边界都在外接矩形内
    for k in range(n):
        # x[k] >= bbox_min_x（chiplet k 的左边界 >= 外接矩形最小x）
        prob += x[k] >= bbox_min_x, f"bbox_min_x_{k}"
        # x[k] + w[k] <= bbox_max_x（chiplet k 的右边界 <= 外接矩形最大x）
        prob += x[k] + w[k] <= bbox_max_x, f"bbox_max_x_{k}"
        # y[k] >= bbox_min_y（chiplet k 的底边界 >= 外接矩形最小y）
        prob += y[k] >= bbox_min_y, f"bbox_min_y_{k}"
        # y[k] + h[k] <= bbox_max_y（chiplet k 的顶边界 <= 外接矩形最大y）
        prob += y[k] + h[k] <= bbox_max_y, f"bbox_max_y_{k}"
    
    # 约束：外接矩形的宽度和高度
    prob += bbox_w == bbox_max_x - bbox_min_x, "bbox_width"
    prob += bbox_h == bbox_max_y - bbox_min_y, "bbox_height"

    # 使用 AM-GM 不等式的凸近似来构造“面积代理”变量 t
    # 目标是用线性约束来逼近 bbox_w * bbox_h，从而在目标函数中最小化 t
    #
    # 参考形式：
    #   t ≤ (bbox_w + bbox_h) / 2
    #   bbox_w ≤ K * t
    #   bbox_h ≤ K * t
    # 在最小化 t 的前提下，这会同时压缩 bbox_w 和 bbox_h，相当于间接减小面积。
    #
    # 这里 K 取一个与边界框尺度同一量级的常数，避免约束过松或不稳定。
    K_bbox = max(W, H)
    t = pulp.LpVariable("bbox_area_proxy_t", lowBound=0)

    # t ≤ (bbox_w + bbox_h) / 2
    prob += t <= (bbox_w + bbox_h) / 2.0, "bbox_t_le_mean_wh"
    # bbox_w ≤ K * t, bbox_h ≤ K * t
    prob += bbox_w <= K_bbox * t, "bbox_w_le_Kt"
    prob += bbox_h <= K_bbox * t, "bbox_h_le_Kt"

    if verbose:
        print("外接方框约束: 已添加（使用 AM-GM 凸近似的面积代理 t）")
    
    # ============ 步骤5: 定义目标函数 ============
    # 多目标优化：
    # 1. 最小化有连接关系的模块之间的中心距离之和
    # 2. 最小化外接方框面积（使用 w + h 作为代理，或使用分段线性化）
    
    # 目标1: 最小化有连接关系的模块之间的中心距离之和
    # 使用 L1 距离（曼哈顿距离）作为线性近似
    distance_terms = []
    for i, j in connected_pairs:
        # 距离辅助变量（用于线性化绝对值）
        dx_abs = pulp.LpVariable(f"dx_abs_{i}_{j}", lowBound=0)
        dy_abs = pulp.LpVariable(f"dy_abs_{i}_{j}", lowBound=0)
        
        # 绝对值线性化：dx_abs >= |cx[i] - cx[j]|
        prob += dx_abs >= cx[i] - cx[j], f"dx_abs_{i}_{j}_pos"
        prob += dx_abs >= cx[j] - cx[i], f"dx_abs_{i}_{j}_neg"
        # dy_abs >= |cy[i] - cy[j]|
        prob += dy_abs >= cy[i] - cy[j], f"dy_abs_{i}_{j}_pos"
        prob += dy_abs >= cy[j] - cy[i], f"dy_abs_{i}_{j}_neg"
        
        distance_terms.append(dx_abs + dy_abs)
    
    # 目标2: 最小化外接方框面积（使用凸近似的面积代理 t）
    # 最终目标形式：β1 * 线长 + β2 * t
    if minimize_bbox_area:
        prob += (
            distance_weight * pulp.lpSum(distance_terms) + area_weight * t,
            "total_objective",
        )
        if verbose:
            print(
                f"目标函数: β1*线长 + β2*t（面积凸近似），其中 "
                f"β1={distance_weight:.2f}, β2={area_weight:.2f}"
            )
    else:
        prob += pulp.lpSum(distance_terms), "total_connection_distance"
        if verbose:
            print("目标函数: 只最小化线长（不考虑外接方框大小）")

    # 返回尚未求解的模型上下文
    return ILPModelContext(
        prob=prob,
        nodes=nodes,
        edges=edges,
        x=x,
        y=y,
        r=r,
        z1=z1,
        z2=z2,
        z1L=z1L,
        z1R=z1R,
        z2D=z2D,
        z2U=z2U,
        connected_pairs=connected_pairs,
        bbox_w=bbox_w,
        bbox_h=bbox_h,
        W=W,
        H=H,
    )


def solve_placement_ilp_from_model(
    ctx: ILPModelContext,
    time_limit: int = 300,
    verbose: bool = True,
) -> ILPPlacementResult:
    """
    在已有 ILPModelContext 上调用求解器并抽取解。

    可以在多轮求解之间往 ctx.prob 上继续添加约束（例如排除解约束）。
    """
    import time

    prob = ctx.prob
    nodes = ctx.nodes
    x, y, r = ctx.x, ctx.y, ctx.r
    W, H = ctx.W, ctx.H

    start_time = time.time()

    if verbose:
        print("\n开始求解ILP问题...")
        print(f"变量数量: {prob.numVariables()}")
        print(f"约束数量: {prob.numConstraints()}")

    # 尝试使用可用的求解器
    solver = None
    solver_name = "默认求解器"

    # 首先尝试 GLPK
    try:
        import subprocess

        result = subprocess.run(
            ["glpsol", "--version"],
            capture_output=True,
            timeout=2,
        )
        if result.returncode == 0:
            solver = pulp.getSolver("GLPK_CMD", timeLimit=time_limit, msg=verbose)
            solver_name = "GLPK"
            if verbose:
                print(f"使用求解器: {solver_name}")
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        # GLPK 不可用，尝试 CBC
        try:
            solver = pulp.getSolver("PULP_CBC_CMD", timeLimit=time_limit, msg=verbose)
            solver_name = "CBC"
            if verbose:
                print(f"使用求解器: {solver_name}")
        except Exception:
            # 如果 CBC 也不可用，使用默认求解器
            if verbose:
                print("警告: GLPK 和 CBC 都不可用，使用默认求解器")
            solver = None
            solver_name = "默认求解器"

    try:
        status = prob.solve(solver)
        solve_time = time.time() - start_time

        if verbose:
            print(f"\n求解状态: {pulp.LpStatus[status]}")
            print(f"求解时间: {solve_time:.2f} 秒")
            if status == pulp.LpStatusOptimal:
                print(f"目标函数值: {pulp.value(prob.objective):.2f}")

        # 提取解
        layout: Dict[str, Tuple[float, float]] = {}
        rotations: Dict[str, bool] = {}
        for k, node in enumerate(nodes):
            if status == pulp.LpStatusOptimal:
                x_val = pulp.value(x[k])
                y_val = pulp.value(y[k])
                r_val = pulp.value(r[k])
                layout[node.name] = (
                    x_val if x_val is not None else 0.0,
                    y_val if y_val is not None else 0.0,
                )
                rotations[node.name] = bool(r_val > 0.5) if r_val is not None else False
            else:
                layout[node.name] = (0.0, 0.0)
                rotations[node.name] = False

        obj_value = (
            pulp.value(prob.objective) if status == pulp.LpStatusOptimal else float("inf")
        )

        # 使用求解得到的 bbox_w / bbox_h 作为返回的边界框尺寸
        try:
            bw_val = pulp.value(ctx.bbox_w)
            bh_val = pulp.value(ctx.bbox_h)
        except Exception:
            bw_val, bh_val = None, None

        bbox_tuple = (
            float(bw_val) if bw_val is not None else 0.0,
            float(bh_val) if bh_val is not None else 0.0,
        )

        return ILPPlacementResult(
            layout=layout,
            rotations=rotations,
            objective_value=obj_value,
            status=pulp.LpStatus[status],
            solve_time=solve_time,
            bounding_box=bbox_tuple,
        )

    except Exception as e:
        solve_time = time.time() - start_time
        if verbose:
            print(f"\n求解出错: {e}")
            import traceback

            traceback.print_exc()

        # 返回空解
        layout = {node.name: (0.0, 0.0) for node in nodes}
        rotations = {node.name: False for node in nodes}
        return ILPPlacementResult(
            layout=layout,
            rotations=rotations,
            objective_value=float("inf"),
            status="Error",
            solve_time=solve_time,
            bounding_box=(W if W else 100.0, H if H else 100.0),
        )


def solve_placement_ilp(
    nodes: List[ChipletNode],
    edges: List[Tuple[str, str]],
    W: Optional[float] = None,
    H: Optional[float] = None,
    time_limit: int = 300,
    verbose: bool = True,
    min_shared_length: float = 0.0,
    minimize_bbox_area: bool = True,
    distance_weight: float = 1.0,
    area_weight: float = 0.1,
) -> ILPPlacementResult:
    """
    兼容旧接口的一站式求解函数：
    内部先调用 :func:`build_placement_ilp_model` 构建模型，
    然后用 :func:`solve_placement_ilp_from_model` 进行一次求解。
    """

    ctx = build_placement_ilp_model(
        nodes=nodes,
        edges=edges,
        W=W,
        H=H,
        verbose=verbose,
        min_shared_length=min_shared_length,
        minimize_bbox_area=minimize_bbox_area,
        distance_weight=distance_weight,
        area_weight=area_weight,
    )

    return solve_placement_ilp_from_model(
        ctx,
        time_limit=time_limit,
        verbose=verbose,
    )


def build_placement_ilp_model_grid(
    nodes: List[ChipletNode],
    edges: List[Tuple[str, str]],
    grid_size: float,
    W: Optional[float] = None,
    H: Optional[float] = None,
    time_limit: int = 300,
    verbose: bool = True,
    min_shared_length: float = 0.0,
    minimize_bbox_area: bool = True,
    distance_weight: float = 1.0,
    area_weight: float = 0.1,
    fixed_chiplet_idx: Optional[int] = None,  # 固定位置的chiplet索引（中心固定在方框中心）
) -> ILPModelContext:
    """
    使用网格化ILP求解chiplet布局。
    
    与build_placement_ilp_model的主要区别：
    1. 坐标变量为整数（grid索引）
    2. 有链接关系的chiplet之间距离不能超过一个grid
    3. 共享边长不超过一个grid的共享范围，且不能小于min_shared_length
    4. 可以固定一个chiplet的中心位置在方框的中心点
    
    参数
    ----
    grid_size: float
        网格大小（实际单位）
    fixed_chiplet_idx: Optional[int]
        固定位置的chiplet索引，如果提供，则该chiplet的中心固定在方框中心
    其他参数同build_placement_ilp_model
    """
    import math
    
    n = len(nodes)
    name_to_idx = {node.name: i for i, node in enumerate(nodes)}
    
    # ============ 步骤1: 读取已知条件 ============
    w_orig = {}
    h_orig = {}
    for i, node in enumerate(nodes):
        w_orig[i] = float(node.dimensions.get("x", 0.0))
        h_orig[i] = float(node.dimensions.get("y", 0.0))
    
    # 找到所有有边连接的模块对
    connected_pairs = []
    for src_name, dst_name in edges:
        if src_name in name_to_idx and dst_name in name_to_idx:
            i = name_to_idx[src_name]
            j = name_to_idx[dst_name]
            if i != j:
                if i > j:
                    i, j = j, i
                connected_pairs.append((i, j))
    connected_pairs = list(set(connected_pairs))
    
    # 估算芯片边界框尺寸
    if W is None or H is None:
        max_w = max(w_orig.values())
        max_h = max(h_orig.values())
        total_area = sum(w_orig[i] * h_orig[i] for i in range(n))
        estimated_side = math.ceil(math.sqrt(total_area * 2))
        if W is None:
            W = max(estimated_side, max_w * 3)
        if H is None:
            H = max(estimated_side, max_h * 3)
    
    # 计算grid数量
    grid_w = int(math.ceil(W / grid_size))
    grid_h = int(math.ceil(H / grid_size))
    
    if verbose:
        print(f"网格化布局: grid_size={grid_size}, grid_w={grid_w}, grid_h={grid_h}")
        print(f"问题规模: {n} 个模块, {len(connected_pairs)} 对有连接的模块对")
    
    # ============ 步骤2: 创建ILP问题 ============
    prob = pulp.LpProblem("ChipletPlacementGrid", pulp.LpMinimize)
    
    # 大M常数
    M = max(grid_w, grid_h) * 2
    
    # ============ 步骤3: 定义变量 ============
    # 3.1 整数变量：每个chiplet在grid中的左下角坐标（grid索引）
    x_grid = {}
    y_grid = {}
    for k in range(n):
        max_dim_k_grid = max(int(math.ceil(w_orig[k] / grid_size)), int(math.ceil(h_orig[k] / grid_size)))
        x_grid[k] = pulp.LpVariable(f"x_grid_{k}", lowBound=0, upBound=grid_w - 1, cat='Integer')
        y_grid[k] = pulp.LpVariable(f"y_grid_{k}", lowBound=0, upBound=grid_h - 1, cat='Integer')
    
    # 3.2 连续变量：实际坐标（用于计算距离和共享边长）
    x = {}
    y = {}
    for k in range(n):
        x[k] = pulp.LpVariable(f"x_{k}", lowBound=0, upBound=W)
        y[k] = pulp.LpVariable(f"y_{k}", lowBound=0, upBound=H)
        # 约束：实际坐标 = grid坐标 * grid_size
        prob += x[k] == x_grid[k] * grid_size, f"x_grid_to_real_{k}"
        prob += y[k] == y_grid[k] * grid_size, f"y_grid_to_real_{k}"
    
    # 3.3 二进制变量：旋转变量
    r = {}
    for k in range(n):
        r[k] = pulp.LpVariable(f"r_{k}", cat='Binary')
    
    # 3.4 连续变量：实际宽度和高度
    w = {}
    h = {}
    for k in range(n):
        w_min = min(w_orig[k], h_orig[k])
        w_max = max(w_orig[k], h_orig[k])
        w[k] = pulp.LpVariable(f"w_{k}", lowBound=w_min, upBound=w_max)
        h[k] = pulp.LpVariable(f"h_{k}", lowBound=w_min, upBound=w_max)
    
    # 3.5 辅助变量：中心坐标
    cx = {}
    cy = {}
    for k in range(n):
        cx[k] = pulp.LpVariable(f"cx_{k}", lowBound=0, upBound=W)
        cy[k] = pulp.LpVariable(f"cy_{k}", lowBound=0, upBound=H)
    
    # 3.6 二进制变量：控制相邻方式
    z1 = {}
    z2 = {}
    z1L = {}
    z1R = {}
    z2D = {}
    z2U = {}
    
    # 3.7 二进制变量：控制相邻方式
    for i, j in connected_pairs:
        z1[(i, j)] = pulp.LpVariable(f"z1_{i}_{j}", cat="Binary")
        z2[(i, j)] = pulp.LpVariable(f"z2_{i}_{j}", cat="Binary")
        z1L[(i, j)] = pulp.LpVariable(f"z1L_{i}_{j}", cat="Binary")
        z1R[(i, j)] = pulp.LpVariable(f"z1R_{i}_{j}", cat="Binary")
        z2D[(i, j)] = pulp.LpVariable(f"z2D_{i}_{j}", cat="Binary")
        z2U[(i, j)] = pulp.LpVariable(f"z2U_{i}_{j}", cat="Binary")
    
    # ============ 步骤4: 定义约束 ============
    
    # 4.1 旋转约束
    for k in range(n):
        prob += w[k] == w_orig[k] + r[k] * (h_orig[k] - w_orig[k]), f"width_rotation_{k}"
        prob += h[k] == h_orig[k] + r[k] * (w_orig[k] - h_orig[k]), f"height_rotation_{k}"
    
    # 4.2 中心坐标定义
    for k in range(n):
        prob += cx[k] == x[k] + w[k] / 2.0, f"cx_def_{k}"
        prob += cy[k] == y[k] + h[k] / 2.0, f"cy_def_{k}"
    
    # 4.3 固定chiplet中心在方框中心
    if fixed_chiplet_idx is not None and 0 <= fixed_chiplet_idx < n:
        prob += cx[fixed_chiplet_idx] == W / 2.0, f"fix_center_x_{fixed_chiplet_idx}"
        prob += cy[fixed_chiplet_idx] == H / 2.0, f"fix_center_y_{fixed_chiplet_idx}"
        if verbose:
            print(f"固定chiplet {fixed_chiplet_idx} 的中心在方框中心 ({W/2:.2f}, {H/2:.2f})")
    
    # 4.4 相邻约束：对于每对有连接的模块对 (i, j)
    for i, j in connected_pairs:
        # 规则1: 必须相邻，且只能选一种方式
        prob += z1[(i, j)] + z2[(i, j)] == 1, f"must_adjacent_{i}_{j}"
        
        # 规则2: 如果水平相邻，要么 i 在左，要么 i 在右
        prob += z1L[(i, j)] + z1R[(i, j)] == z1[(i, j)], f"horizontal_direction_{i}_{j}"
        
        # 规则3: 如果垂直相邻，要么 i 在下，要么 i 在上
        prob += z2D[(i, j)] + z2U[(i, j)] == z2[(i, j)], f"vertical_direction_{i}_{j}"
        
        # 规则4: 距离约束 - 有链接关系的chiplet之间距离不能超过一个grid
        # 简化：直接使用grid坐标约束距离
        # 如果水平相邻：|y_grid_i - y_grid_j| <= 1（在垂直方向上最多相差1个grid）
        # 如果垂直相邻：|x_grid_i - x_grid_j| <= 1（在水平方向上最多相差1个grid）
        
        # 水平相邻时，垂直方向grid距离 <= 1
        diff_y_grid = pulp.LpVariable(f"diff_y_grid_{i}_{j}", lowBound=0, upBound=grid_h, cat='Integer')
        prob += diff_y_grid >= y_grid[i] - y_grid[j], f"diff_y_grid_abs1_{i}_{j}"
        prob += diff_y_grid >= y_grid[j] - y_grid[i], f"diff_y_grid_abs2_{i}_{j}"
        prob += diff_y_grid <= 1 + M * (1 - z1[(i, j)]), f"diff_y_grid_limit_{i}_{j}"
        
        # 垂直相邻时，水平方向grid距离 <= 1
        diff_x_grid = pulp.LpVariable(f"diff_x_grid_{i}_{j}", lowBound=0, upBound=grid_w, cat='Integer')
        prob += diff_x_grid >= x_grid[i] - x_grid[j], f"diff_x_grid_abs1_{i}_{j}"
        prob += diff_x_grid >= x_grid[j] - x_grid[i], f"diff_x_grid_abs2_{i}_{j}"
        prob += diff_x_grid <= 1 + M * (1 - z2[(i, j)]), f"diff_x_grid_limit_{i}_{j}"
        
        # 规则5: 水平相邻的具体约束
        # 约束1：相邻方向的边界距离 ≤ grid_size
        # 如果 i 在左（z1L[i,j] = 1）：x_j - (x_i + w_i) <= grid_size（距离不超过1个grid）
        prob += x[j] - (x[i] + w[i]) <= grid_size + M * (1 - z1L[(i, j)]), f"horizontal_left_dist_{i}_{j}"
        prob += x[j] - (x[i] + w[i]) >= 0 - M * (1 - z1L[(i, j)]), f"horizontal_left_dist_lb_{i}_{j}"
        # 如果 i 在右（z1R[i,j] = 1）：x_i - (x_j + w_j) <= grid_size（距离不超过1个grid）
        prob += x[i] - (x[j] + w[j]) <= grid_size + M * (1 - z1R[(i, j)]), f"horizontal_right_dist_{i}_{j}"
        prob += x[i] - (x[j] + w[j]) >= 0 - M * (1 - z1R[(i, j)]), f"horizontal_right_dist_lb_{i}_{j}"
        
        # 约束2：垂直方向（与相邻方向垂直）的重叠长度 >= min_shared_length
        # 重叠长度 = min(y[i] + h[i], y[j] + h[j]) - max(y[i], y[j])
        # 首先确保有重叠：y[i] < y[j] + h[j] 且 y[j] < y[i] + h[i]
        prob += y[i] - (y[j] + h[j]) <= M * (1 - z1[(i, j)]), f"horizontal_overlap_y1_{i}_{j}"
        prob += y[j] - (y[i] + h[i]) <= M * (1 - z1[(i, j)]), f"horizontal_overlap_y2_{i}_{j}"
        
        # 共享边长度（垂直方向的重叠长度）
        max_shared_y = max(h_orig[i], h_orig[j])
        shared_y = pulp.LpVariable(f"shared_y_{i}_{j}", lowBound=0, upBound=max_shared_y)
        prob += shared_y <= (y[i] + h[i]) - y[j] + M * (1 - z1[(i, j)]), f"shared_y_ub1_{i}_{j}"
        prob += shared_y <= (y[j] + h[j]) - y[i] + M * (1 - z1[(i, j)]), f"shared_y_ub2_{i}_{j}"
        prob += shared_y >= min_shared_length - M * (1 - z1[(i, j)]), f"shared_y_min_{i}_{j}"
        prob += shared_y <= M * z1[(i, j)], f"shared_y_zero_{i}_{j}"
        
        # 规则6: 垂直相邻的具体约束
        # 约束1：相邻方向的边界距离 ≤ grid_size
        # 如果 i 在下（z2D[i,j] = 1）：y_j - (y_i + h_i) <= grid_size（距离不超过1个grid）
        prob += y[j] - (y[i] + h[i]) <= grid_size + M * (1 - z2D[(i, j)]), f"vertical_down_dist_{i}_{j}"
        prob += y[j] - (y[i] + h[i]) >= 0 - M * (1 - z2D[(i, j)]), f"vertical_down_dist_lb_{i}_{j}"
        # 如果 i 在上（z2U[i,j] = 1）：y_i - (y_j + h_j) <= grid_size（距离不超过1个grid）
        prob += y[i] - (y[j] + h[j]) <= grid_size + M * (1 - z2U[(i, j)]), f"vertical_up_dist_{i}_{j}"
        prob += y[i] - (y[j] + h[j]) >= 0 - M * (1 - z2U[(i, j)]), f"vertical_up_dist_lb_{i}_{j}"
        
        # 约束2：水平方向（与相邻方向垂直）的重叠长度 >= min_shared_length
        # 重叠长度 = min(x[i] + w[i], x[j] + w[j]) - max(x[i], x[j])
        # 首先确保有重叠：x[i] < x[j] + w[j] 且 x[j] < x[i] + w[i]
        prob += x[i] - (x[j] + w[j]) <= M * (1 - z2[(i, j)]), f"vertical_overlap_x1_{i}_{j}"
        prob += x[j] - (x[i] + w[i]) <= M * (1 - z2[(i, j)]), f"vertical_overlap_x2_{i}_{j}"
        
        # 共享边长度（水平方向的重叠长度）
        max_shared_x = max(w_orig[i], w_orig[j])
        shared_x = pulp.LpVariable(f"shared_x_{i}_{j}", lowBound=0, upBound=max_shared_x)
        prob += shared_x <= (x[i] + w[i]) - x[j] + M * (1 - z2[(i, j)]), f"shared_x_ub1_{i}_{j}"
        prob += shared_x <= (x[j] + w[j]) - x[i] + M * (1 - z2[(i, j)]), f"shared_x_ub2_{i}_{j}"
        prob += shared_x >= min_shared_length - M * (1 - z2[(i, j)]), f"shared_x_min_{i}_{j}"
        prob += shared_x <= M * z2[(i, j)], f"shared_x_zero_{i}_{j}"
    
    # 4.5 边界约束
    for k in range(n):
        prob += x[k] + w[k] <= W, f"boundary_x_{k}"
        prob += y[k] + h[k] <= H, f"boundary_y_{k}"
    
    # 4.6 非重叠约束
    p_left = {}
    p_right = {}
    p_down = {}
    p_up = {}
    
    all_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            all_pairs.append((i, j))
            p_left[(i, j)] = pulp.LpVariable(f"p_left_{i}_{j}", cat='Binary')
            p_right[(i, j)] = pulp.LpVariable(f"p_right_{i}_{j}", cat='Binary')
            p_down[(i, j)] = pulp.LpVariable(f"p_down_{i}_{j}", cat='Binary')
            p_up[(i, j)] = pulp.LpVariable(f"p_up_{i}_{j}", cat='Binary')
    
    for i, j in all_pairs:
        prob += p_left[(i, j)] + p_right[(i, j)] + p_down[(i, j)] + p_up[(i, j)] == 1, \
               f"non_overlap_choice_{i}_{j}"
        prob += x[i] + w[i] <= x[j] + M * (1 - p_left[(i, j)]), f"non_overlap_left_{i}_{j}"
        prob += x[j] + w[j] <= x[i] + M * (1 - p_right[(i, j)]), f"non_overlap_right_{i}_{j}"
        prob += y[i] + h[i] <= y[j] + M * (1 - p_down[(i, j)]), f"non_overlap_down_{i}_{j}"
        prob += y[j] + h[j] <= y[i] + M * (1 - p_up[(i, j)]), f"non_overlap_up_{i}_{j}"
    
    # 4.7 外接方框约束
    bbox_min_x = pulp.LpVariable("bbox_min_x", lowBound=0, upBound=W)
    bbox_max_x = pulp.LpVariable("bbox_max_x", lowBound=0, upBound=W)
    bbox_min_y = pulp.LpVariable("bbox_min_y", lowBound=0, upBound=H)
    bbox_max_y = pulp.LpVariable("bbox_max_y", lowBound=0, upBound=H)
    bbox_w = pulp.LpVariable("bbox_w", lowBound=0, upBound=W)
    bbox_h = pulp.LpVariable("bbox_h", lowBound=0, upBound=H)
    
    for k in range(n):
        prob += bbox_min_x <= x[k], f"bbox_min_x_{k}"
        prob += bbox_max_x >= x[k] + w[k], f"bbox_max_x_{k}"
        prob += bbox_min_y <= y[k], f"bbox_min_y_{k}"
        prob += bbox_max_y >= y[k] + h[k], f"bbox_max_y_{k}"
    
    prob += bbox_w == bbox_max_x - bbox_min_x, "bbox_w_def"
    prob += bbox_h == bbox_max_y - bbox_min_y, "bbox_h_def"
    
    # ============ 步骤5: 定义目标函数 ============
    # 5.1 线长（曼哈顿距离）
    wirelength = 0
    for i, j in connected_pairs:
        dx_abs = pulp.LpVariable(f"dx_abs_{i}_{j}", lowBound=0)
        dy_abs = pulp.LpVariable(f"dy_abs_{i}_{j}", lowBound=0)
        
        prob += dx_abs >= cx[i] - cx[j], f"dx_abs1_{i}_{j}"
        prob += dx_abs >= cx[j] - cx[i], f"dx_abs2_{i}_{j}"
        prob += dy_abs >= cy[i] - cy[j], f"dy_abs1_{i}_{j}"
        prob += dy_abs >= cy[j] - cy[i], f"dy_abs2_{i}_{j}"
        
        wirelength += dx_abs + dy_abs
    
    # 5.2 面积代理
    t = pulp.LpVariable("bbox_area_proxy_t", lowBound=0)
    prob += t <= (bbox_w + bbox_h) / 2.0, "area_proxy_am_gm1"
    K = max(W, H)
    prob += bbox_w <= K * t, "area_proxy_am_gm2"
    prob += bbox_h <= K * t, "area_proxy_am_gm3"
    
    # 5.3 目标函数
    if minimize_bbox_area:
        prob += distance_weight * wirelength + area_weight * t, "Objective"
    else:
        prob += distance_weight * wirelength, "Objective"
    
    if verbose:
        print(f"目标函数: {distance_weight} * wirelength + {area_weight} * area_proxy")
    
    return ILPModelContext(
        prob=prob,
        nodes=nodes,
        edges=edges,
        x=x,
        y=y,
        r=r,
        z1=z1,
        z2=z2,
        z1L=z1L,
        z1R=z1R,
        z2D=z2D,
        z2U=z2U,
        connected_pairs=connected_pairs,
        bbox_w=bbox_w,
        bbox_h=bbox_h,
        W=W,
        H=H,
        grid_size=grid_size,
        fixed_chiplet_idx=fixed_chiplet_idx,
    )