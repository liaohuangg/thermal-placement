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
    ILPPlacementResult
        包含布局坐标、旋转状态、目标值、求解状态以及边界框大小等信息。
    """
    import time
    start_time = time.time()
    
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
            W = max(estimated_side, max_w * 3)
        if H is None:
            H = max(estimated_side, max_h * 3)
    
    # 大 M 常数：一个足够大的数（通常取 2 倍芯片最大尺寸）
    M = max(W, H) * 2
    
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
        z1[(i, j)] = pulp.LpVariable(f"z1_{i}_{j}", cat='Binary')
        z2[(i, j)] = pulp.LpVariable(f"z2_{i}_{j}", cat='Binary')
        z1L[(i, j)] = pulp.LpVariable(f"z1L_{i}_{j}", cat='Binary')
        z1R[(i, j)] = pulp.LpVariable(f"z1R_{i}_{j}", cat='Binary')
        z2D[(i, j)] = pulp.LpVariable(f"z2D_{i}_{j}", cat='Binary')
        z2U[(i, j)] = pulp.LpVariable(f"z2U_{i}_{j}", cat='Binary')
        
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
    
    # 组合目标函数：最小化总线长 + 外接方框“面积代理”（使用权重平衡）
    if minimize_bbox_area:
        # 目标函数：最小化加权总线长和面积代理 t
        prob += distance_weight * pulp.lpSum(distance_terms) + area_weight * t, \
               "total_objective"
        if verbose:
            print(f"目标函数: β1*线长 + β2*t（面积凸近似），其中 β1={distance_weight:.2f}, β2={area_weight:.2f}")
    else:
        # 只最小化距离
        prob += pulp.lpSum(distance_terms), "total_connection_distance"
        if verbose:
            print("目标函数: 只最小化线长（不考虑外接方框大小）")
    
    # ============ 步骤6: 求解ILP问题 ============
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
        result = subprocess.run(['glpsol', '--version'], 
                              capture_output=True, 
                              timeout=2)
        if result.returncode == 0:
            solver = pulp.getSolver('GLPK_CMD', timeLimit=time_limit, msg=verbose)
            solver_name = "GLPK"
            if verbose:
                print(f"使用求解器: {solver_name}")
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        # GLPK 不可用，尝试 CBC
        try:
            solver = pulp.getSolver('PULP_CBC_CMD', timeLimit=time_limit, msg=verbose)
            solver_name = "CBC"
            if verbose:
                print(f"使用求解器: {solver_name}")
        except:
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
                print(f"目标函数值（总距离）: {pulp.value(prob.objective):.2f}")
        
        # ============ 步骤7: 提取解 ============
        layout = {}
        rotations = {}
        for k, node in enumerate(nodes):
            if status == pulp.LpStatusOptimal:
                x_val = pulp.value(x[k])
                y_val = pulp.value(y[k])
                r_val = pulp.value(r[k])
                layout[node.name] = (x_val if x_val is not None else 0.0,
                                     y_val if y_val is not None else 0.0)
                rotations[node.name] = bool(r_val > 0.5) if r_val is not None else False
            else:
                layout[node.name] = (0.0, 0.0)
                rotations[node.name] = False
        
        obj_value = pulp.value(prob.objective) if status == pulp.LpStatusOptimal else float('inf')
        
        return ILPPlacementResult(
            layout=layout,
            rotations=rotations,
            objective_value=obj_value,
            status=pulp.LpStatus[status],
            solve_time=solve_time,
            bounding_box=(W, H),
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
            objective_value=float('inf'),
            status="Error",
            solve_time=solve_time,
            bounding_box=(W if W else 100.0, H if H else 100.0),
        )


if __name__ == "__main__":
    # 测试：使用ILP求解布局问题
    print("=" * 80)
    print("ILP芯片布局求解器测试（相邻约束模型）")
    print("=" * 80)
    
    # 构建测试图（只取前4个模块，生成4条边）
    nodes, edges = build_random_chiplet_graph(edge_prob=0.2, max_nodes=5, fixed_num_edges=5)
    
    print(f"\n问题规模: {len(nodes)} 个模块, {len(edges)} 条边")
    
    # 求解
    # min_shared_length: 相邻chiplet之间共享边的最小长度
    # - 0.0: 只需要有接触即可（共享边长度 >= 0）
    # - 1.0: 共享边长度必须 >= 1.0
    # - 2.0: 共享边长度必须 >= 2.0
    result = solve_placement_ilp(
        nodes=nodes,
        edges=edges,
        W=None,  # 自动计算
        H=None,  # 自动计算
        time_limit=300,
        verbose=True,
        min_shared_length=0.5,  # 可以修改这个值来控制相邻chiplet共享边的最小长度
    )
    
    print(f"\n求解结果:")
    print(f"  状态: {result.status}")
    if result.status == 'Optimal':
        print(f"  目标值（总距离）: {result.objective_value:.2f}")
        print(f"  边界框尺寸: {result.bounding_box[0]:.2f} × {result.bounding_box[1]:.2f}")
        # 显示旋转信息
        rotated_nodes = [name for name, rotated in result.rotations.items() if rotated]
        if rotated_nodes:
            print(f"  旋转的节点: {', '.join(rotated_nodes)}")
        else:
            print(f"  旋转的节点: 无")
    print(f"  求解时间: {result.solve_time:.2f} 秒")
    
    # 可视化结果
    if result.status == 'Optimal':
        from pathlib import Path
        out_path = Path(__file__).parent.parent / "chiplet_ilp_placement.png"
        
        # 创建旋转后的节点副本用于绘图
        from copy import deepcopy
        nodes_for_draw = []
        for i, node in enumerate(nodes):
            node_copy = deepcopy(node)
            if result.rotations.get(node.name, False):
                # 旋转90度：交换宽度和高度
                orig_w = node.dimensions.get("x", 0.0)
                orig_h = node.dimensions.get("y", 0.0)
                node_copy.dimensions["x"] = orig_h
                node_copy.dimensions["y"] = orig_w
                
                # 旋转接口位置：原来的 (px, py) 变成 (h - py, px)
                if node_copy.phys:
                    rotated_phys = []
                    for p in node.phys:
                        px = float(p.get("x", 0.0))
                        py = float(p.get("y", 0.0))
                        # 旋转90度：相对于新左下角的坐标
                        new_px = orig_h - py
                        new_py = px
                        rotated_phys.append({"x": new_px, "y": new_py})
                    node_copy.phys = rotated_phys
            nodes_for_draw.append(node_copy)
        
        draw_chiplet_diagram(nodes_for_draw, edges, save_path=str(out_path), layout=result.layout)
        print(f"\n布局结果已保存到: {out_path}")
    else:
        print(f"\n求解未成功，状态: {result.status}")
