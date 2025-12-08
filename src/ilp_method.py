"""
使用整数线性规划（ILP）进行 chiplet 布局优化。

主要特性
--------
1. **相邻约束**：只有硅桥互联边的 chiplet 必须水平或垂直相邻（紧靠），并且共享边长度不少于给定下界。普通互联边不需要相邻约束。
2. **旋转约束**：每个 chiplet 允许 0°/90° 旋转，由二进制变量 ``r_k`` 控制宽高交换。注意：正方形chiplet（w == h）自动关闭旋转选项（r_k = None，固定为0），因为旋转不会带来布局优势。
3. **非重叠约束**：任意两块 chiplet 之间不能重叠。
4. **外接方框约束**：显式构造覆盖所有 chiplet 的外接矩形，并对其宽高建立线性约束。
5. **多目标优化**：目标函数为

   ``β1 * wirelength + β2 * t``

   其中 ``wirelength`` 是所有连边中心点间的半周长线长（HPWL）之和，
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
    - `r`     : 每个 chiplet 的旋转变量（正方形chiplet为None，固定不旋转）
    - `z1, z2`: 每对有连边的 chiplet 的"相邻方式"变量（水平/垂直）
    - `z1L, z1R, z2D, z2U`: 每对有连边的 chiplet 的相对方向变量（左、右、下、上）
    - `connected_pairs` : 有连边的 (i, j) 索引列表（i < j）
    - `edge_types` : 边的类型映射，格式为 {(i, j): "silicon_bridge" | "normal"}
    - `bbox_w, bbox_h` : 外接方框宽和高对应的变量
    - `W, H`  : 外接边界框的上界尺寸（建模阶段确定）
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
    edge_types: Dict[Tuple[int, int], str]  # 边的类型映射

    bbox_w: pulp.LpVariable
    bbox_h: pulp.LpVariable

    W: float
    H: float
    min_shared_length: float  # 共享边最小长度


def build_placement_ilp_model(
    nodes: List[ChipletNode],
    edges: List[Tuple[str, str]] | List[Tuple[str, str, str]],
    silicon_bridge_edges: Optional[List[Tuple[str, str]]] = None,
    normal_edges: Optional[List[Tuple[str, str]]] = None,
    W: Optional[float] = None,
    H: Optional[float] = None,
    time_limit: int = 300,
    verbose: bool = True,
    min_shared_length: float = 0.0,
    minimize_bbox_area: bool = True,
    distance_weight: float = 1.0,
    area_weight: float = 0.1,
    silicon_bridge_weight: Optional[float] = None,  # 硅桥互联边的权重（如果指定，会覆盖distance_weight）
    normal_edge_weight: Optional[float] = None,  # 普通链接边的权重（如果指定，会覆盖distance_weight）
) -> ILPModelContext:
    """
    使用 ILP 求解 chiplet 布局（相邻 + 非重叠 + 外接方框 + 多目标）。

    约束概要
    --------
    - **相邻约束**：只有硅桥互联边对应的两个 chiplet 必须相邻：
      - 要么水平相邻（左右紧靠），要么垂直相邻（上下紧靠），二选一；
      - 相邻方向上的共享边长度必须大于等于 ``min_shared_length``（仅适用于硅桥互联的chiplet）。
      - 普通互联边不需要相邻约束，也不需要共享边长度约束，只需要在目标函数中最小化距离。
    - **旋转约束**：每个 chiplet 通过二进制变量 ``r_k`` 决定是否旋转 90°（宽高交换）。正方形chiplet（w == h）自动关闭旋转选项（r_k = None，固定为0），因为旋转不会带来布局优势。
    - **非重叠约束**：任意两个 chiplet 在平面上不能有重叠区域。
    - **外接方框约束**：显式构造一个覆盖所有 chiplet 的外接矩形，并引入宽高 ``bbox_w, bbox_h``。

    目标函数
    --------
    - 线长（wirelength）：所有连边（包括硅桥互联和普通互联）对应中心点之间的半周长线长（HPWL，Half-Perimeter Wirelength）的总和。
    - 面积代理 ``t``：利用 AM–GM 不等式
      ``t ≤ (bbox_w + bbox_h)/2``, ``bbox_w ≤ K·t``, ``bbox_h ≤ K·t`` 构造的凸近似，
      在最小化 ``t`` 时同时推动 ``bbox_w`` 和 ``bbox_h`` 变小。
    - 最终目标：``distance_weight * wirelength + area_weight * t``。
    - 如果指定了 ``silicon_bridge_weight`` 或 ``normal_edge_weight``，则不同类型的边使用不同的权重。

    参数
    ----
    nodes:
        从 JSON 读入的 chiplet 节点列表。
    edges:
        节点之间的连边列表。可以是：
        - 旧格式：``List[Tuple[str, str]]``（向后兼容）
        - 新格式：``List[Tuple[str, str, str]]``，第三个元素为边类型（"silicon_bridge" 或 "normal"）
    silicon_bridge_edges:
        硅桥互联边列表（可选，如果提供，会与edges合并）
    normal_edges:
        普通链接边列表（可选，如果提供，会与edges合并）
    W, H:
        外接芯片区域的最大宽高；若为 ``None`` 则根据 chiplet 总面积和尺寸自动估算。
    time_limit:
        求解时间上限（秒）。
    verbose:
        若为 True，则打印建模与求解过程中的关键信息。
    min_shared_length:
        硅桥互联的 chiplet 之间共享边长度的下界（在相邻方向上）。
        仅适用于硅桥互联边，普通互联边不需要此约束。
    minimize_bbox_area:
        是否在目标函数中加入面积代理项 ``t``。
    distance_weight:
        线长项的默认权重（对应公式中的 β₁），如果未指定特定类型边的权重则使用此值。
    area_weight:
        面积代理 ``t`` 的权重（对应公式中的 β₂）。
    silicon_bridge_weight:
        硅桥互联边的权重（如果指定，会覆盖distance_weight）
    normal_edge_weight:
        普通链接边的权重（如果指定，会覆盖distance_weight）

    返回
    ----
    ILPModelContext
        已经构建好的 ILP 模型上下文（尚未求解），
        可在外部继续添加约束（例如排除解）后再调用求解函数。
    """
    n = len(nodes)
    name_to_idx = {node.name: i for i, node in enumerate(nodes)}
    
    # ============ 步骤1: 处理边输入，统一转换为带类型的边 ============
    # 合并所有边并提取类型信息
    all_edges_with_type: List[Tuple[str, str, str]] = []
    edge_types_map: Dict[Tuple[int, int], str] = {}
    
    # 处理edges参数（可能是旧格式或新格式）
    for edge in edges:
        if len(edge) == 2:
            # 旧格式：(src, dst)，默认为普通链接边
            src_name, dst_name = edge
            edge_type = "normal"
        elif len(edge) == 3:
            # 新格式：(src, dst, type)
            src_name, dst_name, edge_type = edge
        else:
            continue
        
        if src_name in name_to_idx and dst_name in name_to_idx:
            i = name_to_idx[src_name]
            j = name_to_idx[dst_name]
            if i != j:
                # 确保 i < j（避免重复）
                if i > j:
                    i, j = j, i
                all_edges_with_type.append((src_name, dst_name, edge_type))
                edge_types_map[(i, j)] = edge_type
    
    # 处理silicon_bridge_edges参数
    if silicon_bridge_edges:
        for src_name, dst_name in silicon_bridge_edges:
            if src_name in name_to_idx and dst_name in name_to_idx:
                i = name_to_idx[src_name]
                j = name_to_idx[dst_name]
                if i != j:
                    if i > j:
                        i, j = j, i
                    if (src_name, dst_name, "silicon_bridge") not in all_edges_with_type:
                        all_edges_with_type.append((src_name, dst_name, "silicon_bridge"))
                    edge_types_map[(i, j)] = "silicon_bridge"
    
    # 处理normal_edges参数
    if normal_edges:
        for src_name, dst_name in normal_edges:
            if src_name in name_to_idx and dst_name in name_to_idx:
                i = name_to_idx[src_name]
                j = name_to_idx[dst_name]
                if i != j:
                    if i > j:
                        i, j = j, i
                    if (src_name, dst_name, "normal") not in all_edges_with_type:
                        all_edges_with_type.append((src_name, dst_name, "normal"))
                    edge_types_map[(i, j)] = "normal"
    
    # 转换为旧格式的edges列表（用于向后兼容）
    edges_legacy: List[Tuple[str, str]] = [(src, dst) for src, dst, _ in all_edges_with_type]
    
    # ============ 步骤2: 读取已知条件（固定不变的量）============
    # 获取原始矩形尺寸（未旋转时）
    w_orig = {}  # w_k^o: 模块 k 的原始宽度
    h_orig = {}  # h_k^o: 模块 k 的原始高度
    for i, node in enumerate(nodes):
        w_orig[i] = float(node.dimensions.get("x", 0.0))
        h_orig[i] = float(node.dimensions.get("y", 0.0))
    
    # 找到所有有边连接的模块对
    # 分为两类：
    # 1. silicon_bridge_pairs: 硅桥互联边，需要相邻约束
    # 2. all_connected_pairs: 所有边（包括硅桥互联和普通互联），用于目标函数
    silicon_bridge_pairs = []
    all_connected_pairs = []
    
    for src_name, dst_name, edge_type in all_edges_with_type:
        if src_name in name_to_idx and dst_name in name_to_idx:
            i = name_to_idx[src_name]
            j = name_to_idx[dst_name]
            if i != j:
                # 确保 i < j（避免重复）
                if i > j:
                    i, j = j, i
                pair = (i, j)
                all_connected_pairs.append(pair)
                # 只有硅桥互联边需要相邻约束
                if edge_type == "silicon_bridge":
                    silicon_bridge_pairs.append(pair)
    
    silicon_bridge_pairs = list(set(silicon_bridge_pairs))  # 去重
    all_connected_pairs = list(set(all_connected_pairs))  # 去重
    
    if verbose:
        silicon_bridge_count = len(silicon_bridge_pairs)
        normal_count = len(all_connected_pairs) - silicon_bridge_count
        print(f"问题规模: {n} 个模块, {len(all_connected_pairs)} 对有连接的模块对")
        print(f"  其中：硅桥互联边 {silicon_bridge_count} 条（需要相邻约束），普通链接边 {normal_count} 条（仅目标函数）")
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
    M = max(W, H) * 3
    
    if verbose:
        print(f"芯片边界框尺寸: ChipW = {W:.2f}, ChipH = {H:.2f}")
        print(f"大 M 常数: M = {M:.2f}")
        print(f"硅桥互联chiplet共享边最小长度: min_shared_length = {min_shared_length:.2f}")
    
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
    # 注意：如果chiplet是正方形（w_orig == h_orig），则关闭旋转选项（r_k = None，固定为0）
    r = {}
    square_chiplets = []  # 记录正方形chiplet的索引
    for k in range(n):
        # 检查是否为正方形（允许小的数值误差）
        is_square = abs(w_orig[k] - h_orig[k]) < 1e-6
        if is_square:
            # 正方形chiplet：固定 r_k = 0（不旋转）
            r[k] = None  # 不使用变量，直接约束为0
            square_chiplets.append(k)
            if verbose:
                print(f"  模块 {k} ({nodes[k].name}) 是正方形 (w={w_orig[k]:.2f}, h={h_orig[k]:.2f})，关闭旋转选项")
        else:
            # 非正方形chiplet：创建旋转变量
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
    
    # 3.5 二进制变量：控制相邻方式（仅对于硅桥互联边 (i, j)）
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
    
    for i, j in silicon_bridge_pairs:
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
    # 对于正方形chiplet（r_k = None），直接固定 w_k = w_k^o, h_k = h_k^o
    for k in range(n):
        if r[k] is None:
            # 正方形chiplet：固定尺寸，不旋转
            prob += w[k] == w_orig[k], f"width_fixed_{k}"
            prob += h[k] == h_orig[k], f"height_fixed_{k}"
        else:
            # 非正方形chiplet：允许旋转
            prob += w[k] == w_orig[k] + r[k] * (h_orig[k] - w_orig[k]), f"width_rotation_{k}"
            prob += h[k] == h_orig[k] + r[k] * (w_orig[k] - h_orig[k]), f"height_rotation_{k}"
    
    # 4.2 中心坐标定义
    for k in range(n):
        prob += cx[k] == x[k] + w[k] / 2.0, f"cx_def_{k}"
        prob += cy[k] == y[k] + h[k] / 2.0, f"cy_def_{k}"
    
    # 4.3 相邻约束：仅对于硅桥互联边 (i, j)，它们必须相邻
    # 普通互联边不需要相邻约束，只需要在目标函数中最小化距离
    # 对于硅桥互联的chiplet对，必须满足：
    #   1. 必须相邻（左右或上下，二选一）
    #   2. 相邻的边必须共享一定长度的相邻边（>= min_shared_length）
    for i, j in silicon_bridge_pairs:
        # ========== 规则1: 必须相邻，且只能选一种方式 ==========
        # z1[i,j] + z2[i,j] = 1（要么水平相邻，要么垂直相邻，不能不相邻，也不能两种都选）
        prob += z1[(i, j)] + z2[(i, j)] == 1, f"must_adjacent_{i}_{j}"
        
        # ========== 规则2: 水平相邻时的方向选择 ==========
        # z1L[i,j] + z1R[i,j] = z1[i,j]（如果水平相邻，要么 i 在左，要么 i 在右）
        prob += z1L[(i, j)] + z1R[(i, j)] == z1[(i, j)], f"horizontal_direction_{i}_{j}"
        
        # ========== 规则3: 垂直相邻时的方向选择 ==========
        # z2D[i,j] + z2U[i,j] = z2[i,j]（如果垂直相邻，要么 i 在下，要么 i 在上）
        prob += z2D[(i, j)] + z2U[(i, j)] == z2[(i, j)], f"vertical_direction_{i}_{j}"
        
        # ========== 规则4: 水平相邻的具体约束（z1[i,j] = 1 时才生效）==========
        # (4.4.1) 边界对齐（左右紧靠，无间隙）
        # 如果 i 在左（z1L[i,j] = 1）：x_i + w_i = x_j（i 的右边 = j 的左边）
        prob += x[i] + w[i] - x[j] <= M * (1 - z1L[(i, j)]), f"horizontal_left_{i}_{j}"
        prob += x[i] + w[i] - x[j] >= -M * (1 - z1L[(i, j)]), f"horizontal_left_eq_{i}_{j}"
        
        # 如果 i 在右（z1R[i,j] = 1）：x_j + w_j = x_i（j 的右边 = i 的左边）
        prob += x[j] + w[j] - x[i] <= M * (1 - z1R[(i, j)]), f"horizontal_right_{i}_{j}"
        prob += x[j] + w[j] - x[i] >= -M * (1 - z1R[(i, j)]), f"horizontal_right_eq_{i}_{j}"
        
        # (4.4.2) 垂直方向必须重叠，且共享边长度 >= min_shared_length（强制约束）
        # 当 z1[i,j] = 1（水平相邻）时：
        #   - 垂直方向必须重叠：y_i ≤ y_j + h_j 且 y_j ≤ y_i + h_i
        #   - 共享边长度 = min(y_i + h_i, y_j + h_j) - max(y_i, y_j) >= min_shared_length
        prob += y[i] - (y[j] + h[j]) <= M * (1 - z1[(i, j)]), f"horizontal_overlap_y1_{i}_{j}"
        prob += y[j] - (y[i] + h[i]) <= M * (1 - z1[(i, j)]), f"horizontal_overlap_y2_{i}_{j}"
        
        # 强制约束：当 z1[i,j] = 1 时，实际重叠长度必须 >= min_shared_length
        # 即：min((y[i] + h[i]) - y[j], (y[j] + h[j]) - y[i]) >= min_shared_length
        # 这等价于：(y[i] + h[i]) - y[j] >= min_shared_length 且 (y[j] + h[j]) - y[i] >= min_shared_length
        prob += (y[i] + h[i]) - y[j] >= min_shared_length - M * (1 - z1[(i, j)]), \
               f"actual_overlap_y1_min_{i}_{j}"
        prob += (y[j] + h[j]) - y[i] >= min_shared_length - M * (1 - z1[(i, j)]), \
               f"actual_overlap_y2_min_{i}_{j}"
        
        # shared_y 变量的约束（用于记录共享边长度）
        # 上界：shared_y <= min((y[i] + h[i]) - y[j], (y[j] + h[j]) - y[i])
        prob += shared_y[(i, j)] <= (y[i] + h[i]) - y[j] + M * (1 - z1[(i, j)]), \
               f"shared_y_ub1_{i}_{j}"
        prob += shared_y[(i, j)] <= (y[j] + h[j]) - y[i] + M * (1 - z1[(i, j)]), \
               f"shared_y_ub2_{i}_{j}"
        # 下界：当 z1[i,j] = 1 时，shared_y[i,j] >= min_shared_length
        prob += shared_y[(i, j)] >= min_shared_length * z1[(i, j)], \
               f"shared_y_min_{i}_{j}"
        # 当 z1[i,j] = 0 时，shared_y[i,j] = 0
        prob += shared_y[(i, j)] <= M * z1[(i, j)], f"shared_y_zero_{i}_{j}"
        
        # ========== 规则5: 垂直相邻的具体约束（z2[i,j] = 1 时才生效）==========
        # (4.5.1) 边界对齐（上下紧靠，无间隙）
        # 如果 i 在下（z2D[i,j] = 1）：y_i + h_i = y_j（i 的顶部 = j 的底部）
        prob += y[i] + h[i] - y[j] <= M * (1 - z2D[(i, j)]), f"vertical_down_{i}_{j}"
        prob += y[i] + h[i] - y[j] >= -M * (1 - z2D[(i, j)]), f"vertical_down_eq_{i}_{j}"
        
        # 如果 i 在上（z2U[i,j] = 1）：y_j + h_j = y_i（j 的顶部 = i 的底部）
        prob += y[j] + h[j] - y[i] <= M * (1 - z2U[(i, j)]), f"vertical_up_{i}_{j}"
        prob += y[j] + h[j] - y[i] >= -M * (1 - z2U[(i, j)]), f"vertical_up_eq_{i}_{j}"
        
        # (4.5.2) 水平方向必须重叠，且共享边长度 >= min_shared_length（强制约束）
        # 当 z2[i,j] = 1（垂直相邻）时：
        #   - 水平方向必须重叠：x_i ≤ x_j + w_j 且 x_j ≤ x_i + w_i
        #   - 共享边长度 = min(x_i + w_i, x_j + w_j) - max(x_i, x_j) >= min_shared_length
        prob += x[i] - (x[j] + w[j]) <= M * (1 - z2[(i, j)]), f"vertical_overlap_x1_{i}_{j}"
        prob += x[j] - (x[i] + w[i]) <= M * (1 - z2[(i, j)]), f"vertical_overlap_x2_{i}_{j}"
        
        # 强制约束：当 z2[i,j] = 1 时，实际重叠长度必须 >= min_shared_length
        # 即：min((x[i] + w[i]) - x[j], (x[j] + w[j]) - x[i]) >= min_shared_length
        # 这等价于：(x[i] + w[i]) - x[j] >= min_shared_length 且 (x[j] + w[j]) - x[i] >= min_shared_length
        prob += (x[i] + w[i]) - x[j] >= min_shared_length - M * (1 - z2[(i, j)]), \
               f"actual_overlap_x1_min_{i}_{j}"
        prob += (x[j] + w[j]) - x[i] >= min_shared_length - M * (1 - z2[(i, j)]), \
               f"actual_overlap_x2_min_{i}_{j}"
        
        # shared_x 变量的约束（用于记录共享边长度）
        # 上界：shared_x <= min((x[i] + w[i]) - x[j], (x[j] + w[j]) - x[i])
        prob += shared_x[(i, j)] <= (x[i] + w[i]) - x[j] + M * (1 - z2[(i, j)]), \
               f"shared_x_ub1_{i}_{j}"
        prob += shared_x[(i, j)] <= (x[j] + w[j]) - x[i] + M * (1 - z2[(i, j)]), \
               f"shared_x_ub2_{i}_{j}"
        # 下界：当 z2[i,j] = 1 时，shared_x[i,j] >= min_shared_length
        prob += shared_x[(i, j)] >= min_shared_length * z2[(i, j)], \
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

    # 使用 AM-GM 不等式的凸近似来构造"面积代理"变量 t
    # 目标是用线性约束来逼近 bbox_w * bbox_h，从而在目标函数中最小化 t
    #
    # 参考形式：
    #   t ≥ (bbox_w + bbox_h) / 2  （注意：这里改为 >=，确保 t 至少是平均值）
    #   bbox_w ≤ K * t
    #   bbox_h ≤ K * t
    # 在最小化 t 的前提下，这会同时压缩 bbox_w 和 bbox_h，相当于间接减小面积。
    #
    # 这里 K 取一个与边界框尺度同一量级的常数，避免约束过松或不稳定。
    # 使用更小的 K 值可以使约束更紧，但需要确保 K 足够大以避免不可行
    K_bbox = max(W, H) * 0.5  # 减小 K 值，使约束更紧
    t = pulp.LpVariable("bbox_area_proxy_t", lowBound=0)

    # t ≥ (bbox_w + bbox_h) / 2  （确保 t 至少是宽高的平均值）
    prob += t >= (bbox_w + bbox_h) / 2.0, "bbox_t_ge_mean_wh"
    # bbox_w ≤ K * t, bbox_h ≤ K * t（当 t 最小时，bbox_w 和 bbox_h 也会被压缩）
    prob += bbox_w <= K_bbox * t, "bbox_w_le_Kt"
    prob += bbox_h <= K_bbox * t, "bbox_h_le_Kt"
    
    # 额外的约束：确保 t 不能太小（避免不可行）
    # t 的下界应该是所有 chiplet 的最小可能外接矩形的平均值
    min_possible_bbox = max(max(w_orig.values()), max(h_orig.values()))
    prob += t >= min_possible_bbox / 2.0, "bbox_t_min_bound"

    if verbose:
        print("外接方框约束: 已添加（使用 AM-GM 凸近似的面积代理 t）")
    
    # ============ 步骤5: 定义目标函数 ============
    # 多目标优化：
    # 1. 最小化有连接关系的模块之间的中心距离之和
    # 2. 最小化外接方框面积（使用 w + h 作为代理，或使用分段线性化）
    
    # 目标1: 最小化有连接关系的模块之间的半周长线长（HPWL）之和
    # HPWL (Half-Perimeter Wirelength) = |cx[i] - cx[j]| + |cy[i] - cy[j]|
    # 这是曼哈顿距离，是线长估计的标准方法
    # 注意：包括所有边（硅桥互联和普通互联）的HPWL
    distance_terms = []
    weighted_distance_terms = []
    
    for i, j in all_connected_pairs:
        # 距离辅助变量（用于线性化绝对值）
        dx_abs = pulp.LpVariable(f"dx_abs_{i}_{j}", lowBound=0)
        dy_abs = pulp.LpVariable(f"dy_abs_{i}_{j}", lowBound=0)
        
        # 绝对值线性化：dx_abs >= |cx[i] - cx[j]|
        prob += dx_abs >= cx[i] - cx[j], f"dx_abs_{i}_{j}_pos"
        prob += dx_abs >= cx[j] - cx[i], f"dx_abs_{i}_{j}_neg"
        # dy_abs >= |cy[i] - cy[j]|
        prob += dy_abs >= cy[i] - cy[j], f"dy_abs_{i}_{j}_pos"
        prob += dy_abs >= cy[j] - cy[i], f"dy_abs_{i}_{j}_neg"
        
        # HPWL = |cx[i] - cx[j]| + |cy[i] - cy[j]|
        dist_term = dx_abs + dy_abs
        distance_terms.append(dist_term)
        
        # 根据边类型确定权重
        edge_type = edge_types_map.get((i, j), "normal")
        if edge_type == "silicon_bridge" and silicon_bridge_weight is not None:
            weight = silicon_bridge_weight
        elif edge_type == "normal" and normal_edge_weight is not None:
            weight = normal_edge_weight
        else:
            weight = distance_weight
        
        weighted_distance_terms.append(weight * dist_term)
    
    # 目标2: 最小化外接方框面积（使用凸近似的面积代理 t）
    # 最终目标形式：β1 * 线长 + β2 * t
    if minimize_bbox_area:
        prob += (
            pulp.lpSum(weighted_distance_terms) + area_weight * t,
            "total_objective",
        )
        if verbose:
            silicon_bridge_w = silicon_bridge_weight if silicon_bridge_weight is not None else distance_weight
            normal_w = normal_edge_weight if normal_edge_weight is not None else distance_weight
            print(
                f"目标函数: 加权线长（HPWL半周长线长） + β2*t（面积凸近似）"
            )
            if silicon_bridge_weight is not None or normal_edge_weight is not None:
                print(f"  硅桥互联边权重: {silicon_bridge_w:.2f}, 普通链接边权重: {normal_w:.2f}")
            else:
                print(f"  统一权重: β1={distance_weight:.2f}")
            print(f"  面积权重: β2={area_weight:.2f}")
    else:
        prob += pulp.lpSum(weighted_distance_terms), "total_connection_distance"
        if verbose:
            silicon_bridge_w = silicon_bridge_weight if silicon_bridge_weight is not None else distance_weight
            normal_w = normal_edge_weight if normal_edge_weight is not None else distance_weight
            print("目标函数: 只最小化加权线长（HPWL半周长线长，不考虑外接方框大小）")
            if silicon_bridge_weight is not None or normal_edge_weight is not None:
                print(f"  硅桥互联边权重: {silicon_bridge_w:.2f}, 普通链接边权重: {normal_w:.2f}")
            else:
                print(f"  统一权重: {distance_weight:.2f}")

    # 返回尚未求解的模型上下文
    return ILPModelContext(
        prob=prob,
        nodes=nodes,
        edges=edges_legacy,  # 使用旧格式的边列表（向后兼容）
        x=x,
        y=y,
        r=r,
        z1=z1,
        z2=z2,
        z1L=z1L,
        z1R=z1R,
        z2D=z2D,
        z2U=z2U,
        connected_pairs=silicon_bridge_pairs,  # 只包含需要相邻约束的硅桥互联边
        edge_types=edge_types_map,  # 添加边类型映射
        bbox_w=bbox_w,
        bbox_h=bbox_h,
        W=W,
        H=H,
        min_shared_length=min_shared_length,  # 保存共享边最小长度
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
        # 检查求解状态：只有在 Optimal 或至少是可行状态时才提取解
        layout: Dict[str, Tuple[float, float]] = {}
        rotations: Dict[str, bool] = {}
        has_feasible_solution = False
        
        # 检查求解状态：Infeasible 或 Not Solved 状态不应该有可行解
        status_str = pulp.LpStatus[status]
        is_feasible_status = status_str in ["Optimal", "Not Solved"]  # Not Solved 可能是超时但找到了可行解
        
        # 如果状态是 Infeasible，明确标记为无可行解，不提取变量值
        if status_str == "Infeasible":
            has_feasible_solution = False
            if verbose:
                print(f"[警告] 求解状态为 Infeasible（不可行），无法提取有效解")
            # 使用默认值填充layout（避免后续代码出错）
            for k, node in enumerate(nodes):
                layout[node.name] = (0.0, 0.0)
                rotations[node.name] = False
        else:
            # 尝试提取变量值
            non_zero_count = 0  # 统计非零位置的数量
            for k, node in enumerate(nodes):
                x_val = pulp.value(x[k])
                y_val = pulp.value(y[k])
                # 对于正方形chiplet，r[k] 是 None，旋转值固定为 False
                if r[k] is None:
                    r_val = 0.0  # 正方形chiplet不旋转
                else:
                    r_val = pulp.value(r[k])
                
                # 如果变量值存在且不为None，说明至少找到了可行解
                if x_val is not None and y_val is not None:
                    # 检查是否有非零值（避免所有位置都是0,0的情况）
                    if x_val != 0.0 or y_val != 0.0:
                        non_zero_count += 1
                    has_feasible_solution = True
                    layout[node.name] = (
                        x_val if x_val is not None else 0.0,
                        y_val if y_val is not None else 0.0,
                    )
                    rotations[node.name] = bool(r_val > 0.5) if r_val is not None else False
                else:
                    # 如果变量值为None，使用默认值
                    layout[node.name] = (0.0, 0.0)
                    rotations[node.name] = False
            
            # 如果所有位置都是(0,0)，可能是默认值而不是真实解
            if non_zero_count == 0 and has_feasible_solution:
                if verbose:
                    print(f"[警告] 所有chiplet的位置都是(0,0)，可能是默认值而不是真实解")
                    print(f"[警告] 求解状态: {status_str}")
                # 如果状态不是 Optimal，且所有位置都是0，可能没有真正的可行解
                if status_str != "Optimal":
                    has_feasible_solution = False
            
            # 如果状态不是 Optimal 但变量值都是 None，说明没有可行解
            if not has_feasible_solution and status_str != "Optimal":
                if verbose:
                    print(f"[警告] 求解状态为 {status_str}，且无法提取变量值，可能没有可行解")

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
        
        # 计算并打印硅桥互联边的共享边长度（只有在找到可行解时才输出）
        # 如果状态是 Infeasible，不应该输出共享边长度检查
        if has_feasible_solution and status_str != "Infeasible":
            print("\n" + "="*60)
            print("硅桥互联边的共享边长度检查:")
            print("="*60)
            
            # 获取硅桥互联边对：直接使用ctx.connected_pairs（它已经是硅桥互联边的索引对列表）
            # ctx.connected_pairs 在 build_placement_ilp_model 中被设置为 silicon_bridge_pairs
            silicon_bridge_pairs = ctx.connected_pairs if hasattr(ctx, 'connected_pairs') else []
            
            # 调试信息：打印找到的硅桥互联边数量
            if len(silicon_bridge_pairs) == 0:
                print("[DEBUG] 警告：未找到硅桥互联边！")
                print(f"[DEBUG] ctx.connected_pairs = {ctx.connected_pairs if hasattr(ctx, 'connected_pairs') else 'None'}")
                print(f"[DEBUG] ctx.edge_types = {ctx.edge_types if hasattr(ctx, 'edge_types') else 'None'}")
                print(f"[DEBUG] ctx.edges 数量 = {len(ctx.edges) if hasattr(ctx, 'edges') else 'None'}")
                # 尝试从edge_types中找出硅桥互联边
                if hasattr(ctx, 'edge_types'):
                    edge_types_map = ctx.edge_types
                    for (i, j), edge_type in edge_types_map.items():
                        if edge_type == "silicon_bridge":
                            if (i, j) not in silicon_bridge_pairs:
                                silicon_bridge_pairs.append((i, j))
                    print(f"[DEBUG] 从edge_types中找到 {len(silicon_bridge_pairs)} 条硅桥互联边")
            
            # 获取原始尺寸和旋转后的实际尺寸
            w_orig = {}
            h_orig = {}
            w_actual = {}
            h_actual = {}
            for k, node in enumerate(nodes):
                w_orig[k] = float(node.dimensions.get("x", 0.0))
                h_orig[k] = float(node.dimensions.get("y", 0.0))
                # 对于正方形chiplet，r[k] 是 None，旋转值固定为 0.0
                if r[k] is None:
                    r_val = 0.0  # 正方形chiplet不旋转
                else:
                    r_val = pulp.value(r[k])
                    if r_val is None:
                        r_val = 0.0
                if r_val >= 0.5:
                    w_actual[k] = h_orig[k]
                    h_actual[k] = w_orig[k]
                else:
                    w_actual[k] = w_orig[k]
                    h_actual[k] = h_orig[k]
            
            # 检查每个硅桥互联边
            min_shared_length = ctx.min_shared_length
            violation_count = 0
            
            for i, j in silicon_bridge_pairs:
                node_i = nodes[i]
                node_j = nodes[j]
                
                # 获取位置和尺寸
                x_i = pulp.value(x[i]) if x[i] is not None else 0.0
                y_i = pulp.value(y[i]) if y[i] is not None else 0.0
                x_j = pulp.value(x[j]) if x[j] is not None else 0.0
                y_j = pulp.value(y[j]) if y[j] is not None else 0.0
                
                if x_i is None:
                    x_i = 0.0
                if y_i is None:
                    y_i = 0.0
                if x_j is None:
                    x_j = 0.0
                if y_j is None:
                    y_j = 0.0
                
                w_i = w_actual[i]
                h_i = h_actual[i]
                w_j = w_actual[j]
                h_j = h_actual[j]
                
                # 获取z1和z2的值
                z1_val = None
                z2_val = None
                try:
                    z1_var = prob.variablesDict().get(f"z1_{i}_{j}")
                    z2_var = prob.variablesDict().get(f"z2_{i}_{j}")
                    if z1_var is not None:
                        z1_val = pulp.value(z1_var)
                    if z2_var is not None:
                        z2_val = pulp.value(z2_var)
                except:
                    pass
                
                # 计算共享边长度
                shared_length = 0.0
                adjacent_type = "未知"
                is_violation = False
                
                if z1_val is not None and z1_val >= 0.5:
                    # 水平相邻：共享边在垂直方向
                    adjacent_type = "水平相邻"
                    overlap_y_min = min((y_i + h_i) - y_j, (y_j + h_j) - y_i)
                    shared_length = max(0.0, overlap_y_min)
                    if shared_length < min_shared_length - 1e-6:
                        is_violation = True
                        violation_count += 1
                elif z2_val is not None and z2_val >= 0.5:
                    # 垂直相邻：共享边在水平方向
                    adjacent_type = "垂直相邻"
                    overlap_x_min = min((x_i + w_i) - x_j, (x_j + w_j) - x_i)
                    shared_length = max(0.0, overlap_x_min)
                    if shared_length < min_shared_length - 1e-6:
                        is_violation = True
                        violation_count += 1
                else:
                    # 不相邻（违反约束）
                    adjacent_type = "不相邻（违反约束！）"
                    is_violation = True
                    violation_count += 1
                
                status_mark = "❌ 违反约束" if is_violation else "✓"
                print(f"{status_mark} {node_i.name} <-> {node_j.name}:")
                print(f"    相邻类型: {adjacent_type}")
                print(f"    共享边长度: {shared_length:.4f} (要求 >= {min_shared_length:.4f})")
                print(f"    位置: {node_i.name}({x_i:.2f}, {y_i:.2f}, w={w_i:.2f}, h={h_i:.2f}), "
                      f"{node_j.name}({x_j:.2f}, {y_j:.2f}, w={w_j:.2f}, h={h_j:.2f})")
                # 格式化z1和z2的值
                z1_str = f"{z1_val:.2f}" if z1_val is not None else "N/A"
                z2_str = f"{z2_val:.2f}" if z2_val is not None else "N/A"
                print(f"    z1={z1_str}, z2={z2_str}")
                print()
            
            print("="*60)
            if violation_count == 0:
                print(f"✓ 所有 {len(silicon_bridge_pairs)} 条硅桥互联边都满足共享边长度约束")
            else:
                print(f"❌ 有 {violation_count} 条硅桥互联边违反共享边长度约束！")
            print("="*60 + "\n")

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
    edges: List[Tuple[str, str]] | List[Tuple[str, str, str]],
    silicon_bridge_edges: Optional[List[Tuple[str, str]]] = None,
    normal_edges: Optional[List[Tuple[str, str]]] = None,
    W: Optional[float] = None,
    H: Optional[float] = None,
    time_limit: int = 300,
    verbose: bool = True,
    min_shared_length: float = 0.0,
    minimize_bbox_area: bool = True,
    distance_weight: float = 1.0,
    area_weight: float = 1.0,
    silicon_bridge_weight: Optional[float] = 1,
    normal_edge_weight: Optional[float] = 1,
) -> ILPPlacementResult:
    """
    兼容旧接口的一站式求解函数：
    内部先调用 :func:`build_placement_ilp_model` 构建模型，
    然后用 :func:`solve_placement_ilp_from_model` 进行一次求解。
    
    参数说明参见 :func:`build_placement_ilp_model`。
    """

    ctx = build_placement_ilp_model(
        nodes=nodes,
        edges=edges,
        silicon_bridge_edges=silicon_bridge_edges,
        normal_edges=normal_edges,
        W=W,
        H=H,
        verbose=verbose,
        min_shared_length=min_shared_length,
        minimize_bbox_area=minimize_bbox_area,
        distance_weight=distance_weight,
        area_weight=area_weight,
        silicon_bridge_weight=silicon_bridge_weight,
        normal_edge_weight=normal_edge_weight,
    )

    return solve_placement_ilp_from_model(
        ctx,
        time_limit=time_limit,
        verbose=verbose,
    )
