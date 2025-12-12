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
    cx: Optional[Dict[int, pulp.LpVariable]] = None  # 中心坐标x（如果使用网格化模型）
    cy: Optional[Dict[int, pulp.LpVariable]] = None  # 中心坐标y（如果使用网格化模型）



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
    #print("w_orig, h_orig",w_orig, h_orig)
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
    #print("connected_pairs",connected_pairs)
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
    #print("W, H",W, H)
    # 计算grid数量
    grid_w = int(math.ceil(W / grid_size))
    grid_h = int(math.ceil(H / grid_size))
    # print("grid_w, grid_h",grid_w, grid_h)
    if verbose:
        print(f"网格化布局: grid_size={grid_size}, grid_w={grid_w}, grid_h={grid_h}")
        print(f"问题规模: {n} 个模块, {len(connected_pairs)} 对有连接的模块对")
    
    # ============ 步骤2: 创建ILP问题 ============
    prob = pulp.LpProblem("ChipletPlacementGrid", pulp.LpMinimize)
    
    # 大M常数
    M = 200
    
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
    
    # 约束Chiplet宽高为grid_size的整数倍（对齐网格）
    for k in range(n):
        # 1. 计算固定的网格数（向上取整，不是变量）
        w_grid_k = int(math.ceil(w_orig[k] / grid_size))
        h_grid_k = int(math.ceil(h_orig[k] / grid_size))
        
        # 2. 约束芯片宽度=网格数×grid_size（对齐网格）
        prob += w[k] == w_grid_k * grid_size, f"w_align_grid_{k}"
        prob += h[k] == h_grid_k * grid_size, f"h_align_grid_{k}"
        
        # 3. 额外约束：对齐后的尺寸不超过全局范围（可选，确保合理性）
        prob += w[k] <= W, f"w_max_{k}"
        prob += h[k] <= H, f"h_max_{k}"
    
    # 4.4 相邻约束：对于每对有连接的模块对 (i, j)
    for i, j in connected_pairs:
        # 规则1: 必须相邻，且只能选一种方式
        prob += z1[(i, j)] + z2[(i, j)] == 1, f"must_adjacent_{i}_{j}"
        
        # 规则2: 如果水平相邻，要么 i 在左，要么 i 在右
        prob += z1L[(i, j)] + z1R[(i, j)] == z1[(i, j)], f"horizontal_direction_{i}_{j}"
        
        # 规则3: 如果垂直相邻，要么 i 在下，要么 i 在上
        prob += z2D[(i, j)] + z2U[(i, j)] == z2[(i, j)], f"vertical_direction_{i}_{j}"
        
        # 规则4: 水平相邻的具体约束
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
        
        # 规则5: 垂直相邻的具体约束
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

    # 对于每对模块 (i, j)
    for i, j in all_pairs:
        # 修复漏洞1：恢复互斥约束（必须且仅需满足一个非重叠条件）
        prob += p_left[(i, j)] + p_right[(i, j)] + p_down[(i, j)] + p_up[(i, j)] == 1, \
            f"non_overlap_any_{i}_{j}"
        
        # 情况1: i 在 j 的左边（x_i + w_i <= x_j）
        # 正向约束：p_left=1 → x_i + w_i <= x_j
        prob += x[i] + w[i] - x[j] <= M * (1 - p_left[(i, j)]) , \
            f"non_overlap_left_{i}_{j}"
        # 修复漏洞2：正确的反向约束（x_i + w_i <= x_j → p_left=1）
        prob += x[j] - (x[i] + w[i]) <= M * p_left[(i, j)] , \
            f"non_overlap_left_rev_{i}_{j}"
        
        # 情况2: i 在 j 的右边（x_j + w_j <= x_i）
        prob += x[j] + w[j] - x[i] <= M * (1 - p_right[(i, j)]) , \
            f"non_overlap_right_{i}_{j}"
        prob += x[i] - (x[j] + w[j]) <= M * p_right[(i, j)] , \
            f"non_overlap_right_rev_{i}_{j}"
        
        # 情况3: i 在 j 的下边（y_i + h_i <= y_j）
        prob += y[i] + h[i] - y[j] <= M * (1 - p_down[(i, j)]) , \
            f"non_overlap_down_{i}_{j}"
        prob += y[j] - (y[i] + h[i]) <= M * p_down[(i, j)] , \
            f"non_overlap_down_rev_{i}_{j}"
        
        # 情况4: i 在 j 的上边（y_j + h_j <= y_i）
        prob += y[j] + h[j] - y[i] <= M * (1 - p_up[(i, j)]) , \
            f"non_overlap_up_{i}_{j}"
        prob += y[i] - (y[j] + h[j]) <= M * p_up[(i, j)] , \
            f"non_overlap_up_rev_{i}_{j}"

    if verbose:
        print(f"非重叠约束: {len(all_pairs)} 对模块对（所有模块对），M={M}（基板尺寸最大值）")
    
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
    
    # Big-M方法添加绝对值约束的辅助函数
    def add_absolute_value_constraint_big_m(
        prob: pulp.LpProblem,
        abs_var: pulp.LpVariable,
        orig_var: pulp.LpVariable,
        M: float,
        constraint_prefix: str,
    ) -> None:
        """
        使用Big-M方法添加绝对值约束：abs_var = |orig_var|
        
        实现方法（参考Big-M方法）：
        1. 创建二进制变量 is_positive，表示 orig_var >= 0
        2. 使用4个约束强制 abs_var = |orig_var|
           - 当 orig_var >= 0 时 (is_positive=1): abs_var = orig_var
           - 当 orig_var < 0 时 (is_positive=0): abs_var = -orig_var
        3. 使用2个约束强制 is_positive 的正确性
        """
        # 创建二进制变量：is_positive = 1 当且仅当 orig_var >= 0
        is_positive = pulp.LpVariable(f"{constraint_prefix}_is_positive", cat='Binary')
        
        # 约束1: 当 orig_var >= 0 时 (is_positive=1)，约束简化为: abs_var >= orig_var
        prob += abs_var >= orig_var - M * (1 - is_positive), f"{constraint_prefix}_abs_ge_orig"
        
        # 约束2: 当 orig_var >= 0 时 (is_positive=1)，约束简化为: abs_var <= orig_var
        prob += abs_var <= orig_var + M * (1 - is_positive), f"{constraint_prefix}_abs_le_orig"
        
        # 约束3: 当 orig_var < 0 时 (is_positive=0)，约束简化为: abs_var >= -orig_var
        prob += abs_var >= -orig_var - M * is_positive, f"{constraint_prefix}_abs_ge_neg_orig"
        
        # 约束4: 当 orig_var < 0 时 (is_positive=0)，约束简化为: abs_var <= -orig_var
        prob += abs_var <= -orig_var + M * is_positive, f"{constraint_prefix}_abs_le_neg_orig"
        
        # 约束5: 强制 is_positive = 1 当 orig_var >= 0
        prob += orig_var >= -M * (1 - is_positive), f"{constraint_prefix}_force_positive"
        
        # 约束6: 强制 is_positive = 0 当 orig_var < 0
        epsilon = 0.001
        prob += orig_var <= M * is_positive - epsilon, f"{constraint_prefix}_force_negative"
    
    for i, j in connected_pairs:
        dx_abs = pulp.LpVariable(f"dx_abs_{i}_{j}", lowBound=0)
        dy_abs = pulp.LpVariable(f"dy_abs_{i}_{j}", lowBound=0)
        
        # 创建辅助变量表示差值
        dx_diff = pulp.LpVariable(f"dx_diff_{i}_{j}", lowBound=-W, upBound=W, cat='Continuous')
        dy_diff = pulp.LpVariable(f"dy_diff_{i}_{j}", lowBound=-H, upBound=H, cat='Continuous')
        
        # 定义差值
        prob += dx_diff == cx[i] - cx[j], f"dx_diff_def_{i}_{j}"
        prob += dy_diff == cy[i] - cy[j], f"dy_diff_def_{i}_{j}"
        
        # 使用Big-M方法添加绝对值约束
        M_dx = W  # Big-M常数
        M_dy = H  # Big-M常数
        add_absolute_value_constraint_big_m(
            prob=prob,
            abs_var=dx_abs,
            orig_var=dx_diff,
            M=M_dx,
            constraint_prefix=f"dx_abs_{i}_{j}"
        )
        add_absolute_value_constraint_big_m(
            prob=prob,
            abs_var=dy_abs,
            orig_var=dy_diff,
            M=M_dy,
            constraint_prefix=f"dy_abs_{i}_{j}"
        )
        
        wirelength += dx_abs + dy_abs
    
    # 5.2 面积代理
    t = pulp.LpVariable("bbox_area_proxy_t", lowBound=0, upBound=W+H, cat=pulp.LpContinuous)
    # 4. 核心约束：让 t 合理代理面积（无冲突、紧凑）
    ## 约束1：t 至少 ≥ 宽/高（保证 t 不小于单个维度）
    prob += t >= bbox_w, "t_ge_width"
    prob += t >= bbox_h, "t_ge_height"
    
    ## 约束2：t 至少 ≥ 宽×高的“线性近似”（关键：用均值放大系数逼近面积）
    # 系数 alpha 取 0.5~1（平衡近似精度和约束紧凑性）
    alpha = 0.8
    prob += t >= alpha * (bbox_w + bbox_h), "t_ge_scaled_mean"
    
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
        cx=cx,
        cy=cy,
    )


def main():
    """
    主函数：使用网格化ILP模型进行单次求解并可视化结果。
    """
    from pathlib import Path
    import json
    
    # 设置参数
    grid_size = 1.0
    time_limit = 300
    min_shared_length = 0.1
    fixed_chiplet_idx = 0  # 固定第一个chiplet在中心
    
    # 输出目录
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    # JSON文件路径
    json_path = Path(__file__).parent.parent / "baseline" / "ICCAD23" / "test_input" / "2core.json"
    
    print("=" * 80)
    print("ILP单次求解测试")
    print("=" * 80)
    
    # 从JSON文件加载测试数据
    print(f"\n从JSON文件加载测试数据: {json_path}")
    if not json_path.exists():
        raise FileNotFoundError(f"JSON文件不存在: {json_path}")
    
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 处理JSON数据：格式 {"chiplets": [...], "connections": [...]}
    nodes = []
    edges = []
    
    if "chiplets" in data and isinstance(data["chiplets"], list):
        # 构建ChipletNode对象
        for chiplet_info in data["chiplets"]:
            name = chiplet_info.get("name", "")
            width = chiplet_info.get("width", 0.0)
            height = chiplet_info.get("height", 0.0)
            
            nodes.append(
                ChipletNode(
                    name=name,
                    dimensions={"x": width, "y": height},
                    phys=[],  # 测试输入中没有phys信息
                    power=chiplet_info.get("power", 0.0),
                )
            )
        
        # 提取连接关系
        if "connections" in data and isinstance(data["connections"], list):
            for conn in data["connections"]:
                if isinstance(conn, list) and len(conn) >= 2:
                    src, dst = conn[0], conn[1]
                    edges.append((src, dst))
    
    if len(nodes) == 0:
        raise ValueError("未能从JSON文件中加载任何chiplet节点")
    
    print(f"节点数量: {len(nodes)}")
    print(f"边数量: {len(edges)}")
    print(f"节点列表: {[n.name for n in nodes]}")
    print(f"边列表: {edges}")
    
    # 构建ILP模型
    print("\n构建ILP模型...")
    ctx = build_placement_ilp_model_grid(
        nodes=nodes,
        edges=edges,
        grid_size=grid_size,
        W=None,  # 自动估算
        H=None,  # 自动估算
        verbose=True,
        min_shared_length=min_shared_length,
        minimize_bbox_area=True,
        distance_weight=1.0,
        area_weight=0.1,
        fixed_chiplet_idx=fixed_chiplet_idx,
    )
    
    # 导出LP文件
    lp_file = output_dir / "ilp_model.lp"
    ctx.prob.writeLP(str(lp_file))
    print(f"LP模型文件已导出至: {lp_file}")
    
    # 求解
    print("\n开始求解...")
    result = solve_placement_ilp_from_model(
        ctx,
        time_limit=time_limit,
        verbose=True,
    )
    
    # 输出结果
    print("\n" + "=" * 80)
    print("求解结果")
    print("=" * 80)
    print(f"状态: {result.status}")
    print(f"求解时间: {result.solve_time:.2f} 秒")
    print(f"目标函数值: {result.objective_value:.2f}")
    print(f"边界框尺寸: {result.bounding_box[0]:.2f} x {result.bounding_box[1]:.2f}")
    
    print("\n布局结果:")
    for name, (x, y) in result.layout.items():
        rotated = result.rotations.get(name, False)
        rot_str = " (已旋转)" if rotated else ""
        print(f"  {name}: ({x:.2f}, {y:.2f}){rot_str}")
    
    # 打印所有变量的值
    if result.status == "Optimal":
        print("\n" + "=" * 80)
        print("变量值详情")
        print("=" * 80)
        
        import pulp
        
        # 1. 坐标变量 (x, y)
        print("\n【坐标变量】")
        for k in range(len(nodes)):
            x_val = pulp.value(ctx.x[k]) if ctx.x[k] is not None else None
            y_val = pulp.value(ctx.y[k]) if ctx.y[k] is not None else None
            print(f"  x[{k}] ({nodes[k].name}): {x_val}")
            print(f"  y[{k}] ({nodes[k].name}): {y_val}")
        
        # 2. 网格坐标变量 (x_grid, y_grid)
        print("\n【网格坐标变量】")
        for k in range(len(nodes)):
            x_grid_var = ctx.prob.variablesDict().get(f"x_grid_{k}")
            y_grid_var = ctx.prob.variablesDict().get(f"y_grid_{k}")
            x_grid_val = pulp.value(x_grid_var) if x_grid_var is not None else None
            y_grid_val = pulp.value(y_grid_var) if y_grid_var is not None else None
            print(f"  x_grid[{k}] ({nodes[k].name}): {x_grid_val}")
            print(f"  y_grid[{k}] ({nodes[k].name}): {y_grid_val}")
        
        # 3. 旋转变量 (r)
        print("\n【旋转变量】")
        for k in range(len(nodes)):
            r_val = pulp.value(ctx.r[k]) if ctx.r[k] is not None else None
            rotated_str = "是" if (r_val is not None and r_val > 0.5) else "否"
            print(f"  r[{k}] ({nodes[k].name}): {r_val} (旋转: {rotated_str})")
        
        # 4. 宽度和高度变量 (w, h) - 需要从prob中获取
        print("\n【尺寸变量】")
        for k in range(len(nodes)):
            w_var = None
            h_var = None
            for var_name, var in ctx.prob.variablesDict().items():
                if var_name == f"w_{k}":
                    w_var = var
                elif var_name == f"h_{k}":
                    h_var = var
            w_val = pulp.value(w_var) if w_var is not None else None
            h_val = pulp.value(h_var) if h_var is not None else None
            print(f"  w[{k}] ({nodes[k].name}): {w_val}")
            print(f"  h[{k}] ({nodes[k].name}): {h_val}")
        
        # 5. 中心坐标变量 (cx, cy)
        print("\n【中心坐标变量】")
        for k in range(len(nodes)):
            cx_val = pulp.value(ctx.cx[k]) if ctx.cx[k] is not None else None
            cy_val = pulp.value(ctx.cy[k]) if ctx.cy[k] is not None else None
            print(f"  cx[{k}] ({nodes[k].name}): {cx_val}")
            print(f"  cy[{k}] ({nodes[k].name}): {cy_val}")
        
        # 6. 相邻方式变量 (z1, z2, z1L, z1R, z2D, z2U)
        if len(ctx.connected_pairs) > 0:
            print("\n【相邻方式变量】")
            for i, j in ctx.connected_pairs:
                name_i = nodes[i].name
                name_j = nodes[j].name
                z1_val = pulp.value(ctx.z1[(i, j)]) if (i, j) in ctx.z1 else None
                z2_val = pulp.value(ctx.z2[(i, j)]) if (i, j) in ctx.z2 else None
                z1L_val = pulp.value(ctx.z1L[(i, j)]) if (i, j) in ctx.z1L else None
                z1R_val = pulp.value(ctx.z1R[(i, j)]) if (i, j) in ctx.z1R else None
                z2D_val = pulp.value(ctx.z2D[(i, j)]) if (i, j) in ctx.z2D else None
                z2U_val = pulp.value(ctx.z2U[(i, j)]) if (i, j) in ctx.z2U else None
                print(f"  模块对 ({name_i}, {name_j}):")
                print(f"    z1[{i},{j}] (水平相邻): {z1_val}")
                print(f"    z2[{i},{j}] (垂直相邻): {z2_val}")
                if z1_val is not None and z1_val > 0.5:
                    print(f"      z1L[{i},{j}] (i在左): {z1L_val}")
                    print(f"      z1R[{i},{j}] (i在右): {z1R_val}")
                if z2_val is not None and z2_val > 0.5:
                    print(f"      z2D[{i},{j}] (i在下): {z2D_val}")
                    print(f"      z2U[{i},{j}] (i在上): {z2U_val}")
        
        # 7. 非重叠约束变量 (p_left, p_right, p_down, p_up)
        print("\n【非重叠约束变量】")
        all_pairs = []
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                all_pairs.append((i, j))
        
        for i, j in all_pairs:
            name_i = nodes[i].name
            name_j = nodes[j].name
            # 从prob中查找这些变量
            p_left_var = ctx.prob.variablesDict().get(f"p_left_{i}_{j}")
            p_right_var = ctx.prob.variablesDict().get(f"p_right_{i}_{j}")
            p_down_var = ctx.prob.variablesDict().get(f"p_down_{i}_{j}")
            p_up_var = ctx.prob.variablesDict().get(f"p_up_{i}_{j}")
            
            p_left_val = pulp.value(p_left_var) if p_left_var is not None else None
            p_right_val = pulp.value(p_right_var) if p_right_var is not None else None
            p_down_val = pulp.value(p_down_var) if p_down_var is not None else None
            p_up_val = pulp.value(p_up_var) if p_up_var is not None else None
            
            print(f"  模块对 ({name_i}, {name_j}):")
            print(f"    p_left[{i},{j}]: {p_left_val}")
            print(f"    p_right[{i},{j}]: {p_right_val}")
            print(f"    p_down[{i},{j}]: {p_down_val}")
            print(f"    p_up[{i},{j}]: {p_up_val}")
        
        # 8. 边界框变量
        print("\n【边界框变量】")
        bbox_w_val = pulp.value(ctx.bbox_w) if ctx.bbox_w is not None else None
        bbox_h_val = pulp.value(ctx.bbox_h) if ctx.bbox_h is not None else None
        print(f"  bbox_w: {bbox_w_val}")
        print(f"  bbox_h: {bbox_h_val}")
        
        # 9. 其他辅助变量（shared_x, shared_y, dx_abs, dy_abs, bbox_min/max等）
        print("\n【其他辅助变量】")
        other_vars = []
        for var_name, var in ctx.prob.variablesDict().items():
            if var_name.startswith("shared_") or var_name.startswith("dx_abs_") or \
               var_name.startswith("dy_abs_") or var_name.startswith("bbox_") or \
               var_name.startswith("bbox_area_proxy"):
                # 排除排除约束相关的变量（这些会在后面单独打印）
                if not (var_name.startswith("dx_abs_pair_") or var_name.startswith("dy_abs_pair_")):
                    val = pulp.value(var) if var is not None else None
                    if val is not None:
                        other_vars.append((var_name, val))
        
        if other_vars:
            for var_name, val in sorted(other_vars):
                print(f"  {var_name}: {val}")
        else:
            print("  (无)")
        
        # 10. 排除解约束相关变量和约束（仅在第二次及以后的求解中打印）
        # 检查是否存在排除约束相关的变量
        exclude_vars = []
        for var_name, var in ctx.prob.variablesDict().items():
            if var_name.startswith("dx_abs_pair_") or var_name.startswith("dy_abs_pair_") or \
               var_name.startswith("dist_diff_pair_") or var_name.startswith("dist_diff_abs_pair_") or \
               var_name.startswith("diff_dist_pair_"):
                val = pulp.value(var) if var is not None else None
                if val is not None:
                    exclude_vars.append((var_name, val))
        
        if exclude_vars:
            print("\n" + "=" * 80)
            print("排除解约束相关变量和约束")
            print("=" * 80)
            
            # 10.1 打印排除约束相关的变量
            print("\n【排除约束变量】")
            
            # 按变量类型分组
            dx_abs_pair_vars = []
            dy_abs_pair_vars = []
            dist_diff_pair_vars = []
            dist_diff_abs_pair_vars = []
            diff_dist_pair_vars = []
            
            for var_name, val in exclude_vars:
                if var_name.startswith("dx_abs_pair_"):
                    dx_abs_pair_vars.append((var_name, val))
                elif var_name.startswith("dy_abs_pair_"):
                    dy_abs_pair_vars.append((var_name, val))
                elif var_name.startswith("dist_diff_pair_") and not var_name.startswith("dist_diff_abs_pair_"):
                    dist_diff_pair_vars.append((var_name, val))
                elif var_name.startswith("dist_diff_abs_pair_"):
                    dist_diff_abs_pair_vars.append((var_name, val))
                elif var_name.startswith("diff_dist_pair_"):
                    diff_dist_pair_vars.append((var_name, val))
            
            if dx_abs_pair_vars:
                print("\n  dx_abs_pair (chiplet对的x方向距离绝对值):")
                for var_name, val in sorted(dx_abs_pair_vars):
                    print(f"    {var_name}: {val}")
            
            if dy_abs_pair_vars:
                print("\n  dy_abs_pair (chiplet对的y方向距离绝对值):")
                for var_name, val in sorted(dy_abs_pair_vars):
                    print(f"    {var_name}: {val}")
            
            if dist_diff_pair_vars:
                print("\n  dist_diff_pair (chiplet对的距离差):")
                for var_name, val in sorted(dist_diff_pair_vars):
                    print(f"    {var_name}: {val}")
            
            if dist_diff_abs_pair_vars:
                print("\n  dist_diff_abs_pair (chiplet对的距离差绝对值):")
                for var_name, val in sorted(dist_diff_abs_pair_vars):
                    print(f"    {var_name}: {val}")
            
            if diff_dist_pair_vars:
                print("\n  diff_dist_pair (chiplet对的距离是否不同，二进制变量):")
                for var_name, val in sorted(diff_dist_pair_vars):
                    print(f"    {var_name}: {val}")
            
            # 10.2 打印排除约束相关的约束
            print("\n【排除约束】")
            exclude_constraints = []
            for constraint_name, constraint in ctx.prob.constraints.items():
                if constraint_name.startswith("dx_abs_pair_") or constraint_name.startswith("dy_abs_pair_") or \
                   constraint_name.startswith("dist_diff_pair_") or constraint_name.startswith("dist_diff_abs_pair_") or \
                   constraint_name.startswith("exclude_solution_dist_pair_"):
                    exclude_constraints.append(constraint_name)
            
            if exclude_constraints:
                print(f"  共找到 {len(exclude_constraints)} 个排除约束:")
                for constraint_name in sorted(exclude_constraints):
                    constraint = ctx.prob.constraints[constraint_name]
                    # 打印约束的简化形式
                    print(f"    {constraint_name}: {constraint}")
            else:
                print("  (未找到排除约束)")
        else:
            print("\n【排除解约束】")
            print("  (第一次求解，无排除约束)")
    
    # 可视化结果
    if result.status == "Optimal":
        print("\n生成可视化图表...")
        try:
            save_path = output_dir / "ilp_single_solution.png"
            
            draw_chiplet_diagram(
                nodes=nodes,
                edges=edges,
                layout=result.layout,
                save_path=str(save_path),
            )
            print(f"图表已保存至: {save_path}")
        except Exception as e:
            print(f"可视化失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n求解未达到最优解，跳过可视化")
    
    print("\n" + "=" * 80)
    print("完成")
    print("=" * 80)


if __name__ == "__main__":
    main()