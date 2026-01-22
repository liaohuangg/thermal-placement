"""
使用整数线性规划（ILP）进行 chiplet 布局优化（Gurobi版本）。

主要特性
--------
1. **相邻约束**：有连边的 chiplet 必须水平或垂直相邻（紧靠），并且共享边长度不少于给定下界。
2. **旋转约束**：每个 chiplet 允许 0°/90° 旋转，由二进制变量 ``r_k`` 控制宽高交换。
3. **非重叠约束**：任意两块 chiplet 之间不能重叠。
4. **外接方框约束**：显式构造覆盖所有 chiplet 的外接矩形，并对其宽高建立线性约束。
5. **多目标优化**：目标函数为

   ``β1 * wirelength + β2 * t``

   其中 ``wirelength`` 是所有连边中心点间的曼哈顿距离之和，
   ``t`` 是通过 AM–GM 凸近似得到的"面积代理"变量，用来近似外接矩形面积。

实现依赖 Gurobi Optimizer。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import gurobipy as gp
from gurobipy import GRB

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

    - `model`  : 已经构建好的 Gurobi 模型（包含变量、约束和目标函数，但可以继续加约束）
    - `x, y`  : 每个 chiplet 左下角坐标变量（用于排除解等约束）
    - `r`     : 每个 chiplet 的旋转变量
    - `z1, z2`: 每对有连边的 chiplet 的"相邻方式"变量（水平/垂直）
    - `z1L, z1R, z2D, z2U`: 每对有连边的 chiplet 的相对方向变量（左、右、下、上）
    - `connected_pairs` : 有连边的 (i, j) 索引列表（i < j）
    - `bbox_w, bbox_h` : 外接方框宽和高对应的变量
    - `W, H`  : 外接边界框的上界尺寸（建模阶段确定）
    - `grid_size` : 网格大小（如果使用网格化，否则为None）
    - `fixed_chiplet_idx` : 已废弃，不再使用固定芯粒约束（保留此字段以保持接口兼容性）
    """

    model: gp.Model
    nodes: List[ChipletNode]
    edges: List[Tuple[str, str]]

    x: Dict[int, gp.Var]
    y: Dict[int, gp.Var]
    r: Dict[int, gp.Var]
    z1: Dict[Tuple[int, int], gp.Var]
    z2: Dict[Tuple[int, int], gp.Var]
    z1L: Dict[Tuple[int, int], gp.Var]
    z1R: Dict[Tuple[int, int], gp.Var]
    z2D: Dict[Tuple[int, int], gp.Var]
    z2U: Dict[Tuple[int, int], gp.Var]
    connected_pairs: List[Tuple[int, int]]

    bbox_w: gp.Var
    bbox_h: gp.Var

    W: float
    H: float
    grid_size: Optional[float] = None
    fixed_chiplet_idx: Optional[int] = None
    cx: Optional[Dict[int, gp.Var]] = None  # 中心坐标x（如果使用网格化模型）
    cy: Optional[Dict[int, gp.Var]] = None  # 中心坐标y（如果使用网格化模型）

    # 为了兼容性，添加 prob 属性（指向 model）
    @property
    def prob(self):
        """为了兼容性，返回 model"""
        return self.model


def solve_placement_ilp_from_model(
    ctx: ILPModelContext,
    time_limit: int = 300,
    verbose: bool = True,
) -> ILPPlacementResult:
    """
    在已有 ILPModelContext 上调用求解器并抽取解。

    可以在多轮求解之间往 ctx.model 上继续添加约束（例如排除解约束）。
    """
    import time

    model = ctx.model
    nodes = ctx.nodes
    x, y, r = ctx.x, ctx.y, ctx.r
    W, H = ctx.W, ctx.H

    start_time = time.time()

    if verbose:
        print("\n开始求解ILP问题...")
        print(f"变量数量: {model.NumVars}")
        print(f"约束数量: {model.NumConstrs}")

    # 设置 Gurobi 参数
    model.setParam('TimeLimit', time_limit)
    model.setParam('OutputFlag', 1 if verbose else 0)
    model.setParam('LogToConsole', 1 if verbose else 0)

    try:
        model.optimize()
        solve_time = time.time() - start_time

        # 获取求解状态
        status_map = {
            GRB.OPTIMAL: "Optimal",
            GRB.INFEASIBLE: "Infeasible",
            GRB.UNBOUNDED: "Unbounded",
            GRB.TIME_LIMIT: "TimeLimit",
            GRB.INTERRUPTED: "Interrupted",
        }
        status_str = status_map.get(model.status, f"Unknown({model.status})")

        if verbose:
            print(f"\n求解状态: {status_str}")
            print(f"求解时间: {solve_time:.2f} 秒")
            if model.status == GRB.OPTIMAL:
                print(f"目标函数值: {model.ObjVal:.2f}")

        # 提取解
        layout: Dict[str, Tuple[float, float]] = {}
        rotations: Dict[str, bool] = {}
        for k, node in enumerate(nodes):
            if model.status == GRB.OPTIMAL:
                x_val = x[k].X if x[k] is not None else 0.0
                y_val = y[k].X if y[k] is not None else 0.0
                r_val = r[k].X if r[k] is not None else 0.0
                layout[node.name] = (x_val, y_val)
                rotations[node.name] = bool(r_val > 0.5)
            else:
                layout[node.name] = (0.0, 0.0)
                rotations[node.name] = False

        obj_value = (
            model.ObjVal if model.status == GRB.OPTIMAL else float("inf")
        )

        # 使用求解得到的 bbox_w / bbox_h 作为返回的边界框尺寸
        try:
            bw_val = ctx.bbox_w.X if ctx.bbox_w is not None else None
            bh_val = ctx.bbox_h.X if ctx.bbox_h is not None else None
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
            status=status_str,
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
    fixed_chiplet_idx: Optional[int] = None,  # 已废弃，不再使用固定芯粒约束
    min_aspect_ratio: float = 0.8,
    max_aspect_ratio: float = 1.25,
) -> ILPModelContext:
    """
    使用网格化ILP求解chiplet布局。
    
    与build_placement_ilp_model的主要区别：
    1. 坐标变量为整数（grid索引）
    2. 有链接关系的chiplet之间距离不能超过一个grid
    3. 共享边长不超过一个grid的共享范围，且不能小于min_shared_length
    
    参数
    ----
    grid_size: float
        网格大小（实际单位）
    fixed_chiplet_idx: Optional[int]
        已废弃，不再使用固定芯粒约束（保留此参数以保持接口兼容性）
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
    model = gp.Model("ChipletPlacementGrid")
    
    # 大M常数
    M = 200
    
    # ============ 步骤3: 定义变量 ============
    # 3.1 整数变量：每个chiplet在grid中的左下角坐标（grid索引）
    x_grid = {}
    y_grid = {}
    for k in range(n):
        x_grid[k] = model.addVar(
            name=f"x_grid_{k}",
            lb=0,
            ub=grid_w - 1,
            vtype=GRB.INTEGER
        )
        y_grid[k] = model.addVar(
            name=f"y_grid_{k}",
            lb=0,
            ub=grid_h - 1,
            vtype=GRB.INTEGER
        )
    
    # 3.2 连续变量：实际坐标（用于计算距离和共享边长）
    x = {}
    y = {}
    for k in range(n):
        x[k] = model.addVar(name=f"x_{k}", lb=0, ub=W, vtype=GRB.CONTINUOUS)
        y[k] = model.addVar(name=f"y_{k}", lb=0, ub=H, vtype=GRB.CONTINUOUS)
        # 约束：实际坐标 = grid坐标 * grid_size
        model.addConstr(x[k] == x_grid[k] * grid_size, name=f"x_grid_to_real_{k}")
        model.addConstr(y[k] == y_grid[k] * grid_size, name=f"y_grid_to_real_{k}")
    
    # 3.3 二进制变量：旋转变量
    r = {}
    for k in range(n):
        r[k] = model.addVar(name=f"r_{k}", vtype=GRB.BINARY)
    
    # 3.4 连续变量：实际宽度和高度
    w = {}
    h = {}
    for k in range(n):
        w_min = min(w_orig[k], h_orig[k])
        w_max = max(w_orig[k], h_orig[k])
        w[k] = model.addVar(name=f"w_{k}", lb=w_min, ub=w_max, vtype=GRB.CONTINUOUS)
        h[k] = model.addVar(name=f"h_{k}", lb=w_min, ub=w_max, vtype=GRB.CONTINUOUS)
    
    # 3.5 辅助变量：中心坐标
    cx = {}
    cy = {}
    for k in range(n):
        cx[k] = model.addVar(name=f"cx_{k}", lb=0, ub=W, vtype=GRB.CONTINUOUS)
        cy[k] = model.addVar(name=f"cy_{k}", lb=0, ub=H, vtype=GRB.CONTINUOUS)
    
    # 3.6 二进制变量：控制相邻方式
    z1 = {}
    z2 = {}
    z1L = {}
    z1R = {}
    z2D = {}
    z2U = {}
    
    # 3.7 二进制变量：控制相邻方式
    for i, j in connected_pairs:
        z1[(i, j)] = model.addVar(name=f"z1_{i}_{j}", vtype=GRB.BINARY)
        z2[(i, j)] = model.addVar(name=f"z2_{i}_{j}", vtype=GRB.BINARY)
        z1L[(i, j)] = model.addVar(name=f"z1L_{i}_{j}", vtype=GRB.BINARY)
        z1R[(i, j)] = model.addVar(name=f"z1R_{i}_{j}", vtype=GRB.BINARY)
        z2D[(i, j)] = model.addVar(name=f"z2D_{i}_{j}", vtype=GRB.BINARY)
        z2U[(i, j)] = model.addVar(name=f"z2U_{i}_{j}", vtype=GRB.BINARY)
    
    # ============ 步骤4: 定义约束 ============
    
    # 4.1 旋转约束
    for k in range(n):
        model.addConstr(
            w[k] == w_orig[k] + r[k] * (h_orig[k] - w_orig[k]),
            name=f"width_rotation_{k}"
        )
        model.addConstr(
            h[k] == h_orig[k] + r[k] * (w_orig[k] - h_orig[k]),
            name=f"height_rotation_{k}"
        )
    
    # 4.2 中心坐标定义
    for k in range(n):
        model.addConstr(cx[k] == x[k] + w[k] / 2.0, name=f"cx_def_{k}")
        model.addConstr(cy[k] == y[k] + h[k] / 2.0, name=f"cy_def_{k}")
    
    # 约束Chiplet宽高为grid_size的整数倍（对齐网格）
    for k in range(n):
        # 1. 计算固定的网格数（向上取整，不是变量）
        w_grid_k = int(math.ceil(w_orig[k] / grid_size))
        h_grid_k = int(math.ceil(h_orig[k] / grid_size))
        
        # 2. 约束芯片宽度=网格数×grid_size（对齐网格）
        model.addConstr(w[k] == w_grid_k * grid_size, name=f"w_align_grid_{k}")
        model.addConstr(h[k] == h_grid_k * grid_size, name=f"h_align_grid_{k}")
        
        # 3. 额外约束：对齐后的尺寸不超过全局范围（可选，确保合理性）
        model.addConstr(w[k] <= W, name=f"w_max_{k}")
        model.addConstr(h[k] <= H, name=f"h_max_{k}")
    
    # 4.4 相邻约束：对于每对有连接的模块对 (i, j)
    for i, j in connected_pairs:
        # 规则1: 必须相邻，且只能选一种方式
        model.addConstr(
            z1[(i, j)] + z2[(i, j)] == 1,
            name=f"must_adjacent_{i}_{j}"
        )
        
        # 规则2: 如果水平相邻，要么 i 在左，要么 i 在右
        model.addConstr(
            z1L[(i, j)] + z1R[(i, j)] == z1[(i, j)],
            name=f"horizontal_direction_{i}_{j}"
        )
        
        # 规则3: 如果垂直相邻，要么 i 在下，要么 i 在上
        model.addConstr(
            z2D[(i, j)] + z2U[(i, j)] == z2[(i, j)],
            name=f"vertical_direction_{i}_{j}"
        )
        
        # 规则4: 水平相邻的具体约束
        # 约束1：相邻方向的边界距离 ≤ grid_size
        # 如果 i 在左（z1L[i,j] = 1）：x_j - (x_i + w_i) <= grid_size（距离不超过1个grid）
        model.addConstr(
            x[j] - (x[i] + w[i]) <= grid_size + M * (1 - z1L[(i, j)]),
            name=f"horizontal_left_dist_{i}_{j}"
        )
        model.addConstr(
            x[j] - (x[i] + w[i]) >= 0 - M * (1 - z1L[(i, j)]),
            name=f"horizontal_left_dist_lb_{i}_{j}"
        )
        # 如果 i 在右（z1R[i,j] = 1）：x_i - (x_j + w_j) <= grid_size（距离不超过1个grid）
        model.addConstr(
            x[i] - (x[j] + w[j]) <= grid_size + M * (1 - z1R[(i, j)]),
            name=f"horizontal_right_dist_{i}_{j}"
        )
        model.addConstr(
            x[i] - (x[j] + w[j]) >= 0 - M * (1 - z1R[(i, j)]),
            name=f"horizontal_right_dist_lb_{i}_{j}"
        )
        
        # 约束2：垂直方向（与相邻方向垂直）的重叠长度 >= min_shared_length
        # 重叠长度 = min(y[i] + h[i], y[j] + h[j]) - max(y[i], y[j])
        # 首先确保有重叠：y[i] < y[j] + h[j] 且 y[j] < y[i] + h[i]
        model.addConstr(
            y[i] - (y[j] + h[j]) <= M * (1 - z1[(i, j)]),
            name=f"horizontal_overlap_y1_{i}_{j}"
        )
        model.addConstr(
            y[j] - (y[i] + h[i]) <= M * (1 - z1[(i, j)]),
            name=f"horizontal_overlap_y2_{i}_{j}"
        )
        
        # 共享边长度（垂直方向的重叠长度）
        max_shared_y = max(h_orig[i], h_orig[j])
        shared_y = model.addVar(
            name=f"shared_y_{i}_{j}",
            lb=0,
            ub=max_shared_y,
            vtype=GRB.CONTINUOUS
        )
        model.addConstr(
            shared_y <= (y[i] + h[i]) - y[j] + M * (1 - z1[(i, j)]),
            name=f"shared_y_ub1_{i}_{j}"
        )
        model.addConstr(
            shared_y <= (y[j] + h[j]) - y[i] + M * (1 - z1[(i, j)]),
            name=f"shared_y_ub2_{i}_{j}"
        )
        model.addConstr(
            shared_y >= min_shared_length - M * (1 - z1[(i, j)]),
            name=f"shared_y_min_{i}_{j}"
        )
        model.addConstr(
            shared_y <= M * z1[(i, j)],
            name=f"shared_y_zero_{i}_{j}"
        )
        
        # 规则5: 垂直相邻的具体约束
        # 约束1：相邻方向的边界距离 ≤ grid_size
        # 如果 i 在下（z2D[i,j] = 1）：y_j - (y_i + h_i) <= grid_size（距离不超过1个grid）
        model.addConstr(
            y[j] - (y[i] + h[i]) <= grid_size + M * (1 - z2D[(i, j)]),
            name=f"vertical_down_dist_{i}_{j}"
        )
        model.addConstr(
            y[j] - (y[i] + h[i]) >= 0 - M * (1 - z2D[(i, j)]),
            name=f"vertical_down_dist_lb_{i}_{j}"
        )
        # 如果 i 在上（z2U[i,j] = 1）：y_i - (y_j + h_j) <= grid_size（距离不超过1个grid）
        model.addConstr(
            y[i] - (y[j] + h[j]) <= grid_size + M * (1 - z2U[(i, j)]),
            name=f"vertical_up_dist_{i}_{j}"
        )
        model.addConstr(
            y[i] - (y[j] + h[j]) >= 0 - M * (1 - z2U[(i, j)]),
            name=f"vertical_up_dist_lb_{i}_{j}"
        )
        
        # 约束2：水平方向（与相邻方向垂直）的重叠长度 >= min_shared_length
        # 重叠长度 = min(x[i] + w[i], x[j] + w[j]) - max(x[i], x[j])
        # 首先确保有重叠：x[i] < x[j] + w[j] 且 x[j] < x[i] + w[i]
        model.addConstr(
            x[i] - (x[j] + w[j]) <= M * (1 - z2[(i, j)]),
            name=f"vertical_overlap_x1_{i}_{j}"
        )
        model.addConstr(
            x[j] - (x[i] + w[i]) <= M * (1 - z2[(i, j)]),
            name=f"vertical_overlap_x2_{i}_{j}"
        )
        
        # 共享边长度（水平方向的重叠长度）
        max_shared_x = max(w_orig[i], w_orig[j])
        shared_x = model.addVar(
            name=f"shared_x_{i}_{j}",
            lb=0,
            ub=max_shared_x,
            vtype=GRB.CONTINUOUS
        )
        model.addConstr(
            shared_x <= (x[i] + w[i]) - x[j] + M * (1 - z2[(i, j)]),
            name=f"shared_x_ub1_{i}_{j}"
        )
        model.addConstr(
            shared_x <= (x[j] + w[j]) - x[i] + M * (1 - z2[(i, j)]),
            name=f"shared_x_ub2_{i}_{j}"
        )
        model.addConstr(
            shared_x >= min_shared_length - M * (1 - z2[(i, j)]),
            name=f"shared_x_min_{i}_{j}"
        )
        model.addConstr(
            shared_x <= M * z2[(i, j)],
            name=f"shared_x_zero_{i}_{j}"
        )
    
    # 4.5 边界约束
    for k in range(n):
        model.addConstr(x[k] + w[k] <= W, name=f"boundary_x_{k}")
        model.addConstr(y[k] + h[k] <= H, name=f"boundary_y_{k}")
    
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
            p_left[(i, j)] = model.addVar(name=f"p_left_{i}_{j}", vtype=GRB.BINARY)
            p_right[(i, j)] = model.addVar(name=f"p_right_{i}_{j}", vtype=GRB.BINARY)
            p_down[(i, j)] = model.addVar(name=f"p_down_{i}_{j}", vtype=GRB.BINARY)
            p_up[(i, j)] = model.addVar(name=f"p_up_{i}_{j}", vtype=GRB.BINARY)

    # 对于每对模块 (i, j)
    for i, j in all_pairs:
        # 修复漏洞1：恢复互斥约束（必须且仅需满足一个非重叠条件）
        model.addConstr(
            p_left[(i, j)] + p_right[(i, j)] + p_down[(i, j)] + p_up[(i, j)] == 1,
            name=f"non_overlap_any_{i}_{j}"
        )
        
        # 情况1: i 在 j 的左边（x_i + w_i <= x_j）
        # 正向约束：p_left=1 → x_i + w_i <= x_j
        model.addConstr(
            x[i] + w[i] - x[j] <= M * (1 - p_left[(i, j)]),
            name=f"non_overlap_left_{i}_{j}"
        )
        # 修复漏洞2：正确的反向约束（x_i + w_i <= x_j → p_left=1）
        model.addConstr(
            x[j] - (x[i] + w[i]) <= M * p_left[(i, j)],
            name=f"non_overlap_left_rev_{i}_{j}"
        )
        
        # 情况2: i 在 j 的右边（x_j + w_j <= x_i）
        model.addConstr(
            x[j] + w[j] - x[i] <= M * (1 - p_right[(i, j)]),
            name=f"non_overlap_right_{i}_{j}"
        )
        model.addConstr(
            x[i] - (x[j] + w[j]) <= M * p_right[(i, j)],
            name=f"non_overlap_right_rev_{i}_{j}"
        )
        
        # 情况3: i 在 j 的下边（y_i + h_i <= y_j）
        model.addConstr(
            y[i] + h[i] - y[j] <= M * (1 - p_down[(i, j)]),
            name=f"non_overlap_down_{i}_{j}"
        )
        model.addConstr(
            y[j] - (y[i] + h[i]) <= M * p_down[(i, j)],
            name=f"non_overlap_down_rev_{i}_{j}"
        )
        
        # 情况4: i 在 j 的上边（y_j + h_j <= y_i）
        model.addConstr(
            y[j] + h[j] - y[i] <= M * (1 - p_up[(i, j)]),
            name=f"non_overlap_up_{i}_{j}"
        )
        model.addConstr(
            y[i] - (y[j] + h[j]) <= M * p_up[(i, j)],
            name=f"non_overlap_up_rev_{i}_{j}"
        )

    if verbose:
        print(f"非重叠约束: {len(all_pairs)} 对模块对（所有模块对），M={M}（基板尺寸最大值）")
    
    # 4.7 外接方框约束
    bbox_min_x = model.addVar(name="bbox_min_x", lb=0, ub=W, vtype=GRB.CONTINUOUS)
    bbox_max_x = model.addVar(name="bbox_max_x", lb=0, ub=W, vtype=GRB.CONTINUOUS)
    bbox_min_y = model.addVar(name="bbox_min_y", lb=0, ub=H, vtype=GRB.CONTINUOUS)
    bbox_max_y = model.addVar(name="bbox_max_y", lb=0, ub=H, vtype=GRB.CONTINUOUS)
    bbox_w = model.addVar(name="bbox_w", lb=0, ub=W, vtype=GRB.CONTINUOUS)
    bbox_h = model.addVar(name="bbox_h", lb=0, ub=H, vtype=GRB.CONTINUOUS)
    
    for k in range(n):
        model.addConstr(bbox_min_x <= x[k], name=f"bbox_min_x_{k}")
        model.addConstr(bbox_max_x >= x[k] + w[k], name=f"bbox_max_x_{k}")
        model.addConstr(bbox_min_y <= y[k], name=f"bbox_min_y_{k}")
        model.addConstr(bbox_max_y >= y[k] + h[k], name=f"bbox_max_y_{k}")
    
    model.addConstr(bbox_w == bbox_max_x - bbox_min_x, name="bbox_w_def")
    model.addConstr(bbox_h == bbox_max_y - bbox_min_y, name="bbox_h_def")
    
    # 4.8 长宽比约束
    if min_aspect_ratio is not None:
        # bbox_w / bbox_h >= min_aspect_ratio
        # 转换为线性约束: bbox_w >= min_aspect_ratio * bbox_h
        model.addConstr(
            bbox_w >= min_aspect_ratio * bbox_h,
            name="aspect_ratio_min"
        )
        if verbose:
            print(f"长宽比约束: bbox_w/bbox_h >= {min_aspect_ratio}")
    
    if max_aspect_ratio is not None:
        # bbox_w / bbox_h <= max_aspect_ratio
        # 转换为线性约束: bbox_w <= max_aspect_ratio * bbox_h
        model.addConstr(
            bbox_w <= max_aspect_ratio * bbox_h,
            name="aspect_ratio_max"
        )
        if verbose:
            print(f"长宽比约束: bbox_w/bbox_h <= {max_aspect_ratio}")
    
    # 5.3 长宽比优化目标（最小化长宽比与理想值的偏差）
    # 理想长宽比设为1.0（正方形），使用 |bbox_w/bbox_h - 1| 的线性近似
    aspect_ratio_penalty = None
    if minimize_bbox_area:  # 只在最小化面积时考虑长宽比优化
        # 使用辅助变量表示长宽比偏差
        # 由于 bbox_w/bbox_h 是非线性的，我们使用 |bbox_w - bbox_h| 作为近似
        # 这鼓励长宽接近，从而接近正方形
        aspect_ratio_diff = model.addVar(
            name="aspect_ratio_diff",
            lb=0,
            ub=max(W, H),
            vtype=GRB.CONTINUOUS
        )
        # |bbox_w - bbox_h| <= aspect_ratio_diff
        model.addConstr(
            aspect_ratio_diff >= bbox_w - bbox_h,
            name="aspect_ratio_diff_ge_w_minus_h"
        )
        model.addConstr(
            aspect_ratio_diff >= bbox_h - bbox_w,
            name="aspect_ratio_diff_ge_h_minus_w"
        )
        aspect_ratio_penalty = aspect_ratio_diff
    
    # ============ 步骤5: 定义目标函数 ============
    # 5.1 线长（曼哈顿距离）
    wirelength = 0
    
    # Big-M方法添加绝对值约束的辅助函数
    def add_absolute_value_constraint_big_m(
        model: gp.Model,
        abs_var: gp.Var,
        orig_var: gp.Var,
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
        is_positive = model.addVar(
            name=f"{constraint_prefix}_is_positive",
            vtype=GRB.BINARY
        )
        
        # 约束1: 当 orig_var >= 0 时 (is_positive=1)，约束简化为: abs_var >= orig_var
        model.addConstr(
            abs_var >= orig_var - M * (1 - is_positive),
            name=f"{constraint_prefix}_abs_ge_orig"
        )
        
        # 约束2: 当 orig_var >= 0 时 (is_positive=1)，约束简化为: abs_var <= orig_var
        model.addConstr(
            abs_var <= orig_var + M * (1 - is_positive),
            name=f"{constraint_prefix}_abs_le_orig"
        )
        
        # 约束3: 当 orig_var < 0 时 (is_positive=0)，约束简化为: abs_var >= -orig_var
        model.addConstr(
            abs_var >= -orig_var - M * is_positive,
            name=f"{constraint_prefix}_abs_ge_neg_orig"
        )
        
        # 约束4: 当 orig_var < 0 时 (is_positive=0)，约束简化为: abs_var <= -orig_var
        model.addConstr(
            abs_var <= -orig_var + M * is_positive,
            name=f"{constraint_prefix}_abs_le_neg_orig"
        )
        
        # 约束5: 强制 is_positive = 1 当 orig_var >= 0
        model.addConstr(
            orig_var >= -M * (1 - is_positive),
            name=f"{constraint_prefix}_force_positive"
        )
        
        # 约束6: 强制 is_positive = 0 当 orig_var < 0
        epsilon = 0.001
        model.addConstr(
            orig_var <= M * is_positive - epsilon,
            name=f"{constraint_prefix}_force_negative"
        )
    
    for i, j in connected_pairs:
        dx_abs = model.addVar(name=f"dx_abs_{i}_{j}", lb=0, vtype=GRB.CONTINUOUS)
        dy_abs = model.addVar(name=f"dy_abs_{i}_{j}", lb=0, vtype=GRB.CONTINUOUS)
        
        # 创建辅助变量表示差值
        dx_diff = model.addVar(
            name=f"dx_diff_{i}_{j}",
            lb=-W,
            ub=W,
            vtype=GRB.CONTINUOUS
        )
        dy_diff = model.addVar(
            name=f"dy_diff_{i}_{j}",
            lb=-H,
            ub=H,
            vtype=GRB.CONTINUOUS
        )
        
        # 定义差值
        model.addConstr(dx_diff == cx[i] - cx[j], name=f"dx_diff_def_{i}_{j}")
        model.addConstr(dy_diff == cy[i] - cy[j], name=f"dy_diff_def_{i}_{j}")
        
        # 使用Big-M方法添加绝对值约束
        M_dx = W  # Big-M常数
        M_dy = H  # Big-M常数
        add_absolute_value_constraint_big_m(
            model=model,
            abs_var=dx_abs,
            orig_var=dx_diff,
            M=M_dx,
            constraint_prefix=f"dx_abs_{i}_{j}"
        )
        add_absolute_value_constraint_big_m(
            model=model,
            abs_var=dy_abs,
            orig_var=dy_diff,
            M=M_dy,
            constraint_prefix=f"dy_abs_{i}_{j}"
        )
        
        wirelength += dx_abs + dy_abs
    
    # 5.2 面积代理
    t = model.addVar(
        name="bbox_area_proxy_t",
        lb=0,
        ub=W+H,
        vtype=GRB.CONTINUOUS
    )
    # 4. 核心约束：让 t 合理代理面积（无冲突、紧凑）
    ## 约束1：t 至少 ≥ 宽/高（保证 t 不小于单个维度）
    model.addConstr(t >= bbox_w, name="t_ge_width")
    model.addConstr(t >= bbox_h, name="t_ge_height")
    
    ## 约束2：t 至少 ≥ 宽×高的"线性近似"（关键：用均值放大系数逼近面积）
    # 系数 alpha 取 0.5~1（平衡近似精度和约束紧凑性）
    alpha = 0.8
    model.addConstr(t >= alpha * (bbox_w + bbox_h), name="t_ge_scaled_mean")
    
    # 5.4 目标函数
    aspect_ratio_weight = 0.05  # 长宽比优化权重（相对于其他项）
    
    if minimize_bbox_area:
        if aspect_ratio_penalty is not None:
            objective = (distance_weight * wirelength + 
                        area_weight * t + 
                        aspect_ratio_weight * aspect_ratio_penalty)
            model.setObjective(objective, GRB.MINIMIZE)
            if verbose:
                print(f"目标函数: {distance_weight} * wirelength + {area_weight} * area_proxy + {aspect_ratio_weight} * aspect_ratio_penalty")
        else:
            model.setObjective(distance_weight * wirelength + area_weight * t, GRB.MINIMIZE)
            if verbose:
                print(f"目标函数: {distance_weight} * wirelength + {area_weight} * area_proxy")
    else:
        model.setObjective(distance_weight * wirelength, GRB.MINIMIZE)
        if verbose:
            print(f"目标函数: {distance_weight} * wirelength")
    
    return ILPModelContext(
        model=model,
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
    fixed_chiplet_idx = None  # 不再使用固定芯粒约束
    
    # 输出目录
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    # JSON文件路径
    json_path = Path(__file__).parent.parent / "baseline" / "ICCAD23" / "test_input" / "2core.json"
    
    print("=" * 80)
    print("ILP单次求解测试 (Gurobi版本)")
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
    lp_file = output_dir / "ilp_model_gurobi.lp"
    ctx.model.write(str(lp_file))
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
    
    # 可视化结果
    if result.status == "Optimal":
        print("\n生成可视化图表...")
        try:
            save_path = output_dir / "ilp_single_solution_gurobi.png"
            
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

