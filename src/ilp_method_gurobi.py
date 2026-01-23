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
    - `silicon_bridge_pairs` : silicon_bridge 连接的 (i, j) 索引列表（i < j）
    - `standard_pairs` : standard 连接的 (i, j) 索引列表（i < j）
    - `bbox_w, bbox_h` : 外接方框宽和高对应的变量
    - `W, H`  : 外接边界框的上界尺寸（建模阶段确定）
    - `grid_size` : 网格大小（如果使用网格化，否则为None）
    - `fixed_chiplet_idx` : 已废弃，不再使用固定芯粒约束（保留此字段以保持接口兼容性）
    """

    model: gp.Model
    nodes: List[ChipletNode]
    edges: List[Tuple[str, str, int]]  # (src, dst, connection_type): 1=silicon_bridge, 0=standard

    x_grid_var: Dict[int, gp.Var]
    y_grid_var: Dict[int, gp.Var]
    r: Dict[int, gp.Var]
    z1: Dict[Tuple[int, int], gp.Var]
    z2: Dict[Tuple[int, int], gp.Var]
    z1L: Dict[Tuple[int, int], gp.Var]
    z1R: Dict[Tuple[int, int], gp.Var]
    z2D: Dict[Tuple[int, int], gp.Var]
    z2U: Dict[Tuple[int, int], gp.Var]
    silicon_bridge_pairs: List[Tuple[int, int]]
    standard_pairs: List[Tuple[int, int]]

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
    x_grid_var, y_grid_var, r = ctx.x_grid_var, ctx.y_grid_var, ctx.r
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
                x_val = x_grid_var[k].X if x_grid_var[k] is not None else 0.0
                y_val = y_grid_var[k].X if y_grid_var[k] is not None else 0.0
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


def build_placement_ilp_model_grid(
    nodes: List[ChipletNode],
    edges: List[Tuple[str, str, int]],  # (src, dst, connection_type): 1=silicon_bridge, 0=standard
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
    # 1.1 读取芯片原始尺寸 chiplet_w_orig chiplet_h_orig 表示chiplet的长宽
    chiplet_w_orig = {}
    chiplet_h_orig = {}
    for i, node in enumerate(nodes):
        chiplet_w_orig[i] = float(node.dimensions.get("x", 0.0))
        chiplet_h_orig[i] = float(node.dimensions.get("y", 0.0))
        print(f"node {i} w: {chiplet_w_orig[i]}, h: {chiplet_h_orig[i]}")
    
    chiplet_w_orig_grid = {} # 网格化后的尺寸
    chiplet_h_orig_grid = {} # 网格化后的尺寸
    for i, node in enumerate(nodes):
        chiplet_w_orig_grid[i] = int(chiplet_w_orig[i] / grid_size)
        chiplet_h_orig_grid[i] = int(chiplet_h_orig[i] / grid_size)
        print(f"node {i} chiplet_w_orig_grid: {chiplet_w_orig_grid[i]}, chiplet_h_orig_grid: {chiplet_h_orig_grid[i]}")
    
    # 1.2 找到所有有边连接的模块对，并区分连接类型
    silicon_bridge_pairs = []  # silicon_bridge 连接的模块对（需要紧邻约束）
    standard_pairs = []  # standard 连接的模块对（可以更宽松的约束）
    
    for edge in edges:
        # edges 必须是三元组格式 (src, dst, conn_type)
        if len(edge) != 3:
            raise ValueError(f"边格式错误：每条边必须是三元组 (src, dst, conn_type)，其中 conn_type 为 0 (standard) 或 1 (silicon_bridge)。当前边: {edge}")     
        src_name, dst_name, conn_type = edge
        
        if conn_type not in [0, 1]:
            raise ValueError(f"连接类型错误：conn_type 必须是 0 (standard) 或 1 (silicon_bridge)。当前值: {conn_type}，边: {edge}")
        
        if src_name in name_to_idx and dst_name in name_to_idx:
            i = name_to_idx[src_name]
            j = name_to_idx[dst_name]
            if i != j:
                if i > j:
                    i, j = j, i
                pair = (i, j)
                # 根据连接类型分类
                if conn_type == 1:  # silicon_bridge
                    if pair not in silicon_bridge_pairs:
                        silicon_bridge_pairs.append(pair)
                else:  # standard (conn_type == 0)
                    if pair not in standard_pairs:
                        standard_pairs.append(pair)
    
    silicon_bridge_pairs = list(set(silicon_bridge_pairs))
    standard_pairs = list(set(standard_pairs))
    all_connected_pairs = silicon_bridge_pairs + standard_pairs  # 用于统计和线长计算
    
    if verbose:
        print(f"连接类型统计: silicon_bridge={len(silicon_bridge_pairs)}, standard={len(standard_pairs)}, 总计={len(all_connected_pairs)}")
    
    # 1.3 估算芯片边界框尺寸,使用网格化后的尺寸
    if W is None or H is None:
        total_area = sum(chiplet_w_orig_grid[i] * chiplet_h_orig_grid[i] for i in range(n))
        print(f"total_area: {total_area}")
        estimated_side = math.ceil(math.sqrt(total_area * 2))
        print(f"estimated_side: {estimated_side}")
        if W is None:
            W = estimated_side * 3
        if H is None:
            H = estimated_side * 3
        print(f"Estimated W: {W}, H: {H}")
    
    if verbose:
        print(f"网格化布局: grid_size={grid_size}, W={W}, grid_h={H}")
        print(f"问题规模: {n} 个模块, {len(all_connected_pairs)} 对有连接的模块对")
    
    # 1.4 计算共享边长(硅桥互联硬约束)网格数
    min_shared_length_grid = int(math.ceil(min_shared_length / grid_size))
    
    # ============ 步骤2: 创建ILP问题 ============
    model = gp.Model("ChipletPlacementGrid")
    
    # 大M常数
    M = max(W, H) * 3 # 确保 M 足够覆盖任何两个组件之间的距离

    # ============ 步骤3: 定义变量 ============
    # 3.1 二进制变量：旋转变量
    r = {}
    for k in range(n):
        r[k] = model.addVar(name=f"r_{k}", vtype=GRB.BINARY)
    
    # 3.2 整数变量：实际宽度和高度
    w_var = {}
    h_var = {}
    for k in range(n):
        w_min = min(chiplet_w_orig_grid[k], chiplet_h_orig_grid[k])
        w_max = max(chiplet_w_orig_grid[k], chiplet_h_orig_grid[k])
        w_var[k] = model.addVar(name=f"w_var_{k}", lb=w_min, ub=w_max, vtype=GRB.INTEGER)
        h_var[k] = model.addVar(name=f"h_var_{k}", lb=w_min, ub=w_max, vtype=GRB.INTEGER)
    
    # 3.3 整数变量：每个chiplet在grid中的左下角坐标（grid索引）
    # 坐标的上下界为0到W - w_var[k]和0到H - h_var[k] 不能超过边界框 - 实际长宽（考虑旋转）
    x_grid_var = {}
    y_grid_var = {}
    for k in range(n):
        x_grid_var[k] = model.addVar(
            name=f"x_grid_var_{k}",
            lb=0,
            ub=W,
            vtype=GRB.INTEGER
        )
        y_grid_var[k] = model.addVar(
            name=f"y_grid_var_{k}",
            lb=0,
            ub=H,
            vtype=GRB.INTEGER
        )

    # 3.4 辅助变量：中心坐标
    cx = {}
    cy = {}
    for k in range(n):
        cx[k] = model.addVar(name=f"cx_{k}", lb=0, ub=W, vtype=GRB.INTEGER)
        cy[k] = model.addVar(name=f"cy_{k}", lb=0, ub=H, vtype=GRB.INTEGER)
    
    # 3.5 二进制变量：控制相邻方式
    z1 = {}
    z2 = {}
    z1L = {}
    z1R = {}
    z2D = {}
    z2U = {}
    
    # 3.6 二进制变量：控制相邻方式（仅对 silicon_bridge 连接需要）
    for i, j in silicon_bridge_pairs:
        z1[(i, j)] = model.addVar(name=f"z1_{i}_{j}", vtype=GRB.BINARY)
        z2[(i, j)] = model.addVar(name=f"z2_{i}_{j}", vtype=GRB.BINARY)
        z1L[(i, j)] = model.addVar(name=f"z1L_{i}_{j}", vtype=GRB.BINARY)
        z1R[(i, j)] = model.addVar(name=f"z1R_{i}_{j}", vtype=GRB.BINARY)
        z2D[(i, j)] = model.addVar(name=f"z2D_{i}_{j}", vtype=GRB.BINARY)
        z2U[(i, j)] = model.addVar(name=f"z2U_{i}_{j}", vtype=GRB.BINARY)
    
    # ============ 步骤4: 定义约束 ============

    # 4.1 旋转约束 & 边界约束
    for k in range(n):
        # 旋转约束  
        model.addConstr(
            w_var[k] == chiplet_w_orig_grid[k] + r[k] * (chiplet_h_orig_grid[k] - chiplet_w_orig_grid[k]),
            name=f"width_rotation_{k}"
        )
        model.addConstr(
            h_var[k] == chiplet_h_orig_grid[k] + r[k] * (chiplet_w_orig_grid[k] - chiplet_h_orig_grid[k]),
            name=f"height_rotation_{k}"
        )
        # 边界约束
        model.addConstr(x_grid_var[k] <= W - w_var[k], name=f"x_grid_var_ub_{k}")
        model.addConstr(y_grid_var[k] <= H - h_var[k], name=f"y_grid_var_ub_{k}")
    
    # 4.2 中心坐标定义
    # for k in range(n):
    #     model.addConstr(cx[k] == x_grid_var[k] + w_var[k] / 2.0, name=f"cx_def_{k}")
    #     model.addConstr(cy[k] == y_grid_var[k] + h_var[k] / 2.0, name=f"cy_def_{k}")
   
    # 4.3 相邻约束：根据连接类型应用不同的约束
    # 4.3.1 silicon_bridge 连接：必须紧邻（当前约束）
    for i, j in silicon_bridge_pairs:
        # 规则1: 必须相邻，且只能选一种方式
        model.addConstr(
            z1[(i, j)] + z2[(i, j)] == 1,
            name=f"must_adjacent_sb_{i}_{j}"
        )
        
        # 规则2: 如果水平相邻，要么 i 在左，要么 i 在右
        model.addConstr(
            z1L[(i, j)] + z1R[(i, j)] == z1[(i, j)],
            name=f"horizontal_direction_sb_{i}_{j}"
        )
        
        # 规则3: 如果垂直相邻，要么 i 在下，要么 i 在上
        model.addConstr(
            z2D[(i, j)] + z2U[(i, j)] == z2[(i, j)],
            name=f"vertical_direction_sb_{i}_{j}"
        )
        
        # 规则4: 水平相邻的具体约束（silicon_bridge：必须紧邻）
        # 约束：相邻方向的边界距离 ≤ grid_size
        # 如果 i 在左（z1L[i,j] = 1）：x_j - (x_i + w_i) <= grid_size（距离不超过1个grid）
        model.addConstr(
            x_grid_var[j] - (x_grid_var[i] + w_var[i]) <= grid_size + M * (1 - z1L[(i, j)]),
            name=f"horizontal_left_dist_{i}_{j}"
        )
        model.addConstr(
            x_grid_var[j] - (x_grid_var[i] + w_var[i]) >= 0 - M * (1 - z1L[(i, j)]),
            name=f"horizontal_left_dist_lb_{i}_{j}"
        )
        # 如果 i 在右（z1R[i,j] = 1）：x_i - (x_j + w_j) <= grid_size（距离不超过1个grid）
        model.addConstr(
            x_grid_var[i] - (x_grid_var[j] + w_var[j]) <= grid_size + M * (1 - z1R[(i, j)]),
            name=f"horizontal_right_dist_{i}_{j}"
        )
        model.addConstr(
            x_grid_var[i] - (x_grid_var[j] + w_var[j]) >= 0 - M * (1 - z1R[(i, j)]),
            name=f"horizontal_right_dist_lb_{i}_{j}"
        )
        
        # 规则5: 垂直相邻的具体约束（silicon_bridge：必须紧邻）
        # 约束：相邻方向的边界距离 ≤ grid_size
        # 如果 i 在下（z2D[i,j] = 1）：y_j - (y_i + h_i) <= grid_size（距离不超过1个grid）
        model.addConstr(
            y_grid_var[j] - (y_grid_var[i] + h_var[i]) <= grid_size + M * (1 - z2D[(i, j)]),
            name=f"vertical_down_dist_sb_{i}_{j}"
        )
        model.addConstr(
            y_grid_var[j] - (y_grid_var[i] + h_var[i]) >= 0 - M * (1 - z2D[(i, j)]),
            name=f"vertical_down_dist_lb_sb_{i}_{j}"
        )
        # 如果 i 在上（z2U[i,j] = 1）：y_i - (y_j + h_j) <= grid_size（距离不超过1个grid）
        model.addConstr(
            y_grid_var[i] - (y_grid_var[j] + h_var[j]) <= grid_size + M * (1 - z2U[(i, j)]),
            name=f"vertical_up_dist_sb_{i}_{j}"
        )
        model.addConstr(
            y_grid_var[i] - (y_grid_var[j] + h_var[j]) >= 0 - M * (1 - z2U[(i, j)]),
            name=f"vertical_up_dist_lb_sb_{i}_{j}"
        )


        # 4.4 共享边长约束：如果紧邻，需要有一段长度为min_shared_length的共享边
        # 约束1：如果水平相邻，那么垂直方向的重叠长度 >= min_shared_length_grid
        shared_y = model.addVar(
            name=f"shared_y_{i}_{j}",
            lb=0,
            ub=min(chiplet_w_orig_grid[i], chiplet_h_orig_grid[i], chiplet_w_orig_grid[j], chiplet_h_orig_grid[j]),
            vtype=GRB.INTEGER
        )
        model.addConstr(
            shared_y <= (y_grid_var[i] + h_var[i]) - y_grid_var[j] + M * (1 - z1[(i, j)]),
            name=f"shared_y_ub1_{i}_{j}"
        )
        model.addConstr(
            shared_y <= (y_grid_var[j] + h_var[j]) - y_grid_var[i] + M * (1 - z1[(i, j)]),
            name=f"shared_y_ub2_{i}_{j}"
        )
        model.addConstr(
            shared_y >= min_shared_length_grid - M * (1 - z1[(i, j)]),
            name=f"shared_y_min_{i}_{j}"
        )

        # 约束2：水平方向（与相邻方向垂直）的重叠长度 >= min_shared_length_grid 
        shared_x = model.addVar(
            name=f"shared_x_{i}_{j}",
            lb=0,
            ub=min(chiplet_w_orig_grid[i], chiplet_h_orig_grid[i], chiplet_w_orig_grid[j], chiplet_h_orig_grid[j]),
            vtype=GRB.INTEGER
        )
        model.addConstr(
            shared_x <= (x_grid_var[i] + w_var[i]) - x_grid_var[j] + M * (1 - z2[(i, j)]),
            name=f"shared_x_ub1_{i}_{j}"
        )
        model.addConstr(
            shared_x <= (x_grid_var[j] + w_var[j]) - x_grid_var[i] + M * (1 - z2[(i, j)]),
            name=f"shared_x_ub2_{i}_{j}"
        )
        model.addConstr(
            shared_x >= min_shared_length_grid - M * (1 - z2[(i, j)]),
            name=f"shared_x_min_{i}_{j}"
        )
       
    # 4.5 非重叠约束
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
        # 放宽约束：至少满足一个非重叠条件
        model.addConstr(
            p_left[(i, j)] + p_right[(i, j)] + p_down[(i, j)] + p_up[(i, j)] >= 1,
            name=f"non_overlap_any_{i}_{j}"
        )
        
        # 情况1: i 在 j 的左边（x_i + w_i <= x_j）
        # 正向约束：p_left=1 → x_i + w_i <= x_j
        model.addConstr(
            x_grid_var[i] + w_var[i] - x_grid_var[j] <= M * (1 - p_left[(i, j)]),
            name=f"non_overlap_left_{i}_{j}"
        )
        # 修复漏洞2：正确的反向约束（x_i + w_i <= x_j → p_left=1）
        model.addConstr(
            x_grid_var[j] - (x_grid_var[i] + w_var[i]) <= M * p_left[(i, j)],
            name=f"non_overlap_left_rev_{i}_{j}"
        )
        
        # 情况2: i 在 j 的右边（x_j + w_j <= x_i）
        model.addConstr(
            x_grid_var[j] + w_var[j] - x_grid_var[i] <= M * (1 - p_right[(i, j)]),
            name=f"non_overlap_right_{i}_{j}"
        )
        model.addConstr(
            x_grid_var[i] - (x_grid_var[j] + w_var[j]) <= M * p_right[(i, j)],
            name=f"non_overlap_right_rev_{i}_{j}"
        )
        
        # 情况3: i 在 j 的下边（y_i + h_i <= y_j）
        model.addConstr(
            y_grid_var[i] + h_var[i] - y_grid_var[j] <= M * (1 - p_down[(i, j)]),
            name=f"non_overlap_down_{i}_{j}"
        )
        model.addConstr(
            y_grid_var[j] - (y_grid_var[i] + h_var[i]) <= M * p_down[(i, j)],
            name=f"non_overlap_down_rev_{i}_{j}"
        )
        
        # 情况4: i 在 j 的上边（y_j + h_j <= y_i）
        model.addConstr(
            y_grid_var[j] + h_var[j] - y_grid_var[i] <= M * (1 - p_up[(i, j)]),
            name=f"non_overlap_up_{i}_{j}"
        )
        model.addConstr(
            y_grid_var[i] - (y_grid_var[j] + h_var[j]) <= M * p_up[(i, j)],
            name=f"non_overlap_up_rev_{i}_{j}"
        )

    if verbose:
        print(f"非重叠约束: {len(all_pairs)} 对模块对(所有模块对)M={M}（基板尺寸最大值）")
    
    # 4.4.2 standard 连接：允许更宽松的约束（不强制紧邻，但鼓励靠近）
    # 对于 standard 连接，不强制紧邻
    # 可以通过线长目标函数来鼓励它们靠近
    # 不添加额外的相邻约束，让优化器通过最小化线长来自然靠近
    # if verbose and len(standard_pairs) > 0:
    #     print(f"standard 连接 ({len(standard_pairs)} 对): 不强制紧邻，通过线长优化鼓励靠近")

    # 4.6 外接方框约束
    bbox_min_x = model.addVar(name="bbox_min_x", lb=0, ub=W, vtype=GRB.INTEGER)
    bbox_max_x = model.addVar(name="bbox_max_x", lb=0, ub=W, vtype=GRB.INTEGER)
    bbox_min_y = model.addVar(name="bbox_min_y", lb=0, ub=H, vtype=GRB.INTEGER)
    bbox_max_y = model.addVar(name="bbox_max_y", lb=0, ub=H, vtype=GRB.INTEGER)
    bbox_w = model.addVar(name="bbox_w", lb=0, ub=W, vtype=GRB.INTEGER)
    bbox_h = model.addVar(name="bbox_h", lb=0, ub=H, vtype=GRB.INTEGER)
    
    for k in range(n):
        model.addConstr(bbox_min_x <= x_grid_var[k], name=f"bbox_min_x_{k}")
        model.addConstr(bbox_max_x >= x_grid_var[k] + w_var[k], name=f"bbox_max_x_{k}")
        model.addConstr(bbox_min_y <= y_grid_var[k], name=f"bbox_min_y_{k}")
        model.addConstr(bbox_max_y >= y_grid_var[k] + h_var[k], name=f"bbox_max_y_{k}")
    
    model.addConstr(bbox_w == bbox_max_x - bbox_min_x, name="bbox_w_def")
    model.addConstr(bbox_h == bbox_max_y - bbox_min_y, name="bbox_h_def")
    
    # 4.8 长宽比约束
    # if min_aspect_ratio is not None:
    #     # bbox_w / bbox_h >= min_aspect_ratio
    #     # 转换为线性约束: bbox_w >= min_aspect_ratio * bbox_h
    #     model.addConstr(
    #         bbox_w >= min_aspect_ratio * bbox_h,
    #         name="aspect_ratio_min"
    #     )
    #     if verbose:
    #         print(f"长宽比约束: bbox_w/bbox_h >= {min_aspect_ratio}")
    
    # if max_aspect_ratio is not None:
    #     # bbox_w / bbox_h <= max_aspect_ratio
    #     # 转换为线性约束: bbox_w <= max_aspect_ratio * bbox_h
    #     model.addConstr(
    #         bbox_w <= max_aspect_ratio * bbox_h,
    #         name="aspect_ratio_max"
    #     )
    #     if verbose:
    #         print(f"长宽比约束: bbox_w/bbox_h <= {max_aspect_ratio}")
    
    # 5.3 长宽比优化目标（最小化长宽比与理想值的偏差）
    # 理想长宽比设为1.0（正方形），使用 |bbox_w/bbox_h - 1| 的线性近似
    aspect_ratio_penalty = None
    # if minimize_bbox_area:  # 只在最小化面积时考虑长宽比优化
    #     # 使用辅助变量表示长宽比偏差
    #     # 由于 bbox_w/bbox_h 是非线性的，我们使用 |bbox_w - bbox_h| 作为近似
    #     # 这鼓励长宽接近，从而接近正方形
    #     aspect_ratio_diff = model.addVar(
    #         name="aspect_ratio_diff",
    #         lb=0,
    #         ub=max(W, H),
    #         vtype=GRB.CONTINUOUS
    #     )
    #     # |bbox_w - bbox_h| <= aspect_ratio_diff
    #     model.addConstr(
    #         aspect_ratio_diff >= bbox_w - bbox_h,
    #         name="aspect_ratio_diff_ge_w_minus_h"
    #     )
    #     model.addConstr(
    #         aspect_ratio_diff >= bbox_h - bbox_w,
    #         name="aspect_ratio_diff_ge_h_minus_w"
    #     )
    #     aspect_ratio_penalty = aspect_ratio_diff
    
    # ============ 步骤5: 定义目标函数 ============
    # 5.1 线长（曼哈顿距离）
    wirelength = 0
    
    # 计算所有连接的线长（包括 silicon_bridge 和 standard）
    for i, j in all_connected_pairs:
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
        M_dx = M  # Big-M常数
        M_dy = M  # Big-M常数
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

    
    return ILPModelContext(
        model=model,
        nodes=nodes,
        edges=edges,
        x_grid_var=x_grid_var,
        y_grid_var=y_grid_var,
        r=r,
        z1=z1,
        z2=z2,
        z1L=z1L,
        z1R=z1R,
        z2D=z2D,
        z2U=z2U,
        silicon_bridge_pairs=silicon_bridge_pairs,
        standard_pairs=standard_pairs,
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
        
        # 提取连接关系，支持连接类型（第四列：1=silicon_bridge, 0=standard）
        if "connections" in data and isinstance(data["connections"], list):
            for conn in data["connections"]:
                if isinstance(conn, list) and len(conn) >= 3:
                    src, dst = conn[0], conn[1]
                    # 读取连接类型：必须有第四列
                    if len(conn) < 4:
                        raise ValueError(f"连接格式错误：必须包含4列 [src, dst, weight, conn_type]。当前连接: {conn}")
                    conn_type = conn[3]
                    if conn_type not in [0, 1]:
                        raise ValueError(f"连接类型错误：conn_type 必须是 0 (standard) 或 1 (silicon_bridge)。当前值: {conn_type}，连接: {conn}")
                    edges.append((src, dst, conn_type))
                else:
                    raise ValueError(f"连接格式错误：每个连接必须是包含至少3个元素的列表 [src, dst, weight, conn_type]。当前连接: {conn}")
    
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
                layout=result.layout,  # 网格坐标
                save_path=str(save_path),
                grid_size=grid_size,
                rotations=result.rotations,
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

