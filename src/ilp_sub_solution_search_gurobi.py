from __future__ import annotations

from typing import Dict, Tuple, List, Optional
from pathlib import Path
from copy import deepcopy
import math

import gurobipy as gp
from gurobipy import GRB

from tool import build_random_chiplet_graph, draw_chiplet_diagram, print_constraint_formal, print_pair_distances_only, print_all_variables, get_var_value
from ilp_method_gurobi import (
    ILPModelContext,
    ILPPlacementResult,
    build_placement_ilp_model_grid,
    solve_placement_ilp_from_model,
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
    
    参数:
        model: Gurobi模型
        abs_var: 绝对值变量（>= 0）
        orig_var: 原始变量（可以是正数或负数）
        M: Big-M常数（必须 >= |orig_var|的最大可能值）
        constraint_prefix: 约束名称前缀（用于生成唯一的约束名称）
    
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
    # 当 orig_var < 0 时 (is_positive=0)，约束不起作用（M很大）
    model.addConstr(
        abs_var >= orig_var - M * (1 - is_positive),
        name=f"{constraint_prefix}_abs_ge_orig"
    )
    
    # 约束2: 当 orig_var >= 0 时 (is_positive=1)，约束简化为: abs_var <= orig_var
    # 当 orig_var < 0 时 (is_positive=0)，约束不起作用（M很大）
    model.addConstr(
        abs_var <= orig_var + M * (1 - is_positive),
        name=f"{constraint_prefix}_abs_le_orig"
    )
    
    # 约束3: 当 orig_var < 0 时 (is_positive=0)，约束简化为: abs_var >= -orig_var
    # 当 orig_var >= 0 时 (is_positive=1)，约束不起作用（M很大）
    model.addConstr(
        abs_var >= -orig_var - M * is_positive,
        name=f"{constraint_prefix}_abs_ge_neg_orig"
    )
    
    # 约束4: 当 orig_var < 0 时 (is_positive=0)，约束简化为: abs_var <= -orig_var
    # 当 orig_var >= 0 时 (is_positive=1)，约束不起作用（M很大）
    model.addConstr(
        abs_var <= -orig_var + M * is_positive,
        name=f"{constraint_prefix}_abs_le_neg_orig"
    )
    
    # 约束5: 强制 is_positive = 1 当 orig_var >= 0
    # 如果 orig_var >= 0，则必须 is_positive = 1（否则约束不满足）
    model.addConstr(
        orig_var >= -M * (1 - is_positive),
        name=f"{constraint_prefix}_force_positive"
    )
    
    # 约束6: 强制 is_positive = 0 当 orig_var < 0
    # 如果 orig_var < 0，则必须 is_positive = 0（否则约束不满足）
    # 使用一个很小的epsilon来避免边界情况
    epsilon = 0.001
    model.addConstr(
        orig_var <= M * is_positive - epsilon,
        name=f"{constraint_prefix}_force_negative"
    )


def _add_exclude_dist_constraint(
    ctx: ILPModelContext,
    *,
    solution_index_suffix: str,
    min_pair_dist_diff: float,
    prev_pair_distances_list: List[Dict[Tuple[int, int], float]],
    M: float,
) -> None:
    """
    添加距离排除约束：至少有一对chiplet的距离与之前所有解对应chiplet对之间的距离不同。
    
    注意：只有当当前解的chiplet对之间的距离与之前所有解的对应chiplet对之间的距离，
    全部都相差至少min_pair_dist_diff阈值时，才能说当前chiplet的距离与之前所有解对应chiplet对之间的距离不同。
    
    参数:
        ctx: ILP模型上下文
        solution_index_suffix: 用于生成唯一约束名称的后缀
        min_pair_dist_diff: chiplet对之间距离差异的最小阈值
        prev_pair_distances_list: 之前所有解的chiplet对距离列表，每个元素是一个字典，key是(i,j)元组，value是距离
        M: Big-M常数
    """
    model = ctx.model
    n = len(ctx.nodes)
    W = ctx.W
    H = ctx.H
    grid_size = ctx.grid_size
    
    if grid_size is None:
        print(f"[WARNING] grid_size为None，跳过排除约束")
        return
    
    if len(prev_pair_distances_list) == 0:
        print(f"[WARNING] prev_pair_distances_list为空，跳过排除约束")
        return
    
    print(f"[DEBUG] 开始添加排除约束，之前解数量: {len(prev_pair_distances_list)}, min_pair_dist_diff: {min_pair_dist_diff}")
    
    # 将min_pair_dist_diff从实际坐标单位转换为grid坐标单位
    # 因为约束中使用的是grid坐标距离，所以阈值也需要是grid坐标单位
    min_pair_dist_diff_grid = min_pair_dist_diff / grid_size
    print(f"[DEBUG] min_pair_dist_diff (实际坐标): {min_pair_dist_diff}, 转换为grid坐标: {min_pair_dist_diff_grid}")
    
    # 更新模型以确保变量名称可用
    try:
        model.update()
    except Exception as e:
        print(f"[WARNING] 模型更新失败: {e}")
        return
    
    # 计算grid的上界（需要先计算，因为后面会用到）
    grid_w = int(math.ceil(W / grid_size))
    grid_h = int(math.ceil(H / grid_size))
    
    # 获取x_grid和y_grid变量（左下角坐标）
    # 使用 getVars() 遍历所有变量，通过 VarName 查找，避免 getVarByName() 的错误
    x_grid = {}
    y_grid = {}
    all_vars_dict = {v.VarName: v for v in model.getVars()}
    
    for k in range(n):
        var_name_x = f"x_grid_{k}"
        var_name_y = f"y_grid_{k}"
        x_grid_var = all_vars_dict.get(var_name_x)
        y_grid_var = all_vars_dict.get(var_name_y)
        if x_grid_var is None or y_grid_var is None:
            print(f"[WARNING] 无法找到变量 {var_name_x} 或 {var_name_y}，跳过排除约束")
            # 列出所有变量名以便调试
            all_var_names = list(all_vars_dict.keys())[:20]
            print(f"[DEBUG] 前20个变量名: {all_var_names}")
            return  # 如果没有grid变量，说明不是网格化模型
        x_grid[k] = x_grid_var
        y_grid[k] = y_grid_var
    
    # 获取或计算中心坐标（grid坐标）
    # 中心坐标 = 左下角坐标 + (宽度/2, 高度/2)
    # 在grid坐标系统中，chiplet的宽度和高度是grid_size的整数倍
    cx_grid = {}  # 中心坐标x（grid单位）
    cy_grid = {}  # 中心坐标y（grid单位）
    
    for k in range(n):
        node = ctx.nodes[k]
        # 获取chiplet的实际尺寸
        if hasattr(node, 'dimensions') and isinstance(node.dimensions, dict):
            width = node.dimensions.get('x', 0.0)
            height = node.dimensions.get('y', 0.0)
        else:
            width = 0.0
            height = 0.0
        
        # 转换为grid坐标的尺寸（grid单位）
        w_grid = width / grid_size
        h_grid = height / grid_size
        
        # 计算中心坐标（grid单位）：中心 = 左下角 + (宽度/2, 高度/2)
        # 创建辅助变量表示中心坐标
        # 上界需要更大，因为中心坐标可能超出grid_w/grid_h（由于chiplet有宽度和高度）
        cx_grid_k = model.addVar(
            name=f"cx_grid_center_{solution_index_suffix}_{k}",
            lb=0,
            ub=grid_w + 10,  # 增加上界以容纳chiplet宽度
            vtype=GRB.CONTINUOUS
        )
        cy_grid_k = model.addVar(
            name=f"cy_grid_center_{solution_index_suffix}_{k}",
            lb=0,
            ub=grid_h + 10,  # 增加上界以容纳chiplet高度
            vtype=GRB.CONTINUOUS
        )
        
        # 约束：中心坐标 = 左下角坐标 + (宽度/2, 高度/2)
        model.addConstr(
            cx_grid_k == x_grid[k] + w_grid / 2.0,
            name=f"cx_grid_center_def_{solution_index_suffix}_{k}"
        )
        model.addConstr(
            cy_grid_k == y_grid[k] + h_grid / 2.0,
            name=f"cy_grid_center_def_{solution_index_suffix}_{k}"
        )
        
        cx_grid[k] = cx_grid_k
        cy_grid[k] = cy_grid_k
    
    print(f"[DEBUG] 成功获取所有grid变量和中心坐标，n={n}")
    
    # 生成所有chiplet对（i < j，避免重复）
    chiplet_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            chiplet_pairs.append((i, j))
    
    print(f"[DEBUG] 生成 {len(chiplet_pairs)} 个chiplet对")
    
    # 为每对chiplet计算当前解的距离（使用grid坐标）
    # 创建辅助变量表示距离的绝对值（所有chiplet对共享，避免重复创建）
    dx_abs_dict = {}
    dy_abs_dict = {}
    dist_curr_dict = {}
    
    # 为每对chiplet创建二进制变量，表示该对的距离是否与之前所有解都不同
    diff_dist_pair = {}
    # 为每对chiplet和每个之前解创建二进制变量，表示该对的距离是否与解s的距离相同
    same_dist_pair_prev = {}  # key: ((i,j), prev_idx)
    
    # max_dist_diff使用grid单位（最大可能的grid距离）
    max_dist_diff = grid_w + grid_h
    epsilon = max(0.001, min_pair_dist_diff_grid * 0.01)
    
    for i, j in chiplet_pairs:
        # 创建辅助变量表示grid坐标中心点差的绝对值
        # 上界需要更大，因为中心坐标可能超出grid_w/grid_h（由于chiplet有宽度和高度）
        dx_grid_abs_ij = model.addVar(
            name=f"dx_grid_abs_pair_{solution_index_suffix}_{i}_{j}",
            lb=0,
            ub=grid_w + 10,  # 增加上界以容纳chiplet宽度
            vtype=GRB.CONTINUOUS
        )
        dy_grid_abs_ij = model.addVar(
            name=f"dy_grid_abs_pair_{solution_index_suffix}_{i}_{j}",
            lb=0,
            ub=grid_h + 10,  # 增加上界以容纳chiplet高度
            vtype=GRB.CONTINUOUS
        )
        
        # 计算grid坐标中心点的差
        # dx_grid_ij = cx_grid[i] - cx_grid[j]  (中心点x坐标差)
        # dy_grid_ij = cy_grid[i] - cy_grid[j]  (中心点y坐标差)
        
        # 使用Big-M方法添加绝对值约束：dx_grid_abs_ij = |cx_grid[i] - cx_grid[j]|
        # 上界需要更大以容纳chiplet宽度
        max_diff_x = grid_w + 10
        max_diff_y = grid_h + 10
        dx_grid_diff = model.addVar(
            name=f"dx_grid_diff_{solution_index_suffix}_{i}_{j}",
            lb=-max_diff_x,
            ub=max_diff_x,
            vtype=GRB.CONTINUOUS
        )
        model.addConstr(
            dx_grid_diff == cx_grid[i] - cx_grid[j],
            name=f"dx_grid_diff_def_{solution_index_suffix}_{i}_{j}"
        )
        M_dx = max_diff_x  # Big-M常数
        add_absolute_value_constraint_big_m(
            model=model,
            abs_var=dx_grid_abs_ij,
            orig_var=dx_grid_diff,
            M=M_dx,
            constraint_prefix=f"dx_grid_abs_pair_{solution_index_suffix}_{i}_{j}"
        )
        
        # 使用Big-M方法添加绝对值约束：dy_grid_abs_ij = |cy_grid[i] - cy_grid[j]|
        dy_grid_diff = model.addVar(
            name=f"dy_grid_diff_{solution_index_suffix}_{i}_{j}",
            lb=-max_diff_y,
            ub=max_diff_y,
            vtype=GRB.CONTINUOUS
        )
        model.addConstr(
            dy_grid_diff == cy_grid[i] - cy_grid[j],
            name=f"dy_grid_diff_def_{solution_index_suffix}_{i}_{j}"
        )
        M_dy = max_diff_y  # Big-M常数
        add_absolute_value_constraint_big_m(
            model=model,
            abs_var=dy_grid_abs_ij,
            orig_var=dy_grid_diff,
            M=M_dy,
            constraint_prefix=f"dy_grid_abs_pair_{solution_index_suffix}_{i}_{j}"
        )
        
        # 创建ILP变量表示当前距离（grid单位）
        max_dist = grid_w + grid_h
        dist_curr_ij = model.addVar(
            name=f"dist_curr_pair_{solution_index_suffix}_{i}_{j}",
            lb=0,
            ub=max_dist,
            vtype=GRB.CONTINUOUS
        )
        
        # 约束：当前距离 = dx_grid_abs_ij + dy_grid_abs_ij
        constraint_name = f"dist_curr_pair_def_{solution_index_suffix}_{i}_{j}"
        constr = model.addConstr(
            dist_curr_ij == dx_grid_abs_ij + dy_grid_abs_ij,
            name=constraint_name
        )
        print_constraint_formal(constr)
        
        dx_abs_dict[(i, j)] = dx_grid_abs_ij
        dy_abs_dict[(i, j)] = dy_grid_abs_ij
        dist_curr_dict[(i, j)] = dist_curr_ij
        
        # 检查该对是否在所有之前解中都存在
        pair_exists_in_all = True
        for prev_idx, prev_pair_distances in enumerate(prev_pair_distances_list):
            if (i, j) not in prev_pair_distances:
                pair_exists_in_all = False
                break
        
        if not pair_exists_in_all:
            # 跳过不在所有之前解中都存在的对
            continue
        
        # 创建二进制变量：该对的距离是否与之前所有解都不同
        diff_dist_pair[(i, j)] = model.addVar(
            name=f"diff_dist_pair_{solution_index_suffix}_{i}_{j}",
            vtype=GRB.BINARY
        )
        
        # 对于每个之前解，创建二进制变量表示该对的距离是否与该解的距离相同
        for prev_idx, prev_pair_distances in enumerate(prev_pair_distances_list):
            dist_prev_ij = prev_pair_distances[(i, j)]
            # 创建二进制变量：当前距离是否与解prev_idx的距离相同
            same_dist_pair_prev[((i, j), prev_idx)] = model.addVar(
                name=f"same_dist_pair_{solution_index_suffix}_{i}_{j}_prev{prev_idx}",
                vtype=GRB.BINARY
            )
            
            # 创建辅助变量表示距离差
            dist_diff_ij = model.addVar(
                name=f"dist_diff_pair_{solution_index_suffix}_{i}_{j}_prev{prev_idx}",
                lb=-max_dist_diff,
                ub=max_dist_diff,
                vtype=GRB.CONTINUOUS
            )
            
            # 距离差 = 当前距离 - 之前距离
            constraint_name = f"dist_diff_pair_def_{solution_index_suffix}_{i}_{j}_prev{prev_idx}"
            constr = model.addConstr(
                dist_diff_ij == dist_curr_ij - dist_prev_ij,
                name=constraint_name
            )
            print_constraint_formal(constr)
            
            # 使用Big-M方法添加绝对值约束：dist_diff_abs_ij = |dist_diff_ij|
            dist_diff_abs_ij = model.addVar(
                name=f"dist_diff_abs_pair_{solution_index_suffix}_{i}_{j}_prev{prev_idx}",
                lb=0,
                ub=max_dist_diff,
                vtype=GRB.CONTINUOUS
            )
            M_dist = max_dist_diff  # Big-M常数
            add_absolute_value_constraint_big_m(
                model=model,
                abs_var=dist_diff_abs_ij,
                orig_var=dist_diff_ij,
                M=M_dist,
                constraint_prefix=f"dist_diff_abs_pair_{solution_index_suffix}_{i}_{j}_prev{prev_idx}"
            )
            
            # 约束逻辑：same_dist_pair_prev[((i,j), prev_idx)] = 1 当且仅当 dist_diff_abs_ij < min_pair_dist_diff
            # 即：same = 1 表示当前解的距离与之前解prev_idx的距离相差 < min_pair_dist_diff（距离相同或相近）
            #     same = 0 表示当前解的距离与之前解prev_idx的距离相差 >= min_pair_dist_diff（距离不同）
            # 如果 same_dist_pair_prev = 1，则 dist_diff_abs_ij < min_pair_dist_diff
            constraint_name = f"same_dist_pair_upper_{solution_index_suffix}_{i}_{j}_prev{prev_idx}"
            constr = model.addConstr(
                dist_diff_abs_ij <= min_pair_dist_diff - epsilon + M_dist * (1 - same_dist_pair_prev[((i, j), prev_idx)]),
                name=constraint_name
            )
            print_constraint_formal(constr)
            
            # 如果 same_dist_pair_prev = 0，则 dist_diff_abs_ij >= min_pair_dist_diff
            constraint_name = f"same_dist_pair_lower_{solution_index_suffix}_{i}_{j}_prev{prev_idx}"
            constr = model.addConstr(
                dist_diff_abs_ij >= min_pair_dist_diff - M_dist * same_dist_pair_prev[((i, j), prev_idx)],
                name=constraint_name
            )
            print_constraint_formal(constr)
        
        # 约束：diff_dist_pair[(i,j)] = 1 当且仅当对于所有之前解，same_dist_pair_prev = 0
        # 即：diff_dist_pair[(i,j)] = 1 表示当前解的距离与所有之前解的距离都相差 >= min_pair_dist_diff
        # 注意：只有当当前解的chiplet对之间的距离与之前所有解的对应chiplet对之间的距离，
        #      全部都相差至少min_pair_dist_diff阈值时，diff_dist_pair = 1
        for prev_idx in range(len(prev_pair_distances_list)):
            # 如果 diff_dist_pair = 1，则 same_dist_pair_prev = 0
            constraint_name = f"diff_dist_pair_implies_not_same_{solution_index_suffix}_{i}_{j}_prev{prev_idx}"
            constr = model.addConstr(
                same_dist_pair_prev[((i, j), prev_idx)] <= 1 - diff_dist_pair[(i, j)],
                name=constraint_name
            )
            print_constraint_formal(constr)
            
            # 如果至少有一个 same_dist_pair_prev = 1，则 diff_dist_pair = 0
            constraint_name = f"not_same_implies_diff_dist_pair_{solution_index_suffix}_{i}_{j}_prev{prev_idx}"
            constr = model.addConstr(
                diff_dist_pair[(i, j)] <= 1 - same_dist_pair_prev[((i, j), prev_idx)],
                name=constraint_name
            )
            print_constraint_formal(constr)
        
        # 如果所有 same_dist_pair_prev = 0，则 diff_dist_pair = 1
        constraint_name = f"all_not_same_implies_diff_dist_pair_{solution_index_suffix}_{i}_{j}"
        constr = model.addConstr(
            diff_dist_pair[(i, j)] >= 1 - gp.quicksum([same_dist_pair_prev[((i, j), prev_idx)] for prev_idx in range(len(prev_pair_distances_list))]),
            name=constraint_name
        )
        print_constraint_formal(constr)
    
    # 顶层约束：至少有一对chiplet的距离与之前所有解都不同
    # 即：至少存在一对chiplet，使得当前解的距离与所有之前解的距离都相差 >= min_pair_dist_diff
    if len(diff_dist_pair) > 0:
        constraint_name = f"exclude_solution_dist_pair_{solution_index_suffix}"
        exclude_constr = model.addConstr(
            gp.quicksum([diff_dist_pair[pair] for pair in diff_dist_pair.keys()]) >= 1,
            name=constraint_name
        )
        # 调试信息：确认约束已添加
        print(f"[DEBUG] ✓ 添加排除约束: {constraint_name}, 涉及 {len(diff_dist_pair)} 对chiplet: {list(diff_dist_pair.keys())[:5]}...")
    else:
        print(f"[WARNING] ✗ diff_dist_pair为空，未添加排除约束！chiplet对总数: {len(chiplet_pairs)}, 之前解数量: {len(prev_pair_distances_list)}")


def add_exclude_constraint(
    ctx: ILPModelContext,
    *,
    require_change_pairs: int = 1,  # 目前没有用到这个参数，先保留接口
    solution_index: int = 0,  # 解的索引，用于生成唯一的约束名称
    min_diff: Optional[float] = None,  # 判断"不同"的最小差异阈值，如果为None则使用grid_size（已废弃，使用min_pos_diff）
    min_pos_diff: Optional[float] = None,  # 位置排除约束的最小变化量，如果为None则使用min_diff或grid_size
    min_pair_dist_diff: Optional[float] = None,  # chiplet对之间距离差异的最小阈值，如果为None则使用min_pos_diff
    prev_positions: Optional[List[Dict[int, Tuple[float, float]]]] = None,  # 之前所有解的位置列表
    prev_pair_distances_list: Optional[List[Dict[Tuple[int, int], float]]] = None,  # 之前所有解的chiplet对距离列表
    constraint_counter: Optional[List[int]] = None,  # 全局约束计数器，确保每个约束名称唯一
) -> None:
    """
    在已有 ILP 模型上添加排除解的约束。
    
    排除整个解，只要下一个解的chiplet集合中有chiplet的位置和上一个解不同即可。
    注意：固定的chiplet位置不会被约束，只考虑非固定的chiplet。
    
    参数:
        min_diff: 判断位置"不同"的最小差异阈值。如果为None，则使用grid_size（如果存在）或默认值0.01。
    """

    model = ctx.model
    x = ctx.x
    y = ctx.y
    n = len(ctx.nodes)
    W = ctx.W
    H = ctx.H
    fixed_chiplet_idx = ctx.fixed_chiplet_idx  # 获取固定的chiplet索引
    
    
    # 如果没有提供之前解的位置和距离列表，则尝试读取当前解作为上一解
    # 注意：这个分支通常不应该被执行，因为 add_exclude_constraint 是在求解之前调用的
    # 只有在求解之后需要添加排除约束时才会使用这个逻辑
    # 但是，由于我们在求解之前调用此函数，所以如果 prev_positions 为空，应该跳过位置排除约束
    # 只使用 prev_pair_distances_list 来添加排除约束
    if prev_positions is None or len(prev_positions) == 0:
        # 在求解之前调用时，无法读取变量值，跳过位置排除约束
        # 只使用 prev_pair_distances_list 来添加排除约束
        pass  # 不执行位置排除约束，继续执行距离排除约束
    
    # 获取位置排除约束的最小变化量
    # 优先级：min_pos_diff > min_diff > grid_size > 默认值0.01
    if min_pos_diff is None:
        if min_diff is not None:
            min_pos_diff = min_diff
        else:
            grid_size = ctx.grid_size
            if grid_size is not None:
                min_pos_diff = grid_size
            else:
                min_pos_diff = 0.01
    
    # 获取距离排除约束的最小变化量
    # 如果 min_pair_dist_diff 为 None，则使用 min_pos_diff
    if min_pair_dist_diff is None:
        min_pair_dist_diff = min_pos_diff
    
    M = max(W, H) * 2  # Big-M常数
    
    # 调用距离排除约束函数（一次性处理所有之前解）
    print(f"[DEBUG add_exclude_constraint] 检查条件: prev_pair_distances_list={prev_pair_distances_list is not None}, len={len(prev_pair_distances_list) if prev_pair_distances_list is not None else 0}, grid_size={ctx.grid_size}")
    if prev_pair_distances_list is not None and len(prev_pair_distances_list) > 0:
        print(f"[DEBUG add_exclude_constraint] 准备添加排除约束，之前解数量: {len(prev_pair_distances_list)}")
        # 确保每个约束都有唯一的名称，避免冲突
        # 使用全局约束计数器确保唯一性
        if constraint_counter is not None:
            current_counter = constraint_counter[0]
            solution_index_suffix = f"c{current_counter}"
            constraint_counter[0] += 1
        else:
            solution_index_suffix = f"{solution_index}"
        
        print(f"[DEBUG add_exclude_constraint] 调用 _add_exclude_dist_constraint，solution_index_suffix={solution_index_suffix}")
        _add_exclude_dist_constraint(
            ctx=ctx,
            solution_index_suffix=solution_index_suffix,
            min_pair_dist_diff=min_pair_dist_diff,
            prev_pair_distances_list=prev_pair_distances_list,
            M=M,
        )
        print(f"[DEBUG add_exclude_constraint] _add_exclude_dist_constraint 返回")
    else:
        print(f"[DEBUG add_exclude_constraint] prev_pair_distances_list为空或None，跳过排除约束")


# print_pair_distances_only 和 print_all_variables 函数已移动到 tool.py
# 从 tool.py 导入这些函数

def search_multiple_solutions(
    num_solutions: int = 3,
    min_shared_length: float = 0.5,
    input_json_path: Optional[str] = None,
    nodes: Optional[List] = None,
    edges: Optional[List[Tuple[int, int]]] = None,
    grid_size: Optional[float] = None,
    fixed_chiplet_idx: Optional[int] = None,
    min_pair_dist_diff: Optional[float] = None,  # chiplet对之间距离差异的最小阈值，如果为None则使用grid_size或默认值1.0；此参数控制距离排除约束：至少有一对chiplet的距离差必须 >= min_pair_dist_diff
    output_dir: Optional[str] = None,  # 输出目录，用于保存.lp文件和图片；如果为None，则使用默认路径
) -> List[ILPPlacementResult]:
    """
    搜索多个不同的解。
    
    参数:
        num_solutions: 需要搜索的解的数量
        min_shared_length: 相邻chiplet之间的最小共享边长
        input_json_path: 可选，从JSON文件加载输入
        nodes: 可选，chiplet节点列表（如果提供input_json_path则忽略此参数）
        edges: 可选，连接关系列表（如果提供input_json_path则忽略此参数）
        grid_size: 网格大小，如果提供则使用网格化布局
        fixed_chiplet_idx: 固定位置的chiplet索引
        min_pair_dist_diff: chiplet对之间距离差异的最小阈值，如果为None则使用grid_size或默认值1.0；此参数控制距离排除约束：至少有一对chiplet的距离差必须 >= min_pair_dist_diff
        output_dir: 输出目录，用于保存.lp文件和图片；如果为None，则使用默认路径（相对于项目根目录的output目录）
    """
    # 如果提供了input_json_path，则从JSON文件加载
    if input_json_path is not None:
        import json
        from tool import ChipletNode
        
        with open(input_json_path, 'r') as f:
            data = json.load(f)
        
        # 处理两种JSON格式：
        # 格式1: {"chiplets": [...], "connections": [...]} (ICCAD23格式)
        # 格式2: {"chiplet_name": {"dimensions": ..., "phys": ..., "power": ...}} (旧格式)
        nodes = []
        edges = []
        
        if "chiplets" in data and isinstance(data["chiplets"], list):
            # 格式1: ICCAD23格式
            for chiplet_info in data["chiplets"]:
                name = chiplet_info.get("name", "")
                width = chiplet_info.get("width", 0.0)
                height = chiplet_info.get("height", 0.0)
                
                nodes.append(
                    ChipletNode(
                        name=name,
                        dimensions={"x": width, "y": height},
                        phys=[],
                        power=chiplet_info.get("power", 0.0),
                    )
                )
            
            # 提取连接关系
            if "connections" in data and isinstance(data["connections"], list):
                for conn in data["connections"]:
                    if isinstance(conn, list) and len(conn) >= 2:
                        src, dst = conn[0], conn[1]
                        # 确保边是唯一的（避免重复）
                        edge = (src, dst) if src < dst else (dst, src)
                        if edge not in edges:
                            edges.append(edge)
        else:
            # 格式2: 旧格式（字典格式）
            from input_process import build_chiplet_table
            table = build_chiplet_table(data)
            for row in table:
                nodes.append(
                    ChipletNode(
                        name=row["name"],
                        dimensions=row["dimensions"],
                        phys=row["phys"],
                        power=row["power"],
                    )
                )
            
            # 从JSON数据中提取边（如果存在）
            for chiplet_name, chiplet_data in data.items():
                if isinstance(chiplet_data, dict) and "connections" in chiplet_data:
                    for conn in chiplet_data["connections"]:
                        if isinstance(conn, dict) and "target" in conn:
                            target = conn["target"]
                            # 确保边是唯一的（避免重复）
                            edge = (chiplet_name, target) if chiplet_name < target else (target, chiplet_name)
                            if edge not in edges:
                                edges.append(edge)
        
        # 如果没有找到边，使用默认的随机生成方法
        if len(edges) == 0:
            from tool import generate_random_links
            names = [n.name for n in nodes]
            edges = generate_random_links(names, edge_prob=0.2, fixed_num_edges=4)
    elif nodes is None or edges is None:
        raise ValueError("必须提供 input_json_path 或 nodes 和 edges 参数")
    
    solutions = []
    all_prev_pair_distances = []
    
    # 全局约束计数器，确保每个约束名称唯一
    constraint_counter = [0]
    
    # 确定 min_pair_dist_diff 的值：如果为None，则使用grid_size或默认值1.0
    if min_pair_dist_diff is None:
        if grid_size is not None:
            min_pair_dist_diff = grid_size
        else:
            min_pair_dist_diff = 1.0
    
    for i in range(num_solutions):
        # 构建ILP模型（只使用网格化版本）
        ctx = build_placement_ilp_model_grid(
            nodes=nodes,
            edges=edges,
            grid_size=grid_size if grid_size is not None else 1.0,  # 如果未指定grid_size，使用默认值1.0
            fixed_chiplet_idx=fixed_chiplet_idx,
            min_shared_length=min_shared_length,
        )
        
        # 添加排除约束（排除之前找到的所有解）
        print(f"[DEBUG search_multiple_solutions] 解 {i+1}: all_prev_pair_distances长度={len(all_prev_pair_distances)}")
        if len(all_prev_pair_distances) > 0:
            print(f"[DEBUG search_multiple_solutions] 调用 add_exclude_constraint，之前解数量: {len(all_prev_pair_distances)}")
            add_exclude_constraint(
                ctx=ctx,
                solution_index=i,
                min_pair_dist_diff=min_pair_dist_diff,
                prev_pair_distances_list=all_prev_pair_distances,
                constraint_counter=constraint_counter,
            )
            print(f"[DEBUG search_multiple_solutions] add_exclude_constraint 调用完成")
        else:
            print(f"[DEBUG search_multiple_solutions] all_prev_pair_distances为空，跳过排除约束")
        
        # 导出LP文件（在求解之前，包含所有约束）
        # 确定输出目录
        if output_dir is None:
            # 默认输出目录：相对于项目根目录的output目录
            default_output = Path(__file__).parent.parent / "output"
            output_dir_path = default_output
        else:
            output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        lp_file = output_dir_path / f"constraints_solution_{i+1}_gurobi.lp"
        ctx.model.write(str(lp_file))
        # 简化输出：不再打印LP文件保存信息
        # print(f"\nLP文件已保存: {lp_file}")
        
        # 求解
        result = solve_placement_ilp_from_model(ctx)
        
        # result.status 是字符串类型（如 "Optimal"），需要与字符串比较
        if result.status != "Optimal":
            print(f"\n求解状态: {result.status}，停止搜索")
            break

        solutions.append(result)
        
        # 简化输出：只打印相对距离和比较信息
        print_pair_distances_only(
            ctx, result, 
            solution_idx=i,
            prev_pair_distances_list=all_prev_pair_distances,
            min_pair_dist_diff=min_pair_dist_diff
        )
        
        # 计算并保存当前解的chiplet对之间的距离（使用grid坐标中心点的曼哈顿距离）
        # 注意：使用中心点坐标计算距离，而不是左下角坐标
        pair_distances = {}
        grid_size_val = ctx.grid_size if ctx.grid_size is not None else 1.0
        
        # 获取当前解的实际坐标（左下角坐标，不受旋转影响）
        x_coords = {}
        y_coords = {}
        for k, node in enumerate(nodes):
            node_name = node.name if hasattr(node, 'name') else f"Chiplet_{k}"
            if node_name in result.layout:
                x_coords[k], y_coords[k] = result.layout[node_name]
            else:
                x_coords[k] = 0.0
                y_coords[k] = 0.0
        
        # 获取grid坐标值（左下角）
        x_grid_coords = {}
        y_grid_coords = {}
        for k in range(len(nodes)):
            x_grid_var = ctx.model.getVarByName(f"x_grid_{k}")
            y_grid_var = ctx.model.getVarByName(f"y_grid_{k}")
            if x_grid_var is not None and y_grid_var is not None:
                x_grid_val = x_grid_var.X
                y_grid_val = y_grid_var.X
                if x_grid_val is not None and y_grid_val is not None:
                    x_grid_coords[k] = float(x_grid_val)
                    y_grid_coords[k] = float(y_grid_val)
                else:
                    # 如果无法获取grid坐标，使用实际坐标转换为grid坐标
                    x_grid_coords[k] = x_coords[k] / grid_size_val
                    y_grid_coords[k] = y_coords[k] / grid_size_val
            else:
                # 如果没有grid变量，使用实际坐标转换为grid坐标
                x_grid_coords[k] = x_coords[k] / grid_size_val
                y_grid_coords[k] = y_coords[k] / grid_size_val
        
        # 计算每对chiplet之间的曼哈顿距离（使用grid坐标中心点）
        for i_idx in range(len(nodes)):
            for j_idx in range(i_idx + 1, len(nodes)):
                if i_idx in x_grid_coords and j_idx in x_grid_coords and i_idx in y_grid_coords and j_idx in y_grid_coords:
                    # 获取chiplet的尺寸
                    node_i = nodes[i_idx]
                    node_j = nodes[j_idx]
                    width_i = node_i.dimensions.get('x', 0.0) if hasattr(node_i, 'dimensions') and isinstance(node_i.dimensions, dict) else 0.0
                    height_i = node_i.dimensions.get('y', 0.0) if hasattr(node_i, 'dimensions') and isinstance(node_i.dimensions, dict) else 0.0
                    width_j = node_j.dimensions.get('x', 0.0) if hasattr(node_j, 'dimensions') and isinstance(node_j.dimensions, dict) else 0.0
                    height_j = node_j.dimensions.get('y', 0.0) if hasattr(node_j, 'dimensions') and isinstance(node_j.dimensions, dict) else 0.0
                    
                    # 转换为grid坐标的尺寸
                    w_grid_i = width_i / grid_size_val
                    h_grid_i = height_i / grid_size_val
                    w_grid_j = width_j / grid_size_val
                    h_grid_j = height_j / grid_size_val
                    
                    # 计算中心点坐标（grid单位）
                    cx_grid_i = x_grid_coords[i_idx] + w_grid_i / 2.0
                    cy_grid_i = y_grid_coords[i_idx] + h_grid_i / 2.0
                    cx_grid_j = x_grid_coords[j_idx] + w_grid_j / 2.0
                    cy_grid_j = y_grid_coords[j_idx] + h_grid_j / 2.0
                    
                    # 计算中心点之间的grid坐标距离（绝对值差）
                    dx_grid = abs(cx_grid_i - cx_grid_j)
                    dy_grid = abs(cy_grid_i - cy_grid_j)
                    # grid坐标的曼哈顿距离
                    grid_manhattan_dist = dx_grid + dy_grid
                    pair_distances[(i_idx, j_idx)] = grid_manhattan_dist
        all_prev_pair_distances.append(pair_distances)
        
        # 输出当前解的基本信息
        print(f"\n=== 解 {i+1} ===")
        print(f"目标函数值: {result.objective_value:.4f}")
        
        # 绘制并保存布局图片
        layout_dict = {}
        fixed_chiplet_names = set()
        for k, node in enumerate(nodes):
            node_name = node.name if hasattr(node, 'name') else f"Chiplet_{k}"
            # 直接从result.layout获取坐标（左下角坐标，不受旋转影响）
            if node_name in result.layout:
                layout_dict[node_name] = result.layout[node_name]
            else:
                layout_dict[node_name] = (0.0, 0.0)
            if fixed_chiplet_idx is not None and k == fixed_chiplet_idx:
                fixed_chiplet_names.add(node_name)
        
        # 保存图片到输出目录
        # 使用相同的输出目录（如果output_dir为None，则使用默认路径）
        if output_dir is None:
            default_output = Path(__file__).parent.parent / "output"
            output_dir_path = default_output
        else:
            output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        image_path = output_dir_path / f"solution_{i+1}_layout_gurobi.png"
        
        try:
            draw_chiplet_diagram(
                nodes=nodes,
                edges=edges,
                save_path=str(image_path),
                layout=layout_dict,
                fixed_chiplet_names=fixed_chiplet_names if fixed_chiplet_names else None,
            )
            # 简化输出：不再打印布局图片保存信息
            # print(f"\n布局图片已保存: {image_path}")
        except Exception as e:
            # 简化输出：不再打印错误信息
            # print(f"\n警告: 保存布局图片时出错: {e}")
            pass
    
    return solutions


if __name__ == "__main__":
    import json
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python ilp_sub_solution_search_gurobi.py <input_json_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    nodes, edges = build_random_chiplet_graph(data)
    
    solutions = search_multiple_solutions(
        nodes=nodes,
        edges=edges,
        num_solutions=5,
        grid_size=0.5,
        fixed_chiplet_idx=0,
        min_pair_dist_diff=0.5,
    )
    
    print(f"\n总共找到 {len(solutions)} 个解")

