from __future__ import annotations

from typing import Dict, Tuple, List, Optional
from pathlib import Path
from copy import deepcopy
import math

import pulp

from tool import build_random_chiplet_graph, draw_chiplet_diagram, print_constraint_formal
from ilp_method import (
    ILPModelContext,
    ILPPlacementResult,
    build_placement_ilp_model_grid,
    solve_placement_ilp_from_model,
)


def add_absolute_value_constraint_big_m(
    prob: pulp.LpProblem,
    abs_var: pulp.LpVariable,
    orig_var: pulp.LpVariable,
    M: float,
    constraint_prefix: str,
) -> None:
    """
    使用Big-M方法添加绝对值约束：abs_var = |orig_var|
    
    参数:
        prob: ILP问题
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
    is_positive = pulp.LpVariable(f"{constraint_prefix}_is_positive", cat='Binary')
    
    # 约束1: 当 orig_var >= 0 时 (is_positive=1)，约束简化为: abs_var >= orig_var
    # 当 orig_var < 0 时 (is_positive=0)，约束不起作用（M很大）
    prob += abs_var >= orig_var - M * (1 - is_positive), f"{constraint_prefix}_abs_ge_orig"
    
    # 约束2: 当 orig_var >= 0 时 (is_positive=1)，约束简化为: abs_var <= orig_var
    # 当 orig_var < 0 时 (is_positive=0)，约束不起作用（M很大）
    prob += abs_var <= orig_var + M * (1 - is_positive), f"{constraint_prefix}_abs_le_orig"
    
    # 约束3: 当 orig_var < 0 时 (is_positive=0)，约束简化为: abs_var >= -orig_var
    # 当 orig_var >= 0 时 (is_positive=1)，约束不起作用（M很大）
    prob += abs_var >= -orig_var - M * is_positive, f"{constraint_prefix}_abs_ge_neg_orig"
    
    # 约束4: 当 orig_var < 0 时 (is_positive=0)，约束简化为: abs_var <= -orig_var
    # 当 orig_var >= 0 时 (is_positive=1)，约束不起作用（M很大）
    prob += abs_var <= -orig_var + M * is_positive, f"{constraint_prefix}_abs_le_neg_orig"
    
    # 约束5: 强制 is_positive = 1 当 orig_var >= 0
    # 如果 orig_var >= 0，则必须 is_positive = 1（否则约束不满足）
    prob += orig_var >= -M * (1 - is_positive), f"{constraint_prefix}_force_positive"
    
    # 约束6: 强制 is_positive = 0 当 orig_var < 0
    # 如果 orig_var < 0，则必须 is_positive = 0（否则约束不满足）
    # 使用一个很小的epsilon来避免边界情况
    epsilon = 0.001
    prob += orig_var <= M * is_positive - epsilon, f"{constraint_prefix}_force_negative"


# def _add_exclude_pos_constraint(
#     ctx: ILPModelContext,
#     x_prev: Dict[int, float],
#     y_prev: Dict[int, float],
#     *,
#     solution_index_suffix: str,
#     min_pos_diff: float,
#     M: float,
# ) -> None:
#     """
#     添加位置排除约束：至少有一个非固定chiplet的位置与之前解不同。
    
#     参数:
#         ctx: ILP模型上下文
#         x_prev: 之前解中每个chiplet的x坐标
#         y_prev: 之前解中每个chiplet的y坐标
#         solution_index_suffix: 用于生成唯一约束名称的后缀
#         min_pos_diff: 位置变化的最小阈值
#         M: Big-M常数
#     """
#     prob = ctx.prob
#     x = ctx.x
#     y = ctx.y
#     n = len(ctx.nodes)
#     fixed_chiplet_idx = ctx.fixed_chiplet_idx
    
#     # 为每个chiplet创建二进制变量，表示该chiplet的位置是否与之前解不同
#     diff_pos = {}
#     for k in range(n):
#         diff_pos[k] = pulp.LpVariable(f"diff_pos_{solution_index_suffix}_{k}", cat='Binary')
    
#     # 对于每个chiplet k，判断其位置是否与之前解不同
#     for k in range(n):
#         if fixed_chiplet_idx is not None and k == fixed_chiplet_idx:
#             # 固定chiplet的位置不能改变
#             constraint_name = f"diff_pos_fixed_{solution_index_suffix}_{k}"
#             prob += diff_pos[k] == 0, constraint_name
#             print_constraint_formal(prob.constraints[constraint_name])
#             continue
        
#         # 创建辅助变量判断x坐标是否不同
#         x_diff = pulp.LpVariable(f"x_diff_{solution_index_suffix}_{k}", cat='Binary')
#         x_diff_plus = pulp.LpVariable(f"x_diff_plus_{solution_index_suffix}_{k}", cat='Binary')
#         x_diff_minus = pulp.LpVariable(f"x_diff_minus_{solution_index_suffix}_{k}", cat='Binary')
        
#         constraint_name = f"x_diff_plus_upper_{solution_index_suffix}_{k}"
#         prob += x[k] - x_prev[k] <= min_pos_diff - 0.001 + M * x_diff_plus, constraint_name
#         print_constraint_formal(prob.constraints[constraint_name])
        
#         constraint_name = f"x_diff_plus_lower_{solution_index_suffix}_{k}"
#         prob += x[k] - x_prev[k] >= min_pos_diff - M * (1 - x_diff_plus), constraint_name
#         print_constraint_formal(prob.constraints[constraint_name])
        
#         constraint_name = f"x_diff_minus_upper_{solution_index_suffix}_{k}"
#         prob += x[k] - x_prev[k] >= -min_pos_diff + 0.001 - M * x_diff_minus, constraint_name
#         print_constraint_formal(prob.constraints[constraint_name])
        
#         constraint_name = f"x_diff_minus_lower_{solution_index_suffix}_{k}"
#         prob += x[k] - x_prev[k] <= -min_pos_diff + M * (1 - x_diff_minus), constraint_name
#         print_constraint_formal(prob.constraints[constraint_name])
        
#         constraint_name = f"x_diff_from_plus_{solution_index_suffix}_{k}"
#         prob += x_diff >= x_diff_plus, constraint_name
#         print_constraint_formal(prob.constraints[constraint_name])
        
#         constraint_name = f"x_diff_from_minus_{solution_index_suffix}_{k}"
#         prob += x_diff >= x_diff_minus, constraint_name
#         print_constraint_formal(prob.constraints[constraint_name])
        
#         constraint_name = f"x_diff_upper_{solution_index_suffix}_{k}"
#         prob += x_diff <= x_diff_plus + x_diff_minus, constraint_name
#         print_constraint_formal(prob.constraints[constraint_name])
        
#         # 类似的约束对于y坐标
#         y_diff = pulp.LpVariable(f"y_diff_{solution_index_suffix}_{k}", cat='Binary')
#         y_diff_plus = pulp.LpVariable(f"y_diff_plus_{solution_index_suffix}_{k}", cat='Binary')
#         y_diff_minus = pulp.LpVariable(f"y_diff_minus_{solution_index_suffix}_{k}", cat='Binary')
        
#         constraint_name = f"y_diff_plus_upper_{solution_index_suffix}_{k}"
#         prob += y[k] - y_prev[k] <= min_pos_diff - 0.001 + M * y_diff_plus, constraint_name
#         print_constraint_formal(prob.constraints[constraint_name])
        
#         constraint_name = f"y_diff_plus_lower_{solution_index_suffix}_{k}"
#         prob += y[k] - y_prev[k] >= min_pos_diff - M * (1 - y_diff_plus), constraint_name
#         print_constraint_formal(prob.constraints[constraint_name])
        
#         constraint_name = f"y_diff_minus_upper_{solution_index_suffix}_{k}"
#         prob += y[k] - y_prev[k] >= -min_pos_diff + 0.001 - M * y_diff_minus, constraint_name
#         print_constraint_formal(prob.constraints[constraint_name])
        
#         constraint_name = f"y_diff_minus_lower_{solution_index_suffix}_{k}"
#         prob += y[k] - y_prev[k] <= -min_pos_diff + M * (1 - y_diff_minus), constraint_name
#         print_constraint_formal(prob.constraints[constraint_name])
        
#         constraint_name = f"y_diff_from_plus_{solution_index_suffix}_{k}"
#         prob += y_diff >= y_diff_plus, constraint_name
#         print_constraint_formal(prob.constraints[constraint_name])
        
#         constraint_name = f"y_diff_from_minus_{solution_index_suffix}_{k}"
#         prob += y_diff >= y_diff_minus, constraint_name
#         print_constraint_formal(prob.constraints[constraint_name])
        
#         constraint_name = f"y_diff_upper_{solution_index_suffix}_{k}"
#         prob += y_diff <= y_diff_plus + y_diff_minus, constraint_name
#         print_constraint_formal(prob.constraints[constraint_name])
        
#         constraint_name = f"diff_pos_from_x_{solution_index_suffix}_{k}"
#         prob += diff_pos[k] >= x_diff, constraint_name
#         print_constraint_formal(prob.constraints[constraint_name])
        
#         constraint_name = f"diff_pos_from_y_{solution_index_suffix}_{k}"
#         prob += diff_pos[k] >= y_diff, constraint_name
#         print_constraint_formal(prob.constraints[constraint_name])
        
#         constraint_name = f"diff_pos_upper_{solution_index_suffix}_{k}"
#         prob += diff_pos[k] <= x_diff + y_diff, constraint_name
#         print_constraint_formal(prob.constraints[constraint_name])
    
#     # 位置排除约束：至少有一个非固定chiplet的位置不同
#     non_fixed_indices = [k for k in range(n) if fixed_chiplet_idx is None or k != fixed_chiplet_idx]
#     if len(non_fixed_indices) > 0:
#         constraint_name = f"exclude_solution_pos_{solution_index_suffix}"
#         prob += pulp.lpSum([diff_pos[k] for k in non_fixed_indices]) >= 1, constraint_name
#         print_constraint_formal(prob.constraints[constraint_name])


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
    
    参数:
        ctx: ILP模型上下文
        solution_index_suffix: 用于生成唯一约束名称的后缀
        min_pair_dist_diff: chiplet对之间距离差异的最小阈值
        prev_pair_distances_list: 之前所有解的chiplet对距离列表，每个元素是一个字典，key是(i,j)元组，value是距离
        M: Big-M常数
    """
    prob = ctx.prob
    n = len(ctx.nodes)
    W = ctx.W
    H = ctx.H
    grid_size = ctx.grid_size
    
    if grid_size is None:
        return
    
    if len(prev_pair_distances_list) == 0:
        return
    
    # 获取x_grid和y_grid变量
    x_grid = {}
    y_grid = {}
    for k in range(n):
        x_grid_var = prob.variablesDict().get(f"x_grid_{k}")
        y_grid_var = prob.variablesDict().get(f"y_grid_{k}")
        if x_grid_var is None or y_grid_var is None:
            return  # 如果没有grid变量，说明不是网格化模型
        x_grid[k] = x_grid_var
        y_grid[k] = y_grid_var
    
    # 计算grid的上界
    grid_w = int(math.ceil(W / grid_size))
    grid_h = int(math.ceil(H / grid_size))
    
    # 生成所有chiplet对（i < j，避免重复）
    chiplet_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            chiplet_pairs.append((i, j))
    
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
    epsilon = max(0.001, min_pair_dist_diff * 0.01)
    
    for i, j in chiplet_pairs:
        # 创建辅助变量表示grid坐标差的绝对值
        dx_grid_abs_ij = pulp.LpVariable(f"dx_grid_abs_pair_{solution_index_suffix}_{i}_{j}", lowBound=0, upBound=grid_w - 1, cat='Continuous')
        dy_grid_abs_ij = pulp.LpVariable(f"dy_grid_abs_pair_{solution_index_suffix}_{i}_{j}", lowBound=0, upBound=grid_h - 1, cat='Continuous')
        
        # 计算grid坐标的差
        # dx_grid_ij = x_grid[i] - x_grid[j]
        # dy_grid_ij = y_grid[i] - y_grid[j]
        
        # 使用Big-M方法添加绝对值约束：dx_grid_abs_ij = |x_grid[i] - x_grid[j]|
        dx_grid_diff = pulp.LpVariable(
            f"dx_grid_diff_{solution_index_suffix}_{i}_{j}",
            lowBound=-(grid_w - 1),
            upBound=grid_w - 1,
            cat='Continuous'
        )
        prob += dx_grid_diff == x_grid[i] - x_grid[j], f"dx_grid_diff_def_{solution_index_suffix}_{i}_{j}"
        M_dx = grid_w  # Big-M常数
        add_absolute_value_constraint_big_m(
            prob=prob,
            abs_var=dx_grid_abs_ij,
            orig_var=dx_grid_diff,
            M=M_dx,
            constraint_prefix=f"dx_grid_abs_pair_{solution_index_suffix}_{i}_{j}"
        )
        
        # 使用Big-M方法添加绝对值约束：dy_grid_abs_ij = |y_grid[i] - y_grid[j]|
        dy_grid_diff = pulp.LpVariable(
            f"dy_grid_diff_{solution_index_suffix}_{i}_{j}",
            lowBound=-(grid_h - 1),
            upBound=grid_h - 1,
            cat='Continuous'
        )
        prob += dy_grid_diff == y_grid[i] - y_grid[j], f"dy_grid_diff_def_{solution_index_suffix}_{i}_{j}"
        M_dy = grid_h  # Big-M常数
        add_absolute_value_constraint_big_m(
            prob=prob,
            abs_var=dy_grid_abs_ij,
            orig_var=dy_grid_diff,
            M=M_dy,
            constraint_prefix=f"dy_grid_abs_pair_{solution_index_suffix}_{i}_{j}"
        )
        
        # 创建ILP变量表示当前距离（grid单位）
        max_dist = grid_w + grid_h
        dist_curr_ij = pulp.LpVariable(
            f"dist_curr_pair_{solution_index_suffix}_{i}_{j}",
            lowBound=0,
            upBound=max_dist,
            cat='Continuous'
        )
        
        # 约束：当前距离 = dx_grid_abs_ij + dy_grid_abs_ij
        constraint_name = f"dist_curr_pair_def_{solution_index_suffix}_{i}_{j}"
        prob += dist_curr_ij == dx_grid_abs_ij + dy_grid_abs_ij, constraint_name
        print_constraint_formal(prob.constraints[constraint_name])
        
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
            continue
        
        # 创建二进制变量：该对的距离是否与之前所有解都不同
        diff_dist_pair[(i, j)] = pulp.LpVariable(f"diff_dist_pair_{solution_index_suffix}_{i}_{j}", cat='Binary')
        
        # 对于每个之前解，创建二进制变量表示该对的距离是否与该解的距离相同
        for prev_idx, prev_pair_distances in enumerate(prev_pair_distances_list):
            dist_prev_ij = prev_pair_distances[(i, j)]
            # 创建二进制变量：当前距离是否与解prev_idx的距离相同
            same_dist_pair_prev[((i, j), prev_idx)] = pulp.LpVariable(
                f"same_dist_pair_{solution_index_suffix}_{i}_{j}_prev{prev_idx}", 
                cat='Binary'
            )
            
            # 创建辅助变量表示距离差
            dist_diff_ij = pulp.LpVariable(
                f"dist_diff_pair_{solution_index_suffix}_{i}_{j}_prev{prev_idx}", 
                lowBound=-max_dist_diff, 
                upBound=max_dist_diff
            )
            
            # 距离差 = 当前距离 - 之前距离
            constraint_name = f"dist_diff_pair_def_{solution_index_suffix}_{i}_{j}_prev{prev_idx}"
            prob += dist_diff_ij == dist_curr_ij - dist_prev_ij, constraint_name
            print_constraint_formal(prob.constraints[constraint_name])
            
            # 使用Big-M方法添加绝对值约束：dist_diff_abs_ij = |dist_diff_ij|
            dist_diff_abs_ij = pulp.LpVariable(
                f"dist_diff_abs_pair_{solution_index_suffix}_{i}_{j}_prev{prev_idx}", 
                lowBound=0, 
                upBound=max_dist_diff
            )
            M_dist = max_dist_diff  # Big-M常数
            add_absolute_value_constraint_big_m(
                prob=prob,
                abs_var=dist_diff_abs_ij,
                orig_var=dist_diff_ij,
                M=M_dist,
                constraint_prefix=f"dist_diff_abs_pair_{solution_index_suffix}_{i}_{j}_prev{prev_idx}"
            )
            
            # 约束逻辑：same_dist_pair_prev[((i,j), prev_idx)] = 1 当且仅当 dist_diff_abs_ij < min_pair_dist_diff
            # 如果 same_dist_pair_prev = 1，则 dist_diff_abs_ij < min_pair_dist_diff
            constraint_name = f"same_dist_pair_upper_{solution_index_suffix}_{i}_{j}_prev{prev_idx}"
            prob += dist_diff_abs_ij <= min_pair_dist_diff - epsilon + M * (1 - same_dist_pair_prev[((i, j), prev_idx)]), constraint_name
            print_constraint_formal(prob.constraints[constraint_name])
            
            # 如果 same_dist_pair_prev = 0，则 dist_diff_abs_ij >= min_pair_dist_diff
            constraint_name = f"same_dist_pair_lower_{solution_index_suffix}_{i}_{j}_prev{prev_idx}"
            prob += dist_diff_abs_ij >= min_pair_dist_diff - M * same_dist_pair_prev[((i, j), prev_idx)], constraint_name
            print_constraint_formal(prob.constraints[constraint_name])
        
        # 约束：如果 diff_dist_pair[(i,j)] = 1，则对于所有之前解，same_dist_pair_prev = 0
        # 即：diff_dist_pair[(i,j)] = 1 当且仅当对于所有之前解，same_dist_pair_prev = 0
        for prev_idx in range(len(prev_pair_distances_list)):
            # 如果 diff_dist_pair = 1，则 same_dist_pair_prev = 0
            constraint_name = f"diff_dist_pair_implies_not_same_{solution_index_suffix}_{i}_{j}_prev{prev_idx}"
            prob += same_dist_pair_prev[((i, j), prev_idx)] <= 1 - diff_dist_pair[(i, j)], constraint_name
            print_constraint_formal(prob.constraints[constraint_name])
            
            # 如果至少有一个 same_dist_pair_prev = 1，则 diff_dist_pair = 0
            constraint_name = f"not_same_implies_diff_dist_pair_{solution_index_suffix}_{i}_{j}_prev{prev_idx}"
            prob += diff_dist_pair[(i, j)] <= 1 - same_dist_pair_prev[((i, j), prev_idx)], constraint_name
            print_constraint_formal(prob.constraints[constraint_name])
        
        # 如果所有 same_dist_pair_prev = 0，则 diff_dist_pair = 1
        constraint_name = f"all_not_same_implies_diff_dist_pair_{solution_index_suffix}_{i}_{j}"
        prob += diff_dist_pair[(i, j)] >= 1 - pulp.lpSum([same_dist_pair_prev[((i, j), prev_idx)] for prev_idx in range(len(prev_pair_distances_list))]), constraint_name
        print_constraint_formal(prob.constraints[constraint_name])
    
    # 顶层约束：至少有一对chiplet的距离与之前所有解都不同
    if len(diff_dist_pair) > 0:
        constraint_name = f"exclude_solution_dist_pair_{solution_index_suffix}"
        prob += pulp.lpSum([diff_dist_pair[pair] for pair in diff_dist_pair.keys()]) >= 1, constraint_name
        

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

    prob = ctx.prob
    x = ctx.x
    y = ctx.y
    n = len(ctx.nodes)
    W = ctx.W
    H = ctx.H
    fixed_chiplet_idx = ctx.fixed_chiplet_idx  # 获取固定的chiplet索引
    
    
    # 如果没有提供之前解的位置和距离列表，则读取当前解作为上一解
    if prev_positions is None or len(prev_positions) == 0:
        # 读取当前解中每个chiplet的位置
        x_prev = {}
        y_prev = {}
        valid_count = 0
        
        for k in range(n):
            x_val = pulp.value(x[k])
            y_val = pulp.value(y[k])
            
            if x_val is None or y_val is None:
                continue
            
            x_prev[k] = float(x_val)
            y_prev[k] = float(y_val)
            valid_count += 1
        
        if valid_count < n:
            return
        
        prev_positions = [{k: (x_prev[k], y_prev[k]) for k in x_prev.keys()}]
    
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
    if prev_pair_distances_list is not None and len(prev_pair_distances_list) > 0:
        # 确保每个约束都有唯一的名称，避免冲突
        # 使用全局约束计数器确保唯一性
        if constraint_counter is not None:
            current_counter = constraint_counter[0]
            solution_index_suffix = f"c{current_counter}"
            constraint_counter[0] += 1
        else:
            solution_index_suffix = f"{solution_index}"
        
        _add_exclude_dist_constraint(
            ctx=ctx,
            solution_index_suffix=solution_index_suffix,
            min_pair_dist_diff=min_pair_dist_diff,
            prev_pair_distances_list=prev_pair_distances_list,
            M=M,
        )


def print_all_variables(
    ctx: ILPModelContext, 
    result: ILPPlacementResult,
    prev_pair_distances_list: Optional[List[Dict[Tuple[int, int], float]]] = None
) -> None:
    """
    打印所有变量的值，包括排除约束相关的变量。
    
    参数:
        ctx: ILP模型上下文
        result: 求解结果
        prev_pair_distances_list: 可选，之前所有解的chiplet对距离列表，用于显示对比信息
    """
    if result.status != "Optimal":
        return
    
    nodes = ctx.nodes
    n = len(nodes)
    
    print("\n" + "=" * 80)
    print("变量值详情")
    print("=" * 80)
    
    # 1. 坐标变量 (x, y)
    print("\n【坐标变量】")
    for k in range(n):
        x_val = pulp.value(ctx.x[k]) if ctx.x[k] is not None else None
        y_val = pulp.value(ctx.y[k]) if ctx.y[k] is not None else None
        node_name = nodes[k].name if hasattr(nodes[k], 'name') else f"Chiplet_{k}"
        print(f"  x[{k}] ({node_name}): {x_val}")
        print(f"  y[{k}] ({node_name}): {y_val}")
    
    # 2. 网格坐标变量 (x_grid, y_grid)
    print("\n【网格坐标变量】")
    for k in range(n):
        x_grid_var = ctx.prob.variablesDict().get(f"x_grid_{k}")
        y_grid_var = ctx.prob.variablesDict().get(f"y_grid_{k}")
        x_grid_val = pulp.value(x_grid_var) if x_grid_var is not None else None
        y_grid_val = pulp.value(y_grid_var) if y_grid_var is not None else None
        node_name = nodes[k].name if hasattr(nodes[k], 'name') else f"Chiplet_{k}"
        print(f"  x_grid[{k}] ({node_name}): {x_grid_val}")
        print(f"  y_grid[{k}] ({node_name}): {y_grid_val}")
    
    # 3. 旋转变量 (r)
    print("\n【旋转变量】")
    for k in range(n):
        r_val = pulp.value(ctx.r[k]) if ctx.r[k] is not None else None
        rotated_str = "是" if (r_val is not None and r_val > 0.5) else "否"
        node_name = nodes[k].name if hasattr(nodes[k], 'name') else f"Chiplet_{k}"
        print(f"  r[{k}] ({node_name}): {r_val} (旋转: {rotated_str})")
    
    # 4. 宽度和高度变量 (w, h)
    print("\n【尺寸变量】")
    for k in range(n):
        w_var = ctx.prob.variablesDict().get(f"w_{k}")
        h_var = ctx.prob.variablesDict().get(f"h_{k}")
        w_val = pulp.value(w_var) if w_var is not None else None
        h_val = pulp.value(h_var) if h_var is not None else None
        node_name = nodes[k].name if hasattr(nodes[k], 'name') else f"Chiplet_{k}"
        print(f"  w[{k}] ({node_name}): {w_val}")
        print(f"  h[{k}] ({node_name}): {h_val}")
    
    # 5. 中心坐标变量 (cx, cy)
    if hasattr(ctx, 'cx') and ctx.cx is not None:
        print("\n【中心坐标变量】")
        for k in range(n):
            cx_val = pulp.value(ctx.cx[k]) if ctx.cx[k] is not None else None
            cy_val = pulp.value(ctx.cy[k]) if ctx.cy[k] is not None else None
            node_name = nodes[k].name if hasattr(nodes[k], 'name') else f"Chiplet_{k}"
            print(f"  cx[{k}] ({node_name}): {cx_val}")
            print(f"  cy[{k}] ({node_name}): {cy_val}")
    
    # 6. 相邻方式变量 (z1, z2, z1L, z1R, z2D, z2U)
    if len(ctx.connected_pairs) > 0:
        print("\n【相邻方式变量】")
        for i, j in ctx.connected_pairs:
            name_i = nodes[i].name if hasattr(nodes[i], 'name') else f"Chiplet_{i}"
            name_j = nodes[j].name if hasattr(nodes[j], 'name') else f"Chiplet_{j}"
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
    for i in range(n):
        for j in range(i + 1, n):
            all_pairs.append((i, j))
    
    for i, j in all_pairs:
        name_i = nodes[i].name if hasattr(nodes[i], 'name') else f"Chiplet_{i}"
        name_j = nodes[j].name if hasattr(nodes[j], 'name') else f"Chiplet_{j}"
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
            if not (var_name.startswith("dx_abs_pair_") or var_name.startswith("dy_abs_pair_") or \
                    var_name.startswith("dx_grid_abs_pair_") or var_name.startswith("dy_grid_abs_pair_")):
                val = pulp.value(var) if var is not None else None
                if val is not None:
                    other_vars.append((var_name, val))
    
    if other_vars:
        for var_name, val in sorted(other_vars):
            print(f"  {var_name}: {val}")
    else:
        print("  (无)")
    
    # 10. 排除解约束相关变量和约束（仅在第二次及以后的求解中打印）
    exclude_vars = []
    # 收集所有排除解约束相关的变量，包括所有可能的变量名模式
    for var_name, var in ctx.prob.variablesDict().items():
        # 检查是否是排除解约束相关的变量
        is_exclude_var = (
            var_name.startswith("dx_abs_pair_") or 
            var_name.startswith("dy_abs_pair_") or 
            var_name.startswith("dx_grid_abs_pair_") or 
            var_name.startswith("dy_grid_abs_pair_") or 
            var_name.startswith("dist_curr_pair_") or 
            var_name.startswith("dist_diff_pair_") or 
            var_name.startswith("dist_diff_abs_pair_") or 
            var_name.startswith("diff_dist_pair_") or 
            var_name.startswith("same_dist_pair_")
        )
        if is_exclude_var:
            val = pulp.value(var) if var is not None else None
            # 即使值为None也记录，以便调试
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
        dx_grid_abs_pair_vars = []
        dy_grid_abs_pair_vars = []
        dist_curr_pair_vars = []
        dist_diff_pair_vars = []
        dist_diff_abs_pair_vars = []
        diff_dist_pair_vars = []
        same_dist_pair_vars = []
        
        for var_name, val in exclude_vars:
            if var_name.startswith("dx_grid_abs_pair_"):
                dx_grid_abs_pair_vars.append((var_name, val))
            elif var_name.startswith("dy_grid_abs_pair_"):
                dy_grid_abs_pair_vars.append((var_name, val))
            elif var_name.startswith("dist_curr_pair_"):
                dist_curr_pair_vars.append((var_name, val))
            elif var_name.startswith("dx_abs_pair_"):
                dx_abs_pair_vars.append((var_name, val))
            elif var_name.startswith("dy_abs_pair_"):
                dy_abs_pair_vars.append((var_name, val))
            elif var_name.startswith("dist_diff_pair_") and not var_name.startswith("dist_diff_abs_pair_"):
                dist_diff_pair_vars.append((var_name, val))
            elif var_name.startswith("dist_diff_abs_pair_"):
                dist_diff_abs_pair_vars.append((var_name, val))
            elif var_name.startswith("diff_dist_pair_"):
                diff_dist_pair_vars.append((var_name, val))
            elif var_name.startswith("same_dist_pair_"):
                same_dist_pair_vars.append((var_name, val))
        
        if dx_grid_abs_pair_vars:
            print("\n  dx_grid_abs_pair (chiplet对的x方向grid坐标距离绝对值):")
            for var_name, val in sorted(dx_grid_abs_pair_vars):
                print(f"    {var_name}: {val}")
        
        if dy_grid_abs_pair_vars:
            print("\n  dy_grid_abs_pair (chiplet对的y方向grid坐标距离绝对值):")
            for var_name, val in sorted(dy_grid_abs_pair_vars):
                print(f"    {var_name}: {val}")
        
        # 按chiplet对组织显示，使输出更清晰
        import re
        pair_info = {}  # key: (i, j), value: dict with all related vars
        
        # 解析所有变量，按chiplet对分组
        unmatched_vars = []  # 记录无法匹配的变量
        for var_name, val in exclude_vars:
            # 匹配模式：{prefix}_{suffix}_{i}_{j} 或 {prefix}_{suffix}_{i}_{j}_prev{prev_idx}
            # 注意：变量名可能是 dist_diff_abs_pair_{suffix}_{i}_{j}_prev{prev_idx}
            match = re.search(r'([^_]+(?:_[^_]+)*)_[^_]+_(\d+)_(\d+)(?:_prev(\d+))?', var_name)
            if match:
                prefix = match.group(1)
                i_val = int(match.group(2))
                j_val = int(match.group(3))
                prev_idx = match.group(4)
                pair_key = (i_val, j_val)
                
                if pair_key not in pair_info:
                    pair_info[pair_key] = {
                        'dx_grid_abs': None,
                        'dy_grid_abs': None,
                        'dist_curr': None,
                        'dist_diff': {},
                        'dist_diff_abs': {},
                        'diff_dist': None,
                        'same_dist': {}
                    }
                
                # 处理各种变量前缀
                if prefix == 'dx_grid_abs_pair':
                    pair_info[pair_key]['dx_grid_abs'] = val
                elif prefix == 'dy_grid_abs_pair':
                    pair_info[pair_key]['dy_grid_abs'] = val
                elif prefix == 'dist_curr_pair':
                    pair_info[pair_key]['dist_curr'] = val
                elif prefix == 'dist_diff_pair' and prev_idx:
                    pair_info[pair_key]['dist_diff'][int(prev_idx)] = val
                elif prefix == 'dist_diff_abs_pair' and prev_idx:
                    pair_info[pair_key]['dist_diff_abs'][int(prev_idx)] = val
                elif prefix == 'diff_dist_pair':
                    pair_info[pair_key]['diff_dist'] = val
                elif prefix == 'same_dist_pair' and prev_idx:
                    pair_info[pair_key]['same_dist'][int(prev_idx)] = val
                else:
                    # 无法匹配的变量，记录到unmatched_vars
                    unmatched_vars.append((var_name, val))
            else:
                # 无法解析的变量，记录到unmatched_vars
                unmatched_vars.append((var_name, val))
        
        # 按chiplet对显示详细信息
        if pair_info:
            print("\n  【按chiplet对分组显示】")
            for (i, j) in sorted(pair_info.keys()):
                info = pair_info[(i, j)]
                name_i = nodes[i].name if hasattr(nodes[i], 'name') and i < len(nodes) else f"Chiplet_{i}"
                name_j = nodes[j].name if hasattr(nodes[j], 'name') and j < len(nodes) else f"Chiplet_{j}"
                
                print(f"\n    模块对 ({name_i}, {name_j}) [索引: ({i}, {j})]:")
                
                if info['dx_grid_abs'] is not None:
                    print(f"      dx_grid_abs (x方向grid距离): {info['dx_grid_abs']:.2f}")
                if info['dy_grid_abs'] is not None:
                    print(f"      dy_grid_abs (y方向grid距离): {info['dy_grid_abs']:.2f}")
                if info['dist_curr'] is not None:
                    print(f"      dist_curr (当前距离，grid单位): {info['dist_curr']:.2f}")
                    print(f"        验证: dx_grid_abs + dy_grid_abs = {info['dx_grid_abs']:.2f} + {info['dy_grid_abs']:.2f} = {info['dx_grid_abs'] + info['dy_grid_abs']:.2f}")
                
                if info['dist_diff'] or info['dist_diff_abs']:
                    print(f"      与之前解的距离比较:")
                    for prev_idx in sorted(set(list(info['dist_diff'].keys()) + list(info['dist_diff_abs'].keys()))):
                        dist_diff = info['dist_diff'].get(prev_idx, None)
                        dist_diff_abs = info['dist_diff_abs'].get(prev_idx, None)
                        same_dist = info['same_dist'].get(prev_idx, None)
                        
                        # 显示之前解的距离（如果可用）
                        prev_dist = None
                        if prev_pair_distances_list and prev_idx < len(prev_pair_distances_list):
                            prev_dist = prev_pair_distances_list[prev_idx].get((i, j), None)
                        
                        print(f"        解 {prev_idx}:")
                        if prev_dist is not None:
                            print(f"          之前解的距离: {prev_dist:.2f} (grid单位)")
                        if info['dist_curr'] is not None:
                            print(f"          当前解的距离: {info['dist_curr']:.2f} (grid单位)")
                        if dist_diff is not None:
                            print(f"          距离差 (dist_diff): {dist_diff:.2f}")
                            if prev_dist is not None and info['dist_curr'] is not None:
                                print(f"            验证: {info['dist_curr']:.2f} - {prev_dist:.2f} = {dist_diff:.2f}")
                        if dist_diff_abs is not None:
                            print(f"          距离差绝对值 (dist_diff_abs): {dist_diff_abs:.2f}")
                        if same_dist is not None:
                            same_str = "是" if same_dist > 0.5 else "否"
                            print(f"          是否相同 (same_dist_pair): {same_dist} ({same_str})")
                            if dist_diff_abs is not None:
                                if same_dist > 0.5:
                                    print(f"            → 距离差 {dist_diff_abs:.2f} < 阈值，标记为相同")
                                else:
                                    print(f"            → 距离差 {dist_diff_abs:.2f} >= 阈值，标记为不同")
                
                if info['diff_dist'] is not None:
                    diff_str = "是" if info['diff_dist'] > 0.5 else "否"
                    print(f"      diff_dist_pair (与所有之前解都不同): {info['diff_dist']} ({diff_str})")
                    if info['diff_dist'] > 0.5:
                        print(f"        → 该chiplet对的距离与所有之前解都不同，满足排除约束")
        
        # 保留原有的详细变量列表输出（作为补充）
        if dist_curr_pair_vars:
            print("\n  【详细变量列表 - dist_curr_pair】")
            for var_name, val in sorted(dist_curr_pair_vars):
                print(f"    {var_name}: {val:.2f}")
        
        if dx_abs_pair_vars:
            print("\n  【详细变量列表 - dx_abs_pair (旧版本)】")
            for var_name, val in sorted(dx_abs_pair_vars):
                print(f"    {var_name}: {val:.2f}")
        
        if dy_abs_pair_vars:
            print("\n  【详细变量列表 - dy_abs_pair (旧版本)】")
            for var_name, val in sorted(dy_abs_pair_vars):
                print(f"    {var_name}: {val:.2f}")
        
        if dist_diff_pair_vars:
            print("\n  【详细变量列表 - dist_diff_pair】")
            for var_name, val in sorted(dist_diff_pair_vars):
                print(f"    {var_name}: {val:.2f}")
        
        if dist_diff_abs_pair_vars:
            print("\n  【详细变量列表 - dist_diff_abs_pair】")
            for var_name, val in sorted(dist_diff_abs_pair_vars):
                if val is not None:
                    print(f"    {var_name}: {val:.2f}")
                else:
                    print(f"    {var_name}: None (未求解)")
        
        if diff_dist_pair_vars:
            print("\n  【详细变量列表 - diff_dist_pair (二进制)】")
            for var_name, val in sorted(diff_dist_pair_vars):
                if val is not None:
                    diff_str = "是" if val > 0.5 else "否"
                    print(f"    {var_name}: {val} ({diff_str})")
                else:
                    print(f"    {var_name}: None (未求解)")
        
        # 打印所有其他排除解约束相关的变量（包括无法匹配的）
        if unmatched_vars:
            print("\n  【其他排除解约束相关变量（未在分组中显示）】")
            for var_name, val in sorted(unmatched_vars):
                if val is not None:
                    print(f"    {var_name}: {val}")
                else:
                    print(f"    {var_name}: None (未求解)")
        
        # 打印所有排除解约束相关变量的完整列表（用于调试）
        print("\n  【完整变量列表（所有排除解约束相关变量）】")
        for var_name, val in sorted(exclude_vars):
            if val is not None:
                # 根据变量类型格式化输出
                if var_name.startswith("diff_dist_pair_") or var_name.startswith("same_dist_pair_"):
                    # 二进制变量
                    binary_str = "是" if val > 0.5 else "否"
                    print(f"    {var_name}: {val} ({binary_str})")
                elif isinstance(val, (int, float)):
                    # 数值变量
                    print(f"    {var_name}: {val:.4f}")
                else:
                    print(f"    {var_name}: {val}")
            else:
                print(f"    {var_name}: None (未求解)")
        
        if same_dist_pair_vars:
            print("\n  same_dist_pair (chiplet对的距离是否与某个之前解相同，二进制变量):")
            # 按chiplet对和之前解索引分组显示
            same_dist_by_pair = {}
            import re
            for var_name, val in same_dist_pair_vars:
                # 解析变量名：same_dist_pair_{suffix}_{i}_{j}_prev{prev_idx}
                # 使用正则表达式匹配：same_dist_pair_*_数字_数字_prev数字
                match = re.search(r'same_dist_pair_[^_]+_(\d+)_(\d+)_prev(\d+)', var_name)
                if match:
                    i_val = int(match.group(1))
                    j_val = int(match.group(2))
                    prev_idx = int(match.group(3))
                    pair_key = (i_val, j_val, prev_idx)
                    if pair_key not in same_dist_by_pair:
                        same_dist_by_pair[pair_key] = []
                    same_dist_by_pair[pair_key].append((var_name, val))
                else:
                    # 如果正则匹配失败，直接显示变量名
                    if "unknown" not in same_dist_by_pair:
                        same_dist_by_pair["unknown"] = []
                    same_dist_by_pair["unknown"].append((var_name, val))
            
            # 按chiplet对和之前解索引排序显示
            for key, vars_list in sorted(same_dist_by_pair.items()):
                if key == "unknown":
                    print("    无法解析的变量:")
                    for var_name, val in sorted(vars_list):
                        print(f"      {var_name}: {val}")
                else:
                    i, j, prev_idx = key
                    name_i = nodes[i].name if hasattr(nodes[i], 'name') and i < len(nodes) else f"Chiplet_{i}"
                    name_j = nodes[j].name if hasattr(nodes[j], 'name') and j < len(nodes) else f"Chiplet_{j}"
                    print(f"    模块对 ({name_i}, {name_j}) 与解 {prev_idx}:")
                    for var_name, val in sorted(vars_list):
                        print(f"      {var_name}: {val}")
        
        # 10.2 打印排除约束相关的约束
        print("\n【排除约束】")
        exclude_constraints = []
        for constraint_name, constraint in ctx.prob.constraints.items():
            if constraint_name.startswith("dx_abs_pair_") or constraint_name.startswith("dy_abs_pair_") or \
               constraint_name.startswith("dx_grid_abs_pair_") or constraint_name.startswith("dy_grid_abs_pair_") or \
               constraint_name.startswith("dist_curr_pair_") or \
               constraint_name.startswith("dist_diff_pair_") or constraint_name.startswith("dist_diff_abs_pair_") or \
               constraint_name.startswith("exclude_solution_dist_pair_") or \
               constraint_name.startswith("same_dist_pair_") or constraint_name.startswith("diff_dist_pair_implies_") or \
               constraint_name.startswith("not_same_implies_") or constraint_name.startswith("all_not_same_implies_"):
                exclude_constraints.append(constraint_name)
        
        if exclude_constraints:
            print(f"  共找到 {len(exclude_constraints)} 个排除约束:")
            for constraint_name in sorted(exclude_constraints):
                constraint = ctx.prob.constraints[constraint_name]
                print(f"    {constraint_name}: {constraint}")
        else:
            print("  (未找到排除约束)")
    else:
        print("\n【排除解约束】")
        print("  (第一次求解，无排除约束)")


def search_multiple_solutions(
    num_solutions: int = 3,
    min_shared_length: float = 0.5,
    input_json_path: Optional[str] = None,
    nodes: Optional[List] = None,
    edges: Optional[List[Tuple[int, int]]] = None,
    grid_size: Optional[float] = None,
    fixed_chiplet_idx: Optional[int] = None,
    min_pos_diff: Optional[float] = None,  # 位置排除约束的最小变化量，如果为None则使用grid_size或默认值
    min_pair_dist_diff: Optional[float] = None,  # chiplet对之间距离差异的最小阈值，如果为None则使用min_pos_diff；如果min_pos_diff也为None，则使用grid_size或默认值；此参数控制距离排除约束：至少有一对chiplet的距离差必须 >= min_pair_dist_diff
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
        min_pos_diff: 位置排除约束的最小变化量，如果为None则使用grid_size或默认值
        min_pair_dist_diff: chiplet对之间距离差异的最小阈值，如果为None则使用min_pos_diff；如果min_pos_diff也为None，则使用grid_size或默认值；此参数控制距离排除约束：至少有一对chiplet的距离差必须 >= min_pair_dist_diff
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
    all_prev_positions = []
    all_prev_grid_positions = []  # 保存之前解的grid坐标
    all_prev_pair_distances = []
    
    # 全局约束计数器，确保每个约束名称唯一
    constraint_counter = [0]
    
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
        if len(all_prev_positions) > 0:
            add_exclude_constraint(
                ctx=ctx,
                solution_index=i,
                min_pos_diff=min_pos_diff,
                min_pair_dist_diff=min_pair_dist_diff,
                prev_positions=all_prev_positions,
                prev_pair_distances_list=all_prev_pair_distances,
                constraint_counter=constraint_counter,
            )
        
        # 导出LP文件（在求解之前，包含所有约束）
        # 确定输出目录
        if output_dir is None:
            # 默认输出目录：相对于项目根目录的output目录
            default_output = Path(__file__).parent.parent / "output"
            output_dir_path = default_output
        else:
            output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        lp_file = output_dir_path / f"constraints_solution_{i+1}.lp"
        ctx.prob.writeLP(str(lp_file))
        print(f"\nLP文件已保存: {lp_file}")
        
        # 求解
        result = solve_placement_ilp_from_model(ctx)
        
        # result.status 是字符串类型（如 "Optimal"），需要与字符串比较
        if result.status != "Optimal":
            print(f"\n求解状态: {result.status}，停止搜索")
            break
        
        solutions.append(result)
        
        # 打印所有变量的值（包括排除解约束相关变量）
        print_all_variables(ctx, result, prev_pair_distances_list=all_prev_pair_distances)
        
        # 保存当前解的位置
        # result.layout 是 Dict[str, Tuple[float, float]]，key 是 chiplet 名称
        x_prev = {}
        y_prev = {}
        for k, node in enumerate(nodes):
            node_name = node.name if hasattr(node, 'name') else f"Chiplet_{k}"
            if node_name in result.layout:
                x_prev[k], y_prev[k] = result.layout[node_name]
            else:
                # 如果找不到，尝试使用索引作为key（向后兼容）
                x_prev[k] = 0.0
                y_prev[k] = 0.0
        all_prev_positions.append({k: (x_prev[k], y_prev[k]) for k in x_prev.keys()})
        
        # 计算并保存当前解的chiplet对之间的距离（使用grid坐标）
        pair_distances = {}
        grid_size = ctx.grid_size if ctx.grid_size is not None else 1.0
        
        # 获取grid坐标值
        x_grid_prev = {}
        y_grid_prev = {}
        for k in range(len(nodes)):
            x_grid_var = ctx.prob.variablesDict().get(f"x_grid_{k}")
            y_grid_var = ctx.prob.variablesDict().get(f"y_grid_{k}")
            if x_grid_var is not None and y_grid_var is not None:
                x_grid_val = pulp.value(x_grid_var)
                y_grid_val = pulp.value(y_grid_var)
                if x_grid_val is not None and y_grid_val is not None:
                    x_grid_prev[k] = int(x_grid_val)
                    y_grid_prev[k] = int(y_grid_val)
                else:
                    # 如果无法获取grid坐标，使用实际坐标转换为grid坐标
                    x_grid_prev[k] = int(round(x_prev[k] / grid_size))
                    y_grid_prev[k] = int(round(y_prev[k] / grid_size))
            else:
                # 如果没有grid变量，使用实际坐标转换为grid坐标
                x_grid_prev[k] = int(round(x_prev[k] / grid_size))
                y_grid_prev[k] = int(round(y_prev[k] / grid_size))
        
        # 保存当前解的grid坐标
        all_prev_grid_positions.append({k: (x_grid_prev[k], y_grid_prev[k]) for k in x_grid_prev.keys()})
        
        for i_idx in range(len(nodes)):
            for j_idx in range(i_idx + 1, len(nodes)):
                # 使用grid坐标计算距离（曼哈顿距离）
                dx_grid = abs(x_grid_prev[i_idx] - x_grid_prev[j_idx])
                dy_grid = abs(y_grid_prev[i_idx] - y_grid_prev[j_idx])
                dist = dx_grid + dy_grid
                pair_distances[(i_idx, j_idx)] = dist
        all_prev_pair_distances.append(pair_distances)
        
        # 输出当前解的信息
        print(f"\n=== 解 {i+1} ===")
        print(f"目标函数值: {result.objective_value:.4f}")
        
        # 输出chiplet坐标（左下角）
        print("\nChiplet坐标（左下角）:")
        for k, node in enumerate(nodes):
            node_name = node.name if hasattr(node, 'name') else f"Chiplet_{k}"
            print(f"  {node_name}: ({x_prev[k]:.2f}, {y_prev[k]:.2f})")
        
        # 输出chiplet中心坐标
        print("\nChiplet中心坐标:")
        for k, node in enumerate(nodes):
            node_name = node.name if hasattr(node, 'name') else f"Chiplet_{k}"
            w = float(node.dimensions.get("x", 0.0))
            h = float(node.dimensions.get("y", 0.0))
            cx_k = x_prev[k] + w / 2.0
            cy_k = y_prev[k] + h / 2.0
            print(f"  {node_name}: ({cx_k:.2f}, {cy_k:.2f})")
        
        # 检查chiplet之间是否发生重叠
        print("\n重叠检查:")
        overlaps = []
        for i_idx in range(len(nodes)):
            for j_idx in range(i_idx + 1, len(nodes)):
                node_i = nodes[i_idx]
                node_j = nodes[j_idx]
                name_i = node_i.name if hasattr(node_i, 'name') else f"Chiplet_{i_idx}"
                name_j = node_j.name if hasattr(node_j, 'name') else f"Chiplet_{j_idx}"
                
                w_i = float(node_i.dimensions.get("x", 0.0))
                h_i = float(node_i.dimensions.get("y", 0.0))
                w_j = float(node_j.dimensions.get("x", 0.0))
                h_j = float(node_j.dimensions.get("y", 0.0))
                
                x1_min, x1_max = x_prev[i_idx], x_prev[i_idx] + w_i
                y1_min, y1_max = y_prev[i_idx], y_prev[i_idx] + h_i
                x2_min, x2_max = x_prev[j_idx], x_prev[j_idx] + w_j
                y2_min, y2_max = y_prev[j_idx], y_prev[j_idx] + h_j
                
                # 计算重叠区域
                x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
                y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
                overlap_area = x_overlap * y_overlap
                
                if overlap_area > 0:
                    overlaps.append({
                        'pair': (name_i, name_j),
                        'overlap_area': overlap_area,
                        'x_overlap': x_overlap,
                        'y_overlap': y_overlap,
                        'rect1': (x1_min, y1_min, x1_max, y1_max),
                        'rect2': (x2_min, y2_min, x2_max, y2_max),
                    })
        
        if len(overlaps) == 0:
            print("  ✓ 无重叠：所有chiplet之间都没有重叠")
        else:
            print(f"  ✗ 发现 {len(overlaps)} 对chiplet发生重叠:")
            for overlap in overlaps:
                name_i, name_j = overlap['pair']
                print(f"    - {name_i} 和 {name_j}:")
                print(f"      重叠面积: {overlap['overlap_area']:.2f}")
                print(f"      x方向重叠长度: {overlap['x_overlap']:.2f}")
                print(f"      y方向重叠长度: {overlap['y_overlap']:.2f}")
                x1_min, y1_min, x1_max, y1_max = overlap['rect1']
                x2_min, y2_min, x2_max, y2_max = overlap['rect2']
                print(f"      {name_i} 范围: x=[{x1_min:.2f}, {x1_max:.2f}], y=[{y1_min:.2f}, {y1_max:.2f}]")
                print(f"      {name_j} 范围: x=[{x2_min:.2f}, {x2_max:.2f}], y=[{y2_min:.2f}, {y2_max:.2f}]")
        
        # 输出chiplet对之间的距离
        print("\nChiplet对之间的距离:")
        for (i_idx, j_idx), dist in sorted(pair_distances.items()):
            node_i_name = nodes[i_idx].name if hasattr(nodes[i_idx], 'name') else f"Chiplet_{i_idx}"
            node_j_name = nodes[j_idx].name if hasattr(nodes[j_idx], 'name') else f"Chiplet_{j_idx}"
            print(f"  ({node_i_name}, {node_j_name}): {dist:.2f}")
        
        # 输出与历史所有解的变化（使用grid坐标）
        if i > 0:
            # 位置变化：与上一个解比较
            prev_grid_pos_dict = all_prev_grid_positions[i-1]
            changed_chiplets = []
            for k in range(len(nodes)):
                if k in prev_grid_pos_dict:
                    prev_x_grid, prev_y_grid = prev_grid_pos_dict[k]
                    # 使用grid坐标比较位置变化
                    if x_grid_prev[k] != prev_x_grid or y_grid_prev[k] != prev_y_grid:
                        node_name = nodes[k].name if hasattr(nodes[k], 'name') else f"Chiplet_{k}"
                        changed_chiplets.append(node_name)
            
            # 距离变化：与历史上所有解比较
            # 找出所有与历史上所有解的距离都不同的chiplet对
            changed_pairs = []
            for (i_idx, j_idx), curr_dist in pair_distances.items():
                # 检查当前距离是否与历史上所有解的距离都不同
                is_different_from_all = True
                for prev_sol_idx in range(i):  # 遍历所有之前的解
                    if (i_idx, j_idx) in all_prev_pair_distances[prev_sol_idx]:
                        prev_dist = all_prev_pair_distances[prev_sol_idx][(i_idx, j_idx)]
                        if curr_dist == prev_dist:
                            # 如果与任何一个历史解的距离相同，则不算"距离变化"
                            is_different_from_all = False
                            break
                
                if is_different_from_all:
                    node_i_name = nodes[i_idx].name if hasattr(nodes[i_idx], 'name') else f"Chiplet_{i_idx}"
                    node_j_name = nodes[j_idx].name if hasattr(nodes[j_idx], 'name') else f"Chiplet_{j_idx}"
                    changed_pairs.append((node_i_name, node_j_name))
            
            print("\n与历史解的变化（基于grid坐标）:")
            if changed_chiplets:
                print(f"  位置变化的chiplet（与上一解比较）: {', '.join(changed_chiplets)}")
            else:
                print("  位置变化的chiplet（与上一解比较）: 无")
            if changed_pairs:
                print(f"  距离变化的chiplet对（与所有历史解都不同）: {', '.join([f'({p[0]},{p[1]})' for p in changed_pairs])}")
            else:
                print("  距离变化的chiplet对（与所有历史解都不同）: 无")
        
        # 绘制并保存布局图片
        layout_dict = {}
        fixed_chiplet_names = set()
        for k, node in enumerate(nodes):
            node_name = node.name if hasattr(node, 'name') else f"Chiplet_{k}"
            layout_dict[node_name] = (x_prev[k], y_prev[k])
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
        image_path = output_dir_path / f"solution_{i+1}_layout.png"
        
        try:
            draw_chiplet_diagram(
                nodes=nodes,
                edges=edges,
                save_path=str(image_path),
                layout=layout_dict,
                fixed_chiplet_names=fixed_chiplet_names if fixed_chiplet_names else None,
            )
            print(f"\n布局图片已保存: {image_path}")
        except Exception as e:
            print(f"\n警告: 保存布局图片时出错: {e}")
    
    return solutions


if __name__ == "__main__":
    import json
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python ilp_sub_solution_search.py <input_json_file>")
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
        min_pos_diff=0.5,
        min_pair_dist_diff=0.5,
    )
    
    print(f"\n总共找到 {len(solutions)} 个解")
