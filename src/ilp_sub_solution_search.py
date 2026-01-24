from __future__ import annotations

from typing import Dict, Tuple, List, Optional
from pathlib import Path
from copy import deepcopy
import math

import pulp

from tool import build_random_chiplet_graph, draw_chiplet_diagram, print_constraint_formal, print_pair_distances_only, print_all_variables
from ilp_method import (
    ILPModelContext,
    ILPPlacementResult,
    build_placement_ilp_model,
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
            # 即：same = 1 表示当前解的距离与之前解prev_idx的距离相差 < min_pair_dist_diff（距离相同或相近）
            #     same = 0 表示当前解的距离与之前解prev_idx的距离相差 >= min_pair_dist_diff（距离不同）
            # 如果 same_dist_pair_prev = 1，则 dist_diff_abs_ij < min_pair_dist_diff
            constraint_name = f"same_dist_pair_upper_{solution_index_suffix}_{i}_{j}_prev{prev_idx}"
            prob += dist_diff_abs_ij <= min_pair_dist_diff - epsilon + M * (1 - same_dist_pair_prev[((i, j), prev_idx)]), constraint_name
            print_constraint_formal(prob.constraints[constraint_name])
            
            # 如果 same_dist_pair_prev = 0，则 dist_diff_abs_ij >= min_pair_dist_diff
            constraint_name = f"same_dist_pair_lower_{solution_index_suffix}_{i}_{j}_prev{prev_idx}"
            prob += dist_diff_abs_ij >= min_pair_dist_diff - M * same_dist_pair_prev[((i, j), prev_idx)], constraint_name
            print_constraint_formal(prob.constraints[constraint_name])
        
        # 约束：diff_dist_pair[(i,j)] = 1 当且仅当对于所有之前解，same_dist_pair_prev = 0
        # 即：diff_dist_pair[(i,j)] = 1 表示当前解的距离与所有之前解的距离都相差 >= min_pair_dist_diff
        # 注意：只有当当前解的chiplet对之间的距离与之前所有解的对应chiplet对之间的距离，
        #      全部都相差至少min_pair_dist_diff阈值时，diff_dist_pair = 1
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
    # 即：至少存在一对chiplet，使得当前解的距离与所有之前解的距离都相差 >= min_pair_dist_diff
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
        ctx = build_placement_ilp_model(
        nodes=nodes,
        edges=edges,
            grid_size=grid_size if grid_size is not None else 1.0,  # 如果未指定grid_size，使用默认值1.0
            fixed_chiplet_idx=fixed_chiplet_idx,
        min_shared_length=min_shared_length,
        )
        
        # 添加排除约束（排除之前找到的所有解）
        if len(all_prev_pair_distances) > 0:
            add_exclude_constraint(
                ctx=ctx,
                solution_index=i,
                min_pair_dist_diff=min_pair_dist_diff,
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
        
        # 计算并保存当前解的chiplet对之间的距离（使用grid坐标）
        pair_distances = {}
        grid_size_val = ctx.grid_size if ctx.grid_size is not None else 1.0
        
        # 获取当前解的实际坐标（用于打印输出）
        x_prev = {}
        y_prev = {}
        for k, node in enumerate(nodes):
            node_name = node.name if hasattr(node, 'name') else f"Chiplet_{k}"
            if node_name in result.layout:
                x_prev[k], y_prev[k] = result.layout[node_name]
            else:
                x_prev[k] = 0.0
                y_prev[k] = 0.0
        
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
                    x_grid_prev[k] = int(round(x_prev[k] / grid_size_val))
                    y_grid_prev[k] = int(round(y_prev[k] / grid_size_val))
            else:
                # 如果没有grid变量，使用实际坐标转换为grid坐标
                x_grid_prev[k] = int(round(x_prev[k] / grid_size_val))
                y_grid_prev[k] = int(round(y_prev[k] / grid_size_val))
        
        for i_idx in range(len(nodes)):
            for j_idx in range(i_idx + 1, len(nodes)):
                # 使用grid坐标计算距离（曼哈顿距离）
                dx_grid = abs(x_grid_prev[i_idx] - x_grid_prev[j_idx])
                dy_grid = abs(y_grid_prev[i_idx] - y_grid_prev[j_idx])
                dist = dx_grid + dy_grid
                pair_distances[(i_idx, j_idx)] = dist
        all_prev_pair_distances.append(pair_distances)
        
        # 输出当前解的基本信息
        print(f"\n=== 解 {i+1} ===")
        print(f"目标函数值: {result.objective_value:.4f}")
        
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
        min_pair_dist_diff=0.5,
    )
    
    print(f"\n总共找到 {len(solutions)} 个解")
