from __future__ import annotations

from typing import Dict, Tuple, List, Optional
from pathlib import Path
from copy import deepcopy

import pulp

from tool import build_random_chiplet_graph, draw_chiplet_diagram
from ilp_method import (
    ILPModelContext,
    ILPPlacementResult,
    build_placement_ilp_model,
    build_placement_ilp_model_grid,
    solve_placement_ilp_from_model,
)


def _add_exclude_single_solution_constraint(
    ctx: ILPModelContext,
    x_prev: Dict[int, float],
    y_prev: Dict[int, float],
    *,
    solution_index_suffix: str,
    min_diff: float,
    min_pair_dist_diff: Optional[float],  # chiplet对之间距离差异的最小阈值，如果为None则跳过距离排除约束
    M: float,
) -> None:
    """
    为单个之前的解添加排除约束（内部辅助函数）。
    """
    prob = ctx.prob
    x = ctx.x
    y = ctx.y
    n = len(ctx.nodes)
    fixed_chiplet_idx = ctx.fixed_chiplet_idx
    W = ctx.W
    H = ctx.H
    cx = ctx.cx
    cy = ctx.cy
    
    # 为每个chiplet创建二进制变量，表示该chiplet的位置是否与之前解不同
    diff_pos = {}
    for k in range(n):
        diff_pos[k] = pulp.LpVariable(f"diff_pos_{solution_index_suffix}_{k}", cat='Binary')
    
    # 对于每个chiplet k，判断其位置是否与之前解不同
    for k in range(n):
        if fixed_chiplet_idx is not None and k == fixed_chiplet_idx:
            prob += diff_pos[k] == 0, f"diff_pos_fixed_{solution_index_suffix}_{k}"
            continue
        
        # 创建辅助变量判断x坐标是否不同
        x_diff = pulp.LpVariable(f"x_diff_{solution_index_suffix}_{k}", cat='Binary')
        x_diff_plus = pulp.LpVariable(f"x_diff_plus_{solution_index_suffix}_{k}", cat='Binary')
        x_diff_minus = pulp.LpVariable(f"x_diff_minus_{solution_index_suffix}_{k}", cat='Binary')
        
        prob += x[k] - x_prev[k] <= min_diff - 0.001 + M * x_diff_plus, f"x_diff_plus_upper_{solution_index_suffix}_{k}"
        prob += x[k] - x_prev[k] >= min_diff - M * (1 - x_diff_plus), f"x_diff_plus_lower_{solution_index_suffix}_{k}"
        prob += x[k] - x_prev[k] >= -min_diff + 0.001 - M * x_diff_minus, f"x_diff_minus_upper_{solution_index_suffix}_{k}"
        prob += x[k] - x_prev[k] <= -min_diff + M * (1 - x_diff_minus), f"x_diff_minus_lower_{solution_index_suffix}_{k}"
        
        prob += x_diff >= x_diff_plus, f"x_diff_from_plus_{solution_index_suffix}_{k}"
        prob += x_diff >= x_diff_minus, f"x_diff_from_minus_{solution_index_suffix}_{k}"
        prob += x_diff <= x_diff_plus + x_diff_minus, f"x_diff_upper_{solution_index_suffix}_{k}"
        
        # 类似的约束对于y坐标
        y_diff = pulp.LpVariable(f"y_diff_{solution_index_suffix}_{k}", cat='Binary')
        y_diff_plus = pulp.LpVariable(f"y_diff_plus_{solution_index_suffix}_{k}", cat='Binary')
        y_diff_minus = pulp.LpVariable(f"y_diff_minus_{solution_index_suffix}_{k}", cat='Binary')
        
        prob += y[k] - y_prev[k] <= min_diff - 0.001 + M * y_diff_plus, f"y_diff_plus_upper_{solution_index_suffix}_{k}"
        prob += y[k] - y_prev[k] >= min_diff - M * (1 - y_diff_plus), f"y_diff_plus_lower_{solution_index_suffix}_{k}"
        prob += y[k] - y_prev[k] >= -min_diff + 0.001 - M * y_diff_minus, f"y_diff_minus_upper_{solution_index_suffix}_{k}"
        prob += y[k] - y_prev[k] <= -min_diff + M * (1 - y_diff_minus), f"y_diff_minus_lower_{solution_index_suffix}_{k}"
        
        prob += y_diff >= y_diff_plus, f"y_diff_from_plus_{solution_index_suffix}_{k}"
        prob += y_diff >= y_diff_minus, f"y_diff_from_minus_{solution_index_suffix}_{k}"
        prob += y_diff <= y_diff_plus + y_diff_minus, f"y_diff_upper_{solution_index_suffix}_{k}"
        
        prob += diff_pos[k] >= x_diff, f"diff_pos_from_x_{solution_index_suffix}_{k}"
        prob += diff_pos[k] >= y_diff, f"diff_pos_from_y_{solution_index_suffix}_{k}"
        prob += diff_pos[k] <= x_diff + y_diff, f"diff_pos_upper_{solution_index_suffix}_{k}"
    
    # 位置排除约束：至少有一个非固定chiplet的位置不同
    non_fixed_indices = [k for k in range(n) if fixed_chiplet_idx is None or k != fixed_chiplet_idx]
    if len(non_fixed_indices) > 0:
        constraint_name = f"exclude_solution_pos_{solution_index_suffix}"
        prob += pulp.lpSum([diff_pos[k] for k in non_fixed_indices]) >= 1, constraint_name
        print(f"[约束调试] 添加位置排除约束: {constraint_name}")
        print(f"  - 非固定chiplet索引: {non_fixed_indices}")
        print(f"  - 上一解位置: {[(k, ctx.nodes[k].name, x_prev.get(k, 'N/A'), y_prev.get(k, 'N/A')) for k in non_fixed_indices]}")
        print(f"  - min_diff: {min_diff}")
    
    # 距离排除约束：chiplet之间两两距离，需要有其中一对的距离和之前的解不同
    # 逻辑：计算所有chiplet对之间的距离，至少有一对的距离必须不同
    # 距离不同的阈值由 min_pair_dist_diff 参数控制（用户可配置变量）
    # 如果 min_pair_dist_diff 为 None，则跳过距离排除约束
    if cx is not None and cy is not None and min_pair_dist_diff is not None:
        # 生成所有chiplet对（i < j，避免重复）
        chiplet_pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                chiplet_pairs.append((i, j))
        
        # 为每对chiplet创建二进制变量，表示该对的距离是否不同
        diff_dist_pair = {}
        valid_pairs = []
        
        for i, j in chiplet_pairs:
            # 计算当前解中chiplet i和j的中心之间的距离
            # 创建辅助变量表示距离的绝对值
            dx_abs_ij = pulp.LpVariable(f"dx_abs_pair_{solution_index_suffix}_{i}_{j}", lowBound=0, upBound=W)
            dy_abs_ij = pulp.LpVariable(f"dy_abs_pair_{solution_index_suffix}_{i}_{j}", lowBound=0, upBound=H)
            
            prob += dx_abs_ij >= cx[i] - cx[j], f"dx_abs_pair_pos_{solution_index_suffix}_{i}_{j}"
            prob += dx_abs_ij >= cx[j] - cx[i], f"dx_abs_pair_neg_{solution_index_suffix}_{i}_{j}"
            prob += dy_abs_ij >= cy[i] - cy[j], f"dy_abs_pair_pos_{solution_index_suffix}_{i}_{j}"
            prob += dy_abs_ij >= cy[j] - cy[i], f"dy_abs_pair_neg_{solution_index_suffix}_{i}_{j}"
            
            # 当前距离 = dx_abs_ij + dy_abs_ij
            dist_curr_ij = dx_abs_ij + dy_abs_ij
            
            # 计算上一解中chiplet i和j之间的距离
            # 从之前解的位置计算距离
            if i in x_prev and j in x_prev and i in y_prev and j in y_prev:
                # 计算上一解中chiplet i和j的中心坐标
                node_i = ctx.nodes[i]
                node_j = ctx.nodes[j]
                w_i = float(node_i.dimensions.get("x", 0.0))
                h_i = float(node_i.dimensions.get("y", 0.0))
                w_j = float(node_j.dimensions.get("x", 0.0))
                h_j = float(node_j.dimensions.get("y", 0.0))
                
                cx_i_prev = x_prev[i] + w_i / 2.0
                cy_i_prev = y_prev[i] + h_i / 2.0
                cx_j_prev = x_prev[j] + w_j / 2.0
                cy_j_prev = y_prev[j] + h_j / 2.0
                
                dist_prev_ij = abs(cx_i_prev - cx_j_prev) + abs(cy_i_prev - cy_j_prev)
                
                # 创建辅助变量表示距离差
                max_dist_diff = W + H
                dist_diff_ij = pulp.LpVariable(f"dist_diff_pair_{solution_index_suffix}_{i}_{j}", lowBound=-max_dist_diff, upBound=max_dist_diff)
                prob += dist_diff_ij == dist_curr_ij - dist_prev_ij, f"dist_diff_pair_def_{solution_index_suffix}_{i}_{j}"
                
                dist_diff_abs_ij = pulp.LpVariable(f"dist_diff_abs_pair_{solution_index_suffix}_{i}_{j}", lowBound=0, upBound=max_dist_diff)
                prob += dist_diff_abs_ij >= dist_diff_ij, f"dist_diff_abs_pair_pos_{solution_index_suffix}_{i}_{j}"
                prob += dist_diff_abs_ij >= -dist_diff_ij, f"dist_diff_abs_pair_neg_{solution_index_suffix}_{i}_{j}"
                
                # 为这对chiplet创建二进制变量，表示距离是否不同
                diff_dist_pair[(i, j)] = pulp.LpVariable(f"diff_dist_pair_{solution_index_suffix}_{i}_{j}", cat='Binary')
                
                # 约束逻辑：
                # - 如果 dist_diff_abs_ij >= min_pair_dist_diff，则 diff_dist_pair[(i,j)] 可以是 1
                # - 如果 dist_diff_abs_ij < min_pair_dist_diff，则 diff_dist_pair[(i,j)] 必须是 0
                # 使用 Big M 方法实现：
                # dist_diff_abs_ij >= min_pair_dist_diff - M * (1 - diff_dist_pair[(i,j)])
                # dist_diff_abs_ij <= min_pair_dist_diff - epsilon + M * diff_dist_pair[(i,j)]
                epsilon = max(0.001, min_pair_dist_diff * 0.01)
                # 如果 diff_dist_pair[(i,j)] = 1，则 dist_diff_abs_ij >= min_pair_dist_diff
                prob += dist_diff_abs_ij >= min_pair_dist_diff - M * (1 - diff_dist_pair[(i, j)]), f"dist_diff_abs_pair_lower_{solution_index_suffix}_{i}_{j}"
                # 如果 diff_dist_pair[(i,j)] = 0，则 dist_diff_abs_ij <= min_pair_dist_diff - epsilon
                prob += dist_diff_abs_ij <= min_pair_dist_diff - epsilon + M * diff_dist_pair[(i, j)], f"dist_diff_abs_pair_upper_{solution_index_suffix}_{i}_{j}"
                
                valid_pairs.append((i, j))
        
        # 顶层约束：至少有一对chiplet的距离不同
        if len(valid_pairs) > 0:
            constraint_name = f"exclude_solution_dist_pair_{solution_index_suffix}"
            prob += pulp.lpSum([diff_dist_pair[pair] for pair in valid_pairs]) >= 1, constraint_name
            print(f"[约束调试] 添加距离排除约束: {constraint_name}")
            print(f"  - 有效chiplet对数量: {len(valid_pairs)}")
            print(f"  - min_pair_dist_diff: {min_pair_dist_diff}")
            print(f"  - 上一解chiplet对距离: {[(pair, ctx.nodes[pair[0]].name, ctx.nodes[pair[1]].name, x_prev.get(pair[0], 'N/A'), y_prev.get(pair[0], 'N/A'), x_prev.get(pair[1], 'N/A'), y_prev.get(pair[1], 'N/A')) for pair in valid_pairs[:5]]}")  # 只打印前5对


def add_exclude_layout_constraint(
    ctx: ILPModelContext,
    *,
    require_change_pairs: int = 1,  # 目前没有用到这个参数，先保留接口
    solution_index: int = 0,  # 解的索引，用于生成唯一的约束名称
    min_diff: Optional[float] = None,  # 判断"不同"的最小差异阈值，如果为None则使用grid_size（已废弃，使用min_pos_diff）
    min_pos_diff: Optional[float] = None,  # 位置排除约束的最小变化量，如果为None则使用min_diff或grid_size
    min_pair_dist_diff: Optional[float] = None,  # chiplet对之间距离差异的最小阈值，如果为None则使用min_pos_diff
    prev_positions: Optional[List[Dict[int, Tuple[float, float]]]] = None,  # 之前所有解的位置列表
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
    
    # 对每个之前的解都添加排除约束
    for prev_idx, prev_pos_dict in enumerate(prev_positions):
        # 将位置字典转换为x和y字典
        x_prev = {k: pos[0] for k, pos in prev_pos_dict.items()}
        y_prev = {k: pos[1] for k, pos in prev_pos_dict.items()}
        
        # 确保每个约束都有唯一的名称，避免冲突
        # 使用全局约束计数器确保唯一性
        if constraint_counter is not None:
            current_counter = constraint_counter[0]
            solution_index_suffix = f"c{current_counter}_p{prev_idx}"
            constraint_counter[0] += 1
        else:
            solution_index_suffix = f"{solution_index}_prev{prev_idx}"
        
        # 调用辅助函数添加排除约束
        print(f"\n[约束调试] 为解 {solution_index} 添加排除约束（相对于解 {prev_idx}）")
        print(f"  - solution_index_suffix: {solution_index_suffix}")
        print(f"  - min_pos_diff: {min_pos_diff}")
        print(f"  - min_pair_dist_diff: {min_pair_dist_diff}")
        print(f"  - M (Big-M常数): {M}")
        _add_exclude_single_solution_constraint(
            ctx=ctx,
            x_prev=x_prev,
            y_prev=y_prev,
            solution_index_suffix=solution_index_suffix,
            min_diff=min_pos_diff,  # 使用min_pos_diff作为位置排除约束的最小变化量
            min_pair_dist_diff=min_pair_dist_diff,  # 传递chiplet对之间距离差异的最小阈值
            M=M,
        )


def search_multiple_solutions(
    num_solutions: int = 3,
    min_shared_length: float = 0.5,
    input_json_path: Optional[str] = None,  # 可选：从JSON文件加载输入
    grid_size: Optional[float] = None,  # 网格大小，如果提供则使用网格化布局
    fixed_chiplet_idx: Optional[int] = None,  # 固定位置的chiplet索引
    exclude_min_diff: Optional[float] = None,  # 排除解约束的最小差异阈值，如果为None则使用grid_size（已废弃，使用min_pos_diff）
    min_pos_diff: Optional[float] = None,  # 位置排除约束的最小变化量，如果为None则使用grid_size或exclude_min_diff
    min_pair_dist_diff: Optional[float] = None,  # chiplet对之间距离差异的最小阈值（用户可配置变量）
    # 如果为None，则使用min_pos_diff；如果min_pos_diff也为None，则使用grid_size或默认值0.01
    # 此参数控制距离排除约束：至少有一对chiplet的距离差必须 >= min_pair_dist_diff
) -> List[ILPPlacementResult]:
    """
    演示：在同一个 problem 上反复添加排除解约束，搜索多个不同的可行解（子调度）。
    
    参数:
        num_solutions: 要搜索的解的数量
        min_shared_length: 相邻chiplet之间共享边的最小长度
        input_json_path: 可选的JSON输入文件路径，如果提供则从文件加载，否则使用随机生成的图
        grid_size: 网格大小，如果提供则使用网格化布局
        fixed_chiplet_idx: 固定位置的chiplet索引
        exclude_min_diff: 排除解约束的最小差异阈值。如果为None，则使用grid_size（如果存在）或默认值0.01。
                         这个值决定了两个解被认为是"不同"的最小位置差异。
        min_pos_diff: 位置排除约束的最小变化量（用户可配置变量）。
                      如果为None，则使用grid_size或exclude_min_diff。
                      此参数控制位置排除约束：至少有一个chiplet的位置变化必须 >= min_pos_diff。
        min_pair_dist_diff: chiplet对之间距离差异的最小阈值（用户可配置变量）。
                            如果为None，则使用min_pos_diff；如果min_pos_diff也为None，则使用grid_size或默认值0.01。
                            此参数控制距离排除约束：至少有一对chiplet的距离差必须 >= min_pair_dist_diff。
    """
    # 1. 构建初始 graph
    if input_json_path:
        # 从JSON文件加载输入
        from load_test_input import load_test_case
        nodes, edges = load_test_case(input_json_path)
    else:
        # 使用随机生成的图
        nodes, edges = build_random_chiplet_graph(edge_prob=0.2, max_nodes=8, fixed_num_edges=4)

    # 2. 首次建模（不求解）
    if grid_size is not None:
        # 使用网格化布局
        ctx = build_placement_ilp_model_grid(
            nodes=nodes,
            edges=edges,
            grid_size=grid_size,
            W=None,
            H=None,
            verbose=False,
            min_shared_length=min_shared_length,
            minimize_bbox_area=True,
            distance_weight=1.0,
            area_weight=0.1,
            fixed_chiplet_idx=fixed_chiplet_idx,
        )
        print(f"使用网格化布局，grid_size={grid_size}")
    else:
        # 使用连续布局
        ctx = build_placement_ilp_model(
            nodes=nodes,
            edges=edges,
            W=None,
            H=None,
            verbose=False,
            min_shared_length=min_shared_length,
            minimize_bbox_area=True,
            distance_weight=1.0,
            area_weight=0.1,
        )

    results: List[ILPPlacementResult] = []
    
    # 维护所有之前解的位置列表
    all_prev_positions: List[Dict[int, Tuple[float, float]]] = []
    all_prev_pair_distances: List[Dict[Tuple[int, int], float]] = []  # 保存上一解的所有chiplet对距离
    
    # 全局约束计数器，确保每个约束名称唯一（使用列表以便在函数内部修改）
    constraint_counter = [0]

    # 输出目录：placement/thermal-placement/output
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(num_solutions):
        # 3. 在当前模型上求解
        # 打印当前模型的约束数量
        constraint_count = len(ctx.prob.constraints)
        print(f"\n[约束调试] 求解解 {i+1} 前的约束数量: {constraint_count}")
        
        res = solve_placement_ilp_from_model(ctx, time_limit=3000, verbose=False)
        if res.status != "Optimal":
            print(f"解 {i+1}: 无最优解（状态={res.status}），搜索结束。")
            print(f"[约束调试] 求解失败时的约束数量: {len(ctx.prob.constraints)}")
            break

        print(f"\n解 {i+1}:")
        print(f"[约束调试] 求解成功后的约束数量: {len(ctx.prob.constraints)}")
        results.append(res)

        # 输出chiplet坐标（左下角坐标）
        print("\nChiplet坐标（左下角坐标）：")
        current_positions_dict = {}
        for idx, node in enumerate(ctx.nodes):
            x_val = pulp.value(ctx.x[idx])
            y_val = pulp.value(ctx.y[idx])
            if x_val is not None and y_val is not None:
                x_val = float(x_val)
                y_val = float(y_val)
                current_positions_dict[idx] = (x_val, y_val)
                print(f"  {node.name:3s}: ({x_val:8.2f}, {y_val:8.2f})")
        
        # 计算并输出chiplet对之间的距离
        print("\nChiplet对之间的距离：")
        current_pair_distances = {}
        if ctx.cx is not None and ctx.cy is not None:
            for i in range(len(ctx.nodes)):
                for j in range(i + 1, len(ctx.nodes)):
                    cx_i = pulp.value(ctx.cx[i])
                    cy_i = pulp.value(ctx.cy[i])
                    cx_j = pulp.value(ctx.cx[j])
                    cy_j = pulp.value(ctx.cy[j])
                    if cx_i is not None and cy_i is not None and cx_j is not None and cy_j is not None:
                        dist = abs(float(cx_i) - float(cx_j)) + abs(float(cy_i) - float(cy_j))
                        current_pair_distances[(i, j)] = dist
                        node_i_name = ctx.nodes[i].name
                        node_j_name = ctx.nodes[j].name
                        print(f"  ({node_i_name}, {node_j_name}): {dist:8.2f}")
        
        # 输出与上一解的变化
        if i > 0 and len(all_prev_positions) > 0:
            print("\n与上一解的变化：")
            # 检查位置变化
            prev_positions = all_prev_positions[-1]
            changed_positions = []
            for idx, node in enumerate(ctx.nodes):
                if idx in current_positions_dict and idx in prev_positions:
                    curr_pos = current_positions_dict[idx]
                    prev_pos = prev_positions[idx]
                    if abs(curr_pos[0] - prev_pos[0]) > 0.01 or abs(curr_pos[1] - prev_pos[1]) > 0.01:
                        changed_positions.append(node.name)
            
            if changed_positions:
                print(f"  位置变化的chiplet: {', '.join(changed_positions)}")
            else:
                print(f"  位置变化的chiplet: 无")
            
            # 检查距离变化
            changed_pairs = []
            if len(all_prev_pair_distances) > 0:
                prev_pair_distances = all_prev_pair_distances[-1]
                # 比较当前解和上一解的距离
                for (i_idx, j_idx), curr_dist in current_pair_distances.items():
                    if (i_idx, j_idx) in prev_pair_distances:
                        prev_dist = prev_pair_distances[(i_idx, j_idx)]
                        if abs(curr_dist - prev_dist) > 0.01:
                            node_i_name = ctx.nodes[i_idx].name
                            node_j_name = ctx.nodes[j_idx].name
                            changed_pairs.append(f"({node_i_name}, {node_j_name})")
            
            if changed_pairs:
                print(f"  距离变化的chiplet对: {', '.join(changed_pairs)}")
            else:
                print(f"  距离变化的chiplet对: 无")
        
        print()

        # 4. 为当前解生成并保存布局图
        nodes_for_draw = []
        for node in ctx.nodes:
            node_copy = deepcopy(node)
            if res.rotations.get(node.name, False):
                # 根据旋转结果调整宽高
                orig_w = node.dimensions.get("x", 0.0)
                orig_h = node.dimensions.get("y", 0.0)
                node_copy.dimensions["x"] = orig_h
                node_copy.dimensions["y"] = orig_w

                # 同步旋转 phys 坐标： (px, py) -> (h - py, px)
                if node_copy.phys:
                    rotated_phys = []
                    for p in node.phys:
                        px = float(p.get("x", 0.0))
                        py = float(p.get("y", 0.0))
                        new_px = orig_h - py
                        new_py = px
                        rotated_phys.append({"x": new_px, "y": new_py})
                    node_copy.phys = rotated_phys
            nodes_for_draw.append(node_copy)

        img_path = output_dir / f"ilp_solution_{i+1}.png"
        
        # 确定固定chiplet的名称集合
        fixed_chiplet_names = None
        if ctx.fixed_chiplet_idx is not None and ctx.fixed_chiplet_idx < len(ctx.nodes):
            fixed_chiplet_name = ctx.nodes[ctx.fixed_chiplet_idx].name
            fixed_chiplet_names = {fixed_chiplet_name}
        
        draw_chiplet_diagram(
            nodes_for_draw,
            ctx.edges,
            save_path=str(img_path),
            layout=res.layout,
            fixed_chiplet_names=fixed_chiplet_names,
        )

        # 5. 保存当前解的位置和chiplet对距离
        current_positions: Dict[int, Tuple[float, float]] = {}
        
        for idx in range(len(ctx.nodes)):
            x_val = pulp.value(ctx.x[idx])
            y_val = pulp.value(ctx.y[idx])
            if x_val is not None and y_val is not None:
                current_positions[idx] = (float(x_val), float(y_val))
        
        # 6. 基于所有之前的解添加排除约束
        #    新解必须与所有之前的解都不同（至少有一个chiplet的位置不同，且至少有一个chiplet的距离不同）
        # 确定位置排除约束的最小变化量
        actual_min_pos_diff = min_pos_diff
        if actual_min_pos_diff is None:
            if exclude_min_diff is not None:
                actual_min_pos_diff = exclude_min_diff
            elif grid_size is not None:
                actual_min_pos_diff = grid_size
            else:
                actual_min_pos_diff = 0.01
        
        add_exclude_layout_constraint(
            ctx, 
            require_change_pairs=1, 
            solution_index=i,
            min_pos_diff=actual_min_pos_diff,  # 使用min_pos_diff参数
            min_pair_dist_diff=min_pair_dist_diff,  # 传递chiplet对之间距离差异的最小阈值
            prev_positions=all_prev_positions.copy(),
            constraint_counter=constraint_counter,  # 传递约束计数器（列表引用）
        )
        
        # 7. 将当前解添加到之前解的列表中
        all_prev_positions.append(current_positions.copy())
        # 保存当前解的所有chiplet对距离（在添加排除约束之后，这样下一解可以比较）
        all_prev_pair_distances.append(current_pair_distances.copy())

    return results


if __name__ == "__main__":
    import sys
    
    # 如果提供了命令行参数，使用JSON文件作为输入
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
        print(f"使用输入文件: {json_path}")
        # 使用网格化布局，grid_size=1.0（chiplet位置只能是整数坐标点）
        sols = search_multiple_solutions(
            num_solutions=10, 
            min_shared_length=0.5,
            input_json_path=json_path,
            grid_size=1.0,  # 网格大小为1.0，chiplet位置只能是整数坐标点
            fixed_chiplet_idx=0,  # 固定第一个chiplet的中心位置
        )
    else:
        # 默认使用随机生成的图
        print("使用随机生成的图")
        sols = search_multiple_solutions(
            num_solutions=10, 
            min_shared_length=0.5,
            grid_size=1.0,  # 网格大小为1.0
            fixed_chiplet_idx=0,
        )
    
    print(f"共找到 {len(sols)} 个不同的 ILP 可行解。")