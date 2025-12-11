from __future__ import annotations

from typing import Dict, Tuple, List, Optional
from pathlib import Path
from copy import deepcopy

import pulp

from tool import build_random_chiplet_graph, draw_chiplet_diagram, print_constraint_formal
from ilp_method import (
    ILPModelContext,
    ILPPlacementResult,
    build_placement_ilp_model_grid,
    solve_placement_ilp_from_model,
)


def _add_exclude_pos_constraint(
    ctx: ILPModelContext,
    x_prev: Dict[int, float],
    y_prev: Dict[int, float],
    *,
    solution_index_suffix: str,
    min_pos_diff: float,
    M: float,
) -> None:
    """
    添加位置排除约束：至少有一个非固定chiplet的位置与之前解不同。
    
    参数:
        ctx: ILP模型上下文
        x_prev: 之前解中每个chiplet的x坐标
        y_prev: 之前解中每个chiplet的y坐标
        solution_index_suffix: 用于生成唯一约束名称的后缀
        min_pos_diff: 位置变化的最小阈值
        M: Big-M常数
    """
    prob = ctx.prob
    x = ctx.x
    y = ctx.y
    n = len(ctx.nodes)
    fixed_chiplet_idx = ctx.fixed_chiplet_idx
    
    # 为每个chiplet创建二进制变量，表示该chiplet的位置是否与之前解不同
    diff_pos = {}
    for k in range(n):
        diff_pos[k] = pulp.LpVariable(f"diff_pos_{solution_index_suffix}_{k}", cat='Binary')
    
    # 对于每个chiplet k，判断其位置是否与之前解不同
    for k in range(n):
        if fixed_chiplet_idx is not None and k == fixed_chiplet_idx:
            # 固定chiplet的位置不能改变
            constraint_name = f"diff_pos_fixed_{solution_index_suffix}_{k}"
            prob += diff_pos[k] == 0, constraint_name
            print_constraint_formal(prob.constraints[constraint_name])
            continue
        
        # 创建辅助变量判断x坐标是否不同
        x_diff = pulp.LpVariable(f"x_diff_{solution_index_suffix}_{k}", cat='Binary')
        x_diff_plus = pulp.LpVariable(f"x_diff_plus_{solution_index_suffix}_{k}", cat='Binary')
        x_diff_minus = pulp.LpVariable(f"x_diff_minus_{solution_index_suffix}_{k}", cat='Binary')
        
        constraint_name = f"x_diff_plus_upper_{solution_index_suffix}_{k}"
        prob += x[k] - x_prev[k] <= min_pos_diff - 0.001 + M * x_diff_plus, constraint_name
        print_constraint_formal(prob.constraints[constraint_name])
        
        constraint_name = f"x_diff_plus_lower_{solution_index_suffix}_{k}"
        prob += x[k] - x_prev[k] >= min_pos_diff - M * (1 - x_diff_plus), constraint_name
        print_constraint_formal(prob.constraints[constraint_name])
        
        constraint_name = f"x_diff_minus_upper_{solution_index_suffix}_{k}"
        prob += x[k] - x_prev[k] >= -min_pos_diff + 0.001 - M * x_diff_minus, constraint_name
        print_constraint_formal(prob.constraints[constraint_name])
        
        constraint_name = f"x_diff_minus_lower_{solution_index_suffix}_{k}"
        prob += x[k] - x_prev[k] <= -min_pos_diff + M * (1 - x_diff_minus), constraint_name
        print_constraint_formal(prob.constraints[constraint_name])
        
        constraint_name = f"x_diff_from_plus_{solution_index_suffix}_{k}"
        prob += x_diff >= x_diff_plus, constraint_name
        print_constraint_formal(prob.constraints[constraint_name])
        
        constraint_name = f"x_diff_from_minus_{solution_index_suffix}_{k}"
        prob += x_diff >= x_diff_minus, constraint_name
        print_constraint_formal(prob.constraints[constraint_name])
        
        constraint_name = f"x_diff_upper_{solution_index_suffix}_{k}"
        prob += x_diff <= x_diff_plus + x_diff_minus, constraint_name
        print_constraint_formal(prob.constraints[constraint_name])
        
        # 类似的约束对于y坐标
        y_diff = pulp.LpVariable(f"y_diff_{solution_index_suffix}_{k}", cat='Binary')
        y_diff_plus = pulp.LpVariable(f"y_diff_plus_{solution_index_suffix}_{k}", cat='Binary')
        y_diff_minus = pulp.LpVariable(f"y_diff_minus_{solution_index_suffix}_{k}", cat='Binary')
        
        constraint_name = f"y_diff_plus_upper_{solution_index_suffix}_{k}"
        prob += y[k] - y_prev[k] <= min_pos_diff - 0.001 + M * y_diff_plus, constraint_name
        print_constraint_formal(prob.constraints[constraint_name])
        
        constraint_name = f"y_diff_plus_lower_{solution_index_suffix}_{k}"
        prob += y[k] - y_prev[k] >= min_pos_diff - M * (1 - y_diff_plus), constraint_name
        print_constraint_formal(prob.constraints[constraint_name])
        
        constraint_name = f"y_diff_minus_upper_{solution_index_suffix}_{k}"
        prob += y[k] - y_prev[k] >= -min_pos_diff + 0.001 - M * y_diff_minus, constraint_name
        print_constraint_formal(prob.constraints[constraint_name])
        
        constraint_name = f"y_diff_minus_lower_{solution_index_suffix}_{k}"
        prob += y[k] - y_prev[k] <= -min_pos_diff + M * (1 - y_diff_minus), constraint_name
        print_constraint_formal(prob.constraints[constraint_name])
        
        constraint_name = f"y_diff_from_plus_{solution_index_suffix}_{k}"
        prob += y_diff >= y_diff_plus, constraint_name
        print_constraint_formal(prob.constraints[constraint_name])
        
        constraint_name = f"y_diff_from_minus_{solution_index_suffix}_{k}"
        prob += y_diff >= y_diff_minus, constraint_name
        print_constraint_formal(prob.constraints[constraint_name])
        
        constraint_name = f"y_diff_upper_{solution_index_suffix}_{k}"
        prob += y_diff <= y_diff_plus + y_diff_minus, constraint_name
        print_constraint_formal(prob.constraints[constraint_name])
        
        constraint_name = f"diff_pos_from_x_{solution_index_suffix}_{k}"
        prob += diff_pos[k] >= x_diff, constraint_name
        print_constraint_formal(prob.constraints[constraint_name])
        
        constraint_name = f"diff_pos_from_y_{solution_index_suffix}_{k}"
        prob += diff_pos[k] >= y_diff, constraint_name
        print_constraint_formal(prob.constraints[constraint_name])
        
        constraint_name = f"diff_pos_upper_{solution_index_suffix}_{k}"
        prob += diff_pos[k] <= x_diff + y_diff, constraint_name
        print_constraint_formal(prob.constraints[constraint_name])
    
    # 位置排除约束：至少有一个非固定chiplet的位置不同
    non_fixed_indices = [k for k in range(n) if fixed_chiplet_idx is None or k != fixed_chiplet_idx]
    if len(non_fixed_indices) > 0:
        constraint_name = f"exclude_solution_pos_{solution_index_suffix}"
        prob += pulp.lpSum([diff_pos[k] for k in non_fixed_indices]) >= 1, constraint_name
        print_constraint_formal(prob.constraints[constraint_name])


def _add_exclude_dist_constraint(
    ctx: ILPModelContext,
    x_prev: Dict[int, float],
    y_prev: Dict[int, float],
    *,
    solution_index_suffix: str,
    min_pair_dist_diff: float,
    M: float,
) -> None:
    """
    添加距离排除约束：至少有一对chiplet的距离与之前解不同。
    
    参数:
        ctx: ILP模型上下文
        x_prev: 之前解中每个chiplet的x坐标
        y_prev: 之前解中每个chiplet的y坐标
        solution_index_suffix: 用于生成唯一约束名称的后缀
        min_pair_dist_diff: chiplet对之间距离差异的最小阈值
        M: Big-M常数
    """
    prob = ctx.prob
    n = len(ctx.nodes)
    W = ctx.W
    H = ctx.H
    cx = ctx.cx
    cy = ctx.cy
    
    if cx is None or cy is None:
        return
    
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
        
        constraint_name = f"dx_abs_pair_pos_{solution_index_suffix}_{i}_{j}"
        prob += dx_abs_ij >= cx[i] - cx[j], constraint_name
        print_constraint_formal(prob.constraints[constraint_name])
        
        constraint_name = f"dx_abs_pair_neg_{solution_index_suffix}_{i}_{j}"
        prob += dx_abs_ij >= cx[j] - cx[i], constraint_name
        print_constraint_formal(prob.constraints[constraint_name])
        
        constraint_name = f"dy_abs_pair_pos_{solution_index_suffix}_{i}_{j}"
        prob += dy_abs_ij >= cy[i] - cy[j], constraint_name
        print_constraint_formal(prob.constraints[constraint_name])
        
        constraint_name = f"dy_abs_pair_neg_{solution_index_suffix}_{i}_{j}"
        prob += dy_abs_ij >= cy[j] - cy[i], constraint_name
        print_constraint_formal(prob.constraints[constraint_name])
        
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
            
            constraint_name = f"dist_diff_pair_def_{solution_index_suffix}_{i}_{j}"
            prob += dist_diff_ij == dist_curr_ij - dist_prev_ij, constraint_name
            print_constraint_formal(prob.constraints[constraint_name])
            
            dist_diff_abs_ij = pulp.LpVariable(f"dist_diff_abs_pair_{solution_index_suffix}_{i}_{j}", lowBound=0, upBound=max_dist_diff)
            
            constraint_name = f"dist_diff_abs_pair_pos_{solution_index_suffix}_{i}_{j}"
            prob += dist_diff_abs_ij >= dist_diff_ij, constraint_name
            print_constraint_formal(prob.constraints[constraint_name])
            
            constraint_name = f"dist_diff_abs_pair_neg_{solution_index_suffix}_{i}_{j}"
            prob += dist_diff_abs_ij >= -dist_diff_ij, constraint_name
            print_constraint_formal(prob.constraints[constraint_name])
            
            # 为这对chiplet创建二进制变量，表示距离是否不同
            diff_dist_pair[(i, j)] = pulp.LpVariable(f"diff_dist_pair_{solution_index_suffix}_{i}_{j}", cat='Binary')
            
            # 约束逻辑：
            # - 如果 dist_diff_abs_ij >= min_pair_dist_diff，则 diff_dist_pair[(i,j)] 可以是 1
            # - 如果 dist_diff_abs_ij < min_pair_dist_diff，则 diff_dist_pair[(i,j)] 必须是 0
            # 使用 Big M 方法实现：
            epsilon = max(0.001, min_pair_dist_diff * 0.01)
            # 如果 diff_dist_pair[(i,j)] = 1，则 dist_diff_abs_ij >= min_pair_dist_diff
            constraint_name = f"dist_diff_abs_pair_lower_{solution_index_suffix}_{i}_{j}"
            prob += dist_diff_abs_ij >= min_pair_dist_diff - M * (1 - diff_dist_pair[(i, j)]), constraint_name
            print_constraint_formal(prob.constraints[constraint_name])
            # 如果 diff_dist_pair[(i,j)] = 0，则 dist_diff_abs_ij <= min_pair_dist_diff - epsilon
            constraint_name = f"dist_diff_abs_pair_upper_{solution_index_suffix}_{i}_{j}"
            prob += dist_diff_abs_ij <= min_pair_dist_diff - epsilon + M * diff_dist_pair[(i, j)], constraint_name
            print_constraint_formal(prob.constraints[constraint_name])
            
            valid_pairs.append((i, j))
    
    # 顶层约束：至少有一对chiplet的距离不同
    if len(valid_pairs) > 0:
        constraint_name = f"exclude_solution_dist_pair_{solution_index_suffix}"
        prob += pulp.lpSum([diff_dist_pair[pair] for pair in valid_pairs]) >= 1, constraint_name
        print_constraint_formal(prob.constraints[constraint_name])


def add_exclude_constraint(
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
        
        # 调用位置排除约束函数
        _add_exclude_pos_constraint(
            ctx=ctx,
            x_prev=x_prev,
            y_prev=y_prev,
            solution_index_suffix=solution_index_suffix,
            min_pos_diff=min_pos_diff,
            M=M,
        )
        
        # 调用距离排除约束函数
        _add_exclude_dist_constraint(
            ctx=ctx,
            x_prev=x_prev,
            y_prev=y_prev,
            solution_index_suffix=solution_index_suffix,
            min_pair_dist_diff=min_pair_dist_diff,
            M=M,
        )


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
                constraint_counter=constraint_counter,
            )
        
        # 导出LP文件（在求解之前，包含所有约束）
        output_dir = Path("/root/placement/thermal-placement/output")
        output_dir.mkdir(parents=True, exist_ok=True)
        lp_file = output_dir / f"constraints_solution_{i+1}.lp"
        ctx.prob.writeLP(str(lp_file))
        
        # 求解
        result = solve_placement_ilp_from_model(ctx)
        
        # result.status 是字符串类型（如 "Optimal"），需要与字符串比较
        if result.status != "Optimal":
            print(f"\n求解状态: {result.status}，停止搜索")
            break
        
        solutions.append(result)
        
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
        
        # 计算并保存当前解的chiplet对之间的距离
        pair_distances = {}
        for i_idx in range(len(nodes)):
            for j_idx in range(i_idx + 1, len(nodes)):
                node_i = nodes[i_idx]
                node_j = nodes[j_idx]
                w_i = float(node_i.dimensions.get("x", 0.0))
                h_i = float(node_i.dimensions.get("y", 0.0))
                w_j = float(node_j.dimensions.get("x", 0.0))
                h_j = float(node_j.dimensions.get("y", 0.0))
                
                cx_i = x_prev[i_idx] + w_i / 2.0
                cy_i = y_prev[i_idx] + h_i / 2.0
                cx_j = x_prev[j_idx] + w_j / 2.0
                cy_j = y_prev[j_idx] + h_j / 2.0
                
                dist = abs(cx_i - cx_j) + abs(cy_i - cy_j)
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
        
        # 输出与上一解的变化（仅比较与上一个解）
        if i > 0:
            prev_pos_dict = all_prev_positions[i-1]
            prev_pair_distances = all_prev_pair_distances[i-1]
            
            changed_chiplets = []
            for k in range(len(nodes)):
                if k in prev_pos_dict:
                    prev_x, prev_y = prev_pos_dict[k]
                    if abs(x_prev[k] - prev_x) > 0.01 or abs(y_prev[k] - prev_y) > 0.01:
                        node_name = nodes[k].name if hasattr(nodes[k], 'name') else f"Chiplet_{k}"
                        changed_chiplets.append(node_name)
            
            changed_pairs = []
            for (i_idx, j_idx), curr_dist in pair_distances.items():
                if (i_idx, j_idx) in prev_pair_distances:
                    prev_dist = prev_pair_distances[(i_idx, j_idx)]
                    if abs(curr_dist - prev_dist) > 0.01:
                        node_i_name = nodes[i_idx].name if hasattr(nodes[i_idx], 'name') else f"Chiplet_{i_idx}"
                        node_j_name = nodes[j_idx].name if hasattr(nodes[j_idx], 'name') else f"Chiplet_{j_idx}"
                        changed_pairs.append((node_i_name, node_j_name))
            
            print("\n与上一解的变化:")
            if changed_chiplets:
                print(f"  位置变化的chiplet: {', '.join(changed_chiplets)}")
            else:
                print("  位置变化的chiplet: 无")
            if changed_pairs:
                print(f"  距离变化的chiplet对: {', '.join([f'({p[0]},{p[1]})' for p in changed_pairs])}")
            else:
                print("  距离变化的chiplet对: 无")
        
        # 绘制并保存布局图片
        layout_dict = {}
        fixed_chiplet_names = set()
        for k, node in enumerate(nodes):
            node_name = node.name if hasattr(node, 'name') else f"Chiplet_{k}"
            layout_dict[node_name] = (x_prev[k], y_prev[k])
            if fixed_chiplet_idx is not None and k == fixed_chiplet_idx:
                fixed_chiplet_names.add(node_name)
        
        # 保存图片到输出目录
        output_dir = Path("/root/placement/thermal-placement/output")
        output_dir.mkdir(parents=True, exist_ok=True)
        image_path = output_dir / f"solution_{i+1}_layout.png"
        
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
