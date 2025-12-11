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
    dist_prev: Optional[Dict[int, float]],
    *,
    solution_index_suffix: str,
    min_diff: float,
    min_dist_diff: float,
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
        prob += pulp.lpSum([diff_pos[k] for k in non_fixed_indices]) >= 1, f"exclude_solution_pos_{solution_index_suffix}"
        print(f"[DEBUG] 排除约束1（解 {solution_index_suffix}）：要求至少 {len(non_fixed_indices)} 个非固定chiplet中有1个位置不同")
    
    # 距离排除约束（如果存在固定chiplet）
    # 逻辑：如果某个chiplet的位置发生改变（diff_pos[k] = 1），那么该chiplet到固定chiplet的距离也必须发生改变
    if fixed_chiplet_idx is not None and cx is not None and cy is not None:
        cx_fixed = W / 2.0
        cy_fixed = H / 2.0
        
        # 为每个非固定chiplet创建二进制变量，表示其到固定chiplet的距离是否不同
        diff_dist = {}
        for k in range(n):
            if k != fixed_chiplet_idx:
                diff_dist[k] = pulp.LpVariable(f"diff_dist_{solution_index_suffix}_{k}", cat='Binary')
        
        # 对于每个非固定chiplet k，如果位置改变，则距离必须改变
        for k in range(n):
            if k == fixed_chiplet_idx:
                continue
            
            # 计算当前解中chiplet k的中心到固定chiplet中心的距离
            # 创建辅助变量表示距离的绝对值
            dx_abs = pulp.LpVariable(f"dx_abs_dist_{solution_index_suffix}_{k}", lowBound=0, upBound=W)
            dy_abs = pulp.LpVariable(f"dy_abs_dist_{solution_index_suffix}_{k}", lowBound=0, upBound=H)
            
            prob += dx_abs >= cx[k] - cx_fixed, f"dx_abs_dist_pos_{solution_index_suffix}_{k}"
            prob += dx_abs >= cx_fixed - cx[k], f"dx_abs_dist_neg_{solution_index_suffix}_{k}"
            prob += dy_abs >= cy[k] - cy_fixed, f"dy_abs_dist_pos_{solution_index_suffix}_{k}"
            prob += dy_abs >= cy_fixed - cy[k], f"dy_abs_dist_neg_{solution_index_suffix}_{k}"
            
            # 当前距离 = dx_abs + dy_abs
            dist_curr = dx_abs + dy_abs
            
            # 计算上一解中chiplet k到固定chiplet的距离
            # 从之前解的位置计算距离
            if k in x_prev and k in y_prev:
                # 计算上一解中chiplet k的中心坐标
                # 需要知道chiplet的宽度和高度
                node_k = ctx.nodes[k]
                # ChipletNode使用dimensions字典存储宽度和高度
                w_k = float(node_k.dimensions.get("x", 0.0))
                h_k = float(node_k.dimensions.get("y", 0.0))
                cx_k_prev = x_prev[k] + w_k / 2.0
                cy_k_prev = y_prev[k] + h_k / 2.0
                dist_prev_k = abs(cx_fixed - cx_k_prev) + abs(cy_fixed - cy_k_prev)
            else:
                continue  # 跳过无法计算距离的chiplet
            
            # 创建辅助变量表示距离差
            max_dist_diff = W + H
            dist_diff = pulp.LpVariable(f"dist_diff_{solution_index_suffix}_{k}", lowBound=-max_dist_diff, upBound=max_dist_diff)
            prob += dist_diff == dist_curr - dist_prev_k, f"dist_diff_def_{solution_index_suffix}_{k}"
            
            dist_diff_abs = pulp.LpVariable(f"dist_diff_abs_{solution_index_suffix}_{k}", lowBound=0, upBound=max_dist_diff)
            prob += dist_diff_abs >= dist_diff, f"dist_diff_abs_pos_{solution_index_suffix}_{k}"
            prob += dist_diff_abs >= -dist_diff, f"dist_diff_abs_neg_{solution_index_suffix}_{k}"
            
            # 约束：如果 diff_pos[k] = 1（位置改变），则 diff_dist[k] = 1（距离必须改变）
            # 使用Big M方法：diff_dist[k] >= diff_pos[k]
            prob += diff_dist[k] >= diff_pos[k], f"dist_change_if_pos_change_{solution_index_suffix}_{k}"
            
            # 约束：如果 diff_dist[k] = 1，则 dist_diff_abs >= min_dist_diff
            prob += dist_diff_abs <= min_dist_diff - 0.001 + M * diff_dist[k], f"dist_diff_abs_upper_{solution_index_suffix}_{k}"
            prob += dist_diff_abs >= min_dist_diff - M * (1 - diff_dist[k]), f"dist_diff_abs_lower_{solution_index_suffix}_{k}"
        
        print(f"[DEBUG] 排除约束2（解 {solution_index_suffix}）：如果chiplet位置改变，则其到固定chiplet的距离必须改变")


def add_exclude_layout_constraint(
    ctx: ILPModelContext,
    *,
    require_change_pairs: int = 1,  # 目前没有用到这个参数，先保留接口
    solution_index: int = 0,  # 解的索引，用于生成唯一的约束名称
    min_diff: Optional[float] = None,  # 判断"不同"的最小差异阈值，如果为None则使用grid_size（已废弃，使用min_pos_diff）
    min_pos_diff: Optional[float] = None,  # 位置排除约束的最小变化量，如果为None则使用min_diff或grid_size
    min_dist_diff: Optional[float] = None,  # 距离差异的最小阈值，如果为None则使用min_pos_diff
    prev_positions: Optional[List[Dict[int, Tuple[float, float]]]] = None,  # 之前所有解的位置列表
    prev_distances: Optional[List[Dict[int, float]]] = None,  # 之前所有解的距离列表
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
    
    print(f"[DEBUG] add_exclude_layout_constraint: 共有 {n} 个 chiplet")
    if fixed_chiplet_idx is not None:
        print(f"[DEBUG] 固定chiplet索引: {fixed_chiplet_idx}，将跳过该chiplet的排除约束")
    
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
                print(f"  [警告] chiplet {k} 的位置未求解，跳过")
                continue
            
            x_prev[k] = float(x_val)
            y_prev[k] = float(y_val)
            valid_count += 1
        
        if valid_count < n:
            print(f"[DEBUG] 警告：只有 {valid_count}/{n} 个chiplet的位置可用，无法添加排除约束")
            return
        
        # 计算当前解的距离
        dist_prev = {}
        if fixed_chiplet_idx is not None and ctx.cx is not None and ctx.cy is not None:
            cx_fixed = W / 2.0
            cy_fixed = H / 2.0
            for k in range(n):
                if k == fixed_chiplet_idx:
                    continue
                cx_k_prev = pulp.value(ctx.cx[k])
                cy_k_prev = pulp.value(ctx.cy[k])
                if cx_k_prev is not None and cy_k_prev is not None:
                    dist_prev[k] = abs(cx_fixed - float(cx_k_prev)) + abs(cy_fixed - float(cy_k_prev))
        
        prev_positions = [{k: (x_prev[k], y_prev[k]) for k in x_prev.keys()}]
        prev_distances = [dist_prev] if dist_prev else [{}]
    
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
    print(f"[DEBUG] 位置排除约束的最小变化量 (min_pos_diff): {min_pos_diff}")
    
    # 获取距离差异的最小阈值
    if min_dist_diff is None:
        min_dist_diff = min_pos_diff
    print(f"[DEBUG] 距离差异的最小阈值 (min_dist_diff): {min_dist_diff}")
    
    M = max(W, H) * 2  # Big-M常数
    
    # 对每个之前的解都添加排除约束
    print(f"[DEBUG] 将对 {len(prev_positions)} 个之前的解添加排除约束")
    for prev_idx, prev_pos_dict in enumerate(prev_positions):
        # 将位置字典转换为x和y字典
        x_prev = {k: pos[0] for k, pos in prev_pos_dict.items()}
        y_prev = {k: pos[1] for k, pos in prev_pos_dict.items()}
        
        # 获取对应的距离
        dist_prev = prev_distances[prev_idx] if prev_idx < len(prev_distances) else {}
        
        solution_index_suffix = f"{solution_index}_prev{prev_idx}" if prev_idx > 0 else str(solution_index)
        print(f"[DEBUG] 排除解 {prev_idx+1}（共 {len(prev_positions)} 个之前的解）")
        print(f"[DEBUG]   位置: {list(prev_pos_dict.items())[:3]}...")  # 只显示前3个
        if dist_prev:
            print(f"[DEBUG]   距离: {list(dist_prev.items())[:3]}...")  # 只显示前3个
        
        # 调用辅助函数添加排除约束
        _add_exclude_single_solution_constraint(
            ctx=ctx,
            x_prev=x_prev,
            y_prev=y_prev,
            dist_prev=dist_prev if dist_prev else None,
            solution_index_suffix=solution_index_suffix,
            min_diff=min_pos_diff,  # 使用min_pos_diff作为位置排除约束的最小变化量
            min_dist_diff=min_dist_diff,
            M=M,
        )
    
    print(f"[DEBUG] 已添加排除所有之前解的约束（共 {len(prev_positions)} 个解）")


def search_multiple_solutions(
    num_solutions: int = 3,
    min_shared_length: float = 0.5,
    input_json_path: Optional[str] = None,  # 可选：从JSON文件加载输入
    grid_size: Optional[float] = None,  # 网格大小，如果提供则使用网格化布局
    fixed_chiplet_idx: Optional[int] = None,  # 固定位置的chiplet索引
    exclude_min_diff: Optional[float] = None,  # 排除解约束的最小差异阈值，如果为None则使用grid_size（已废弃，使用min_pos_diff）
    min_pos_diff: Optional[float] = None,  # 位置排除约束的最小变化量，如果为None则使用grid_size或exclude_min_diff
    min_dist_diff: Optional[float] = None,  # 距离差异的最小阈值，如果为None则使用min_pos_diff
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
    """
    # 1. 构建初始 graph
    if input_json_path:
        # 从JSON文件加载输入
        from load_test_input import load_test_case
        nodes, edges = load_test_case(input_json_path)
        print(f"从文件加载输入: {input_json_path}")
        print(f"  节点数: {len(nodes)}, 边数: {len(edges)}")
    else:
        # 使用随机生成的图
        nodes, edges = build_random_chiplet_graph(edge_prob=0.2, max_nodes=8, fixed_num_edges=4)
        print("使用随机生成的图")

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
        print("使用连续布局（非网格化）")

    results: List[ILPPlacementResult] = []
    
    # 维护所有之前解的位置和距离列表
    all_prev_positions: List[Dict[int, Tuple[float, float]]] = []
    all_prev_distances: List[Dict[int, float]] = []

    # 输出目录：placement/thermal-placement/output
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(num_solutions):
        # 3. 在当前模型上求解
        res = solve_placement_ilp_from_model(ctx, time_limit=3000, verbose=False)
        if res.status != "Optimal":
            print(f"第 {i+1} 个解：无最优解（状态={res.status}），搜索结束。")
            break

        print(f"第 {i+1} 个解：目标值={res.objective_value:.2f}")
        results.append(res)

        # 输出chiplet坐标和相对位置距离
        print(f"\n{'='*60}")
        print(f"解 {i+1} 的详细信息")
        print(f"{'='*60}")
        
        # 输出左下角坐标
        print("\nChiplet坐标（左下角坐标）：")
        for idx, node in enumerate(ctx.nodes):
            x_val = pulp.value(ctx.x[idx])
            y_val = pulp.value(ctx.y[idx])
            if x_val is not None and y_val is not None:
                print(f"  {node.name:3s}: ({x_val:8.2f}, {y_val:8.2f})")
        
        # 输出中心坐标
        if ctx.cx is not None and ctx.cy is not None:
            print("\nChiplet中心坐标：")
            for idx, node in enumerate(ctx.nodes):
                cx_val = pulp.value(ctx.cx[idx])
                cy_val = pulp.value(ctx.cy[idx])
                if cx_val is not None and cy_val is not None:
                    print(f"  {node.name:3s}: ({cx_val:8.2f}, {cy_val:8.2f})")
        
        # 输出相对位置距离（固定chiplet中心到其他chiplet的距离）
        if ctx.fixed_chiplet_idx is not None and ctx.cx is not None and ctx.cy is not None:
            fixed_idx = ctx.fixed_chiplet_idx
            fixed_name = ctx.nodes[fixed_idx].name
            
            # 固定chiplet的中心坐标
            cx_fixed = ctx.W / 2.0
            cy_fixed = ctx.H / 2.0
            
            print(f"\n相对位置距离（固定chiplet {fixed_name} 的中心 ({cx_fixed:.2f}, {cy_fixed:.2f}) 到其他chiplet的曼哈顿距离）：")
            for idx, node in enumerate(ctx.nodes):
                if idx != fixed_idx:
                    cx_val = pulp.value(ctx.cx[idx])
                    cy_val = pulp.value(ctx.cy[idx])
                    if cx_val is not None and cy_val is not None:
                        # 计算曼哈顿距离
                        dist = abs(cx_fixed - float(cx_val)) + abs(cy_fixed - float(cy_val))
                        print(f"  {node.name:3s}: {dist:8.2f}")
            
            # 如果是第2个解及以后，比较与上一解的距离
            if i > 0 and hasattr(ctx, '_prev_distances') and ctx._prev_distances:
                print(f"\n距离变化检查（与上一解比较）：")
                all_same = True
                for idx, node in enumerate(ctx.nodes):
                    if idx != fixed_idx:
                        node_name = node.name
                        cx_val = pulp.value(ctx.cx[idx])
                        cy_val = pulp.value(ctx.cy[idx])
                        if cx_val is not None and cy_val is not None and node_name in ctx._prev_distances:
                            curr_dist = abs(cx_fixed - float(cx_val)) + abs(cy_fixed - float(cy_val))
                            prev_dist = ctx._prev_distances[node_name]
                            diff = abs(curr_dist - prev_dist)
                            min_diff_val = ctx.grid_size if ctx.grid_size else 0.01
                            if diff >= min_diff_val:
                                print(f"  {node_name}: {prev_dist:.2f} -> {curr_dist:.2f} (变化 {diff:.2f}) ✓")
                                all_same = False
                            else:
                                print(f"  {node_name}: {prev_dist:.2f} -> {curr_dist:.2f} (变化 {diff:.2f}) ✗ 相同")
                if all_same:
                    print(f"  [警告] 所有chiplet的相对位置距离都与上一解相同！约束可能未生效！")
                # 更新距离字典
                ctx._prev_distances = {}
                for idx, node in enumerate(ctx.nodes):
                    if idx != fixed_idx:
                        cx_val = pulp.value(ctx.cx[idx])
                        cy_val = pulp.value(ctx.cy[idx])
                        if cx_val is not None and cy_val is not None:
                            ctx._prev_distances[node.name] = abs(cx_fixed - float(cx_val)) + abs(cy_fixed - float(cy_val))
            else:
                # 保存第一个解的距离
                ctx._prev_distances = {}
                for idx, node in enumerate(ctx.nodes):
                    if idx != fixed_idx:
                        cx_val = pulp.value(ctx.cx[idx])
                        cy_val = pulp.value(ctx.cy[idx])
                        if cx_val is not None and cy_val is not None:
                            ctx._prev_distances[node.name] = abs(cx_fixed - float(cx_val)) + abs(cy_fixed - float(cy_val))
        
        print(f"{'='*60}\n")

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
        print(f"  解 {i+1} 的布局图已保存到: {img_path}")

        # 5. 保存当前解的位置和距离
        current_positions: Dict[int, Tuple[float, float]] = {}
        current_distances: Dict[int, float] = {}
        
        for idx in range(len(ctx.nodes)):
            x_val = pulp.value(ctx.x[idx])
            y_val = pulp.value(ctx.y[idx])
            if x_val is not None and y_val is not None:
                current_positions[idx] = (float(x_val), float(y_val))
        
        if ctx.fixed_chiplet_idx is not None and ctx.cx is not None and ctx.cy is not None:
            cx_fixed = ctx.W / 2.0
            cy_fixed = ctx.H / 2.0
            for idx in range(len(ctx.nodes)):
                if idx != ctx.fixed_chiplet_idx:
                    cx_val = pulp.value(ctx.cx[idx])
                    cy_val = pulp.value(ctx.cy[idx])
                    if cx_val is not None and cy_val is not None:
                        current_distances[idx] = abs(cx_fixed - float(cx_val)) + abs(cy_fixed - float(cy_val))
        
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
            min_dist_diff=min_dist_diff,
            prev_positions=all_prev_positions.copy(),
            prev_distances=all_prev_distances.copy()
        )
        
        # 7. 将当前解添加到之前解的列表中
        all_prev_positions.append(current_positions.copy())
        all_prev_distances.append(current_distances.copy())

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