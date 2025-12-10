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


def add_exclude_layout_constraint(
    ctx: ILPModelContext,
    *,
    require_change_pairs: int = 1,  # 目前没有用到这个参数，先保留接口
    solution_index: int = 0,  # 解的索引，用于生成唯一的约束名称
    min_diff: Optional[float] = None,  # 判断"不同"的最小差异阈值，如果为None则使用grid_size
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
    
    # 读取上一解中每个chiplet的位置
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
    
    # 获取最小差异阈值
    if min_diff is None:
        # 如果未指定，使用grid_size（如果存在）或默认值
        grid_size = ctx.grid_size
        if grid_size is not None:
            min_diff = grid_size
        else:
            min_diff = 0.01  # 默认值
    print(f"[DEBUG] 排除解约束的最小差异阈值: {min_diff}")
    
    M = max(W, H) * 2  # Big-M常数
    
    # 为每个chiplet创建二进制变量，表示该chiplet的位置是否与上一解不同
    diff_pos = {}
    for k in range(n):
        diff_pos[k] = pulp.LpVariable(f"diff_pos_{solution_index}_{k}", cat='Binary')
    
    # 对于每个chiplet k，判断其位置是否与上一解不同（至少相差一个网格宽度）
    # 注意：跳过固定的chiplet
    for k in range(n):
        # 如果是固定的chiplet，跳过，不为其添加约束
        if fixed_chiplet_idx is not None and k == fixed_chiplet_idx:
            # 固定chiplet的diff_pos始终为0（因为位置不变）
            prob += diff_pos[k] == 0, f"diff_pos_fixed_{solution_index}_{k}"
            continue
        
        # 创建辅助变量表示x坐标是否不同（至少相差min_diff）
        x_diff = pulp.LpVariable(f"x_diff_{solution_index}_{k}", cat='Binary')
        # 创建辅助变量表示y坐标是否不同（至少相差min_diff）
        y_diff = pulp.LpVariable(f"y_diff_{solution_index}_{k}", cat='Binary')
        
        # 判断 x 坐标是否不同：|x[k] - x_prev[k]| >= min_diff
        # 使用两个辅助变量：x_diff_plus 和 x_diff_minus
        x_diff_plus = pulp.LpVariable(f"x_diff_plus_{solution_index}_{k}", cat='Binary')  # x >= x_prev + min_diff
        x_diff_minus = pulp.LpVariable(f"x_diff_minus_{solution_index}_{k}", cat='Binary')  # x <= x_prev - min_diff
        
        # 如果 x[k] - x_prev[k] >= min_diff，则 x_diff_plus = 1
        # 约束1: 如果 x_diff_plus = 0，则 x[k] - x_prev[k] < min_diff
        prob += x[k] - x_prev[k] <= min_diff - 0.001 + M * x_diff_plus, f"x_diff_plus_upper_{solution_index}_{k}"
        # 约束2: 如果 x[k] - x_prev[k] >= min_diff，则 x_diff_plus = 1
        prob += x[k] - x_prev[k] >= min_diff - M * (1 - x_diff_plus), f"x_diff_plus_lower_{solution_index}_{k}"
        
        # 如果 x[k] - x_prev[k] <= -min_diff，则 x_diff_minus = 1
        # 约束1: 如果 x_diff_minus = 0，则 x[k] - x_prev[k] > -min_diff
        prob += x[k] - x_prev[k] >= -min_diff + 0.001 - M * x_diff_minus, f"x_diff_minus_upper_{solution_index}_{k}"
        # 约束2: 如果 x[k] - x_prev[k] <= -min_diff，则 x_diff_minus = 1
        prob += x[k] - x_prev[k] <= -min_diff + M * (1 - x_diff_minus), f"x_diff_minus_lower_{solution_index}_{k}"
        
        # x_diff = 1 当且仅当 x_diff_plus = 1 或 x_diff_minus = 1
        prob += x_diff >= x_diff_plus, f"x_diff_from_plus_{solution_index}_{k}"
        prob += x_diff >= x_diff_minus, f"x_diff_from_minus_{solution_index}_{k}"
        prob += x_diff <= x_diff_plus + x_diff_minus, f"x_diff_upper_{solution_index}_{k}"
        
        # 类似的约束对于y坐标
        y_diff_plus = pulp.LpVariable(f"y_diff_plus_{solution_index}_{k}", cat='Binary')
        y_diff_minus = pulp.LpVariable(f"y_diff_minus_{solution_index}_{k}", cat='Binary')
        
        prob += y[k] - y_prev[k] <= min_diff - 0.001 + M * y_diff_plus, f"y_diff_plus_upper_{solution_index}_{k}"
        prob += y[k] - y_prev[k] >= min_diff - M * (1 - y_diff_plus), f"y_diff_plus_lower_{solution_index}_{k}"
        prob += y[k] - y_prev[k] >= -min_diff + 0.001 - M * y_diff_minus, f"y_diff_minus_upper_{solution_index}_{k}"
        prob += y[k] - y_prev[k] <= -min_diff + M * (1 - y_diff_minus), f"y_diff_minus_lower_{solution_index}_{k}"
        
        prob += y_diff >= y_diff_plus, f"y_diff_from_plus_{solution_index}_{k}"
        prob += y_diff >= y_diff_minus, f"y_diff_from_minus_{solution_index}_{k}"
        prob += y_diff <= y_diff_plus + y_diff_minus, f"y_diff_upper_{solution_index}_{k}"
        
        # diff_pos[k] = 1 当且仅当 x_diff = 1 或 y_diff = 1（至少有一个坐标不同至少一个网格宽度）
        prob += diff_pos[k] >= x_diff, f"diff_pos_from_x_{solution_index}_{k}"
        prob += diff_pos[k] >= y_diff, f"diff_pos_from_y_{solution_index}_{k}"
        prob += diff_pos[k] <= x_diff + y_diff, f"diff_pos_upper_{solution_index}_{k}"
    
    # 排除整个解：至少有一个非固定的chiplet的位置不同
    # 只对非固定的chiplet求和
    non_fixed_indices = [k for k in range(n) if fixed_chiplet_idx is None or k != fixed_chiplet_idx]
    if len(non_fixed_indices) > 0:
        prob += pulp.lpSum([diff_pos[k] for k in non_fixed_indices]) >= 1, f"exclude_solution_{solution_index}"
        print(f"[DEBUG] 排除约束：要求至少 {len(non_fixed_indices)} 个非固定chiplet中有1个位置不同")
    else:
        print(f"[DEBUG] 警告：所有chiplet都是固定的，无法添加排除约束")
    
    print(f"[DEBUG] 已添加排除整个解的约束（排除解 {solution_index}）")
    print(f"  上一解的位置: {[(x_prev[k], y_prev[k]) for k in range(n)]}")


def search_multiple_solutions(
    num_solutions: int = 3,
    min_shared_length: float = 0.5,
    input_json_path: Optional[str] = None,  # 可选：从JSON文件加载输入
    grid_size: Optional[float] = None,  # 网格大小，如果提供则使用网格化布局
    fixed_chiplet_idx: Optional[int] = None,  # 固定位置的chiplet索引
    exclude_min_diff: Optional[float] = None,  # 排除解约束的最小差异阈值，如果为None则使用grid_size
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

        # 5. 基于本次解添加"排除该解"的约束，然后继续下一轮搜索
        #    只要下一个解的chiplet集合中有chiplet的位置和上一个解不同即可
        add_exclude_layout_constraint(
            ctx, 
            require_change_pairs=1, 
            solution_index=i,
            min_diff=exclude_min_diff
        )

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