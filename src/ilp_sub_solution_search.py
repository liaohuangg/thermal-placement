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
) -> None:
    """
    在已有 ILP 模型上添加排除解的约束。
    
    排除整个解，只要下一个解的chiplet集合中有chiplet的位置和上一个解不同即可。
    """

    prob = ctx.prob
    x = ctx.x
    y = ctx.y
    n = len(ctx.nodes)
    W = ctx.W
    H = ctx.H
    
    print(f"[DEBUG] add_exclude_layout_constraint: 共有 {n} 个 chiplet")
    
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
    
    # 为每个chiplet创建二进制变量，表示该chiplet的位置是否与上一解不同
    diff_pos = {}
    for k in range(n):
        diff_pos[k] = pulp.LpVariable(f"diff_pos_{solution_index}_{k}", cat='Binary')
    
    # 小常数，用于判断"不同"（避免浮点数精度问题）
    epsilon = 0.01
    M = max(W, H) * 2
    
    # 对于每个chiplet k，如果 diff_pos[k] = 0，则位置与上一解相同（在epsilon范围内）
    for k in range(n):
        prob += x[k] - x_prev[k] <= epsilon + M * diff_pos[k], f"exclude_x_upper_{solution_index}_{k}"
        prob += x[k] - x_prev[k] >= -epsilon - M * diff_pos[k], f"exclude_x_lower_{solution_index}_{k}"
        prob += y[k] - y_prev[k] <= epsilon + M * diff_pos[k], f"exclude_y_upper_{solution_index}_{k}"
        prob += y[k] - y_prev[k] >= -epsilon - M * diff_pos[k], f"exclude_y_lower_{solution_index}_{k}"
    
    # 排除整个解：至少有一个chiplet的位置不同
    prob += pulp.lpSum([diff_pos[k] for k in range(n)]) >= 1, f"exclude_solution_{solution_index}"
    
    print(f"[DEBUG] 已添加排除整个解的约束（排除解 {solution_index}）")
    print(f"  上一解的位置: {[(x_prev[k], y_prev[k]) for k in range(n)]}")


def search_multiple_solutions(
    num_solutions: int = 3,
    min_shared_length: float = 0.5,
    input_json_path: Optional[str] = None,  # 可选：从JSON文件加载输入
    grid_size: Optional[float] = None,  # 网格大小，如果提供则使用网格化布局
    fixed_chiplet_idx: Optional[int] = None,  # 固定位置的chiplet索引
) -> List[ILPPlacementResult]:
    """
    演示：在同一个 problem 上反复添加排除解约束，搜索多个不同的可行解（子调度）。
    
    参数:
        num_solutions: 要搜索的解的数量
        min_shared_length: 相邻chiplet之间共享边的最小长度
        input_json_path: 可选的JSON输入文件路径，如果提供则从文件加载，否则使用随机生成的图
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
        draw_chiplet_diagram(
            nodes_for_draw,
            ctx.edges,
            save_path=str(img_path),
            layout=res.layout,
        )
        print(f"  解 {i+1} 的布局图已保存到: {img_path}")

        # 5. 基于本次解添加"排除该解"的约束，然后继续下一轮搜索
        #    只要下一个解的chiplet集合中有chiplet的位置和上一个解不同即可
        add_exclude_layout_constraint(
            ctx, 
            require_change_pairs=1, 
            solution_index=i
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