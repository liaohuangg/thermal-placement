from __future__ import annotations

from typing import Dict, Tuple, List
from pathlib import Path
from copy import deepcopy

import pulp

from tool import build_random_chiplet_graph, draw_chiplet_diagram
from ilp_method import (
    ILPModelContext,
    ILPPlacementResult,
    build_placement_ilp_model,
    solve_placement_ilp_from_model,
)


def add_exclude_layout_constraint(
    ctx: ILPModelContext,
    *,
    require_change_pairs: int = 1,  # 目前没有用到这个参数，先保留接口
) -> None:
    """
    在已有 ILP 模型上添加“排除特定相对位置”的约束。

    对每一对有连边的 (i, j)，读取当前解中的相邻方式和方向：
      - z1 = 1: 水平相邻
        z1L = 1: i 在 j 左边
        z1R = 1: i 在 j 右边
      - z2 = 1: 垂直相邻
        z2D = 1: i 在 j 下边
        z2U = 1: i 在 j 上边

    然后禁止“上一解使用过的 (方式 + 方向) 组合”，例如：
      - 若上一解为“水平 & i 在 j 左边”，则加入约束：z1 + z1L <= 1
      - 若上一解为“垂直 & i 在 j 上边”，则加入约束：z2 + z2U <= 1
    """

    prob = ctx.prob
    z1 = ctx.z1
    z1L = ctx.z1L
    z1R = ctx.z1R
    z2 = ctx.z2
    z2D = ctx.z2D
    z2U = ctx.z2U
    pair_list = ctx.connected_pairs

    print(f"[DEBUG] add_exclude_layout_constraint: 共有 {len(pair_list)} 对连接的 chiplet")
    
    if not pair_list:
        print("[DEBUG] 警告：没有连接的 chiplet 对，无法添加排除约束")
        return

    for (i, j) in pair_list:
        z1_val = pulp.value(z1[(i, j)])
        z2_val = pulp.value(z2[(i, j)])
        z1L_val = pulp.value(z1L[(i, j)])
        z1R_val = pulp.value(z1R[(i, j)])
        z2D_val = pulp.value(z2D[(i, j)])
        z2U_val = pulp.value(z2U[(i, j)])
        
        print(f"pair ({i}, {j}): z1={z1_val}, z2={z2_val}, z1L={z1L_val}, z1R={z1R_val}, z2D={z2D_val}, z2U={z2U_val}")
        
        # 还没解出来 / 上一次求解失败时，直接跳过
        if z1_val is None or z2_val is None:
            print(f"  [跳过] pair ({i}, {j}) 的变量值未求解")
            continue

        # ---- 排除“水平 + 左/右”的组合 ----
        if z1_val >= 0.5:
            if z1L_val is not None and z1L_val >= 0.5:
                # 上一解：水平相邻 & i 在 j 左边
                # 禁止：z1 = 1 且 z1L = 1
                prob += z1[(i, j)] + z1L[(i, j)] <= 1, f"exclude_h_left_{i}_{j}"
            elif z1R_val is not None and z1R_val >= 0.5:
                # 上一解：水平相邻 & i 在 j 右边
                prob += z1[(i, j)] + z1R[(i, j)] <= 1, f"exclude_h_right_{i}_{j}"

        # ---- 排除“垂直 + 上/下”的组合 ----
        if z2_val >= 0.5:
            if z2D_val is not None and z2D_val >= 0.5:
                # 上一解：垂直相邻 & i 在 j 下边
                prob += z2[(i, j)] + z2D[(i, j)] <= 1, f"exclude_v_down_{i}_{j}"
            elif z2U_val is not None and z2U_val >= 0.5:
                # 上一解：垂直相邻 & i 在 j 上边
                prob += z2[(i, j)] + z2U[(i, j)] <= 1, f"exclude_v_up_{i}_{j}"


def search_multiple_solutions(
    num_solutions: int = 3,
    min_shared_length: float = 0.5,
) -> List[ILPPlacementResult]:
    """
    演示：在同一个 problem 上反复添加排除解约束，搜索多个不同的可行解（子调度）。
    """
    # 1. 构建初始 graph
    nodes, edges = build_random_chiplet_graph(edge_prob=0.2, max_nodes=8, fixed_num_edges=4)

    # 2. 首次建模（不求解）
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

    # 输出目录：placement/thermal-placement/output
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(num_solutions):
        # 3. 在当前模型上求解
        res = solve_placement_ilp_from_model(ctx, time_limit=300, verbose=False)
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

        # 5. 基于本次解添加“排除该解”的约束，然后继续下一轮搜索
        #    这里只要求：至少 1 对有连边的 chiplet 改变相邻方式
        add_exclude_layout_constraint(ctx, require_change_pairs=1)

    return results


if __name__ == "__main__":
    sols = search_multiple_solutions(num_solutions=10, min_shared_length=0.5)
    print(f"共找到 {len(sols)} 个不同的 ILP 可行解。")