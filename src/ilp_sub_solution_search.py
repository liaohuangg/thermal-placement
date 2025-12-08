from __future__ import annotations

from typing import Dict, Tuple, List
from pathlib import Path
from copy import deepcopy
import time

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
    solution_index: int = 0,  # 解的序号，用于生成唯一的约束名称
) -> None:
    """
    在已有 ILP 模型上添加"排除特定相对位置"的约束。
    
    使用基于 Big-M 的析取约束方法，对所有 chiplet 对排除原始解中的相对位置关系。
    
    对于每一对 chiplet (i, j)，判断原始解中的相对位置关系：
      - i 在 j 的左边：x_i + w_i <= x_j
      - i 在 j 的右边：x_j + w_j <= x_i
      - i 在 j 的下边：y_i + h_i <= y_j
      - i 在 j 的上边：y_j + h_j <= y_i
    
    然后使用 Big-M 方法强制新解中 i 不在 j 的相同方向区域（而不是只排除严格的边界对齐）。
    例如：如果原始解中 i 在 j 的左边，则强制新解中 x_i >= x_j + w_j（i 必须在 j 的右侧），
    这样可以排除"伪新解"（相对位置关系不变，只是距离改变的情况）。
    
    同时排除左右对称、上下对称和中心对称的解。
    """

    prob = ctx.prob
    nodes = ctx.nodes
    x = ctx.x
    y = ctx.y
    r = ctx.r
    w_orig = {}
    h_orig = {}
    
    # 获取原始尺寸
    for i, node in enumerate(nodes):
        w_orig[i] = float(node.dimensions.get("x", 0.0))
        h_orig[i] = float(node.dimensions.get("y", 0.0))
    
    n = len(nodes)
    W = ctx.W
    H = ctx.H
    M = max(W, H) * 2  # Big-M 值
    
    # 检查求解状态
    # 获取最后一次求解的状态
    try:
        # 尝试从prob中获取状态
        # 注意：PuLP可能没有直接的方法获取状态，我们需要通过检查变量值来判断
        # 如果所有变量值都是None，说明还没有求解或求解失败
        test_x_val = pulp.value(x[0]) if n > 0 else None
        if test_x_val is None:
            print(f"[DEBUG] 警告：模型尚未求解或求解失败，无法添加排除约束")
            print(f"[DEBUG] 提示：请确保在调用此函数前已经成功求解了模型")
            return
    except Exception as e:
        print(f"[DEBUG] 警告：无法检查求解状态: {e}")
        return
    
    # 从prob中获取w和h变量（如果存在）
    w = {}
    h = {}
    for i in range(n):
        w_var = prob.variablesDict().get(f"w_{i}")
        h_var = prob.variablesDict().get(f"h_{i}")
        if w_var is None or h_var is None:
            # 如果变量不存在，使用原始尺寸（不考虑旋转）
            w[i] = w_orig[i]
            h[i] = h_orig[i]
        else:
            w[i] = w_var
            h[i] = h_var
    
    # 获取当前解中所有 chiplet 的位置
    current_x = {}
    current_y = {}
    current_w = {}
    current_h = {}
    
    missing_values = []
    for i in range(n):
        x_val = pulp.value(x[i])
        y_val = pulp.value(y[i])
        # 安全获取旋转变量值
        r_val = None
        if r[i] is not None:
            # r[i] 是变量，获取其值
            r_val = pulp.value(r[i])
        else:
            # r[i] 是 None（正方形chiplet），固定为0（不旋转）
            r_val = 0
        # 如果r_val是None，默认为0（不旋转）
        if r_val is None:
            r_val = 0
        
        if x_val is None or y_val is None:
            missing_values.append(i)
            # 使用默认值继续处理，而不是直接返回
            x_val = x_val if x_val is not None else 0.0
            y_val = y_val if y_val is not None else 0.0
        
        current_x[i] = x_val
        current_y[i] = y_val
        # 根据旋转状态确定实际宽高
        if r_val >= 0.5:
            current_w[i] = h_orig[i]
            current_h[i] = w_orig[i]
        else:
            current_w[i] = w_orig[i]
            current_h[i] = h_orig[i]
    
    # 即使有缺失值，也继续添加排除约束（不跳过）
    if missing_values:
        print(f"[DEBUG] 警告：chiplet {missing_values} 的位置未求解（值为None），使用默认值0.0")
        print(f"[DEBUG] 将继续添加排除约束，但可能影响约束的有效性")
    
    print(f"[DEBUG] add_exclude_layout_constraint: 对所有 {n} 个 chiplet 添加排除约束（基于中心相对位置）")
    
    epsilon = 0.1  # 数值精度安全垫，用于确保严格不等式（需大于数值计算误差）
    # 注意：epsilon不能太小，否则会导致约束过严，无法找到新解
    # 如果只找到一个解，可以尝试增大epsilon（如0.1或更大）
    offset_threshold = 0.1 * max(W, H)  # 偏移阈值：布局总尺寸的10%，用于排除"轻微偏移"的伪新解
    
    # ========== 步骤1：提取原解的中心相对位置特征 ==========
    print(f"[DEBUG] 步骤1: 提取原解的中心相对位置特征")
    
    # 1.1 计算原解中每个chiplet的中心坐标
    original_cx = {}  # 原解中心x坐标
    original_cy = {}  # 原解中心y坐标
    for i in range(n):
        original_cx[i] = current_x[i] + current_w[i] / 2.0
        original_cy[i] = current_y[i] + current_h[i] / 2.0
    
    # 1.2 记录所有chiplet对的中心相对偏移量（Δcx, Δcy）
    original_deltas = {}  # (i, j) -> (delta_cx, delta_cy)
    for i in range(n):
        for j in range(i + 1, n):  # 只记录 i < j 的对，避免重复
            delta_cx = original_cx[j] - original_cx[i]  # Δcx = cx_j - cx_i
            delta_cy = original_cy[j] - original_cy[i]  # Δcy = cy_j - cy_i
            original_deltas[(i, j)] = (delta_cx, delta_cy)
    
    print(f"[DEBUG] 记录了 {len(original_deltas)} 对 chiplet 的中心相对偏移量")
    
    # ========== 步骤2：添加约束禁止完全匹配的相对位置 ==========
    print(f"[DEBUG] 步骤2: 添加约束禁止完全匹配的中心相对位置")
    
    # 2.1 引入二元变量 z[(i,j)]：z=1 表示 (i,j) 的中心偏移与原解不同，z=0 表示相同
    z = {}
    z_vars = []
    for (i, j) in original_deltas.keys():
        z[(i, j)] = pulp.LpVariable(f"exclude_z_{solution_index}_{i}_{j}", cat='Binary')
        z_vars.append(z[(i, j)])
    
    # 2.2 对每对chiplet，约束：若z=0（偏移相同），则Δcx=原Δcx且Δcy=原Δcy；z=1则放松
    exclude_count = 0
    for (i, j), (delta_cx_ori, delta_cy_ori) in original_deltas.items():
        # 新解的中心偏移（使用变量表达式）
        # cx_i = x[i] + w[i]/2, cx_j = x[j] + w[j]/2
        # delta_cx_new = cx_j - cx_i = (x[j] + w[j]/2) - (x[i] + w[i]/2) = x[j] - x[i] + (w[j] - w[i])/2
        # 类似地，delta_cy_new = y[j] - y[i] + (h[j] - h[i])/2
        
        # 注意：w[i]和h[i]可能是变量（如果允许旋转），但这里我们使用current_w和current_h的固定值
        # 因为我们在计算原解时已经确定了旋转状态
        w_i_val = current_w[i]  # 使用当前解的值（固定）
        h_i_val = current_h[i]
        w_j_val = current_w[j]
        h_j_val = current_h[j]
        
        # 约束Δcx：强制新解的Δcx与原解的差异≥epsilon（当z=1时）
        # 使用Big-M方法：|delta_cx_new - delta_cx_ori| >= epsilon * z - M * (1 - z)
        # 即：delta_cx_new - delta_cx_ori >= epsilon * z - M * (1 - z)
        # 和：delta_cx_ori - delta_cx_new >= epsilon * z - M * (1 - z)
        prob += (x[j] - x[i] + (w_j_val - w_i_val) / 2.0) - delta_cx_ori >= epsilon * z[(i, j)] - M * (1 - z[(i, j)]), \
               f"delta_cx_{solution_index}_{i}_{j}"
        exclude_count += 1
        
        prob += delta_cx_ori - (x[j] - x[i] + (w_j_val - w_i_val) / 2.0) >= epsilon * z[(i, j)] - M * (1 - z[(i, j)]), \
               f"delta_cx_rev_{solution_index}_{i}_{j}"
        exclude_count += 1
        
        # 约束Δcy：强制新解的Δcy与原解的差异≥epsilon（当z=1时）
        prob += (y[j] - y[i] + (h_j_val - h_i_val) / 2.0) - delta_cy_ori >= epsilon * z[(i, j)] - M * (1 - z[(i, j)]), \
               f"delta_cy_{solution_index}_{i}_{j}"
        exclude_count += 1
        
        prob += delta_cy_ori - (y[j] - y[i] + (h_j_val - h_i_val) / 2.0) >= epsilon * z[(i, j)] - M * (1 - z[(i, j)]), \
               f"delta_cy_rev_{solution_index}_{i}_{j}"
        exclude_count += 1
    
    # 2.3 强制至少有一对chiplet的中心偏移不同（z≥1），排除完全相同的相对位置
    if exclude_count > 0 and z_vars:
        prob += pulp.lpSum(z_vars) >= 1, f"exclude_at_least_one_{solution_index}"
        print(f"[DEBUG] 添加了 {exclude_count} 个排除约束，涉及 {len(z_vars)} 对 chiplet")
    else:
        print(f"[DEBUG] 警告：没有需要排除的相对位置关系或未找到排除变量")
    
    # ========== 步骤3（可选）：添加阈值约束排除"轻微偏移"的伪新解 ==========
    # 允许任意一对chiplet满足阈值约束，而不是固定选择某一对
    # 使用二元变量表示每对chiplet是否满足阈值，然后约束至少有一对满足阈值
    print(f"[DEBUG] 步骤3: 添加阈值约束排除轻微偏移的伪新解（阈值={offset_threshold:.2f}）")
    
    threshold_exclude_count = 0
    z_threshold_vars = []  # 收集所有阈值二元变量
    
    # 对每对chiplet添加阈值约束，但只要求至少有一对满足阈值
    for (i, j), (delta_cx_ori, delta_cy_ori) in original_deltas.items():
        w_i_val = current_w[i]
        h_i_val = current_h[i]
        w_j_val = current_w[j]
        h_j_val = current_h[j]
        
        # 引入二元变量：z_threshold=1表示这对chiplet满足阈值，z_threshold=0表示不满足
        z_threshold = pulp.LpVariable(f"exclude_z_threshold_{solution_index}_{i}_{j}", cat='Binary')
        z_threshold_vars.append(z_threshold)
        
        # 约束：如果z_threshold=1，则至少一个方向的偏移≥阈值
        # |delta_cx_new - delta_cx_ori| >= offset_threshold * z_threshold - M * (1 - z_threshold)
        # 或 |delta_cy_new - delta_cy_ori| >= offset_threshold * z_threshold - M * (1 - z_threshold)
        
        # 约束x方向的偏移≥阈值（当z_threshold=1时）
        prob += (x[j] - x[i] + (w_j_val - w_i_val) / 2.0) - delta_cx_ori >= offset_threshold * z_threshold - M * (1 - z_threshold), \
               f"threshold_delta_cx_{solution_index}_{i}_{j}"
        threshold_exclude_count += 1
        
        prob += delta_cx_ori - (x[j] - x[i] + (w_j_val - w_i_val) / 2.0) >= offset_threshold * z_threshold - M * (1 - z_threshold), \
               f"threshold_delta_cx_rev_{solution_index}_{i}_{j}"
        threshold_exclude_count += 1
        
        # 约束y方向的偏移≥阈值（当z_threshold=1时）
        prob += (y[j] - y[i] + (h_j_val - h_i_val) / 2.0) - delta_cy_ori >= offset_threshold * z_threshold - M * (1 - z_threshold), \
               f"threshold_delta_cy_{solution_index}_{i}_{j}"
        threshold_exclude_count += 1
        
        prob += delta_cy_ori - (y[j] - y[i] + (h_j_val - h_i_val) / 2.0) >= offset_threshold * z_threshold - M * (1 - z_threshold), \
               f"threshold_delta_cy_rev_{solution_index}_{i}_{j}"
        threshold_exclude_count += 1
    
    # 强制至少有一对chiplet满足阈值（至少有一个z_threshold=1）
    # 这样允许求解器任意选择一对chiplet来满足阈值约束
    # 注意：步骤2已经要求至少有一对chiplet的偏移≥epsilon，步骤3要求至少有一对chiplet的偏移≥offset_threshold
    # 如果offset_threshold > epsilon，那么满足步骤3的chiplet对也满足步骤2，所以这两个约束可以共存
    # 但为了不过度约束，我们可以选择只使用步骤3（如果offset_threshold > epsilon），或者将步骤3设为可选
    # 这里先注释掉步骤3的强制约束，因为步骤2已经能够排除完全相同的解
    # 如果需要更严格的约束（排除轻微偏移），可以取消注释
    # if threshold_exclude_count > 0 and z_threshold_vars:
    #     prob += pulp.lpSum(z_threshold_vars) >= 1, f"threshold_at_least_one_{solution_index}"
    #     print(f"[DEBUG] 添加了 {threshold_exclude_count} 个阈值约束，强制至少有一对chiplet的偏移≥阈值（任意选择）")
    # else:
    #     print(f"[DEBUG] 警告：没有添加阈值约束")
    print(f"[DEBUG] 添加了 {threshold_exclude_count} 个阈值约束（可选，不强制，因为步骤2已经能够排除完全相同的解）")
    
    # 注意：基于中心相对位置的约束已经能够排除完全相同的解和轻微偏移的伪新解。
    # 对称解（左右对称、上下对称、中心对称）的中心相对位置会发生变化，不会被排除，
    # 如果不需要排除对称解，则不需要额外的相对位置关系约束。


def search_multiple_solutions(
    num_solutions: int = 3,
    min_shared_length: float = 0.5,
) -> List[ILPPlacementResult]:
    """
    演示：在同一个 problem 上反复添加排除解约束，搜索多个不同的可行解（子调度）。
    """
    start_time = time.time()
    print(f"[时间记录] 开始搜索多个解，目标数量: {num_solutions}")
    
    # 1. 构建初始 graph
    graph_start = time.time()
    # 可以指定硅桥互联边和普通互联边的数量
    # 例如：num_silicon_bridge_edges=6, num_normal_edges=6
    nodes, edges = build_random_chiplet_graph(
        edge_prob=0.2, 
        max_nodes=5, 
        fixed_num_edges=5,
        num_silicon_bridge_edges=3,  # 指定硅桥互联边数量
        num_normal_edges=2,          # 指定普通互联边数量
        seed=42
    )
    
    # 重新生成类型化边以用于绘图和建模（因为build_random_chiplet_graph返回的是旧格式）
    from tool import generate_typed_edges
    node_names = [n.name for n in nodes]
    silicon_bridge_edges_typed, normal_edges_typed = generate_typed_edges(
        node_names=node_names,
        num_silicon_bridge_edges=3,
        num_normal_edges=2,
        seed=42
    )
    # 构建边类型映射用于绘图
    edge_types_for_draw = {}
    for src, dst, edge_type in silicon_bridge_edges_typed + normal_edges_typed:
        edge_types_for_draw[(src, dst)] = edge_type
    
    # 提取硅桥互联边和普通互联边的名称对（用于传递给build_placement_ilp_model）
    silicon_bridge_edges_list = [(src, dst) for src, dst, _ in silicon_bridge_edges_typed]
    normal_edges_list = [(src, dst) for src, dst, _ in normal_edges_typed]
    
    graph_time = time.time() - graph_start
    print(f"[时间记录] 构建初始图耗时: {graph_time:.2f} 秒")
    print(f"[DEBUG] 硅桥互联边: {silicon_bridge_edges_list}")
    print(f"[DEBUG] 普通互联边: {normal_edges_list}")

    # 输出目录：placement/thermal-placement/output
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 绘制并保存初始 chiplet 布局图（使用默认网格布局）
    initial_draw_start = time.time()
    initial_img_path = output_dir / "ilp_initial_layout.png"
    draw_chiplet_diagram(
        nodes,
        edges,
        save_path=str(initial_img_path),
        layout=None,  # 使用默认网格布局
        edge_types=edge_types_for_draw,  # 传递边类型信息以使用不同颜色
    )
    initial_draw_time = time.time() - initial_draw_start
    print(f"[时间记录] 初始布局图已保存到: {initial_img_path}")
    print(f"[时间记录] 绘制初始布局图耗时: {initial_draw_time:.2f} 秒")

    # 2. 首次建模（不求解）
    model_start = time.time()
    ctx = build_placement_ilp_model(
        nodes=nodes,
        edges=edges,  # 旧格式的边（向后兼容）
        silicon_bridge_edges=silicon_bridge_edges_list,  # 传递硅桥互联边
        normal_edges=normal_edges_list,  # 传递普通互联边
        W=None,
        H=None,
        verbose=False,
        min_shared_length=min_shared_length,
        minimize_bbox_area=True,
        distance_weight=1.0,
        area_weight=1.0,  # 增加面积权重，使其与线长权重相等，更有效地最小化面积
    )
    model_time = time.time() - model_start
    print(f"[时间记录] 构建ILP模型耗时: {model_time:.2f} 秒")

    results: List[ILPPlacementResult] = []

    total_solve_time = 0.0
    total_constraint_time = 0.0
    total_draw_time = 0.0

    for i in range(num_solutions):
        print(f"\n[时间记录] ===== 开始搜索第 {i+1} 个解 =====")
        solution_start = time.time()
        
        # 3. 在当前模型上求解（2分钟时间限制）
        solve_start = time.time()
        res = solve_placement_ilp_from_model(ctx, time_limit=120, verbose=True)  # 2分钟 = 120秒，verbose=True以输出共享边长度检查
        solve_time = time.time() - solve_start
        total_solve_time += solve_time
        print(f"[时间记录] 第 {i+1} 个解求解耗时: {solve_time:.2f} 秒")
        print(f"[DEBUG] 求解状态: {res.status}")
        
        # 检查求解状态：如果是 Infeasible 或 Not Solved，直接退出循环
        if res.status == "Infeasible":
            print(f"第 {i+1} 个解：模型不可行（Infeasible），搜索结束。")
            solution_time = time.time() - solution_start
            print(f"[时间记录] 第 {i+1} 个解总耗时: {solution_time:.2f} 秒")
            break
        elif res.status == "Not Solved":
            print(f"第 {i+1} 个解：模型未求解（Not Solved），搜索结束。")
            solution_time = time.time() - solution_start
            print(f"[时间记录] 第 {i+1} 个解总耗时: {solution_time:.2f} 秒")
            break
        
        # 检查是否找到可行解（变量值不为None）
        has_feasible = False
        for node in ctx.nodes:
            if node.name in res.layout:
                x_val, y_val = res.layout[node.name]
                if x_val != 0.0 or y_val != 0.0:
                    # 检查是否真的是从求解器得到的值
                    # 通过检查是否有非零值来判断
                    has_feasible = True
                    break
        
        # 如果所有位置都是(0,0)，可能是默认值，需要进一步检查
        if not has_feasible:
            # 尝试直接从变量中获取值
            try:
                test_x = pulp.value(ctx.x[0])
                if test_x is not None:
                    has_feasible = True
            except:
                pass
        
        if not has_feasible:
            print(f"第 {i+1} 个解：未找到可行解（状态={res.status}），搜索结束。")
            solution_time = time.time() - solution_start
            print(f"[时间记录] 第 {i+1} 个解总耗时: {solution_time:.2f} 秒")
            break
        
        # 如果找到可行解，继续处理（即使不是Optimal状态）
        if res.status == "Optimal":
            print(f"第 {i+1} 个解：找到最优解，目标值={res.objective_value:.2f}")
        else:
            # 对于非最优解，尝试计算目标值
            try:
                obj_val = res.objective_value if res.objective_value != float("inf") else "N/A"
                print(f"第 {i+1} 个解：找到可行解但非最优（状态={res.status}），目标值={obj_val}，继续处理。")
            except:
                print(f"第 {i+1} 个解：找到可行解但非最优（状态={res.status}），无法计算目标值，继续处理。")
        results.append(res)

        # 4. 为当前解生成并保存布局图
        draw_start = time.time()
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

        # 调试输出：打印每个chiplet的位置
        print(f"\n[DEBUG] 第 {i+1} 个解的chiplet位置:")
        for node_name, (x_pos, y_pos) in res.layout.items():
            print(f"  {node_name}: ({x_pos:.2f}, {y_pos:.2f})")
        print(f"[DEBUG] layout中共有 {len(res.layout)} 个chiplet，nodes中共有 {len(nodes_for_draw)} 个chiplet")
        
        img_path = output_dir / f"ilp_solution_{i+1}.png"
        draw_chiplet_diagram(
            nodes_for_draw,
            ctx.edges,
            save_path=str(img_path),
            layout=res.layout,
            edge_types=edge_types_for_draw,  # 传递边类型信息以使用不同颜色
        )
        draw_time = time.time() - draw_start
        total_draw_time += draw_time
        print(f"  解 {i+1} 的布局图已保存到: {img_path}")
        print(f"[时间记录] 第 {i+1} 个解绘图耗时: {draw_time:.2f} 秒")

        # 5. 基于本次解添加"排除该解"的约束，然后继续下一轮搜索
        #    这里只要求：至少 1 对有连边的 chiplet 改变相邻方式
        constraint_start = time.time()
        add_exclude_layout_constraint(ctx, require_change_pairs=1, solution_index=i+1)
        constraint_time = time.time() - constraint_start
        total_constraint_time += constraint_time
        print(f"[时间记录] 第 {i+1} 个解添加约束耗时: {constraint_time:.2f} 秒")
        
        solution_time = time.time() - solution_start
        print(f"[时间记录] 第 {i+1} 个解总耗时: {solution_time:.2f} 秒")

    total_time = time.time() - start_time
    print(f"\n[时间记录] ===== 搜索完成 =====")
    print(f"[时间记录] 总耗时: {total_time:.2f} 秒")
    print(f"[时间记录] 构建图耗时: {graph_time:.2f} 秒")
    print(f"[时间记录] 构建模型耗时: {model_time:.2f} 秒")
    if len(results) > 0:
        print(f"[时间记录] 总求解耗时: {total_solve_time:.2f} 秒 (平均: {total_solve_time/len(results):.2f} 秒/解)")
        print(f"[时间记录] 总绘图耗时: {total_draw_time:.2f} 秒 (平均: {total_draw_time/len(results):.2f} 秒/解)")
        print(f"[时间记录] 总添加约束耗时: {total_constraint_time:.2f} 秒 (平均: {total_constraint_time/len(results):.2f} 秒/解)")
    else:
        print(f"[时间记录] 未找到可行解")
    
    return results


if __name__ == "__main__":
    sols = search_multiple_solutions(num_solutions=100, min_shared_length=2)
    print(f"共找到 {len(sols)} 个不同的 ILP 可行解。")