from __future__ import annotations

from typing import Dict, Tuple, List, Optional
from pathlib import Path
from copy import deepcopy
import math

import gurobipy as gp
from gurobipy import GRB

from tool import draw_chiplet_diagram, print_constraint_formal, print_pair_distances_only
from ilp_method_gurobi import (
    ILPModelContext,
    ILPPlacementResult,
    add_absolute_value_constraint_big_m,
    build_placement_ilp_model,
    solve_placement_ilp_from_model,
)


def _get_var_value(model: gp.Model, var_name: str) -> Optional[float]:
    v = model.getVarByName(var_name)
    if v is None:
        return None
    try:
        return float(v.X)
    except Exception:
        return None


def _compute_objective_terms_from_model(
    ctx: ILPModelContext,
) -> Tuple[float, Optional[float], Optional[float]]:
    """
    计算并返回目标函数中的关键项数值（与 ilp_method_gurobi.py 的建模变量一致）：
    - wirelength: sum(dx_abs_{i}_{j} + dy_abs_{i}_{j})
    - t: bbox_area_proxy_t
    - aspect_ratio_penalty: aspect_ratio_diff（若未启用则为 None）
    """
    model = ctx.model

    # wirelength：由 build_placement_ilp_model 创建的 dx_abs_{i}_{j}, dy_abs_{i}_{j} 组成
    wirelength_val = 0.0
    connected_pairs: List[Tuple[int, int]] = []
    if getattr(ctx, "silicon_bridge_pairs", None):
        connected_pairs.extend(ctx.silicon_bridge_pairs)
    if getattr(ctx, "standard_pairs", None):
        connected_pairs.extend(ctx.standard_pairs)

    for (i, j) in connected_pairs:
        dx = _get_var_value(model, f"dx_abs_{i}_{j}")
        dy = _get_var_value(model, f"dy_abs_{i}_{j}")
        # 保险：若变量名方向不同，尝试互换
        if dx is None:
            dx = _get_var_value(model, f"dx_abs_{j}_{i}")
        if dy is None:
            dy = _get_var_value(model, f"dy_abs_{j}_{i}")
        if dx is None or dy is None:
            continue
        wirelength_val += dx + dy

    # 面积代理变量 t
    t_val = _get_var_value(model, "bbox_area_proxy_t")

    # 长宽比偏差（如果启用 minimize_bbox_area 才会存在）
    aspect_val = _get_var_value(model, "aspect_ratio_diff")

    return wirelength_val, t_val, aspect_val


# 单次求解（固定 60s），允许返回可行解（非最优）
def _solve_once_with_gap(
    *,
    ctx: ILPModelContext,
    nodes: List,
    gap: float,
    time_limit: int = 60,
    mip_focus: int = 3,
    heuristics: float = 0.5,
) -> ILPPlacementResult:
    """
    单次求解（固定 time_limit 秒），允许返回可行解（非最优）。
    - 有可行解：返回 status="Optimal" 或 "Feasible"，objective_value 为 ObjVal
    - 无可行解：返回 status="NoSolution"，objective_value 为 inf
    """
    import time as _time

    model = ctx.model
    model.Params.TimeLimit = time_limit
    model.Params.MIPGap = gap
    model.Params.MIPFocus = mip_focus
    model.Params.Heuristics = heuristics
    model.Params.LogToConsole = True

    start = _time.time()
    model.optimize()
    solve_time = _time.time() - start

    status = model.Status
    sol_count = int(getattr(model, "SolCount", 0))

    # 有可行解（包括最优/非最优/超时有解）
    if sol_count > 0 and status in (GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT):
        layout: Dict[str, Tuple[float, float]] = {}
        rotations: Dict[str, bool] = {}
        cx_grid_val: Dict[str, float] = {}
        cy_grid_val: Dict[str, float] = {}
        for k, node in enumerate(nodes):
            x_val = float(ctx.x_grid_var[k].X) if ctx.x_grid_var.get(k) is not None else 0.0
            y_val = float(ctx.y_grid_var[k].X) if ctx.y_grid_var.get(k) is not None else 0.0
            r_val = float(ctx.r[k].X) if ctx.r.get(k) is not None else 0.0
            layout[node.name] = (x_val, y_val)
            rotations[node.name] = bool(r_val > 0.5)
            cx_grid_val[node.name] = float(ctx.cx_grid_var[k].X) if ctx.cx_grid_var.get(k) is not None else 0.0
            cy_grid_val[node.name] = float(ctx.cy_grid_var[k].X) if ctx.cy_grid_var.get(k) is not None else 0.0

        try:
            bw_val = float(ctx.bbox_w.X) if ctx.bbox_w is not None else 0.0
            bh_val = float(ctx.bbox_h.X) if ctx.bbox_h is not None else 0.0
        except Exception:
            bw_val, bh_val = 0.0, 0.0

        status_str = "Optimal" if status == GRB.OPTIMAL else "Feasible"
        obj_val = float(model.ObjVal)
        print(
            f"[EMIB] 单次求解完成: MIPGap={gap}, 状态={status_str}, Obj={obj_val:.6f}, 用时={solve_time:.2f}s, SolCount={sol_count}"
        )

        return ILPPlacementResult(
            layout=layout,
            rotations=rotations,
            objective_value=obj_val,
            status=status_str,
            solve_time=solve_time,
            bounding_box=(bw_val, bh_val),
            cx_grid_var=cx_grid_val,
            cy_grid_var=cy_grid_val,
        )

    # 无可行解
    print(f"[EMIB] 单次求解无可行解: MIPGap={gap}, 状态码={status}, 用时={solve_time:.2f}s, SolCount={sol_count}")
    empty_layout = {node.name: (0.0, 0.0) for node in nodes}
    empty_rot = {node.name: False for node in nodes}
    return ILPPlacementResult(
        layout=empty_layout,
        rotations=empty_rot,
        objective_value=float("inf"),
        status="NoSolution",
        solve_time=solve_time,
        bounding_box=(0.0, 0.0),
        cx_grid_var={node.name: 0.0 for node in nodes},
        cy_grid_var={node.name: 0.0 for node in nodes},
    )


# print_pair_distances_only 和 print_all_variables 函数已移动到 tool.py
# 从 tool.py 导入这些函数

def search_multiple_solutions(
    num_solutions: int = 3,
    min_shared_length: float = 0.5,
    input_json_path: Optional[str] = None,
    nodes: Optional[List] = None,
    edges: Optional[List[Tuple[int, int]]] = None,
    fixed_chiplet_idx: Optional[int] = None,
    min_pair_dist_diff: Optional[float] = None,  # chiplet对之间距离差异的最小阈值；此参数控制距离排除约束：至少有一对chiplet的距离差必须 >= min_pair_dist_diff
    time_limit: int = 600,  # 求解时间限制（秒），默认10分钟
    output_dir: Optional[str] = None,  # 输出目录，用于保存.lp文件；如果为None，则使用默认路径
    image_output_dir: Optional[str] = None,  # 图片输出目录，用于保存图片；如果为None则使用output_dir
) -> List[ILPPlacementResult]:
    """
    搜索多个不同的解。
    
    参数:
        num_solutions: 需要搜索的解的数量
        min_shared_length: 相邻chiplet之间的最小共享边长
        input_json_path: 可选，从JSON文件加载输入
        nodes: 可选，chiplet节点列表（如果提供input_json_path则忽略此参数）
        edges: 可选，连接关系列表（如果提供input_json_path则忽略此参数）
        fixed_chiplet_idx: 固定位置的chiplet索引
        min_pair_dist_diff: chiplet对之间距离差异的最小阈值；此参数控制距离排除约束：至少有一对chiplet的距离差必须 >= min_pair_dist_diff
        output_dir: 输出目录，用于保存.lp文件；如果为None，则使用默认路径（相对于项目根目录的output目录）
        image_output_dir: 图片输出目录，用于保存图片；如果为None则使用output_dir
    """
    # 如果提供了input_json_path，则从JSON文件加载
    if input_json_path is not None:
        import json
        from tool import ChipletNode
        
        with open(input_json_path, 'r') as f:
            data = json.load(f)
        
        # 处理JSON格式：{"chiplets": [...], "connections": [...]}
        nodes = []
        edges = []
        
        if "chiplets" not in data or not isinstance(data["chiplets"], list):
            raise ValueError(f"JSON文件格式错误：必须包含 'chiplets' 列表字段。文件: {input_json_path}")
        
        # 构建 ChipletNode 对象
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
        
        # 提取连接关系，支持连接类型（第四列：1=silicon_bridge, 0=standard）
        if "connections" in data and isinstance(data["connections"], list):
            for conn in data["connections"]:
                if isinstance(conn, list) and len(conn) >= 3:
                    src, dst = conn[0], conn[1]
                    # 读取连接类型：必须有第四列
                    if len(conn) < 4:
                        raise ValueError(f"连接格式错误：必须包含4列 [src, dst, weight, conn_type]。当前连接: {conn}")
                    conn_type = conn[3]
                    if conn_type not in [0, 1]:
                        raise ValueError(f"连接类型错误：conn_type 必须是 0 (standard) 或 1 (silicon_bridge)。当前值: {conn_type}")
                    # 确保边是唯一的（避免重复），但保留连接类型信息
                    if src < dst:
                        edge = (src, dst, conn_type)
                    else:
                        edge = (dst, src, conn_type)
                    # 检查是否已存在相同的边（忽略连接类型）
                    edge_exists = any(e[0] == edge[0] and e[1] == edge[1] for e in edges)
                    if not edge_exists:
                        edges.append(edge)
                else:
                    raise ValueError(f"连接格式错误：每个连接必须是包含至少3个元素的列表 [src, dst, weight, conn_type]。当前连接: {conn}")
        
        if len(edges) == 0:
            raise ValueError(f"JSON文件中没有找到有效的连接关系。文件: {input_json_path}")
    elif nodes is None or edges is None:
        raise ValueError("必须提供 input_json_path 或 nodes 和 edges 参数")
    
    solutions = []
    all_prev_pair_distances = []
    
    # 全局约束计数器，确保每个约束名称唯一
    constraint_counter = [0]
    
    # 确定 min_pair_dist_diff 的值：如果为None，则使用默认值1.0
    if min_pair_dist_diff is None:
        min_pair_dist_diff = 1.0
    
    # =========================
    # EMIB(硅桥)降级循环：
    # - 先按当前 conn_type 进行求解
    # - 若不可行：从 silicon_bridge 边中挑 1 条降级为 standard，再求解
    #   选择规则：按 weight 升序；weight 相同按 max(indeg(u), indeg(v)) 升序
    # =========================

    # 重新从 JSON 中读取带 weight 的 connections（否则 edges 中丢失 weight）
    connections: List[List] = []
    if input_json_path is not None:
        import json
        with open(input_json_path, "r") as f:
            data = json.load(f)
        if "connections" not in data or not isinstance(data["connections"], list):
            raise ValueError(f"JSON文件格式错误：必须包含 'connections' 列表字段。文件: {input_json_path}")
        for conn in data["connections"]:
            if not (isinstance(conn, list) and len(conn) == 4):
                raise ValueError(f"连接格式错误：必须是4列 [src, dst, weight, conn_type]。当前连接: {conn}")
            if conn[3] not in [0, 1]:
                raise ValueError(f"连接类型错误：conn_type 必须是 0 或 1。当前连接: {conn}")
            connections.append(conn)
    else:
        # 若未提供 json 路径，无法进行 weight 排序降级
        raise ValueError("EMIB降级搜索需要提供 input_json_path 以读取连接权重")

    # 归一化：按无向边合并（保留最小 weight；conn_type 取 max）
    edge_map: Dict[Tuple[str, str], Dict[str, float]] = {}
    for (s, t, w, ct) in connections:
        a, b = (s, t) if s <= t else (t, s)
        if (a, b) not in edge_map:
            edge_map[(a, b)] = {"weight": float(w), "conn_type": int(ct)}
        else:
            edge_map[(a, b)]["weight"] = min(edge_map[(a, b)]["weight"], float(w))
            edge_map[(a, b)]["conn_type"] = max(edge_map[(a, b)]["conn_type"], int(ct))

    name_to_idx = {node.name: k for k, node in enumerate(nodes)}

    # 最多降级次数 = 当前硅桥边数（每次降 1 条）
    max_relax = sum(1 for v in edge_map.values() if int(v["conn_type"]) == 1)

    for relax_iter in range(max_relax + 1):
        # 当前用于求解的 edges（三元组：src, dst, conn_type）
        edges = [(a, b, int(v["conn_type"])) for (a, b), v in edge_map.items()]

        # 分类得到 silicon_bridge_pairs / standard_pairs
        silicon_bridge_pairs: List[Tuple[int, int]] = []
        standard_pairs: List[Tuple[int, int]] = []
        for (src, dst, conn_type) in edges:
            if src not in name_to_idx or dst not in name_to_idx:
                continue
            i = name_to_idx[src]
            j = name_to_idx[dst]
            if i == j:
                continue
            if i > j:
                i, j = j, i
            if conn_type == 1:
                silicon_bridge_pairs.append((i, j))
            else:
                standard_pairs.append((i, j))
        silicon_bridge_pairs = list(set(silicon_bridge_pairs))
        standard_pairs = list(set(standard_pairs))

        print(f"\n[EMIB] 第 {relax_iter+1} 轮求解：silicon_bridge={len(silicon_bridge_pairs)}, standard={len(standard_pairs)}")

        # 构建ILP模型
        ctx = build_placement_ilp_model(
            nodes=nodes,
            edges=edges,
            fixed_chiplet_idx=fixed_chiplet_idx,
            min_shared_length=min_shared_length,
            silicon_bridge_pairs=silicon_bridge_pairs,
            standard_pairs=standard_pairs,
        )
        # 导出LP文件（在求解之前，包含所有约束）
        # 确定输出目录
        if output_dir is None:
            # 默认输出目录：相对于项目根目录的output目录
            default_output = Path(__file__).parent.parent / "output_gurobi"
            output_dir_path = default_output
        else:
            output_dir_path = Path(output_dir)
            # 如果是相对路径，将其解析为相对于项目根目录的路径
            if not output_dir_path.is_absolute():
                project_root = Path(__file__).parent.parent
                output_dir_path = project_root / output_dir_path
        output_dir_path.mkdir(parents=True, exist_ok=True)
        lp_file = output_dir_path / f"constraints_relax_{relax_iter+1}_gurobi.lp"
        ctx.model.write(str(lp_file))
        # 简化输出：不再打印LP文件保存信息
        # print(f"\nLP文件已保存: {lp_file}")
        
        # =========================
        # 两阶段求解（每次边降级会进行两次求解）：
        # 1) MIPGap = 0.05（质量更高）
        # 2) 若仍无解：MIPGap = 0.3（更激进找可行解）
        # 单次求解 TimeLimit 固定为 60s
        # 只要任意阶段找到可行解（包括非最优），就输出并停止；否则再进行边降级
        # =========================

        # 第一阶段：MIPGap=0.05
        result = _solve_once_with_gap(ctx=ctx, nodes=nodes, gap=0.05, time_limit=60)

        # 若第一阶段仍无可行解，则第二阶段放宽到 MIPGap=0.3
        if result.status == "NoSolution":
            print(f"[EMIB] 第一阶段无可行解，切换到第二阶段 MIPGap=0.3 继续尝试。")
            result = _solve_once_with_gap(ctx=ctx, nodes=nodes, gap=0.3, time_limit=60)

        # 若任一阶段得到可行/最优解：输出并结束
        if result.status in ("Optimal", "Feasible"):
            print(f"[EMIB] 求解成功（{result.status}），不再降级硅桥边。")

            # 输出修改后的互联关系（connections，4列）
            updated_connections = []
            for (a, b), v in edge_map.items():
                updated_connections.append([a, b, v["weight"], int(v["conn_type"])])
            updated_connections.sort(key=lambda x: (x[3], x[2], x[0], x[1]))  # 先按类型，再按权重
            print("[EMIB] 修改后的互联关系（[src, dst, weight, conn_type]）:")
            for c in updated_connections:
                print(f"  {c}")

            # === 保存布局图片（保留可视化输出）===
            try:
                layout_dict: Dict[str, Tuple[float, float]] = {}
                fixed_chiplet_names = set()
                for k, node in enumerate(nodes):
                    node_name = node.name if hasattr(node, "name") else f"Chiplet_{k}"
                    if node_name in result.layout:
                        x_val, y_val = result.layout[node_name]
                        layout_dict[node_name] = (float(x_val), float(y_val))
                    else:
                        layout_dict[node_name] = (0.0, 0.0)
                    if fixed_chiplet_idx is not None and k == fixed_chiplet_idx:
                        fixed_chiplet_names.add(node_name)

                # 边类型映射：用于绘图区分硅桥/普通互联
                edge_type_map: Dict[Tuple[str, str], str] = {}
                for (src, dst, conn_type) in edges:
                    if conn_type == 1:
                        edge_type_map[(src, dst)] = "silicon_bridge"
                        edge_type_map[(dst, src)] = "silicon_bridge"
                    else:
                        edge_type_map[(src, dst)] = "normal"
                        edge_type_map[(dst, src)] = "normal"

                project_root = Path(__file__).parent.parent
                if image_output_dir is not None:
                    image_output_dir_path = Path(image_output_dir)
                    if not image_output_dir_path.is_absolute():
                        image_output_dir_path = project_root / image_output_dir_path
                elif output_dir is not None:
                    # 若 output_dir 类似 output_gurobi/lp/<case>_core，则图片保存到 output_gurobi/fig/<case>_core
                    outp = Path(output_dir)
                    if not outp.is_absolute():
                        outp = project_root / outp
                    if outp.name:
                        image_output_dir_path = outp.parent.parent / "fig" / outp.name
                    else:
                        image_output_dir_path = outp.parent / "fig"
                else:
                    image_output_dir_path = project_root / "output_gurobi" / "fig"

                image_output_dir_path.mkdir(parents=True, exist_ok=True)
                image_path = image_output_dir_path / f"solution_1_layout_gurobi.png"

                draw_chiplet_diagram(
                    nodes=nodes,
                    edges=edges,
                    save_path=str(image_path),
                    layout=layout_dict,
                    fixed_chiplet_names=fixed_chiplet_names if fixed_chiplet_names else None,
                    rotations=result.rotations if hasattr(result, "rotations") else None,
                    edge_types=edge_type_map,
                )
                print(f"[EMIB] 布局图片已保存: {image_path}")
            except Exception as e:
                print(f"[EMIB] 警告：保存布局图片失败: {e}")
                import traceback
                traceback.print_exc()

            solutions.append(result)
            break

        # 两阶段都无可行解：进行边降级后进入下一轮
        print(f"[EMIB] 两阶段均无可行解，准备降级1条硅桥边后继续。")

        # 没有硅桥边可降级了：停止
        # 入度（无向图度数）先全图统计，再用于 tie-break，避免遍历顺序影响结果
        deg: Dict[str, int] = {}
        for (u, v) in edge_map.keys():
            deg[u] = deg.get(u, 0) + 1
            deg[v] = deg.get(v, 0) + 1

        sb_edges = []
        for (a, b), v in edge_map.items():
            if int(v["conn_type"]) == 1:
                sb_edges.append((a, b, float(v["weight"]), max(deg.get(a, 0), deg.get(b, 0))))

        if not sb_edges:
            print("[EMIB] 已无 silicon_bridge 边可降级，停止。")
            break

        # 选择要降级的硅桥边：weight 升序；weight 相同按 max indeg 升序
        sb_edges.sort(key=lambda x: (x[2], x[3], x[0], x[1]))
        a, b, w, md = sb_edges[0]
        print(f"[EMIB] 降级硅桥边: ({a}, {b}), weight={w}, max_indeg={md} -> conn_type=0")
        edge_map[(min(a, b), max(a, b))]["conn_type"] = 0

    return solutions
