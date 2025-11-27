"""
使用整数线性规划（ILP）进行芯片布局优化。

目标：
1. 有互联关系的方形需要相邻
2. 最小化有互联关系的方形外接圆圆心距离

使用 PuLP + GLPK 求解器实现。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pulp

from tool import ChipletNode, build_random_chiplet_graph, draw_chiplet_diagram


@dataclass
class ILPPlacementResult:
    """ILP求解结果"""
    layout: Dict[str, Tuple[float, float]]  # name -> (x, y)
    rotations: Dict[str, bool]  # name -> 是否旋转
    objective_value: float
    status: str
    solve_time: float


def solve_placement_ilp(
    nodes: List[ChipletNode],
    edges: List[Tuple[str, str]],
    adjacency_threshold: float = 3.0,
    max_side: float | None = None,
    time_limit: int = 300,
    verbose: bool = True,
) -> ILPPlacementResult:
    """
    使用ILP求解芯片布局问题。
    
    方法：连续变量 + 距离阈值约束（使用GLPK求解器）
    
    参数
    ----
    nodes:
        Chiplet节点列表
    edges:
        边列表，表示有互联关系的节点对
    adjacency_threshold:
        相邻的阈值（中心距离小于此值视为相邻）
    max_side:
        最大布局区域边长（如果不指定，自动计算）
    time_limit:
        求解时间限制（秒）
    verbose:
        是否打印详细信息
    
    返回
    ----
    ILPPlacementResult
        包含布局、目标值、求解状态等信息
    """
    import time
    start_time = time.time()
    
    n = len(nodes)
    name_to_idx = {node.name: i for i, node in enumerate(nodes)}
    
    # 获取原始矩形尺寸（未旋转时）
    widths_orig = [float(node.dimensions.get("x", 0.0)) for node in nodes]
    heights_orig = [float(node.dimensions.get("y", 0.0)) for node in nodes]
    
    # 获取原始接口位置（未旋转时）
    phys_orig = []
    for node in nodes:
        phys_list = []
        for p in node.phys:
            phys_list.append({
                'x': float(p.get("x", 0.0)),
                'y': float(p.get("y", 0.0))
            })
        phys_orig.append(phys_list)
    
    # 找到所有有边连接的矩形对
    connected_pairs = []
    for src_name, dst_name in edges:
        if src_name in name_to_idx and dst_name in name_to_idx:
            i = name_to_idx[src_name]
            j = name_to_idx[dst_name]
            if i != j:
                connected_pairs.append((min(i, j), max(i, j)))
    connected_pairs = list(set(connected_pairs))  # 去重
    
    if verbose:
        print(f"问题规模: {n} 个矩形, {len(connected_pairs)} 个有连接的矩形对")
    
    # 估算最大布局区域（考虑旋转，取较大值）
    if max_side is None:
        max_w = max(max(widths_orig), max(heights_orig))
        max_h = max(max(widths_orig), max(heights_orig))
        total_area = sum(w * h for w, h in zip(widths_orig, heights_orig))
        max_side = math.ceil(math.sqrt(total_area) * 1.5)
        max_side = max(max_side, max_w * 2, max_h * 2)
    
    # Big-M 常数
    M = max_side * 3
    
    if verbose:
        print(f"最大布局区域: {max_side:.2f} x {max_side:.2f}")
        print(f"相邻阈值: {adjacency_threshold:.2f}")
        print(f"支持旋转: 是")
    
    # 创建ILP问题
    prob = pulp.LpProblem("ChipletPlacement", pulp.LpMinimize)
    
    # ============ 变量定义 ============
    
    # 旋转变量：矩形 i 是否旋转90度（二进制变量）
    r = {}
    for i in range(n):
        r[i] = pulp.LpVariable(f"r_{i}", cat='Binary')
    
    # 实际宽度和高度（连续变量，取决于旋转状态）
    # 根据公式：w = r * h_orig + (1-r) * w_orig = w_orig + r * (h_orig - w_orig)
    #          h = r * w_orig + (1-r) * h_orig = h_orig + r * (w_orig - h_orig)
    w = {}
    h = {}
    for i in range(n):
        w_min = min(widths_orig[i], heights_orig[i])
        w_max = max(widths_orig[i], heights_orig[i])
        w[i] = pulp.LpVariable(f"w_{i}", lowBound=w_min, upBound=w_max)
        h[i] = pulp.LpVariable(f"h_{i}", lowBound=w_min, upBound=w_max)
    
    # 位置变量：矩形 i 的左下角坐标（连续变量）
    x = {}
    y = {}
    for i in range(n):
        # 上界需要考虑旋转后的最大尺寸
        max_w_i = max(widths_orig[i], heights_orig[i])
        max_h_i = max(widths_orig[i], heights_orig[i])
        x[i] = pulp.LpVariable(f"x_{i}", lowBound=0, upBound=max_side - min(widths_orig[i], heights_orig[i]))
        y[i] = pulp.LpVariable(f"y_{i}", lowBound=0, upBound=max_side - min(widths_orig[i], heights_orig[i]))
    
    # 中心坐标（辅助变量，取决于旋转后的尺寸）
    cx = {}
    cy = {}
    for i in range(n):
        cx[i] = pulp.LpVariable(f"cx_{i}", lowBound=0)
        cy[i] = pulp.LpVariable(f"cy_{i}", lowBound=0)
    
    # 距离辅助变量（用于线性化绝对值）
    dx = {}
    dy = {}
    for i, j in connected_pairs:
        dx[(i, j)] = pulp.LpVariable(f"dx_{i}_{j}", lowBound=0)
        dy[(i, j)] = pulp.LpVariable(f"dy_{i}_{j}", lowBound=0)
    
    # 相邻关系变量：矩形 i 和 j 是否相邻（用于约束）
    adjacent = {}
    for i, j in connected_pairs:
        adjacent[(i, j)] = pulp.LpVariable(f"adjacent_{i}_{j}", cat='Binary')
    
    # 非重叠约束的辅助变量
    # 对于每对矩形 (i,j)，使用4个二进制变量表示4种不重叠情况
    non_overlap = {}
    for i in range(n):
        for j in range(i + 1, n):
            non_overlap[(i, j, 0)] = pulp.LpVariable(f"no_{i}_{j}_0", cat='Binary')  # i在j左边
            non_overlap[(i, j, 1)] = pulp.LpVariable(f"no_{i}_{j}_1", cat='Binary')  # i在j右边
            non_overlap[(i, j, 2)] = pulp.LpVariable(f"no_{i}_{j}_2", cat='Binary')  # i在j下边
            non_overlap[(i, j, 3)] = pulp.LpVariable(f"no_{i}_{j}_3", cat='Binary')  # i在j上边
    
    # ============ 约束 ============
    
    # 约束0: 宽度和高度的旋转约束
    # 根据公式：w = r * h_orig + (1-r) * w_orig = w_orig + r * (h_orig - w_orig)
    #          h = r * w_orig + (1-r) * h_orig = h_orig + r * (w_orig - h_orig)
    # 使用 Big-M 方法线性化二进制变量与常数的乘积
    # 引入辅助变量来表示 r[i] * constant
    aux_w = {}
    aux_h = {}
    for i in range(n):
        aux_w[i] = pulp.LpVariable(f"aux_w_{i}", lowBound=0)
        aux_h[i] = pulp.LpVariable(f"aux_h_{i}", lowBound=0)
    
    # 定义辅助变量约束和宽度/高度约束
    for i in range(n):
        w_diff = heights_orig[i] - widths_orig[i]
        h_diff = widths_orig[i] - heights_orig[i]
        
        # aux_w[i] = r[i] * w_diff 的线性化（使用 Big-M 方法）
        # 当 r[i] = 0 时，aux_w[i] = 0
        # 当 r[i] = 1 时，aux_w[i] = w_diff
        prob += aux_w[i] <= M * r[i], f"aux_w_{i}_ub"
        prob += aux_w[i] <= w_diff + M * (1 - r[i]), f"aux_w_{i}_ub2"
        prob += aux_w[i] >= w_diff - M * (1 - r[i]), f"aux_w_{i}_lb"
        prob += aux_w[i] >= 0, f"aux_w_{i}_nonneg"
        
        # aux_h[i] = r[i] * h_diff 的线性化
        prob += aux_h[i] <= M * r[i], f"aux_h_{i}_ub"
        prob += aux_h[i] <= h_diff + M * (1 - r[i]), f"aux_h_{i}_ub2"
        prob += aux_h[i] >= h_diff - M * (1 - r[i]), f"aux_h_{i}_lb"
        prob += aux_h[i] >= 0, f"aux_h_{i}_nonneg"
        
        # 定义宽度和高度
        prob += w[i] == widths_orig[i] + aux_w[i], f"width_rotation_{i}"
        prob += h[i] == heights_orig[i] + aux_h[i], f"height_rotation_{i}"
    
    # 约束0.5: 中心坐标定义
    for i in range(n):
        prob += cx[i] == x[i] + w[i] / 2.0, f"cx_def_{i}"
        prob += cy[i] == y[i] + h[i] / 2.0, f"cy_def_{i}"
    
    # 约束1: 非重叠约束（使用旋转后的宽度和高度）
    for i in range(n):
        for j in range(i + 1, n):
            # 至少有一种不重叠关系
            prob += non_overlap[(i, j, 0)] + non_overlap[(i, j, 1)] + \
                   non_overlap[(i, j, 2)] + non_overlap[(i, j, 3)] >= 1, \
                   f"non_overlap_{i}_{j}_any"
            
            # i在j左边（使用旋转后的宽度 w[i]）
            prob += x[i] + w[i] <= x[j] + M * (1 - non_overlap[(i, j, 0)]), \
                   f"non_overlap_{i}_{j}_left"
            
            # i在j右边（使用旋转后的宽度 w[j]）
            prob += x[j] + w[j] <= x[i] + M * (1 - non_overlap[(i, j, 1)]), \
                   f"non_overlap_{i}_{j}_right"
            
            # i在j下边（使用旋转后的高度 h[i]）
            prob += y[i] + h[i] <= y[j] + M * (1 - non_overlap[(i, j, 2)]), \
                   f"non_overlap_{i}_{j}_bottom"
            
            # i在j上边（使用旋转后的高度 h[j]）
            prob += y[j] + h[j] <= y[i] + M * (1 - non_overlap[(i, j, 3)]), \
                   f"non_overlap_{i}_{j}_top"
    
    # 约束2: 相邻关系约束（有连接的矩形必须相邻）
    # 【已注释】暂时关闭相邻约束，以便观察是否能放置所有方块
    # 注意：这里使用中心距离，但理想情况下应该使用接口位置之间的距离
    # 为了简化，我们仍使用中心距离，但可以考虑接口位置优化
    # for i, j in connected_pairs:
    #     # 强制相邻
    #     prob += adjacent[(i, j)] == 1, f"adjacent_{i}_{j}_required"
    #     
    #     # 距离约束：如果相邻，则中心距离必须小于阈值
    #     # （可选：也可以约束接口之间的距离，但需要更复杂的建模）
    #     prob += dx[(i, j)] + dy[(i, j)] <= adjacency_threshold, \
    #             f"distance_{i}_{j}_constraint"
    
    # 约束3: 距离辅助变量的定义（绝对值线性化）
    for i, j in connected_pairs:
        prob += dx[(i, j)] >= cx[i] - cx[j], f"dx_{i}_{j}_pos"
        prob += dx[(i, j)] >= cx[j] - cx[i], f"dx_{i}_{j}_neg"
        prob += dy[(i, j)] >= cy[i] - cy[j], f"dy_{i}_{j}_pos"
        prob += dy[(i, j)] >= cy[j] - cy[i], f"dy_{i}_{j}_neg"
    
    # ============ 目标函数 ============
    # 最小化有连接的矩形对的中心距离之和
    prob += pulp.lpSum([dx[(i, j)] + dy[(i, j)] for i, j in connected_pairs]), "total_distance"
    
    # ============ 求解 ============
    if verbose:
        print("\n开始求解ILP问题...")
        print(f"变量数量: {prob.numVariables()}")
        print(f"约束数量: {prob.numConstraints()}")
    
    # 使用GLPK求解器
    try:
        solver = pulp.getSolver('GLPK_CMD', timeLimit=time_limit, msg=verbose)
    except:
        # 如果GLPK不可用，尝试使用默认求解器
        if verbose:
            print("警告: GLPK不可用，使用默认求解器")
        solver = None
    
    try:
        status = prob.solve(solver)
        solve_time = time.time() - start_time
        
        if verbose:
            print(f"\n求解状态: {pulp.LpStatus[status]}")
            print(f"求解时间: {solve_time:.2f} 秒")
            if status == pulp.LpStatusOptimal:
                print(f"目标函数值: {pulp.value(prob.objective):.2f}")
        
        # 提取解
        layout = {}
        rotations = {}
        for i, node in enumerate(nodes):
            if status == pulp.LpStatusOptimal:
                x_val = pulp.value(x[i])
                y_val = pulp.value(y[i])
                r_val = pulp.value(r[i])
                layout[node.name] = (x_val if x_val is not None else 0.0,
                                     y_val if y_val is not None else 0.0)
                rotations[node.name] = bool(r_val > 0.5) if r_val is not None else False
            else:
                layout[node.name] = (0.0, 0.0)
                rotations[node.name] = False
        
        obj_value = pulp.value(prob.objective) if status == pulp.LpStatusOptimal else float('inf')
        
        return ILPPlacementResult(
            layout=layout,
            rotations=rotations,
            objective_value=obj_value,
            status=pulp.LpStatus[status],
            solve_time=solve_time,
        )
    
    except Exception as e:
        solve_time = time.time() - start_time
        if verbose:
            print(f"\n求解出错: {e}")
            import traceback
            traceback.print_exc()
        
        # 返回空解
        layout = {node.name: (0.0, 0.0) for node in nodes}
        rotations = {node.name: False for node in nodes}
        return ILPPlacementResult(
            layout=layout,
            rotations=rotations,
            objective_value=float('inf'),
            status="Error",
            solve_time=solve_time,
        )


if __name__ == "__main__":
    # 测试：使用ILP求解布局问题
    print("=" * 80)
    print("ILP芯片布局求解器测试")
    print("=" * 80)
    
    # 构建测试图（使用所有节点）
    nodes, edges = build_random_chiplet_graph(edge_prob=0.2)
    
    print(f"\n问题规模: {len(nodes)} 个矩形, {len(edges)} 条边")
    
    # 求解
    result = solve_placement_ilp(
        nodes=nodes,
        edges=edges,
        adjacency_threshold=5.0,
        time_limit=120,
        verbose=True,
    )
    
    print(f"\n求解结果:")
    print(f"  状态: {result.status}")
    if result.status == 'Optimal':
        print(f"  目标值: {result.objective_value:.2f}")
        # 显示旋转信息
        rotated_nodes = [name for name, rotated in result.rotations.items() if rotated]
        if rotated_nodes:
            print(f"  旋转的节点: {', '.join(rotated_nodes)}")
        else:
            print(f"  旋转的节点: 无")
    print(f"  求解时间: {result.solve_time:.2f} 秒")
    
    # 可视化结果
    if result.status == 'Optimal':
        from pathlib import Path
        out_path = Path(__file__).parent.parent / "chiplet_ilp_placement.png"
        
        # 创建旋转后的节点副本用于绘图
        from copy import deepcopy
        nodes_for_draw = []
        for i, node in enumerate(nodes):
            node_copy = deepcopy(node)
            if result.rotations.get(node.name, False):
                # 旋转90度：交换宽度和高度
                orig_w = node.dimensions.get("x", 0.0)
                orig_h = node.dimensions.get("y", 0.0)
                node_copy.dimensions["x"] = orig_h
                node_copy.dimensions["y"] = orig_w
                
                # 旋转接口位置：原来的 (px, py) 变成 (h - py, px)
                if node_copy.phys:
                    rotated_phys = []
                    for p in node.phys:
                        px = float(p.get("x", 0.0))
                        py = float(p.get("y", 0.0))
                        # 旋转90度：相对于新左下角的坐标
                        new_px = orig_h - py
                        new_py = px
                        rotated_phys.append({"x": new_px, "y": new_py})
                    node_copy.phys = rotated_phys
            nodes_for_draw.append(node_copy)
        
        draw_chiplet_diagram(nodes_for_draw, edges, save_path=str(out_path), layout=result.layout)
        print(f"\n布局结果已保存到: {out_path}")
    else:
        print(f"\n求解未成功，状态: {result.status}")
