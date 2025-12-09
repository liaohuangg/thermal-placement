# -*- coding: utf-8 -*-
"""
TCG合法化模块 - EMIB Legalization

基于论文算法实现EMIB合法化:
1. 在Ch和Cv中分别添加EMIB边来表示桥连接约束
2. 调整node值(坐标)而非修改reduction边
3. 使用最长路径算法计算满足所有约束的坐标
"""
import json
import random
from typing import Dict, List, Tuple, Optional, Set
from TCG import TCG, generate_layout_from_tcg, print_layout_info,get_layout_area   
from Generate_initial_TCG import generate_initial_TCG
import copy
import networkx as nx
from unit import load_problem_from_json,save_layout_to_json
from chiplet_model import Chiplet, LayoutProblem, MIN_OVERLAP, get_adjacency_info, EPSILON
from Bridge_Overlap_Adjustment import (
    SiliconBridge, generate_silicon_bridges, 
    SiliconBridge_is_legal, SILICONBRIDGE_LENGTH
)


def add_emib_edges(tcg: TCG, problem: LayoutProblem, verbose: bool = False) -> TCG:
    """
    添加EMIB边到TCG中
    
    根据算法描述:
    - 如果在Ch中有reduction edge ni→nj,且di和dj之间有桥连接
    - 则在Cv中添加无向EMIB边Eij
    - 反之亦然
    
    Args:
        tcg: 输入TCG
        problem: 布局问题(包含连接图)
        verbose: 是否输出详细信息
        
    Returns:
        添加了EMIB边的新TCG
    """
    # 创建TCG副本
    new_tcg = copy.deepcopy(tcg)
    
    if verbose:
        print("  添加EMIB边...")
    
    # 遍历所有需要连接的芯片对(桥连接)
    for chip1_id, chip2_id in problem.connection_graph.edges():
        # 检查Ch中是否有reduction edge
        has_ch_edge = (new_tcg.Ch.has_edge(chip1_id, chip2_id) or 
                      new_tcg.Ch.has_edge(chip2_id, chip1_id))
        
        # 检查Cv中是否有reduction edge  
        has_cv_edge = (new_tcg.Cv.has_edge(chip1_id, chip2_id) or 
                      new_tcg.Cv.has_edge(chip2_id, chip1_id))
        
        if has_ch_edge:
            # Ch中有边,在Cv中添加EMIB边表示垂直方向的桥连接约束
            # 注意:EMIB边是无向的,但NetworkX的DiGraph需要添加双向边来模拟
            if not new_tcg.Cv.has_edge(chip1_id, chip2_id):
                new_tcg.Cv.add_edge(chip1_id, chip2_id)
            if not new_tcg.Cv.has_edge(chip2_id, chip1_id):
                new_tcg.Cv.add_edge(chip2_id, chip1_id)
            
            if verbose:
                print(f"    Ch中有{chip1_id}-{chip2_id},在Cv中添加EMIB边")
        
        elif has_cv_edge:
            # Cv中有边,在Ch中添加EMIB边表示水平方向的桥连接约束
            if not new_tcg.Ch.has_edge(chip1_id, chip2_id):
                new_tcg.Ch.add_edge(chip1_id, chip2_id)
            if not new_tcg.Ch.has_edge(chip2_id, chip1_id):
                new_tcg.Ch.add_edge(chip2_id, chip1_id)
            
            if verbose:
                print(f"    Cv中有{chip1_id}-{chip2_id},在Ch中添加EMIB边")
    
    return new_tcg


def compute_constrained_longest_path(graph: nx.DiGraph, problem: LayoutProblem,
                                     dimension: str, emib_edges: Set[Tuple[str, str]],
                                     verbose: bool = False) -> Dict[str, float]:
    """
    计算满足EMIB约束的最长路径坐标 - 简化版本
    
    先使用标准最长路径算法,然后迭代调整以满足EMIB约束
    
    Args:
        graph: Ch或Cv图
        problem: 布局问题
        dimension: 'width'或'height'
        emib_edges: EMIB边集合
        verbose: 是否输出详细信息
        
    Returns:
        坐标字典
    """
    from TCG import compute_longest_path_lengths
    
    # 使用标准最长路径算法
    coordinates = compute_longest_path_lengths(graph, problem, dimension)
    
    # 迭代调整以满足EMIB约束
    for iteration in range(50):
        adjusted = False
        
        for chip1_id, chip2_id in emib_edges:
            chip1 = problem.get_chiplet(chip1_id)
            chip2 = problem.get_chiplet(chip2_id)
            
            coord1 = coordinates[chip1_id]
            coord2 = coordinates[chip2_id]
            
            size1 = chip1.width if dimension == 'width' else chip1.height
            size2 = chip2.width if dimension == 'width' else chip2.height
            
            # 计算重叠
            overlap = min(coord1 + size1, coord2 + size2) - max(coord1, coord2)
            
            if overlap < MIN_OVERLAP:
                # 调整以满足最小重叠
                max_overlap = min(size1, size2)
                if coord1 < coord2:
                    # chip2向左/下移
                    new_coord2 = coord1 + size1 - max_overlap
                    if new_coord2 != coord2:
                        coordinates[chip2_id] = new_coord2
                        adjusted = True
                else:
                    # chip1向左/下移
                    new_coord1 = coord2 + size2 - max_overlap
                    if new_coord1 != coord1:
                        coordinates[chip1_id] = new_coord1
                        adjusted = True
        
        if not adjusted:
            break
    
    return coordinates


def detect_illegal_loop(emib_edges: Set[Tuple[str, str]], layout: Dict[str, Chiplet], 
                        verbose: bool = False) -> bool:
    """
    检测是否存在非法循环(Illegal Loop) - 图8(b)
    
    非法循环:多个EMIB边形成的约束无法同时满足
    例如:三个芯片A,B,C形成环形连接,但物理上无法同时满足所有重叠要求
    
    Args:
        emib_edges: EMIB边集合
        layout: 当前布局
        verbose: 是否输出详细信息
        
    Returns:
        是否存在非法循环
    """
    # 构建EMIB边的图
    emib_graph = nx.Graph()
    for chip1_id, chip2_id in emib_edges:
        emib_graph.add_edge(chip1_id, chip2_id)
    
    # 检查是否有环
    try:
        cycles = nx.find_cycle(emib_graph, orientation='ignore')
        if cycles:
            if verbose:
                cycle_nodes = [edge[0] for edge in cycles]
                print(f"  检测到EMIB边形成环: {cycle_nodes}")
            
            # 检查环中的节点是否能同时满足所有重叠约束
            cycle_nodes = list(set([edge[0] for edge in cycles] + [edge[1] for edge in cycles]))
            
            # 简单启发式:如果环中超过3个节点,很可能是非法循环
            if len(cycle_nodes) > 3:
                if verbose:
                    print(f"  环中节点数={len(cycle_nodes)} > 3, 判定为非法循环")
                return True
                
    except nx.NetworkXNoCycle:
        pass
    
    return False


def detect_illegal_crossing(emib_edges: Set[Tuple[str, str]], layout: Dict[str, Chiplet],
                            tcg: TCG, verbose: bool = False) -> bool:
    """
    检测是否存在非法交叉(Illegal Crossing) - 图8(c)
    
    非法交叉:两条EMIB边在拓扑上交叉,导致无法合法化
    暂时禁用此检查,因为很多正常情况也会触发
    
    Args:
        emib_edges: EMIB边集合
        layout: 当前布局
        tcg: TCG拓扑
        verbose: 是否输出详细信息
        
    Returns:
        是否存在非法交叉
    """
    # 暂时禁用非法交叉检测
    return False


def legalize_tcg(tcg: TCG, problem: LayoutProblem, max_iterations: int = 100,
                 verbose: bool = False) -> Tuple[bool, TCG, Dict[str, Chiplet]]:
    """
    TCG合法化 - 简化实用版本
    
    策略:
    1. 从TCG生成初始布局
    2. 迭代调整以满足EMIB约束(邻接+重叠)
    3. 调整时遵守TCG的拓扑约束
    
    Args:
        tcg: 输入TCG
        problem: 布局问题
        max_iterations: 最大迭代次数
        verbose: 是否输出详细信息
        
    Returns:
        (是否成功, TCG, 最终布局)
    """
    if verbose:
        print("\n" + "=" * 70)
        print("TCG合法化开始")
        print("=" * 70)
    
    # 收集桥连接约束
    bridge_connections = list(problem.connection_graph.edges())
    
    if verbose:
        print(f"\n[步骤1] 桥连接约束: {bridge_connections}")
    
    # 生成初始布局
    layout = generate_layout_from_tcg(tcg, problem)
    
    if verbose:
        print(f"\n[步骤2] 初始布局:")
        for chip_id, chip in layout.items():
            print(f"  {chip_id}: ({chip.x:.1f}, {chip.y:.1f})")
    
    # 迭代调整以满足EMIB约束
    for iteration in range(max_iterations):
        if verbose:
            print(f"\n  --- 迭代 {iteration + 1} ---")
        
        adjusted = False
        
        # 对每个桥连接约束进行检查和调整
        for chip1_id, chip2_id in bridge_connections:
            chip1 = layout[chip1_id]
            chip2 = layout[chip2_id]
            
            is_adj, overlap_len, direction = get_adjacency_info(chip1, chip2)
            
            if is_adj and overlap_len >= MIN_OVERLAP:
                continue  # 已满足
            
            # 需要调整 - 根据TCG约束关系决定调整方向
            has_ch_12 = tcg.Ch.has_edge(chip1_id, chip2_id)
            has_ch_21 = tcg.Ch.has_edge(chip2_id, chip1_id)
            has_cv_12 = tcg.Cv.has_edge(chip1_id, chip2_id)
            has_cv_21 = tcg.Cv.has_edge(chip2_id, chip1_id)
            
            if has_cv_12 or has_cv_21:
                # Cv中有边:垂直方向有序,需要x方向重叠
                x_overlap = min(chip1.x + chip1.width, chip2.x + chip2.width) - max(chip1.x, chip2.x)
                
                if x_overlap < MIN_OVERLAP:
                    # 需要增加x方向重叠 - 采用随机目标重叠策略(40%-100%芯片宽度)
                    overlap_ratio = random.uniform(0.4, 1)
                    target_overlap = max(MIN_OVERLAP, min(chip1.width, chip2.width) * overlap_ratio)
                    if has_ch_12:
                        # Ch中chip1→chip2: chip2在chip1右边,只能右移chip1
                        new_x1 = chip2.x + target_overlap - chip1.width
                        if new_x1 > chip1.x:
                            chip1.x = new_x1
                            adjusted = True
                            if verbose:
                                actual_overlap = min(chip1.x + chip1.width, chip2.x + chip2.width) - max(chip1.x, chip2.x)
                                print(f"    调整{chip1_id}.x → {new_x1:.1f} (与{chip2_id}重叠={actual_overlap:.1f})")
                    
                    elif has_ch_21:
                        # Ch中chip2→chip1: chip1在chip2右边,只能右移chip2  
                        new_x2 = chip1.x + target_overlap - chip2.width
                        if new_x2 > chip2.x:
                            chip2.x = new_x2
                            adjusted = True
                            if verbose:
                                actual_overlap = min(chip1.x + chip1.width, chip2.x + chip2.width) - max(chip1.x, chip2.x)
                                print(f"    调整{chip2_id}.x → {new_x2:.1f} (与{chip1_id}重叠={actual_overlap:.1f})")
                    
                    else:
                        # Ch中没有约束,选择左边的芯片右移
                        if chip1.x < chip2.x:
                            new_x1 = chip2.x + target_overlap - chip1.width
                            if new_x1 > chip1.x:
                                chip1.x = new_x1
                                adjusted = True
                                if verbose:
                                    actual_overlap = min(chip1.x + chip1.width, chip2.x + chip2.width) - max(chip1.x, chip2.x)
                                    print(f"    调整{chip1_id}.x → {new_x1:.1f} (与{chip2_id}重叠={actual_overlap:.1f})")
                        else:
                            new_x2 = chip1.x + target_overlap - chip2.width
                            if new_x2 > chip2.x:
                                chip2.x = new_x2
                                adjusted = True
                                if verbose:
                                    actual_overlap = min(chip1.x + chip1.width, chip2.x + chip2.width) - max(chip1.x, chip2.x)
                                    print(f"    调整{chip2_id}.x → {new_x2:.1f} (与{chip1_id}重叠={actual_overlap:.1f})")
            
            elif has_ch_12 or has_ch_21:
                # Ch中有边:水平方向有序,需要y方向重叠
                y_overlap = min(chip1.y + chip1.height, chip2.y + chip2.height) - max(chip1.y, chip2.y)
                
                if y_overlap < MIN_OVERLAP:
                    # 需要增加y方向重叠 - 采用随机目标重叠策略(40%-100%芯片高度)
                    overlap_ratio = random.uniform(0.4, 1)
                    target_overlap = max(MIN_OVERLAP, min(chip1.height, chip2.height) * overlap_ratio)
                    if has_cv_12:
                        # Cv中chip1→chip2: chip2在chip1上方,只能上移chip1
                        new_y1 = chip2.y + target_overlap - chip1.height
                        if new_y1 > chip1.y:
                            chip1.y = new_y1
                            adjusted = True
                            if verbose:
                                actual_overlap = min(chip1.y + chip1.height, chip2.y + chip2.height) - max(chip1.y, chip2.y)
                                print(f"    调整{chip1_id}.y → {new_y1:.1f} (与{chip2_id}重叠={actual_overlap:.1f})")
                    
                    elif has_cv_21:
                        # Cv中chip2→chip1: chip1在chip2上方,只能上移chip2
                        new_y2 = chip1.y + target_overlap - chip2.height
                        if new_y2 > chip2.y:
                            chip2.y = new_y2
                            adjusted = True
                            if verbose:
                                actual_overlap = min(chip1.y + chip1.height, chip2.y + chip2.height) - max(chip1.y, chip2.y)
                                print(f"    调整{chip2_id}.y → {new_y2:.1f} (与{chip1_id}重叠={actual_overlap:.1f})")
                    
                    else:
                        # Cv中没有约束,选择下方的芯片上移
                        if chip1.y < chip2.y:
                            new_y1 = chip2.y + target_overlap - chip1.height
                            if new_y1 > chip1.y:
                                chip1.y = new_y1
                                adjusted = True
                                if verbose:
                                    actual_overlap = min(chip1.y + chip1.height, chip2.y + chip2.height) - max(chip1.y, chip2.y)
                                    print(f"    调整{chip1_id}.y → {new_y1:.1f} (与{chip2_id}重叠={actual_overlap:.1f})")
                        else:
                            new_y2 = chip1.y + target_overlap - chip2.height
                            if new_y2 > chip2.y:
                                chip2.y = new_y2
                                adjusted = True
                                if verbose:
                                    actual_overlap = min(chip1.y + chip1.height, chip2.y + chip2.height) - max(chip1.y, chip2.y)
                                    print(f"    调整{chip2_id}.y → {new_y2:.1f} (与{chip1_id}重叠={actual_overlap:.1f})")
        
        if not adjusted:
            if verbose:
                print(f"  ✓ 收敛于第{iteration + 1}次迭代")
            break
        
        # 每次迭代后检查是否有新的非法边
        illegal_edge_found = False
        for chip1_id, chip2_id in bridge_connections:
            chip1 = layout[chip1_id]
            chip2 = layout[chip2_id]
            is_adj, overlap_len, _ = get_adjacency_info(chip1, chip2)
            if not is_adj or overlap_len < MIN_OVERLAP:
                illegal_edge_found = True
                break
        
        if not illegal_edge_found:
            break
    
    # 步骤6: 邻接调整(Place adjacently) - 图8(d)
    if verbose:
        print(f"\n[步骤6] 邻接调整优化...")
    
    for chip1_id, chip2_id in bridge_connections:
        chip1 = layout[chip1_id]
        chip2 = layout[chip2_id]
        
        is_adj, overlap_len, direction = get_adjacency_info(chip1, chip2)
        
        if is_adj and overlap_len >= MIN_OVERLAP:
            # 已经邻接,尝试进一步紧凑排列
            # 这里可以添加更细致的调整策略
            pass
    
    # 步骤7: 最终验证
    if verbose:
        print(f"\n[步骤7] 最终验证...")
        print(f"  最终布局:")
        for chip_id, chip in layout.items():
            print(f"    {chip_id}: ({chip.x:.1f}, {chip.y:.1f})")
    
    # 验证所有桥连接是否满足
    all_valid = True
    for chip1_id, chip2_id in bridge_connections:
        chip1 = layout[chip1_id]
        chip2 = layout[chip2_id]
        
        is_adj, overlap_len, direction = get_adjacency_info(chip1, chip2)
        
        if verbose:
            print(f"  {chip1_id}-{chip2_id}: 邻接={is_adj}, 重叠={overlap_len:.2f}")
        
        if not is_adj or overlap_len < MIN_OVERLAP:
            all_valid = False
    
    # 验证无重叠
    chip_ids = list(layout.keys())
    for i in range(len(chip_ids)):
        for j in range(i + 1, len(chip_ids)):
            chip1_id = chip_ids[i]
            chip2_id = chip_ids[j]
            chip1 = layout[chip1_id]
            chip2 = layout[chip2_id]
            
            x_overlap = min(chip1.x + chip1.width, chip2.x + chip2.width) - max(chip1.x, chip2.x)
            y_overlap = min(chip1.y + chip1.height, chip2.y + chip2.height) - max(chip1.y, chip2.y)
            
            if x_overlap > EPSILON and y_overlap > EPSILON:
                all_valid = False
                if verbose:
                    print(f"  ✗ {chip1_id}-{chip2_id} 重叠! (x:{x_overlap:.2f}, y:{y_overlap:.2f})")
    
    if not all_valid:
        if verbose:
            print("\n" + "=" * 70)
            print("✗ TCG合法化失败 - EMIB边无法全部合法化")
            print("=" * 70)
        return False, tcg, layout
    
    if verbose:
        print("\n" + "=" * 70)
        print("✓ TCG合法化成功")
        print("=" * 70)
    
    return True, tcg, layout


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    from chiplet_model import is_layout_valid
    
    # print("TCG合法化测试 - 简单迭代调整")
    # print("=" * 70)





    
    
    # # 测试1: 3芯片案例 (详细调试)
    # print("\n测试1: 3芯片案例 (详细调试)")
    # print("-" * 70)
    
    # problem1 = load_problem_from_json("../test_input/5core.json")
    # tcg1= generate_initial_TCG(problem1, seed=None)

    # # tcg1 =TCG(['A','B','C'])
    # # tcg1.add_horizontal_constraint('A','C')
    # # tcg1.add_vertical_constraint('B','A')
    # # tcg1.add_vertical_constraint('B','C')
     
    
    # result1 = legalize_tcg(tcg1, problem1, verbose=True)

    # success, legal_tcg1, legal_layout1 = result1
    # if success:
    #     is_valid_layout = is_layout_valid(legal_layout1, problem1, verbose=True)
    #     print(f"\n布局是否有效？: {'✓ 有效' if is_valid_layout else '✗ 无效'}")
    #     print(f"\n✓ 测试1通过")
    # else:
    #     print("\n✗ 测试1失败")
    # from unit import visualize_layout_with_bridges
    # visualize_layout_with_bridges(legal_layout1, problem1, "../output/test1_layout.png")



    
    # 测试2: 12芯片复杂拓扑 - 1000次尝试
    print("\n\n" + "=" * 70)
    print("测试2: 12芯片复杂拓扑 - 10000次随机尝试")
    print("=" * 70)
    
    problem2 = load_problem_from_json("../test_input/8core.json")
    
    max_attempts = 10000
    success_count = 0
    best_layout = None
    best_area = float('inf')
    best_utilization = 0.0
    
    print(f"\n开始生成{max_attempts}个随机TCG并尝试合法化...")
    print("=" * 70)
     
    for attempt in range(max_attempts):
        # 生成随机TCG
        tcg2 = generate_initial_TCG(problem2, seed=None)
        
        # 尝试合法化
        result2 = legalize_tcg(tcg2, problem2, verbose=False)
        success, legal_tcg2, legal_layout2 = result2
        
        # 验证布局
        if success:
            is_valid_layout = is_layout_valid(legal_layout2, problem2, verbose=False) and SiliconBridge_is_legal(legal_layout2, problem2, verbose=False)
            
            if is_valid_layout:
                success_count += 1
                
                # 计算面积
                x_coords = [chip.x for chip in legal_layout2.values()]
                y_coords = [chip.y for chip in legal_layout2.values()]
                x_max = max(chip.x + chip.width for chip in legal_layout2.values())
                y_max = max(chip.y + chip.height for chip in legal_layout2.values())
                width = x_max - min(x_coords)
                height = y_max - min(y_coords)
                area = width * height
                total_chip_area = sum(c.width * c.height for c in legal_layout2.values())
                utilization = total_chip_area / area * 100
                
                # 更新最佳布局
                if utilization > best_utilization:
                    best_utilization = utilization
                    best_layout = legal_layout2
                    best_utilization = utilization
                    print(f"  [{attempt+1}/{max_attempts}] ✓ 成功! 利用率={utilization:.2f}% (面积={area:.1f}, 宽={width:.1f}, 高={height:.1f}) [新最佳]")
                else:
                    print(f"  [{attempt+1}/{max_attempts}] ✓ 成功! 利用率={utilization:.2f}% (面积={area:.1f}, 宽={width:.1f}, 高={height:.1f})")
        
        # 每100次报告进度
        if (attempt + 1) % 100 == 0:
            print(f"\n进度: {attempt+1}/{max_attempts}, 成功率: {success_count}/{attempt+1} = {success_count/(attempt+1)*100:.1f}%")
            print("-" * 70)
    
    # 最终统计
    print("\n" + "=" * 70)
    print("最终统计")
    print("=" * 70)
    print(f"总尝试次数: {max_attempts}")
    print(f"成功次数: {success_count}")
    print(f"成功率: {success_count/max_attempts*100:.2f}%")
    
    if best_layout:
        print(f"\n最佳布局:")
        print(f"  利用率: {best_utilization:.2f}%")
        
        # 可视化最佳布局
        from unit import visualize_layout_with_bridges
        visualize_layout_with_bridges(best_layout, problem2, "../output/best_layout.png")
        save_layout_to_json(best_layout, "../output/best_layout.json")
        
        print(f"\n✓ 最佳布局已保存:")
        print(f"  - 可视化: ../output/best_layout.png")
        print(f"  - JSON: ../output/best_layout.json")
    else:
        print("\n✗ 未找到任何合法布局")
        print("建议: 增加尝试次数或调整问题约束")
        
