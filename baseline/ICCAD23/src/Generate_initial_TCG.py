"""
初始TCG生成器

将无向连接图转换为初始的TCG（传递闭包图）和对应的几何布局。
使用最大生成树（MST）作为骨架，随机分配边到水平/垂直约束图中。
"""
import json
import networkx as nx
import random
from typing import Dict, Tuple, List, Set
from chiplet_model import Chiplet, LayoutProblem
from TCG import TCG, generate_layout_from_tcg


def generate_initial_TCG(problem: LayoutProblem, seed: int = None) -> TCG:
    """
    从问题的无向连接图生成初始的TCG候选解
    
    算法步骤：
    1. 计算最大生成树（MST）- 使用边的weight作为权重
    2. 选择根节点并为MST边定向
    3. 构建TCG.Ch（水平约束图）- 基于MST的传递闭包
    4. 构建TCG.Cv（垂直约束图）- 补充Ch中没有的节点对约束
    
    Args:
        problem: 布局问题对象，包含芯片和无向连接图
        seed: 随机种子，用于可重复性（可选）
        
    Returns:
        tcg: 生成的TCG对象
        
    Raises:
        ValueError: 如果问题中没有芯片或连接图为空
    """
    if seed is not None:
        random.seed(seed)
    
    # 验证输入
    if len(problem.chiplets) == 0:
        raise ValueError("问题中没有芯片")
    
    if problem.connection_graph.number_of_edges() == 0:
        raise ValueError("连接图为空，无法生成MST")
    
    # 步骤1: 计算最大生成树（MST）
    # NetworkX的minimum_spanning_tree使用权重的负值来找最大生成树
    mst = _compute_maximum_spanning_tree(problem.connection_graph)
    
    # 步骤2: 选择根节点并为MST定向
    root = random.choice(list(problem.chiplets.keys()))
    # print(f"\n当前选择的MST根节点: {root}")
    directed_mst_edges = _orient_tree_from_root(mst, root)
    # print(f"定向后的MST边: {directed_mst_edges}")
    
    # 步骤3: 创建初始TCG
    chip_ids = list(problem.chiplets.keys())
    tcg = TCG(chip_ids)

    # 步骤4：构建TCG.Ch
    # print("构建TCG.Ch边集...")
    ch_edges = creat_tcg_ch(directed_mst_edges)
    tcg.Ch.add_edges_from(ch_edges)  # 使用add_edges_from方法添加边
    # print(f"当前构建的(Ch): {list(tcg.Ch.edges())}")
   
    # 记录已经在Ch中有路径的节点对
    ch_reachable = _build_reachability_set(directed_mst_edges, is_mst=True)
    # print(f"Ch中可达节点对: {ch_reachable}")

    # 步骤5：构建TCG.Cv
    # print("\n构建TCG.Cv边集...")
    cv_edges = creat_tcg_cv(chip_ids, tcg.Ch)
    tcg.Cv.add_edges_from(cv_edges)
    # print(f"当前构建的(Cv): {list(tcg.Cv.edges())}")
    
    return tcg


def _compute_maximum_spanning_tree(graph: nx.Graph) -> nx.Graph:
    """
    计算无向图的最大生成树
    
    使用边的weight作为权重，weight越大的边越优先选择。
    
    Args:
        graph: 无向图，边应包含'weight'属性
        
    Returns:
        最大生成树（nx.Graph对象）
    """
    # 创建一个副本，将权重取负
    weighted_graph = nx.Graph()
    for u, v, data in graph.edges(data=True):
        weight = data.get('weight', 1.0)
        weighted_graph.add_edge(u, v, weight=-weight)  # 取负值
    
    # 使用最小生成树算法（实际上找的是最大生成树）
    mst = nx.minimum_spanning_tree(weighted_graph)
    
    return mst


def _orient_tree_from_root(tree: nx.Graph, root: str) -> List[Tuple[str, str]]:
    """
    从根节点出发，为树的所有边定向（远离根的方向）
    
    使用BFS遍历树，确保所有边都指向远离根的方向。
    
    Args:
        tree: 无向树
        root: 根节点
        
    Returns:
        有向边列表 [(source, target), ...]，所有边都指向远离根的方向
    """
    directed_edges = []
    visited = {root}
    queue = [root]
    
    while queue:
        current = queue.pop(0)
        
        for neighbor in tree.neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                directed_edges.append((current, neighbor))  # 从current指向neighbor
                queue.append(neighbor)
    
    return directed_edges


def _build_reachability_set(edges: List[Tuple[str, str]], is_mst: bool = True) -> Set[Tuple[str, str]]:
    """
    构建可达性集合（用于记录哪些节点对在某个图中有路径）
    
    Args:
        edges: 有向边列表
        is_mst: 是否是MST边（当前未使用，保留用于扩展）
        
    Returns:
        可达节点对的集合
    """
    reachable = set()
    for source, target in edges:
        reachable.add((source, target))
    return reachable


def _has_path_in_graph(graph: nx.DiGraph, source: str, target: str) -> bool:
    """
    检查有向图中是否存在从source到target的路径(单向即可)
    
    Args:
        graph: 有向图
        source: 起始节点
        target: 目标节点
        
    Returns:
        如果存在路径返回True，否则返回False
    """
    try:
        return nx.has_path(graph, source, target)
    except nx.NodeNotFound:
        return False


def print_generation_info(problem: LayoutProblem, tcg: TCG) -> None:
    """
    打印初始TCG生成信息
    
    Args:
        problem: 原始问题
        tcg: 生成的TCG
    """
    print("\n初始TCG生成信息")
    print("=" * 60)
    print(f"\n创建的TCG: {tcg}")
    print(f"  水平约束 (Ch): {list(tcg.Ch.edges())}")
    print(f"  垂直约束 (Cv): {list(tcg.Cv.edges())}")

    print(f"\n问题规模:")
    print(f"  芯片数量: {len(problem.chiplets)}")
    print(f"  连接数量: {problem.connection_graph.number_of_edges()}")
    
    print(f"\nTCG结构:")
    print(f"  水平约束数: {tcg.Ch.number_of_edges()}")
    print(f"  垂直约束数: {tcg.Cv.number_of_edges()}")
    
    is_valid, message = tcg.is_valid()
    print(f"\nTCG有效性: {is_valid} - {message}")


def creat_tcg_ch(directed_mst_edges: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """
    从定向后的MST边创建传递闭包边集
    
    添加传递约束边以保持TCG的传递属性：
    对于每个ni和nj，如果ni是nj的祖先，则添加约束边ni → nj
    
    Args:
        directed_mst_edges: 已定向的MST边列表 [(source, target), ...]
        
    Returns:
        包含所有MST边和传递闭包边的有向边集合
    """
    # 首先包含所有原始MST边
    all_edges = set(directed_mst_edges)
    
    # 构建一个有向图来计算祖先关系
    graph = nx.DiGraph()
    graph.add_edges_from(directed_mst_edges)
    
    # 获取所有节点
    all_nodes = list(graph.nodes())
    
    # 对于每对节点(ni, nj)，如果ni是nj的祖先，添加边ni → nj
    for ni in all_nodes:
        for nj in all_nodes:
            if ni != nj:
                # 检查ni是否是nj的祖先（即从ni到nj存在路径）
                if nx.has_path(graph, ni, nj):
                    # 添加传递闭包边
                    all_edges.add((ni, nj))
    
    return list(all_edges)


def creat_tcg_cv(chip_ids: List[str], ch_graph: nx.DiGraph) -> List[Tuple[str, str]]:
    """
    构建TCG.Cv的边集
    
    选择所有在Ch中没有约束关系的节点对，随机分配方向后添加到Cv。
    不添加传递闭包，保持稀疏性。添加边时检查是否会产生环，如果会则尝试反向。
    
    Args:
        chip_ids: 所有芯片ID列表
        ch_graph: 已构建的Ch图（水平约束图）
        
    Returns:
        Cv的有向边列表（不含传递闭包）
    """
    # 第一步：选择Ch中没有约束的节点对
    unconstrained_pairs = []
    
    for i, chip1 in enumerate(chip_ids):
        for j, chip2 in enumerate(chip_ids):
            if i < j:  # 只考虑每对节点一次
                # 检查Ch中是否已有这对节点的约束关系（任意方向）
                has_ch_constraint = (
                    _has_path_in_graph(ch_graph, chip1, chip2) or
                    _has_path_in_graph(ch_graph, chip2, chip1)
                )
                
                # 如果Ch中没有约束，则记录这对节点
                if not has_ch_constraint:
                    unconstrained_pairs.append((chip1, chip2))
    
    # 第二步：为未约束的节点对随机分配方向，同时避免产生环
    # 随机打乱节点对的顺序
    random.shuffle(unconstrained_pairs)
    
    cv_graph = nx.DiGraph()
    cv_edges = []
    
    for chip1, chip2 in unconstrained_pairs:
        # 随机选择方向
        if random.random() < 0.5:
            edge = (chip1, chip2)
        else:
            edge = (chip2, chip1)
        
        # 尝试添加这条边，检查是否会产生环
        cv_graph.add_edge(edge[0], edge[1])
        
        if not nx.is_directed_acyclic_graph(cv_graph):
            # 产生环了，尝试反向
            cv_graph.remove_edge(edge[0], edge[1])
            reverse_edge = (edge[1], edge[0])
            cv_graph.add_edge(reverse_edge[0], reverse_edge[1])
            
            if not nx.is_directed_acyclic_graph(cv_graph):
                # 反向也产生环，放弃这条边
                cv_graph.remove_edge(reverse_edge[0], reverse_edge[1])
            else:
                # 反向可行
                cv_edges.append(reverse_edge)
        else:
            # 原方向可行
            cv_edges.append(edge)
    
    return cv_edges
    

if __name__ == "__main__":
    # 测试示例
    # print("初始TCG生成器测试")
    # print("=" * 60)
    # from unit import load_problem_from_json, save_layout_to_json
    # # 创建一个简单的测试问题
    # problem =  load_problem_from_json("../test_input/12core.json")
    
    # # # 添加芯片
    # # chips = [
    # #     Chiplet("A", 15, 10),
    # #     Chiplet("B", 6, 10),
    # #     Chiplet("C", 10, 4),
    # #     Chiplet("D", 3, 3),
    # # ]
    
    # # for chip in chips:
    # #     problem.add_chiplet(chip)
    
    # # # 添加带权重的连接
    # # problem.add_connection("A", "B", weight=5.0)
    # # problem.add_connection("B", "D", weight=4.0)
    # # problem.add_connection("C", "D", weight=2.0)
    # # # problem.add_connection("A", "D", weight=1.0)
    # # problem.add_connection("A", "C", weight=3.0)
    
    # print(f"\n创建的问题: {problem}")
    # print(f"连接关系:")
    # for u, v, data in problem.connection_graph.edges(data=True):
    #     print(f"  {u} - {v}: weight={data.get('weight', 1.0)}")
    
    # # 生成初始候选解
    # print("\n" + "=" * 60)
    # print("生成初始TCG...")
    # print("=" * 60)
    
    # try:
    #     # 生成TCG
    #     tcg = generate_initial_TCG(problem, seed=None)
    #     print_generation_info(problem, tcg)
        
    #     # 使用TCG.py中的函数生成布局
    #     print("\n" + "=" * 60)
    #     print("从TCG生成几何布局...")
    #     print("=" * 60)
        
    #     layout = generate_layout_from_tcg(tcg, problem)
        
    #     # 打印布局详情
    #     from TCG import print_layout_info, get_layout_area
    #     print_layout_info(layout, "生成的初始布局")
        
    #     # 验证布局
    #     from chiplet_model import is_layout_valid
    #     is_valid = is_layout_valid(layout, problem, verbose=True)
    #     print(f"\n最终布局有效性: {'✓ 有效' if is_valid else '✗ 无效'}")
    #     print(f"布局面积: {get_layout_area(layout):.1f}")
    #     from unit import visualize_layout_with_bridges
    #     visualize_layout_with_bridges(layout, problem, "initial_layout.png", show_bridges=True, show_coordinates=True)
        
    # except ValueError as e:
    #     print(f"\n生成失败: {e}")

    print("\n\n" + "=" * 70)
    print("测试2: 12芯片复杂拓扑 - 1000次随机尝试")
    print("=" * 70)
    from unit import load_problem_from_json, save_layout_to_json
    from chiplet_model import is_layout_valid
    problem2 = load_problem_from_json("../test_input/8core.json")
    
    max_attempts = 10000
    success_count = 0
    best_layout = None
    best_area = float('inf')
    
    print(f"\n开始生成{max_attempts}个随机TCG并尝试合法化...")
    print("=" * 70)
     
    for attempt in range(max_attempts):
        # 生成随机TCG
        tcg2 = generate_initial_TCG(problem2, seed=None)
        layout = generate_layout_from_tcg(tcg2, problem2)
        is_valid_layout = is_layout_valid(layout, problem2, verbose=False)
        
        
        
        # 验证布局
        if is_valid_layout:
            success_count += 1
            
                # 计算面积
            x_coords = [chip.x for chip in layout.values()]
            y_coords = [chip.y for chip in layout.values()]
            x_max = max(chip.x + chip.width for chip in layout.values())
            y_max = max(chip.y + chip.height for chip in layout.values())
            width = x_max - min(x_coords)
            height = y_max - min(y_coords)
            area = width * height
                
                # 更新最佳布局
            if area < best_area:
                    best_area = area
                    best_layout = layout
                    print(f"  [{attempt+1}/{max_attempts}] ✓ 成功! 面积={area:.1f} (宽={width:.1f}, 高={height:.1f}) [新最佳]")
            else:
                    print(f"  [{attempt+1}/{max_attempts}] ✓ 成功! 面积={area:.1f} (宽={width:.1f}, 高={height:.1f})")
        
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
        print(f"  面积: {best_area:.1f}")
        
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
        



 

