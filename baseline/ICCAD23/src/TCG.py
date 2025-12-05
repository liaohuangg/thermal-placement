"""
传递闭包图 (Transitive Closure Graph - TCG) 数据结构与几何转换器

TCG是一种用于芯片布局的拓扑表示方法，使用两个有向图来表示芯片之间的相对位置关系：
- Ch (水平图): 表示芯片的水平相对位置（左右关系）
- Cv (垂直图): 表示芯片的垂直相对位置（上下关系）

通过最长路径算法可以将TCG转换为具体的几何布局。
"""

import networkx as nx
import json
from typing import Dict, List, Tuple, Set, Optional
from chiplet_model import Chiplet, LayoutProblem
from chiplet_model import (
    Chiplet, LayoutProblem, is_layout_valid, 
    has_overlap, get_adjacency_info, MIN_OVERLAP
)






class TCG:
    """
    传递闭包图 (Transitive Closure Graph)
    
    使用两个有向无环图来表示芯片布局的拓扑关系：
    - Ch: 水平约束图，如果存在边 u->v，表示芯片u在芯片v的左边
    - Cv: 垂直约束图，如果存在边 u->v，表示芯片u在芯片v的下面
    
    Attributes:
        Ch (nx.DiGraph): 水平约束有向图
        Cv (nx.DiGraph): 垂直约束有向图
        chip_ids (List[str]): 所有芯片ID的列表
    """
    
    def __init__(self, chip_ids: List[str] = None):
        """
        初始化TCG
        
        Args:
            chip_ids: 芯片ID列表，如果提供则初始化图中的节点
        """
        self.Ch = nx.DiGraph()  # 水平约束图
        self.Cv = nx.DiGraph()  # 垂直约束图
        self.chip_ids = chip_ids if chip_ids else []
        
        # 如果提供了芯片ID，添加节点
        if chip_ids:
            for chip_id in chip_ids:
                self.Ch.add_node(chip_id)
                self.Cv.add_node(chip_id)
    
    def add_chip(self, chip_id: str) -> None:
        """
        添加一个芯片到TCG中
        
        Args:
            chip_id: 芯片的ID
        """
        if chip_id not in self.chip_ids:
            self.chip_ids.append(chip_id)
            self.Ch.add_node(chip_id)
            self.Cv.add_node(chip_id)
    
    def add_horizontal_constraint(self, left_chip: str, right_chip: str) -> None:
        """
        添加水平约束：left_chip 在 right_chip 的左边
        
        Args:
            left_chip: 左边的芯片ID
            right_chip: 右边的芯片ID
        """
        self.Ch.add_edge(left_chip, right_chip)
    
    def add_vertical_constraint(self, bottom_chip: str, top_chip: str) -> None:
        """
        添加垂直约束：bottom_chip 在 top_chip 的下面
        
        Args:
            bottom_chip: 下面的芯片ID
            top_chip: 上面的芯片ID
        """
        self.Cv.add_edge(bottom_chip, top_chip)
    
    def is_valid(self) -> Tuple[bool, str]:
        """
        检查TCG是否有效
        
        TCG的有效性要求：
        
        1. 无环性：Ch 和 Cv 都必须是有向无环图（DAG）
           - 不能有循环的相对位置关系（如 A在B左边，B在C左边，C又在A左边）
        
        2. 完备性：对于任意一对不同的芯片 (i, j)，它们之间必须有且仅有一个相对位置约束
           - 要么 i 在 j 的左边（Ch中有边 i->j）
           - 要么 j 在 i 的左边（Ch中有边 j->i）
           - 要么 i 在 j 的下方（Cv中有边 i->j）
           - 要么 j 在 i 的下方（Cv中有边 j->i）
           - 四种情况中必须恰好满足一种，不能是0种或多于1种
        
        特殊情况：
        - 如果所有约束都在Ch中，Cv为空，这个TCG是合法的（所有芯片按水平顺序排列）
        - 如果所有约束都在Cv中，Ch为空，这个TCG是合法的（所有芯片按垂直顺序排列）
        
        Returns:
            (is_valid, message): 是否有效以及说明信息
        """
        # 检查Ch是否有环
        if not nx.is_directed_acyclic_graph(self.Ch):
            return False, "水平约束图Ch包含环"
        
        # 检查Cv是否有环
        if not nx.is_directed_acyclic_graph(self.Cv):
            return False, "垂直约束图Cv包含环"
        
        # 检查TCG的完备性：对于任意两个不同的芯片，必须恰好有一个相对位置约束
        n = len(self.chip_ids)
        for i in range(n):
            for j in range(i + 1, n):
                chip_i = self.chip_ids[i]
                chip_j = self.chip_ids[j]
                
                # 统计Ch中的约束：i->j 或 j->i
                has_ch_edge_ij = self.Ch.has_edge(chip_i, chip_j)
                has_ch_edge_ji = self.Ch.has_edge(chip_j, chip_i)
                ch_constraint_count = sum([has_ch_edge_ij, has_ch_edge_ji])
                
                # 统计Cv中的约束：i->j 或 j->i
                has_cv_edge_ij = self.Cv.has_edge(chip_i, chip_j)
                has_cv_edge_ji = self.Cv.has_edge(chip_j, chip_i)
                cv_constraint_count = sum([has_cv_edge_ij, has_cv_edge_ji])
                
                # 总约束数
                total_constraints = ch_constraint_count + cv_constraint_count
                
                # 必须恰好有一个约束
                if total_constraints == 0:
                    return False, (f"芯片对 ({chip_i}, {chip_j}) 缺少相对位置约束：\n"
                                 f"  Ch中无边 {chip_i}->{chip_j} 或 {chip_j}->{chip_i}\n"
                                 f"  Cv中无边 {chip_i}->{chip_j} 或 {chip_j}->{chip_i}")
                
                if total_constraints > 1:
                    edges_desc = []
                    if has_ch_edge_ij:
                        edges_desc.append(f"Ch: {chip_i}->{chip_j}")
                    if has_ch_edge_ji:
                        edges_desc.append(f"Ch: {chip_j}->{chip_i}")
                    if has_cv_edge_ij:
                        edges_desc.append(f"Cv: {chip_i}->{chip_j}")
                    if has_cv_edge_ji:
                        edges_desc.append(f"Cv: {chip_j}->{chip_i}")
                    
                    return False, (f"芯片对 ({chip_i}, {chip_j}) 过度约束，有 {total_constraints} 条边：\n"
                                 f"  {', '.join(edges_desc)}")
        
        return True, "TCG有效"
    
    def get_sources(self, graph: nx.DiGraph) -> List[str]:
        """
        获取图中的源节点（没有入边的节点）
        
        Args:
            graph: 有向图
            
        Returns:
            源节点列表
        """
        return [node for node in graph.nodes() if graph.in_degree(node) == 0]
    
    def __repr__(self) -> str:
        """返回TCG的字符串表示"""
        return (f"TCG(chips={len(self.chip_ids)}, "
                f"h_edges={self.Ch.number_of_edges()}, "
                f"v_edges={self.Cv.number_of_edges()})")


def compute_longest_path_lengths(graph: nx.DiGraph, problem: LayoutProblem, 
                                  dimension: str = 'width') -> Dict[str, float]:
    """
    计算从源节点到每个节点的最长路径长度（以芯片尺寸累加）
    
    使用拓扑排序和动态规划计算最长路径。
    
    Args:
        graph: 有向无环图（Ch或Cv）
        problem: 布局问题，包含芯片尺寸信息
        dimension: 使用的尺寸维度，'width'用于水平，'height'用于垂直
        
    Returns:
        字典，键为芯片ID，值为该芯片的坐标（从源点的最长路径长度）
    """
    # 初始化所有节点的距离为0
    distances = {node: 0.0 for node in graph.nodes()}
    
    # 拓扑排序
    try:
        topo_order = list(nx.topological_sort(graph))
    except nx.NetworkXError:
        # 如果图有环，抛出错误
        raise ValueError("图包含环，无法计算最长路径")
    
    # 按拓扑顺序处理每个节点
    for node in topo_order:
        # 获取当前节点的芯片
        chip = problem.get_chiplet(node)
        if chip is None:
            raise ValueError(f"芯片 {node} 在问题中不存在")
        
        # 获取芯片的尺寸（宽度或高度）
        size = chip.width if dimension == 'width' else chip.height
        
        # 更新所有后继节点的距离
        for successor in graph.successors(node):
            # 到达successor的路径长度 = 到达node的路径长度 + node的尺寸
            new_distance = distances[node] + size
            # 取最长路径
            if new_distance > distances[successor]:
                distances[successor] = new_distance
    
    return distances


def generate_layout_from_tcg(tcg: TCG, problem: LayoutProblem) -> Dict[str, Chiplet]:
    """
    从TCG生成几何布局
    
    使用最长路径算法计算每个芯片的x和y坐标：
    - x坐标：在Ch图上从源节点到该节点的最长路径（累加宽度）
    - y坐标：在Cv图上从源节点到该节点的最长路径（累加高度）
    
    Args:
        tcg: 传递闭包图
        problem: 布局问题，包含芯片尺寸信息
        
    Returns:
        布局字典 {chip_id: chip_object}，每个芯片的x和y坐标已设置
        
    Raises:
        ValueError: 如果TCG无效（包含环）或芯片信息不匹配
    """
    # 验证TCG是否有效
    is_valid, message = tcg.is_valid()
    if not is_valid:
        raise ValueError(f"TCG无效: {message}")
    
    # 计算x坐标（基于Ch图和芯片宽度）
    x_coordinates = compute_longest_path_lengths(tcg.Ch, problem, dimension='width')
    
    # 计算y坐标（基于Cv图和芯片高度）
    y_coordinates = compute_longest_path_lengths(tcg.Cv, problem, dimension='height')
    
    # 创建布局字典
    layout = {}
    
    for chip_id in tcg.chip_ids:
        # 获取原始芯片对象
        original_chip = problem.get_chiplet(chip_id)
        if original_chip is None:
            raise ValueError(f"芯片 {chip_id} 在问题中不存在")
        
        # 创建新的芯片对象，设置计算出的坐标
        chip = Chiplet(
            chip_id=original_chip.id,
            width=original_chip.width,
            height=original_chip.height,
            x=x_coordinates[chip_id],
            y=y_coordinates[chip_id]
        )
        
        layout[chip_id] = chip
    
    return layout


def get_layout_bounds(layout: Dict[str, Chiplet]) -> Tuple[float, float, float, float]:
    """
    获取布局的边界框
    
    Args:
        layout: 布局字典
        
    Returns:
        (x_min, y_min, x_max, y_max): 布局的边界坐标
    """
    if not layout:
        return (0, 0, 0, 0)
    
    x_min = min(chip.x for chip in layout.values())
    y_min = min(chip.y for chip in layout.values())
    x_max = max(chip.x + chip.width for chip in layout.values())
    y_max = max(chip.y + chip.height for chip in layout.values())
    
    return (x_min, y_min, x_max, y_max)


def get_layout_area(layout: Dict[str, Chiplet]) -> float:
    """
    计算布局的总面积（边界框面积）
    
    Args:
        layout: 布局字典
        
    Returns:
        布局面积
    """
    x_min, y_min, x_max, y_max = get_layout_bounds(layout)
    width = x_max - x_min
    height = y_max - y_min
    return width * height


def print_layout_info(layout: Dict[str, Chiplet], title: str = "布局信息") -> None:
    """
    打印布局信息
    
    Args:
        layout: 布局字典
        title: 标题
    """
    print(f"\n{title}")
    print("=" * 60)
    
    print("\n芯片位置:")
    for chip_id, chip in sorted(layout.items()):
        bounds = chip.get_bounds()
        print(f"  {chip_id}: 位置({chip.x:.1f}, {chip.y:.1f}), "
              f"大小({chip.width}x{chip.height}), "
              f"边界{bounds}")
    
    x_min, y_min, x_max, y_max = get_layout_bounds(layout)
    width = x_max - x_min
    height = y_max - y_min
    area = width * height
    
    print(f"\n布局统计:")
    print(f"  边界框: ({x_min:.1f}, {y_min:.1f}) 到 ({x_max:.1f}, {y_max:.1f})")
    print(f"  宽度: {width:.1f}")
    print(f"  高度: {height:.1f}")
    print(f"  面积: {area:.1f}")


if __name__ == "__main__":
    #核心作用：根据TCG生成布局

#第一步：先手动创建一个简单的问题和TCG

    # 简单示例
    print("TCG 核心数据结构与几何转换器")
    print("=" * 60)
    
    # 创建一个简单的问题
    from chiplet_model import Chiplet, LayoutProblem
    
    problem = LayoutProblem()
    
    # 添加芯片
    chips = [
        Chiplet("A", 10, 20),
        Chiplet("B", 11, 10),
        Chiplet("C", 10, 15),
        Chiplet("D", 10, 10),
    ]
    
    for chip in chips:
            problem.add_chiplet(chip)
                
    print(problem.get_chiplet(chip_id="B"))
     #注意：添加连接要求
    problem.add_connection("A", "B")
    problem.add_connection("B", "D")
    # problem.add_connection("C", "D")
    # problem.add_connection("A", "D")    
    problem.add_connection("A", "C")        
        
        # 创建TCG
    tcg = TCG(["A", "B", "C", "D"])
    
    # 添加约束
    # A在B的左边，B在C的左边
    tcg.add_horizontal_constraint("B", "D")
    tcg.add_horizontal_constraint("A", "B")
    tcg.add_horizontal_constraint("A", "C") 
    tcg.add_horizontal_constraint("A", "D")
    
    # A在C的下面
    tcg.add_vertical_constraint("D", "C")
    tcg.add_vertical_constraint("B", "C")
 
     
 #第2步：根据TCG生成布局   
    print(f"\n创建的TCG: {tcg}")
    print(f"  水平约束 (Ch): {list(tcg.Ch.edges())}")
    print(f"  垂直约束 (Cv): {list(tcg.Cv.edges())}")
    
    # 验证TCG
    is_valid, message = tcg.is_valid()
    print(f"\nTCG有效性: {is_valid} - {message}")
    
    # 生成布局
    print("\n生成几何布局...")
    layout = generate_layout_from_tcg(tcg, problem)
    print_layout_info(layout, "从TCG生成的布局")
    
    # 打印连接关系
    # print("\n" + "=" * 60)
    # print("连接关系 (来自 problem.connection_graph)")
    # print("=" * 60)
    # print(f"\n总连接数: {problem.connection_graph.number_of_edges()}")
    
    # if problem.connection_graph.number_of_edges() > 0:
    #     print("\n所有连接:")
    #     for edge in problem.connection_graph.edges(data=True):
    #         chip1, chip2, data = edge
    #         weight = data.get('weight', 1.0)
    #         print(f"  {chip1} <-> {chip2}: weight={weight}")
        
    #     print("\n每个芯片的连接:")
    #     for chip_id in sorted(problem.chiplets.keys()):
    #         neighbors = problem.get_neighbors(chip_id)
    #         print(f"  {chip_id}: 连接到 {neighbors}")
        
    #     print("\n连接的物理状态:")
    #     for edge in problem.connection_graph.edges():
    #         chip1_id, chip2_id = edge
    #         chip1 = layout[chip1_id]
    #         chip2 = layout[chip2_id]
            
    #         is_adj, overlap_len, direction = get_adjacency_info(chip1, chip2)
    #         status = "✓ 邻接" if is_adj else "✗ 不邻接"
            
    #         print(f"  {chip1_id} - {chip2_id}: {status}", end="")
    #         if is_adj:
    #             print(f" (方向={direction}, 共享长度={overlap_len:.1f})")
    #         else:
    #             print(f" (间隙存在)")
    # else:
    #     print("\n  (无连接要求)")
    
    print("\n" + "😊"*30)
    
    is_valid_layout = is_layout_valid(layout, problem, verbose=True)
    print(f"\n布局是否有效？: {'✓ 有效' if is_valid_layout else '✗ 无效'}")

#第三步：保存生成的布局到layout.json文件
    # 将布局保存到 layout.json
    print("\n" + "=" * 60)
    print("保存布局到 layout.json...")
    print("=" * 60)
    
    # 构建JSON数据
    layout_data = {
        "chiplets": []
    }
    
    for chip_id, chip in layout.items():
        chiplet_data = {
            "id": chip.id,
            "width": chip.width,
            "height": chip.height,
            "x": chip.x,
            "y": chip.y
        }
        layout_data["chiplets"].append(chiplet_data)
    
    # 写入JSON文件
    with open('layout.json', 'w', encoding='utf-8') as f:
        json.dump(layout_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ 布局已保存到 layout.json")
    print(f"  - 芯片数量: {len(layout)}")
    print(f"  - 布局面积: {get_layout_area(layout):.1f}")




#从test_complix.json加载问题，并生成TCG和布局
    print("\n" + "=" * 60)
    print("从 test_complex.json 加载问题并生成TCG和布局...")
    print("=" * 60)
    #todo
    

   
