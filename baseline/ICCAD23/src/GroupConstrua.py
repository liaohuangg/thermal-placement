"""
相似度树（Similarity Tree）构建模块

相似度树是一个N叉树结构，用于组织和管理多个布局解：
- 每个节点代表一个合法的布局解（TCG + Layout）
- 节点之间通过相似度度量建立父子关系
- 相似度计算基于芯片的相对位置关系（四个方向的最近邻居）

主要功能：
1. 根据SA得到的合法解构建相似度树
2. 计算布局解之间的相似度（SD值，范围0-1）
3. 基于相似度树进行进一步的优化搜索
"""

import copy
from typing import Dict, List, Tuple, Optional, Set
from TCG import TCG
from chiplet_model import Chiplet, LayoutProblem, EPSILON


class SimilarityTreeNode:
    """
    相似度树节点
    
    每个节点代表一个合法的布局解。
    
    Attributes:
        tcg (TCG): 该节点对应的TCG拓扑
        layout (Dict[str, Chiplet]): 该节点对应的几何布局
        parent (SimilarityTreeNode): 父节点（可为None表示根节点）
        children (List[SimilarityTreeNode]): 子节点列表
        similarity_to_parent (float): 与父节点的相似度（根节点为None）
        cost (float): 布局的评价成本（如面积利用率、线长等）
        node_id (int): 节点唯一标识
    """
    
    _node_counter = 0  # 类变量，用于生成唯一ID
    
    def __init__(self, tcg: TCG, layout: Dict[str, Chiplet], 
                 parent: Optional['SimilarityTreeNode'] = None,
                 similarity_to_parent: Optional[float] = None,
                 cost: float = 0.0):
        """
        初始化相似度树节点
        
        Args:
            tcg: TCG拓扑
            layout: 几何布局
            parent: 父节点（根节点为None）
            similarity_to_parent: 与父节点的相似度（根节点为None）
            cost: 布局评价成本
        """
        self.tcg = copy.deepcopy(tcg)
        self.layout = copy.deepcopy(layout)
        self.parent = parent
        self.children: List[SimilarityTreeNode] = []
        self.similarity_to_parent = similarity_to_parent
        self.cost = cost
        
        # 分配唯一ID
        SimilarityTreeNode._node_counter += 1
        self.node_id = SimilarityTreeNode._node_counter
    
    def add_child(self, child: 'SimilarityTreeNode') -> None:
        """
        添加子节点
        
        Args:
            child: 子节点
        """
        self.children.append(child)
        child.parent = self
    
    def is_root(self) -> bool:
        """判断是否为根节点"""
        return self.parent is None
    
    def is_leaf(self) -> bool:
        """判断是否为叶节点"""
        return len(self.children) == 0
    
    def get_depth(self) -> int:
        """获取节点深度（根节点深度为0）"""
        depth = 0
        node = self
        while node.parent is not None:
            depth += 1
            node = node.parent
        return depth
    
    def __repr__(self) -> str:
        return (f"TreeNode(id={self.node_id}, depth={self.get_depth()}, "
                f"children={len(self.children)}, cost={self.cost:.4f})")


class SimilarityTree:
    """
    相似度树
    
    管理整个相似度树结构，提供树的构建和查询功能。
    
    Attributes:
        root (SimilarityTreeNode): 树的根节点
        all_nodes (List[SimilarityTreeNode]): 所有节点的列表
    """
    
    def __init__(self, root: SimilarityTreeNode):
        """
        初始化相似度树
        
        Args:
            root: 根节点
        """
        self.root = root
        self.all_nodes: List[SimilarityTreeNode] = [root]
    
    def add_node(self, parent: SimilarityTreeNode, child: SimilarityTreeNode) -> None:
        """
        添加节点到树中
        
        Args:
            parent: 父节点
            child: 要添加的子节点
        """
        parent.add_child(child)
        self.all_nodes.append(child)
    
    def get_all_leaves(self) -> List[SimilarityTreeNode]:
        """获取所有叶节点"""
        return [node for node in self.all_nodes if node.is_leaf()]
    
    def get_nodes_at_depth(self, depth: int) -> List[SimilarityTreeNode]:
        """
        获取指定深度的所有节点
        
        Args:
            depth: 深度（根节点深度为0）
            
        Returns:
            该深度的所有节点列表
        """
        return [node for node in self.all_nodes if node.get_depth() == depth]
    
    def get_tree_statistics(self) -> Dict:
        """
        获取树的统计信息
        
        Returns:
            包含节点数、深度、叶节点数等信息的字典
        """
        max_depth = max(node.get_depth() for node in self.all_nodes) if self.all_nodes else 0
        num_leaves = len(self.get_all_leaves())
        
        return {
            'total_nodes': len(self.all_nodes),
            'max_depth': max_depth,
            'num_leaves': num_leaves,
            'avg_children': sum(len(node.children) for node in self.all_nodes) / len(self.all_nodes)
        }
    
    def __repr__(self) -> str:
        stats = self.get_tree_statistics()
        return (f"SimilarityTree(nodes={stats['total_nodes']}, "
                f"depth={stats['max_depth']}, leaves={stats['num_leaves']})")


def get_nearest_neighbors(chip_id: str, layout: Dict[str, Chiplet]) -> Dict[str, Optional[str]]:
    """
    获取指定芯片在四个方向上的最近邻居
    
    四个方向：left, right, top, bottom
    最近邻居定义：在某个方向上，与该芯片中心点距离最近的芯片
    
    Args:
        chip_id: 目标芯片ID
        layout: 布局字典
        
    Returns:
        字典 {'left': chip_id, 'right': chip_id, 'top': chip_id, 'bottom': chip_id}
        如果某个方向没有邻居，则值为None
    """
    target_chip = layout[chip_id]
    target_cx = target_chip.x + target_chip.width / 2
    target_cy = target_chip.y + target_chip.height / 2
    
    neighbors = {
        'left': None,
        'right': None,
        'top': None,
        'bottom': None
    }
    
    distances = {
        'left': float('inf'),
        'right': float('inf'),
        'top': float('inf'),
        'bottom': float('inf')
    }
    
    # 遍历所有其他芯片
    for other_id, other_chip in layout.items():
        if other_id == chip_id:
            continue
        
        other_cx = other_chip.x + other_chip.width / 2
        other_cy = other_chip.y + other_chip.height / 2
        
        # 判断方向并计算距离
        # Left: other在target左边（other_cx < target_cx）
        if other_cx < target_cx - EPSILON:
            dist = abs(target_cx - other_cx)
            if dist < distances['left']:
                distances['left'] = dist
                neighbors['left'] = other_id
        
        # Right: other在target右边（other_cx > target_cx）
        if other_cx > target_cx + EPSILON:
            dist = abs(other_cx - target_cx)
            if dist < distances['right']:
                distances['right'] = dist
                neighbors['right'] = other_id
        
        # Bottom: other在target下方（other_cy < target_cy）
        if other_cy < target_cy - EPSILON:
            dist = abs(target_cy - other_cy)
            if dist < distances['bottom']:
                distances['bottom'] = dist
                neighbors['bottom'] = other_id
        
        # Top: other在target上方（other_cy > target_cy）
        if other_cy > target_cy + EPSILON:
            dist = abs(other_cy - target_cy)
            if dist < distances['top']:
                distances['top'] = dist
                neighbors['top'] = other_id
    
    return neighbors


def compute_similarity(layout1: Dict[str, Chiplet], 
                       layout2: Dict[str, Chiplet]) -> float:
    """
    计算两个布局之间的相似度 SD(σ_i, T_j)
    
    相似度计算方法：
    1. 对每个芯片，找出其在四个方向上的最近邻居
    2. 比较两个布局中，每个芯片的邻居是否相同
    3. 相似度 = 匹配的邻居关系数 / 总的邻居关系数
    
    SD值范围：[0, 1]
    - 1: 完全相同（所有芯片的邻居关系都一致）
    - 0: 完全不同（没有任何邻居关系匹配）
    
    Args:
        layout1: 第一个布局
        layout2: 第二个布局
        
    Returns:
        相似度值（0到1之间）
        
    Raises:
        ValueError: 如果两个布局包含的芯片不一致
    """
    # 检查两个布局是否包含相同的芯片
    if set(layout1.keys()) != set(layout2.keys()):
        raise ValueError("两个布局包含的芯片不一致")
    
    chip_ids = list(layout1.keys())
    
    if len(chip_ids) == 0:
        return 1.0  # 空布局认为完全相似
    
    # 统计匹配的邻居关系数
    total_relations = 0  # 总的邻居关系数
    matched_relations = 0  # 匹配的邻居关系数
    
    for chip_id in chip_ids:
        # 获取该芯片在两个布局中的邻居
        neighbors1 = get_nearest_neighbors(chip_id, layout1)
        neighbors2 = get_nearest_neighbors(chip_id, layout2)
        
        # 比较四个方向的邻居
        for direction in ['left', 'right', 'top', 'bottom']:
            # 只统计至少有一个布局中该方向存在邻居的情况
            if neighbors1[direction] is not None or neighbors2[direction] is not None:
                total_relations += 1
                
                # 如果两个布局中该方向的邻居相同，则匹配
                if neighbors1[direction] == neighbors2[direction]:
                    matched_relations += 1
    
    # 计算相似度
    if total_relations == 0:
        return 1.0  # 如果没有邻居关系（只有一个芯片），认为完全相似
    
    similarity = matched_relations / total_relations
    
    return similarity


def build_similarity_tree(legal_tcgs: List[TCG], 
                          legal_layouts: List[Dict[str, Chiplet]],
                          costs: List[float]) -> SimilarityTree:
    """
    从合法解集合构建相似度树
    
    构建策略：
    1. 选择成本最优的解作为根节点
    2. 其他解按照与根节点的相似度排序
    3. 为每个解找到最相似的已有节点作为父节点
    
    Args:
        legal_tcgs: 合法TCG列表
        legal_layouts: 合法布局列表
        costs: 每个布局的成本列表
        
    Returns:
        构建好的相似度树
        
    Raises:
        ValueError: 如果输入列表为空或长度不一致
    """
    if len(legal_tcgs) == 0:
        raise ValueError("合法解列表为空")
    
    if not (len(legal_tcgs) == len(legal_layouts) == len(costs)):
        raise ValueError("TCG、布局和成本列表长度不一致")
    
    # 步骤1：选择成本最优的解作为根节点
    best_idx = min(range(len(costs)), key=lambda i: costs[i])
    root = SimilarityTreeNode(
        tcg=legal_tcgs[best_idx],
        layout=legal_layouts[best_idx],
        parent=None,
        similarity_to_parent=None,
        cost=costs[best_idx]
    )
    
    tree = SimilarityTree(root)
    
    # 步骤2：处理其他解
    remaining_indices = [i for i in range(len(legal_tcgs)) if i != best_idx]
    
    for idx in remaining_indices:
        # 为当前解找到树中最相似的节点作为父节点
        max_similarity = -1.0
        best_parent = None
        
        for node in tree.all_nodes:
            similarity = compute_similarity(legal_layouts[idx], node.layout)
            if similarity > max_similarity:
                max_similarity = similarity
                best_parent = node
        
        # 创建新节点并添加到树中
        new_node = SimilarityTreeNode(
            tcg=legal_tcgs[idx],
            layout=legal_layouts[idx],
            parent=best_parent,
            similarity_to_parent=max_similarity,
            cost=costs[idx]
        )
        
        tree.add_node(best_parent, new_node)
    
    return tree


def compute_neighbor_cost(base_layout: Dict[str, Chiplet],
                         neighbor_layout: Dict[str, Chiplet],
                         alpha_x: float = 1.0,
                         beta_y: float = 1.0,
                         gamma_s: float = 1.0,
                         expected_similarity: float = 0.8) -> float:
    """
    计算邻居拓扑的成本函数: C_neighbor = α_x * P_x + β_y * P_y + γ_s * P_s
    
    Args:
        base_layout: 基础拓扑布局 (σ_i)
        neighbor_layout: 邻居拓扑布局 (T_j)
        alpha_x: 宽度惩罚系数
        beta_y: 高度惩罚系数
        gamma_s: 相似度惩罚系数
        expected_similarity: 期望相似度 S_e (默认0.8)
        
    Returns:
        邻居拓扑的总成本
    """
    from TCG import get_layout_bounds
    
    # 计算基础拓扑和邻居拓扑的宽高
    base_x_min, base_y_min, base_x_max, base_y_max = get_layout_bounds(base_layout)
    base_width = base_x_max - base_x_min
    base_height = base_y_max - base_y_min
    
    neighbor_x_min, neighbor_y_min, neighbor_x_max, neighbor_y_max = get_layout_bounds(neighbor_layout)
    neighbor_width = neighbor_x_max - neighbor_x_min
    neighbor_height = neighbor_y_max - neighbor_y_min
    
    # 1. 形状约束项：P_x 和 P_y
    P_x = neighbor_width / base_width if base_width > 0 else 1.0
    P_y = neighbor_height / base_height if base_height > 0 else 1.0
    
    # 2. 相似性力项：P_s = |SD - S_e| / (1 - S_e)
    SD = compute_similarity(base_layout, neighbor_layout)
    if expected_similarity >= 1.0:
        P_s = abs(SD - expected_similarity)
    else:
        P_s = abs(SD - expected_similarity) / (1.0 - expected_similarity)
    
    # 总成本
    C_neighbor = alpha_x * P_x + beta_y * P_y + gamma_s * P_s
    return C_neighbor


def SA_neighbor_generation(base_node: SimilarityTreeNode,
                          problem: LayoutProblem,
                          max_iterations: int = 5000,
                          initial_temp: float = 50.0,
                          cooling_rate: float = 0.95,
                          alpha_x: float = 1.0,
                          beta_y: float = 1.0,
                          gamma_s: float = 2.0,
                          expected_similarity: float = 0.8,
                          max_neighbors: int = 10,
                          verbose: bool = False) -> List[Tuple[TCG, Dict[str, Chiplet], float]]:
    """
    基于相似度的SA算法，为基础节点生成邻居节点
    
    Args:
        base_node: 基础节点（父节点）
        problem: 布局问题
        max_iterations: 最大迭代次数
        initial_temp: 初始温度
        cooling_rate: 冷却速率
        alpha_x, beta_y, gamma_s: 成本系数
        expected_similarity: 期望相似度
        max_neighbors: 最多生成的邻居数量
        verbose: 是否输出详细信息
        
    Returns:
        邻居解列表 [(TCG, Layout, cost), ...]
    """
    import random
    import math
    from TCG import generate_layout_from_tcg
    from legalize_tcg import legalize_tcg
    from Legality_optimized_SA import _generate_neighbor_tcg, cost_legal
    
    if verbose:
        print(f"\n  为节点#{base_node.node_id}生成邻居...")
    
    base_layout = base_node.layout
    current_tcg = copy.deepcopy(base_node.tcg)
    neighbors = []
    temp = initial_temp
    best_cost = float('inf')
    
    for iteration in range(max_iterations):
        neighbor_tcg = _generate_neighbor_tcg(current_tcg, problem)
        is_valid, _ = neighbor_tcg.is_valid()
        if not is_valid:
            continue
        
        try:
            neighbor_layout = generate_layout_from_tcg(neighbor_tcg, problem)
            legal_cost = cost_legal(neighbor_tcg, problem, neighbor_layout, alpha_c=1.0, beta_l=10.0)
            
            # 如果不合法，尝试legalize
            if abs(legal_cost) > 1e-6:
                success, legalized_tcg, legalized_layout = legalize_tcg(neighbor_tcg, problem, verbose=False)
                if success:
                    neighbor_tcg, neighbor_layout = legalized_tcg, legalized_layout
                    legal_cost = cost_legal(neighbor_tcg, problem, neighbor_layout, alpha_c=1.0, beta_l=10.0)
                else:
                    continue
            
            if abs(legal_cost) > 1e-6:
                continue
            
            # 计算邻居成本
            neighbor_cost = compute_neighbor_cost(base_layout, neighbor_layout,
                                                alpha_x, beta_y, gamma_s, expected_similarity)
            
            if neighbor_cost < best_cost:
                best_cost = neighbor_cost
                
                # 避免重复
                is_duplicate = any(compute_similarity(neighbor_layout, ex_layout) > 0.95 
                                 for _, ex_layout, _ in neighbors)
                
                if not is_duplicate:
                    neighbors.append((copy.deepcopy(neighbor_tcg), 
                                    copy.deepcopy(neighbor_layout), neighbor_cost))
                    if verbose and len(neighbors) <= 3:
                        print(f"    找到邻居#{len(neighbors)}: cost={neighbor_cost:.4f}")
                    if len(neighbors) >= max_neighbors:
                        if verbose:
                            print(f"    已找到{max_neighbors}个邻居，提前终止")
                        break
                
                current_tcg = copy.deepcopy(neighbor_tcg)
            else:
                delta = neighbor_cost - best_cost
                if random.random() < math.exp(-delta / temp):
                    current_tcg = copy.deepcopy(neighbor_tcg)
        
        except Exception:
            continue
        
        if iteration % 100 == 0:
            temp *= cooling_rate
    
    if verbose:
        print(f"    完成: 共找到{len(neighbors)}个合法邻居")
    return neighbors


def GroupConstrua_full(root_nodes: List[SimilarityTreeNode],
                      problem: LayoutProblem,
                      num_topologies_to_generate: int = 100,
                      children_per_node: int = 5,
                      max_iterations_sa: int = 5000,
                      alpha_x: float = 1.0,
                      beta_y: float = 1.0,
                      gamma_s: float = 2.0,
                      expected_similarity: float = 0.8,
                      verbose: bool = True) -> List[SimilarityTree]:
    """
    Group Construction: 构建多个相似度树（广度优先构建）
    
    构建策略（按论文方法）：
    1. 对根节点进行SA扰动，生成大量候选拓扑
    2. 使用广度优先方式构建树：
       - Step 1: 计算当前节点与所有未连接拓扑的相似度
       - Step 2: 选择相似度最高的前N个作为当前节点的子节点
       - Step 3: 按BFS顺序选择下一个当前节点，重复直到所有拓扑都连接
    
    Args:
        root_nodes: 基础导出得到的根节点列表
        problem: 布局问题
        num_topologies_to_generate: 为每棵树生成的候选拓扑总数
        children_per_node: 每个节点最多拥有的子节点数（N值）
        max_iterations_sa: SA最大迭代次数
        alpha_x, beta_y, gamma_s: 成本参数
        expected_similarity: 期望相似度
        verbose: 是否输出详细信息
        
    Returns:
        相似度树列表（森林）
    """
    from collections import deque
    
    if verbose:
        print("="*80)
        print("Group Construction - 构建相似度树（BFS方式）")
        print("="*80)
        print(f"根节点数量: {len(root_nodes)}")
        print(f"每棵树生成拓扑数: {num_topologies_to_generate}")
        print(f"每节点子节点数: {children_per_node}")
        print(f"SA参数: max_iter={max_iterations_sa}")
        print(f"成本参数: α_x={alpha_x}, β_y={beta_y}, γ_s={gamma_s}, S_e={expected_similarity}")
        print("="*80 + "\n")
    
    similarity_trees = []
    
    for tree_idx, root in enumerate(root_nodes):
        if verbose:
            print(f"\n{'─'*80}")
            print(f"处理树 #{tree_idx + 1}/{len(root_nodes)} (根节点ID={root.node_id})")
            print(f"{'─'*80}")
        
        # 创建树（初始只有根节点）
        tree = SimilarityTree(root)
        
        # Step 1: 对根节点进行SA扰动，生成大量候选拓扑
        if verbose:
            print(f"\n  生成候选拓扑...")
        
        candidate_topologies = []  # [(TCG, Layout, cost), ...]
        
        # 使用SA生成候选拓扑
        generated_count = 0
        while generated_count < num_topologies_to_generate:
            neighbors = SA_neighbor_generation(
                base_node=root, problem=problem,
                max_iterations=max_iterations_sa,
                initial_temp=50.0, cooling_rate=0.95,
                alpha_x=alpha_x, beta_y=beta_y, gamma_s=gamma_s,
                expected_similarity=expected_similarity,
                max_neighbors=min(20, num_topologies_to_generate - generated_count),
                verbose=False
            )
            candidate_topologies.extend(neighbors)
            generated_count = len(candidate_topologies)
            
            if len(neighbors) == 0:
                if verbose:
                    print(f"    警告: SA无法生成更多拓扑，当前共{generated_count}个")
                break
        
        if verbose:
            print(f"    共生成 {len(candidate_topologies)} 个候选拓扑")
        
        if len(candidate_topologies) == 0:
            if verbose:
                print(f"  → 树#{tree_idx + 1}完成: 仅包含根节点（无候选拓扑）")
            similarity_trees.append(tree)
            continue
        
        # Step 2: 使用BFS方式构建树
        if verbose:
            print(f"\n  使用BFS方式构建树...")
        
        # 未连接的拓扑集合
        unconnected_topologies = candidate_topologies.copy()
        
        # BFS队列：存储待处理的节点
        bfs_queue = deque([root])
        
        while bfs_queue and unconnected_topologies:
            # Step 3: 按BFS顺序选择当前节点
            current_node = bfs_queue.popleft()
            
            if verbose:
                print(f"    处理节点#{current_node.node_id}, 剩余未连接拓扑: {len(unconnected_topologies)}")
            
            # Step 1: 计算当前节点与所有未连接拓扑的相似度
            similarities = []
            for idx, (tcg, layout, cost) in enumerate(unconnected_topologies):
                sim = compute_similarity(current_node.layout, layout)
                similarities.append((idx, sim, tcg, layout, cost))
            
            # Step 2: 选择相似度最高的前N个作为子节点
            similarities.sort(key=lambda x: x[1], reverse=True)  # 按相似度降序排序
            
            num_children = min(children_per_node, len(similarities))
            selected_indices = set()
            
            for i in range(num_children):
                idx, sim, tcg, layout, cost = similarities[i]
                selected_indices.add(idx)
                
                # 创建子节点
                child_node = SimilarityTreeNode(
                    tcg=tcg, layout=layout,
                    parent=current_node,
                    similarity_to_parent=sim,
                    cost=cost
                )
                
                # 添加到树中
                tree.add_node(current_node, child_node)
                
                # 将子节点加入BFS队列
                bfs_queue.append(child_node)
                
                if verbose and i < 3:  # 只打印前3个
                    print(f"      添加子节点: 相似度={sim:.4f}, cost={cost:.4f}")
            
            # 从未连接集合中移除已选择的拓扑
            unconnected_topologies = [
                topo for i, topo in enumerate(unconnected_topologies) 
                if i not in selected_indices
            ]
        
        if verbose:
            stats = tree.get_tree_statistics()
            print(f"  → 树#{tree_idx + 1}完成: {stats['total_nodes']}个节点, "
                  f"最大深度={stats['max_depth']}, 剩余未连接={len(unconnected_topologies)}")
        
        similarity_trees.append(tree)
    
    if verbose:
        print("\n" + "="*80)
        print("Group Construction 完成")
        print("="*80)
        print(f"生成的树数量: {len(similarity_trees)}")
        for i, tree in enumerate(similarity_trees):
            stats = tree.get_tree_statistics()
            print(f"  树#{i+1}: {stats['total_nodes']}个节点, "
                  f"最大深度={stats['max_depth']}, "
                  f"叶节点={stats['num_leaves']}, "
                  f"平均子节点数={stats['avg_children']:.2f}")
        print("="*80)
    
    return similarity_trees


def GroupConstrua(root_nodes: List[SimilarityTreeNode]) -> List[SimilarityTree]:
    """
    简化版本：根据基础导出得到的根节点列表，构建多个相似度树（仅包含根节点）
    
    Args:
        root_nodes: 基础导出得到的根节点列表
        
    Returns:
        相似度树列表
    """
    similarity_trees = []
    for root in root_nodes:
        tree = SimilarityTree(root)
        similarity_trees.append(tree)
    return similarity_trees
#添加GroupConstrua：
#根据BaseDerivation.py中返回的List[SimilarityTreeNode]，对于每个根节点，构建一个相似度树
#构建方法：对于每个根节点，调用Legalize_optimize.py中的SA_1函数，SA扰动后如果不合法要进行legalize_tcg，如果合法直接计算相似度插入树中
#SA的方向（即cost）与原始不同：具体的计算公式如下：$$C_{neighbor} = \alpha_x P_x(\sigma_i, T_j) + \beta_y P_y(\sigma_i, T_j) + \gamma_s P_s(\sigma_i, T_j)$$
#下面详细拆解这三个项是怎么算的：. 形状约束项：$P_x$ 和 $P_y$（控制胖瘦）这两项是为了防止生成的邻居拓扑长得太“畸形”（比如变得特别宽），因为过宽的布局在后期很难压缩面积。计算方法：$P_x = \frac{W(T_j)}{W(\sigma_i)}$：新拓扑的宽度 / 基础拓扑的宽度。$P_y = \frac{H(T_j)}{H(\sigma_i)}$：新拓扑的高度 / 基础拓扑的高度。
#2. 相似性力项：$P_s$（控制相似度）这项是为了鼓励生成的邻居拓扑与基础拓扑在结构上保持一定的相似性，从而更好地继承基础解的优点。 第一步：算出相似度 $SD(\sigma_i, T_j)$算法会比较两个拓扑中每个芯片的相对位置关系：对每个芯片，观察它在 TCG 图中四个方向最近的邻居是谁。如果两个拓扑中，芯片 $c_k$ 的邻居和方向大部分都对得上，那相似度就高。$SD$ 的值在 $0$ 到 $1$ 之间（1表示完全一样，0表示完全不同）。第二步：计算惩罚 $P_s$$$P_s(\sigma_i, T_j) = \frac{|SD(\sigma_i, T_j) - S_e|}{1 - S_e}$$$S_e$ (Expected Similarity)：这是作者预设的一个**“期望相似度”**（比如设定为0.8）。逻辑：如果 $SD$ 刚好等于 $S_e$，代价 $P_s$ 就是 0，SA最开心。如果 $SD$ 太高（跟原件一模一样），$P_s$ 会变大，SA会拒绝，逼它去变点花样。如果 $SD$ 太低（变得面目全非），$P_s$ 也会变大，SA也会拒绝，怕它跑丢了。



if __name__ == "__main__":
    # # 简单测试相似度计算
    # print("Testing similarity computation...")
    # chip_a1 = Chiplet("A", 10, 10, 0, 0)
    # chip_b1 = Chiplet("B", 10, 10, 10, 0)
    # chip_c1 = Chiplet("C", 10, 10, 0, 10)
    
    # layout1 = {
    #     "A": chip_a1,
    #     "B": chip_b1,
    #     "C": chip_c1
    # }
    
    # chip_a2 = Chiplet("A", 10, 10, 0, 0)
    # chip_b2 = Chiplet("B", 10, 10, 10, 2)
    # chip_c2 = Chiplet("C", 10, 10, 5, 10)  # C位置不同
    
    # layout2 = {
    #     "A": chip_a2,
    #     "B": chip_b2,
    #     "C": chip_c2
    # }
    
    # similarity = compute_similarity(layout1, layout2)
    # print(f"Similarity between layout1 and layout2 : {similarity:.4f}")
    from BaseDerivation import run_base_derivation
    from unit import load_problem_from_json
    problem = load_problem_from_json("../test_input/8core.json")




# 2. 基础导出（生成根节点）
    root_nodes = run_base_derivation(problem, num_runs=5, min_similarity=0.4)
    # print(f"\n基础导出完成，生成 {len(root_nodes)} 个根节点。")
    from unit import visualize_layout_with_bridges, save_layout_image
    # image_path = f"../output/BaseDerivation/root_node_{0+1}_layout.png"
    # save_layout_image(root_nodes[0].layout, problem, image_path)
    # print(f"  根节点 #{0+1} 布局图已保存: {image_path}")


    similarity_trees = GroupConstrua_full(
        root_nodes=root_nodes,
        problem=problem,
        num_topologies_to_generate=50,  # 为每棵树生成6个候选拓扑
        children_per_node=5,  # 每个节点最多2个子节点
        max_iterations_sa=5000,
        alpha_x=1.0,
        beta_y=1.0,
        gamma_s=2.0,
        expected_similarity=0.8,
        verbose=True
    )
    print(f"\nGroup Construction 完成，生成 {len(similarity_trees)} 棵相似度树。")
 
    for i, tree in enumerate(similarity_trees):
        stats = tree.get_tree_statistics()
        print(f"  树#{i+1}: {stats['total_nodes']}个节点, 最大深度={stats['max_depth']}")

    print(similarity_trees[0])
    #保存每个节点的布局图
    import os
    output_dir = "../output/GroupConstrua"
    os.makedirs(output_dir, exist_ok=True)  # 创建目录（如果不存在）
    
    for node in similarity_trees[0].all_nodes:
        image_path = f"{output_dir}/tree1_node{node.node_id}_layout.png"
        save_layout_image(node.layout, problem, image_path)
        print(f"  节点 #{node.node_id} 布局图已保存: {image_path}")
        