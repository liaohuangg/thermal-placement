"""
合法性优化的模拟退火算法
使用合法性成本函数来引导TCG优化
"""
import networkx as nx
from typing import Dict, Set, Tuple, List
from TCG import TCG
from chiplet_model import Chiplet, LayoutProblem, MIN_OVERLAP, get_adjacency_info


def count_emib_closure_edges(tcg: TCG, problem: LayoutProblem) -> int:
    """
    计算EMIB边中有多少是闭包边
    
    关键约束: 所有EMIB边必须是归约边(reduction edge),
    如果EMIB边是闭包边,说明两个芯片之间有其他芯片阻隔,无法直接邻接
    
    Args:
        tcg: TCG拓扑图
        problem: 布局问题
        
    Returns:
        EMIB边中闭包边的数量
    """
    emib_closure_count = 0
    
    # 遍历所有EMIB连接
    for chip1_id, chip2_id in problem.connection_graph.edges():
        # 检查这个EMIB边在Ch或Cv中是否是闭包边
        
        # 检查Ch中的情况
        if tcg.Ch.has_edge(chip1_id, chip2_id):
            # 检查是否是闭包边
            is_closure = False
            for nk in tcg.Ch.nodes():
                if nk != chip1_id and nk != chip2_id:
                    if tcg.Ch.has_edge(chip1_id, nk) and tcg.Ch.has_edge(nk, chip2_id):
                        is_closure = True
                        break
            if is_closure:
                emib_closure_count += 1
        
        elif tcg.Ch.has_edge(chip2_id, chip1_id):
            # 反向边
            is_closure = False
            for nk in tcg.Ch.nodes():
                if nk != chip1_id and nk != chip2_id:
                    if tcg.Ch.has_edge(chip2_id, nk) and tcg.Ch.has_edge(nk, chip1_id):
                        is_closure = True
                        break
            if is_closure:
                emib_closure_count += 1
        
        # 检查Cv中的情况
        elif tcg.Cv.has_edge(chip1_id, chip2_id):
            is_closure = False
            for nk in tcg.Cv.nodes():
                if nk != chip1_id and nk != chip2_id:
                    if tcg.Cv.has_edge(chip1_id, nk) and tcg.Cv.has_edge(nk, chip2_id):
                        is_closure = True
                        break
            if is_closure:
                emib_closure_count += 1
        
        elif tcg.Cv.has_edge(chip2_id, chip1_id):
            is_closure = False
            for nk in tcg.Cv.nodes():
                if nk != chip1_id and nk != chip2_id:
                    if tcg.Cv.has_edge(chip2_id, nk) and tcg.Cv.has_edge(nk, chip1_id):
                        is_closure = True
                        break
            if is_closure:
                emib_closure_count += 1
    
    return emib_closure_count


def count_closure_edges(graph: nx.DiGraph) -> int:
    """
    计算图中闭合边的数量
    
    闭合边定义: 如果边 ni → nj 存在,且存在另一个节点 nk,
    使得 ni → nk 和 nk → nj 都存在,则 ni → nj 是闭合边
    (可以通过传递关系推导出来的边)
    
    Args:
        graph: 有向图
        
    Returns:
        闭合边的数量
    """
    closure_count = 0
    
    for ni, nj in graph.edges():
        # 检查是否存在中间节点 nk
        is_closure = False
        for nk in graph.nodes():
            if nk != ni and nk != nj:
                # 检查是否存在路径 ni -> nk -> nj
                if graph.has_edge(ni, nk) and graph.has_edge(nk, nj):
                    is_closure = True
                    break
        
        if is_closure:
            closure_count += 1
    
    return closure_count


def get_emib_edges(problem: LayoutProblem) -> Set[Tuple[str, str]]:
    """
    获取所有EMIB边(桥连接边)
    
    Args:
        problem: 布局问题
        
    Returns:
        EMIB边集合
    """
    emib_edges = set()
    for chip1_id, chip2_id in problem.connection_graph.edges():
        emib_edges.add((chip1_id, chip2_id))
        emib_edges.add((chip2_id, chip1_id))  # 双向
    return emib_edges


def count_legal_emib_edges(tcg: TCG, problem: LayoutProblem, layout: Dict[str, Chiplet]) -> int:
    """
    计算当前合法的EMIB边数量
    
    合法EMIB边: 对应的两个芯片满足邻接且重叠 >= MIN_OVERLAP
    
    Args:
        tcg: TCG拓扑
        problem: 布局问题
        layout: 当前布局
        
    Returns:
        合法EMIB边的数量
    """
    legal_count = 0
    
    for chip1_id, chip2_id in problem.connection_graph.edges():
        chip1 = layout.get(chip1_id)
        chip2 = layout.get(chip2_id)
        
        if chip1 and chip2:
            is_adj, overlap_len, _ = get_adjacency_info(chip1, chip2)
            if is_adj and overlap_len >= MIN_OVERLAP:
                legal_count += 1
    
    return legal_count


def cost_legal(tcg: TCG, problem: LayoutProblem, layout: Dict[str, Chiplet],
               alpha_c: float = 1.0, beta_l: float = 1.0) -> float:
    """
    计算TCG的合法性成本
    
    公式: Clegal = αc * Pc(Ti) + βl * Pl(Ti)
    
    其中:
    - Pc(Ti): TCG中EMIB边为闭包边的比例 (关键修正!)
      * EMIB边必须是归约边,否则两芯片间有阻隔,无法直接邻接
    - Pl(Ti): TCG中非法EMIB边和EMIB边的比例
      * 非法EMIB边 = EMIB边总数 - 当前合法EMIB边数量
    
    Args:
        tcg: 传递闭包图
        problem: 布局问题
        layout: 当前布局
        alpha_c: EMIB闭包边惩罚系数
        beta_l: 非法EMIB边惩罚系数
        
    Returns:
        合法性成本值(越小越好)
    """
    total_emib_edges = len(problem.connection_graph.edges())  # 无向边数量
    
    if total_emib_edges == 0:
        # 没有桥连接,返回0
        return 0.0
    
    # 1. 计算EMIB边为闭包边的比例 Pc
    # 关键: 只统计EMIB边中有多少是闭包边,而不是所有闭包边
    emib_closure_count = count_emib_closure_edges(tcg, problem)
    Pc = emib_closure_count / total_emib_edges if total_emib_edges > 0 else 0.0
    
    # 2. 计算非法EMIB边比例 Pl
    legal_emib_count = count_legal_emib_edges(tcg, problem, layout)
    illegal_emib_count = total_emib_edges - legal_emib_count
    
    Pl = illegal_emib_count / total_emib_edges if total_emib_edges > 0 else 0.0
    
    # 3. 计算总成本
    Clegal = alpha_c * Pc + beta_l * Pl
    
    return Clegal


def SA_1(initial_tcg: TCG, problem: LayoutProblem, 
         max_iterations: int = 10000,
         initial_temp: float = 100.0,
         cooling_rate: float = 0.95,
         alpha_c: float = 1.0,
         beta_l: float = 2.0,
         use_legalize: bool = True,
         verbose: bool = False) -> Tuple[List[TCG], List[Dict[str, Chiplet]], float]:
    """
    合法性优化的模拟退火算法
    
    通过模拟退火搜索,将不合法的TCG优化为合法的TCG(cost_legal = 0)
    关键改进: 集成legalize_tcg方法,对每个邻域解尝试合法化
    
    TCG操作:
    1. Op1: 交换两个节点在某个序列中的位置
    2. Op2: 旋转一个芯片(交换宽高)
    3. Op3: 将某条边从Ch移到Cv或反之
    
    Args:
        initial_tcg: 初始TCG(可能不合法)
        problem: 布局问题
        max_iterations: 最大迭代次数
        initial_temp: 初始温度
        cooling_rate: 冷却速率
        alpha_c: EMIB闭包边惩罚系数
        beta_l: 非法EMIB边惩罚系数
        use_legalize: 是否在邻域生成后尝试legalize_tcg
        verbose: 是否输出详细信息
        
    Returns:
        (legal_tcgs, legal_layouts, best_cost): 
        - legal_tcgs: 找到的所有合法TCG列表
        - legal_layouts: 对应的合法布局列表
        - best_cost: 最佳成本值
    """
    import copy
    import random
    import math
    from TCG import generate_layout_from_tcg
    from legalize_tcg import legalize_tcg
    
    # 初始化
    current_tcg = copy.deepcopy(initial_tcg)
    current_layout = generate_layout_from_tcg(current_tcg, problem)
    current_cost = cost_legal(current_tcg, problem, current_layout, alpha_c, beta_l)
    
    # 如果初始TCG已经合法,直接返回
    if abs(current_cost) < 1e-6:
        if verbose:
            print(f"SA_1 合法性优化: 初始TCG已合法,直接返回")
            print(f"  初始成本: {current_cost:.4f}")
        return [copy.deepcopy(current_tcg)], [copy.deepcopy(current_layout)], 0.0
    
    # 如果启用legalize,先尝试合法化初始TCG
    if use_legalize:
        success, legalized_tcg, legalized_layout = legalize_tcg(current_tcg, problem, verbose=False)
        if success:
            legalized_cost = cost_legal(legalized_tcg, problem, legalized_layout, alpha_c, beta_l)
            if legalized_cost < current_cost:
                current_tcg = legalized_tcg
                current_layout = legalized_layout
                current_cost = legalized_cost
                if verbose:
                    print(f"  初始TCG经legalize后成本改善: {current_cost:.4f}")
                if abs(current_cost) < 1e-6:
                    if verbose:
                        print(f"  初始TCG经legalize后已合法!")
                    return [copy.deepcopy(current_tcg)], [copy.deepcopy(current_layout)], 0.0
    
    best_tcg = copy.deepcopy(current_tcg)
    best_layout = copy.deepcopy(current_layout)
    best_cost = current_cost
    
    # 存储所有找到的合法解
    legal_tcgs = []
    legal_layouts = []
    
    if verbose:
        print(f"SA_1 合法性优化开始")
        print(f"初始成本: {current_cost:.4f}")
        print(f"目标: cost_legal = 0.0")
        print(f"启用legalize_tcg: {use_legalize}")
        print("=" * 70)
    
    temp = initial_temp
    iterations_without_improvement = 0
    legalize_success_count = 0
    
    for iteration in range(max_iterations):
        # 生成邻域解 - 随机选择一种操作
        neighbor_tcg = _generate_neighbor_tcg(current_tcg, problem)
        
        # 检查TCG有效性
        is_valid, msg = neighbor_tcg.is_valid()
        if not is_valid:
            continue  # 跳过无效的TCG
        
        # 生成布局并计算成本
        try:
            neighbor_layout = generate_layout_from_tcg(neighbor_tcg, problem)
            neighbor_cost = cost_legal(neighbor_tcg, problem, neighbor_layout, alpha_c, beta_l)
            
            # 尝试使用legalize_tcg改进邻域解
            if use_legalize:
                success, legalized_tcg, legalized_layout = legalize_tcg(neighbor_tcg, problem, verbose=False)
                if success:
                    legalized_cost = cost_legal(legalized_tcg, problem, legalized_layout, alpha_c, beta_l)
                    if legalized_cost < neighbor_cost:
                        # legalize改进了解
                        neighbor_tcg = legalized_tcg
                        neighbor_layout = legalized_layout
                        neighbor_cost = legalized_cost
                        legalize_success_count += 1
                        
                        # 如果legalize直接找到合法解
                        if abs(neighbor_cost) < 1e-6:
                            if verbose and len(legal_tcgs) < 10:  # 只打印前10个
                                print(f"  [迭代 {iteration}] legalize找到合法解! (第{len(legal_tcgs)+1}个)")
                            legal_tcgs.append(copy.deepcopy(neighbor_tcg))
                            legal_layouts.append(copy.deepcopy(neighbor_layout))
        
        except Exception as e:
            if verbose and iteration % 100 == 0:
                print(f"  布局生成失败: {e}")
            continue
        
        # Metropolis准则
        delta_cost = neighbor_cost - current_cost
        
        if delta_cost < 0 or random.random() < math.exp(-delta_cost / temp):
            # 接受新解
            current_tcg = neighbor_tcg
            current_layout = neighbor_layout
            current_cost = neighbor_cost
            
            # 更新最佳解
            if current_cost < best_cost:
                best_tcg = copy.deepcopy(current_tcg)
                best_layout = copy.deepcopy(current_layout)
                best_cost = current_cost
                iterations_without_improvement = 0
                
                if verbose:
                    print(f"  [迭代 {iteration}] 新最佳成本: {best_cost:.4f} (温度={temp:.2f})")
                
                # 检查是否找到合法解
                if abs(best_cost) < 1e-6:  # cost ≈ 0
                    if best_tcg not in [tcg for tcg in legal_tcgs]:
                        legal_tcgs.append(copy.deepcopy(best_tcg))
                        legal_layouts.append(copy.deepcopy(best_layout))
                        
                        if verbose:
                            print(f"  ✓ 找到合法解! (第{len(legal_tcgs)}个)")
            else:
                iterations_without_improvement += 1
        
        # 降温
        if iteration % 100 == 0:
            temp *= cooling_rate
        
        # 进度输出
        if verbose and iteration % 500 == 0 and iteration > 0:
            print(f"  [迭代 {iteration}/{max_iterations}] 当前成本={current_cost:.4f}, "
                  f"最佳成本={best_cost:.4f}, 温度={temp:.2f}, 合法解数={len(legal_tcgs)}, "
                  f"legalize成功={legalize_success_count}")
        
        # 早停条件
        if len(legal_tcgs) >= 1000:  # 找到100个合法解就停止
            if verbose:
                print(f"  已找到{len(legal_tcgs)}个合法解,提前终止")
            break
        
        if iterations_without_improvement > 30000:
            if verbose:
                print(f"  连续30000次迭代无改善,提前终止")
            break
    
    if verbose:
        print("=" * 70)
        print(f"SA_1 完成:")
        print(f"  最佳成本: {best_cost:.4f}")
        print(f"  找到合法解数量: {len(legal_tcgs)}")
        print(f"  总迭代次数: {iteration + 1}")
        if use_legalize:
            print(f"  legalize成功次数: {legalize_success_count}")
    
    # 返回结果: 如果找到合法解返回合法解集合,否则返回空列表和最佳成本
    if len(legal_tcgs) > 0:
        return legal_tcgs, legal_layouts, 0.0
    else:
        # 未找到合法解,返回空列表
        return [], [], best_cost


def _generate_neighbor_tcg(tcg: TCG, problem: LayoutProblem) -> TCG:
    """
    生成邻域TCG - 随机应用一种操作
    
    操作类型:
    1. 交换Ch或Cv中两条边的方向
    2. 将一条非EMIB边从Ch移到Cv或反之
    3. 删除一条闭包边并重新添加
    
    Args:
        tcg: 当前TCG
        problem: 布局问题
        
    Returns:
        新的邻域TCG
    """
    import copy
    import random
    
    new_tcg = copy.deepcopy(tcg)
    
    # 随机选择操作类型
    op_type = random.choice([1, 2, 3])
    
    if op_type == 1:
        # 操作1: 反转一条边的方向
        graph_choice = random.choice(['Ch', 'Cv'])
        graph = new_tcg.Ch if graph_choice == 'Ch' else new_tcg.Cv
        
        if graph.number_of_edges() > 0:
            edge = random.choice(list(graph.edges()))
            graph.remove_edge(edge[0], edge[1])
            graph.add_edge(edge[1], edge[0])
    
    elif op_type == 2:
        # 操作2: 将边从Ch移到Cv或反之
        if new_tcg.Ch.number_of_edges() > 0 and random.random() < 0.5:
            # Ch -> Cv
            edge = random.choice(list(new_tcg.Ch.edges()))
            # 检查是否是EMIB边,EMIB边不能移动
            is_emib = problem.connection_graph.has_edge(edge[0], edge[1])
            if not is_emib:
                new_tcg.Ch.remove_edge(edge[0], edge[1])
                new_tcg.Cv.add_edge(edge[0], edge[1])
        elif new_tcg.Cv.number_of_edges() > 0:
            # Cv -> Ch
            edge = random.choice(list(new_tcg.Cv.edges()))
            is_emib = problem.connection_graph.has_edge(edge[0], edge[1])
            if not is_emib:
                new_tcg.Cv.remove_edge(edge[0], edge[1])
                new_tcg.Ch.add_edge(edge[0], edge[1])
    
    elif op_type == 3:
        # 操作3: 随机添加或删除一条边
        graph_choice = random.choice(['Ch', 'Cv'])
        graph = new_tcg.Ch if graph_choice == 'Ch' else new_tcg.Cv
        
        nodes = list(graph.nodes())
        if len(nodes) >= 2:
            n1, n2 = random.sample(nodes, 2)
            if graph.has_edge(n1, n2):
                # 检查是否是EMIB边
                is_emib = problem.connection_graph.has_edge(n1, n2)
                if not is_emib:
                    graph.remove_edge(n1, n2)
            else:
                graph.add_edge(n1, n2)
    
    return new_tcg


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    from unit import load_problem_from_json
    from Generate_initial_TCG import generate_initial_TCG
    from TCG import generate_layout_from_tcg
    from legalize_tcg import legalize_tcg
    
    print("=" * 70)
    print("合法性成本函数测试")
    print("=" * 70)
    
    # 测试1: 3芯片简单案例
    print("\n测试1: 3芯片案例")
    print("-" * 70)
    
    problem = load_problem_from_json("../test_input/3core.json")
    print(f"加载问题: {len(problem.chiplets)}个芯片, {problem.connection_graph.number_of_edges()}个连接")
    
    # 生成随机TCG
    # tcg = generate_initial_TCG(problem, seed=42)
   

    tcg=TCG(["A","B","C"])
    tcg.add_horizontal_constraint("C","B")
    tcg.add_horizontal_constraint("A","C")
    tcg.add_horizontal_constraint("A","B")


    print(f"\n生成TCG: Ch边={tcg.Ch.number_of_edges()}, Cv边={tcg.Cv.number_of_edges()}")
    print(f"TCG.Ch边: {list(tcg.Ch.edges())}")
    print(f"TCG.Cv边: {list(tcg.Cv.edges())}")
    
    # 生成初始布局
    layout = generate_layout_from_tcg(tcg, problem)
    print(f"\n初始布局:")
    for chip_id, chip in layout.items():
        print(f"  {chip_id}: ({chip.x:.1f}, {chip.y:.1f})")
    
    # 计算合法性成本
    print(f"\n计算合法性成本...")
    
    # 详细计算过程
    ch_closure = count_closure_edges(tcg.Ch)
    cv_closure = count_closure_edges(tcg.Cv)
    total_closure = ch_closure + cv_closure
    
    total_emib = len(problem.connection_graph.edges())
    legal_emib = count_legal_emib_edges(tcg, problem, layout)
    illegal_emib = total_emib - legal_emib
    
    emib_closure = count_emib_closure_edges(tcg, problem)
    Pc = emib_closure / total_emib if total_emib > 0 else 0.0
    Pl = illegal_emib / total_emib if total_emib > 0 else 0.0
    
    print(f"  Ch闭合边总数: {ch_closure} (可能含非EMIB边)")
    print(f"  Cv闭合边总数: {cv_closure}")
    print(f"  总闭包边数: {total_closure}")
    print(f"  ────────────────")
    print(f"  EMIB连接总数: {total_emib}")
    print(f"  EMIB中闭包边: {emib_closure} ⚠️ 必须=0")
    print(f"  合法EMIB边数: {legal_emib}")
    print(f"  非法EMIB边数: {illegal_emib}")
    print(f"  ────────────────")
    print(f"  Pc (EMIB闭包比): {Pc:.4f}")
    print(f"  Pl (非法EMIB比): {Pl:.4f}")
    
    cost = cost_legal(tcg, problem, layout, alpha_c=1.0, beta_l=1.0)
    print(f"\n  Clegal = {Pc:.4f} + {Pl:.4f} = {cost:.4f}")
    
    # 尝试合法化
    print(f"\n尝试合法化TCG...")
    success, legal_tcg, legal_layout = legalize_tcg(tcg, problem, verbose=False)
    
    if success:
        print(f"✓ 合法化成功!")
        print(f"\n合法化后布局:")
        for chip_id, chip in legal_layout.items():
            print(f"  {chip_id}: ({chip.x:.1f}, {chip.y:.1f})")
        
        # 计算合法化后的成本
        legal_cost = cost_legal(legal_tcg, problem, legal_layout, alpha_c=1.0, beta_l=1.0)
        
        legal_emib_after = count_legal_emib_edges(legal_tcg, problem, legal_layout)
        illegal_emib_after = total_emib - legal_emib_after
        Pl_after = illegal_emib_after / total_emib if total_emib > 0 else 0.0
        
        print(f"\n合法化后:")
        print(f"  合法EMIB边: {legal_emib_after}/{total_emib}")
        print(f"  非法EMIB边: {illegal_emib_after}")
        print(f"  Pl (非法边比例): {Pl_after:.4f}")
        print(f"  Clegal = {legal_cost:.4f}")
        print(f"  成本改善: {cost - legal_cost:.4f} ({'降低' if legal_cost < cost else '升高'})")
    else:
        print(f"✗ 合法化失败")
    
    # # 测试2: 8芯片复杂案例
    # print("\n\n" + "=" * 70)
    # print("测试2: 8芯片案例 - 验证EMIB闭包边检测")
    # print("=" * 70)
    
    # problem2 = load_problem_from_json("../test_input/8core.json")
    # print(f"加载问题: {len(problem2.chiplets)}个芯片, {problem2.connection_graph.number_of_edges()}个连接")
    
    # # 测试多个随机TCG
    # print(f"\n测试10个随机TCG的EMIB闭包边情况...")
    # costs = []
    
    # for i in range(1000):
    #     tcg2 = generate_initial_TCG(problem2, seed=None)
    #     layout2 = generate_layout_from_tcg(tcg2, problem2)
        
    #     emib_closure2 = count_emib_closure_edges(tcg2, problem2)
    #     cost2 = cost_legal(tcg2, problem2, layout2, alpha_c=1.0, beta_l=2.0)
    #     costs.append(cost2)
        
    #     legal_count2 = count_legal_emib_edges(tcg2, problem2, layout2)
    #     total_emib2 = len(problem2.connection_graph.edges())
        
    #     print(f"  TCG {i+1}: EMIB闭包={emib_closure2}/{total_emib2}, "
    #           f"合法边={legal_count2}/{total_emib2}, Clegal={cost2:.4f}")
    
    # print(f"\n统计:")
    # print(f"  平均成本: {sum(costs)/len(costs):.4f}")
    # print(f"  最小成本: {min(costs):.4f}")
    # print(f"  最大成本: {max(costs):.4f}")
    
    # print("\n" + "=" * 70)
    # print("✓ 测试完成")
    # print("=" * 70)