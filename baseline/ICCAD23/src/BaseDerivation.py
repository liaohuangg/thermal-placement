
#完成代码：
#对于测试数据，多次运行SA，
#多次运行SA的结果，有3种情况：1.初始化的布局就合法 2.初始化布局不合法，但是通过 EMIB Legalization合法化后，得到了很多解（假设有1000个） 3.初始布局不合法，EMIB Legalization后也没有合法解
#对于情况1，保存初始布局，保存为一个相似度树的根节点
#对于情况2，将第一个合法化后的解保存为相似度树的根节点，对于其他的解如果与当前所有根节点的相似度小于<min_similarity,则把他保存为一个根节点，否则直接舍弃
#对于情况3，舍弃
#最后你返回的数据结构应该是一个森林，每个树就是一个根节点，此时任一棵树不应该有任何子节点



"""
基础导出（Base Derivation）模块

从多次SA运行结果中筛选出多样化的合法解，构建相似度树森林。
每棵树的根节点代表一个独特的布局解（相似度低于阈值）。

主要流程：
1. 多次运行SA生成大量候选解
2. 根据合法性和相似度筛选解
3. 构建相似度树森林（此阶段每棵树只有根节点）
"""

import sys
import os
import random
import time
from typing import List, Dict, Tuple

from unit import load_problem_from_json, calculate_layout_utilization
from Generate_initial_TCG import generate_initial_TCG
from TCG import TCG, generate_layout_from_tcg
from Legality_optimized_SA import cost_legal, SA_1
from GroupConstrua import SimilarityTreeNode, compute_similarity
from chiplet_model import Chiplet, LayoutProblem


def run_base_derivation(problem: LayoutProblem, 
                        num_runs: int = 10,
                        min_similarity: float = 0.7,
                        max_iterations: int = 50000,
                        initial_temp: float = 200.0,
                        cooling_rate: float = 0.98,
                        alpha_c: float = 1.0,
                        beta_l: float = 10.0,
                        verbose: bool = True) -> List[SimilarityTreeNode]:
    """
    基础导出：多次运行SA并筛选出多样化的合法解作为森林的根节点
    
    处理三种情况：
    1. 初始布局就合法：直接保存为根节点
    2. 初始布局不合法，但legalization后得到合法解：
       - 第一个解作为根节点
       - 其他解如果与所有根节点相似度都 < min_similarity，则保存为新根节点
    3. 无合法解：舍弃
    
    Args:
        problem: 布局问题
        num_runs: SA运行次数
        min_similarity: 相似度阈值（小于此值才保存为新根节点）
        max_iterations: SA最大迭代次数
        initial_temp: SA初始温度
        cooling_rate: SA冷却速率
        alpha_c: 闭包边惩罚系数
        beta_l: 非法边惩罚系数
        verbose: 是否输出详细信息
        
    Returns:
        根节点列表（森林），每个节点代表一个独特的合法布局解
    """
    if verbose:
        print("="*80)
        print("基础导出 (Base Derivation) - 构建相似度树森林")
        print("="*80)
        print(f"问题规模: {len(problem.chiplets)}个芯片, {problem.connection_graph.number_of_edges()}个连接")
        print(f"运行次数: {num_runs}")
        print(f"相似度阈值: {min_similarity} (低于此值才保存为新根节点)")
        print(f"SA参数: max_iter={max_iterations}, T0={initial_temp}, cooling={cooling_rate}")
        print("="*80 + "\n")
    
    root_nodes: List[SimilarityTreeNode] = []  # 森林的根节点列表
    total_legal_solutions = 0  # 统计找到的合法解总数
    
    for run_idx in range(num_runs):
        if verbose:
            print(f"\n{'─'*80}")
            print(f"第 {run_idx + 1}/{num_runs} 次运行")
            print(f"{'─'*80}")
        
        # 设置不同的随机种子
        seed = int(time.time() * 1000) % 100000 + run_idx
        random.seed(seed)
        
        if verbose:
            print(f"随机种子: {seed}")
        
        # 生成初始TCG
        tcg = generate_initial_TCG(problem, seed=seed)
        layout = generate_layout_from_tcg(tcg, problem)
        
        # 计算初始成本
        initial_cost = cost_legal(tcg, problem, layout, alpha_c, beta_l)
        
        if verbose:
            print(f"初始成本: {initial_cost:.4f}")
        
        # 情况1：初始布局已经合法
        if abs(initial_cost) < 1e-6:
            if verbose:
                print("✓ 初始布局已合法（情况1）")
            
            # 计算布局利用率作为成本
            utilization, _, _, _, _ = calculate_layout_utilization(layout)
            cost = -utilization  # 负值，因为利用率越高越好
            
            # 检查是否与现有根节点相似度太高
            is_unique = _is_unique_solution(layout, root_nodes, min_similarity, verbose)
            
            if is_unique:
                node = SimilarityTreeNode(
                    tcg=tcg,
                    layout=layout,
                    parent=None,
                    similarity_to_parent=None,
                    cost=cost
                )
                root_nodes.append(node)
                total_legal_solutions += 1
                
                if verbose:
                    print(f"  → 保存为根节点 #{len(root_nodes)} (利用率={utilization:.2f}%)")
            else:
                if verbose:
                    print(f"  → 与现有根节点相似度过高，舍弃")
            
            continue
        
        # 初始布局不合法，运行SA寻找合法解
        if verbose:
            print(f"初始布局不合法，运行SA寻找合法解...")
        
        start_time = time.time()
        legal_tcgs, legal_layouts, final_cost = SA_1(
            tcg, problem,
            max_iterations=max_iterations,
            initial_temp=initial_temp,
            cooling_rate=cooling_rate,
            alpha_c=alpha_c,
            beta_l=beta_l,
            use_legalize=True,
            verbose=False
        )
        elapsed_time = time.time() - start_time
        
        num_legal = len(legal_tcgs)
        
        if verbose:
            print(f"SA完成: 找到{num_legal}个合法解, 耗时{elapsed_time:.2f}秒")
        
        # 情况3：没有找到合法解
        if num_legal == 0:
            if verbose:
                print("✗ 未找到合法解（情况3），舍弃")
            continue
        
        # 情况2：找到了合法解
        if verbose:
            print(f"✓ 找到{num_legal}个合法解（情况2）")
        
        total_legal_solutions += num_legal
        
        # 处理第一个合法解：直接作为根节点
        first_layout = legal_layouts[0]
        first_tcg = legal_tcgs[0]
        utilization, _, _, _, _ = calculate_layout_utilization(first_layout)
        cost = -utilization
        
        # 检查第一个解是否唯一
        is_unique = _is_unique_solution(first_layout, root_nodes, min_similarity, verbose)
        
        if is_unique:
            node = SimilarityTreeNode(
                tcg=first_tcg,
                layout=first_layout,
                parent=None,
                similarity_to_parent=None,
                cost=cost
            )
            root_nodes.append(node)
            
            if verbose:
                print(f"  解1: 保存为根节点 #{len(root_nodes)} (利用率={utilization:.2f}%)")
        else:
            if verbose:
                print(f"  解1: 与现有根节点相似度过高，舍弃")
        
        # 处理其他合法解：只有与所有根节点相似度都小于min_similarity才保存
        added_count = 1 if is_unique else 0
        skipped_count = 0 if is_unique else 1
        
        for idx in range(1, num_legal):
            layout_i = legal_layouts[idx]
            tcg_i = legal_tcgs[idx]
            
            # 检查是否唯一
            is_unique = _is_unique_solution(layout_i, root_nodes, min_similarity, verbose=False)
            
            if is_unique:
                utilization, _, _, _, _ = calculate_layout_utilization(layout_i)
                cost = -utilization
                
                node = SimilarityTreeNode(
                    tcg=tcg_i,
                    layout=layout_i,
                    parent=None,
                    similarity_to_parent=None,
                    cost=cost
                )
                root_nodes.append(node)
                added_count += 1
                
                if verbose and added_count <= 5:  # 只打印前5个
                    print(f"  解{idx+1}: 保存为根节点 #{len(root_nodes)} (利用率={utilization:.2f}%)")
            else:
                skipped_count += 1
        
        if verbose:
            print(f"  → 本轮添加{added_count}个根节点, 跳过{skipped_count}个相似解")
    
    # 输出最终统计
    if verbose:
        print("\n" + "="*80)
        print("基础导出完成")
        print("="*80)
        print(f"总运行次数: {num_runs}")
        print(f"找到合法解总数: {total_legal_solutions}")
        print(f"森林根节点数量: {len(root_nodes)}")
        print(f"平均相似度阈值: {min_similarity}")
        
        if len(root_nodes) > 0:
            print(f"\n根节点详情:")
            for i, node in enumerate(root_nodes):
                utilization, _, _, _, _ = calculate_layout_utilization(node.layout)
                print(f"  根节点 #{i+1}: 利用率={utilization:.2f}%")
        
        print("="*80)
    
    return root_nodes


def _is_unique_solution(layout: Dict[str, Chiplet], 
                        root_nodes: List[SimilarityTreeNode],
                        min_similarity: float,
                        verbose: bool = False) -> bool:
    """
    检查布局是否与所有现有根节点的相似度都小于阈值
    
    Args:
        layout: 待检查的布局
        root_nodes: 现有根节点列表
        min_similarity: 相似度阈值
        verbose: 是否输出详细信息
        
    Returns:
        True: 该解唯一（与所有根节点相似度都 < min_similarity）
        False: 该解不唯一（存在某个根节点相似度 >= min_similarity）
    """
    if len(root_nodes) == 0:
        return True  # 没有根节点，当然是唯一的
    
    for root_node in root_nodes:
        similarity = compute_similarity(layout, root_node.layout)
        
        if verbose:
            print(f"    与根节点#{root_node.node_id}的相似度: {similarity:.4f}")
        
        if similarity >= min_similarity:
            return False  # 相似度过高，不唯一
    
    return True  # 与所有根节点相似度都小于阈值，唯一


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    print("基础导出模块测试\n")
    
    # 测试：10芯片案例
    print("="*80)
    print("测试案例: 10芯片系统")
    print("="*80)
    
    # 加载问题
    problem = load_problem_from_json("../../../benchmark/test_input/cpu-dram.json")
    
    # 运行基础导出
    root_nodes = run_base_derivation(
        problem=problem,
        num_runs=5,  # 运行5次
        min_similarity=0.5,  # 相似度阈值
        max_iterations=30000,
        initial_temp=150.0,
        cooling_rate=0.98,
        alpha_c=1.0,
        beta_l=10.0,
        verbose=True
    )
    
    print(f"\n最终结果: 构建了包含 {len(root_nodes)} 个根节点的森林")
    #把所有根节点的布局图片保存下来
    for i, root in enumerate(root_nodes):
        from unit import visualize_layout_with_bridges, save_layout_image
        image_path = f"../output/BaseDerivation/root_node_{i+1}_layout.png"
        save_layout_image(root.layout, problem, image_path)
        print(f"  根节点 #{i+1} 布局图已保存: {image_path}")
 
    
    # 验证：确保所有根节点之间相似度都小于阈值
    if len(root_nodes) > 1:
        print(f"\n验证根节点之间的相似度:")
        for i in range(len(root_nodes)):
            for j in range(i + 1, len(root_nodes)):
                sim = compute_similarity(root_nodes[i].layout, root_nodes[j].layout)
                print(f"  根节点#{i+1} vs 根节点#{j+1}: 相似度={sim:.4f}")
                
                if sim >= 0.7:
                    print(f"   相似度过高")
    
    print("\n测试完成")
