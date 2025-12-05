"""
12core多次运行SA_1寻找最优解

策略:
1. 多次运行SA_1，每次使用不同的随机种子
2. 收集所有找到的合法解
3. 从合法解中选择利用率最优的
"""
import sys
import os
# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from unit import load_problem_from_json, best_utilization, visualize_layout_with_bridges
from Generate_initial_TCG import generate_initial_TCG
from TCG import generate_layout_from_tcg
from Legality_optimized_SA import cost_legal, count_emib_closure_edges, count_legal_emib_edges, SA_1
import random
import time


def run_multiple_SA_1(problem, num_runs=10, max_iterations=50000, 
                      initial_temp=200.0, cooling_rate=0.98, 
                      alpha_c=1.0, beta_l=10.0, use_legalize=True, verbose=False):
    """
    多次运行SA_1寻找最优解
    
    Args:
        problem: 布局问题
        num_runs: 运行次数
        max_iterations: 每次SA的最大迭代次数
        initial_temp: 初始温度
        cooling_rate: 冷却速率
        alpha_c: 闭包边惩罚系数
        beta_l: 非法边惩罚系数
        use_legalize: 是否使用legalize_tcg加速
        verbose: 是否显示详细信息
        
    Returns:
        all_legal_tcgs: 所有合法TCG列表
        all_legal_layouts: 所有合法布局列表
        run_stats: 每次运行的统计信息
    """
    all_legal_tcgs = []
    all_legal_layouts = []
    run_stats = []
    
    print(f"{'='*70}")
    print(f"开始多次运行SA_1优化")
    print(f"{'='*70}")
    print(f"问题规模: {len(problem.chiplets)}个芯片, {problem.connection_graph.number_of_edges()}个EMIB连接")
    print(f"运行次数: {num_runs}")
    print(f"每次参数: max_iter={max_iterations}, T0={initial_temp}, cooling={cooling_rate}")
    print(f"成本参数: α_c={alpha_c}, β_l={beta_l}")
    print(f"使用legalize: {use_legalize}")
    print(f"{'='*70}\n")
    
    for run_idx in range(num_runs):
        print(f"\n{'─'*70}")
        print(f"第 {run_idx + 1}/{num_runs} 次运行")
        print(f"{'─'*70}")
        
        # 设置不同的随机种子
        seed = int(time.time() * 1000) % 100000 + run_idx
        random.seed(seed)
        print(f"随机种子: {seed}")
        
        # 生成初始TCG
        tcg = generate_initial_TCG(problem, seed=seed)
        layout = generate_layout_from_tcg(tcg, problem)
        
        # 计算初始成本
        initial_cost = cost_legal(tcg, problem, layout, alpha_c, beta_l)
        emib_closure = count_emib_closure_edges(tcg, problem)
        legal_emib = count_legal_emib_edges(tcg, problem, layout)
        total_emib = problem.connection_graph.number_of_edges()
        
        print(f"初始状态: cost={initial_cost:.4f}, EMIB闭包={emib_closure}/{total_emib}, 合法EMIB={legal_emib}/{total_emib}")
        
        # 运行SA_1
        start_time = time.time()
        legal_tcgs, legal_layouts, final_cost = SA_1(
            tcg, problem,
            max_iterations=max_iterations,
            initial_temp=initial_temp,
            cooling_rate=cooling_rate,
            alpha_c=alpha_c,
            beta_l=beta_l,
            use_legalize=use_legalize,
            verbose=verbose
        )
        elapsed_time = time.time() - start_time
        
        # 统计结果
        num_legal = len(legal_tcgs)
        print(f"\n结果: 找到{num_legal}个合法解, 耗时{elapsed_time:.2f}秒, 最终成本={final_cost:.4f}")
        
        # 收集合法解
        if num_legal > 0:
            all_legal_tcgs.extend(legal_tcgs)
            all_legal_layouts.extend(legal_layouts)
            print(f"✓ 累计找到 {len(all_legal_tcgs)} 个合法解")
        else:
            print(f"⚠ 本次未找到合法解")
        
        # 保存统计信息
        run_stats.append({
            'run': run_idx + 1,
            'seed': seed,
            'initial_cost': initial_cost,
            'final_cost': final_cost,
            'num_legal': num_legal,
            'time': elapsed_time
        })
    
    return all_legal_tcgs, all_legal_layouts, run_stats


def print_summary(run_stats, all_legal_tcgs):
    """打印运行总结"""
    print(f"\n\n{'='*70}")
    print(f"运行总结")
    print(f"{'='*70}")
    
    total_runs = len(run_stats)
    successful_runs = sum(1 for stat in run_stats if stat['num_legal'] > 0)
    total_legal = sum(stat['num_legal'] for stat in run_stats)
    total_time = sum(stat['time'] for stat in run_stats)
    
    print(f"总运行次数: {total_runs}")
    print(f"成功运行次数: {successful_runs} ({successful_runs/total_runs*100:.1f}%)")
    print(f"找到合法解总数: {total_legal}")
    print(f"总耗时: {total_time:.2f}秒 (平均每次{total_time/total_runs:.2f}秒)")
    
    if total_legal > 0:
        avg_legal_per_success = total_legal / successful_runs if successful_runs > 0 else 0
        print(f"平均每次成功找到: {avg_legal_per_success:.1f}个合法解")
    
    print(f"\n每次运行详情:")
    print(f"{'运行':^6} {'种子':^8} {'初始成本':^10} {'最终成本':^10} {'合法解数':^10} {'耗时(s)':^10}")
    print(f"{'-'*70}")
    for stat in run_stats:
        print(f"{stat['run']:^6} {stat['seed']:^8} {stat['initial_cost']:^10.4f} "
              f"{stat['final_cost']:^10.4f} {stat['num_legal']:^10} {stat['time']:^10.2f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    # 加载12core问题
    print("\n加载12core问题...")
    problem = load_problem_from_json("../test_input/10core.json")
    
    # 12core问题非常困难，需要更激进的参数
    # 问题分析: 从前7次运行看，成本只能降到6.5-7.0
    # 这意味着还有约45-50%的EMIB边不合法
    # 建议策略: 先不用legalize (避免deepcopy性能问题)，纯SA搜索
    
    print("\n建议: 12core问题非常困难，可以尝试:")
    print("1. 不使用legalize (use_legalize=False) 提高速度")
    print("2. 增加运行次数，用多样性弥补单次的困难")
    print("3. 调整参数后多次尝试\n")
    
    num_runs = 10 # 先运行10次看效果
    all_legal_tcgs, all_legal_layouts, run_stats = run_multiple_SA_1(
        problem,
        num_runs=num_runs,
        max_iterations=50000,   # 10万次迭代
        initial_temp=200.0,      # 高初始温度
        cooling_rate=0.95,      # 慢冷却
        alpha_c=18.0,             # 提高闭包边惩罚
        beta_l=2.0,             # 加大非法边惩罚
        use_legalize=True,      # 不使用legalize，提高性能
        verbose=False           # 显示详细信息
    )
    
    # 打印总结
    print_summary(run_stats, all_legal_tcgs)
    
    # 如果找到合法解，选择最优的
    if len(all_legal_tcgs) > 0:
        print(f"\n{'='*70}")
        print(f"选择最优解")
        print(f"{'='*70}")
        
        # 找到利用率最优的解
        best_idx, best_layout, best_util = best_utilization(all_legal_tcgs, all_legal_layouts)
        best_tcg = all_legal_tcgs[best_idx]
        
        print(f"最优解索引: {best_idx + 1}/{len(all_legal_tcgs)}")
        print(f"最优利用率: {best_util:.2f}%")
        
        # 验证成本
        verify_cost = cost_legal(best_tcg, problem, best_layout, alpha_c=1.0, beta_l=10.0)
        print(f"验证成本: {verify_cost:.6f}")
        
        # 计算布局信息
        chip_area = sum(chip.width * chip.height for chip in problem.chiplets.values())
        max_x = max(chip.x + chip.width for chip in best_layout.values())
        max_y = max(chip.y + chip.height for chip in best_layout.values())
        layout_area = max_x * max_y
        
        print(f"\n布局详情:")
        print(f"  芯片总面积: {chip_area:.2f}")
        print(f"  布局尺寸: {max_x:.2f} x {max_y:.2f}")
        print(f"  布局面积: {layout_area:.2f}")
        print(f"  利用率: {chip_area/layout_area*100:.2f}%")
        
        # 可视化最优解
        output_file = "../output/12core_best_solution.png"
        visualize_layout_with_bridges(best_layout, problem, output_file)
        print(f"\n✓ 最优布局已保存到: {output_file}")
        
    else:
        print(f"\n⚠ 所有运行均未找到合法解")
        print(f"建议:")
        print(f"  1. 增加运行次数 (num_runs)")
        print(f"  2. 增加每次迭代次数 (max_iterations)")
        print(f"  3. 调高初始温度 (initial_temp)")
        print(f"  4. 调慢冷却速率 (cooling_rate更接近1.0)")
