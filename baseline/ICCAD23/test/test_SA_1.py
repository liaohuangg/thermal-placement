"""
测试SA_1模拟退火算法
"""
import sys
import os
# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from unit import load_problem_from_json, visualize_layout_with_bridges, best_utilization
from Generate_initial_TCG import generate_initial_TCG
from TCG import generate_layout_from_tcg
from Legality_optimized_SA import cost_legal, count_emib_closure_edges, count_legal_emib_edges, SA_1, _generate_neighbor_tcg

# # 测试3芯片案例
# print("\n" + "="*70)
# print("测试案例: 3芯片系统")
# print("="*70)

# problem_3 = load_problem_from_json("../test_input/3core.json")

# # 生成初始TCG
# tcg_3 = generate_initial_TCG(problem_3)
# layout_3 = generate_layout_from_tcg(tcg_3, problem_3)

# print(f"\n初始TCG状态:")
# cost_3 = cost_legal(tcg_3, problem_3, layout_3, alpha_c=1.0, beta_l=2.0)
# print(f"  初始成本: {cost_3:.4f}")

# emib_closure_3 = count_emib_closure_edges(tcg_3, problem_3)
# legal_emib_3 = count_legal_emib_edges(tcg_3, problem_3, layout_3)
# total_emib_3 = problem_3.connection_graph.number_of_edges()

# print(f"  EMIB边总数: {total_emib_3}")
# print(f"  EMIB闭包边数: {emib_closure_3}")
# print(f"  合法EMIB边数: {legal_emib_3}")
# print(f"  Pc = {emib_closure_3}/{total_emib_3} = {emib_closure_3/total_emib_3:.4f}")
# print(f"  Pl = {total_emib_3-legal_emib_3}/{total_emib_3} = {(total_emib_3-legal_emib_3)/total_emib_3:.4f}")

# # 运行SA_1
# print("\n" + "-"*70)
# print("运行SA_1模拟退火...")
# print("-"*70)

# legal_tcgs, legal_layouts, final_cost = SA_1(
#     tcg_3, problem_3,
#     max_iterations=5000,
#     initial_temp=50.0,
#     cooling_rate=0.95,
#     alpha_c=1.0,
#     beta_l=2.0,
#     verbose=True
# )

# print("\n" + "="*70)
# print("SA_1 结果总结:")
# print("="*70)
# print(f"找到合法解数量: {len(legal_tcgs)}")
# print(f"最终成本: {final_cost:.4f}")

# if len(legal_tcgs) > 0:
#     print(f"\n✓ 成功找到合法TCG!")
#     for i, (tcg, layout) in enumerate(zip(legal_tcgs, legal_layouts)):
#         cost = cost_legal(tcg, problem_3, layout, alpha_c=1.0, beta_l=2.0)
#         print(f"  解 {i+1}: cost = {cost:.6f}")
# else:
#     print(f"\n⚠ 未找到完全合法的解,最佳成本为 {final_cost:.4f}")




# 测试8芯片案例--测试维度：利用率
print("\n\n" + "="*70)
print("测试案例: 8芯片系统")
print("="*70)

problem_8 = load_problem_from_json("../test_input/6core.json")

tcg_8 = generate_initial_TCG(problem_8, seed=None)
layout_8 = generate_layout_from_tcg(tcg_8, problem_8)

print(f"\n初始TCG状态:")
cost_8 = cost_legal(tcg_8, problem_8, layout_8, alpha_c=1.0, beta_l=2.0)
print(f"  初始成本: {cost_8:.4f}")

emib_closure_8 = count_emib_closure_edges(tcg_8, problem_8)
legal_emib_8 = count_legal_emib_edges(tcg_8, problem_8, layout_8)
total_emib_8 = problem_8.connection_graph.number_of_edges()

print(f"  EMIB边总数: {total_emib_8}")
print(f"  EMIB闭包边数: {emib_closure_8}")
print(f"  合法EMIB边数: {legal_emib_8}")

print("\n" + "-"*70)
print("运行SA_1模拟退火...")
print("-"*70)

legal_tcgs_8, legal_layouts_8, final_cost_8 = SA_1(
    tcg_8, problem_8,
    max_iterations=50000,
    initial_temp=100.0,
    cooling_rate=0.95,
    alpha_c=10,
    beta_l=3,
    verbose=True
)
print("\n" + "="*70)
print("SA_1 结果总结:")
print("="*70)
print(f"找到合法解数量: {len(legal_tcgs_8)}")
print(f"最终成本: {final_cost_8:.4f}")

if len(legal_tcgs_8) > 0:
    print(f"\n✓ 成功找到{len(legal_tcgs_8)}个合法TCG!")
    
    # 显示所有解的成本
    for i, (tcg, layout) in enumerate(zip(legal_tcgs_8, legal_layouts_8)):
        verify_cost = cost_legal(tcg, problem_8, layout, alpha_c=1.0, beta_l=2.0)
        print(f"  解 {i+1}: cost = {verify_cost:.6f}")
    
    # 选择利用率最高的解
    best_idx, best_layout, best_util = best_utilization(legal_tcgs_8, legal_layouts_8)
    print(f"\n最佳利用率解:")
    print(f"  索引: {best_idx + 1}")
    print(f"  利用率: {best_util:.2f}%")
    
    # 可视化利用率最高的解
    visualize_layout_with_bridges(best_layout, problem_8, "../output/SA_1_8core_best_util.png")
    print(f"\n最佳利用率布局已保存到: ../output/SA_1_8core_best_util.png")
else:
    print(f"\n⚠ 未找到完全合法的解,最佳成本为 {final_cost_8:.4f}")
   

# #测试维度：最短线长
# print("\n\n" + "="*70)
# print("测试案例: 8芯片系统 - 最短线长优化")
# print("="*70)

# problem_8_wl = load_problem_from_json("../test_input/6core.json")

# tcg_8_wl = generate_initial_TCG(problem_8_wl, seed=None)
# layout_8_wl = generate_layout_from_tcg(tcg_8_wl, problem_8_wl)

# print(f"\n初始TCG状态:")
# cost_8_wl = cost_legal(tcg_8_wl, problem_8_wl, layout_8_wl, alpha_c=1.0, beta_l=2.0)
# print(f"  初始成本: {cost_8_wl:.4f}")

# emib_closure_8_wl = count_emib_closure_edges(tcg_8_wl, problem_8_wl)
# legal_emib_8_wl = count_legal_emib_edges(tcg_8_wl, problem_8_wl, layout_8_wl)
# total_emib_8_wl = problem_8_wl.connection_graph.number_of_edges()

# print(f"  EMIB边总数: {total_emib_8_wl}")
# print(f"  EMIB闭包边数: {emib_closure_8_wl}")
# print(f"  合法EMIB边数: {legal_emib_8_wl}")

# print("\n" + "-"*70)
# print("运行SA_1模拟退火...")
# print("-"*70)

# legal_tcgs_8_wl, legal_layouts_8_wl, final_cost_8_wl = SA_1(
#     tcg_8_wl, problem_8_wl,
#     max_iterations=50000,
#     initial_temp=100.0,
#     cooling_rate=0.95,
#     alpha_c=10,
#     beta_l=3,
#     verbose=True
# )
# print("\n" + "="*70)
# print("SA_1 结果总结:")
# print("="*70)
# print(f"找到合法解数量: {len(legal_tcgs_8_wl)}")
# print(f"最终成本: {final_cost_8_wl:.4f}")

# if len(legal_tcgs_8_wl) > 0:
#     print(f"\n✓ 成功找到{len(legal_tcgs_8_wl)}个合法TCG!")
    
#     # 显示所有解的成本和线长
#     from unit import calculate_wirelength
    
#     for i, (tcg, layout) in enumerate(zip(legal_tcgs_8_wl, legal_layouts_8_wl)):
#         verify_cost = cost_legal(tcg, problem_8_wl, layout, alpha_c=1.0, beta_l=2.0)
#         wirelength = calculate_wirelength(layout, problem_8_wl)
#         print(f"  解 {i+1}: cost = {verify_cost:.6f}, wirelength = {wirelength:.2f}")
    
#     # 选择线长最短的解
#     best_wl_idx = -1
#     best_wl_layout = None
#     best_wl_value = float('inf')
    
#     for i, layout in enumerate(legal_layouts_8_wl):
#         wirelength = calculate_wirelength(layout, problem_8_wl)
#         if wirelength < best_wl_value:
#             best_wl_value = wirelength
#             best_wl_idx = i
#             best_wl_layout = layout
    
#     print(f"\n最短线长解:")
#     print(f"  索引: {best_wl_idx + 1}")
#     print(f"  线长: {best_wl_value:.2f}")
    
#     # 可视化线长最短的解
#     visualize_layout_with_bridges(best_wl_layout, problem_8_wl, "../output/SA_1_8core_best_wirelength.png")
#     print(f"\n最短线长布局已保存到: ../output/SA_1_8core_best_wirelength.png")
# else:
#     print(f"\n⚠ 未找到完全合法的解,最佳成本为 {final_cost_8_wl:.4f}")
