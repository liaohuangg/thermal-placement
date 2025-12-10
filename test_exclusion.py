#!/usr/bin/env python3
"""
快速测试排除约束是否有效
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from ilp_sub_solution_search import search_multiple_solutions

def main():
    json_path = Path(__file__).parent / "baseline" / "ICCAD23" / "test_input" / "5core.json"
    
    print("=" * 80)
    print(f"测试排除约束：运行 {json_path}")
    print("=" * 80)
    
    # 只搜索前5个解，快速验证
    sols = search_multiple_solutions(
        num_solutions=20, 
        min_shared_length=0.1,
        input_json_path=str(json_path),
        grid_size=1.0,
        fixed_chiplet_idx=0,
        exclude_min_diff=3.0
    )
    
    print("\n" + "=" * 80)
    print(f"共找到 {len(sols)} 个解")
    print("=" * 80)
    
    # 检查是否有相同位置的解
    positions_list = []
    for i, sol in enumerate(sols):
        positions = sorted([(k, v[0], v[1]) for k, v in sol.layout.items()])
        positions_tuple = tuple(positions)
        positions_list.append(positions_tuple)
        
        print(f"\n解 {i+1}: 目标值={sol.objective_value:.2f}")
        print(f"  位置: {[(k, f'({x:.2f}, {y:.2f})') for k, x, y in positions]}")
    
    # 检查重复
    unique_positions = set(positions_list)
    if len(unique_positions) < len(positions_list):
        print("\n" + "!" * 80)
        print("警告：发现相同位置的解！")
        print("!" * 80)
        for i, pos in enumerate(positions_list):
            if positions_list.count(pos) > 1:
                print(f"解 {i+1} 的位置重复出现")
    else:
        print("\n" + "✓" * 80)
        print("成功：所有解的位置都不同！")
        print("✓" * 80)

if __name__ == "__main__":
    main()

