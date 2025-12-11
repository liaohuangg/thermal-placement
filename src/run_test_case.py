#!/usr/bin/env python3
"""
运行测试用例的示例脚本。

用法：
    python3 run_test_case.py <json_file_path>
    
例如：
    python3 run_test_case.py ../baseline/ICCAD23/test_input/5core.json
    python3 run_test_case.py ../baseline/ICCAD23/test_input/8core.json
"""

import sys
from pathlib import Path

# 确保可以导入 ilp_sub_solution_search
sys.path.insert(0, str(Path(__file__).parent))

from ilp_sub_solution_search import search_multiple_solutions


def main():
    if len(sys.argv) < 2:
        print("用法: python3 run_test_case.py <json_file_path>")
        print("例如: python3 run_test_case.py ../baseline/ICCAD23/test_input/5core.json")
        sys.exit(1)
    
    json_path = sys.argv[1]
    json_file = Path(json_path)
    
    # 如果路径是相对路径，尝试相对于当前脚本目录
    if not json_file.is_absolute():
        json_file = Path(__file__).parent / json_path
    
    if not json_file.exists():
        print(f"错误：文件不存在: {json_file}")
        print(f"  尝试的路径: {json_file.absolute()}")
        sys.exit(1)
    
    print("=" * 80)
    print(f"运行测试用例: {json_file}")
    print("=" * 80)
    
    # 运行求解搜索
    try:
        sols = search_multiple_solutions(
            num_solutions=100, 
            min_shared_length=0.1,
            input_json_path=str(json_file.absolute()),
            grid_size=0.5,  # 使用网格化布局，grid_size=1.0（chiplet位置只能是整数坐标点）
            fixed_chiplet_idx=0,  # 固定第一个chiplet的中心位置
            min_dist_diff=2.0
        )
        
        print("=" * 80)
        print(f"共找到 {len(sols)} 个不同的 ILP 可行解。")
        print("=" * 80)
        
        # 打印每个解的信息
        for i, sol in enumerate(sols):
            print(f"\n解 {i+1}:")
            print(f"  状态: {sol.status}")
            print(f"  目标函数值: {sol.objective_value:.2f}")
            print(f"  求解时间: {sol.solve_time:.2f} 秒")
            print(f"  边界框: {sol.bounding_box[0]:.2f} x {sol.bounding_box[1]:.2f}")
        
        print("\n" + "=" * 80)
        print("求解完成！")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n错误：求解过程中出现异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
