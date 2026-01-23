#!/bin/bash
#
# 批量测试脚本：对 benchmark/test_input 目录下的所有 JSON 文件运行 batch_test_solutions_gurobi.py
#
# 用法:
#     ./run_all_tests.sh [test_name]
#     ./run_all_tests.sh all          # 运行所有测试
#     ./run_all_tests.sh acend910      # 运行单个测试
#     ./run_all_tests.sh              # 无参数时运行所有测试

log_dir="../output_gurobi/log"

# 创建日志目录（如果不存在）
mkdir -p "$log_dir"

if [ $# -gt 0 ]; then
    case "$1" in
        "all"|"ALL")
            # 运行所有测试
            if [ -f "$log_dir/acend910.log" ]; then rm -f "$log_dir/acend910.log"; fi
            touch "$log_dir/acend910.log"
            python3 batch_test_solutions_gurobi.py --num-solutions 5 --grid-size 0.01  --min-shared-length 2 --min-pair-dist-diff 0.1 --files acend910.json > "$log_dir/acend910.log" 2>&1
            
            if [ -f "$log_dir/cpu-dram.log" ]; then rm -f "$log_dir/cpu-dram.log"; fi
            touch "$log_dir/cpu-dram.log"
            python3 batch_test_solutions_gurobi.py --num-solutions 5 --grid-size 0.01  --min-shared-length 2 --min-pair-dist-diff 0.1 --files cpu-dram.json > "$log_dir/cpu-dram.log" 2>&1
            
            if [ -f "$log_dir/hp11_m.log" ]; then rm -f "$log_dir/hp11_m.log"; fi
            touch "$log_dir/hp11_m.log"
            python3 batch_test_solutions_gurobi.py --num-solutions 5 --grid-size 0.01  --min-shared-length 2 --min-pair-dist-diff 0.1 --files hp11_m.json > "$log_dir/hp11_m.log" 2>&1
            
            if [ -f "$log_dir/hp6_m.log" ]; then rm -f "$log_dir/hp6_m.log"; fi
            touch "$log_dir/hp6_m.log"
            python3 batch_test_solutions_gurobi.py --num-solutions 5 --grid-size 0.01  --min-shared-length 2 --min-pair-dist-diff 0.1 --files hp6_m.json > "$log_dir/hp6_m.log" 2>&1
            
            if [ -f "$log_dir/hp8_m.log" ]; then rm -f "$log_dir/hp8_m.log"; fi
            touch "$log_dir/hp8_m.log"
            python3 batch_test_solutions_gurobi.py --num-solutions 5 --grid-size 0.01  --min-shared-length 2 --min-pair-dist-diff 0.1 --files hp8_m.json > "$log_dir/hp8_m.log" 2>&1
            
            if [ -f "$log_dir/multigpu.log" ]; then rm -f "$log_dir/multigpu.log"; fi
            touch "$log_dir/multigpu.log"
            python3 batch_test_solutions_gurobi.py --num-solutions 5 --grid-size 0.01  --min-shared-length 2 --min-pair-dist-diff 0.1 --files multigpu.json > "$log_dir/multigpu.log" 2>&1
            
            if [ -f "$log_dir/syn1.log" ]; then rm -f "$log_dir/syn1.log"; fi
            touch "$log_dir/syn1.log"
            python3 batch_test_solutions_gurobi.py --num-solutions 5 --grid-size 0.01  --min-shared-length 2 --min-pair-dist-diff 0.1 --files syn1.json > "$log_dir/syn1.log" 2>&1
            
            if [ -f "$log_dir/syn2.log" ]; then rm -f "$log_dir/syn2.log"; fi
            touch "$log_dir/syn2.log"
            python3 batch_test_solutions_gurobi.py --num-solutions 5 --grid-size 0.01  --min-shared-length 2 --min-pair-dist-diff 0.1 --files syn2.json > "$log_dir/syn2.log" 2>&1
            
            if [ -f "$log_dir/syn3.log" ]; then rm -f "$log_dir/syn3.log"; fi
            touch "$log_dir/syn3.log"
            python3 batch_test_solutions_gurobi.py --num-solutions 5 --grid-size 0.01  --min-shared-length 2 --min-pair-dist-diff 0.1 --files syn3.json > "$log_dir/syn3.log" 2>&1
            
            if [ -f "$log_dir/syn4.log" ]; then rm -f "$log_dir/syn4.log"; fi
            touch "$log_dir/syn4.log"
            python3 batch_test_solutions_gurobi.py --num-solutions 5 --grid-size 0.01  --min-shared-length 2 --min-pair-dist-diff 0.1 --files syn4.json > "$log_dir/syn4.log" 2>&1
            
            if [ -f "$log_dir/syn5.log" ]; then rm -f "$log_dir/syn5.log"; fi
            touch "$log_dir/syn5.log"
            python3 batch_test_solutions_gurobi.py --num-solutions 5 --grid-size 0.01  --min-shared-length 2 --min-pair-dist-diff 0.1 --files syn5.json > "$log_dir/syn5.log" 2>&1
            
            if [ -f "$log_dir/syn6.log" ]; then rm -f "$log_dir/syn6.log"; fi
            touch "$log_dir/syn6.log"
            python3 batch_test_solutions_gurobi.py --num-solutions 5 --grid-size 0.01  --min-shared-length 2 --min-pair-dist-diff 0.1 --files syn6.json > "$log_dir/syn6.log" 2>&1
            
            if [ -f "$log_dir/xerox6_m.log" ]; then rm -f "$log_dir/xerox6_m.log"; fi
            touch "$log_dir/xerox6_m.log"
            python3 batch_test_solutions_gurobi.py --num-solutions 5 --grid-size 0.01  --min-shared-length 2 --min-pair-dist-diff 0.1 --files xerox6_m.json > "$log_dir/xerox6_m.log" 2>&1
            
            if [ -f "$log_dir/xerox7_m.log" ]; then rm -f "$log_dir/xerox7_m.log"; fi
            touch "$log_dir/xerox7_m.log"
            python3 batch_test_solutions_gurobi.py --num-solutions 5 --grid-size 0.01  --min-shared-length 2 --min-pair-dist-diff 0.1 --files xerox7_m.json > "$log_dir/xerox7_m.log" 2>&1
            
            if [ -f "$log_dir/xerox8_m.log" ]; then rm -f "$log_dir/xerox8_m.log"; fi
            touch "$log_dir/xerox8_m.log"
            python3 batch_test_solutions_gurobi.py --num-solutions 5 --grid-size 0.01  --min-shared-length 2 --min-pair-dist-diff 0.1 --files xerox8_m.json > "$log_dir/xerox8_m.log" 2>&1
            ;;
        "acend910")
            if [ -f "$log_dir/acend910.log" ]; then rm -f "$log_dir/acend910.log"; fi
            touch "$log_dir/acend910.log"
            python3 batch_test_solutions_gurobi.py --num-solutions 5 --grid-size 0.01  --min-shared-length 2 --min-pair-dist-diff 0.1 --files acend910.json > "$log_dir/acend910.log" 2>&1
            ;;
        "cpu-dram")
            if [ -f "$log_dir/cpu-dram.log" ]; then rm -f "$log_dir/cpu-dram.log"; fi
            touch "$log_dir/cpu-dram.log"
            python3 batch_test_solutions_gurobi.py --num-solutions 5 --grid-size 0.01  --min-shared-length 2 --min-pair-dist-diff 0.1 --files cpu-dram.json > "$log_dir/cpu-dram.log" 2>&1
            ;;
        "hp11_m")
            if [ -f "$log_dir/hp11_m.log" ]; then rm -f "$log_dir/hp11_m.log"; fi
            touch "$log_dir/hp11_m.log"
            python3 batch_test_solutions_gurobi.py --num-solutions 5 --grid-size 0.01  --min-shared-length 2 --min-pair-dist-diff 0.1 --files hp11_m.json > "$log_dir/hp11_m.log" 2>&1
            ;;
        "hp6_m")
            if [ -f "$log_dir/hp6_m.log" ]; then rm -f "$log_dir/hp6_m.log"; fi
            touch "$log_dir/hp6_m.log"
            python3 batch_test_solutions_gurobi.py --num-solutions 5 --grid-size 0.01  --min-shared-length 2 --min-pair-dist-diff 0.1 --files hp6_m.json > "$log_dir/hp6_m.log" 2>&1
            ;;
        "hp8_m")
            if [ -f "$log_dir/hp8_m.log" ]; then rm -f "$log_dir/hp8_m.log"; fi
            touch "$log_dir/hp8_m.log"
            python3 batch_test_solutions_gurobi.py --num-solutions 5 --grid-size 0.01  --min-shared-length 2 --min-pair-dist-diff 0.1 --files hp8_m.json > "$log_dir/hp8_m.log" 2>&1
            ;;
        "multigpu")
            if [ -f "$log_dir/multigpu.log" ]; then rm -f "$log_dir/multigpu.log"; fi
            touch "$log_dir/multigpu.log"
            python3 batch_test_solutions_gurobi.py --num-solutions 5 --grid-size 0.01  --min-shared-length 2 --min-pair-dist-diff 0.1 --files multigpu.json > "$log_dir/multigpu.log" 2>&1
            ;;
        "syn1")
            if [ -f "$log_dir/syn1.log" ]; then rm -f "$log_dir/syn1.log"; fi
            touch "$log_dir/syn1.log"
            python3 batch_test_solutions_gurobi.py --num-solutions 5 --grid-size 0.01  --min-shared-length 2 --min-pair-dist-diff 0.1 --files syn1.json > "$log_dir/syn1.log" 2>&1
            ;;
        "syn2")
            if [ -f "$log_dir/syn2.log" ]; then rm -f "$log_dir/syn2.log"; fi
            touch "$log_dir/syn2.log"
            python3 batch_test_solutions_gurobi.py --num-solutions 5 --grid-size 0.01  --min-shared-length 2 --min-pair-dist-diff 0.1 --files syn2.json > "$log_dir/syn2.log" 2>&1
            ;;
        "syn3")
            if [ -f "$log_dir/syn3.log" ]; then rm -f "$log_dir/syn3.log"; fi
            touch "$log_dir/syn3.log"
            python3 batch_test_solutions_gurobi.py --num-solutions 5 --grid-size 0.01  --min-shared-length 2 --min-pair-dist-diff 0.1 --files syn3.json > "$log_dir/syn3.log" 2>&1
            ;;
        "syn4")
            if [ -f "$log_dir/syn4.log" ]; then rm -f "$log_dir/syn4.log"; fi
            touch "$log_dir/syn4.log"
            python3 batch_test_solutions_gurobi.py --num-solutions 5 --grid-size 0.01  --min-shared-length 2 --min-pair-dist-diff 0.1 --files syn4.json > "$log_dir/syn4.log" 2>&1
            ;;
        "syn5")
            if [ -f "$log_dir/syn5.log" ]; then rm -f "$log_dir/syn5.log"; fi
            touch "$log_dir/syn5.log"
            python3 batch_test_solutions_gurobi.py --num-solutions 5 --grid-size 0.01  --min-shared-length 2 --min-pair-dist-diff 0.1 --files syn5.json > "$log_dir/syn5.log" 2>&1
            ;;
        "syn6")
            if [ -f "$log_dir/syn6.log" ]; then rm -f "$log_dir/syn6.log"; fi
            touch "$log_dir/syn6.log"
            python3 batch_test_solutions_gurobi.py --num-solutions 5 --grid-size 0.01  --min-shared-length 2 --min-pair-dist-diff 0.1 --files syn6.json > "$log_dir/syn6.log" 2>&1
            ;;
        "xerox6_m")
            if [ -f "$log_dir/xerox6_m.log" ]; then rm -f "$log_dir/xerox6_m.log"; fi
            touch "$log_dir/xerox6_m.log"
            python3 batch_test_solutions_gurobi.py --num-solutions 5 --grid-size 0.01  --min-shared-length 2 --min-pair-dist-diff 0.1 --files xerox6_m.json > "$log_dir/xerox6_m.log" 2>&1
            ;;
        "xerox7_m")
            if [ -f "$log_dir/xerox7_m.log" ]; then rm -f "$log_dir/xerox7_m.log"; fi
            touch "$log_dir/xerox7_m.log"
            python3 batch_test_solutions_gurobi.py --num-solutions 5 --grid-size 0.01  --min-shared-length 2 --min-pair-dist-diff 0.1 --files xerox7_m.json > "$log_dir/xerox7_m.log" 2>&1
            ;;
        "xerox8_m")
            if [ -f "$log_dir/xerox8_m.log" ]; then rm -f "$log_dir/xerox8_m.log"; fi
            touch "$log_dir/xerox8_m.log"
            python3 batch_test_solutions_gurobi.py --num-solutions 5 --grid-size 0.01  --min-shared-length 2 --min-pair-dist-diff 0.1 --files xerox8_m.json > "$log_dir/xerox8_m.log" 2>&1
            ;;
        *)
            echo "错误: 未知的测试名称: $1"
            echo ""
            echo "可用的测试:"
            echo "  all, acend910, cpu-dram, hp11_m, hp6_m, hp8_m, multigpu,"
            echo "  syn1, syn2, syn3, syn4, syn5, syn6,"
            echo "  xerox6_m, xerox7_m, xerox8_m"
            exit 1
            ;;
    esac
else
    # 如果没有参数，运行所有测试
    if [ -f "$log_dir/acend910.log" ]; then rm -f "$log_dir/acend910.log"; fi
    touch "$log_dir/acend910.log"
    python3 batch_test_solutions_gurobi.py --num-solutions 5 --grid-size 0.01  --min-shared-length 2 --min-pair-dist-diff 0.1 --files acend910.json > "$log_dir/acend910.log" 2>&1
    
    if [ -f "$log_dir/cpu-dram.log" ]; then rm -f "$log_dir/cpu-dram.log"; fi
    touch "$log_dir/cpu-dram.log"
    python3 batch_test_solutions_gurobi.py --num-solutions 5 --grid-size 0.01  --min-shared-length 2 --min-pair-dist-diff 0.1 --files cpu-dram.json > "$log_dir/cpu-dram.log" 2>&1
    
    if [ -f "$log_dir/hp11_m.log" ]; then rm -f "$log_dir/hp11_m.log"; fi
    touch "$log_dir/hp11_m.log"
    python3 batch_test_solutions_gurobi.py --num-solutions 5 --grid-size 0.01  --min-shared-length 2 --min-pair-dist-diff 0.1 --files hp11_m.json > "$log_dir/hp11_m.log" 2>&1
    
    if [ -f "$log_dir/hp6_m.log" ]; then rm -f "$log_dir/hp6_m.log"; fi
    touch "$log_dir/hp6_m.log"
    python3 batch_test_solutions_gurobi.py --num-solutions 5 --grid-size 0.01  --min-shared-length 2 --min-pair-dist-diff 0.1 --files hp6_m.json > "$log_dir/hp6_m.log" 2>&1
    
    if [ -f "$log_dir/hp8_m.log" ]; then rm -f "$log_dir/hp8_m.log"; fi
    touch "$log_dir/hp8_m.log"
    python3 batch_test_solutions_gurobi.py --num-solutions 5 --grid-size 0.01  --min-shared-length 2 --min-pair-dist-diff 0.1 --files hp8_m.json > "$log_dir/hp8_m.log" 2>&1
    
    if [ -f "$log_dir/multigpu.log" ]; then rm -f "$log_dir/multigpu.log"; fi
    touch "$log_dir/multigpu.log"
    python3 batch_test_solutions_gurobi.py --num-solutions 5 --grid-size 0.01  --min-shared-length 2 --min-pair-dist-diff 0.1 --files multigpu.json > "$log_dir/multigpu.log" 2>&1
    
    if [ -f "$log_dir/syn1.log" ]; then rm -f "$log_dir/syn1.log"; fi
    touch "$log_dir/syn1.log"
    python3 batch_test_solutions_gurobi.py --num-solutions 5 --grid-size 0.01  --min-shared-length 2 --min-pair-dist-diff 0.1 --files syn1.json > "$log_dir/syn1.log" 2>&1
    
    if [ -f "$log_dir/syn2.log" ]; then rm -f "$log_dir/syn2.log"; fi
    touch "$log_dir/syn2.log"
    python3 batch_test_solutions_gurobi.py --num-solutions 5 --grid-size 0.01  --min-shared-length 2 --min-pair-dist-diff 0.1 --files syn2.json > "$log_dir/syn2.log" 2>&1
    
    if [ -f "$log_dir/syn3.log" ]; then rm -f "$log_dir/syn3.log"; fi
    touch "$log_dir/syn3.log"
    python3 batch_test_solutions_gurobi.py --num-solutions 5 --grid-size 0.01  --min-shared-length 2 --min-pair-dist-diff 0.1 --files syn3.json > "$log_dir/syn3.log" 2>&1
    
    if [ -f "$log_dir/syn4.log" ]; then rm -f "$log_dir/syn4.log"; fi
    touch "$log_dir/syn4.log"
    python3 batch_test_solutions_gurobi.py --num-solutions 5 --grid-size 0.01  --min-shared-length 2 --min-pair-dist-diff 0.1 --files syn4.json > "$log_dir/syn4.log" 2>&1
    
    if [ -f "$log_dir/syn5.log" ]; then rm -f "$log_dir/syn5.log"; fi
    touch "$log_dir/syn5.log"
    python3 batch_test_solutions_gurobi.py --num-solutions 5 --grid-size 0.01  --min-shared-length 2 --min-pair-dist-diff 0.1 --files syn5.json > "$log_dir/syn5.log" 2>&1
    
    if [ -f "$log_dir/syn6.log" ]; then rm -f "$log_dir/syn6.log"; fi
    touch "$log_dir/syn6.log"
    python3 batch_test_solutions_gurobi.py --num-solutions 5 --grid-size 0.01  --min-shared-length 2 --min-pair-dist-diff 0.1 --files syn6.json > "$log_dir/syn6.log" 2>&1
    
    if [ -f "$log_dir/xerox6_m.log" ]; then rm -f "$log_dir/xerox6_m.log"; fi
    touch "$log_dir/xerox6_m.log"
    python3 batch_test_solutions_gurobi.py --num-solutions 5 --grid-size 0.01  --min-shared-length 2 --min-pair-dist-diff 0.1 --files xerox6_m.json > "$log_dir/xerox6_m.log" 2>&1
    
    if [ -f "$log_dir/xerox7_m.log" ]; then rm -f "$log_dir/xerox7_m.log"; fi
    touch "$log_dir/xerox7_m.log"
    python3 batch_test_solutions_gurobi.py --num-solutions 5 --grid-size 0.01  --min-shared-length 2 --min-pair-dist-diff 0.1 --files xerox7_m.json > "$log_dir/xerox7_m.log" 2>&1
    
    if [ -f "$log_dir/xerox8_m.log" ]; then rm -f "$log_dir/xerox8_m.log"; fi
    touch "$log_dir/xerox8_m.log"
    python3 batch_test_solutions_gurobi.py --num-solutions 5 --grid-size 0.01  --min-shared-length 2 --min-pair-dist-diff 0.1 --files xerox8_m.json > "$log_dir/xerox8_m.log" 2>&1
fi