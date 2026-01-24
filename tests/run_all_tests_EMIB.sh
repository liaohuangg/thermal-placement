#!/bin/bash
#
# 批量测试脚本：对 benchmark/test_input 目录下的所有 JSON 文件运行 batch_test_solutions_gurobi_EMIB.py
#
# 用法:
#     ./run_all_tests_EMIB.sh [test_name]   # 运行单个测试
#     ./run_all_tests_EMIB.sh               # 无参数时运行所有测试

log_dir="../output_gurobi/log"
fig_dir="../output_gurobi/fig"

# 创建日志目录（如果不存在）
mkdir -p "$log_dir"

# 清除所有 *_core 子目录下的图片文件
# if [ -d "$fig_dir" ]; then
#     echo "清除 $fig_dir 目录下所有 *_core 子目录中的图片文件..."
#     find "$fig_dir" -type d -name "*_core" -exec sh -c 'rm -f "$1"/*.png "$1"/*.jpg "$1"/*.jpeg 2>/dev/null' _ {} \;
#     echo "图片清除完成。"
# fi

if [ $# -gt 0 ]; then
    case "$1" in
        "acend910")
            if [ -f "$log_dir/acend910.log" ]; then rm -f "$log_dir/acend910.log"; fi
            touch "$log_dir/acend910.log"
            python3 batch_test_solutions_gurobi_EMIB.py --min-shared-length 1 --min-pair-dist-diff 1 --files acend910.json > "$log_dir/acend910.log" 2>&1
            echo "acend910.log 求解完成"
            ;;
        "cpu-dram")
            if [ -f "$log_dir/cpu-dram.log" ]; then rm -f "$log_dir/cpu-dram.log"; fi
            touch "$log_dir/cpu-dram.log"
            python3 batch_test_solutions_gurobi_EMIB.py --min-shared-length 1 --min-pair-dist-diff 1 --files cpu-dram.json > "$log_dir/cpu-dram.log" 2>&1
            echo "cpu-dram.log 求解完成"
            ;;
        "hp11_m")
            if [ -f "$log_dir/hp11_m.log" ]; then rm -f "$log_dir/hp11_m.log"; fi
            touch "$log_dir/hp11_m.log"
            python3 batch_test_solutions_gurobi_EMIB.py --min-shared-length 1 --min-pair-dist-diff 1 --files hp11_m.json > "$log_dir/hp11_m.log" 2>&1
            echo "hp11_m.log 求解完成"
            ;;
        "hp6_m")
            if [ -f "$log_dir/hp6_m.log" ]; then rm -f "$log_dir/hp6_m.log"; fi
            touch "$log_dir/hp6_m.log"
            python3 batch_test_solutions_gurobi_EMIB.py --min-shared-length 1 --min-pair-dist-diff 1 --files hp6_m.json > "$log_dir/hp6_m.log" 2>&1
            echo "hp6_m.log 求解完成"
            ;;
        "hp8_m")
            if [ -f "$log_dir/hp8_m.log" ]; then rm -f "$log_dir/hp8_m.log"; fi
            touch "$log_dir/hp8_m.log"
            python3 batch_test_solutions_gurobi_EMIB.py --min-shared-length 1 --min-pair-dist-diff 1 --files hp8_m.json > "$log_dir/hp8_m.log" 2>&1
            echo "hp8_m.log 求解完成"
            ;;
        "multigpu")
            if [ -f "$log_dir/multigpu.log" ]; then rm -f "$log_dir/multigpu.log"; fi
            touch "$log_dir/multigpu.log"
            python3 batch_test_solutions_gurobi_EMIB.py --min-shared-length 1 --min-pair-dist-diff 1 --files multigpu.json > "$log_dir/multigpu.log" 2>&1
            echo "multigpu.log 求解完成"
            ;;
        "xerox6_m")
            if [ -f "$log_dir/xerox6_m.log" ]; then rm -f "$log_dir/xerox6_m.log"; fi
            touch "$log_dir/xerox6_m.log"
            python3 batch_test_solutions_gurobi_EMIB.py --min-shared-length 1 --min-pair-dist-diff 1 --files xerox6_m.json > "$log_dir/xerox6_m.log" 2>&1
            echo "xerox6_m.log 求解完成"
            ;;
        "xerox7_m")
            if [ -f "$log_dir/xerox7_m.log" ]; then rm -f "$log_dir/xerox7_m.log"; fi
            touch "$log_dir/xerox7_m.log"
            python3 batch_test_solutions_gurobi_EMIB.py --min-shared-length 1 --min-pair-dist-diff 1 --files xerox7_m.json > "$log_dir/xerox7_m.log" 2>&1
            echo "xerox7_m.log 求解完成"
            ;;
        "xerox8_m")
            if [ -f "$log_dir/xerox8_m.log" ]; then rm -f "$log_dir/xerox8_m.log"; fi
            touch "$log_dir/xerox8_m.log"
            python3 batch_test_solutions_gurobi_EMIB.py --min-shared-length 1 --min-pair-dist-diff 1 --files xerox8_m.json > "$log_dir/xerox8_m.log" 2>&1
            echo "xerox8_m.log 求解完成"
            ;;
        "syn1")
            if [ -f "$log_dir/syn1.log" ]; then rm -f "$log_dir/syn1.log"; fi
            touch "$log_dir/syn1.log"
            python3 batch_test_solutions_gurobi_EMIB.py --min-shared-length 1 --min-pair-dist-diff 1 --files syn1.json > "$log_dir/syn1.log" 2>&1
            echo "syn1.log 求解完成"
            ;;
        "syn2")
            if [ -f "$log_dir/syn2.log" ]; then rm -f "$log_dir/syn2.log"; fi
            touch "$log_dir/syn2.log"
            python3 batch_test_solutions_gurobi_EMIB.py --min-shared-length 1 --min-pair-dist-diff 1 --files syn2.json > "$log_dir/syn2.log" 2>&1
            echo "syn2.log 求解完成"
            ;;
        "syn3")
            if [ -f "$log_dir/syn3.log" ]; then rm -f "$log_dir/syn3.log"; fi
            touch "$log_dir/syn3.log"
            python3 batch_test_solutions_gurobi_EMIB.py --min-shared-length 1 --min-pair-dist-diff 1 --files syn3.json > "$log_dir/syn3.log" 2>&1
            echo "syn3.log 求解完成"
            ;;
        "syn4")
            if [ -f "$log_dir/syn4.log" ]; then rm -f "$log_dir/syn4.log"; fi
            touch "$log_dir/syn4.log"
            python3 batch_test_solutions_gurobi_EMIB.py --min-shared-length 1 --min-pair-dist-diff 1 --files syn4.json > "$log_dir/syn4.log" 2>&1
            echo "syn4.log 求解完成"
            ;;
        "syn5")
            if [ -f "$log_dir/syn5.log" ]; then rm -f "$log_dir/syn5.log"; fi
            touch "$log_dir/syn5.log"
            python3 batch_test_solutions_gurobi_EMIB.py --min-shared-length 1 --min-pair-dist-diff 1 --files syn5.json > "$log_dir/syn5.log" 2>&1
            echo "syn5.log 求解完成"
            ;;
        "syn6")
            if [ -f "$log_dir/syn6.log" ]; then rm -f "$log_dir/syn6.log"; fi
            touch "$log_dir/syn6.log"
            python3 batch_test_solutions_gurobi_EMIB.py --min-shared-length 1 --min-pair-dist-diff 1 --files syn6.json > "$log_dir/syn6.log" 2>&1
            echo "syn6.log 求解完成"
            ;;
        *)
            echo "错误: 未知的测试名称: $1"
            echo ""
            echo "可用的测试:"
            echo "  acend910, cpu-dram, hp11_m, hp6_m, hp8_m, multigpu,"
            echo "  xerox6_m, xerox7_m, xerox8_m,"
            echo "  syn1, syn2, syn3, syn4, syn5, syn6"
            exit 1
            ;;
    esac
else
    # 如果没有参数，运行所有测试
    if [ -f "$log_dir/acend910.log" ]; then rm -f "$log_dir/acend910.log"; fi
    touch "$log_dir/acend910.log"
    python3 batch_test_solutions_gurobi_EMIB.py --min-shared-length 1 --min-pair-dist-diff 1 --files acend910.json > "$log_dir/acend910.log" 2>&1
    echo "acend910.log 求解完成"
    
    if [ -f "$log_dir/cpu-dram.log" ]; then rm -f "$log_dir/cpu-dram.log"; fi
    touch "$log_dir/cpu-dram.log"
    python3 batch_test_solutions_gurobi_EMIB.py --min-shared-length 1 --min-pair-dist-diff 1 --files cpu-dram.json > "$log_dir/cpu-dram.log" 2>&1
    echo "cpu-dram.log 求解完成"
    
    if [ -f "$log_dir/hp6_m.log" ]; then rm -f "$log_dir/hp6_m.log"; fi
    touch "$log_dir/hp6_m.log"
    python3 batch_test_solutions_gurobi_EMIB.py --min-shared-length 1 --min-pair-dist-diff 1 --files hp6_m.json > "$log_dir/hp6_m.log" 2>&1
    echo "hp6_m.log 求解完成"
    
    if [ -f "$log_dir/hp8_m.log" ]; then rm -f "$log_dir/hp8_m.log"; fi
    touch "$log_dir/hp8_m.log"
    python3 batch_test_solutions_gurobi_EMIB.py --min-shared-length 1 --min-pair-dist-diff 1 --files hp8_m.json > "$log_dir/hp8_m.log" 2>&1
    echo "hp8_m.log 求解完成"
    
    if [ -f "$log_dir/hp11_m.log" ]; then rm -f "$log_dir/hp11_m.log"; fi
    touch "$log_dir/hp11_m.log"
    python3 batch_test_solutions_gurobi_EMIB.py --min-shared-length 1 --min-pair-dist-diff 1 --files hp11_m.json > "$log_dir/hp11_m.log" 2>&1
    echo "hp11_m.log 求解完成"
    
    if [ -f "$log_dir/multigpu.log" ]; then rm -f "$log_dir/multigpu.log"; fi
    touch "$log_dir/multigpu.log"
    python3 batch_test_solutions_gurobi_EMIB.py --min-shared-length 1 --min-pair-dist-diff 1 --files multigpu.json > "$log_dir/multigpu.log" 2>&1
    echo "multigpu.log 求解完成"
    
    if [ -f "$log_dir/xerox6_m.log" ]; then rm -f "$log_dir/xerox6_m.log"; fi
    touch "$log_dir/xerox6_m.log"
    python3 batch_test_solutions_gurobi_EMIB.py --min-shared-length 1 --min-pair-dist-diff 1 --files xerox6_m.json > "$log_dir/xerox6_m.log" 2>&1
    echo "xerox6_m.log 求解完成"
    
    if [ -f "$log_dir/xerox7_m.log" ]; then rm -f "$log_dir/xerox7_m.log"; fi
    touch "$log_dir/xerox7_m.log"
    python3 batch_test_solutions_gurobi_EMIB.py --min-shared-length 1 --min-pair-dist-diff 1 --files xerox7_m.json > "$log_dir/xerox7_m.log" 2>&1
    echo "xerox7_m.log 求解完成"
    
    if [ -f "$log_dir/xerox8_m.log" ]; then rm -f "$log_dir/xerox8_m.log"; fi
    touch "$log_dir/xerox8_m.log"
    python3 batch_test_solutions_gurobi_EMIB.py --min-shared-length 1 --min-pair-dist-diff 1 --files xerox8_m.json > "$log_dir/xerox8_m.log" 2>&1
    echo "xerox8_m.log 求解完成"

    if [ -f "$log_dir/syn1.log" ]; then rm -f "$log_dir/syn1.log"; fi
    touch "$log_dir/syn1.log"
    python3 batch_test_solutions_gurobi_EMIB.py --min-shared-length 1 --min-pair-dist-diff 1 --files syn1.json > "$log_dir/syn1.log" 2>&1
    echo "syn1.log 求解完成"

    if [ -f "$log_dir/syn2.log" ]; then rm -f "$log_dir/syn2.log"; fi
    touch "$log_dir/syn2.log"
    python3 batch_test_solutions_gurobi_EMIB.py --min-shared-length 1 --min-pair-dist-diff 1 --files syn2.json > "$log_dir/syn2.log" 2>&1
    echo "syn2.log 求解完成"

    if [ -f "$log_dir/syn3.log" ]; then rm -f "$log_dir/syn3.log"; fi
    touch "$log_dir/syn3.log"
    python3 batch_test_solutions_gurobi_EMIB.py --min-shared-length 1 --min-pair-dist-diff 1 --files syn3.json > "$log_dir/syn3.log" 2>&1
    echo "syn3.log 求解完成"

    if [ -f "$log_dir/syn4.log" ]; then rm -f "$log_dir/syn4.log"; fi
    touch "$log_dir/syn4.log"
    python3 batch_test_solutions_gurobi_EMIB.py --min-shared-length 1 --min-pair-dist-diff 1 --files syn4.json > "$log_dir/syn4.log" 2>&1
    echo "syn4.log 求解完成"

    if [ -f "$log_dir/syn5.log" ]; then rm -f "$log_dir/syn5.log"; fi
    touch "$log_dir/syn5.log"
    python3 batch_test_solutions_gurobi_EMIB.py --min-shared-length 1 --min-pair-dist-diff 1 --files syn5.json > "$log_dir/syn5.log" 2>&1
    echo "syn5.log 求解完成"

    if [ -f "$log_dir/syn6.log" ]; then rm -f "$log_dir/syn6.log"; fi
    touch "$log_dir/syn6.log"
    python3 batch_test_solutions_gurobi_EMIB.py --min-shared-length 1 --min-pair-dist-diff 1 --files syn6.json > "$log_dir/syn6.log" 2>&1
    echo "syn6.log 求解完成"
fi