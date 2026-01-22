#!/bin/bash
#
# 批量测试脚本：对 benchmark/test_input 目录下的所有 JSON 文件运行 batch_test_solutions_gurobi.py
#
# 用法:
#     ./run_all_tests.sh

log_dir="output/log"

# 创建日志目录（如果不存在）
mkdir -p "$log_dir"

# 清理并创建日志文件，然后运行测试
if [ -f "$log_dir/acend910.log" ]; then rm -f "$log_dir/acend910.log"; fi
touch "$log_dir/acend910.log"
python3 batch_test_solutions_gurobi.py --num-solutions 30 --grid-size 0.1 --min-pair-dist-diff 0.1 --files acend910.json > "$log_dir/acend910.log" 2>&1

if [ -f "$log_dir/cpu-dram.log" ]; then rm -f "$log_dir/cpu-dram.log"; fi
touch "$log_dir/cpu-dram.log"
python3 batch_test_solutions_gurobi.py --num-solutions 30 --grid-size 0.1 --min-pair-dist-diff 0.1 --files cpu-dram.json > "$log_dir/cpu-dram.log" 2>&1

if [ -f "$log_dir/hp11_m.log" ]; then rm -f "$log_dir/hp11_m.log"; fi
touch "$log_dir/hp11_m.log"
python3 batch_test_solutions_gurobi.py --num-solutions 30 --grid-size 0.1 --min-pair-dist-diff 0.1 --files hp11_m.json > "$log_dir/hp11_m.log" 2>&1

if [ -f "$log_dir/hp6_m.log" ]; then rm -f "$log_dir/hp6_m.log"; fi
touch "$log_dir/hp6_m.log"
python3 batch_test_solutions_gurobi.py --num-solutions 30 --grid-size 0.1 --min-pair-dist-diff 0.1 --files hp6_m.json > "$log_dir/hp6_m.log" 2>&1

if [ -f "$log_dir/hp8_m.log" ]; then rm -f "$log_dir/hp8_m.log"; fi
touch "$log_dir/hp8_m.log"
python3 batch_test_solutions_gurobi.py --num-solutions 30 --grid-size 0.1 --min-pair-dist-diff 0.1 --files hp8_m.json > "$log_dir/hp8_m.log" 2>&1

if [ -f "$log_dir/multigpu.log" ]; then rm -f "$log_dir/multigpu.log"; fi
touch "$log_dir/multigpu.log"
python3 batch_test_solutions_gurobi.py --num-solutions 30 --grid-size 0.1 --min-pair-dist-diff 0.1 --files multigpu.json > "$log_dir/multigpu.log" 2>&1

if [ -f "$log_dir/syn1.log" ]; then rm -f "$log_dir/syn1.log"; fi
touch "$log_dir/syn1.log"
python3 batch_test_solutions_gurobi.py --num-solutions 30 --grid-size 0.1 --min-pair-dist-diff 0.1 --files syn1.json > "$log_dir/syn1.log" 2>&1

if [ -f "$log_dir/syn2.log" ]; then rm -f "$log_dir/syn2.log"; fi
touch "$log_dir/syn2.log"
python3 batch_test_solutions_gurobi.py --num-solutions 30 --grid-size 0.1 --min-pair-dist-diff 0.1 --files syn2.json > "$log_dir/syn2.log" 2>&1

if [ -f "$log_dir/syn3.log" ]; then rm -f "$log_dir/syn3.log"; fi
touch "$log_dir/syn3.log"
python3 batch_test_solutions_gurobi.py --num-solutions 30 --grid-size 0.1 --min-pair-dist-diff 0.1 --files syn3.json > "$log_dir/syn3.log" 2>&1

if [ -f "$log_dir/syn4.log" ]; then rm -f "$log_dir/syn4.log"; fi
touch "$log_dir/syn4.log"
python3 batch_test_solutions_gurobi.py --num-solutions 30 --grid-size 0.1 --min-pair-dist-diff 0.1 --files syn4.json > "$log_dir/syn4.log" 2>&1

if [ -f "$log_dir/syn5.log" ]; then rm -f "$log_dir/syn5.log"; fi
touch "$log_dir/syn5.log"
python3 batch_test_solutions_gurobi.py --num-solutions 30 --grid-size 0.1 --min-pair-dist-diff 0.1 --files syn5.json > "$log_dir/syn5.log" 2>&1

if [ -f "$log_dir/syn6.log" ]; then rm -f "$log_dir/syn6.log"; fi
touch "$log_dir/syn6.log"
python3 batch_test_solutions_gurobi.py --num-solutions 30 --grid-size 0.1 --min-pair-dist-diff 0.1 --files syn6.json > "$log_dir/syn6.log" 2>&1

if [ -f "$log_dir/xerox6_m.log" ]; then rm -f "$log_dir/xerox6_m.log"; fi
touch "$log_dir/xerox6_m.log"
python3 batch_test_solutions_gurobi.py --num-solutions 30 --grid-size 0.1 --min-pair-dist-diff 0.1 --files xerox6_m.json > "$log_dir/xerox6_m.log" 2>&1

if [ -f "$log_dir/xerox7_m.log" ]; then rm -f "$log_dir/xerox7_m.log"; fi
touch "$log_dir/xerox7_m.log"
python3 batch_test_solutions_gurobi.py --num-solutions 30 --grid-size 0.1 --min-pair-dist-diff 0.1 --files xerox7_m.json > "$log_dir/xerox7_m.log" 2>&1

if [ -f "$log_dir/xerox8_m.log" ]; then rm -f "$log_dir/xerox8_m.log"; fi
touch "$log_dir/xerox8_m.log"
python3 batch_test_solutions_gurobi.py --num-solutions 30 --grid-size 0.1 --min-pair-dist-diff 0.1 --files xerox8_m.json > "$log_dir/xerox8_m.log" 2>&1
