
import sys
import os
# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import json
import random
from typing import Dict, List, Tuple, Optional, Set
from TCG import TCG, generate_layout_from_tcg, print_layout_info,get_layout_area   
from Generate_initial_TCG import generate_initial_TCG
import copy
import networkx as nx
from unit import load_problem_from_json,save_layout_to_json,visualize_layout_with_bridges,best_utilization
from chiplet_model import Chiplet, LayoutProblem, MIN_OVERLAP, get_adjacency_info, EPSILON
from Bridge_Overlap_Adjustment import (
    SiliconBridge, generate_silicon_bridges, 
    SiliconBridge_is_legal, SILICONBRIDGE_LENGTH
)
from legalize_tcg import legalize_tcg

problem=load_problem_from_json("../test_input/3core.json")
tcg=generate_initial_TCG(problem, seed=None)

# 打印TCG信息
print("初始TCG信息:")
print(f"  {tcg}")
print(f"  芯片数量: {len(tcg.chip_ids)}")
print(f"  芯片列表: {tcg.chip_ids}")
print(f"  水平约束边 (Ch): {list(tcg.Ch.edges())}")
print(f"  垂直约束边 (Cv): {list(tcg.Cv.edges())}")

layout = generate_layout_from_tcg(tcg, problem)

# 合法化TCG
print("\n合法化TCG...")
success, tcg_legalized, layout2 = legalize_tcg(tcg, problem, verbose=True)

if success:
    print("\n✓ TCG合法化成功")
else:
    print("\n⚠ TCG合法化未完全成功，使用最佳结果")

visualize_layout_with_bridges(layout2, problem, "../output/1.png")
