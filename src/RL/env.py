"""
RL环境模块 - 芯片布局强化学习环境

提供标准的 Gym 风格接口，用于 PPO/DQN 等 RL 算法训练
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from copy import deepcopy

import numpy as np
import json

# 导入 baseline 中的数据结构和验证函数
baseline_path = Path(__file__).parent.parent.parent / "baseline" / "ICCAD23" / "src"
sys.path.insert(0, str(baseline_path))

from chiplet_model import (
    Chiplet, 
    LayoutProblem, 
    has_overlap, 
    get_adjacency_info, 
    load_problem_from_json
)
from unit import (
    calculate_wirelength,
    calculate_manhattan_wirelength,
    calculate_layout_utilization,
    visualize_layout_with_bridges
)


@dataclass
class ChipletState:
    """布局状态快照"""
    layout: Dict[str, Chiplet]  # 当前布局
    placed: List[str]  # 已放置的芯片ID列表
    remaining: List[str]  # 未放置的芯片ID列表
    current_step: int = 0  # 当前步骤（固定顺序时）
    
    def copy(self) -> ChipletState:
        """深拷贝状态"""
        return ChipletState(
            layout={k: deepcopy(v) for k, v in self.layout.items()},
            placed=self.placed.copy(),
            remaining=self.remaining.copy(),
            current_step=self.current_step
        )


class ChipletPlacementEnv:
    """
    芯片布局强化学习环境 - 连接约束驱动版本
    
    核心思想：
    1. 按照固定顺序（例如DFS）放置芯片
    2. 第一个芯片可以放在任何位置
    3. 后续芯片必须与已放置的邻接芯片在物理上相邻（边接触 + 重叠 >= min_overlap）
    4. 所有非邻接芯片间必须不重叠
    5. 只有满足约束的位置才是有效动作
    
    动作空间：(网格x, 网格y, 旋转)，但只有有效位置可以选择
    状态空间：当前布局编码 + 当前芯片的有效位置掩码
    """
    
    def __init__(
        self,
        problem: LayoutProblem,
        placement_order: Optional[List[str]] = None,  # 固定的放置顺序
        grid_resolution: int = 50,
        max_width: Optional[float] = None,
        max_height: Optional[float] = None,
        min_overlap: float = 1.0,
        # 奖励权重
        overlap_penalty: float = 1000.0,
        adjacency_penalty: float = 500.0,  # 违反相邻约束的惩罚
        adjacency_reward: float = 100.0,  # 满足相邻约束的奖励
        area_reward_scale: float = 10.0,
        placement_reward: float = 50.0,
    ):
        """
        初始化环境
        
        Args:
            problem: LayoutProblem 对象
            placement_order: 放置顺序（DFS生成）
            grid_resolution: 网格分辨率
            max_width/max_height: 边界框尺寸
            min_overlap: 相邻最小重叠长度
            overlap_penalty: 重叠惩罚
            adjacency_penalty: 违反相邻约束的惩罚
            adjacency_reward: 满足相邻约束的奖励
            area_reward_scale: 面积利用率奖励
            placement_reward: 成功放置奖励
        """
        self.problem = problem
        self.grid_resolution = grid_resolution
        self.min_overlap = min_overlap
        
        # 奖励权重
        self.overlap_penalty = overlap_penalty
        self.adjacency_penalty = adjacency_penalty
        self.adjacency_reward = adjacency_reward
        self.area_reward_scale = area_reward_scale
        self.placement_reward = placement_reward
        
        # 初始化芯片
        self.chiplets: Dict[str, Chiplet] = {
            chip_id: deepcopy(chiplet) 
            for chip_id, chiplet in problem.chiplets.items()
        }
        
        # 放置顺序
        if placement_order is None:
            self.placement_order = creat_order_dfs(problem)
        else:
            for chip_id in placement_order:
                if chip_id not in self.chiplets:
                    raise ValueError(f"芯片 {chip_id} 不存在")
            self.placement_order = placement_order
        
        self.num_chiplets = len(self.placement_order)
        
        # 计算边界框
        if max_width is None or max_height is None:
            widths = [c.width for c in self.chiplets.values()]
            heights = [c.height for c in self.chiplets.values()]
            total_area = sum(w * h for w, h in zip(widths, heights))
            
            if max_width is None:
                max_width = int(np.sqrt(total_area * 2))
            if max_height is None:
                max_height = int(np.sqrt(total_area * 2))
        
        self.max_width = float(max_width)
        self.max_height = float(max_height)
        
        # 动作空间
        self.grid_size_x = grid_resolution
        self.grid_size_y = grid_resolution
        self.num_rotations = 2
        
        self.action_dim = (
            self.grid_size_x * 
            self.grid_size_y * 
            self.num_rotations
        )
        
        # 观察维度（需要与get_observation()一致）
        # 观察包含：已放置数(1) + 所有芯片的bbox(4*n) + 当前利用率(1) + 当前线长(1) + 有效动作数(1) + 当前步骤(1)
        # 为简化起见，使用固定维度
        self.observation_dim = 10
        
        # 网格步长
        self.step_x = self.max_width / self.grid_resolution
        self.step_y = self.max_height / self.grid_resolution
        
        # 初始化状态
        self.state: ChipletState = self._init_state()
    
    def _init_state(self) -> ChipletState:
        """初始化状态"""
        return ChipletState(
            layout={},
            placed=[],
            remaining=self.placement_order.copy(),
            current_step=0
        )
    
    def reset(self) -> np.ndarray:
        """重置环境"""
        self.state = self._init_state()
        return self.get_observation()
    
    def _get_current_chip_id(self) -> Optional[str]:
        """获取当前要放置的芯片ID"""
        if self.state.current_step < len(self.placement_order):
            return self.placement_order[self.state.current_step]
        return None
    
    def _get_adjacent_neighbors(self, chip_id: str) -> List[str]:
        """
        获取芯片的邻接邻域（在连接图中相邻且已放置的芯片）
        
        Args:
            chip_id: 芯片ID
            
        Returns:
            已放置的邻接芯片ID列表
        """
        neighbors = self.problem.get_neighbors(chip_id)
        return [n for n in neighbors if n in self.state.layout]
    
    def _is_valid_placement(
        self,
        new_chiplet: Chiplet,
        chip_id: str
    ) -> Tuple[bool, str]:
        """
        检查芯片放置是否合法
        
        Rules:
        1. 不能超出边界
        2. 不能与任何已放置芯片重叠（包括邻接和非邻接芯片）
        3. 如果有邻接邻域，必须与至少一个邻接邻域相邻（边接触+充分重叠）
        
        Args:
            new_chiplet: 新芯片对象
            chip_id: 芯片ID
            
        Returns:
            (is_valid, reason)
        """
        # 1. 检查边界
        if new_chiplet.x < 0 or new_chiplet.y < 0 or \
           new_chiplet.x + new_chiplet.width > self.max_width or \
           new_chiplet.y + new_chiplet.height > self.max_height:
            return False, "out_of_bounds"
        
        # 2. 检查与已放置芯片的关系
        adjacent_neighbors = self._get_adjacent_neighbors(chip_id)
        is_adjacent_to_any_neighbor = False
        
        for placed_chip_id, placed_chip in self.state.layout.items():
            # 首先检查是否有重叠（所有芯片都不能重叠，无论是否邻接）
            if has_overlap(new_chiplet, placed_chip):
                if placed_chip_id in adjacent_neighbors:
                    return False, f"overlap_with_neighbor_{placed_chip_id}"
                else:
                    return False, f"overlap_with_non_neighbor_{placed_chip_id}"
            
            # 对于邻接芯片，检查是否满足相邻约束
            if placed_chip_id in adjacent_neighbors:
                is_adjacent, overlap_len, direction = get_adjacency_info(new_chiplet, placed_chip)
                if is_adjacent and overlap_len >= self.min_overlap:
                    is_adjacent_to_any_neighbor = True
        
        # 3. 如果有邻接邻域，必须与至少一个相邻
        if adjacent_neighbors and not is_adjacent_to_any_neighbor:
            return False, "not_adjacent_to_any_neighbor"
        
        return True, "ok"
    
    def _get_valid_positions(self, chip_id: str) -> List[Tuple[int, int, int]]:
        """
        获取芯片的有效放置位置列表
        
        返回所有满足约束的 (x_idx, y_idx, rotation) 组合
        
        Args:
            chip_id: 芯片ID
            
        Returns:
            有效位置列表 [(x_idx, y_idx, rotation), ...]
        """
        valid_positions = []
        chiplet_template = self.chiplets[chip_id]
        adjacent_neighbors = self._get_adjacent_neighbors(chip_id)
        
        # 如果是第一个芯片，所有位置都有效
        if not self.state.layout:
            for x_idx in range(self.grid_size_x):
                for y_idx in range(self.grid_size_y):
                    for rotation in range(self.num_rotations):
                        valid_positions.append((x_idx, y_idx, rotation))
            return valid_positions
        
        # 对于后续芯片，只考虑邻接邻域附近的位置
        # 优化：只检查与邻接邻域接近的网格位置
        for x_idx in range(self.grid_size_x):
            for y_idx in range(self.grid_size_y):
                for rotation in range(self.num_rotations):
                    # 创建测试芯片
                    test_chiplet = deepcopy(chiplet_template)
                    if rotation == 1:
                        test_chiplet.width, test_chiplet.height = test_chiplet.height, test_chiplet.width
                    
                    test_chiplet.x = x_idx * self.step_x
                    test_chiplet.y = y_idx * self.step_y
                    
                    # 检查是否合法
                    is_valid, reason = self._is_valid_placement(test_chiplet, chip_id)
                    if is_valid:
                        valid_positions.append((x_idx, y_idx, rotation))
        
        return valid_positions
    
    def get_observation(self) -> np.ndarray:
        """
        获取观察向量
        
        包含：
        1. 已放置/剩余芯片数
        2. 边界框信息
        3. 面积利用率
        4. 线长
        5. 相邻约束满足情况
        6. 当前可用位置数（作为难度指标）
        """
        features = []
        
        # 1. 基本统计
        features.append(len(self.state.placed))
        features.append(len(self.state.remaining))
        
        # 2. 边界框
        if self.state.layout:
            chiplets = list(self.state.layout.values())
            x_coords = [c.x for c in chiplets] + [c.x + c.width for c in chiplets]
            y_coords = [c.y for c in chiplets] + [c.y + c.height for c in chiplets]
            
            bbox_w = max(x_coords) - min(x_coords)
            bbox_h = max(y_coords) - min(y_coords)
            features.extend([bbox_w, bbox_h, bbox_w * bbox_h])
        else:
            features.extend([0.0, 0.0, 0.0])
        
        # 3. 面积利用率
        if self.state.layout:
            util = sum(c.width * c.height for c in self.state.layout.values())
            features.append(util / (self.max_width * self.max_height))
        else:
            features.append(0.0)
        
        # 4. 线长
        if self.state.layout and len(self.state.layout) > 1:
            wirelength = calculate_manhattan_wirelength(self.state.layout, self.problem)
            features.append(wirelength)
        else:
            features.append(0.0)
        
        # 5. 相邻约束
        satisfied_count = 0
        for chip_id1, chip_id2 in self.problem.connection_graph.edges():
            if chip_id1 in self.state.layout and chip_id2 in self.state.layout:
                is_adj, overlap, _ = get_adjacency_info(
                    self.state.layout[chip_id1],
                    self.state.layout[chip_id2]
                )
                if is_adj and overlap >= self.min_overlap:
                    satisfied_count += 1
        
        features.append(satisfied_count)
        features.append(len(self.problem.connection_graph.edges()))
        
        # 6. 当前可用位置数
        current_chip = self._get_current_chip_id()
        if current_chip:
            valid_pos_count = len(self._get_valid_positions(current_chip))
            features.append(valid_pos_count)
        else:
            features.append(0.0)
        
        return np.array(features, dtype=np.float32)
    
    def get_valid_actions(self) -> List[int]:
        """
        获取当前有效的动作列表（编码的有效位置）
        
        Returns:
            有效动作索引列表
        """
        if self.state.current_step >= self.num_chiplets:
            return []
        
        current_chip = self._get_current_chip_id()
        if current_chip is None:
            return []
        
        valid_positions = self._get_valid_positions(current_chip)
        valid_actions = [self._encode_action(x, y, r) for x, y, r in valid_positions]
        
        return valid_actions
    
    def _encode_action(self, x_idx: int, y_idx: int, rotation: int) -> int:
        """编码动作"""
        return (
            x_idx * (self.grid_size_y * self.num_rotations) +
            y_idx * self.num_rotations +
            rotation
        )
    
    def _decode_action(self, action: int) -> Tuple[int, int, int]:
        """解码动作"""
        x_idx = action // (self.grid_size_y * self.num_rotations)
        remainder = action % (self.grid_size_y * self.num_rotations)
        y_idx = remainder // self.num_rotations
        rotation = remainder % self.num_rotations
        return x_idx, y_idx, rotation
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        执行动作
        
        Args:
            action: 动作索引
            
        Returns:
            (observation, reward, done, info)
        """
        if self.state.current_step >= self.num_chiplets:
            return self.get_observation(), -self.overlap_penalty, True, {
                "step": self.state.current_step,
                "total_steps": self.num_chiplets,
                "error": "done"
            }
        
        current_chip_id = self._get_current_chip_id()
        if current_chip_id is None:
            return self.get_observation(), -self.overlap_penalty, False, {
                "step": self.state.current_step,
                "total_steps": self.num_chiplets,
                "error": "no_chip"
            }
        
        # 解码动作
        x_idx, y_idx, rotation = self._decode_action(action)
        chiplet_template = self.chiplets[current_chip_id]
        
        # 创建新芯片
        new_chiplet = deepcopy(chiplet_template)
        if rotation == 1:
            new_chiplet.width, new_chiplet.height = new_chiplet.height, new_chiplet.width
        
        new_chiplet.x = x_idx * self.step_x
        new_chiplet.y = y_idx * self.step_y
        
        # 检查合法性
        is_valid, reason = self._is_valid_placement(new_chiplet, current_chip_id)
        if not is_valid:
            return self.get_observation(), -self.overlap_penalty, False, {
                "step": self.state.current_step,
                "total_steps": self.num_chiplets,
                "chip_id": current_chip_id,
                "error": reason
            }
        
        # 放置芯片
        self.state.layout[current_chip_id] = new_chiplet
        self.state.placed.append(current_chip_id)
        self.state.remaining.remove(current_chip_id)
        self.state.current_step += 1
        
        # 计算奖励
        reward = self.placement_reward
        
        # 检查相邻约束
        neighbors = self.problem.get_neighbors(current_chip_id)
        for neighbor_id in neighbors:
            if neighbor_id in self.state.layout:
                neighbor_chip = self.state.layout[neighbor_id]
                is_adj, overlap_len, _ = get_adjacency_info(new_chiplet, neighbor_chip)
                if is_adj and overlap_len >= self.min_overlap:
                    reward += self.adjacency_reward
                else:
                    reward -= self.adjacency_penalty
        
        # 检查完成
        done = self.state.current_step >= self.num_chiplets
        
        if done:
            utilization, _, _, _, _ = calculate_layout_utilization(self.state.layout)
            reward += utilization * self.area_reward_scale
        
        info = {
            "chip_id": current_chip_id,
            "step": self.state.current_step,
            "total_steps": self.num_chiplets,
            "valid_positions": len(self._get_valid_positions(current_chip_id) if not done else []),
            "rotation": rotation
        }
        
        return self.get_observation(), reward, done, info
    
    
    def render(self, mode: str = 'human') -> Optional[str]:
        """渲染布局"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
        except ImportError:
            print("matplotlib 未安装")
            return None
        
        if not self.state.layout:
            print("布局为空")
            return None
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        for chip_id, chiplet in self.state.layout.items():
            rect = patches.Rectangle(
                (chiplet.x, chiplet.y),
                chiplet.width,
                chiplet.height,
                linewidth=2,
                edgecolor='black',
                facecolor='lightblue',
                alpha=0.7
            )
            ax.add_patch(rect)
            ax.text(
                chiplet.x + chiplet.width / 2,
                chiplet.y + chiplet.height / 2,
                chip_id,
                ha='center',
                va='center',
                fontsize=10
            )
        
        ax.set_xlim(0, self.max_width)
        ax.set_ylim(0, self.max_height)
        ax.set_aspect('equal')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Chiplet Layout ({len(self.state.placed)}/{self.num_chiplets})')
        ax.grid(True, alpha=0.3)
        
        if mode == 'human':
            plt.show()
        
        return None
    
    def get_layout_dict(self) -> Dict[str, Tuple[float, float]]:
        """获取布局坐标字典"""
        return {chip_id: (c.x, c.y) for chip_id, c in self.state.layout.items()}
    
    def save_layout_json(self, filepath: str) -> None:
        """保存布局为JSON"""
        data = {
            "chiplets": [
                {
                    "id": chip_id,
                    "x": float(chiplet.x),
                    "y": float(chiplet.y),
                    "width": float(chiplet.width),
                    "height": float(chiplet.height)
                }
                for chip_id, chiplet in self.state.layout.items()
            ],
            "connections": list(self.problem.connection_graph.edges())
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


# 便捷函数
def create_env_from_json(json_path: str, **kwargs) -> ChipletPlacementEnv:
    """
    从 JSON 文件创建环境
    
    Args:
        json_path: JSON 文件路径
        **kwargs: 其他环境参数
        
    Returns:
        ChipletPlacementEnv 实例
    """
    problem = load_problem_from_json(json_path)
    return ChipletPlacementEnv(problem, **kwargs)


def creat_order_dfs(problem: LayoutProblem) -> List[str]:
    """
    基于深度优先搜索生成芯片放置顺序
    
    Args:
        problem: LayoutProblem 对象
        
    Returns:
        芯片ID列表，表示放置顺序
    """
    from collections import deque
    
    visited = set()
    order = []
    
    def dfs(chip_id: str):
        visited.add(chip_id)
        order.append(chip_id)
        
        for neighbor in problem.connection_graph.neighbors(chip_id):
            if neighbor not in visited:
                dfs(neighbor)
    
    # 从第一个芯片开始DFS
    start_chip = list(problem.chiplets.keys())[0]
    dfs(start_chip)
    
    # 添加未访问的芯片（孤立芯片）
    for chip_id in problem.chiplets.keys():
        if chip_id not in visited:
            order.append(chip_id)
    
    return order



def creat_order_bfs(problem: LayoutProblem) -> List[str]:
    """
    基于广度优先搜索生成芯片放置顺序
    
    Args:
        problem: LayoutProblem 对象
        
    Returns:
        芯片ID列表，表示放置顺序
    """
    from collections import deque
    
    visited = set()
    order = []
    queue = deque()
    
    # 从第一个芯片开始BFS
    start_chip = list(problem.chiplets.keys())[0]
    queue.append(start_chip)
    visited.add(start_chip)
    
    while queue:
        chip_id = queue.popleft()
        order.append(chip_id)
        
        for neighbor in problem.connection_graph.neighbors(chip_id):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    # 添加未访问的芯片（孤立芯片）
    for chip_id in problem.chiplets.keys():
        if chip_id not in visited:
            order.append(chip_id)
    
    return order



if __name__ == "__main__":
    # 测试环境
    print("=" * 70)
    print("测试 ChipletPlacementEnv - 连接约束驱动版本（细粒度网格）")
    print("=" * 70)
    
    # 创建简单测试用例
    # problem = LayoutProblem()
    # problem.add_chiplet(Chiplet("A", 10, 20))
    # problem.add_chiplet(Chiplet("B", 15, 25))
    # problem.add_chiplet(Chiplet("C", 12, 18))
    # problem.add_chiplet(Chiplet("D", 14, 22))
    
    # problem.add_connection("A", "B")
    # problem.add_connection("B", "C")
    # problem.add_connection("C", "D")
    
    # 从JSON加载问题（支持'chiplets'格式）
    import json
    json_file = "../../baseline/ICCAD23/test_input/12core.json"
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    problem = LayoutProblem()
    
    # 支持'chiplets'或'dies'字段
    chiplets_data = data.get('chiplets', data.get('dies', []))
    for chiplet_data in chiplets_data:
        name = chiplet_data.get('name') or chiplet_data.get('id')
        width = chiplet_data.get('width', 10)
        height = chiplet_data.get('height', 10)
        problem.add_chiplet(Chiplet(name, width, height))
    
    # 添加连接
    connections = data.get('connections', [])
    for conn in connections:
        if len(conn) >= 2:
            problem.add_connection(conn[0], conn[1])
    
    placement_order = creat_order_dfs(problem)
    
    # 创建环境 - 使用很细的网格（50×50）和较小的min_overlap
    env = ChipletPlacementEnv(
        problem,
        placement_order=placement_order,
        grid_resolution=50,  # 增加到50×50
        max_width=100,
        max_height=100,
        min_overlap=0.5  # 减小重叠要求
    )
    
    print(f"\n环境参数:")
    print(f"  芯片数量: {env.num_chiplets}")
    print(f"  放置顺序: {env.placement_order}")
    print(f"  动作空间维度: {env.action_dim}")
    print(f"  网格步长: ({env.step_x:.2f}, {env.step_y:.2f})")
    print(f"  边界框: {env.max_width} × {env.max_height}")
    print(f"  连接关系: {list(problem.connection_graph.edges())}")
    
    # 重置并开始
    obs = env.reset()
    print(f"\n初始状态观察维度: {obs.shape}")
    
    # 执行一个完整episode
    print(f"\n开始放置:")
    print("-" * 70)
    
    total_reward = 0.0
    step_count = 0
    
    while True:
        step_count += 1
        current_chip = env._get_current_chip_id()
        valid_actions = env.get_valid_actions()
        
        print(f"\n  步骤 {step_count}:")
        print(f"    放置芯片: {current_chip}")
        print(f"    有效位置数: {len(valid_actions)}")
        
        if not valid_actions:
            print(f"    ✗ ERROR: 无有效动作可选!")
            break
        
        # 随机选择一个有效动作
        import random
        action = random.choice(valid_actions)

        #todo
        #action=RL_agent.select_action(obs,valid_actions)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        
        print(f"    本步奖励: {reward:.2f}")
        print(f"    累计奖励: {total_reward:.2f}")
        print(f"    进度: {info['step']}/{info['total_steps']}")
        
        if done:
            print(f"\n✓ 完成！总奖励: {total_reward:.2f}")
            
            # 显示最终布局
            print(f"\n最终布局:")
            layout = env.get_layout_dict()
            for chip_id, (x, y) in layout.items():
                chip = env.state.layout[chip_id]
                print(f"  {chip_id}: ({x:.2f}, {y:.2f}) size=({chip.width:.1f}×{chip.height:.1f})")
            break

    print("\n可视化布局和硅桥")
    print("-" * 70)
    # 传递Chiplet对象而非坐标字典
    visualize_layout_with_bridges(
        env.state.layout,  # 使用Chiplet对象字典
        problem, 
        output_file='output/layout_with_bridges.png',
        show_bridges=True,
        show_coordinates=True
    )
    
    print("\n✓ 所有测试完成")
