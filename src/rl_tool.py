"""
可复用的强化学习工具函数和环境类
用于chiplet布局优化问题
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tool import ChipletNode


@dataclass
class ChipletPlacement:
    """单个chiplet的放置信息"""
    chiplet_idx: int  # chiplet在nodes列表中的索引
    grid_x: int  # 右下角在grid中的x坐标
    grid_y: int  # 右下角在grid中的y坐标
    rotated: bool  # 是否旋转90度


class PlacementState:
    """布局状态：已放置的chiplet列表和grid mask"""
    def __init__(self, placed: List[ChipletPlacement], remaining: List[int], grid_mask: np.ndarray):
        self.placed = placed  # 已放置的chiplet
        self.remaining = remaining  # 待放置的chiplet索引
        self.grid_mask = grid_mask  # grid占用mask，1表示已占用，0表示空闲


class PlacementEnv:
    """
    强化学习环境：
    - 状态：已放置的chiplet位置和旋转状态，以及待放置chiplet的信息
    - 动作：选择下一个chiplet，选择其位置(x, y)和旋转状态
    - 奖励：基于重叠惩罚、相邻约束惩罚、面积奖励
    """
    
    def __init__(
        self,
        nodes: List[ChipletNode],
        edges: List[Tuple[str, str]],
        min_shared_length: float = 0.5,
        grid_resolution: int = 25,
        overlap_penalty: float = 1000.0,
        adjacency_penalty: float = 500.0,
        area_reward_scale: float = 1.0,
        gap_penalty: float = 100.0,
        adjacency_reward: float = 50.0,
    ):
        self.nodes = nodes
        self.edges = edges
        self.min_shared_length = min_shared_length
        self.grid_resolution = grid_resolution
        self.overlap_penalty = overlap_penalty
        self.adjacency_penalty = adjacency_penalty
        self.area_reward_scale = area_reward_scale
        self.gap_penalty = gap_penalty
        self.adjacency_reward = adjacency_reward
        
        # 构建连接关系映射
        self.name_to_idx = {node.name: i for i, node in enumerate(nodes)}
        self.connected_pairs = set()
        for s_name, d_name in edges:
            if s_name in self.name_to_idx and d_name in self.name_to_idx:
                i = self.name_to_idx[s_name]
                j = self.name_to_idx[d_name]
                if i != j:
                    self.connected_pairs.add((min(i, j), max(i, j)))
        
        # 估算边界框大小（用于离散化位置）
        total_area = sum(
            node.dimensions.get("x", 0.0) * node.dimensions.get("y", 0.0)
            for node in nodes
        )
        self.estimated_bound = math.sqrt(total_area) * 2.0
        
        # 计算chiplet在grid中的尺寸（需要将实际尺寸转换为grid单位）
        self.grid_size_per_unit = self.estimated_bound / self.grid_resolution
        
        self.reset()
    
    def _get_chiplet_grid_size(self, chiplet_idx: int, rotated: bool) -> Tuple[int, int]:
        """获取chiplet在grid中的尺寸（grid单位）"""
        node = self.nodes[chiplet_idx]
        w_orig = node.dimensions.get("x", 0.0)
        h_orig = node.dimensions.get("y", 0.0)
        w = h_orig if rotated else w_orig
        h = w_orig if rotated else h_orig
        
        grid_w = max(1, int(math.ceil(w / self.grid_size_per_unit)))
        grid_h = max(1, int(math.ceil(h / self.grid_size_per_unit)))
        
        return grid_w, grid_h
    
    def _get_chiplet_grid_region(self, grid_x: int, grid_y: int, grid_w: int, grid_h: int) -> Tuple[int, int, int, int]:
        """获取chiplet占用的grid区域"""
        min_x = grid_x - grid_w + 1
        max_x = grid_x + 1
        min_y = grid_y - grid_h + 1
        max_y = grid_y + 1
        return min_x, max_x, min_y, max_y
    
    def _check_grid_overlap(self, grid_x: int, grid_y: int, grid_w: int, grid_h: int) -> bool:
        """检查chiplet在grid中的位置是否与已占用的grid重叠"""
        min_x, max_x, min_y, max_y = self._get_chiplet_grid_region(grid_x, grid_y, grid_w, grid_h)
        
        if min_x < 0 or min_y < 0 or max_x > self.grid_resolution or max_y > self.grid_resolution:
            return True
        
        region_mask = self.state.grid_mask[min_y:max_y, min_x:max_x]
        return np.any(region_mask > 0.5)
    
    def _update_grid_mask(self, grid_x: int, grid_y: int, grid_w: int, grid_h: int):
        """更新grid mask，标记chiplet占用的区域"""
        min_x, max_x, min_y, max_y = self._get_chiplet_grid_region(grid_x, grid_y, grid_w, grid_h)
        
        if min_x < 0 or min_y < 0 or max_x > self.grid_resolution or max_y > self.grid_resolution:
            raise ValueError(
                f"Cannot update mask: chiplet region [{min_y}:{max_y}, {min_x}:{max_x}] "
                f"exceeds grid bounds [0:{self.grid_resolution}, 0:{self.grid_resolution}]"
            )
        
        self.state.grid_mask[min_y:max_y, min_x:max_x] = 1.0
    
    def reset(self) -> np.ndarray:
        """重置环境，返回初始状态向量"""
        grid_mask = np.zeros((self.grid_resolution, self.grid_resolution), dtype=np.float32)
        
        self.state = PlacementState(
            placed=[],
            remaining=list(range(len(self.nodes))),
            grid_mask=grid_mask
        )
        random.shuffle(self.state.remaining)
        return self._state_to_vector()
    
    def step(self, action: Tuple[int, int, int, int]) -> Tuple[np.ndarray, float, bool, dict]:
        """执行动作"""
        chiplet_idx_in_remaining, grid_x, grid_y, rotation = action
        
        if chiplet_idx_in_remaining >= len(self.state.remaining):
            return self._state_to_vector(), -self.overlap_penalty, False, {"invalid": True}
        
        if grid_x < 0 or grid_x >= self.grid_resolution or grid_y < 0 or grid_y >= self.grid_resolution:
            return self._state_to_vector(), -self.overlap_penalty, False, {"invalid": True}
        
        chiplet_idx = self.state.remaining[chiplet_idx_in_remaining]
        rotated = (rotation == 1)
        grid_w, grid_h = self._get_chiplet_grid_size(chiplet_idx, rotated)
        
        overlap = self._check_grid_overlap(grid_x, grid_y, grid_w, grid_h)
        if overlap:
            reward = -self.overlap_penalty
            done = False
            info = {"overlap": True}
            return self._state_to_vector(), reward, done, info
        
        placement = ChipletPlacement(
            chiplet_idx=chiplet_idx,
            grid_x=grid_x,
            grid_y=grid_y,
            rotated=rotated
        )
        
        self._update_grid_mask(grid_x, grid_y, grid_w, grid_h)
        self.state.placed.append(placement)
        self.state.remaining.remove(chiplet_idx)
        
        reward = self._compute_reward()
        done = len(self.state.remaining) == 0
        
        info = {
            "overlap": False,
            "num_placed": len(self.state.placed),
            "num_remaining": len(self.state.remaining),
        }
        
        return self._state_to_vector(), reward, done, info
    
    def _grid_to_real_coords(self, grid_x: int, grid_y: int) -> Tuple[float, float]:
        """将grid坐标转换为实际坐标"""
        x = (grid_x / self.grid_resolution) * self.estimated_bound
        y = (grid_y / self.grid_resolution) * self.estimated_bound
        return x, y
    
    def _compute_reward(self) -> float:
        """计算奖励"""
        reward = 0.0
        
        # 1. 检查相邻约束
        adjacency_violations = 0
        for i, j in self.connected_pairs:
            if not self._check_adjacency(i, j):
                adjacency_violations += 1
        
        if adjacency_violations > 0:
            reward -= self.adjacency_penalty * adjacency_violations
        
        # 2. 计算外接框面积
        bbox_w, bbox_h = self._compute_bounding_box()
        area = bbox_w * bbox_h
        reward -= self.area_reward_scale * area
        
        # 3. 惩罚间隔
        if len(self.state.placed) > 1:
            total_gap = self._compute_total_gap()
            reward -= self.gap_penalty * total_gap
        
        # 4. 奖励紧邻
        if len(self.state.placed) > 1:
            adjacency_count = self._count_adjacent_pairs()
            reward += self.adjacency_reward * adjacency_count
        
        return reward
    
    def _compute_total_gap(self) -> float:
        """计算所有chiplet之间的总间隔"""
        if len(self.state.placed) < 2:
            return 0.0
        
        total_gap = 0.0
        
        for i, p1 in enumerate(self.state.placed):
            grid_w1, grid_h1 = self._get_chiplet_grid_size(p1.chiplet_idx, p1.rotated)
            min_x1, max_x1, min_y1, max_y1 = self._get_chiplet_grid_region(
                p1.grid_x, p1.grid_y, grid_w1, grid_h1
            )
            
            for j, p2 in enumerate(self.state.placed):
                if i >= j:
                    continue
                
                grid_w2, grid_h2 = self._get_chiplet_grid_size(p2.chiplet_idx, p2.rotated)
                min_x2, max_x2, min_y2, max_y2 = self._get_chiplet_grid_region(
                    p2.grid_x, p2.grid_y, grid_w2, grid_h2
                )
                
                if max_x1 <= min_x2:
                    gap_x_grid = min_x2 - max_x1
                elif max_x2 <= min_x1:
                    gap_x_grid = min_x1 - max_x2
                else:
                    gap_x_grid = 0
                
                if max_y1 <= min_y2:
                    gap_y_grid = min_y2 - max_y1
                elif max_y2 <= min_y1:
                    gap_y_grid = min_y1 - max_y2
                else:
                    gap_y_grid = 0
                
                gap_x = gap_x_grid * self.grid_size_per_unit
                gap_y = gap_y_grid * self.grid_size_per_unit
                
                if gap_x > 0 and gap_y > 0:
                    gap = min(gap_x, gap_y)
                else:
                    gap = max(gap_x, gap_y)
                
                total_gap += gap
        
        return total_gap
    
    def _count_adjacent_pairs(self) -> int:
        """统计紧邻的chiplet对数量"""
        if len(self.state.placed) < 2:
            return 0
        
        count = 0
        
        for i, p1 in enumerate(self.state.placed):
            grid_w1, grid_h1 = self._get_chiplet_grid_size(p1.chiplet_idx, p1.rotated)
            min_x1, max_x1, min_y1, max_y1 = self._get_chiplet_grid_region(
                p1.grid_x, p1.grid_y, grid_w1, grid_h1
            )
            
            for j, p2 in enumerate(self.state.placed):
                if i >= j:
                    continue
                
                grid_w2, grid_h2 = self._get_chiplet_grid_size(p2.chiplet_idx, p2.rotated)
                min_x2, max_x2, min_y2, max_y2 = self._get_chiplet_grid_region(
                    p2.grid_x, p2.grid_y, grid_w2, grid_h2
                )
                
                is_adjacent = False
                
                if max_x1 == min_x2 or max_x2 == min_x1:
                    y_overlap_start = max(min_y1, min_y2)
                    y_overlap_end = min(max_y1, max_y2)
                    if y_overlap_end > y_overlap_start:
                        is_adjacent = True
                
                if not is_adjacent:
                    if max_y1 == min_y2 or max_y2 == min_y1:
                        x_overlap_start = max(min_x1, min_x2)
                        x_overlap_end = min(max_x1, max_x2)
                        if x_overlap_end > x_overlap_start:
                            is_adjacent = True
                
                if is_adjacent:
                    count += 1
        
        return count
    
    def _check_adjacency(self, i: int, j: int) -> bool:
        """检查chiplet i和j是否满足相邻约束"""
        placed_i = None
        placed_j = None
        for p in self.state.placed:
            if p.chiplet_idx == i:
                placed_i = p
            if p.chiplet_idx == j:
                placed_j = p
        
        if placed_i is None or placed_j is None:
            return True
        
        grid_w_i, grid_h_i = self._get_chiplet_grid_size(placed_i.chiplet_idx, placed_i.rotated)
        min_x_i, max_x_i, min_y_i, max_y_i = self._get_chiplet_grid_region(
            placed_i.grid_x, placed_i.grid_y, grid_w_i, grid_h_i
        )
        
        grid_w_j, grid_h_j = self._get_chiplet_grid_size(placed_j.chiplet_idx, placed_j.rotated)
        min_x_j, max_x_j, min_y_j, max_y_j = self._get_chiplet_grid_region(
            placed_j.grid_x, placed_j.grid_y, grid_w_j, grid_h_j
        )
        
        horizontal_adjacent = False
        
        if max_x_i == min_x_j:
            y_overlap_start = max(min_y_i, min_y_j)
            y_overlap_end = min(max_y_i, max_y_j)
            if y_overlap_end > y_overlap_start:
                shared_length_h_grid = y_overlap_end - y_overlap_start
                shared_length_h = shared_length_h_grid * self.grid_size_per_unit
                if shared_length_h >= self.min_shared_length:
                    horizontal_adjacent = True
        
        if not horizontal_adjacent and max_x_j == min_x_i:
            y_overlap_start = max(min_y_i, min_y_j)
            y_overlap_end = min(max_y_i, max_y_j)
            if y_overlap_end > y_overlap_start:
                shared_length_h_grid = y_overlap_end - y_overlap_start
                shared_length_h = shared_length_h_grid * self.grid_size_per_unit
                if shared_length_h >= self.min_shared_length:
                    horizontal_adjacent = True
        
        vertical_adjacent = False
        
        if max_y_i == min_y_j:
            x_overlap_start = max(min_x_i, min_x_j)
            x_overlap_end = min(max_x_i, max_x_j)
            if x_overlap_end > x_overlap_start:
                shared_length_v_grid = x_overlap_end - x_overlap_start
                shared_length_v = shared_length_v_grid * self.grid_size_per_unit
                if shared_length_v >= self.min_shared_length:
                    vertical_adjacent = True
        
        if not vertical_adjacent and max_y_j == min_y_i:
            x_overlap_start = max(min_x_i, min_x_j)
            x_overlap_end = min(max_x_i, max_x_j)
            if x_overlap_end > x_overlap_start:
                shared_length_v_grid = x_overlap_end - x_overlap_start
                shared_length_v = shared_length_v_grid * self.grid_size_per_unit
                if shared_length_v >= self.min_shared_length:
                    vertical_adjacent = True
        
        return horizontal_adjacent or vertical_adjacent
    
    def _compute_bounding_box(self) -> Tuple[float, float]:
        """计算当前布局的外接框尺寸"""
        if not self.state.placed:
            return 0.0, 0.0
        
        min_x_grid = self.grid_resolution
        min_y_grid = self.grid_resolution
        max_x_grid = 0
        max_y_grid = 0
        
        for p in self.state.placed:
            grid_w, grid_h = self._get_chiplet_grid_size(p.chiplet_idx, p.rotated)
            min_x, max_x, min_y, max_y = self._get_chiplet_grid_region(
                p.grid_x, p.grid_y, grid_w, grid_h
            )
            
            min_x_grid = min(min_x_grid, min_x)
            min_y_grid = min(min_y_grid, min_y)
            max_x_grid = max(max_x_grid, max_x)
            max_y_grid = max(max_y_grid, max_y)
        
        width_grid = max_x_grid - min_x_grid
        height_grid = max_y_grid - min_y_grid
        width_real = width_grid * self.grid_size_per_unit
        height_real = height_grid * self.grid_size_per_unit
        
        return width_real, height_real
    
    def _state_to_vector(self) -> np.ndarray:
        """将状态转换为向量表示"""
        n = len(self.nodes)
        
        grid_mask_size = self.grid_resolution * self.grid_resolution
        num_connections = n * (n - 1) // 2
        state_dim = grid_mask_size + n * 6 + num_connections
        state_vec = np.zeros(state_dim, dtype=np.float32)
        
        # 1. 填充grid mask
        state_vec[:grid_mask_size] = self.state.grid_mask.flatten()
        idx = grid_mask_size
        
        # 2. 填充chiplet信息
        placed_indices = {p.chiplet_idx for p in self.state.placed}
        
        for i in range(n):
            if i in placed_indices:
                placement = next(p for p in self.state.placed if p.chiplet_idx == i)
                grid_w, grid_h = self._get_chiplet_grid_size(placement.chiplet_idx, placement.rotated)
                
                state_vec[idx] = placement.grid_x / self.grid_resolution
                state_vec[idx + 1] = placement.grid_y / self.grid_resolution
                state_vec[idx + 2] = grid_w / self.grid_resolution
                state_vec[idx + 3] = grid_h / self.grid_resolution
                state_vec[idx + 4] = 1.0 if placement.rotated else 0.0
                state_vec[idx + 5] = 1.0
            else:
                node = self.nodes[i]
                w_orig = node.dimensions.get("x", 0.0)
                h_orig = node.dimensions.get("y", 0.0)
                grid_w = max(1, int(math.ceil(w_orig / self.grid_size_per_unit)))
                grid_h = max(1, int(math.ceil(h_orig / self.grid_size_per_unit)))
                state_vec[idx + 2] = grid_w / self.grid_resolution
                state_vec[idx + 3] = grid_h / self.grid_resolution
                state_vec[idx + 5] = 0.0
            
            idx += 6
        
        # 3. 填充连接关系
        conn_idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                if (i, j) in self.connected_pairs:
                    state_vec[idx + conn_idx] = 1.0
                conn_idx += 1
        
        return state_vec
    
    def get_valid_actions(self) -> List[int]:
        """获取当前状态下的有效动作索引列表"""
        if not self.state.remaining:
            return []
        
        valid_actions = []
        num_remaining = len(self.state.remaining)
        actions_per_chiplet = self.grid_resolution * self.grid_resolution * 2
        
        for chiplet_idx_in_remaining in range(num_remaining):
            chiplet_idx = self.state.remaining[chiplet_idx_in_remaining]
            
            for rotation in [0, 1]:
                rotated = (rotation == 1)
                grid_w, grid_h = self._get_chiplet_grid_size(chiplet_idx, rotated)
                
                for grid_y in range(self.grid_resolution):
                    for grid_x in range(self.grid_resolution):
                        if not self._check_grid_overlap(grid_x, grid_y, grid_w, grid_h):
                            base_idx = chiplet_idx_in_remaining * actions_per_chiplet
                            rotation_offset = rotation
                            grid_offset = (grid_y * self.grid_resolution + grid_x) * 2
                            action_idx = base_idx + grid_offset + rotation_offset
                            valid_actions.append(action_idx)
        
        return valid_actions
    
    def decode_action(self, action_idx: int) -> Tuple[int, int, int, int]:
        """将动作索引解码为具体的动作参数"""
        if not self.state.remaining:
            raise ValueError("No remaining chiplets")
        
        num_remaining = len(self.state.remaining)
        actions_per_chiplet = self.grid_resolution * self.grid_resolution * 2
        
        chiplet_idx_in_remaining = action_idx // actions_per_chiplet
        remaining_action = action_idx % actions_per_chiplet
        
        rotation = remaining_action % 2
        grid_xy = remaining_action // 2
        grid_y = grid_xy // self.grid_resolution
        grid_x = grid_xy % self.grid_resolution
        
        return (chiplet_idx_in_remaining, grid_x, grid_y, rotation)
    
    def get_current_layout(self) -> Dict[str, Tuple[float, float]]:
        """获取当前布局（用于可视化）"""
        layout = {}
        rotations = {}
        
        for p in self.state.placed:
            node = self.nodes[p.chiplet_idx]
            grid_w, grid_h = self._get_chiplet_grid_size(p.chiplet_idx, p.rotated)
            min_x_grid, max_x_grid, min_y_grid, max_y_grid = self._get_chiplet_grid_region(
                p.grid_x, p.grid_y, grid_w, grid_h
            )
            x_real, y_real = self._grid_to_real_coords(min_x_grid, min_y_grid)
            layout[node.name] = (x_real, y_real)
            rotations[node.name] = p.rotated
        
        return layout, rotations


class DQN(nn.Module):
    """DQN网络：输入状态向量，输出每个动作的Q值"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class ReplayBuffer:
    """经验回放缓冲区，用于DQN等离线学习算法"""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """存储经验"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        """随机采样一批经验"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
        )
    
    def __len__(self):
        return len(self.buffer)

