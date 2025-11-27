"""
DQN-based skeleton for chiplet placement optimization.

目标（奖励越大越好）：
1. 所有 chiplet 在 2D 平面上不重叠；
2. 覆盖所有淡蓝色方块的外接正方形面积尽量小；
3. 对于有连线的方块，方块外接圆圆心之间的距离之和尽量小。

这里实现一个环境 + DQN 框架，并调用 :mod:`tool` 中的绘图方法展示结果。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from tool import (
    ChipletNode,
    build_random_chiplet_graph,
    draw_chiplet_diagram,
)


@dataclass
class PlacementState:
    """
    简单的状态表示：chiplet 排列顺序（一个 permutation）。

    我们使用一个一维数组表示排列，环境内部根据排列顺序用简单的行优先排布算法
    计算每个 chiplet 的 (x, y) 位置。
    """

    order: List[int]  # indices into `nodes` list


class PlacementEnv:
    """
    强化学习环境：

    - 状态: chiplet 的排列顺序（Permutation）；
    - 动作: 任选两个位置 i, j，对应的 chiplet 交换位置；
    - 奖励: 负的“目标函数”：外接正方形面积 + 连线距离加权和。
    """

    def __init__(
        self,
        nodes: List[ChipletNode],
        edges: List[Tuple[str, str]],
        max_steps: int = 50,
        lambda_area: float = 1.0,
        lambda_wire: float = 0.1,
    ) -> None:
        self.nodes = nodes
        self.edges = edges
        self.max_steps = max_steps
        self.lambda_area = lambda_area
        self.lambda_wire = lambda_wire

        self.name_to_idx = {n.name: i for i, n in enumerate(nodes)}

        # 动作空间：所有 i < j 的 pair
        self.actions: List[Tuple[int, int]] = []
        n = len(nodes)
        for i in range(n):
            for j in range(i + 1, n):
                self.actions.append((i, j))

        self.num_actions = len(self.actions)
        self.reset()

    # ----------------------------- Env API -----------------------------

    def reset(self) -> np.ndarray:
        self.state = PlacementState(order=list(range(len(self.nodes))))
        random.shuffle(self.state.order)
        self.step_count = 0
        return self._state_to_vector()

    def step(self, action_idx: int):
        """
        执行动作（交换两个位置），返回 (next_state, reward, done, info)。
        """

        i, j = self.actions[action_idx]
        self.state.order[i], self.state.order[j] = (
            self.state.order[j],
            self.state.order[i],
        )
        self.step_count += 1

        reward = -self._objective()
        done = self.step_count >= self.max_steps

        return self._state_to_vector(), reward, done, {}

    # ----------------------------- Helpers -----------------------------

    def _state_to_vector(self) -> np.ndarray:
        """
        把 permutation 转成一个简单的数值向量，用于喂给 DQN。
        这里直接用 [0, 1] 归一化的 index。
        """

        arr = np.array(self.state.order, dtype=np.float32)
        return arr / max(len(self.state.order) - 1, 1)

    def _compute_positions(self) -> Dict[int, Tuple[float, float]]:
        """
        根据当前排列顺序，使用简单的“行优先”排布算法，计算每个 chiplet 左下角坐标。

        - 按顺序从左到右放，当当前行宽度超过阈值时换行；
        - 行宽阈值设置为 sum(widths) / 2，目的是让不同行的高度变化，从而影响外接方形面积。
        """

        widths = [float(n.dimensions.get("x", 0.0)) for n in self.nodes]
        heights = [float(n.dimensions.get("y", 0.0)) for n in self.nodes]

        total_w = sum(widths)
        row_limit = max(total_w / 2.0, max(widths))

        positions: Dict[int, Tuple[float, float]] = {}
        cur_x = 0.0
        cur_y = 0.0
        cur_row_h = 0.0

        for idx in self.state.order:
            w = widths[idx]
            h = heights[idx]
            if cur_x + w > row_limit and cur_x > 0:
                # 换行
                cur_x = 0.0
                cur_y += cur_row_h
                cur_row_h = 0.0

            positions[idx] = (cur_x, cur_y)
            cur_x += w
            cur_row_h = max(cur_row_h, h)

        return positions

    def _objective(self) -> float:
        """
        计算当前布局的目标函数值（越小越好）：

        obj = lambda_area * (bounding_square_area) +
              lambda_wire * (sum over edges of center_distance)
        """

        positions = self._compute_positions()
        widths = [float(n.dimensions.get("x", 0.0)) for n in self.nodes]
        heights = [float(n.dimensions.get("y", 0.0)) for n in self.nodes]

        # 计算外接矩形
        max_x = 0.0
        max_y = 0.0
        for idx, (x, y) in positions.items():
            w = widths[idx]
            h = heights[idx]
            max_x = max(max_x, x + w)
            max_y = max(max_y, y + h)

        side = max(max_x, max_y)  # 外接正方形的边长
        area = side * side

        # 计算连线距离（外接圆圆心之间的距离）
        name_to_idx = self.name_to_idx
        centers: Dict[int, Tuple[float, float]] = {}
        for idx, (x, y) in positions.items():
            centers[idx] = (x + widths[idx] / 2.0, y + heights[idx] / 2.0)

        wire_len = 0.0
        for s_name, d_name in self.edges:
            if s_name not in name_to_idx or d_name not in name_to_idx:
                continue
            si = name_to_idx[s_name]
            di = name_to_idx[d_name]
            if si not in centers or di not in centers:
                continue
            sx, sy = centers[si]
            dx, dy = centers[di]
            wire_len += math.hypot(sx - dx, sy - dy)

        return self.lambda_area * area + self.lambda_wire * wire_len

    # 提供给绘图的布局
    def current_layout(self) -> Dict[str, Tuple[float, float]]:
        positions = self._compute_positions()
        layout: Dict[str, Tuple[float, float]] = {}
        for idx, (x, y) in positions.items():
            name = self.nodes[idx].name
            layout[name] = (x, y)
        return layout


# ----------------------------- DQN -----------------------------


class DQN(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def push(self, transition):
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        return map(np.array, zip(*batch))

    def __len__(self):
        return len(self.buffer)


def train_dqn(
    env: PlacementEnv,
    num_episodes: int = 100,
    batch_size: int = 64,
    gamma: float = 0.99,
    lr: float = 1e-3,
    eps_start: float = 1.0,
    eps_end: float = 0.1,
    eps_decay: float = 0.995,
) -> Tuple[DQN, PlacementState]:
    """
    一个简单的 DQN 训练框架，用于在给定环境上搜索较好的布局。
    """

    state_dim = len(env.state.order)
    action_dim = env.num_actions

    policy_net = DQN(state_dim, action_dim)
    target_net = DQN(state_dim, action_dim)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    buffer = ReplayBuffer()

    eps = eps_start

    best_state = env.state
    best_obj = env._objective()

    for ep in range(num_episodes):
        state = env.reset()
        done = False

        while not done:
            # epsilon-greedy
            if random.random() < eps:
                action = random.randrange(action_dim)
            else:
                with torch.no_grad():
                    q = policy_net(torch.from_numpy(state).unsqueeze(0))
                    action = int(q.argmax().item())

            next_state, reward, done, _ = env.step(action)

            buffer.push((state, action, reward, next_state, float(done)))
            state = next_state

            # 更新最优解（注意 reward 是负的 obj）
            obj = env._objective()
            if obj < best_obj:
                best_obj = obj
                best_state = PlacementState(order=list(env.state.order))

            # 采样更新
            if len(buffer) >= batch_size:
                (
                    states,
                    actions,
                    rewards,
                    next_states,
                    dones,
                ) = buffer.sample(batch_size)

                states_t = torch.from_numpy(states).float()
                actions_t = torch.from_numpy(actions).long()
                rewards_t = torch.from_numpy(rewards).float()
                next_states_t = torch.from_numpy(next_states).float()
                dones_t = torch.from_numpy(dones).float()

                q_values = policy_net(states_t).gather(
                    1, actions_t.unsqueeze(1)
                ).squeeze(1)

                with torch.no_grad():
                    next_q = target_net(next_states_t).max(1)[0]
                    target = rewards_t + gamma * next_q * (1.0 - dones_t)

                loss = nn.functional.mse_loss(q_values, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # 软更新 target 网
        target_net.load_state_dict(policy_net.state_dict())
        eps = max(eps_end, eps * eps_decay)

        print(f"Episode {ep+1}/{num_episodes}, best_obj={best_obj:.3f}, eps={eps:.3f}")

    # 恢复到 best_state
    env.state = best_state
    return policy_net, best_state


if __name__ == "__main__":
    # 1. 构建初始图
    nodes, edges = build_random_chiplet_graph(edge_prob=0.3)

    # 2. 构建环境并训练 DQN（这里只跑少量 episode 作为示例）
    env = PlacementEnv(nodes, edges, max_steps=30)
    print(f"Env actions: {env.num_actions}, nodes: {len(nodes)}")

    policy, best_state = train_dqn(env, num_episodes=20, batch_size=32)

    # 3. 用当前 env.state 的布局生成可视化
    layout = env.current_layout()
    out_path = "/root/placement/thermal-placement/chiplet_dqn_placement.png"
    draw_chiplet_diagram(nodes, edges, save_path=out_path, layout=layout)
    print(f"DQN placement diagram saved to: {out_path}")

