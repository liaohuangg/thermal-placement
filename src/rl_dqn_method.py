"""
基于DQN的chiplet布局优化算法。

问题描述：
- 给定一组方块（chiplets），每个有长宽，方块之间有互联关系
- 要求：
  1. 无重叠排布在平面上
  2. 有互联关系的芯粒需要边缘贴近，并且相邻的边需要有一定长度的共享范围（可指定）
  3. 最小化覆盖所有方块的外接方框的面积

方法：
- 使用DQN（Deep Q-Network）强化学习算法
- 状态：已放置chiplet的位置、旋转状态，以及待放置chiplet的信息
- 动作：选择下一个chiplet，选择其位置和旋转
- 奖励：惩罚重叠、惩罚不满足相邻约束，奖励小面积
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from rl_tool import (
    PlacementEnv,
    PlacementState,
    ChipletPlacement,
    DQN,
    ReplayBuffer,
)
from tool import (
    build_random_chiplet_graph,
    draw_chiplet_diagram,
)


def train_dqn(
    env: PlacementEnv,
    num_episodes: int = 500,
    batch_size: int = 64,
    gamma: float = 0.99,
    lr: float = 1e-4,
    eps_start: float = 1.0,
    eps_end: float = 0.01,
    eps_decay: float = 0.995,
    target_update: int = 10,
    replay_buffer_size: int = 10000,
    max_steps_per_episode: int = 100,
    use_gpu: bool = True,  # 是否使用GPU
) -> Tuple[DQN, PlacementState]:
    """
    训练DQN算法
    
    参数:
        env: 布局环境
        num_episodes: 训练轮数
        batch_size: 批次大小
        gamma: 折扣因子
        lr: 学习率
        eps_start: 初始探索率
        eps_end: 最终探索率
        eps_decay: 探索率衰减
        target_update: 目标网络更新频率
        replay_buffer_size: 经验回放缓冲区大小
        max_steps_per_episode: 每个episode最大步数
        use_gpu: 是否使用GPU（如果可用）
    
    返回:
        (训练好的policy网络, 最佳状态)
    """
    # 检测设备（添加实际GPU可用性测试）
    device = torch.device("cpu")
    if use_gpu and torch.cuda.is_available():
        try:
            # 尝试创建一个小的tensor来测试GPU是否真的可用
            test_tensor = torch.zeros(1).cuda()
            _ = test_tensor + 1  # 执行一个简单操作
            device = torch.device("cuda")
            print(f"使用GPU: {torch.cuda.get_device_name(0)}")
            del test_tensor
            torch.cuda.empty_cache()
        except (RuntimeError, Exception) as e:
            # GPU不可用，回退到CPU
            error_msg = str(e)[:150] if len(str(e)) > 150 else str(e)
            print(f"警告: GPU检测到但无法使用，回退到CPU")
            print(f"  错误信息: {error_msg}")
            device = torch.device("cpu")
            print("使用CPU进行训练")
    else:
        print("使用CPU")
    
    # 计算状态和动作维度
    state_dim = len(env._state_to_vector())
    
    # 动作空间：最大chiplet数 * 网格位置 * 旋转
    # 注意：实际动作空间大小会根据remaining chiplet数量动态变化
    max_chiplets = len(env.nodes)
    max_action_dim = max_chiplets * env.grid_resolution * env.grid_resolution * 2
    
    # 创建网络（使用最大动作空间）并移到GPU
    policy_net = DQN(state_dim, max_action_dim).to(device)
    target_net = DQN(state_dim, max_action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    replay_buffer = ReplayBuffer(replay_buffer_size)
    
    eps = eps_start
    best_reward = float('-inf')
    best_state = None
    
    print(f"开始训练DQN: state_dim={state_dim}, max_action_dim={max_action_dim}")
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0.0
        episode_steps = 0
        
        while episode_steps < max_steps_per_episode:
            # 获取有效动作
            valid_actions = env.get_valid_actions()
            
            if not valid_actions:
                break
            
            # 选择动作（epsilon-greedy）
            if random.random() < eps:
                action_idx = random.choice(valid_actions)
            else:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                with torch.no_grad():
                    q_values = policy_net(state_tensor)
                    # 只考虑有效动作
                    valid_q_values = q_values[0][valid_actions]
                    best_valid_idx = torch.argmax(valid_q_values).item()
                    action_idx = valid_actions[best_valid_idx]
            
            # 执行动作
            action = env.decode_action(action_idx)
            next_state, reward, done, info = env.step(action)
            
            # 存储经验
            replay_buffer.push(state, action_idx, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            episode_steps += 1
            
            if done:
                break
        
        # 训练网络
        if len(replay_buffer) >= batch_size:
            states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
            
            # 将数据移到GPU
            states_t = torch.FloatTensor(states).to(device)
            actions_t = torch.LongTensor(actions).to(device)
            rewards_t = torch.FloatTensor(rewards).to(device)
            next_states_t = torch.FloatTensor(next_states).to(device)
            dones_t = torch.FloatTensor(dones).to(device)
            
            # 计算当前Q值
            q_values = policy_net(states_t)
            q_value = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)
            
            # 计算目标Q值
            with torch.no_grad():
                next_q_values = target_net(next_states_t)
                # 对于每个next_state，需要找到有效动作的最大Q值
                # 简化处理：使用全局最大值（实际应该只考虑有效动作）
                max_next_q = next_q_values.max(1)[0]
                target_q = rewards_t + gamma * max_next_q * (1 - dones_t)
            
            # 计算损失
            loss = F.mse_loss(q_value, target_q)
            
            # 优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # 更新目标网络
        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        # 衰减探索率
        eps = max(eps_end, eps * eps_decay)
        
        # 记录最佳状态
        is_complete = len(env.state.remaining) == 0
        
        should_update = False
        if best_state is None:
            should_update = True
        else:
            best_is_complete = len(best_state.remaining) == 0
            if is_complete and not best_is_complete:
                should_update = True
            elif is_complete == best_is_complete:
                if episode_reward > best_reward:
                    should_update = True
        
        if should_update:
            best_reward = episode_reward
            best_state = PlacementState(
                placed=list(env.state.placed),
                remaining=list(env.state.remaining),
                grid_mask=env.state.grid_mask.copy()
            )
        
        if (episode + 1) % 50 == 0:
            completion_status = "✓" if is_complete else "✗"
            best_completion_status = "✓" if (best_state and len(best_state.remaining) == 0) else "✗"
            print(f"Episode {episode+1}/{num_episodes}, "
                  f"Reward: {episode_reward:.2f} ({completion_status}), "
                  f"Best Reward: {best_reward:.2f} ({best_completion_status}), "
                  f"Placed: {len(env.state.placed)}/{len(env.nodes)}, "
                  f"Epsilon: {eps:.3f}")
    
    # 恢复最佳状态
    if best_state:
        env.state = best_state
    
    return policy_net, best_state


if __name__ == "__main__":
    # 1. 构建初始图
    nodes, edges = build_random_chiplet_graph(max_nodes=4, fixed_num_edges=4)
    print(f"加载了 {len(nodes)} 个chiplet，{len(edges)} 条边")
    
    # 2. 创建环境
    env = PlacementEnv(
        nodes=nodes,
        edges=edges,
        min_shared_length=0.5,  # 最小共享边长度
        grid_resolution=25,
        overlap_penalty=1000.0,
        adjacency_penalty=50000.0,  # 大幅增加相邻约束惩罚，确保有链接关系的chiplet必须相邻！
        area_reward_scale=10.0,
        gap_penalty=200.0,
        adjacency_reward=100.0,
    )
    
    # 3. 训练DQN
    print("开始训练...")
    policy_net, best_state = train_dqn(
        env,
        num_episodes=500,
        batch_size=64,
        gamma=0.99,
        lr=1e-4,
        eps_start=1.0,
        eps_end=0.01,
        eps_decay=0.995,
        target_update=10,
        replay_buffer_size=10000,
        max_steps_per_episode=50,
        use_gpu=True,  # 使用GPU加速
    )
    
    # 4. 可视化最佳布局
    layout, rotations = env.get_current_layout()
    
    # 调整nodes以反映旋转
    nodes_for_draw = []
    for node in nodes:
        if node.name in rotations and rotations[node.name]:
            from copy import deepcopy
            node_copy = deepcopy(node)
            orig_w = node.dimensions.get("x", 0.0)
            orig_h = node.dimensions.get("y", 0.0)
            node_copy.dimensions["x"] = orig_h
            node_copy.dimensions["y"] = orig_w
            nodes_for_draw.append(node_copy)
        else:
            nodes_for_draw.append(node)
    
    from pathlib import Path
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)
    out_path = output_dir / "chiplet_dqn_placement.png"
    draw_chiplet_diagram(nodes_for_draw, edges, save_path=str(out_path), layout=layout)
    print(f"DQN布局图已保存到: {out_path}")
    
    # 5. 打印布局统计信息
    bbox_w, bbox_h = env._compute_bounding_box()
    print(f"外接框尺寸: {bbox_w:.2f} x {bbox_h:.2f}, 面积: {bbox_w * bbox_h:.2f}")
    
    # 检查相邻约束满足情况
    satisfied = 0
    total = len(env.connected_pairs)
    for i, j in env.connected_pairs:
        if env._check_adjacency(i, j):
            satisfied += 1
    print(f"相邻约束满足: {satisfied}/{total}")
