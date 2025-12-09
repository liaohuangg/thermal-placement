"""
基于PPO（Proximal Policy Optimization）的chiplet布局优化算法。

PPO算法特点：
- Actor网络：学习策略π(a|s)，输出动作概率分布
- Critic网络：学习价值函数V(s)，评估状态价值
- 使用clipped objective限制策略更新幅度，提高稳定性
- 使用重要性采样比率（importance sampling ratio）
- 通常使用多个epoch更新策略，提高样本效率
"""

from __future__ import annotations

import math
import random
from typing import List, Tuple, Optional
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from rl_tool import PlacementEnv, PlacementState, ChipletPlacement
from tool import build_random_chiplet_graph, draw_chiplet_diagram


class ActorCritic(nn.Module):
    """Actor-Critic网络：共享特征提取层，分别输出策略和价值"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(ActorCritic, self).__init__()
        
        # 共享特征提取层
        self.shared_fc1 = nn.Linear(state_dim, hidden_dim)
        self.shared_fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Actor头：输出动作概率分布
        self.actor_fc = nn.Linear(hidden_dim, action_dim)
        
        # Critic头：输出状态价值
        self.critic_fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        返回:
            action_logits: 动作的对数概率（未归一化）
            value: 状态价值V(s)
        """
        # 共享特征提取
        x = F.relu(self.shared_fc1(state))
        x = F.relu(self.shared_fc2(x))
        
        # Actor：输出动作logits
        action_logits = self.actor_fc(x)
        
        # Critic：输出状态价值
        value = self.critic_fc(x)
        
        return action_logits, value
    
    def get_action_and_value(
        self, 
        state: torch.Tensor, 
        valid_actions: List[int],
        action_mask: Optional[torch.Tensor] = None
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        根据状态选择动作并计算价值
        
        参数:
            state: 状态向量 (batch_size, state_dim) 或 (state_dim,)
            valid_actions: 有效动作索引列表
            action_mask: 动作mask（可选）
        
        返回:
            action_idx: 选择的动作索引
            log_prob: 动作的对数概率
            value: 状态价值
        """
        action_logits, value = self.forward(state)
        
        # 处理batch维度：如果state是1D，action_logits会是1D；如果是2D，会是2D
        # 确保action_logits是2D的 (batch_size, action_dim)
        if action_logits.dim() == 1:
            action_logits = action_logits.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # 创建mask，只考虑有效动作
        if action_mask is None:
            action_mask = torch.zeros(action_logits.shape[-1], dtype=torch.bool, device=action_logits.device)
            action_mask[valid_actions] = True
        
        # 将无效动作的logits设为负无穷
        # action_logits shape: (batch_size, action_dim)
        # action_mask shape: (action_dim,)
        masked_logits = action_logits.clone()
        masked_logits[:, ~action_mask] = float('-inf')
        
        # 计算动作概率分布
        action_probs = F.softmax(masked_logits, dim=-1)
        
        # 从有效动作中采样
        if len(valid_actions) == 0:
            raise ValueError("No valid actions available")
        
        # 只考虑有效动作的概率（取第一个batch）
        valid_probs = action_probs[0, valid_actions]
        valid_probs = valid_probs / valid_probs.sum()  # 重新归一化
        
        # 采样动作
        action_idx_in_valid = torch.multinomial(valid_probs.unsqueeze(0), 1).item()
        action_idx = valid_actions[action_idx_in_valid]
        
        # 计算对数概率（取第一个batch）
        log_prob = F.log_softmax(masked_logits, dim=-1)[0, action_idx]
        
        # 如果原始输入是1D，squeeze value
        if squeeze_output:
            value = value.squeeze(0)
        
        return action_idx, log_prob, value.squeeze()
    
    def get_log_prob(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        valid_actions_list: List[List[int]]
    ) -> torch.Tensor:
        """
        计算给定状态和动作的对数概率（用于PPO的重要性采样）
        
        参数:
            state: 状态向量 (batch_size, state_dim)
            action: 动作索引 (batch_size,)
            valid_actions_list: 每个状态的有效动作列表
        
        返回:
            log_prob: 对数概率 (batch_size,)
        """
        action_logits, _ = self.forward(state)
        
        # 为每个状态创建mask
        batch_size = state.shape[0]
        action_dim = action_logits.shape[-1]
        masked_logits = action_logits.clone()
        
        for i in range(batch_size):
            if i < len(valid_actions_list):
                valid_actions = valid_actions_list[i]
                action_mask = torch.zeros(action_dim, dtype=torch.bool, device=action_logits.device)
                action_mask[valid_actions] = True
                masked_logits[i, ~action_mask] = float('-inf')
        
        # 计算对数概率
        log_probs = F.log_softmax(masked_logits, dim=-1)
        selected_log_probs = log_probs.gather(1, action.unsqueeze(1)).squeeze(1)
        
        return selected_log_probs


def train_ppo(
    env: PlacementEnv,
    num_episodes: int = 500,
    max_steps_per_episode: int = 100,
    gamma: float = 0.99,
    lr: float = 3e-4,
    clip_ratio: float = 0.2,  # PPO的clip参数
    value_coef: float = 0.5,  # Critic损失的权重
    entropy_coef: float = 0.01,  # 熵正则化权重
    ppo_epochs: int = 4,  # PPO更新epoch数
    use_gpu: bool = True,  # 是否使用GPU
) -> Tuple[ActorCritic, PlacementState]:
    """
    训练PPO算法
    
    参数:
        env: 布局环境
        num_episodes: 训练轮数
        max_steps_per_episode: 每个episode最大步数
        gamma: 折扣因子
        lr: 学习率
        clip_ratio: PPO的clip比率（通常0.1-0.3）
        value_coef: Critic损失权重
        entropy_coef: 熵正则化权重（鼓励探索）
        ppo_epochs: PPO更新epoch数（通常3-10）
        use_gpu: 是否使用GPU（如果可用）
    
    返回:
        (训练好的Actor-Critic网络, 最佳状态)
    """
    # 检测设备（添加实际GPU可用性测试）
    device = torch.device("cpu")
    if use_gpu and torch.cuda.is_available():
        try:
            # 尝试创建一个小的tensor来测试GPU是否真的可用
            # 这可以检测CUDA架构不兼容等问题
            test_tensor = torch.zeros(1).cuda()
            _ = test_tensor + 1  # 执行一个简单操作
            device = torch.device("cuda")
            print(f"使用GPU: {torch.cuda.get_device_name(0)}")
            del test_tensor
            torch.cuda.empty_cache()  # 清理测试缓存
        except (RuntimeError, Exception) as e:
            # GPU不可用（可能是架构不支持，如RTX 5090的sm_120），回退到CPU
            error_msg = str(e)[:150] if len(str(e)) > 150 else str(e)
            print(f"警告: GPU检测到但无法使用，回退到CPU")
            print(f"  错误信息: {error_msg}")
            print(f"  提示: RTX 5090 (sm_120) 需要PyTorch nightly版本支持")
            device = torch.device("cpu")
            print("使用CPU进行训练")
    else:
        print("使用CPU")
    
    # 计算状态和动作维度
    state_dim = len(env._state_to_vector())
    max_chiplets = len(env.nodes)
    max_action_dim = max_chiplets * env.grid_resolution * env.grid_resolution * 2
    
    # 创建Actor-Critic网络并移到GPU
    ac_net = ActorCritic(state_dim, max_action_dim).to(device)
    optimizer = optim.Adam(ac_net.parameters(), lr=lr)
    
    best_reward = float('-inf')
    best_state = None
    
    print(f"开始训练PPO: state_dim={state_dim}, max_action_dim={max_action_dim}, clip_ratio={clip_ratio}, ppo_epochs={ppo_epochs}")
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0.0
        episode_steps = 0
        
        # 存储轨迹
        states = []
        actions = []
        rewards = []
        old_log_probs = []  # 旧策略的对数概率（用于重要性采样）
        values = []
        dones = []
        valid_actions_list = []  # 每个状态的有效动作列表
        
        while episode_steps < max_steps_per_episode:
            # 获取有效动作
            valid_actions = env.get_valid_actions()
            
            if not valid_actions:
                break
            
            # 将状态转换为tensor并移到GPU
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            
            # 选择动作（使用当前策略）
            with torch.no_grad():
                action_idx, log_prob, value = ac_net.get_action_and_value(
                    state_tensor, valid_actions
                )
            
            # 执行动作
            action = env.decode_action(action_idx)
            next_state, reward, done, info = env.step(action)
            
            # 存储轨迹
            states.append(state)
            actions.append(action_idx)
            rewards.append(reward)
            old_log_probs.append(log_prob)  # 保存旧策略的对数概率
            values.append(value)
            dones.append(done)
            valid_actions_list.append(valid_actions)  # 保存有效动作列表
            
            state = next_state
            episode_reward += reward
            episode_steps += 1
            
            if done:
                break
        
        # 计算Advantage和回报
        if len(rewards) > 0:
            # 计算回报（使用GAE）
            returns = []
            advantages = []
            
            # 计算最后一个状态的价值（用于bootstrap）
            if not dones[-1]:
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(device)
                with torch.no_grad():
                    _, _, next_value = ac_net.get_action_and_value(
                        next_state_tensor, env.get_valid_actions()
                    )
            else:
                next_value = torch.tensor(0.0).to(device)
            
            # 从后往前计算回报和advantage（使用GAE）
            gae = torch.tensor(0.0).to(device)
            for t in reversed(range(len(rewards))):
                if dones[t]:
                    gae = torch.tensor(0.0).to(device)
                    next_value = torch.tensor(0.0).to(device)
                
                # TD error
                delta = torch.tensor(rewards[t]).to(device) + gamma * next_value - values[t]
                gae = delta + gamma * 0.95 * gae  # λ=0.95 for GAE
                
                advantages.insert(0, gae)
                returns.insert(0, gae + values[t])
                
                next_value = values[t]
            
            # 转换为tensor并移到GPU
            states_t = torch.FloatTensor(np.array(states)).to(device)
            actions_t = torch.LongTensor(actions).to(device)
            old_log_probs_t = torch.stack(old_log_probs).to(device)
            returns_t = torch.stack(returns).to(device)
            advantages_t = torch.stack(advantages).to(device)
            
            # 归一化advantages（减少方差）
            advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
            
            # PPO多epoch更新
            for epoch in range(ppo_epochs):
                # 前向传播计算当前策略和价值
                action_logits, values_pred = ac_net(states_t)
                
                # 计算当前策略的对数概率
                new_log_probs = ac_net.get_log_prob(states_t, actions_t, valid_actions_list)
                
                # 计算重要性采样比率
                ratio = torch.exp(new_log_probs - old_log_probs_t)
                
                # PPO clipped objective
                # L^CLIP = E[min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)]
                surr1 = ratio * advantages_t
                surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages_t
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # 计算熵（用于正则化）
                action_probs_all = F.softmax(action_logits, dim=-1)
                action_log_probs_all = F.log_softmax(action_logits, dim=-1)
                entropy = -(action_probs_all * action_log_probs_all).sum(dim=-1).mean()
                
                # Critic损失：(V(s) - Returns)²
                critic_loss = F.mse_loss(values_pred.squeeze(), returns_t)
                
                # 总损失
                total_loss = actor_loss + value_coef * critic_loss - entropy_coef * entropy
                
                # 反向传播
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(ac_net.parameters(), 0.5)
                optimizer.step()
        
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
                  f"Placed: {len(env.state.placed)}/{len(env.nodes)}")
    
    # 恢复最佳状态
    if best_state:
        env.state = best_state
    
    return ac_net, best_state


if __name__ == "__main__":
    # 1. 构建初始图
    nodes, edges = build_random_chiplet_graph(max_nodes=4, fixed_num_edges=4)
    print(f"加载了 {len(nodes)} 个chiplet，{len(edges)} 条边")
    
    # 2. 创建环境
    env = PlacementEnv(
        nodes=nodes,
        edges=edges,
        min_shared_length=0.5,  # 最小共享边长度
        grid_resolution=100,  # 与DQN保持一致
        overlap_penalty=1000.0,
        adjacency_penalty=50000.0,  # 大幅增加相邻约束惩罚，确保有链接关系的chiplet必须相邻！
        area_reward_scale=10.0,
        gap_penalty=200.0,
        adjacency_reward=100.0,
    )
    
    # 3. 训练PPO
    print("开始训练...")
    ac_net, best_state = train_ppo(
        env,
        num_episodes=5000,
        max_steps_per_episode=50,
        gamma=0.99,
        lr=3e-4,
        clip_ratio=0.2,  # PPO clip参数
        value_coef=0.5,
        entropy_coef=0.01,
        ppo_epochs=4,  # PPO更新epoch数
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
    
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)
    out_path = output_dir / "chiplet_ppo_placement.png"
    draw_chiplet_diagram(nodes_for_draw, edges, save_path=str(out_path), layout=layout)
    print(f"PPO布局图已保存到: {out_path}")
    
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

