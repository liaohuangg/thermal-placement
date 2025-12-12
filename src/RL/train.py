"""
PPO训练脚本 - 芯片布局强化学习

使用Proximal Policy Optimization (PPO)算法训练芯片布局策略
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from typing import List, Tuple, Dict
import json
from pathlib import Path

from env import ChipletPlacementEnv, create_env_from_json

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


class ActorCritic(nn.Module):
    """Actor-Critic网络"""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        # 共享特征提取层
        self.feature = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Actor网络（策略）
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        
        # Critic网络（价值函数）
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, x):
        """前向传播"""
        features = self.feature(x)
        action_logits = self.actor(features)
        value = self.critic(features)
        return action_logits, value
    
    def get_action(self, obs, valid_actions):
        """
        根据观察选择动作
        
        Args:
            obs: 观察向量 (已在设备上)
            valid_actions: 有效动作列表
            
        Returns:
            action, log_prob, value
        """
        action_logits, value = self.forward(obs)
        
        # 创建动作掩码（只有有效动作可以选择）

        mask = torch.ones(action_logits.shape[-1], device=obs.device) * float('-inf')
        mask[valid_actions] = 0
        masked_logits = action_logits + mask
        
        # 采样动作
        probs = torch.softmax(masked_logits, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob, value


class PPOTrainer:
    """PPO训练器"""
    
    def __init__(
        self,
        env: ChipletPlacementEnv,
        lr: float = 0.0003,
        gamma: float = 0.99,
        epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.2,
        max_grad_norm: float = 0.5,
        hidden_dim: int = 256,
    ):
        """
        初始化PPO训练器
        
        Args:
            env: 环境
            lr: 学习率
            gamma: 折扣因子
            epsilon: PPO裁剪参数
            value_coef: 价值损失系数
            entropy_coef: 熵奖励系数
            max_grad_norm: 梯度裁剪
            hidden_dim: 隐藏层维度
        """
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # 创建网络并移到GPU
        self.model = ActorCritic(env.observation_dim, env.action_dim, hidden_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # 经验缓冲
        self.reset_buffer()
    
    def reset_buffer(self):
        """重置经验缓冲"""
        self.observations = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
    
    def collect_episode(self) -> Tuple[float, bool]:
        """
        收集一个episode的数据
        
        Returns:
            (total_reward, success)
        """
        obs = self.env.reset()
        done = False
        total_reward = 0.0
        step_count = 0
        max_steps = self.env.num_chiplets * 10  # 防止无限循环
        
        while not done and step_count < max_steps:
            step_count += 1
            
            # 获取有效动作
            valid_actions = self.env.get_valid_actions()
            
            if not valid_actions:
                # 无有效动作，episode失败
                return total_reward, False
            
            # 选择动作
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                action, log_prob, value = self.model.get_action(obs_tensor, valid_actions)
            
            # 执行动作
            next_obs, reward, done, info = self.env.step(action)
            
            # 存储经验（至少需要1个样本）
            self.observations.append(obs)
            self.actions.append(action)
            self.log_probs.append(log_prob)
            self.values.append(value)
            self.rewards.append(reward)
            self.dones.append(done)
            
            obs = next_obs
            total_reward += reward
        
        return total_reward, done
    
    def compute_returns(self) -> torch.Tensor:
        """计算折扣回报"""
        returns = []
        R = 0
        
        for reward, done in zip(reversed(self.rewards), reversed(self.dones)):
            if done:
                R = 0
            R = reward + self.gamma * R
            returns.insert(0, R)
        
        return torch.FloatTensor(returns)
    
    def update(self, epochs: int = 4):
        """
        PPO更新
        
        Args:
            epochs: 更新轮数
        """
        # 需要至少1个样本才能更新
        if len(self.observations) == 0:
            return {}
        
        # 如果只有1个样本，跳过标准化（会导致std=0）
        if len(self.observations) == 1:
            self.reset_buffer()
            return {}
        
        # 转换为tensor并移到GPU
        obs_tensor = torch.FloatTensor(np.array(self.observations)).to(device)
        actions_tensor = torch.LongTensor(self.actions).to(device)
        old_log_probs = torch.stack(self.log_probs).to(device)
        old_values = torch.cat(self.values).to(device)
        
        # 计算回报和优势
        returns = self.compute_returns().to(device)
        advantages = returns - old_values.detach()
        
        # 裁剪advantages防止极端值
        advantages = torch.clamp(advantages, -10.0, 10.0)
        
        # 数值稳定的标准化（避免除以0）
        adv_mean = advantages.mean()
        adv_std = advantages.std()
        if adv_std > 1e-8:  # 避免除以接近0的数
            advantages = (advantages - adv_mean) / (adv_std + 1e-8)
        else:
            advantages = advantages - adv_mean  # 至少中心化
        
        # 再次裁剪标准化后的advantages
        advantages = torch.clamp(advantages, -5.0, 5.0)
        
        # 多轮更新
        total_loss = 0
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        total_kl = 0
        
        for epoch in range(epochs):
            # 前向传播
            action_logits, values = self.model(obs_tensor)
            
            # 计算当前策略的log概率
            probs = torch.softmax(action_logits, dim=-1)
            
            # 检查并修复NaN
            if torch.isnan(probs).any():
                print(f"警告: 检测到NaN概率，使用均匀分布代替")
                probs = torch.ones_like(probs) / probs.shape[-1]
            
            dist = Categorical(probs)
            log_probs = dist.log_prob(actions_tensor)
            entropy = dist.entropy().mean()
            
            # 计算KL散度（检测策略变化）
            with torch.no_grad():
                kl_div = (old_log_probs - log_probs).mean()
                total_kl += kl_div.item()
            
            # 如果KL散度过大，提前停止更新
            if epoch > 0 and kl_div > 0.015:  # KL阈值
                break
            
            # PPO损失
            ratio = torch.exp(log_probs - old_log_probs.detach())
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # 价值损失
            critic_loss = nn.MSELoss()(values.squeeze(-1), returns)
            
            # 总损失
            loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            total_loss += loss.item()
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_entropy += entropy.item()
        
        # 清空缓冲
        self.reset_buffer()
        
        actual_epochs = epoch + 1 if 'epoch' in locals() else epochs
        
        return {
            'loss': total_loss / actual_epochs,
            'actor_loss': total_actor_loss / actual_epochs,
            'critic_loss': total_critic_loss / actual_epochs,
            'entropy': total_entropy / actual_epochs,
            'kl_div': total_kl / actual_epochs,
        }
    
    def save(self, path: str):
        """保存模型和环境配置"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'env_config': {
                'grid_resolution': self.env.grid_resolution,
                'max_width': self.env.max_width,
                'max_height': self.env.max_height,
                'min_overlap': self.env.min_overlap,
                'observation_dim': self.env.observation_dim,
                'action_dim': self.env.action_dim,
            }
        }, path)
        print(f"模型已保存到 {path}")
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"模型已从 {path} 加载")


def train(
    json_path: str,
    num_episodes: int = 1000,
    save_interval: int = 100,
    log_interval: int = 10,
    **env_kwargs
):
    """
    训练函数
    
    Args:
        json_path: 输入JSON路径
        num_episodes: 训练episode数
        save_interval: 保存间隔
        log_interval: 日志间隔
        **env_kwargs: 环境参数
    """
    print("=" * 70)
    print("PPO训练 - 芯片布局优化")
    print("=" * 70)
    
    # 创建环境
    env = create_env_from_json(json_path, **env_kwargs)
    print(f"\n环境信息:")
    print(f"  芯片数量: {env.num_chiplets}")
    print(f"  放置顺序: {env.placement_order}")
    print(f"  观察维度: {env.observation_dim}")
    print(f"  动作维度: {env.action_dim}")
    
    # 创建训练器
    trainer = PPOTrainer(env)
    
    # 训练统计
    episode_rewards = []
    success_count = 0
    best_avg_reward = float('-inf')
    best_model_path = None
    patience_counter = 0
    max_patience = 5  # 连续5次保存间隔性能下降就警告
    
    print(f"\n开始训练...")
    print("-" * 70)
    
    for episode in range(1, num_episodes + 1):
        # 收集经验
        reward, success = trainer.collect_episode()
        episode_rewards.append(reward)
        
        if success:
            success_count += 1
        
        # 更新策略
        metrics = trainer.update(epochs=4)
        
        # 日志
        if episode % log_interval == 0:
            avg_reward = np.mean(episode_rewards[-log_interval:])
            success_rate = success_count / episode
            
            print(f"Episode {episode}/{num_episodes}")
            print(f"  平均奖励: {avg_reward:.2f}")
            print(f"  成功率: {success_rate*100:.1f}%")
            if metrics:
                print(f"  损失: {metrics['loss']:.4f}")
                print(f"  熵: {metrics['entropy']:.4f}")
                print(f"  KL散度: {metrics.get('kl_div', 0.0):.6f}")
            
            # 检测性能下降
            if episode > log_interval * 2:
                prev_avg = np.mean(episode_rewards[-log_interval*2:-log_interval])
                if avg_reward < prev_avg * 0.8:  # 下降超过20%
                    print(f"  ⚠️ 警告：性能下降 {prev_avg:.1f} → {avg_reward:.1f}")
        
        # 保存模型
        if episode % save_interval == 0:
            Path("checkpoints").mkdir(exist_ok=True)
            checkpoint_path = f"checkpoints/ppo_episode_{episode}.pt"
            trainer.save(checkpoint_path)
            
            # 检查是否是最佳模型
            recent_avg = np.mean(episode_rewards[-save_interval:])
            if recent_avg > best_avg_reward:
                best_avg_reward = recent_avg
                best_model_path = checkpoint_path
                patience_counter = 0
                print(f"  ✓ 新的最佳模型！平均奖励: {recent_avg:.2f}")
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    print(f"  ⚠️ 警告：连续{max_patience}次保存未改进，考虑提前停止")
    
    # 保存最终模型
    Path("checkpoints").mkdir(exist_ok=True)
    trainer.save("checkpoints/ppo_model.pt")
    
    # 如果有更好的模型，复制为最佳模型
    if best_model_path and best_model_path != "checkpoints/ppo_model.pt":
        import shutil
        shutil.copy(best_model_path, "checkpoints/ppo_best.pt")
        print(f"\n✓ 最佳模型已保存: {best_model_path}")
        print(f"  最佳平均奖励: {best_avg_reward:.2f}")
    
    print("\n" + "=" * 70)
    print("训练完成！")
    print(f"  总episodes: {num_episodes}")
    print(f"  最终成功率: {success_count/num_episodes*100:.1f}%")
    print(f"  平均奖励: {np.mean(episode_rewards[-100:]):.2f}")
    print("=" * 70)
    
    return trainer


if __name__ == "__main__":
    # 训练配置
    json_file = "../../baseline/ICCAD23/test_input/6core.json"
    
    trained_model = train(
        json_path=json_file,
        num_episodes=5000,
        save_interval=100,
        log_interval=100,
        grid_resolution=50,
        max_width=100.0,
        max_height=100.0,
        min_overlap=0.5,
        # 优化目标：只关注利用率
        placement_reward=10,  # 降低放置奖励
        adjacency_reward=10,   # 降低邻接奖励
        compact = 80,
        min_wirelength_reward_scale =20   
    )
    
    print(f"\n✓ 模型已保存到 checkpoints/ppo_model.pt")
