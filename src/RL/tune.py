"""
迁移学习脚本 - 从6core到12core

使用在6core上训练的模型，在12core上继续训练
"""

import torch
import sys
from pathlib import Path

# 添加路径
sys.path.append(str(Path(__file__).parent))

from train import PPOTrainer, train
from env import create_env_from_json

def transfer_learning():
    """迁移学习：6core -> 12core"""

    print("=" * 70)
    print("迁移学习：6core -> 12core")
    print("=" * 70)

    # 1. 创建12core环境
    env_12core = create_env_from_json(
        "../../baseline/ICCAD23/test_input/12core.json",
        grid_resolution=100,
        max_width=100.0,
        max_height=100.0,
        min_overlap=0.5,
        placement_reward=20,
        adjacency_reward=20,
        compact=20,
        min_wirelength_reward_scale=0,
        extra_adjacency_reward=350
    )

    print("12core环境信息:")
    print(f"  芯片数量: {env_12core.num_chiplets}")
    print(f"  观察维度: {env_12core.observation_dim}")
    print(f"  动作维度: {env_12core.action_dim}")

    # 2. 创建训练器（针对12core）
    trainer = PPOTrainer(
        env_12core,
        lr=5e-4,  # 降低学习率
        entropy_coef=0.6,  # 增加熵系数鼓励探索
        # max_grad_norm=0.5,   # 减小梯度裁剪阈值
        load_optimizer=True  # 加载优化器状态
    )

    # 3. 加载6core预训练模型
    model_path = "checkpoints/12ppo_episode_500.pt"  # 训练好的model路径
    if Path(model_path).exists():
        trainer.load(model_path)
        print(f"✓ 已加载预训练模型: {model_path}")
    else:
        print(f"⚠️ 预训练模型不存在: {model_path}，将从头训练")

    # 4. 在12core上继续训练
    print("\n开始在12core上fine-tune...")
    trained_model = train(
        json_path="../../baseline/ICCAD23/test_input/12core.json",
        num_episodes=2000,  # fine-tune episode数
        save_interval=100,
        log_interval=100,
        trainer=trainer,  # 使用已加载的trainer
        grid_resolution=100,
        max_width=100.0, 
        max_height=100.0,
        min_overlap=0.5,
        placement_reward=20,  # 增加放置奖励
        adjacency_reward=20,
        compact=20,
        min_wirelength_reward_scale=0,
        extra_adjacency_reward=350
    )

    print("\n✓ 迁移学习完成！模型已保存")

if __name__ == "__main__":
    transfer_learning()
