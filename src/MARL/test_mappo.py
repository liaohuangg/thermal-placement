"""
Quick test script for MAPPO implementation
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
from MARL.config import MAPPOConfig
from MARL.envs.chiplet_env import MultiAgentChipletEnv
from MARL.agents.marl_solver import MAPPOSolver


def test_environment():
    """Test environment functionality"""
    print("Testing Environment...")
    
    # Simple test case
    chiplet_sizes = [(10, 10), (10, 10), (10, 10)]
    connections = [(0, 1), (1, 2)]
    
    config = MAPPOConfig()
    config.grid_resolution = 30
    
    env = MultiAgentChipletEnv(chiplet_sizes, connections, config)
    
    # Test reset
    obs = env.reset()
    print(f"  Observation shape: {obs[0].shape}")
    print(f"  Number of agents: {env.num_agents}")
    
    # Test valid actions
    valid_actions = env.get_valid_actions(0)
    print(f"  Valid actions for agent 0: {len(valid_actions)}")
    
    # Test step
    if len(valid_actions) > 0:
        actions = [valid_actions[0] if i == 0 else (0, 0, 0) for i in range(env.num_agents)]
        next_obs, rewards, done, info = env.step(actions)
        print(f"  Step successful: placed={info['num_placed']}, remaining={info['num_remaining']}")
    
    print("  Environment test passed!\n")


def test_policy_network():
    """Test policy network"""
    print("Testing Policy Network...")
    
    config = MAPPOConfig()
    obs_dim = 100
    action_dim = 50
    
    from MARL.agents.policy_net import Actor, Critic
    
    actor = Actor(obs_dim, action_dim)
    critic = Critic(obs_dim * 3)  # 3 agents
    
    # Test forward pass
    obs = torch.randn(1, obs_dim)
    action_logits = actor(obs)
    print(f"  Actor output shape: {action_logits.shape}")
    
    global_obs = torch.randn(1, obs_dim * 3)
    value = critic(global_obs)
    print(f"  Critic output shape: {value.shape}")
    
    # Test action selection
    action, log_prob = actor.get_action(obs.squeeze(0), valid_actions=[1, 2, 3, 5, 10])
    print(f"  Selected action: {action.item()}, log_prob: {log_prob.item():.4f}")
    
    print("  Policy network test passed!\n")


def test_training_loop():
    """Test a short training loop"""
    print("Testing Training Loop...")
    
    chiplet_sizes = [(8, 8), (8, 8), (8, 8)]
    connections = [(0, 1), (1, 2)]
    
    config = MAPPOConfig()
    config.grid_resolution = 25
    config.num_episodes = 5
    config.log_interval = 2
    
    env = MultiAgentChipletEnv(chiplet_sizes, connections, config)
    
    device = torch.device("cpu")
    solver = MAPPOSolver(env, config, device)
    
    # Train for a few episodes
    best_placements = solver.train(num_episodes=5)
    
    print(f"  Training completed!")
    print(f"  Best reward: {solver.best_reward:.2f}")
    print(f"  Episodes run: {len(solver.episode_rewards)}")
    
    print("  Training loop test passed!\n")


if __name__ == "__main__":
    print("=" * 60)
    print("MAPPO Implementation Tests")
    print("=" * 60)
    print()
    
    try:
        test_environment()
        test_policy_network()
        test_training_loop()
        
        print("=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        print()
        print("You can now run the full training with:")
        print("  python src/MARL/train_mappo.py")
        
    except Exception as e:
        print(f"\n❌ Test failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
