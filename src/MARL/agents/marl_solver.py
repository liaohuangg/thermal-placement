"""
MAPPO Solver for Chiplet Placement
"""

import numpy as np
import torch
from typing import List, Dict, Tuple
from collections import defaultdict

from MARL.agents.policy_net import MAPPOAgent
from MARL.envs.chiplet_env import MultiAgentChipletEnv


class RolloutBuffer:
    """Buffer for storing rollout data"""
    
    def __init__(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        self.valid_actions = []
    
    def add(
        self, 
        obs: List[np.ndarray], 
        actions: List[int],
        rewards: List[float],
        log_probs: List[float],
        value: float,
        done: bool,
        valid_actions: List[List[int]]
    ):
        """Add experience to buffer"""
        # Flatten agent-wise data
        for i in range(len(obs)):
            self.observations.append(obs[i])
            self.actions.append(actions[i])
            self.rewards.append(rewards[i])
            self.log_probs.append(log_probs[i])
            self.values.append(value)  # Same value for all agents (centralized critic)
            self.dones.append(done)
            self.valid_actions.append(valid_actions[i] if i < len(valid_actions) else [])
    
    def compute_returns_and_advantages(self, last_value: float, gamma: float, gae_lambda: float):
        """Compute returns and advantages using GAE"""
        advantages = []
        returns = []
        
        gae = 0.0
        next_value = last_value
        
        for t in reversed(range(len(self.rewards))):
            if self.dones[t]:
                next_value = 0.0
                gae = 0.0
            
            delta = self.rewards[t] + gamma * next_value - self.values[t]
            gae = delta + gamma * gae_lambda * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + self.values[t])
            
            next_value = self.values[t]
        
        return returns, advantages
    
    def get(self):
        """Get all data as dictionary"""
        return {
            'observations': self.observations,
            'actions': self.actions,
            'log_probs': self.log_probs,
            'advantages': self.advantages,
            'returns': self.returns,
            'valid_actions': self.valid_actions,
        }
    
    def clear(self):
        """Clear buffer"""
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()
        self.valid_actions.clear()


class MAPPOSolver:
    """MAPPO solver for multi-agent chiplet placement"""
    
    def __init__(
        self,
        env: MultiAgentChipletEnv,
        config,
        device: torch.device
    ):
        self.env = env
        self.config = config
        self.device = device
        
        # Create MAPPO agent
        obs_dim = env.get_observation_space_size()
        action_dim = env.get_action_space_size()
        
        self.agent = MAPPOAgent(
            num_agents=env.num_agents,
            obs_dim=obs_dim,
            action_dim=action_dim,
            config=config,
            device=device
        )
        
        # Statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.best_reward = float('-inf')
        self.best_placements = None
    
    def train(self, num_episodes: int):
        """Train MAPPO agent"""
        
        print(f"Starting MAPPO training for {num_episodes} episodes...")
        print(f"Number of agents: {self.env.num_agents}")
        print(f"Observation dim: {self.env.get_observation_space_size()}")
        print(f"Action dim: {self.env.get_action_space_size()}")
        print(f"Device: {self.device}")
        
        for episode in range(num_episodes):
            episode_reward, episode_length, info = self._run_episode()
            
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            
            # Update best solution
            if episode_reward > self.best_reward:
                self.best_reward = episode_reward
                self.best_placements = [p for p in self.env.placements]
            
            # Logging
            if (episode + 1) % self.config.log_interval == 0:
                avg_reward = np.mean(self.episode_rewards[-self.config.log_interval:])
                avg_length = np.mean(self.episode_lengths[-self.config.log_interval:])
                
                print(f"Episode {episode + 1}/{num_episodes}")
                print(f"  Avg Reward: {avg_reward:.2f}")
                print(f"  Avg Length: {avg_length:.1f}")
                print(f"  Best Reward: {self.best_reward:.2f}")
                print(f"  Placed: {info['num_placed']}/{self.env.num_agents}")
                
                # Check adjacency constraints
                satisfied = 0
                total = len(self.env.connections)
                for i, j in self.env.connections:
                    if (self.env.placements[i] is not None and 
                        self.env.placements[j] is not None):
                        if self.env._check_adjacency_constraint(i, j):
                            satisfied += 1
                
                if total > 0:
                    print(f"  Adjacency: {satisfied}/{total} ({100*satisfied/total:.1f}%)")
                print()
        
        print("Training completed!")
        print(f"Best reward: {self.best_reward:.2f}")
        
        return self.best_placements
    
    def _run_episode(self) -> Tuple[float, int, Dict]:
        """Run one episode and update policy"""
        
        buffer = RolloutBuffer()
        observations = self.env.reset()
        
        episode_reward = 0.0
        step = 0
        done = False
        
        while not done and step < self.config.max_steps_per_episode:
            # Get valid actions for all agents
            valid_actions_list = [
                self.env.get_valid_actions(i) 
                for i in range(self.env.num_agents)
            ]
            
            # Select actions
            actions, log_probs = self.agent.get_actions(
                observations, valid_actions_list, deterministic=False
            )
            
            # Decode actions to (grid_x, grid_y, rotation)
            decoded_actions = []
            for agent_id, action_idx in enumerate(actions):
                if len(valid_actions_list[agent_id]) > 0:
                    # Map to valid action
                    if action_idx < len(valid_actions_list[agent_id]):
                        decoded_action = valid_actions_list[agent_id][action_idx]
                    else:
                        # Fallback: random valid action
                        decoded_action = valid_actions_list[agent_id][
                            action_idx % len(valid_actions_list[agent_id])
                        ]
                else:
                    decoded_action = (0, 0, 0)  # Dummy action
                
                decoded_actions.append(decoded_action)
            
            # Get value estimate
            value = self.agent.get_value(observations)
            
            # Step environment
            next_observations, rewards, done, info = self.env.step(decoded_actions)
            
            # Store in buffer
            buffer.add(
                observations,
                actions,
                rewards,
                log_probs,
                value,
                done,
                valid_actions_list
            )
            
            observations = next_observations
            episode_reward += sum(rewards)
            step += 1
        
        # Compute returns and advantages
        last_value = self.agent.get_value(observations) if not done else 0.0
        returns, advantages = buffer.compute_returns_and_advantages(
            last_value, self.config.gamma, self.config.gae_lambda
        )
        
        buffer.returns = returns
        buffer.advantages = advantages
        
        # Update policy
        if len(buffer.observations) > 0:
            train_info = self.agent.update(buffer.get())
        
        return episode_reward, step, info
    
    def evaluate(self, num_episodes: int = 10) -> Dict:
        """Evaluate current policy"""
        
        eval_rewards = []
        eval_lengths = []
        eval_success = []
        
        for _ in range(num_episodes):
            observations = self.env.reset()
            done = False
            episode_reward = 0.0
            step = 0
            
            while not done and step < self.config.max_steps_per_episode:
                valid_actions_list = [
                    self.env.get_valid_actions(i) 
                    for i in range(self.env.num_agents)
                ]
                
                actions, _ = self.agent.get_actions(
                    observations, valid_actions_list, deterministic=True
                )
                
                decoded_actions = []
                for agent_id, action_idx in enumerate(actions):
                    if len(valid_actions_list[agent_id]) > 0:
                        decoded_action = valid_actions_list[agent_id][
                            action_idx % len(valid_actions_list[agent_id])
                        ]
                    else:
                        decoded_action = (0, 0, 0)
                    decoded_actions.append(decoded_action)
                
                next_observations, rewards, done, info = self.env.step(decoded_actions)
                
                observations = next_observations
                episode_reward += sum(rewards)
                step += 1
            
            eval_rewards.append(episode_reward)
            eval_lengths.append(step)
            eval_success.append(info['num_placed'] == self.env.num_agents)
        
        return {
            'mean_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'mean_length': np.mean(eval_lengths),
            'success_rate': np.mean(eval_success),
        }
