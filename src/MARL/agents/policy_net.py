"""
MAPPO Policy Networks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional, Dict
import numpy as np


class Actor(nn.Module):
    """Actor network for MAPPO (decentralized execution)"""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super(Actor, self).__init__()
        
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.action_head = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            obs: observation tensor [batch_size, obs_dim]
        
        Returns:
            action_logits: [batch_size, action_dim]
        """
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        action_logits = self.action_head(x)
        return action_logits
    
    def get_action(
        self, 
        obs: torch.Tensor, 
        valid_actions: Optional[List[int]] = None,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get action and log probability
        
        Args:
            obs: observation [obs_dim]
            valid_actions: list of valid action indices
            deterministic: if True, select argmax action
        
        Returns:
            action: selected action index
            log_prob: log probability of the action
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        
        action_logits = self.forward(obs)
        
        # Mask invalid actions
        if valid_actions is not None and len(valid_actions) > 0:
            mask = torch.full((action_logits.shape[-1],), float('-inf'), device=obs.device)
            mask[valid_actions] = 0.0
            action_logits = action_logits + mask.unsqueeze(0)
        
        # Compute action probabilities
        action_probs = F.softmax(action_logits, dim=-1)
        
        # Handle NaN in probabilities
        if torch.isnan(action_probs).any():
            # Fallback to uniform distribution over valid actions
            action_probs = torch.zeros_like(action_logits)
            if valid_actions is not None and len(valid_actions) > 0:
                action_probs[0, valid_actions] = 1.0 / len(valid_actions)
            else:
                action_probs.fill_(1.0 / action_probs.shape[-1])
        
        if deterministic:
            if valid_actions is not None and len(valid_actions) > 0:
                # Select best valid action
                valid_probs = action_probs[0, valid_actions]
                best_idx = torch.argmax(valid_probs)
                action = torch.tensor(valid_actions[best_idx], device=obs.device)
            else:
                action = torch.argmax(action_probs, dim=-1).squeeze(0)
        else:
            # Sample from distribution
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample().squeeze(0)
        
        # Compute log probability
        log_prob = torch.log(action_probs[0, action] + 1e-10)
        
        return action, log_prob
    
    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        valid_actions_batch: Optional[List[List[int]]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate actions (for training)
        
        Args:
            obs: observations [batch_size, obs_dim]
            actions: actions [batch_size]
            valid_actions_batch: list of valid action lists for each sample
        
        Returns:
            log_probs: log probabilities [batch_size]
            entropy: entropy [batch_size]
        """
        action_logits = self.forward(obs)
        
        # Mask invalid actions
        if valid_actions_batch is not None:
            for i, valid_actions in enumerate(valid_actions_batch):
                if valid_actions and len(valid_actions) > 0:
                    mask = torch.full((action_logits.shape[-1],), float('-inf'), device=obs.device)
                    mask[valid_actions] = 0.0
                    action_logits[i] = action_logits[i] + mask
        
        log_probs = F.log_softmax(action_logits, dim=-1)
        action_probs = F.softmax(action_logits, dim=-1)
        
        # Get log prob of selected actions
        selected_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute entropy (avoid NaN)
        entropy = -(action_probs * log_probs).sum(dim=-1)
        entropy = torch.nan_to_num(entropy, nan=0.0)
        
        return selected_log_probs, entropy


class Critic(nn.Module):
    """Centralized critic network for MAPPO"""
    
    def __init__(self, global_obs_dim: int, hidden_dim: int = 256):
        super(Critic, self).__init__()
        
        self.fc1 = nn.Linear(global_obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, global_obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            global_obs: global observation [batch_size, global_obs_dim]
        
        Returns:
            value: state value [batch_size, 1]
        """
        x = F.relu(self.fc1(global_obs))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        value = self.value_head(x)
        return value


class MAPPOAgent:
    """MAPPO Agent managing multiple actors and a centralized critic"""
    
    def __init__(
        self,
        num_agents: int,
        obs_dim: int,
        action_dim: int,
        config,
        device: torch.device
    ):
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.config = config
        self.device = device
        
        # Create actors (one per agent, with shared parameters)
        self.actor = Actor(obs_dim, action_dim).to(device)
        
        # Centralized critic (takes concatenated observations)
        global_obs_dim = obs_dim * num_agents
        self.critic = Critic(global_obs_dim).to(device)
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), 
            lr=config.lr_actor
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), 
            lr=config.lr_critic
        )
    
    def get_actions(
        self,
        observations: List[np.ndarray],
        valid_actions_list: List[List[int]],
        deterministic: bool = False
    ) -> Tuple[List[int], List[float]]:
        """
        Get actions for all agents
        
        Args:
            observations: list of observations for each agent
            valid_actions_list: list of valid actions for each agent
            deterministic: if True, select argmax actions
        
        Returns:
            actions: list of action indices
            log_probs: list of log probabilities
        """
        actions = []
        log_probs = []
        
        with torch.no_grad():
            for i in range(self.num_agents):
                obs = torch.FloatTensor(observations[i]).to(self.device)
                valid_actions = valid_actions_list[i] if i < len(valid_actions_list) else None
                
                action, log_prob = self.actor.get_action(
                    obs, valid_actions, deterministic
                )
                
                actions.append(action.item())
                log_probs.append(log_prob.item())
        
        return actions, log_probs
    
    def get_value(self, observations: List[np.ndarray]) -> float:
        """
        Get value estimate from centralized critic
        
        Args:
            observations: list of observations for all agents
        
        Returns:
            value: estimated value
        """
        global_obs = np.concatenate(observations)
        global_obs_tensor = torch.FloatTensor(global_obs).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            value = self.critic(global_obs_tensor)
        
        return value.item()
    
    def update(
        self,
        rollout_buffer,
    ) -> Dict[str, float]:
        """
        Update policy using PPO
        
        Args:
            rollout_buffer: buffer containing rollout data
        
        Returns:
            train_info: dictionary with training statistics
        """
        # Get data from buffer
        obs_batch = rollout_buffer['observations']
        actions_batch = rollout_buffer['actions']
        old_log_probs_batch = rollout_buffer['log_probs']
        advantages_batch = rollout_buffer['advantages']
        returns_batch = rollout_buffer['returns']
        valid_actions_batch = rollout_buffer['valid_actions']
        
        # Convert to tensors
        obs_tensor = torch.FloatTensor(np.array(obs_batch)).to(self.device)
        actions_tensor = torch.LongTensor(actions_batch).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(old_log_probs_batch).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages_batch).to(self.device)
        returns_tensor = torch.FloatTensor(returns_batch).to(self.device)
        
        # Normalize advantages
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (
            advantages_tensor.std() + 1e-8
        )
        
        # Statistics
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        num_updates = 0
        
        # PPO update epochs
        for _ in range(self.config.ppo_epochs):
            # Evaluate actions
            new_log_probs, entropy = self.actor.evaluate_actions(
                obs_tensor, actions_tensor, valid_actions_batch
            )
            
            # Compute ratio
            ratio = torch.exp(new_log_probs - old_log_probs_tensor)
            
            # Clipped surrogate loss
            surr1 = ratio * advantages_tensor
            surr2 = torch.clamp(
                ratio, 
                1.0 - self.config.clip_ratio, 
                1.0 + self.config.clip_ratio
            ) * advantages_tensor
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Entropy bonus
            entropy_loss = -entropy.mean()
            
            # Total actor loss
            actor_total_loss = (
                actor_loss + 
                self.config.entropy_coef * entropy_loss
            )
            
            # Update actor
            self.actor_optimizer.zero_grad()
            actor_total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(), 
                self.config.max_grad_norm
            )
            self.actor_optimizer.step()
            
            # Critic loss (using global observations)
            # Reconstruct global observations
            batch_size = len(obs_batch) // self.num_agents
            global_obs_list = []
            for b in range(batch_size):
                start_idx = b * self.num_agents
                end_idx = start_idx + self.num_agents
                global_obs = np.concatenate(obs_batch[start_idx:end_idx])
                global_obs_list.append(global_obs)
            
            global_obs_tensor = torch.FloatTensor(np.array(global_obs_list)).to(self.device)
            values = self.critic(global_obs_tensor).squeeze()
            
            # Expand returns for comparison
            returns_expanded = returns_tensor.view(batch_size, self.num_agents).mean(dim=1)
            
            # Ensure same shape
            if values.dim() > returns_expanded.dim():
                values = values.squeeze(-1)
            
            critic_loss = F.mse_loss(values, returns_expanded)
            
            # Update critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.critic.parameters(), 
                self.config.max_grad_norm
            )
            self.critic_optimizer.step()
            
            # Record statistics
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_entropy += entropy.mean().item()
            num_updates += 1
        
        train_info = {
            'actor_loss': total_actor_loss / num_updates,
            'critic_loss': total_critic_loss / num_updates,
            'entropy': total_entropy / num_updates,
        }
        
        return train_info
