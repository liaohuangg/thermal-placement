"""
MAPPO Configuration for Chiplet Placement
"""

class MAPPOConfig:
    def __init__(self):
        # Environment settings
        self.grid_resolution = 50  # Grid size for discretization
        self.min_overlap_length = 0.5  # Minimum overlap length for connections
        
        # MAPPO hyperparameters
        self.lr_actor = 3e-4
        self.lr_critic = 3e-4
        self.gamma = 0.99  # Discount factor
        self.gae_lambda = 0.95  # GAE lambda
        self.clip_ratio = 0.2  # PPO clip ratio
        self.value_coef = 0.5  # Value loss coefficient
        self.entropy_coef = 0.01  # Entropy bonus coefficient
        self.max_grad_norm = 0.5  # Gradient clipping
        
        # Training settings
        self.num_episodes = 2000
        self.max_steps_per_episode = 200
        self.ppo_epochs = 4  # Number of PPO update epochs per batch
        self.mini_batch_size = 64
        
        # Reward weights
        self.overlap_penalty = 1000.0
        self.adjacency_penalty = 500.0
        self.adjacency_reward = 100.0
        self.area_penalty = 1.0
        self.gap_penalty = 50.0
        self.boundary_penalty = 100.0
        
        # Device
        self.use_gpu = True
        self.device = "cuda"  # Will auto-switch to cpu if needed
        
        # Logging
        self.log_interval = 50
        self.save_interval = 500
