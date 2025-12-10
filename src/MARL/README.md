# MAPPO for Chiplet Placement

Multi-Agent Proximal Policy Optimization (MAPPO) implementation for chiplet placement optimization.

## Features

- **Multi-Agent Learning**: Each chiplet is controlled by an independent agent
- **Centralized Training, Decentralized Execution (CTDE)**: Agents share a centralized critic during training but execute independently
- **Adjacency Constraints**: Enforces minimum overlap length for connected chiplets
- **Discrete Action Space**: Grid-based placement with rotation support

## Architecture

```
MARL/
├── config.py              # Configuration parameters
├── train_mappo.py         # Main training script
├── test_mappo.py          # Unit tests
├── agents/
│   ├── policy_net.py      # Actor and Critic networks
│   └── marl_solver.py     # MAPPO training algorithm
├── envs/
│   └── chiplet_env.py     # Multi-agent environment
└── utils/
    └── visualize.py       # Visualization utilities
```

## Quick Start

### 1. Run Tests

```bash
cd src/MARL
python test_mappo.py
```

### 2. Train on Test Cases

```bash
python train_mappo.py
```

This will train on all test cases in `baseline/ICCAD23/test_input/`.

## Configuration

Key parameters in `MAPPOConfig`:

```python
# Environment
grid_resolution = 50          # Grid discretization
min_overlap_length = 0.5      # Minimum overlap for connections

# MAPPO Hyperparameters
lr_actor = 3e-4              # Actor learning rate
lr_critic = 3e-4             # Critic learning rate
gamma = 0.99                 # Discount factor
clip_ratio = 0.2             # PPO clip parameter
ppo_epochs = 4               # PPO update epochs

# Rewards
overlap_penalty = 1000.0     # Penalty for overlapping chiplets
adjacency_penalty = 500.0    # Penalty for violated connections
adjacency_reward = 100.0     # Reward for satisfied connections
area_penalty = 1.0           # Penalty for bounding box area
gap_penalty = 50.0           # Penalty for gaps between chiplets
```

## Input Format

JSON files in `baseline/ICCAD23/test_input/`:

```json
{
  "chiplets": [
    {"name": "A", "width": 10.0, "height": 10.0},
    {"name": "B", "width": 10.0, "height": 10.0}
  ],
  "connections": [
    ["A", "B"]
  ]
}
```

## Output

- **Visualizations**: Saved to `output/MARL/`
- **Console Logs**: Training progress, rewards, constraint satisfaction

## Constraints

1. **No Overlap**: Chiplets cannot overlap
2. **Adjacency**: Connected chiplets must share an edge with length ≥ `min_overlap_length`
3. **Boundary**: All chiplets must fit within the estimated boundary

## Algorithm Details

### MAPPO (Multi-Agent PPO)

- **Actors**: Each agent has a policy network (parameter-shared)
- **Critic**: Centralized value network using global observations
- **Update**: PPO with clipped objective and GAE

### State Representation

For each agent:
- Grid occupancy mask (flattened)
- Own size information
- Placement status
- Connection information with other agents

### Action Space

For each agent: `(grid_x, grid_y, rotation)`
- `grid_x, grid_y`: Grid coordinates (0 to grid_resolution-1)
- `rotation`: 0 (no rotation) or 1 (90° rotation)

### Reward Function

```
reward = adjacency_reward * satisfied_connections
       - adjacency_penalty * violated_connections
       - area_penalty * bounding_box_area
       - gap_penalty * total_gaps
       - overlap_penalty * (if overlapping)
```

## Troubleshooting

### GPU Issues

If GPU is detected but not usable (e.g., RTX 5090 with sm_120), the code automatically falls back to CPU.

### Memory Issues

Reduce `grid_resolution` or `num_agents` if running out of memory.

### Slow Training

- Use GPU if available
- Reduce `ppo_epochs`
- Reduce `num_episodes`

## Performance Tips

1. **Grid Resolution**: Start with 30-50 for faster training
2. **Episode Count**: 1000-2000 episodes usually sufficient for small cases
3. **Reward Tuning**: Adjust penalty/reward weights based on problem requirements

## Example Results

After training on `3core.json`:
```
Best reward: 150.23
Placed chiplets: 3/3
Adjacency constraints: 2/2 (100%)
Bounding box area: 425.6
```

## Citation

Based on the MAPPO algorithm from:
- Yu et al., "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games", NeurIPS 2022
