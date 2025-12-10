"""
Quick example: Train MAPPO on a single test case
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import json
from MARL.config import MAPPOConfig
from MARL.envs.chiplet_env import MultiAgentChipletEnv
from MARL.agents.marl_solver import MAPPOSolver
from MARL.utils.visualize import visualize_placement


def main():
    print("=" * 80)
    print("MAPPO Chiplet Placement - Quick Example")
    print("=" * 80)
    print()
    
    # Load 5-core test case
    test_file = Path(__file__).parent.parent.parent / "baseline" / "ICCAD23" / "test_input" / "5core.json"
    
    with open(test_file, 'r') as f:
        data = json.load(f)
    
    chiplets = data['chiplets']
    connections = data['connections']
    
    # Extract info
    chiplet_sizes = [(c['width'], c['height']) for c in chiplets]
    chiplet_names = [c['name'] for c in chiplets]
    name_to_idx = {name: i for i, name in enumerate(chiplet_names)}
    connection_indices = [(name_to_idx[c[0]], name_to_idx[c[1]]) for c in connections]
    
    print(f"Test case: {test_file.name}")
    print(f"Chiplets: {chiplet_names}")
    print(f"Connections: {connections}")
    print()
    
    # Configuration
    config = MAPPOConfig()
    config.num_episodes = 500  # Quick training
    config.log_interval = 100
    config.grid_resolution = 40  # Moderate resolution for speed
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print()
    
    # Create environment
    env = MultiAgentChipletEnv(
        chiplet_sizes=chiplet_sizes,
        connections=connection_indices,
        config=config
    )
    
    # Create solver
    solver = MAPPOSolver(
        env=env,
        config=config,
        device=device
    )
    
    # Train
    print("Training...")
    best_placements = solver.train(num_episodes=config.num_episodes)
    print()
    
    # Results
    print("=" * 80)
    print("Results")
    print("=" * 80)
    print(f"Best reward: {solver.best_reward:.2f}")
    
    placed_count = sum(1 for p in best_placements if p is not None)
    print(f"Placed chiplets: {placed_count}/{len(chiplet_sizes)}")
    
    # Check adjacency
    satisfied = 0
    for i, j in connection_indices:
        if (best_placements[i] is not None and best_placements[j] is not None):
            if env._check_adjacency_constraint(i, j):
                satisfied += 1
    
    print(f"Adjacency constraints: {satisfied}/{len(connection_indices)} " 
          f"({100*satisfied/len(connection_indices) if connection_indices else 0:.1f}%)")
    
    # Bounding box
    if placed_count > 0:
        old_placements = env.placements
        old_placed = env.placed_agents
        env.placements = best_placements
        env.placed_agents = {i for i, p in enumerate(best_placements) if p is not None}
        bbox_area = env._compute_bounding_box_area()
        env.placements = old_placements
        env.placed_agents = old_placed
        print(f"Bounding box area: {bbox_area:.2f}")
    
    # Visualize
    output_dir = Path(__file__).parent.parent.parent / "output" / "MARL"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "example_5core.png"
    
    try:
        visualize_placement(
            env=env,
            placements=best_placements,
            chiplet_names=chiplet_names,
            save_path=str(output_path)
        )
        print(f"\nVisualization saved to: {output_path}")
    except Exception as e:
        print(f"Visualization failed: {e}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
