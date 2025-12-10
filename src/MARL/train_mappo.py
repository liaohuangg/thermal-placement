"""
Main script to train MAPPO on chiplet placement tasks
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from MARL.config import MAPPOConfig
from MARL.envs.chiplet_env import MultiAgentChipletEnv
from MARL.agents.marl_solver import MAPPOSolver
from MARL.utils.visualize import visualize_placement


def load_test_case(json_path: str):
    """Load test case from JSON file"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    chiplets = data['chiplets']
    connections = data['connections']
    
    # Extract sizes
    chiplet_sizes = []
    chiplet_names = []
    for chiplet in chiplets:
        width = chiplet['width']
        height = chiplet['height']
        chiplet_sizes.append((width, height))
        chiplet_names.append(chiplet['name'])
    
    # Convert connections to indices
    name_to_idx = {name: i for i, name in enumerate(chiplet_names)}
    connection_indices = []
    for conn in connections:
        i = name_to_idx[conn[0]]
        j = name_to_idx[conn[1]]
        connection_indices.append((i, j))
    
    return chiplet_sizes, connection_indices, chiplet_names


def main():
    # Configuration
    config = MAPPOConfig()
    
    # Device setup
    device = torch.device("cpu")
    if config.use_gpu and torch.cuda.is_available():
        try:
            test_tensor = torch.zeros(1).cuda()
            _ = test_tensor + 1
            device = torch.device("cuda")
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            del test_tensor
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"GPU detected but not usable, falling back to CPU")
            print(f"  Error: {str(e)[:150]}")
            device = torch.device("cpu")
    else:
        print("Using CPU")
    
    config.device = str(device)
    
    # Load test case
    test_input_dir = Path(__file__).parent.parent.parent / "baseline" / "ICCAD23" / "test_input"
    
    # Test with different cases
    test_files = [
        "3core.json",
        "5core.json",
        "6core.json",
        "8core.json",
        "10core.json",
    ]
    
    for test_file in test_files:
        test_path = test_input_dir / test_file
        
        if not test_path.exists():
            print(f"Skipping {test_file} (not found)")
            continue
        
        print("=" * 80)
        print(f"Training on: {test_file}")
        print("=" * 80)
        
        # Load test case
        chiplet_sizes, connections, chiplet_names = load_test_case(str(test_path))
        
        print(f"Number of chiplets: {len(chiplet_sizes)}")
        print(f"Number of connections: {len(connections)}")
        print(f"Chiplet sizes: {chiplet_sizes}")
        print(f"Connections: {connections}")
        print()
        
        # Create environment
        env = MultiAgentChipletEnv(
            chiplet_sizes=chiplet_sizes,
            connections=connections,
            config=config
        )
        
        # Create solver
        solver = MAPPOSolver(
            env=env,
            config=config,
            device=device
        )
        
        # Train
        best_placements = solver.train(num_episodes=config.num_episodes)
        
        # Evaluate
        print("\nEvaluating trained policy...")
        eval_results = solver.evaluate(num_episodes=20)
        print(f"Evaluation results:")
        print(f"  Mean reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
        print(f"  Mean episode length: {eval_results['mean_length']:.1f}")
        print(f"  Success rate: {eval_results['success_rate']:.2%}")
        
        # Visualize best solution
        if best_placements is not None:
            output_dir = Path(__file__).parent.parent.parent / "output" / "MARL"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_path = output_dir / f"{test_file.replace('.json', '_mappo.png')}"
            
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
        
        # Print statistics
        print("\nFinal Statistics:")
        print(f"  Best reward: {solver.best_reward:.2f}")
        
        # Check constraints
        placed_count = sum(1 for p in best_placements if p is not None)
        print(f"  Placed chiplets: {placed_count}/{len(chiplet_sizes)}")
        
        # Check adjacency constraints
        satisfied = 0
        total = len(connections)
        for i, j in connections:
            if (best_placements[i] is not None and 
                best_placements[j] is not None):
                if env._check_adjacency_constraint(i, j):
                    satisfied += 1
        
        if total > 0:
            print(f"  Adjacency constraints: {satisfied}/{total} ({100*satisfied/total:.1f}%)")
        
        # Compute bounding box
        if placed_count > 0:
            # Temporarily set env placements to best
            old_placements = env.placements
            old_placed = env.placed_agents
            
            env.placements = best_placements
            env.placed_agents = {i for i, p in enumerate(best_placements) if p is not None}
            
            bbox_area = env._compute_bounding_box_area()
            print(f"  Bounding box area: {bbox_area:.2f}")
            
            env.placements = old_placements
            env.placed_agents = old_placed
        
        print("\n")


if __name__ == "__main__":
    main()
