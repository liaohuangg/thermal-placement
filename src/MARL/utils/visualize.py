"""
Visualization utilities for chiplet placement
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from typing import List, Optional


def visualize_placement(
    env,
    placements: List,
    chiplet_names: List[str],
    save_path: Optional[str] = None
):
    """
    Visualize chiplet placement
    
    Args:
        env: MultiAgentChipletEnv instance
        placements: List of ChipletPlacement objects
        chiplet_names: List of chiplet names
        save_path: Path to save figure (optional)
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Colors for different chiplets
    colors = plt.cm.tab20(np.linspace(0, 1, len(chiplet_names)))
    
    min_x_global = float('inf')
    min_y_global = float('inf')
    max_x_global = float('-inf')
    max_y_global = float('-inf')
    
    # Draw chiplets
    for agent_id, placement in enumerate(placements):
        if placement is None:
            continue
        
        # Get region
        min_x, max_x, min_y, max_y = env._get_chiplet_region(
            agent_id, placement.grid_x, placement.grid_y, placement.rotated
        )
        
        # Convert to real coordinates
        min_x_real = min_x * env.grid_size_per_unit
        min_y_real = min_y * env.grid_size_per_unit
        width_real = (max_x - min_x) * env.grid_size_per_unit
        height_real = (max_y - min_y) * env.grid_size_per_unit
        
        # Update global bounds
        min_x_global = min(min_x_global, min_x_real)
        min_y_global = min(min_y_global, min_y_real)
        max_x_global = max(max_x_global, min_x_real + width_real)
        max_y_global = max(max_y_global, min_y_real + height_real)
        
        # Draw rectangle
        rect = patches.Rectangle(
            (min_x_real, min_y_real),
            width_real,
            height_real,
            linewidth=2,
            edgecolor='black',
            facecolor=colors[agent_id],
            alpha=0.6
        )
        ax.add_patch(rect)
        
        # Add label
        center_x = min_x_real + width_real / 2
        center_y = min_y_real + height_real / 2
        
        label = chiplet_names[agent_id]
        if placement.rotated:
            label += "\n(R)"
        
        ax.text(
            center_x, center_y, label,
            ha='center', va='center',
            fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
    
    # Draw connections
    for i, j in env.connections:
        if placements[i] is None or placements[j] is None:
            continue
        
        # Get centers
        region_i = env._get_chiplet_region(
            i, placements[i].grid_x, placements[i].grid_y, placements[i].rotated
        )
        region_j = env._get_chiplet_region(
            j, placements[j].grid_x, placements[j].grid_y, placements[j].rotated
        )
        
        center_i_x = (region_i[0] + region_i[1]) / 2 * env.grid_size_per_unit
        center_i_y = (region_i[2] + region_i[3]) / 2 * env.grid_size_per_unit
        center_j_x = (region_j[0] + region_j[1]) / 2 * env.grid_size_per_unit
        center_j_y = (region_j[2] + region_j[3]) / 2 * env.grid_size_per_unit
        
        # Check if adjacency constraint is satisfied
        is_satisfied = env._check_adjacency_constraint(i, j)
        line_color = 'green' if is_satisfied else 'red'
        line_style = '-' if is_satisfied else '--'
        line_width = 2 if is_satisfied else 1
        
        ax.plot(
            [center_i_x, center_j_x],
            [center_i_y, center_j_y],
            color=line_color,
            linestyle=line_style,
            linewidth=line_width,
            alpha=0.7,
            marker='o',
            markersize=4
        )
    
    # Set axis limits with padding
    if min_x_global != float('inf'):
        padding = 5
        ax.set_xlim(min_x_global - padding, max_x_global + padding)
        ax.set_ylim(min_y_global - padding, max_y_global + padding)
    
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    
    # Title
    num_placed = sum(1 for p in placements if p is not None)
    num_total = len(placements)
    
    satisfied = sum(
        1 for i, j in env.connections
        if placements[i] is not None and placements[j] is not None
        and env._check_adjacency_constraint(i, j)
    )
    total_conn = len(env.connections)
    
    title = f"MAPPO Chiplet Placement\n"
    title += f"Placed: {num_placed}/{num_total}, "
    title += f"Adjacency: {satisfied}/{total_conn}"
    
    if num_placed > 0:
        # Compute area
        old_placements = env.placements
        old_placed = env.placed_agents
        env.placements = placements
        env.placed_agents = {i for i, p in enumerate(placements) if p is not None}
        area = env._compute_bounding_box_area()
        env.placements = old_placements
        env.placed_agents = old_placed
        
        title += f", Area: {area:.2f}"
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Legend
    legend_elements = [
        patches.Patch(facecolor='green', alpha=0.7, label='Satisfied connection'),
        patches.Patch(facecolor='red', alpha=0.7, label='Violated connection')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()
