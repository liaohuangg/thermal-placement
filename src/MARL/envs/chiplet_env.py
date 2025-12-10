"""
Multi-Agent Chiplet Placement Environment for MAPPO
"""

import numpy as np
import math
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class ChipletPlacement:
    """Single chiplet placement information"""
    agent_id: int
    grid_x: int  # Bottom-right corner x in grid
    grid_y: int  # Bottom-right corner y in grid
    rotated: bool


class MultiAgentChipletEnv:
    """
    Multi-Agent Environment for chiplet placement
    Each chiplet is controlled by an independent agent
    """
    
    def __init__(
        self,
        chiplet_sizes: List[Tuple[float, float]],  # [(width, height), ...]
        connections: List[Tuple[int, int]],  # [(chiplet_i, chiplet_j), ...]
        config,
    ):
        """
        Args:
            chiplet_sizes: List of (width, height) for each chiplet
            connections: List of (i, j) indices indicating connections
            config: MAPPOConfig object
        """
        self.chiplet_sizes = chiplet_sizes
        self.num_agents = len(chiplet_sizes)
        self.connections = set()
        
        # Store connections as sorted tuples
        for i, j in connections:
            self.connections.add((min(i, j), max(i, j)))
        
        self.config = config
        self.grid_resolution = config.grid_resolution
        self.min_overlap_length = config.min_overlap_length
        
        # Estimate bounding box
        total_area = sum(w * h for w, h in chiplet_sizes)
        self.estimated_bound = math.sqrt(total_area) * 2.5
        self.grid_size_per_unit = self.estimated_bound / self.grid_resolution
        
        # State
        self.grid_mask = None
        self.placements = None
        self.placed_agents = None
        self.remaining_agents = None
        
        self.reset()
    
    def reset(self) -> List[np.ndarray]:
        """Reset environment and return initial observations for all agents"""
        self.grid_mask = np.zeros(
            (self.grid_resolution, self.grid_resolution), 
            dtype=np.float32
        )
        self.placements = [None] * self.num_agents
        self.placed_agents = set()
        self.remaining_agents = set(range(self.num_agents))
        
        # Get initial observations for all agents
        observations = [self._get_agent_observation(i) for i in range(self.num_agents)]
        return observations
    
    def step(self, actions: List[Tuple[int, int, int]]) -> Tuple[
        List[np.ndarray],  # observations
        List[float],  # rewards
        bool,  # done
        Dict  # info
    ]:
        """
        Execute joint actions from all agents
        
        Args:
            actions: List of (grid_x, grid_y, rotation) for each agent
        
        Returns:
            observations: List of observations for each agent
            rewards: List of rewards for each agent
            done: Whether episode is done
            info: Additional information
        """
        step_rewards = [0.0] * self.num_agents
        conflicts = []
        
        # Collect actions from remaining agents
        pending_placements = []
        for agent_id in self.remaining_agents:
            grid_x, grid_y, rotation = actions[agent_id]
            rotated = (rotation == 1)
            
            # Check validity
            if not self._is_valid_action(agent_id, grid_x, grid_y, rotated):
                step_rewards[agent_id] = -self.config.overlap_penalty
                continue
            
            pending_placements.append((agent_id, grid_x, grid_y, rotated))
        
        # Check for conflicts among pending placements
        valid_placements = []
        conflicting_agents = set()
        
        for i, (agent_i, x_i, y_i, rot_i) in enumerate(pending_placements):
            has_conflict = False
            region_i = self._get_chiplet_region(agent_i, x_i, y_i, rot_i)
            
            for j, (agent_j, x_j, y_j, rot_j) in enumerate(pending_placements):
                if i >= j:
                    continue
                
                region_j = self._get_chiplet_region(agent_j, x_j, y_j, rot_j)
                
                if self._regions_overlap(region_i, region_j):
                    has_conflict = True
                    conflicting_agents.add(agent_i)
                    conflicting_agents.add(agent_j)
                    conflicts.append((agent_i, agent_j))
            
            if not has_conflict:
                valid_placements.append((agent_i, x_i, y_i, rot_i))
        
        # Apply valid placements
        for agent_id, grid_x, grid_y, rotated in valid_placements:
            placement = ChipletPlacement(agent_id, grid_x, grid_y, rotated)
            self.placements[agent_id] = placement
            self.placed_agents.add(agent_id)
            self.remaining_agents.discard(agent_id)
            
            # Update grid mask
            min_x, max_x, min_y, max_y = self._get_chiplet_region(
                agent_id, grid_x, grid_y, rotated
            )
            self.grid_mask[min_y:max_y, min_x:max_x] = 1.0
        
        # Penalize conflicts
        for agent_id in conflicting_agents:
            step_rewards[agent_id] = -self.config.overlap_penalty / 2
        
        # Compute rewards for placed agents
        if len(self.placed_agents) > 0:
            global_reward = self._compute_global_reward()
            
            # Distribute reward to all placed agents
            for agent_id in self.placed_agents:
                if agent_id not in conflicting_agents:
                    step_rewards[agent_id] = global_reward / len(self.placed_agents)
        
        # Check if done
        done = len(self.remaining_agents) == 0
        
        # Get next observations
        observations = [self._get_agent_observation(i) for i in range(self.num_agents)]
        
        info = {
            'num_placed': len(self.placed_agents),
            'num_remaining': len(self.remaining_agents),
            'conflicts': conflicts,
        }
        
        return observations, step_rewards, done, info
    
    def _get_agent_observation(self, agent_id: int) -> np.ndarray:
        """Get observation for a specific agent"""
        # Observation includes:
        # 1. Grid mask (flattened)
        # 2. Agent's own size info
        # 3. Placement status
        # 4. Connection information
        
        grid_mask_flat = self.grid_mask.flatten()
        
        # Own size
        w, h = self.chiplet_sizes[agent_id]
        grid_w = max(1, int(math.ceil(w / self.grid_size_per_unit)))
        grid_h = max(1, int(math.ceil(h / self.grid_size_per_unit)))
        
        own_info = np.array([
            grid_w / self.grid_resolution,
            grid_h / self.grid_resolution,
            1.0 if agent_id in self.placed_agents else 0.0,
        ], dtype=np.float32)
        
        # Connection info (connected agents positions if placed)
        connection_info = []
        for other_id in range(self.num_agents):
            if other_id == agent_id:
                continue
            
            is_connected = (min(agent_id, other_id), max(agent_id, other_id)) in self.connections
            
            if self.placements[other_id] is not None:
                p = self.placements[other_id]
                connection_info.extend([
                    is_connected * 1.0,
                    p.grid_x / self.grid_resolution,
                    p.grid_y / self.grid_resolution,
                    1.0 if p.rotated else 0.0,
                ])
            else:
                connection_info.extend([is_connected * 1.0, 0.0, 0.0, 0.0])
        
        connection_info = np.array(connection_info, dtype=np.float32)
        
        obs = np.concatenate([grid_mask_flat, own_info, connection_info])
        return obs
    
    def _get_chiplet_grid_size(self, agent_id: int, rotated: bool) -> Tuple[int, int]:
        """Get chiplet size in grid units"""
        w, h = self.chiplet_sizes[agent_id]
        if rotated:
            w, h = h, w
        
        grid_w = max(1, int(math.ceil(w / self.grid_size_per_unit)))
        grid_h = max(1, int(math.ceil(h / self.grid_size_per_unit)))
        return grid_w, grid_h
    
    def _get_chiplet_region(
        self, agent_id: int, grid_x: int, grid_y: int, rotated: bool
    ) -> Tuple[int, int, int, int]:
        """Get chiplet occupied region (min_x, max_x, min_y, max_y)"""
        grid_w, grid_h = self._get_chiplet_grid_size(agent_id, rotated)
        
        min_x = grid_x - grid_w + 1
        max_x = grid_x + 1
        min_y = grid_y - grid_h + 1
        max_y = grid_y + 1
        
        return min_x, max_x, min_y, max_y
    
    def _regions_overlap(
        self, region1: Tuple[int, int, int, int], region2: Tuple[int, int, int, int]
    ) -> bool:
        """Check if two regions overlap"""
        min_x1, max_x1, min_y1, max_y1 = region1
        min_x2, max_x2, min_y2, max_y2 = region2
        
        return not (max_x1 <= min_x2 or max_x2 <= min_x1 or 
                   max_y1 <= min_y2 or max_y2 <= min_y1)
    
    def _is_valid_action(
        self, agent_id: int, grid_x: int, grid_y: int, rotated: bool
    ) -> bool:
        """Check if action is valid"""
        # Check if agent already placed
        if agent_id in self.placed_agents:
            return False
        
        # Check grid bounds
        if grid_x < 0 or grid_x >= self.grid_resolution:
            return False
        if grid_y < 0 or grid_y >= self.grid_resolution:
            return False
        
        # Check region bounds and overlap with placed chiplets
        min_x, max_x, min_y, max_y = self._get_chiplet_region(
            agent_id, grid_x, grid_y, rotated
        )
        
        if min_x < 0 or max_x > self.grid_resolution:
            return False
        if min_y < 0 or max_y > self.grid_resolution:
            return False
        
        # Check overlap with grid mask
        region_mask = self.grid_mask[min_y:max_y, min_x:max_x]
        if np.any(region_mask > 0.5):
            return False
        
        return True
    
    def _compute_global_reward(self) -> float:
        """Compute global reward based on current placement"""
        reward = 0.0
        
        # 1. Check adjacency constraints
        adjacency_violations = 0
        adjacency_satisfied = 0
        
        for i, j in self.connections:
            if self.placements[i] is None or self.placements[j] is None:
                continue
            
            if self._check_adjacency_constraint(i, j):
                adjacency_satisfied += 1
                reward += self.config.adjacency_reward
            else:
                adjacency_violations += 1
                reward -= self.config.adjacency_penalty
        
        # 2. Penalize bounding box area
        bbox_area = self._compute_bounding_box_area()
        reward -= self.config.area_penalty * bbox_area
        
        # 3. Penalize gaps
        if len(self.placed_agents) > 1:
            total_gap = self._compute_total_gap()
            reward -= self.config.gap_penalty * total_gap
        
        return reward
    
    def _check_adjacency_constraint(self, agent_i: int, agent_j: int) -> bool:
        """Check if two connected chiplets satisfy adjacency constraint"""
        p_i = self.placements[agent_i]
        p_j = self.placements[agent_j]
        
        if p_i is None or p_j is None:
            return False
        
        min_x_i, max_x_i, min_y_i, max_y_i = self._get_chiplet_region(
            agent_i, p_i.grid_x, p_i.grid_y, p_i.rotated
        )
        min_x_j, max_x_j, min_y_j, max_y_j = self._get_chiplet_region(
            agent_j, p_j.grid_x, p_j.grid_y, p_j.rotated
        )
        
        # Check horizontal adjacency
        if max_x_i == min_x_j or max_x_j == min_x_i:
            y_overlap = min(max_y_i, max_y_j) - max(min_y_i, min_y_j)
            if y_overlap > 0:
                overlap_length = y_overlap * self.grid_size_per_unit
                if overlap_length >= self.min_overlap_length:
                    return True
        
        # Check vertical adjacency
        if max_y_i == min_y_j or max_y_j == min_y_i:
            x_overlap = min(max_x_i, max_x_j) - max(min_x_i, min_x_j)
            if x_overlap > 0:
                overlap_length = x_overlap * self.grid_size_per_unit
                if overlap_length >= self.min_overlap_length:
                    return True
        
        return False
    
    def _compute_bounding_box_area(self) -> float:
        """Compute bounding box area of all placed chiplets"""
        if len(self.placed_agents) == 0:
            return 0.0
        
        min_x = self.grid_resolution
        min_y = self.grid_resolution
        max_x = 0
        max_y = 0
        
        for agent_id in self.placed_agents:
            p = self.placements[agent_id]
            region_min_x, region_max_x, region_min_y, region_max_y = \
                self._get_chiplet_region(agent_id, p.grid_x, p.grid_y, p.rotated)
            
            min_x = min(min_x, region_min_x)
            max_x = max(max_x, region_max_x)
            min_y = min(min_y, region_min_y)
            max_y = max(max_y, region_max_y)
        
        width = (max_x - min_x) * self.grid_size_per_unit
        height = (max_y - min_y) * self.grid_size_per_unit
        
        return width * height
    
    def _compute_total_gap(self) -> float:
        """Compute total gap between placed chiplets"""
        total_gap = 0.0
        placed_list = list(self.placed_agents)
        
        for i, agent_i in enumerate(placed_list):
            p_i = self.placements[agent_i]
            region_i = self._get_chiplet_region(
                agent_i, p_i.grid_x, p_i.grid_y, p_i.rotated
            )
            min_x_i, max_x_i, min_y_i, max_y_i = region_i
            
            for agent_j in placed_list[i+1:]:
                p_j = self.placements[agent_j]
                region_j = self._get_chiplet_region(
                    agent_j, p_j.grid_x, p_j.grid_y, p_j.rotated
                )
                min_x_j, max_x_j, min_y_j, max_y_j = region_j
                
                # Compute gap
                if max_x_i <= min_x_j:
                    gap_x = (min_x_j - max_x_i) * self.grid_size_per_unit
                elif max_x_j <= min_x_i:
                    gap_x = (min_x_i - max_x_j) * self.grid_size_per_unit
                else:
                    gap_x = 0.0
                
                if max_y_i <= min_y_j:
                    gap_y = (min_y_j - max_y_i) * self.grid_size_per_unit
                elif max_y_j <= min_y_i:
                    gap_y = (min_y_i - max_y_j) * self.grid_size_per_unit
                else:
                    gap_y = 0.0
                
                if gap_x > 0 and gap_y > 0:
                    gap = min(gap_x, gap_y)
                else:
                    gap = max(gap_x, gap_y)
                
                total_gap += gap
        
        return total_gap
    
    def get_valid_actions(self, agent_id: int) -> List[Tuple[int, int, int]]:
        """Get valid actions for a specific agent"""
        if agent_id in self.placed_agents:
            return []
        
        valid_actions = []
        
        for rotation in [0, 1]:
            rotated = (rotation == 1)
            grid_w, grid_h = self._get_chiplet_grid_size(agent_id, rotated)
            
            for grid_y in range(grid_h - 1, self.grid_resolution):
                for grid_x in range(grid_w - 1, self.grid_resolution):
                    if self._is_valid_action(agent_id, grid_x, grid_y, rotated):
                        valid_actions.append((grid_x, grid_y, rotation))
        
        return valid_actions
    
    def get_observation_space_size(self) -> int:
        """Get the size of observation space"""
        grid_size = self.grid_resolution * self.grid_resolution
        own_info = 3
        connection_info = (self.num_agents - 1) * 4
        return grid_size + own_info + connection_info
    
    def get_action_space_size(self) -> int:
        """Get the size of action space"""
        return self.grid_resolution * self.grid_resolution * 2
