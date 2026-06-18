""" 
Adaptive Environment Generator for Multi-UAV Pursuit-Evasion
=============================================================
"""

import numpy as np
from collections import deque


class AdaptiveEnvironmentGenerator:
    """
    Generates training scenarios adaptively based on agent performance
    """
    
    def __init__(
        self,
        arena_size=14.0,
        n_pursuers=3,
        min_success_rate=0.3,  
        max_success_rate=0.85, 
        local_prob=0.6,         
        position_noise=0.8,    
        max_archive_size=100,
        eval_episodes=5     
    ):
        self.arena_size = arena_size
        self.n_pursuers = n_pursuers
        self.min_success_rate = min_success_rate
        self.max_success_rate = max_success_rate
        self.local_prob = local_prob
        self.position_noise = position_noise
        self.max_archive_size = max_archive_size
        self.eval_episodes = eval_episodes
        
        # Active archive of tasks with appropriate difficulty
        self.active_archive = []
        
        # Performance tracking: {task_hash: [success_list]}
        self.task_performance = {}
        
        # Initialize with simple seed tasks
        self._initialize_seed_tasks()
    
    def _initialize_seed_tasks(self):
        """
        Create initial VERY EASY tasks to bootstrap learning
        CHANGED: Much closer starting positions, NO obstacles initially
        """
        seed_tasks = [
            # VERY close start, no obstacles (EASIEST)
            {
                'pursuer_positions': [
                    [-1.0, 0.0, 3.0],
                    [-1.0, 0.8, 3.0],
                    [-1.0, -0.8, 3.0]
                ],
                'evader_position': [1.0, 0.0, 3.0],  # Only 2.0 units away!
                'obstacles': []
            },
            # Slightly further, no obstacles
            {
                'pursuer_positions': [
                    [-1.5, 0.0, 3.0],
                    [-1.5, 1.0, 3.0],
                    [-1.5, -1.0, 3.0]
                ],
                'evader_position': [1.5, 0.0, 3.0],  # 3.0 units away
                'obstacles': []
            },
            # Medium distance, no obstacles
            {
                'pursuer_positions': [
                    [-2.0, 0.0, 3.0],
                    [-2.0, 1.2, 3.0],
                    [-2.0, -1.2, 3.0]
                ],
                'evader_position': [2.0, 0.0, 3.0],  # 4.0 units away
                'obstacles': []
            },
            # Close start, flanking positions, no obstacles
            {
                'pursuer_positions': [
                    [-1.2, 0.0, 3.0],
                    [-1.2, 1.5, 3.0],
                    [-1.2, -1.5, 3.0]
                ],
                'evader_position': [1.2, 0.0, 3.0],
                'obstacles': []
            },
            # Medium start, wide formation, no obstacles
            {
                'pursuer_positions': [
                    [-1.8, 0.0, 3.0],
                    [-1.8, 2.0, 3.0],
                    [-1.8, -2.0, 3.0]
                ],
                'evader_position': [1.8, 0.0, 3.0],
                'obstacles': []
            },
        ]
        
        self.active_archive = seed_tasks.copy()
        print(f"[AEG] Initialized with {len(seed_tasks)} EASY seed tasks (NO obstacles)")
    
    def _task_to_hash(self, task):
        """Convert task dict to hashable string for tracking"""
        # Use rounded positions to avoid float precision issues
        pursuer_str = '_'.join([
            f"{p[0]:.2f},{p[1]:.2f},{p[2]:.2f}" 
            for p in task['pursuer_positions']
        ])
        evader_str = f"{task['evader_position'][0]:.2f},{task['evader_position'][1]:.2f},{task['evader_position'][2]:.2f}"
        n_obs = len(task['obstacles'])
        return f"p{pursuer_str}_e{evader_str}_o{n_obs}"
    
    def sample_task(self):
        """
        Sample a task using mixed strategy:
        - 60% Local Expansion (expand from active archive)
        - 40% Global Exploration (random new scenario)
        """
        if np.random.random() < self.local_prob and len(self.active_archive) > 0:
            # Local Expansion - build on successful tasks
            return self._local_expansion()
        else:
            # Global Exploration - try new things
            return self._global_exploration()
    
    def _local_expansion(self):
        """
        Expand from a seed task in active archive by perturbing positions
        Keep obstacles fixed, vary initial positions
        """
        # Sample a seed task from archive
        seed_task = self.active_archive[np.random.randint(len(self.active_archive))]
        
        # Create expanded task by adding noise to positions
        expanded_task = {
            'pursuer_positions': [],
            'evader_position': None,
            'obstacles': seed_task['obstacles'].copy()  # Keep obstacles same
        }
        
        # Perturb pursuer positions
        for pos in seed_task['pursuer_positions']:
            noise = np.random.uniform(-self.position_noise, self.position_noise, 3)
            noise[2] = 0  # Keep z fixed at 3.0
            new_pos = np.array(pos) + noise
            # Clip to arena bounds (pursuers on left side)
            new_pos[0] = np.clip(new_pos[0], -self.arena_size/2 + 1, -0.5)  # Keep on left
            new_pos[1] = np.clip(new_pos[1], -self.arena_size/2 + 1, self.arena_size/2 - 1)
            new_pos[2] = 3.0
            expanded_task['pursuer_positions'].append(new_pos.tolist())
        
        # Perturb evader position
        noise = np.random.uniform(-self.position_noise, self.position_noise, 3)
        noise[2] = 0
        new_evader_pos = np.array(seed_task['evader_position']) + noise
        new_evader_pos[0] = np.clip(new_evader_pos[0], 0.5, self.arena_size/2 - 1)  # Keep on right
        new_evader_pos[1] = np.clip(new_evader_pos[1], -self.arena_size/2 + 1, self.arena_size/2 - 1)
        new_evader_pos[2] = 3.0
        expanded_task['evader_position'] = new_evader_pos.tolist()
        
        return expanded_task
    
    def _global_exploration(self):
        """
        Generate completely random scenario
        CHANGED: Start with fewer obstacles (0-3 instead of 0-5)
        """
        task = {
            'pursuer_positions': [],
            'evader_position': None,
            'obstacles': []
        }
        
        # Random pursuer positions (left side)
        for _ in range(self.n_pursuers):
            x = np.random.uniform(-self.arena_size/2 + 1, -0.5)  # Keep on left
            y = np.random.uniform(-self.arena_size/2 + 1, self.arena_size/2 - 1)
            task['pursuer_positions'].append([x, y, 3.0])
        
        # Random evader position (right side)
        x = np.random.uniform(0.5, self.arena_size/2 - 1)  # Keep on right
        y = np.random.uniform(-self.arena_size/2 + 1, self.arena_size/2 - 1)
        task['evader_position'] = [x, y, 3.0]
        
        # CHANGED: Fewer obstacles initially (0-3 instead of 0-5)
        # This makes global exploration more likely to succeed
        n_obstacles = np.random.randint(0, 4)  # 0, 1, 2, or 3
        for _ in range(n_obstacles):
            obs_x = np.random.uniform(-self.arena_size/2 + 2, self.arena_size/2 - 2)
            obs_y = np.random.uniform(-self.arena_size/2 + 2, self.arena_size/2 - 2)
            
            # Make sure obstacle isn't too close to start positions
            # (avoid blocking capture immediately)
            task['obstacles'].append({
                'type': 'cylinder',
                'position': [obs_x, obs_y, 0.6],
                'size': [0.1, 1.2],
                'color': [0.5, 0.5, 0.5, 1.0]
            })
        
        return task
    
    def update_archive(self, task, success):
        """
        Update archive based on task performance
        
        Args:
            task: Task dictionary
            success: Boolean, whether the task was solved successfully
        """
        task_hash = self._task_to_hash(task)
        
        # Track performance
        if task_hash not in self.task_performance:
            self.task_performance[task_hash] = deque(maxlen=self.eval_episodes)
        
        self.task_performance[task_hash].append(float(success))
        
        # Calculate success rate when we have enough data
        if len(self.task_performance[task_hash]) >= self.eval_episodes:
            success_rate = np.mean(self.task_performance[task_hash])
            
            # Add to archive if difficulty is appropriate
            if self.min_success_rate <= success_rate <= self.max_success_rate:
                # Check if task already in archive
                task_in_archive = any(
                    self._task_to_hash(t) == task_hash 
                    for t in self.active_archive
                )
                
                if not task_in_archive:
                    self.active_archive.append(task)
                    print(f"[AEG] ✓ Added task to archive (success rate: {success_rate:.2f})")
                    
                    # Limit archive size
                    if len(self.active_archive) > self.max_archive_size:
                        # Remove oldest task
                        self.active_archive.pop(0)
                        print(f"[AEG] ↻ Removed oldest task (archive full)")
            
            # Remove from archive if too easy or too hard
            elif success_rate < self.min_success_rate or success_rate > self.max_success_rate:
                task_in_archive_idx = None
                for idx, t in enumerate(self.active_archive):
                    if self._task_to_hash(t) == task_hash:
                        task_in_archive_idx = idx
                        break
                
                if task_in_archive_idx is not None:
                    self.active_archive.pop(task_in_archive_idx)
                    reason = "too hard" if success_rate < self.min_success_rate else "too easy"
                    print(f"[AEG] ✗ Removed task from archive ({reason}: {success_rate:.2f})")
    
    def get_archive_stats(self):
        """Get statistics about the current archive"""
        if len(self.active_archive) == 0:
            return {
                'archive_size': 0,
                'tracked_tasks': len(self.task_performance),
                'avg_obstacles': 0,
                'avg_distance': 0
            }
        
        # Calculate average starting distance
        avg_distance = 0
        for task in self.active_archive:
            evader_pos = np.array(task['evader_position'])
            pursuer_center = np.mean(task['pursuer_positions'], axis=0)
            avg_distance += np.linalg.norm(evader_pos - pursuer_center)
        avg_distance /= len(self.active_archive)
        
        return {
            'archive_size': len(self.active_archive),
            'tracked_tasks': len(self.task_performance),
            'avg_obstacles': np.mean([len(t['obstacles']) for t in self.active_archive]),
            'avg_distance': avg_distance
        }


if __name__ == "__main__":
    # Test the AEG
    print("Testing IMPROVED Adaptive Environment Generator...")
    
    aeg = AdaptiveEnvironmentGenerator(n_pursuers=3)
    
    print(f"\nInitial archive size: {len(aeg.active_archive)}")
    print(f"Settings:")
    print(f"  min_success_rate: {aeg.min_success_rate}")
    print(f"  max_success_rate: {aeg.max_success_rate}")
    print(f"  local_prob: {aeg.local_prob}")
    print(f"  eval_episodes: {aeg.eval_episodes}")
    
    # Sample some tasks
    print("\nSampling 10 tasks:")
    for i in range(10):
        task = aeg.sample_task()
        
        # Calculate distance
        evader_pos = np.array(task['evader_position'])
        pursuer_center = np.mean(task['pursuer_positions'], axis=0)
        dist = np.linalg.norm(evader_pos - pursuer_center)
        
        print(f"\nTask {i+1}:")
        print(f"  Distance: {dist:.2f}")
        print(f"  Obstacles: {len(task['obstacles'])}")
        
        # Simulate success based on distance (closer = more likely to succeed)
        # This simulates the improved environment
        success_prob = max(0.1, 1.0 - (dist / 6.0))  # Closer = higher success
        success = np.random.random() < success_prob
        
        aeg.update_archive(task, success)
        print(f"  Result: {'SUCCESS' if success else 'FAILURE'}")
    
    stats = aeg.get_archive_stats()
    print(f"\nFinal stats:")
    print(f"  Archive size: {stats['archive_size']}")
    print(f"  Tracked tasks: {stats['tracked_tasks']}")
    print(f"  Avg obstacles: {stats['avg_obstacles']:.1f}")
    print(f"  Avg distance: {stats['avg_distance']:.2f}")