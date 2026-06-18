"""
Enhanced Demo Script - Visualize Trained Policy
================================================
"""

import time
import numpy as np
import torch

from PPO_AEG import MAPPOActor
from pursuit_evasion import PursuitEvasionImproved
from Adaptive_Env_Generator import AdaptiveEnvironmentGenerator

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT = "checkpoints_mappo/mappo_aeg_ep3000.pt" 


def load_policy(actor, ckpt):
    """Load trained actor weights - PyTorch 2.6 compatible"""
    data = torch.load(ckpt, map_location=DEVICE, weights_only=False)
    actor.load_state_dict(data["actor"])
    print(f"✓ Loaded checkpoint: {ckpt}")
    print(f"  Episode: {data['episode']}")


def ask_mode():
    """Interactive mode selection"""
    print("\n" + "="*60)
    print("Choose Scenario:")
    print("="*60)
    print("1) Easy Chase (Close start, no obstacles)")
    print("2) Medium Chase (Far start, few obstacles)")
    print("3) Hard Chase (Very far, many obstacles)")
    print("4) Extreme Chase (Maximum distance + obstacles)")
    print("5) Maze Challenge (Dense obstacle field)")
    print("6) Random AEG Task")
    print("7) Custom Distance (You choose)")
    print("8) HUMAN vs AI (You control the evader!)")
    print("0) Quit")
    print("="*60)
    return input("Select mode [0-8]: ").strip()


def create_scenario(mode, aeg, arena_size=14.0):
    """Create scenario with varying difficulty"""
    
    if mode == "1":
        # Easy - Close start
        scenario = {
            'pursuer_positions': [
                [-2.0, 0.0, 3.0],
                [-2.0, 1.5, 3.0],
                [-2.0, -1.5, 3.0]
            ],
            'evader_position': [2.0, 0.0, 3.0],
            'obstacles': []
        }
        print("Easy Chase - Close start (4 units), no obstacles")
        
    elif mode == "2":
        # Medium - Far start with some obstacles
        scenario = {
            'pursuer_positions': [
                [-4.0, 0.0, 3.0],
                [-4.0, 2.5, 3.0],
                [-4.0, -2.5, 3.0]
            ],
            'evader_position': [4.0, 0.0, 3.0],
            'obstacles': [
                {'type': 'cylinder', 'position': [0.0, 2.5, 1.0], 'size': [0.15, 2.0], 'color': [0.5, 0.5, 0.5, 1.0]},
                {'type': 'cylinder', 'position': [0.0, -2.5, 1.0], 'size': [0.15, 2.0], 'color': [0.5, 0.5, 0.5, 1.0]},
                {'type': 'cylinder', 'position': [1.5, 0.0, 1.0], 'size': [0.15, 2.0], 'color': [0.5, 0.5, 0.5, 1.0]},
            ]
        }
        print("Medium Chase - Far start (8 units), 3 obstacles (GRAY)")
        
    elif mode == "3":
        # Hard - Very far with many obstacles
        scenario = {
            'pursuer_positions': [
                [-5.5, 0.0, 3.0],
                [-5.5, 3.0, 3.0],
                [-5.5, -3.0, 3.0]
            ],
            'evader_position': [5.5, 0.0, 3.0],
            'obstacles': [
                {'type': 'cylinder', 'position': [-3.0, 2.0, 1.2], 'size': [0.15, 2.4], 'color': [0.6, 0.4, 0.2, 1.0]},
                {'type': 'cylinder', 'position': [-3.0, -2.0, 1.2], 'size': [0.15, 2.4], 'color': [0.6, 0.4, 0.2, 1.0]},
                {'type': 'cylinder', 'position': [0.0, 3.5, 1.2], 'size': [0.15, 2.4], 'color': [0.6, 0.4, 0.2, 1.0]},
                {'type': 'cylinder', 'position': [0.0, -3.5, 1.2], 'size': [0.15, 2.4], 'color': [0.6, 0.4, 0.2, 1.0]},
                {'type': 'cylinder', 'position': [3.0, 1.5, 1.2], 'size': [0.15, 2.4], 'color': [0.6, 0.4, 0.2, 1.0]},
                {'type': 'cylinder', 'position': [3.0, -1.5, 1.2], 'size': [0.15, 2.4], 'color': [0.6, 0.4, 0.2, 1.0]},
            ]
        }
        print("Hard Chase - Very far (11 units), 6 obstacles (BROWN)")
        
    elif mode == "4":
        # Extreme - Maximum distance with obstacles
        scenario = {
            'pursuer_positions': [
                [-6.5, 0.0, 3.0],
                [-6.5, 3.5, 3.0],
                [-6.5, -3.5, 3.0]
            ],
            'evader_position': [6.5, 0.0, 3.0],
            'obstacles': [
                {'type': 'cylinder', 'position': [-4.5, 0.0, 1.5], 'size': [0.18, 3.0], 'color': [0.8, 0.2, 0.2, 1.0]},
                {'type': 'cylinder', 'position': [-2.5, 3.0, 1.5], 'size': [0.18, 3.0], 'color': [0.8, 0.2, 0.2, 1.0]},
                {'type': 'cylinder', 'position': [-2.5, -3.0, 1.5], 'size': [0.18, 3.0], 'color': [0.8, 0.2, 0.2, 1.0]},
                {'type': 'cylinder', 'position': [0.0, 4.5, 1.5], 'size': [0.18, 3.0], 'color': [0.8, 0.2, 0.2, 1.0]},
                {'type': 'cylinder', 'position': [0.0, -4.5, 1.5], 'size': [0.18, 3.0], 'color': [0.8, 0.2, 0.2, 1.0]},
                {'type': 'cylinder', 'position': [2.5, 2.5, 1.5], 'size': [0.18, 3.0], 'color': [0.8, 0.2, 0.2, 1.0]},
                {'type': 'cylinder', 'position': [2.5, -2.5, 1.5], 'size': [0.18, 3.0], 'color': [0.8, 0.2, 0.2, 1.0]},
                {'type': 'cylinder', 'position': [4.5, 0.0, 1.5], 'size': [0.18, 3.0], 'color': [0.8, 0.2, 0.2, 1.0]},
            ]
        }
        print(" Extreme Chase - Maximum distance (13 units), 8 TALL obstacles (RED)")
        
    elif mode == "5":
        # Maze - Dense obstacle field
        scenario = {
            'pursuer_positions': [
                [-5.0, 0.0, 3.0],
                [-5.0, 2.5, 3.0],
                [-5.0, -2.5, 3.0]
            ],
            'evader_position': [5.0, 0.0, 3.0],
            'obstacles': [
                # Left column - TALL
                {'type': 'cylinder', 'position': [-3.5, 3.5, 2.0], 'size': [0.15, 4.0], 'color': [0.2, 0.7, 0.2, 1.0]},
                {'type': 'cylinder', 'position': [-3.5, 0.0, 2.0], 'size': [0.15, 4.0], 'color': [0.2, 0.7, 0.2, 1.0]},
                {'type': 'cylinder', 'position': [-3.5, -3.5, 2.0], 'size': [0.15, 4.0], 'color': [0.2, 0.7, 0.2, 1.0]},
                # Middle-left column - TALL
                {'type': 'cylinder', 'position': [-1.5, 4.5, 2.0], 'size': [0.15, 4.0], 'color': [0.2, 0.7, 0.2, 1.0]},
                {'type': 'cylinder', 'position': [-1.5, 2.0, 2.0], 'size': [0.15, 4.0], 'color': [0.2, 0.7, 0.2, 1.0]},
                {'type': 'cylinder', 'position': [-1.5, -2.0, 2.0], 'size': [0.15, 4.0], 'color': [0.2, 0.7, 0.2, 1.0]},
                {'type': 'cylinder', 'position': [-1.5, -4.5, 2.0], 'size': [0.15, 4.0], 'color': [0.2, 0.7, 0.2, 1.0]},
                # Middle-right column - TALL
                {'type': 'cylinder', 'position': [1.5, 3.5, 2.0], 'size': [0.15, 4.0], 'color': [0.2, 0.7, 0.2, 1.0]},
                {'type': 'cylinder', 'position': [1.5, 0.0, 2.0], 'size': [0.15, 4.0], 'color': [0.2, 0.7, 0.2, 1.0]},
                {'type': 'cylinder', 'position': [1.5, -3.5, 2.0], 'size': [0.15, 4.0], 'color': [0.2, 0.7, 0.2, 1.0]},
                # Right column - TALL
                {'type': 'cylinder', 'position': [3.5, 2.5, 2.0], 'size': [0.15, 4.0], 'color': [0.2, 0.7, 0.2, 1.0]},
                {'type': 'cylinder', 'position': [3.5, -2.5, 2.0], 'size': [0.15, 4.0], 'color': [0.2, 0.7, 0.2, 1.0]},
            ]
        }
        print("Maze Challenge - Dense obstacle maze, 12 TALL obstacles (GREEN)")
        
    elif mode == "6":
        # Random from AEG
        scenario = aeg.sample_task()
        print(f"Random AEG Task - {len(scenario['obstacles'])} obstacles")
        
    elif mode == "7":
        # Custom distance
        try:
            dist = float(input("Enter distance between teams (2.0 - 12.0): "))
            dist = np.clip(dist, 2.0, 12.0)
            half_dist = dist / 2.0
            
            n_obs = int(input("Enter number of obstacles (0-10): "))
            n_obs = np.clip(n_obs, 0, 10)
            
            scenario = {
                'pursuer_positions': [
                    [-half_dist, 0.0, 3.0],
                    [-half_dist, 2.0, 3.0],
                    [-half_dist, -2.0, 3.0]
                ],
                'evader_position': [half_dist, 0.0, 3.0],
                'obstacles': []
            }
            
            # Add random obstacles
            for _ in range(n_obs):
                x = np.random.uniform(-arena_size/2 + 2, arena_size/2 - 2)
                y = np.random.uniform(-arena_size/2 + 2, arena_size/2 - 2)
                # Random height between 1.5 and 3.5
                height = np.random.uniform(1.5, 3.5)
                scenario['obstacles'].append({
                    'type': 'cylinder',
                    'position': [x, y, height/2],  # Center at half height
                    'size': [0.15, height],
                    'color': [np.random.uniform(0.3, 0.7), 
                             np.random.uniform(0.3, 0.7), 
                             np.random.uniform(0.3, 0.7), 1.0]
                })
            
            print(f"Custom - Distance: {dist:.1f} units, {n_obs} obstacles")
            
        except:
            print("Invalid input, using default easy scenario")
            return create_scenario("1", aeg, arena_size)
    
    elif mode == "8":
        # Human vs AI - Keyboard control (Medium difficulty by default)
        scenario = {
            'pursuer_positions': [
                [-4.5, 0.0, 3.0],
                [-4.5, 2.5, 3.0],
                [-4.5, -2.5, 3.0]
            ],
            'evader_position': [4.5, 0.0, 3.0],
            'obstacles': [
                {'type': 'cylinder', 'position': [0.0, 2.5, 1.0], 'size': [0.15, 2.0], 'color': [0.5, 0.5, 0.5, 1.0]},
                {'type': 'cylinder', 'position': [0.0, -2.5, 1.0], 'size': [0.15, 2.0], 'color': [0.5, 0.5, 0.5, 1.0]},
                {'type': 'cylinder', 'position': [1.5, 0.0, 1.0], 'size': [0.15, 2.0], 'color': [0.5, 0.5, 0.5, 1.0]},
            ]
        }
        print("Human vs AI - Navigate around 3 obstacles with arrow keys!")
        
        return scenario
        
    else:
        return None
    
    return scenario


def run_episode(env, actor, scenario):
    """Run a single episode with visualization"""
    obs = env.reset(scenario=scenario)
    done = False
    
    step = 0
    total_reward = 0
    captured = False
    min_dist_ever = 9999
    
    print("\n" + "-"*60)
    print("EPISODE START")
    print(f"Evader Speed: {env.evader_speed:.3f}")
    print("-"*60)
    
    while not done:
        # Get actions from policy
        with torch.no_grad():
            actions, _ = actor.act(obs, deterministic=True, device=DEVICE)
        
        # Step environment
        obs, rewards, done, info = env.step(actions)
        
        step += 1
        total_reward += np.mean(rewards)
        captured = info["captured"]
        min_dist = info["min_distance"]
        min_dist_ever = min(min_dist_ever, min_dist)
        
        # Print progress every 30 steps
        if step % 30 == 0:
            drone_coll = info.get("drone_collisions", 0)
            obs_coll = info.get("obstacle_collisions", 0)
            print(f"Step {step:3d} | Dist: {min_dist:.2f} | "
                  f"Reward: {total_reward:6.1f} | "
                  f"D-Coll: {drone_coll} | O-Coll: {obs_coll}")
        
        if captured:
            print(f"\n CAPTURED at step {step}!")
            break
        
        # Slow down for better viewing (60 FPS)
        time.sleep(1/60.0)
    
    # Episode summary
    print("-"*60)
    print("EPISODE SUMMARY")
    print("-"*60)
    if captured:
        print(f"Result: ✓ CAPTURED (Success!)")
    else:
        print(f"Result: ✗ FAILED (Evader escaped)")
    print(f"Evader Speed: {env.evader_speed:.3f}")
    print(f"Steps: {step}/{env.max_steps}")
    print(f"Total Reward: {total_reward:.1f}")
    print(f"Min Distance Achieved: {min_dist_ever:.2f}")
    print(f"Capture Radius: {env.capture_radius:.2f}")
    print(f"Drone-Drone Collisions: {info.get('drone_collisions', 0)}")
    print(f"Drone-Obstacle Collisions: {info.get('obstacle_collisions', 0)}")
    print("-"*60)
    
    return {
        'captured': captured,
        'steps': step,
        'reward': total_reward,
        'min_dist': min_dist_ever
    }


def run_keyboard_episode(env, actor, scenario):
    """Run episode where human controls the evader with keyboard"""
    import pybullet as p
    
    obs = env.reset(scenario=scenario)
    done = False
    
    step = 0
    total_reward = 0
    captured = False
    min_dist_ever = 9999
    
    # Evader control parameters
    evader_velocity = [0.0, 0.0, 0.0]
    max_speed = 5.0  # Increased human control speed to match fast evader
    
    print("\n" + "-"*60)
    print("🎮 HUMAN vs AI - YOU CONTROL THE RED EVADER!")
    print("-"*60)
    print(f"Your Max Speed: {max_speed:.2f}")
    print(f"AI Pursuer Speed: ~3.0")
    print("CONTROLS:")
    print("  Arrow Keys: Move X/Y (Up/Down/Left/Right)")
    print("  ESC: Give up")
    print("-"*60)
    print("Survive for 600 steps without being caught!")
    print("-"*60 + "\n")
    
    while not done and step < env.max_steps:
        # Get pursuer actions from AI policy
        with torch.no_grad():
            actions, _ = actor.act(obs, deterministic=True, device=DEVICE)
        
        # Get keyboard input for evader
        keys = p.getKeyboardEvents()
        
        # Update velocity based on keyboard - ARROW KEYS ONLY
        evader_velocity = [0.0, 0.0, 0.0]
        
        # Arrow keys for horizontal movement
        if p.B3G_UP_ARROW in keys:
            evader_velocity[1] += max_speed  # Forward
        if p.B3G_DOWN_ARROW in keys:
            evader_velocity[1] -= max_speed  # Backward
        if p.B3G_LEFT_ARROW in keys:
            evader_velocity[0] -= max_speed  # Left
        if p.B3G_RIGHT_ARROW in keys:
            evader_velocity[0] += max_speed  # Right
        
        # ESC to give up
        if p.B3G_ESCAPE in keys:
            print("\n You gave up!")
            break
        
        # Apply human control to evader
        ev_pos, ev_orn = p.getBasePositionAndOrientation(env.evader_id)
        ev_pos = np.array(ev_pos)
        
        dt = 1/30.0
        new_ev_pos = ev_pos + np.array(evader_velocity) * dt
        
        # Clip to arena bounds
        new_ev_pos = np.clip(new_ev_pos,
                            [-env.arena_size/2+1, -env.arena_size/2+1, 1],
                            [env.arena_size/2-1, env.arena_size/2-1, 5])
        
        # Update evader position (bypass AI control)
        p.resetBasePositionAndOrientation(env.evader_id, new_ev_pos, ev_orn)
        
        # Move pursuers with AI
        for i, pid in enumerate(env.pursuer_ids):
            pos, orn = p.getBasePositionAndOrientation(pid)
            pos = np.array(pos)
            new = pos + actions[i] * dt
            new = np.clip(new, [-env.arena_size/2+1, -env.arena_size/2+1, 1],
                                [env.arena_size/2-1, env.arena_size/2-1, 5])
            p.resetBasePositionAndOrientation(pid, new, orn)
        
        # Step simulation
        p.stepSimulation()
        
        # Calculate distances and check capture
        purs_positions = []
        for pid in env.pursuer_ids:
            pos, _ = p.getBasePositionAndOrientation(pid)
            purs_positions.append(np.array(pos))
        
        dists = [np.linalg.norm(new_ev_pos - p_pos) for p_pos in purs_positions]
        min_dist = min(dists)
        min_dist_ever = min(min_dist_ever, min_dist)
        
        captured = min_dist < env.capture_radius
        
        step += 1
        
        # Update observation for next iteration
        obs = env._get_obs()
        
        # Print progress
        if step % 30 == 0:
            print(f"Step {step:3d}/600 | Min Dist: {min_dist:.2f} | "
                  f"Your Position: ({new_ev_pos[0]:.1f}, {new_ev_pos[1]:.1f}, {new_ev_pos[2]:.1f})")
        
        if captured:
            print(f"\n CAUGHT at step {step}! The AI pursuers got you!")
            break
        
        # Slow down for playability
        time.sleep(1/30.0)  # 30 FPS for human control
    
    # Episode summary
    print("-"*60)
    print("HUMAN vs AI SUMMARY")
    print("-"*60)
    if captured:
        print(f"Result: CAUGHT by AI pursuers!")
        print(f"You survived {step} steps")
    elif step >= env.max_steps:
        print(f"Result: YOU WIN! You escaped!")
        print(f"You survived all {env.max_steps} steps!")
    else:
        print(f"Result: Gave up after {step} steps")
    
    print(f"Closest distance achieved: {min_dist_ever:.2f}")
    print(f"Capture radius was: {env.capture_radius:.2f}")
    print("-"*60)
    
    return {
        'captured': captured,
        'steps': step,
        'reward': 0,  # No reward for human play
        'min_dist': min_dist_ever
    }


def run_demo():
    """Main demo loop"""
    
    print("\n" + "="*60)
    print("Multi-UAV Pursuit-Evasion Demo")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"PyTorch version: {torch.__version__}")
    print("\nCONFIGURATION:")
    print("  • Evader Speed: 2.0 (Fast mode)")
    print("  • Blue Drones = AI Pursuers (your trained team)")
    print("  • Red Drone = Evader (trying to escape)")
    print("\nFEATURES:")
    print("  • Multiple difficulty scenarios")
    print("  • Human vs AI mode with arrow key controls")
    print("  • Tall color-coded obstacles")
    
    # Create environment with GUI
    print("\nInitializing environment...")
    env = PursuitEvasionImproved(
        gui=True,
        n_pursuers=3,
        arena_size=14.0,
        max_steps=600,
        evader_speed=2.0  
    )
    
    env.set_camera()
    
    # Load trained policy
    print("\nLoading trained policy...")
    obs_dim = env.obs_dim
    act_dim = env.action_space.shape[-1]
    
    actor = MAPPOActor(obs_dim, act_dim).to(DEVICE)
    
    try:
        load_policy(actor, CHECKPOINT)
    except FileNotFoundError:
        print(f"\n WARNING: Checkpoint not found: {CHECKPOINT}")
        print("Available checkpoints:")
        import os
        if os.path.exists("checkpoints_mappo"):
            ckpts = [f for f in os.listdir("checkpoints_mappo") if f.endswith('.pt')]
            for ckpt in sorted(ckpts):
                print(f"  - checkpoints_mappo/{ckpt}")
            if ckpts:
                use_ckpt = input("\nEnter checkpoint to use: ").strip()
                if not use_ckpt.startswith("checkpoints_mappo/"):
                    use_ckpt = f"checkpoints_mappo/{use_ckpt}"
                load_policy(actor, use_ckpt)
            else:
                print("\n No checkpoints found. Train first with: python PPO_AEG.py")
                env.close()
                return
        else:
            print("\n Checkpoints directory not found. Train first with: python PPO_AEG.py")
            env.close()
            return
    except Exception as e:
        print(f"\n Error loading checkpoint: {e}")
        print("\nIf you see a PyTorch version error, the checkpoint was saved with a different PyTorch version.")
        env.close()
        return
    
    actor.eval()
    
    # Create AEG
    aeg = AdaptiveEnvironmentGenerator(n_pursuers=3, arena_size=14.0)
    
    print("\n Setup complete! Ready for demo!")
    print("\nCONTROLS:")
    print("  - Select scenarios from menu")
    print("  - Press Ctrl+C during episode to skip")
    print("  - Close PyBullet window to exit")
    
    # Demo loop
    episode_count = 0
    stats_history = []
    
    print(f"\n Evader speed set to: {env.evader_speed:.3f}")
    
    while True:
        mode = ask_mode()
        
        if mode == "0":
            print("\n Exiting demo. Goodbye!")
            break
        
        scenario = create_scenario(mode, aeg, env.arena_size)
        
        if scenario is None:
            print("Invalid mode selected.")
            continue
        
        episode_count += 1
        print(f"\n>>> Episode {episode_count}")
        
        try:
            # Check if keyboard control mode
            if mode == "8":
                stats = run_keyboard_episode(env, actor, scenario)
            else:
                stats = run_episode(env, actor, scenario)
            
            stats_history.append(stats)
            
            # Show cumulative stats
            if len(stats_history) >= 3:
                print("\n" + "="*60)
                print("CUMULATIVE STATISTICS")
                print("="*60)
                cap_rate = 100 * np.mean([s['captured'] for s in stats_history])
                avg_steps = np.mean([s['steps'] for s in stats_history])
                avg_reward = np.mean([s['reward'] for s in stats_history])
                avg_min_dist = np.mean([s['min_dist'] for s in stats_history])
                
                print(f"Episodes: {len(stats_history)}")
                print(f"Capture Rate: {cap_rate:.1f}%")
                print(f"Avg Steps: {avg_steps:.1f}")
                print(f"Avg Reward: {avg_reward:.1f}")
                print(f"Avg Min Distance: {avg_min_dist:.2f}")
                print("="*60)
                
        except KeyboardInterrupt:
            print("\n\n Episode interrupted by user.")
            continue
        except Exception as e:
            print(f"\n Error during episode: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        print("\nPress Enter to continue (or Ctrl+C to exit)...")
        try:
            input()
        except KeyboardInterrupt:
            print("\n Exiting demo. Goodbye!")
            break
    
    env.close()
    
    # Final summary
    if stats_history:
        print("\n" + "="*60)
        print("FINAL DEMO STATISTICS")
        print("="*60)
        print(f"Total Episodes: {len(stats_history)}")
        print(f"Captures: {sum(s['captured'] for s in stats_history)}")
        print(f"Capture Rate: {100*np.mean([s['captured'] for s in stats_history]):.1f}%")
        print(f"Avg Reward: {np.mean([s['reward'] for s in stats_history]):.1f}")
        print(f"Avg Min Distance: {np.mean([s['min_dist'] for s in stats_history]):.2f}")
        print("="*60)
    
    print("Demo terminated.")


if __name__ == "__main__":
    try:
        run_demo()
    except KeyboardInterrupt:
        print("\n\n Demo interrupted. Goodbye!")
    except Exception as e:
        print(f"\n Fatal error: {e}")
        import traceback
        traceback.print_exc()