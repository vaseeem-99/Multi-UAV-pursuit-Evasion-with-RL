"""
Human vs AI Demo - Control the Evader!
======================================

YOU control the RED evader drone with arrow keys.
AI controls the BLUE pursuer drones.
"""

import time
import numpy as np
import torch
import pybullet as p

from PPO_AEG import MAPPOActor
from pursuit_evasion import PursuitEvasionImproved

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT = "checkpoints_mappo/mappo_aeg_ep3000.pt"

# Configuration 1 (Default - typical setup):
UP_AXIS = 0      # X+ (forward)
UP_SIGN = 1      # positive
DOWN_AXIS = 0    # X- (backward)
DOWN_SIGN = -1   # negative
LEFT_AXIS = 1    # Y+ (left)
LEFT_SIGN = 1    # positive
RIGHT_AXIS = 1   # Y- (right)
RIGHT_SIGN = -1  # negative

SHOW_DEBUG = False  # Set to True to see velocity values
# ============================================================


def load_policy(actor, ckpt):
    """Load trained actor weights"""
    data = torch.load(ckpt, map_location=DEVICE, weights_only=False)
    actor.load_state_dict(data["actor"])
    print(f" Loaded checkpoint from episode {data['episode']}")


def get_keyboard_constants():
    """Get keyboard constants that work with current PyBullet version"""
    keys = {}
    
    try:
        keys['UP'] = p.B3G_UP_ARROW
        keys['DOWN'] = p.B3G_DOWN_ARROW
        keys['LEFT'] = p.B3G_LEFT_ARROW
        keys['RIGHT'] = p.B3G_RIGHT_ARROW
    except AttributeError:
        keys['UP'] = 65297
        keys['DOWN'] = 65298
        keys['LEFT'] = 65295
        keys['RIGHT'] = 65296
    
    return keys


def run_human_vs_ai():
    """Human controls evader, AI controls pursuers"""
    
    print("\n" + "="*60)
    print("HUMAN vs AI - Pursuit-Evasion Challenge")
    print("="*60)
    print("YOU (RED) vs AI PURSUERS (BLUE)")
    print("\nCONTROLS:")
    print("  ↑ ↓ ← →  : Move evader")
    print("  +        : Increase your speed")
    print("  -        : Decrease your speed")
    print("  Q or ESC : Quit")
    print("="*60)
    
    # Create environment
    print("\nInitializing...")
    env = PursuitEvasionImproved(
        gui=True,
        n_pursuers=3,
        arena_size=14.0,
        max_steps=1000
    )
    env.set_camera()
    
    # Get keyboard constants
    KEY = get_keyboard_constants()
    
    # Load AI policy
    obs_dim = env.obs_dim
    act_dim = env.action_space.shape[-1]
    actor = MAPPOActor(obs_dim, act_dim).to(DEVICE)
    load_policy(actor, CHECKPOINT)
    actor.eval()
    
    # Starting scenario
    scenario = {
        'pursuer_positions': [
            [-5.0, 0.0, 3.0],
            [-5.0, 2.5, 3.0],
            [-5.0, -2.5, 3.0]
        ],
        'evader_position': [5.0, 0.0, 3.0],
        'obstacles': [
            {'type': 'cylinder', 'position': [0.0, 3.0, 1.0], 'size': [0.15, 2.0], 'color': [0.5, 0.5, 0.5, 1.0]},
            {'type': 'cylinder', 'position': [0.0, -3.0, 1.0], 'size': [0.15, 2.0], 'color': [0.5, 0.5, 0.5, 1.0]},
            {'type': 'cylinder', 'position': [2.5, 0.0, 1.0], 'size': [0.15, 2.0], 'color': [0.5, 0.5, 0.5, 1.0]},
        ]
    }
    
    print("Ready! Game starting...\n")
    time.sleep(1)
    
    # Game statistics
    total_games = 0
    wins = 0
    losses = 0
    
    # Main game loop - keeps restarting
    game_running = True
    
    while game_running:
        # Reset for new game
        obs = env.reset(scenario=scenario)
        step = 0
        captured = False
        min_dist_ever = 9999
        
        # Human control settings
        human_speed = 3.0 
        min_speed = 1.0
        max_speed = 10.0
        speed_increment = 0.5
        
        total_games += 1
        
        print("\n" + "="*60)
        print(f"GAME {total_games} START!")
        print(f"Score: Wins={wins} | Losses={losses}")
        print(f"Your speed: {human_speed:.1f} | AI pursuer speed: ~3.0")
        if SHOW_DEBUG:
            print("DEBUG MODE: ON (will show velocity)")
        print("="*60)
        
        try:
            while step < env.max_steps and not captured:
                # Get keyboard input
                keys = p.getKeyboardEvents()
                
                # Speed controls
                if ord('+') in keys or ord('=') in keys:
                    human_speed = min(human_speed + speed_increment, max_speed)
                    print(f"Speed UP: {human_speed:.1f}")
                
                if ord('-') in keys or ord('_') in keys:
                    human_speed = max(human_speed - speed_increment, min_speed)
                    print(f"Speed DOWN: {human_speed:.1f}")
                
                # Quit controls (ESC=27, Q=113)
                if 27 in keys or ord('q') in keys or ord('Q') in keys:
                    print("\n Quitting game...")
                    game_running = False
                    break
                
                # Movement controls (arrow keys) - CONFIGURABLE
                evader_velocity = [0.0, 0.0, 0.0]
                
                keys_pressed = []
                if KEY['UP'] in keys:
                    evader_velocity[UP_AXIS] += UP_SIGN * human_speed
                    keys_pressed.append("UP")
                if KEY['DOWN'] in keys:
                    evader_velocity[DOWN_AXIS] += DOWN_SIGN * human_speed
                    keys_pressed.append("DOWN")
                if KEY['LEFT'] in keys:
                    evader_velocity[LEFT_AXIS] += LEFT_SIGN * human_speed
                    keys_pressed.append("LEFT")
                if KEY['RIGHT'] in keys:
                    evader_velocity[RIGHT_AXIS] += RIGHT_SIGN * human_speed
                    keys_pressed.append("RIGHT")
                
                # Debug output
                if SHOW_DEBUG and keys_pressed:
                    print(f"Keys: {keys_pressed} | Velocity: [{evader_velocity[0]:.1f}, {evader_velocity[1]:.1f}, {evader_velocity[2]:.1f}]")
                
                # Get AI pursuer actions
                with torch.no_grad():
                    actions, _ = actor.act(obs, deterministic=True, device=DEVICE)
                
                # Update evader position (human control)
                ev_pos, ev_orn = p.getBasePositionAndOrientation(env.evader_id)
                ev_pos = np.array(ev_pos)
                
                dt = 1/30.0
                new_ev_pos = ev_pos + np.array(evader_velocity) * dt
                
                # Clip to arena bounds
                new_ev_pos = np.clip(new_ev_pos,
                                    [-env.arena_size/2+1, -env.arena_size/2+1, 1],
                                    [env.arena_size/2-1, env.arena_size/2-1, 5])
                
                p.resetBasePositionAndOrientation(env.evader_id, new_ev_pos, ev_orn)
                
                # Update pursuer positions (AI control)
                for i, pid in enumerate(env.pursuer_ids):
                    pos, orn = p.getBasePositionAndOrientation(pid)
                    pos = np.array(pos)
                    new_pos = pos + actions[i] * dt
                    new_pos = np.clip(new_pos, 
                                    [-env.arena_size/2+1, -env.arena_size/2+1, 1],
                                    [env.arena_size/2-1, env.arena_size/2-1, 5])
                    p.resetBasePositionAndOrientation(pid, new_pos, orn)
                
                # Step simulation
                p.stepSimulation()
                
                # Check distances
                purs_positions = []
                for pid in env.pursuer_ids:
                    pos, _ = p.getBasePositionAndOrientation(pid)
                    purs_positions.append(np.array(pos))
                
                dists = [np.linalg.norm(new_ev_pos - p_pos) for p_pos in purs_positions]
                min_dist = min(dists)
                min_dist_ever = min(min_dist_ever, min_dist)
                
                # Check capture
                captured = min_dist < env.capture_radius
                
                step += 1
                
                # Update observations for AI
                obs = env._get_obs()
                
                # Print status every 50 steps
                if step % 50 == 0:
                    print(f"Step {step:4d}/1000 | Distance: {min_dist:.2f} | Speed: {human_speed:.1f} | Pos: ({new_ev_pos[0]:.1f}, {new_ev_pos[1]:.1f})")
                
                if captured:
                    print(f"\n CAUGHT at step {step}!")
                    losses += 1
                    break
                
                # Control frame rate
                time.sleep(1/30.0)
            
            # Check if player won
            if step >= env.max_steps and not captured:
                print(f"\n YOU WIN! Escaped for {env.max_steps} steps!")
                wins += 1
        
        except KeyboardInterrupt:
            print("\n\n Game interrupted by user")
            game_running = False
        
        # Show round summary
        if game_running:
            print("\n" + "-"*60)
            print("ROUND OVER")
            print("-"*60)
            
            if captured:
                print(f"Result: CAUGHT by AI pursuers!")
                print(f"You survived: {step} steps")
            else:
                print(f"Result: YOU ESCAPED!")
                print(f"You survived all {env.max_steps} steps!")
            
            print(f"Closest distance: {min_dist_ever:.2f}")
            print(f"Final speed: {human_speed:.1f}")
            print("-"*60)
            print(f"\nTotal Score: Wins={wins} | Losses={losses}")
            print("Restarting in 3 seconds...")
            print("(Press Q to quit)")
            print("-"*60)
            
            # Wait 3 seconds before restarting
            for i in range(30):
                keys = p.getKeyboardEvents()
                if ord('q') in keys or ord('Q') in keys or 27 in keys:
                    print("\n Quitting game...")
                    game_running = False
                    break
                time.sleep(0.1)
    
    # Final statistics
    print("\n" + "="*60)
    print("FINAL STATISTICS")
    print("="*60)
    print(f"Total Games: {total_games}")
    print(f"Wins: {wins}")
    print(f"Losses: {losses}")
    if total_games > 0:
        win_rate = 100 * wins / total_games
        print(f"Win Rate: {win_rate:.1f}%")
    print("="*60)
    
    env.close()
    print("\nThanks for playing!")


if __name__ == "__main__":
    try:
        run_human_vs_ai()
    except KeyboardInterrupt:
        print("\n\n Game interrupted. Goodbye!")
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()