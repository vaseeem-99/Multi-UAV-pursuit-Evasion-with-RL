"""
Quick Test: Environment
=================================

This script tests the improved environment with random actions
to verify it's catchable (unlike the original 0% capture rate).

"""

import numpy as np
import matplotlib.pyplot as plt
from pursuit_evasion import PursuitEvasionImproved


def test_improved_env(n_episodes=50):
    """Test improved environment with random actions"""
    print(f"\n{'='*60}")
    print(f"Testing IMPROVED Environment")
    print(f"{'='*60}")
    print("Using RANDOM actions to test if evader is catchable...")
    print()
    
    env = PursuitEvasionImproved(gui=False, n_pursuers=3)
    
    results = {
        'min_distances': [],
        'episode_lengths': [],
        'captures': [],
        'rewards': []
    }
    
    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        steps = 0
        ep_reward = 0
        
        while not done and steps < 600:
            # Random actions
            actions = np.random.uniform(-1, 1, (3, 3))
            obs, reward, done, info = env.step(actions)
            ep_reward += np.mean(reward)
            steps += 1
        
        results['min_distances'].append(info['min_distance'])
        results['episode_lengths'].append(steps)
        results['captures'].append(1 if info['captured'] else 0)
        results['rewards'].append(ep_reward)
        
        if (ep + 1) % 10 == 0:
            cap_rate = np.mean(results['captures']) * 100
            avg_dist = np.mean(results['min_distances'])
            avg_reward = np.mean(results['rewards'])
            print(f"[{ep+1}/{n_episodes}] Capture: {cap_rate:.1f}% | "
                  f"Avg Min Dist: {avg_dist:.2f} | Avg Reward: {avg_reward:.1f}")
    
    env.close()
    return results


def plot_results(results):
    """Plot test results"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Minimum Distance Distribution
    axes[0].hist(results['min_distances'], bins=20, alpha=0.7, 
                 color='green', edgecolor='black')
    axes[0].axvline(2.0, color='red', linestyle='--', linewidth=2, 
                    label='Capture Radius (2.0)')
    mean_dist = np.mean(results['min_distances'])
    axes[0].axvline(mean_dist, color='blue', 
                    linestyle='--', linewidth=2,
                    label=f'Mean: {mean_dist:.2f}')
    axes[0].set_xlabel('Minimum Distance Achieved')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Minimum Distance Distribution\n(Random Policy)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. Capture Rate
    cap_rate = np.mean(results['captures']) * 100
    fail_rate = 100 - cap_rate
    
    bars = axes[1].bar(['Failed', 'Captured'],
                       [fail_rate, cap_rate],
                       color=['red', 'green'], alpha=0.7, 
                       edgecolor='black', linewidth=2)
    axes[1].set_ylabel('Percentage (%)')
    axes[1].set_title(f'Capture Rate: {cap_rate:.1f}%\n(Random Policy)')
    axes[1].set_ylim([0, 105])
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # 3. Episode Length Distribution
    mean_length = np.mean(results['episode_lengths'])
    axes[2].hist(results['episode_lengths'], bins=20, alpha=0.7,
                 color='blue', edgecolor='black')
    axes[2].axvline(mean_length, color='red', 
                    linestyle='--', linewidth=2,
                    label=f'Mean: {mean_length:.1f}')
    axes[2].set_xlabel('Episode Length (steps)')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Episode Length Distribution\n(Random Policy)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('environment_test.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved test results to environment_test.png")
    
    # Print summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY (Improved Environment)")
    print(f"{'='*60}")
    print(f"Environment Settings:")
    print(f"  - Evader Speed: 0.05 (was 0.10)")
    print(f"  - Capture Radius: 2.0 (was 1.2)")
    print(f"  - Movement Multiplier: 0.25 (was 0.6)")
    print(f"\nResults with RANDOM actions:")
    print(f"  - Capture Rate: {cap_rate:.1f}%")
    print(f"  - Avg Min Distance: {mean_dist:.2f}")
    print(f"  - Avg Episode Length: {mean_length:.1f} steps")
    
    avg_reward = np.mean(results['rewards'])
    print(f"  - Avg Reward: {avg_reward:.1f}")
    print(f"{'='*60}\n")
    
    # Interpretation
    if cap_rate > 5:
        print(f"✓ SUCCESS! Environment is catchable ({cap_rate:.1f}% with random actions)")
        print(f"  With trained MAPPO, expect 40-80% capture rate!")
        print(f"  You can proceed to training with PPO_AEG.py\n")
    elif cap_rate > 0:
        print(f" MARGINAL: Some captures ({cap_rate:.1f}%) but may need easier settings")
        print(f"  Consider reducing evader speed to 0.03 or capture radius to 2.5\n")
    else:
        print(f" PROBLEM: Still 0% capture rate. Evader is STILL too strong!")
        print(f"  MUST reduce evader speed to 0.03 and increase capture radius to 2.5")
        print(f"  Edit pursuit_evasion.py and change these values in __init__\n")


if __name__ == "__main__":
    print("="*60)
    print("ENVIRONMENT TEST (Improved Settings)")
    print("="*60)
    print("\nThis test uses RANDOM actions to verify the environment is catchable.")
    print("Even random actions should occasionally capture in the improved env!\n")
    
    # Test improved environment
    results = test_improved_env(n_episodes=50)
    
    # Plot and analyze
    plot_results(results)
    
    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print("If capture rate > 5%:")
    print("  → Run: python PPO_AEG.py")
    print("\nIf capture rate = 0%:")
    print("  → Edit pursuit_evasion.py:")
    print("     evader_speed=0.03")
    print("     capture_radius=2.5")
    print("  → Run this test again")
    print("="*60)