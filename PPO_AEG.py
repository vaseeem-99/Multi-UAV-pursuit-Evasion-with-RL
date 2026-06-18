"""
MAPPO + AEG Training with Comprehensive Plotting
================================================
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import json
from datetime import datetime

from pursuit_evasion import PursuitEvasionImproved
from Adaptive_Env_Generator import AdaptiveEnvironmentGenerator

#   MAPPO NETWORKS
# ============================================================

class MAPPOActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def get_dist(self, obs):
        mean = torch.tanh(self.net(obs))
        std = torch.exp(self.log_std).expand_as(mean)
        return torch.distributions.Normal(mean, std)

    def act(self, obs_np, deterministic=False, device="cpu"):
        obs = torch.FloatTensor(obs_np).to(device)
        dist = self.get_dist(obs)

        if deterministic:
            actions = dist.mean
        else:
            actions = dist.sample()

        logprobs = dist.log_prob(actions).sum(-1)
        actions_np = actions.detach().cpu().numpy()
        logp_np = logprobs.detach().cpu().numpy()

        return actions_np, logp_np


class CentralCritic(nn.Module):
    def __init__(self, joint_obs_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(joint_obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, joint_obs):
        return self.net(joint_obs).squeeze(-1)

#   GAE
# ============================================================

def compute_gae(rewards, values, dones, gamma=0.99, lam=0.97):
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    gae = 0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values[t+1] * (1-dones[t]) - values[t]
        gae = delta + gamma * lam * (1-dones[t]) * gae
        adv[t] = gae
    returns = adv + values[:-1]
    return returns, adv

#   TRAINING METRICS TRACKER
# ============================================================

class MetricsTracker:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Training metrics
        self.episode_rewards = []
        self.capture_rates = []
        self.min_distances = []
        self.drone_collisions = []
        self.obstacle_collisions = []
        self.episode_lengths = []
        
        # Archive metrics
        self.archive_sizes = []
        self.avg_obstacles = []
        
        # Loss metrics
        self.actor_losses = []
        self.critic_losses = []
        
    def log_episode(self, ep_reward, captured, min_dist, drone_coll, obs_coll, ep_len):
        self.episode_rewards.append(ep_reward)
        self.capture_rates.append(1.0 if captured else 0.0)
        self.min_distances.append(min_dist)
        self.drone_collisions.append(drone_coll)
        self.obstacle_collisions.append(obs_coll)
        self.episode_lengths.append(ep_len)
    
    def log_archive(self, archive_size, avg_obs):
        self.archive_sizes.append(archive_size)
        self.avg_obstacles.append(avg_obs)
    
    def log_losses(self, actor_loss, critic_loss):
        self.actor_losses.append(actor_loss)
        self.critic_losses.append(critic_loss)
    
    def save_metrics(self):
        metrics = {
            'episode_rewards': self.episode_rewards,
            'capture_rates': self.capture_rates,
            'min_distances': self.min_distances,
            'drone_collisions': self.drone_collisions,
            'obstacle_collisions': self.obstacle_collisions,
            'episode_lengths': self.episode_lengths,
            'archive_sizes': self.archive_sizes,
            'avg_obstacles': self.avg_obstacles,
            'actor_losses': self.actor_losses,
            'critic_losses': self.critic_losses,
        }
        
        with open(os.path.join(self.save_dir, 'training_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def plot_training_curves(self, window=50):
        """Generate comprehensive training plots"""
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Helper function for smoothing
        def smooth(data, window):
            if len(data) < window:
                return data
            return np.convolve(data, np.ones(window)/window, mode='valid')
        
        episodes = np.arange(len(self.episode_rewards))
        
        # 1. Episode Rewards
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(episodes, self.episode_rewards, alpha=0.3, label='Raw')
        if len(self.episode_rewards) >= window:
            smoothed = smooth(self.episode_rewards, window)
            ax1.plot(episodes[window-1:], smoothed, linewidth=2, label=f'{window}-ep avg')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.set_title('Episode Rewards')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Capture Rate
        ax2 = fig.add_subplot(gs[0, 1])
        if len(self.capture_rates) >= window:
            capture_smoothed = smooth(self.capture_rates, window) * 100
            ax2.plot(episodes[window-1:], capture_smoothed, linewidth=2, color='green')
        ax2.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='50% target')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Capture Rate (%)')
        ax2.set_title(f'Capture Rate ({window}-ep moving avg)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 105])
        
        # 3. Minimum Distance to Evader
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(episodes, self.min_distances, alpha=0.3, label='Raw')
        if len(self.min_distances) >= window:
            dist_smoothed = smooth(self.min_distances, window)
            ax3.plot(episodes[window-1:], dist_smoothed, linewidth=2, label=f'{window}-ep avg')
        ax3.axhline(y=1.2, color='r', linestyle='--', alpha=0.5, label='Capture threshold')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Min Distance')
        ax3.set_title('Minimum Distance Achieved')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Collisions
        ax4 = fig.add_subplot(gs[1, 0])
        if len(self.drone_collisions) >= window:
            drone_coll_smooth = smooth(self.drone_collisions, window)
            ax4.plot(episodes[window-1:], drone_coll_smooth, linewidth=2, label='Drone-Drone', color='red')
        if len(self.obstacle_collisions) >= window:
            obs_coll_smooth = smooth(self.obstacle_collisions, window)
            ax4.plot(episodes[window-1:], obs_coll_smooth, linewidth=2, label='Drone-Obstacle', color='orange')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Collisions per Episode')
        ax4.set_title(f'Collision Statistics ({window}-ep avg)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Episode Length
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.plot(episodes, self.episode_lengths, alpha=0.3, label='Raw')
        if len(self.episode_lengths) >= window:
            length_smooth = smooth(self.episode_lengths, window)
            ax5.plot(episodes[window-1:], length_smooth, linewidth=2, label=f'{window}-ep avg')
        ax5.set_xlabel('Episode')
        ax5.set_ylabel('Steps')
        ax5.set_title('Episode Length')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Archive Size
        ax6 = fig.add_subplot(gs[1, 2])
        if len(self.archive_sizes) > 0:
            ax6.plot(self.archive_sizes, linewidth=2, color='purple')
        ax6.set_xlabel('Update')
        ax6.set_ylabel('Archive Size')
        ax6.set_title('AEG Archive Growth')
        ax6.grid(True, alpha=0.3)
        
        # 7. Average Obstacles in Archive
        ax7 = fig.add_subplot(gs[2, 0])
        if len(self.avg_obstacles) > 0:
            ax7.plot(self.avg_obstacles, linewidth=2, color='brown')
        ax7.set_xlabel('Update')
        ax7.set_ylabel('Avg # Obstacles')
        ax7.set_title('Curriculum Complexity')
        ax7.grid(True, alpha=0.3)
        
        # 8. Actor Loss
        ax8 = fig.add_subplot(gs[2, 1])
        if len(self.actor_losses) > 0:
            ax8.plot(self.actor_losses, linewidth=2, color='blue', alpha=0.7)
        ax8.set_xlabel('Update')
        ax8.set_ylabel('Loss')
        ax8.set_title('Actor Loss')
        ax8.grid(True, alpha=0.3)
        
        # 9. Critic Loss
        ax9 = fig.add_subplot(gs[2, 2])
        if len(self.critic_losses) > 0:
            ax9.plot(self.critic_losses, linewidth=2, color='red', alpha=0.7)
        ax9.set_xlabel('Update')
        ax9.set_ylabel('Loss')
        ax9.set_title('Critic Loss')
        ax9.grid(True, alpha=0.3)
        
        plt.savefig(os.path.join(self.save_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved training curves to {self.save_dir}/training_curves.png")

#   EVALUATION
# ============================================================

def evaluate_policy(actor, env, n_episodes=100, device="cpu"):
    """Evaluate trained policy"""
    print(f"\n{'='*60}")
    print(f"EVALUATING POLICY ({n_episodes} episodes)")
    print(f"{'='*60}")
    
    results = {
        'rewards': [],
        'captures': [],
        'min_distances': [],
        'episode_lengths': [],
        'drone_collisions': [],
        'obstacle_collisions': []
    }
    
    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        ep_reward = 0
        steps = 0
        drone_coll_total = 0
        obs_coll_total = 0
        
        while not done:
            with torch.no_grad():
                actions, _ = actor.act(obs, deterministic=True, device=device)
            
            obs, rewards, done, info = env.step(actions)
            ep_reward += np.mean(rewards)
            steps += 1
            drone_coll_total += info.get('drone_collisions', 0)
            obs_coll_total += info.get('obstacle_collisions', 0)
        
        results['rewards'].append(ep_reward)
        results['captures'].append(1 if info['captured'] else 0)
        results['min_distances'].append(info['min_distance'])
        results['episode_lengths'].append(steps)
        results['drone_collisions'].append(drone_coll_total)
        results['obstacle_collisions'].append(obs_coll_total)
        
        if (ep + 1) % 10 == 0:
            cap_rate = np.mean(results['captures']) * 100
            avg_reward = np.mean(results['rewards'])
            print(f"[{ep+1}/{n_episodes}] Capture Rate: {cap_rate:.1f}% | Avg Reward: {avg_reward:.1f}")
    
    return results


def plot_evaluation_results(results, save_dir):
    """Plot evaluation results"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Evaluation Results', fontsize=16, fontweight='bold')
    
    # 1. Reward Distribution
    axes[0, 0].hist(results['rewards'], bins=30, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(np.mean(results['rewards']), color='r', linestyle='--', 
                       label=f'Mean: {np.mean(results["rewards"]):.1f}')
    axes[0, 0].set_xlabel('Episode Reward')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Reward Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Capture Rate
    cap_rate = np.mean(results['captures']) * 100
    axes[0, 1].bar(['Failed', 'Captured'], 
                   [100-cap_rate, cap_rate],
                   color=['red', 'green'], alpha=0.7)
    axes[0, 1].set_ylabel('Percentage (%)')
    axes[0, 1].set_title(f'Capture Rate: {cap_rate:.1f}%')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # 3. Min Distance Distribution
    axes[0, 2].hist(results['min_distances'], bins=30, edgecolor='black', alpha=0.7)
    axes[0, 2].axvline(1.2, color='r', linestyle='--', label='Capture threshold')
    axes[0, 2].axvline(np.mean(results['min_distances']), color='g', linestyle='--',
                       label=f'Mean: {np.mean(results["min_distances"]):.2f}')
    axes[0, 2].set_xlabel('Minimum Distance')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Closest Approach Distribution')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Episode Length
    axes[1, 0].hist(results['episode_lengths'], bins=30, edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(np.mean(results['episode_lengths']), color='r', linestyle='--',
                       label=f'Mean: {np.mean(results["episode_lengths"]):.1f}')
    axes[1, 0].set_xlabel('Episode Length (steps)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Episode Length Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Collision Statistics
    avg_drone_coll = np.mean(results['drone_collisions'])
    avg_obs_coll = np.mean(results['obstacle_collisions'])
    axes[1, 1].bar(['Drone-Drone', 'Drone-Obstacle'],
                   [avg_drone_coll, avg_obs_coll],
                   color=['red', 'orange'], alpha=0.7)
    axes[1, 1].set_ylabel('Avg Collisions per Episode')
    axes[1, 1].set_title('Collision Statistics')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # 6. Success vs Episode Length
    captures = np.array(results['captures'])
    lengths = np.array(results['episode_lengths'])
    axes[1, 2].scatter(lengths[captures == 1], 
                       np.arange(np.sum(captures)), 
                       c='green', alpha=0.6, label='Captured', s=50)
    axes[1, 2].scatter(lengths[captures == 0], 
                       np.arange(np.sum(1-captures)), 
                       c='red', alpha=0.6, label='Failed', s=50)
    axes[1, 2].set_xlabel('Episode Length (steps)')
    axes[1, 2].set_ylabel('Episode Index')
    axes[1, 2].set_title('Capture Success vs Episode Length')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'evaluation_results.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved evaluation plots to {save_dir}/evaluation_results.png")
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Capture Rate:          {cap_rate:.2f}%")
    print(f"Avg Reward:            {np.mean(results['rewards']):.2f} ± {np.std(results['rewards']):.2f}")
    print(f"Avg Min Distance:      {np.mean(results['min_distances']):.3f} ± {np.std(results['min_distances']):.3f}")
    print(f"Avg Episode Length:    {np.mean(results['episode_lengths']):.1f} ± {np.std(results['episode_lengths']):.1f}")
    print(f"Avg Drone Collisions:  {avg_drone_coll:.2f}")
    print(f"Avg Obs Collisions:    {avg_obs_coll:.2f}")
    print(f"{'='*60}\n")

#   MAIN TRAINING LOOP
# ============================================================

def train_mappo(
    n_episodes=3000,
    episodes_per_update=10,
    ppo_epochs=10,
    minibatch_size=1024,
    save_dir="checkpoints_mappo",
    eval_interval=500,
    eval_episodes=100,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create environment and AEG
    env = PursuitEvasionImproved(gui=False, n_pursuers=3, arena_size=14.0)
    aeg = AdaptiveEnvironmentGenerator(n_pursuers=3, arena_size=14.0)

    n_agents = env.n_pursuers
    obs_dim = env.obs_dim
    act_dim = env.action_space.shape[-1]
    joint_obs_dim = obs_dim * n_agents

    # Networks
    actor = MAPPOActor(obs_dim, act_dim).to(device)
    critic = CentralCritic(joint_obs_dim).to(device)

    actor_opt = optim.Adam(actor.parameters(), lr=3e-4)
    critic_opt = optim.Adam(critic.parameters(), lr=1e-3)

    # Metrics tracker
    metrics = MetricsTracker(save_dir)

    # Training buffers
    buf_obs, buf_actions, buf_logp = [], [], []
    buf_returns, buf_advs = [], []
    episodes_in_batch = 0

    print(f"\n{'='*60}")
    print("STARTING TRAINING")
    print(f"{'='*60}\n")

    for ep in range(1, n_episodes+1):
        scenario = aeg.sample_task()
        obs = env.reset(scenario=scenario)
        done = False

        ep_rewards = []
        ep_values = []
        ep_dones = []
        ep_obs_list = []
        ep_actions_list = []
        ep_logp_list = []

        captured_flag = False
        team_return = 0
        total_drone_collisions = 0
        total_obstacle_collisions = 0
        steps = 0

        while not done:
            actions_np, logp_np = actor.act(obs, device=device)

            joint_obs = obs.reshape(-1)
            with torch.no_grad():
                V = critic(torch.FloatTensor(joint_obs).unsqueeze(0).to(device)).item()

            next_obs, rewards, done, info = env.step(actions_np)
            team_reward = float(np.mean(rewards))

            ep_obs_list.append(obs)
            ep_actions_list.append(actions_np)
            ep_logp_list.append(logp_np)
            ep_rewards.append(team_reward)
            ep_values.append(V)
            ep_dones.append(float(done))

            team_return += team_reward
            captured_flag = captured_flag or info["captured"]
            total_drone_collisions += info.get("drone_collisions", 0)
            total_obstacle_collisions += info.get("obstacle_collisions", 0)
            steps += 1

            obs = next_obs

        # Bootstrap
        joint_obs = obs.reshape(-1)
        with torch.no_grad():
            last_v = 0.0 if done else critic(torch.FloatTensor(joint_obs).unsqueeze(0).to(device)).item()

        values = np.array(ep_values + [last_v], dtype=np.float32)
        rews = np.array(ep_rewards, dtype=np.float32)
        dones = np.array(ep_dones, dtype=np.float32)

        returns, advs = compute_gae(rews, values, dones)

        T = len(returns)
        ep_obs_arr = np.array(ep_obs_list, dtype=np.float32)
        ep_actions_arr = np.array(ep_actions_list, dtype=np.float32)
        ep_logp_arr = np.array(ep_logp_list, dtype=np.float32)

        returns_expand = np.tile(returns.reshape(T, 1), (1, n_agents))
        advs_expand = np.tile(advs.reshape(T, 1), (1, n_agents))

        buf_obs.append(ep_obs_arr)
        buf_actions.append(ep_actions_arr)
        buf_logp.append(ep_logp_arr)
        buf_returns.append(returns_expand)
        buf_advs.append(advs_expand)

        episodes_in_batch += 1

        # Update AEG
        aeg.update_archive(scenario, captured_flag)

        # Log episode metrics
        metrics.log_episode(team_return, captured_flag, env.min_distance, 
                          total_drone_collisions, total_obstacle_collisions, steps)

        # PPO UPDATE
        if episodes_in_batch >= episodes_per_update:
            obs_batch = np.concatenate(buf_obs, axis=0)
            actions_batch = np.concatenate(buf_actions, axis=0)
            logp_batch = np.concatenate(buf_logp, axis=0)
            ret_batch = np.concatenate(buf_returns, axis=0)
            adv_batch = np.concatenate(buf_advs, axis=0)

            T_tot, N, D = obs_batch.shape

            obs_flat = obs_batch.reshape(T_tot*N, D)
            actions_flat = actions_batch.reshape(T_tot*N, -1)
            logp_flat = logp_batch.reshape(T_tot*N)
            ret_flat = ret_batch.reshape(T_tot*N)
            adv_flat = adv_batch.reshape(T_tot*N)

            joint_obs_per_t = obs_batch.reshape(T_tot, -1)
            joint_obs_flat = np.repeat(joint_obs_per_t, N, axis=0)

            obs_t = torch.FloatTensor(obs_flat).to(device)
            actions_t = torch.FloatTensor(actions_flat).to(device)
            old_logp_t = torch.FloatTensor(logp_flat).to(device)
            returns_t = torch.FloatTensor(ret_flat).to(device)
            adv_t = torch.FloatTensor(adv_flat).to(device)
            joint_obs_t = torch.FloatTensor(joint_obs_flat).to(device)

            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

            epoch_actor_losses = []
            epoch_critic_losses = []

            idxs = np.arange(len(obs_t))
            for _ in range(ppo_epochs):
                np.random.shuffle(idxs)
                for start in range(0, len(idxs), minibatch_size):
                    mb = idxs[start:start+minibatch_size]

                    dist = actor.get_dist(obs_t[mb])
                    new_logp = dist.log_prob(actions_t[mb]).sum(-1)
                    entropy = dist.entropy().sum(-1).mean()

                    V = critic(joint_obs_t[mb])

                    ratio = torch.exp(new_logp - old_logp_t[mb])
                    s1 = ratio * adv_t[mb]
                    s2 = torch.clamp(ratio, 0.8, 1.2) * adv_t[mb]
                    actor_loss = -torch.min(s1, s2).mean()

                    critic_loss = nn.MSELoss()(V, returns_t[mb])

                    loss = actor_loss + 0.5*critic_loss - 0.01*entropy

                    actor_opt.zero_grad()
                    critic_opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
                    nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
                    actor_opt.step()
                    critic_opt.step()

                    epoch_actor_losses.append(actor_loss.item())
                    epoch_critic_losses.append(critic_loss.item())

            # Log losses and archive stats
            metrics.log_losses(np.mean(epoch_actor_losses), np.mean(epoch_critic_losses))
            arc_stats = aeg.get_archive_stats()
            metrics.log_archive(arc_stats['archive_size'], arc_stats['avg_obstacles'])

            # Clear buffer
            buf_obs.clear()
            buf_actions.clear()
            buf_logp.clear()
            buf_returns.clear()
            buf_advs.clear()
            episodes_in_batch = 0

        # LOGGING
        if ep % 20 == 0:
            avg_r = np.mean(metrics.episode_rewards[-20:])
            cap_rate = 100*np.mean(metrics.capture_rates[-20:])
            avg_drone_coll = np.mean(metrics.drone_collisions[-20:])
            avg_obs_coll = np.mean(metrics.obstacle_collisions[-20:])
            avg_min_dist = np.mean(metrics.min_distances[-20:])
            arc_stats = aeg.get_archive_stats()
            
            print(
                f"[EP {ep:4d}] Reward:{avg_r:7.1f} | Capture:{cap_rate:5.1f}% | "
                f"Archive:{arc_stats['archive_size']:3d} | AvgObs:{arc_stats['avg_obstacles']:.1f} | "
                f"MinDist:{avg_min_dist:.2f} | D-Coll:{avg_drone_coll:.1f} | O-Coll:{avg_obs_coll:.1f}"
            )

        # SAVE CHECKPOINT & EVALUATE
        if ep % eval_interval == 0:
            # Save model
            pt = os.path.join(save_dir, f"mappo_aeg_ep{ep}.pt")
            torch.save({
                'actor': actor.state_dict(),
                'critic': critic.state_dict(),
                'episode': ep,
                'metrics': metrics.__dict__
            }, pt)
            print(f"✓ Saved checkpoint: {pt}")
            
            # Save metrics and plot
            metrics.save_metrics()
            metrics.plot_training_curves()
            
            # Evaluate
            print(f"\n--- Running evaluation at episode {ep} ---")
            eval_results = evaluate_policy(actor, env, n_episodes=eval_episodes, device=device)
            plot_evaluation_results(eval_results, save_dir)

    env.close()
    
    # Final save
    metrics.save_metrics()
    metrics.plot_training_curves()
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETED")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    train_mappo(
        n_episodes=3000,
        episodes_per_update=10,
        ppo_epochs=10,
        minibatch_size=1024,
        save_dir="checkpoints_mappo",
        eval_interval=500,
        eval_episodes=100
    )