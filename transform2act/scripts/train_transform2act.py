#!/usr/bin/env python3
"""Training script for Transform2Act with IK environment.

This is a minimal implementation for testing the integration.
"""

import argparse
import os
import pickle
import sys
import time
from datetime import datetime
from typing import Any, Dict

# Add parent directories to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import wandb
import yaml

# Import our Transform2Act IK environment
from reaching_rl_wrapper.transform2act.envs.transform2act_ik_env import Transform2ActIKEnv


class SimpleTransform2ActAgent:
    """Simplified Transform2Act agent for testing.

    In a full implementation, this would use the actual Transform2Act policy networks.
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.exploration_rate = 1.0
        self.exploration_steps = cfg["agent"]["exploration_steps"]
        self.final_exploration = cfg["agent"]["final_exploration"]
        self.total_steps = 0

    def act(self, obs, stage: str, action_space_size: int):
        """Generate action based on observation and current stage.

        Args:
        ----
            obs: Observation from environment
            stage: Current stage ("skel_trans", "attr_trans", or "execution")
            action_space_size: Size of action space

        Returns:
        -------
            Action for the current stage

        """
        # Update exploration rate
        if self.total_steps < self.exploration_steps:
            self.exploration_rate = 1.0 - (1.0 - self.final_exploration) * (
                self.total_steps / self.exploration_steps
            )
        else:
            self.exploration_rate = self.final_exploration

        # Random action for testing (in practice, would use neural network)
        if np.random.random() < self.exploration_rate:
            if stage == "skel_trans":
                # Discrete action: 0 (no change), 1 (add), 2 (remove)
                action = np.random.randint(0, action_space_size)
            else:
                # Continuous action
                action = np.random.uniform(-1, 1, size=action_space_size)
        else:
            # Greedy action (simplified - just go towards target)
            if stage == "skel_trans":
                action = 0  # No change for now
            elif stage == "attr_trans":
                # Small random changes to link lengths
                action = np.random.uniform(-0.2, 0.2, size=action_space_size)
            else:  # execution
                # Improved control strategy
                obs[0]
                n_links = action_space_size

                # Extract relevant info from observation
                # Assuming observation contains joint positions and target info
                # This is a heuristic - proper RL would learn this

                # Strategy 1: Sweep through joint space systematically
                t = self.total_steps * 0.02
                base_positions = np.array(
                    [np.sin(t + i * np.pi / n_links) * 0.8 for i in range(n_links)]
                )

                # Strategy 2: Add some randomness for exploration
                noise = np.random.normal(0, 0.1, size=n_links)

                # Strategy 3: Gradually decrease joint angles for some poses
                if np.random.random() < 0.3:
                    for i in range(n_links):
                        base_positions[i] *= 1.0 - i * 0.2

                action = np.clip(base_positions + noise, -1, 1)

        return action

    def update(self, trajectory):
        """Update agent (placeholder for actual training)."""
        self.total_steps += len(trajectory)
        # In practice, would update neural networks here
        pass

    def save(self, path):
        """Save agent state."""
        state = {
            "total_steps": self.total_steps,
            "exploration_rate": self.exploration_rate,
            "cfg": self.cfg,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load(self, path):
        """Load agent state."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.total_steps = state["total_steps"]
        self.exploration_rate = state["exploration_rate"]


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def evaluate_agent(env: Transform2ActIKEnv, agent: SimpleTransform2ActAgent, n_episodes: int = 10):
    """Evaluate agent performance."""
    rewards = []
    successes = []

    for _ep in range(n_episodes):
        obs = env.reset_model()
        episode_reward = 0
        done = False

        while not done:
            action = agent.act(obs, env.stage, env.action_space)
            obs, reward, done, info = env.step(action)
            episode_reward += reward

        rewards.append(episode_reward)
        successes.append(info.get("is_success", False))

    return {
        "mean_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "success_rate": np.mean(successes),
    }


def main():
    """Run Transform2Act training."""
    parser = argparse.ArgumentParser(description="Train Transform2Act on IK task")
    parser.add_argument(
        "--config",
        type=str,
        default="transform2act/configs/ik_reaching.yml",
        help="Path to configuration file",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--test", action="store_true", help="Run in test mode")
    parser.add_argument("--checkpoint", type=str, default=None, help="Load from checkpoint")
    parser.add_argument(
        "--num-episodes", type=int, default=None, help="Override number of episodes"
    )
    args = parser.parse_args()

    # Load configuration
    cfg = load_config(args.config)

    # Initialize WandB if enabled
    use_wandb = cfg["training"].get("use_wandb", False) and not args.test
    if use_wandb:
        wandb.init(
            project=cfg["training"].get("wandb_project", "transform2act-ik"),
            entity=cfg["training"].get("wandb_entity", None),
            name=f"transform2act_ik_seed{args.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "env_config": cfg["env"],
                "agent_config": cfg["agent"],
                "training_config": cfg["training"],
                "seed": args.seed,
            },
            tags=["transform2act", "ik-reaching", f"seed-{args.seed}"],
        )

    # Create environment
    print("Creating Transform2Act IK environment...")
    env = Transform2ActIKEnv(cfg["env"], random_seed=args.seed)

    # Create agent
    print("Creating Transform2Act agent...")
    agent = SimpleTransform2ActAgent(cfg)

    # Create checkpoint directory
    checkpoint_dir = cfg["training"].get("checkpoint_dir", "./checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    run_name = f"transform2act_ik_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(checkpoint_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Training parameters
    total_timesteps = cfg["training"]["total_timesteps"]
    eval_freq = cfg["training"]["eval_freq"]
    log_interval = cfg["training"]["log_interval"]
    save_freq = cfg["training"].get("save_freq", 50000)

    # Load checkpoint if provided
    start_timesteps = 0
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}...")
        agent.load(args.checkpoint)
        start_timesteps = agent.total_steps
        print(f"Resuming from timestep {start_timesteps}")

    if args.test:
        # Test mode - just run a few episodes
        print("\nRunning in test mode...")
        total_timesteps = 1000
        eval_freq = 500
        log_interval = 100

    # Training loop
    print(f"\nStarting training for {total_timesteps} timesteps...")
    print(f"Evaluation every {eval_freq} timesteps")
    print(f"Logging every {log_interval} timesteps")
    print(f"Saving checkpoints every {save_freq} timesteps")
    print(f"Checkpoint directory: {run_dir}")
    print("-" * 50)

    timesteps = start_timesteps
    episode = 0

    # Training statistics
    episode_rewards = []
    episode_lengths = []
    episode_successes = []
    link_counts = []
    best_reward = -float("inf")

    # Time tracking
    start_time = time.time()
    last_log_time = start_time

    while timesteps < total_timesteps:
        # Reset environment
        obs = env.reset_model()
        episode_reward = 0
        episode_length = 0
        done = False
        trajectory = []

        # Collect episode
        while not done:
            # Get action from agent
            action = agent.act(obs, env.stage, env.action_space)

            # Step environment
            next_obs, reward, done, info = env.step(action)

            # Store transition
            trajectory.append(
                {
                    "obs": obs,
                    "action": action,
                    "reward": reward,
                    "next_obs": next_obs,
                    "done": done,
                    "info": info,
                }
            )

            obs = next_obs
            episode_reward += reward
            episode_length += 1
            timesteps += 1

            # Log progress
            if timesteps % log_interval == 0 and timesteps > 0:
                # Calculate recent statistics
                recent_rewards = (
                    episode_rewards[-10:] if len(episode_rewards) >= 10 else episode_rewards
                )
                recent_successes = (
                    episode_successes[-10:] if len(episode_successes) >= 10 else episode_successes
                )

                # Calculate time statistics
                current_time = time.time()
                current_time - start_time
                time_since_log = current_time - last_log_time
                steps_per_sec = log_interval / time_since_log
                eta_seconds = (total_timesteps - timesteps) / steps_per_sec
                eta_minutes = eta_seconds / 60

                print(
                    f"[{timesteps:6d}/{total_timesteps}] "
                    f"Ep: {episode:4d} | "
                    f"Reward: {episode_reward:7.2f} | "
                    f"Avg(10): {np.mean(recent_rewards) if recent_rewards else 0:7.2f} | "
                    f"Success: {np.mean(recent_successes)*100 if recent_successes else 0:5.1f}% | "
                    f"Links: {env.current_links} | "
                    f"SPS: {steps_per_sec:5.1f} | "
                    f"ETA: {eta_minutes:5.1f}m"
                )

                # Log to WandB
                if use_wandb:
                    wandb.log(
                        {
                            "timesteps": timesteps,
                            "episode": episode,
                            "episode_reward": episode_reward,
                            "mean_reward_10": np.mean(recent_rewards) if recent_rewards else 0,
                            "success_rate_10": np.mean(recent_successes) if recent_successes else 0,
                            "current_links": env.current_links,
                            "steps_per_second": steps_per_sec,
                            "exploration_rate": agent.exploration_rate,
                        },
                        step=timesteps,
                    )

                last_log_time = current_time

            if timesteps >= total_timesteps:
                break

        # Store episode statistics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_successes.append(info.get("is_success", False))
        link_counts.append(env.current_links)

        # Update agent
        agent.update(trajectory)

        # Track best reward
        if episode_reward > best_reward:
            best_reward = episode_reward

        # Log episode to WandB
        if use_wandb:
            wandb.log(
                {
                    "episode_reward": episode_reward,
                    "episode_length": episode_length,
                    "episode_success": info.get("is_success", False),
                    "episode_links": env.current_links,
                    "best_reward": best_reward,
                    "final_distance": info.get("distance_to_target", 0),
                    "final_machine_cost": info.get("machine_cost", 0),
                },
                step=timesteps,
            )

        # Save checkpoint periodically
        if timesteps % save_freq == 0 and timesteps > 0:
            checkpoint_path = os.path.join(run_dir, f"checkpoint_{timesteps}.pkl")
            agent.save(checkpoint_path)
            print(f"\n💾 Checkpoint saved to {checkpoint_path}")

            # Also save training statistics
            stats_path = os.path.join(run_dir, f"stats_{timesteps}.pkl")
            stats = {
                "timesteps": timesteps,
                "episode": episode,
                "episode_rewards": episode_rewards,
                "episode_successes": episode_successes,
                "link_counts": link_counts,
                "best_reward": best_reward,
            }
            with open(stats_path, "wb") as f:
                pickle.dump(stats, f)

        # Evaluate periodically
        if timesteps % eval_freq == 0 and timesteps > 0:
            print("\n" + "=" * 60)
            print("EVALUATION")
            eval_results = evaluate_agent(env, agent, n_episodes=10)
            print(
                f"Mean Reward: {eval_results['mean_reward']:8.2f} ± "
                f"{eval_results['std_reward']:6.2f}"
            )
            print(f"Success Rate: {eval_results['success_rate']:7.1%}")
            print(f"Best Episode Reward: {best_reward:8.2f}")
            if link_counts:
                recent_links = link_counts[-100:] if len(link_counts) >= 100 else link_counts
                unique_links, counts = np.unique(recent_links, return_counts=True)
                link_dist = dict(zip(unique_links, counts))
                print(f"Link Distribution: {link_dist}")
            print("=" * 60 + "\n")

            # Log evaluation to WandB
            if use_wandb:
                wandb.log(
                    {
                        "eval/mean_reward": eval_results["mean_reward"],
                        "eval/std_reward": eval_results["std_reward"],
                        "eval/success_rate": eval_results["success_rate"],
                        "eval/best_reward_ever": best_reward,
                    },
                    step=timesteps,
                )

                # Log link distribution
                if link_counts:
                    for n_links, count in link_dist.items():
                        wandb.log({f"link_dist/{n_links}_links": count}, step=timesteps)

        episode += 1

    print("\nTraining completed!")

    # Save final checkpoint
    final_checkpoint = os.path.join(run_dir, "final_checkpoint.pkl")
    agent.save(final_checkpoint)
    print(f"\n💾 Final checkpoint saved to {final_checkpoint}")

    # Save final statistics
    final_stats_path = os.path.join(run_dir, "final_stats.pkl")
    final_stats = {
        "timesteps": timesteps,
        "episode": episode,
        "episode_rewards": episode_rewards,
        "episode_successes": episode_successes,
        "link_counts": link_counts,
        "best_reward": best_reward,
        "total_time": time.time() - start_time,
    }
    with open(final_stats_path, "wb") as f:
        pickle.dump(final_stats, f)

    # Final evaluation
    print("\nFinal evaluation...")
    eval_results = evaluate_agent(env, agent, n_episodes=20)
    print("\nFinal results: ")
    print(f"  Mean Reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
    print(f"  Success Rate: {eval_results['success_rate']:.2%}")
    print(f"  Best Episode Reward Ever: {best_reward:.2f}")
    print(f"  Total Training Time: {(time.time() - start_time) / 60:.1f} minutes")
    print(f"\nResults saved in: {run_dir}")

    # Log final results to WandB
    if use_wandb:
        wandb.log(
            {
                "final/mean_reward": eval_results["mean_reward"],
                "final/std_reward": eval_results["std_reward"],
                "final/success_rate": eval_results["success_rate"],
                "final/best_reward_ever": best_reward,
                "final/total_episodes": episode,
                "final/total_time_minutes": (time.time() - start_time) / 60,
            }
        )

        # Save model to WandB
        wandb.save(final_checkpoint)
        wandb.save(final_stats_path)

        # Finish WandB run
        wandb.finish()


if __name__ == "__main__":
    main()
