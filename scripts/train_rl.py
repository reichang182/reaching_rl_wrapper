import argparse
import os
import sys

import numpy as np
import torch
import wandb
import warp as wp
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from wandb.integration.sb3 import WandbCallback

from envs.ik_gym_wrapper import InverseKinematicsEnv

# Add parent directory to path to import from envs
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# For video rendering (optional - will skip if not available)
try:
    from pyvirtualdisplay import Display

    PYVIRTUALDISPLAY_AVAILABLE = True
except ImportError:
    PYVIRTUALDISPLAY_AVAILABLE = False
    print("Warning: pyvirtualdisplay not available. Video rendering may fail on headless systems.")


# Custom Callback for logging episode-end metrics
class EpisodeEndMetricsLogger(BaseCallback):
    """Custom callback for logging episode-end metrics."""

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Check if any environment is done
        if self.locals["dones"][0]:  # For a single environment
            info = self.locals["infos"][0]

            # Log all available metrics
            if "distance_to_target" in info:
                wandb.log(
                    {"train/final_distance_to_target": info["distance_to_target"]},
                    step=self.num_timesteps,
                )
            if "is_success" in info:
                wandb.log({"train/is_success": float(info["is_success"])}, step=self.num_timesteps)
            if "machine_cost" in info:
                wandb.log({"train/machine_cost": info["machine_cost"]}, step=self.num_timesteps)
            if "smoothness_penalty" in info:
                wandb.log(
                    {"train/smoothness_penalty": info["smoothness_penalty"]},
                    step=self.num_timesteps,
                )

            if self.verbose > 0:
                print(
                    f"Episode ended at timestep {self.num_timesteps}: "
                    f"distance={info.get('distance_to_target', 'N/A'):.4f}, "
                    f"success={info.get('is_success', False)}"
                )
        return True


# Initialize ArgumentParser
parser = argparse.ArgumentParser(description="Train RL agent for inverse kinematics task.")
parser.add_argument(
    "--algorithm", type=str, default="SAC", choices=["SAC", "PPO"], help="RL algorithm to use"
)
parser.add_argument("--dof", type=int, default=3, help="Degrees of freedom for the robot")
parser.add_argument("--num_targets", type=int, default=1, help="Number of targets")
parser.add_argument(
    "--enable_obstacles", action="store_true", help="Enable obstacles in the environment"
)
parser.add_argument("--num_obstacles", type=int, default=0, help="Number of obstacles")
parser.add_argument(
    "--total-timesteps", type=int, default=1000000, help="Total number of training timesteps"
)
parser.add_argument(
    "--eval-freq", type=int, default=10000, help="Frequency of evaluation during training"
)
parser.add_argument(
    "--n-eval-episodes", type=int, default=10, help="Number of episodes for evaluation"
)
parser.add_argument("--test-episodes", type=int, default=5, help="Number of episodes for testing")
parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
parser.add_argument("--fps", type=int, default=30, help="FPS for environment simulation")
parser.add_argument("--max_episode_steps", type=int, default=200, help="Maximum steps per episode")
parser.add_argument(
    "--target_threshold", type=float, default=0.05, help="Distance threshold for success"
)
parser.add_argument("--collision_penalty", type=float, default=1.0, help="Penalty for collisions")
parser.add_argument(
    "--machine_cost_weight", type=float, default=0.0001, help="Weight for machine cost in reward"
)
parser.add_argument(
    "--terminate_on_collision", action="store_true", help="Terminate episode on collision"
)
parser.add_argument(
    "--learning_rate", type=float, default=3e-4, help="Learning rate for the RL algorithm"
)
parser.add_argument(
    "--batch_size", type=int, default=256, help="Batch size for training (SAC only)"
)
parser.add_argument("--stage_path", type=str, default=".", help="Path to stage directory")

args = parser.parse_args()

# Handle seeding
if args.seed is None:
    args.seed = np.random.randint(0, 1_000_000)
print(f"Using seed: {args.seed}")
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
wp.rand_init(args.seed)

# Experiment naming
exp_name_parts = ["ik", args.algorithm.lower(), f"dof{args.dof}"]
if args.enable_obstacles:
    exp_name_parts.append(f"obs{args.num_obstacles}")
exp_name_parts.append(f"seed{args.seed}")
exp_name = "-".join(exp_name_parts)

# Define output directory
base_output_dir = f"./logs/{exp_name}"
os.makedirs(base_output_dir, exist_ok=True)

# Initialize wandb
wandb.init(project="inverse-kinematics-rl", name=exp_name, sync_tensorboard=True, config=vars(args))


# Create environment function
def make_env(seed_val):
    env = InverseKinematicsEnv(
        random_seed=seed_val,
        dof=args.dof,
        number_of_targets=args.num_targets,
        fps=args.fps,
        max_episode_steps=args.max_episode_steps,
        terminate_on_collision=args.terminate_on_collision,
        collision_penalty=args.collision_penalty,
        target_threshold=args.target_threshold,
        machine_cost_weight=args.machine_cost_weight,
        enable_obstacles=args.enable_obstacles,
        number_of_obstacles=args.num_obstacles,
        stage_path=args.stage_path,
    )
    env = TimeLimit(env, max_episode_steps=args.max_episode_steps)
    env = Monitor(env)
    return env


# Create training and evaluation environments
env = make_env(args.seed)
eval_env = make_env(args.seed + 1)

# Initialize model based on chosen algorithm
print(f"Training {args.algorithm} agent...")
model_log_path = base_output_dir

if args.algorithm == "SAC":
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=model_log_path,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        seed=args.seed,
    )
elif args.algorithm == "PPO":
    # PPO with larger network
    policy_kwargs = {"net_arch": [{"pi": [256, 256], "vf": [256, 256]}]}
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=model_log_path,
        learning_rate=args.learning_rate,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs=policy_kwargs,
        seed=args.seed,
    )
else:
    raise ValueError(f"Unsupported algorithm: {args.algorithm}")

# Setup callbacks
wandb_callback = WandbCallback(
    gradient_save_freq=args.eval_freq,
    model_save_path=f"{base_output_dir}/models/wandb/",
    verbose=2,
)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=f"{base_output_dir}/models/best_model/",
    log_path=f"{base_output_dir}/eval_logs/",
    eval_freq=args.eval_freq,
    n_eval_episodes=args.n_eval_episodes,
    deterministic=True,
    render=False,
    verbose=1,
)

episode_metrics_logger = EpisodeEndMetricsLogger(verbose=0)

# Train the model
print(f"\nStarting training for {args.total_timesteps} timesteps...")
model.learn(
    total_timesteps=args.total_timesteps,
    callback=[wandb_callback, eval_callback, episode_metrics_logger],
    progress_bar=True,
)

# Save final model
os.makedirs(f"{base_output_dir}/models/", exist_ok=True)
model.save(f"{base_output_dir}/models/{exp_name}_final")
print(f"Final model saved to {base_output_dir}/models/{exp_name}_final.zip")

# Evaluate the model
print(f"\nEvaluating {args.algorithm} agent...")
mean_reward, std_reward = evaluate_policy(
    model, eval_env, n_eval_episodes=args.n_eval_episodes, deterministic=True
)
wandb.log(
    {
        f"{args.algorithm.lower()}_eval_mean_reward": mean_reward,
        f"{args.algorithm.lower()}_eval_std_reward": std_reward,
    }
)
print(f"{args.algorithm} Evaluation Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")


# Test the trained model
def test_agent(model_to_test, test_env, num_episodes, base_seed):
    """Test the agent and collect metrics."""
    rewards = []
    successes = []
    final_distances = []

    for episode in range(num_episodes):
        episode_seed = base_seed + 10000 + episode
        obs, info = test_env.reset(seed=episode_seed)

        done = False
        truncated = False
        episode_reward = 0
        num_steps = 0

        while not (done or truncated):
            action, _ = model_to_test.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = test_env.step(action)
            episode_reward += reward
            num_steps += 1

        rewards.append(episode_reward)
        successes.append(float(info.get("is_success", False)))
        final_distances.append(info.get("distance_to_target", float("inf")))

        print(
            f"Test Episode {episode+1}: Reward = {episode_reward:.2f}, "
            f"Steps = {num_steps}, Success = {info.get('is_success', False)}, "
            f"Distance = {info.get('distance_to_target', 'N/A'):.4f}"
        )

    # Log test results
    test_results = {
        f"{args.algorithm.lower()}_test_avg_reward": np.mean(rewards),
        f"{args.algorithm.lower()}_test_success_rate": np.mean(successes),
        f"{args.algorithm.lower()}_test_avg_final_distance": np.mean(final_distances),
    }
    wandb.log(test_results)

    return test_results


if args.test_episodes > 0:
    print(f"\nTesting {args.algorithm} agent for {args.test_episodes} episodes...")
    test_env = make_env(args.seed + 2)

    # Start virtual display if available and needed
    display = None
    if PYVIRTUALDISPLAY_AVAILABLE:
        try:
            display = Display(visible=0, size=(640, 480))
            display.start()
            print("Virtual display started for rendering.")
        except Exception as e:
            print(f"Could not start virtual display: {e}")

    test_results = test_agent(model, test_env, args.test_episodes, args.seed)

    print("\nTest Results:")
    print(f"Average Reward: {test_results[f'{args.algorithm.lower()}_test_avg_reward']:.2f}")
    print(f"Success Rate: {test_results[f'{args.algorithm.lower()}_test_success_rate']:.2%}")
    print(
        f"Average Final Distance: "
        f"{test_results[f'{args.algorithm.lower()}_test_avg_final_distance']:.4f}"
    )

    test_env.close()

    if display:
        try:
            display.stop()
            print("Virtual display stopped.")
        except Exception as e:
            print(f"Error stopping virtual display: {e}")

# Clean up
env.close()
eval_env.close()
wandb.finish()

print(f"\nTraining completed! Results saved to {base_output_dir}")
