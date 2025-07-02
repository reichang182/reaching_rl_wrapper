#!/usr/bin/env python3
"""Run RL training experiments with multiple random seeds.

Results are logged to a wandb project named with timestamp.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


def generate_random_seeds(n_seeds, base_seed=None):
    """Generate n random seeds for experiments."""
    if base_seed is not None:
        np.random.seed(base_seed)
    seeds = np.random.randint(0, 1_000_000, size=n_seeds)
    return seeds.tolist()


def run_single_experiment(cmd_args, seed, project_name, run_name_prefix):
    """Run a single training experiment with given seed."""
    # Construct command
    cmd = [
        sys.executable,
        "scripts/train_rl.py",
        "--seed",
        str(seed),
    ]

    # Add all other arguments
    cmd.extend(cmd_args)

    # Set environment variables for wandb
    env = os.environ.copy()
    env["WANDB_PROJECT"] = project_name
    env["WANDB_RUN_NAME"] = f"{run_name_prefix}_seed{seed}"

    print(f"\n{'='*60}")
    print(f"Running experiment with seed {seed}")
    print(f"Project: {project_name}")
    print(f"Run name: {run_name_prefix}_seed{seed}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    # Run the experiment
    try:
        result = subprocess.run(cmd, env=env, check=True)
        return {"seed": seed, "status": "success", "return_code": result.returncode}
    except subprocess.CalledProcessError as e:
        print(f"Error running seed {seed}: {e}")
        return {"seed": seed, "status": "failed", "return_code": e.returncode}
    except Exception as e:
        print(f"Unexpected error for seed {seed}: {e}")
        return {"seed": seed, "status": "error", "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Run multi-seed RL experiments")

    # Multi-seed specific arguments
    parser.add_argument("--n_seeds", type=int, default=5, help="Number of random seeds to run")
    parser.add_argument(
        "--base_seed", type=int, default=None, help="Base seed for generating random seeds"
    )
    parser.add_argument(
        "--project_suffix", type=str, default="", help="Suffix for wandb project name"
    )
    parser.add_argument("--parallel", action="store_true", help="Run experiments in parallel")
    parser.add_argument("--max_parallel", type=int, default=2, help="Maximum parallel experiments")

    # Training arguments to pass through
    parser.add_argument("--algorithm", type=str, default="SAC", choices=["SAC", "PPO"])
    parser.add_argument("--total_timesteps", type=int, default=100000)
    parser.add_argument("--eval_freq", type=int, default=10000)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--dof", type=int, default=3)
    parser.add_argument("--num_targets", type=int, default=1)
    parser.add_argument("--target_threshold", type=float, default=0.01)
    parser.add_argument("--collision_penalty", type=float, default=0)
    parser.add_argument("--machine_cost_weight", type=float, default=1e-4)
    parser.add_argument("--terminate_on_collision", action="store_true")
    parser.add_argument("--max_episode_steps", type=int, default=200)
    parser.add_argument("--enable_obstacles", action="store_true")
    parser.add_argument("--num_obstacles", type=int, default=5)
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--test_episodes", type=int, default=10)
    parser.add_argument("--save_episode_images", action="store_true", default=True)
    parser.add_argument(
        "--no_save_episode_images", dest="save_episode_images", action="store_false"
    )

    args = parser.parse_args()

    # Generate timestamp for project name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    project_name = f"rl_experiments_{timestamp}"
    if args.project_suffix:
        project_name += f"_{args.project_suffix}"

    # Generate seeds
    seeds = generate_random_seeds(args.n_seeds, args.base_seed)
    print(f"Generated seeds: {seeds}")

    # Create experiment directory
    exp_dir = Path(f"./experiments/{project_name}")
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save experiment configuration
    config = {
        "project_name": project_name,
        "timestamp": timestamp,
        "n_seeds": args.n_seeds,
        "seeds": seeds,
        "args": vars(args),
    }

    with open(exp_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Prepare command arguments for train_rl.py
    cmd_args = []

    # Add algorithm
    cmd_args.extend(["--algorithm", args.algorithm])

    # Add all other arguments (note: train_rl.py uses hyphens in some args)
    if args.total_timesteps != 100000:
        cmd_args.extend(["--total-timesteps", str(args.total_timesteps)])
    if args.eval_freq != 10000:
        cmd_args.extend(["--eval-freq", str(args.eval_freq)])
    if args.learning_rate != 3e-4:
        cmd_args.extend(["--learning_rate", str(args.learning_rate)])
    if args.batch_size != 256:
        cmd_args.extend(["--batch_size", str(args.batch_size)])
    if args.dof != 3:
        cmd_args.extend(["--dof", str(args.dof)])
    if args.num_targets != 1:
        cmd_args.extend(["--num_targets", str(args.num_targets)])
    if args.target_threshold != 0.01:
        cmd_args.extend(["--target_threshold", str(args.target_threshold)])
    if args.collision_penalty != 0:
        cmd_args.extend(["--collision_penalty", str(args.collision_penalty)])
    if args.machine_cost_weight != 1e-4:
        cmd_args.extend(["--machine_cost_weight", str(args.machine_cost_weight)])
    if args.terminate_on_collision:
        cmd_args.append("--terminate_on_collision")
    if args.max_episode_steps != 200:
        cmd_args.extend(["--max_episode_steps", str(args.max_episode_steps)])
    if args.enable_obstacles:
        cmd_args.append("--enable_obstacles")
        cmd_args.extend(["--num_obstacles", str(args.num_obstacles)])
    if args.num_envs != 64:
        cmd_args.extend(["--num_envs", str(args.num_envs)])
    if args.test_episodes != 10:
        cmd_args.extend(["--test-episodes", str(args.test_episodes)])
    if not args.save_episode_images:
        cmd_args.append("--no_save_episode_images")

    # Run name prefix
    run_name_prefix = f"{args.algorithm.lower()}_dof{args.dof}"
    if args.enable_obstacles:
        run_name_prefix += f"_obs{args.num_obstacles}"

    # Run experiments
    results = []

    if args.parallel:
        # Parallel execution using multiprocessing
        from functools import partial
        from multiprocessing import Pool

        run_func = partial(
            run_single_experiment,
            cmd_args,
            project_name=project_name,
            run_name_prefix=run_name_prefix,
        )

        with Pool(processes=min(args.max_parallel, len(seeds))) as pool:
            results = pool.map(run_func, seeds)
    else:
        # Sequential execution
        for seed in seeds:
            result = run_single_experiment(cmd_args, seed, project_name, run_name_prefix)
            results.append(result)

    # Save results
    results_file = exp_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"Project: {project_name}")
    print(f"Total experiments: {len(seeds)}")

    successful = sum(1 for r in results if r["status"] == "success")
    failed = sum(1 for r in results if r["status"] == "failed")
    errors = sum(1 for r in results if r["status"] == "error")

    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Errors: {errors}")
    print(f"\nResults saved to: {results_file}")
    print(f"Experiment config saved to: {exp_dir / 'config.json'}")

    # Print individual results
    print("\nIndividual results:")
    for result in results:
        status_emoji = "✅" if result["status"] == "success" else "❌"
        print(f"  Seed {result['seed']}: {status_emoji} {result['status']}")

    return 0 if successful == len(seeds) else 1


if __name__ == "__main__":
    sys.exit(main())
