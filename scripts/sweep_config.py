import argparse
import os
import sys

import numpy as np
import torch
import wandb
import warp as wp

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add co_design_task to path
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import from co_design_task
from co_design_task.context.inverse_kinematics_context_manager import (
    InverseKinematicsContextManager,
)

# For video rendering (optional)
try:
    from pyvirtualdisplay import Display

    PYVIRTUALDISPLAY_AVAILABLE = True
except ImportError:
    PYVIRTUALDISPLAY_AVAILABLE = False
    print("Warning: pyvirtualdisplay not available. Video rendering may fail on headless systems.")

# Global variables for sweep configuration
SWEEP_CONFIG = {}


def evaluate_ik_config():
    """Evaluate inverse kinematics configuration for W&B sweep.

    Called by wandb.agent, initializes InverseKinematicsTask with parameters
    from wandb.config, evaluates static configuration, and logs results.
    """
    dof = SWEEP_CONFIG["dof"]
    num_targets = SWEEP_CONFIG["num_targets"]
    machine_cost_weight = SWEEP_CONFIG["machine_cost_weight"]

    with wandb.init() as run:
        cfg = wandb.config

        # Setup output directory
        run_output_dir = os.path.join("./sweep_outputs", run.id)
        os.makedirs(run_output_dir, exist_ok=True)

        # Configuration for InverseKinematicsTask
        task_config = {
            "general": {
                "stage_path": run_output_dir,
                "random_seed": np.random.randint(0, 1_000_000),
                "device": "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu",
            },
            "simulation": {"integrator_type": "euler", "sim_substeps": 1, "fps": 30},
            "task": {
                "degrees_of_freedom": dof,
                "number_of_targets": num_targets,
                "number_of_instances": 1,
                "grid_offset": 1.0,
                "table_size_x": 1.0,
                "table_size_y": 0.01,
                "table_size_z": 1.0,
                "robot_base_height": 0.05,
                "default_link_length": 0.2,
                "default_link_radius": 0.02,
                "end_effector_sphere_radius": 0.02,
                "enable_obstacles": False,
                "number_of_obstacles": 0,
                "requires_grad": False,
                "rigid_contact_margin": 0.01,
                "target_lower_bounds": [-0.5, 0.0, -0.5],
                "target_upper_bounds": [0.5, 0.5, 0.5],
                "joint_reset_lower": -np.pi,
                "joint_reset_upper": np.pi,
                # Additional required parameters
                "target_shared_random_position": False,
                "joint_pose_lower_bounds": [-np.pi] * dof,
                "joint_pose_upper_bounds": [np.pi] * dof,
                "link_length_reset_handler": "static",
                "link_length_static_values": [0.2] * dof,
                "collision": True,
                "collision_weight": 1.0,
            },
        }

        # Set random seeds
        current_seed = task_config["general"]["random_seed"]
        wp.rand_init(current_seed)
        np.random.seed(current_seed)
        torch.manual_seed(current_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(current_seed)

        # Create task using context manager
        task = InverseKinematicsContextManager.create(task_config)

        # Reset the task
        task.reset()

        # Prepare link lengths and joint angles from sweep config
        link_lengths = np.zeros(dof, dtype=np.float32)
        joint_angles = np.zeros(dof * num_targets, dtype=np.float32)

        # Extract link lengths from config
        for i in range(dof):
            link_lengths[i] = getattr(cfg, f"link_{i}_length")

        # Extract joint angles from config
        for target_idx in range(num_targets):
            for joint_idx in range(dof):
                if num_targets == 1:
                    # Single target: use joint_i_angle
                    joint_angles[target_idx * dof + joint_idx] = getattr(
                        cfg, f"joint_{joint_idx}_angle"
                    )
                else:
                    # Multiple targets: use joint_i_target_j_angle
                    joint_angles[target_idx * dof + joint_idx] = getattr(
                        cfg, f"joint_{joint_idx}_target_{target_idx}_angle"
                    )

        print(f"Starting trial {run.name} (ID: {run.id}) with DOF={dof}, Targets={num_targets}")
        print(f"  Link Lengths: {link_lengths.tolist()}")
        print(f"  Joint Angles: {joint_angles.tolist()}")

        # Set link lengths (if the task supports it)
        feature_states = task.get_feature_states()
        if "link_lengths" in feature_states:
            # Override the default link lengths using Warp's copy function
            wp.copy(
                task.feature_states["link_lengths"],
                wp.from_numpy(link_lengths, device=task.feature_states["link_lengths"].device),
            )

        # Set joint angles in the task using Warp's copy function
        wp.copy(
            task.feature_states["joint_poses"],
            wp.from_numpy(joint_angles, device=task.feature_states["joint_poses"].device),
        )

        # Step the task
        feature_states = task.step()

        # Get updated feature states
        feature_states = task.get_feature_states()

        # Calculate metrics
        total_distance = 0.0
        for target_idx in range(num_targets):
            ee_pos = feature_states["end_effector_position"][target_idx]
            target_pos = feature_states["target_position"][target_idx]
            distance = np.linalg.norm(ee_pos - target_pos)
            total_distance += distance

            # Log individual target metrics
            run.log(
                {
                    f"target_{target_idx}_distance": distance,
                    f"target_{target_idx}_ee_pos_x": ee_pos[0],
                    f"target_{target_idx}_ee_pos_y": ee_pos[1],
                    f"target_{target_idx}_ee_pos_z": ee_pos[2],
                }
            )

        avg_distance = total_distance / num_targets
        machine_cost = feature_states["machine_cost"][0]

        # Calculate final loss
        final_loss = avg_distance + machine_cost_weight * machine_cost

        # Check for collisions
        collision_count = 0
        if "collisions" in feature_states:
            collision_count = np.sum(feature_states["collisions"])

        # Log metrics
        metrics = {
            "final_loss": final_loss,
            "avg_distance": avg_distance,
            "total_distance": total_distance,
            "machine_cost": machine_cost,
            "collision_count": collision_count,
        }

        # Add configuration to metrics for easy reference
        for i in range(dof):
            metrics[f"config/link_{i}_length"] = link_lengths[i]
            metrics[f"config/joint_{i}_angle"] = joint_angles[i] if num_targets == 1 else "multiple"

        run.log(metrics)

        print(f"Finished trial {run.name}:")
        print(f"  Final Loss: {final_loss:.4f}")
        print(f"  Avg Distance: {avg_distance:.4f}")
        print(f"  Machine Cost: {machine_cost:.4f}")
        print(f"  Collisions: {collision_count}")


def main():
    parser = argparse.ArgumentParser(
        description="Run WandB sweep for InverseKinematicsTask static configuration."
    )
    parser.add_argument(
        "--project_name", type=str, default="ik_static_config_sweep", help="WandB project name"
    )
    parser.add_argument(
        "--sweep_name", type=str, default="default_static_sweep", help="WandB sweep name"
    )
    parser.add_argument("--dof", type=int, default=3, help="Degrees of Freedom for the robot")
    parser.add_argument("--num_targets", type=int, default=1, help="Number of targets")
    parser.add_argument(
        "--link_length_min", type=float, default=0.05, help="Minimum link length for sweep"
    )
    parser.add_argument(
        "--link_length_max", type=float, default=0.3, help="Maximum link length for sweep"
    )
    parser.add_argument(
        "--joint_angle_min", type=float, default=-np.pi, help="Minimum joint angle (radians)"
    )
    parser.add_argument(
        "--joint_angle_max", type=float, default=np.pi, help="Maximum joint angle (radians)"
    )
    parser.add_argument(
        "--num_trials", type=int, default=1000, help="Number of trials for the sweep"
    )
    parser.add_argument(
        "--machine_cost_weight",
        type=float,
        default=0.01,
        help="Weight for machine cost in final loss",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="bayes",
        choices=["grid", "random", "bayes"],
        help="Sweep method",
    )

    args = parser.parse_args()

    # Store configuration in global variable
    global SWEEP_CONFIG
    SWEEP_CONFIG = {
        "dof": args.dof,
        "num_targets": args.num_targets,
        "machine_cost_weight": args.machine_cost_weight,
    }

    # Build sweep configuration
    sweep_config = {
        "method": args.method,
        "metric": {"name": "final_loss", "goal": "minimize"},
        "parameters": {},
    }

    # Add parameters for each link length
    for i in range(args.dof):
        sweep_config["parameters"][f"link_{i}_length"] = {
            "distribution": "uniform",
            "min": args.link_length_min,
            "max": args.link_length_max,
        }

    # Add parameters for joint angles
    if args.num_targets == 1:
        # Single target: one set of joint angles
        for i in range(args.dof):
            sweep_config["parameters"][f"joint_{i}_angle"] = {
                "distribution": "uniform",
                "min": args.joint_angle_min,
                "max": args.joint_angle_max,
            }
    else:
        # Multiple targets: separate joint angles for each target
        for target_idx in range(args.num_targets):
            for joint_idx in range(args.dof):
                sweep_config["parameters"][f"joint_{joint_idx}_target_{target_idx}_angle"] = {
                    "distribution": "uniform",
                    "min": args.joint_angle_min,
                    "max": args.joint_angle_max,
                }

    print("Sweep Configuration:")
    import yaml

    print(yaml.dump(sweep_config))

    # Set up virtual display if available
    display = None
    if PYVIRTUALDISPLAY_AVAILABLE:
        try:
            display = Display(visible=0, size=(640, 480))
            display.start()
            print("Virtual display started for rendering")
        except Exception as e:
            print(f"Could not start virtual display: {e}")

    # Create and run sweep
    sweep_id = wandb.sweep(sweep_config, project=args.project_name)
    print(f"Sweep ID: {sweep_id}")
    print(f"Run `wandb agent {sweep_id}` to start additional agents.")

    print(f"\nStarting agent for {args.num_trials} trials...")
    wandb.agent(sweep_id, function=evaluate_ik_config, count=args.num_trials)

    print("Sweep completed!")

    # Clean up virtual display
    if display:
        try:
            display.stop()
            print("Virtual display stopped")
        except Exception as e:
            print(f"Error stopping virtual display: {e}")


if __name__ == "__main__":
    main()
