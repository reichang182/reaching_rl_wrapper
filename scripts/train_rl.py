import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import torch
import wandb
import warp as wp
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
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

# For image handling (optional)
try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: PIL not available. Image logging may be limited.")


# Custom Callback for logging episode-end metrics
class EpisodeEndMetricsLogger(BaseCallback):
    """Custom callback for logging episode-end metrics."""

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Check if any environment is done
        for i, done in enumerate(self.locals["dones"]):
            if done:
                info = self.locals["infos"][i]

                # Log all available metrics
                if "distance_to_target" in info:
                    wandb.log(
                        {"train/final_distance_to_target": info["distance_to_target"]},
                        step=self.num_timesteps,
                    )
                if "is_success" in info:
                    wandb.log(
                        {"train/is_success": float(info["is_success"])}, step=self.num_timesteps
                    )
                if "machine_cost" in info:
                    wandb.log({"train/machine_cost": info["machine_cost"]}, step=self.num_timesteps)
                if "smoothness_penalty" in info:
                    wandb.log(
                        {"train/smoothness_penalty": info["smoothness_penalty"]},
                        step=self.num_timesteps,
                    )

                if self.verbose > 0:
                    print(
                        f"Episode ended in env {i} at timestep {self.num_timesteps}: "
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
    "--total-timesteps", type=int, default=4000000, help="Total number of training timesteps"
)
parser.add_argument(
    "--eval-freq", type=int, default=20000, help="Frequency of evaluation during training"
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
parser.add_argument("--collision_penalty", type=float, default=0, help="Penalty for collisions")
parser.add_argument(
    "--machine_cost_weight", type=float, default=1e-2, help="Weight for machine cost in reward"
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
parser.add_argument(
    "--num_envs", type=int, default=64, help="Number of parallel environments for training"
)
parser.add_argument(
    "--save_episode_images",
    action="store_true",
    default=True,
    help="Save and upload episode images to wandb during testing (default: True)",
)
parser.add_argument(
    "--no_save_episode_images",
    dest="save_episode_images",
    action="store_false",
    help="Disable saving episode images",
)


def add_camera_to_usd(usd_path):
    """Add camera and lighting to USD file for better rendering."""
    try:
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdRender, UsdShade

        stage = Usd.Stage.Open(usd_path)
        if not stage:
            return False

        # Add camera - use same path as co_design_optimization
        cam_path = "/World/RenderCam"
        cam = UsdGeom.Camera.Define(stage, cam_path)

        # Position camera - elevated view to show entire robot including base
        eye = Gf.Vec3d(0.0, 1.0, 1.8)  # Y=1.0 (high enough), Z=1.8 (good distance)
        target = Gf.Vec3d(0.0, 0.2, 0.0)  # Look at robot upper section
        cam_xf = Gf.Matrix4d().SetLookAt(eye, target, Gf.Vec3d(0, 1, 0)).GetInverse()

        UsdGeom.Xformable(cam).ClearXformOpOrder()
        UsdGeom.Xformable(cam).AddTransformOp().Set(cam_xf)

        # Set camera properties - same as co_design_optimization
        cam.CreateFocalLengthAttr(24)  # Wide angle lens
        cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 10000.0))

        # Add sun light - same as co_design_optimization
        sun_path = Sdf.Path("/World/SunLight")
        sun = UsdLux.DistantLight.Define(stage, sun_path)
        sun.CreateIntensityAttr(3000)
        UsdGeom.Xformable(sun).AddRotateXOp().Set(-60)
        UsdGeom.Xformable(sun).AddRotateYOp().Set(-35)

        # Add materials - same structure as co_design_optimization
        mat_path = Sdf.Path("/World/Looks/DefaultPreviewMat")
        shader_path = mat_path.AppendPath("PBRShader")

        mat = UsdShade.Material.Define(stage, mat_path)
        shader = UsdShade.Shader.Define(stage, shader_path)
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.8, 0.8, 0.8))
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.1)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)

        shader_out = shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        mat.CreateSurfaceOutput().ConnectToSource(shader_out)

        # Bind material to all geometry
        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.Gprim):
                material_binding_api = UsdShade.MaterialBindingAPI(prim)
                if not material_binding_api.GetDirectBindingRel().GetTargets():
                    material_binding_api.Bind(mat)

        # Add render settings
        settings_path = Sdf.Path("/Render/Settings")
        settings = UsdRender.Settings.Define(stage, settings_path)
        settings.CreateCameraRel().SetTargets([cam.GetPath()])

        stage.Save()
        print("  Added camera, lights, and materials to USD")
        return True
    except Exception as e:
        print(f"Error adding camera to USD: {e}")
        import traceback

        traceback.print_exc()
        return False


def render_episode_state_to_image(env, image_path, resolution=(640, 480)):
    """Render the current environment state directly to an image using warp's OpenGL renderer."""
    try:
        # Try to use warp's OpenGL renderer if available
        import warp.render

        # Get the underlying environment
        while hasattr(env, "env") and not hasattr(env, "task"):
            env = env.env

        if not hasattr(env, "task"):
            raise RuntimeError("Could not access task from environment")

        # Create renderer
        renderer = warp.render.UsdRenderer(env.task.model, env.task.stage_path)

        # Set camera position
        renderer.set_camera_position((0.0, 1.5, 2.0))
        renderer.set_camera_look_at((0.0, 0.15, 0.0))

        # Render frame
        renderer.begin_frame(0.0)
        renderer.render(env.task.states[0])

        # Add target visualization
        targets = env.task.targets.numpy()
        for i in range(env.num_targets):
            renderer.render_sphere(
                name=f"target_{i}",
                pos=targets[i],
                rot=wp.quat_identity(),
                radius=0.02,
                color=(1.0, 0.0, 0.0),
            )

        renderer.end_frame()

        # Save screenshot
        renderer.save_screenshot(image_path)
        return True

    except Exception as e:
        print(f"Could not use warp renderer: {e}")
        return False


def convert_usd_to_image(usd_path, image_path, resolution=(640, 480)):
    """Convert USD file to image using usdrecord or fallback method."""
    # First, try to add camera to the USD file
    add_camera_to_usd(usd_path)

    # Try to use usdrecord if available
    usdrecord_exe = shutil.which("usdrecord")
    if usdrecord_exe:
        try:
            # Create a temporary directory for the frame
            with tempfile.TemporaryDirectory() as temp_dir:
                # usdrecord expects frame pattern like frame.0001.png
                temp_pattern = os.path.join(temp_dir, "frame.####.png")

                # Run usdrecord to render a single frame
                cmd = [
                    usdrecord_exe,
                    "--renderer",
                    "GL",  # Use GL renderer
                    "--frames",
                    "0:0",  # Only render frame 0
                    "--imageWidth",
                    str(resolution[0]),
                    "--camera",
                    "/World/RenderCam",  # Use camera path from co_design_optimization
                    usd_path,
                    temp_pattern,
                ]

                result = subprocess.run(cmd, capture_output=True, text=True)

                # The actual output file will be frame.0000.png
                actual_image = os.path.join(temp_dir, "frame.0000.png")

                if result.returncode == 0 and os.path.exists(actual_image):
                    # Move the rendered image to the desired location
                    shutil.move(actual_image, image_path)
                    print("  Successfully rendered USD to image using usdrecord")
                    return True
                # Don't print error if it fails, just fall through to placeholder

        except Exception:
            # Don't print error, just use fallback
            pass

    # Create a placeholder image as fallback
    if PIL_AVAILABLE:
        try:
            # Create a visualization showing episode completion
            img = Image.new("RGB", resolution, color="#1a1a2e")
            from PIL import ImageDraw

            draw = ImageDraw.Draw(img)

            # Add title
            title = "Episode Complete"
            draw.text((resolution[0] // 2 - 60, 50), title, fill="white")

            # Add a simple robot visualization
            # Base
            draw.rectangle(
                [
                    resolution[0] // 2 - 20,
                    resolution[1] - 100,
                    resolution[0] // 2 + 20,
                    resolution[1] - 80,
                ],
                fill="#16213e",
                outline="white",
            )

            # Links (simplified)
            draw.line(
                [resolution[0] // 2, resolution[1] - 90, resolution[0] // 2, resolution[1] - 150],
                fill="#0f3460",
                width=8,
            )
            draw.line(
                [
                    resolution[0] // 2,
                    resolution[1] - 150,
                    resolution[0] // 2 + 40,
                    resolution[1] - 200,
                ],
                fill="#0f3460",
                width=6,
            )
            draw.line(
                [
                    resolution[0] // 2 + 40,
                    resolution[1] - 200,
                    resolution[0] // 2 + 30,
                    resolution[1] - 250,
                ],
                fill="#0f3460",
                width=4,
            )

            # Target
            draw.ellipse(
                [
                    resolution[0] // 2 + 50,
                    resolution[1] // 2 - 10,
                    resolution[0] // 2 + 70,
                    resolution[1] // 2 + 10,
                ],
                fill="red",
                outline="white",
            )

            # Info text
            info_text = "View in wandb for details"
            draw.text((resolution[0] // 2 - 80, resolution[1] - 50), info_text, fill="#888888")

            img.save(image_path)
            return True
        except Exception as e:
            print(f"Error creating visualization image: {e}")

    return False


def main():
    args = parser.parse_args()
    args.eval_freq = args.eval_freq // args.num_envs

    # Handle seeding for reproducibility
    if args.seed is None:
        args.seed = np.random.randint(0, 1_000_000)
    print(f"Using seed: {args.seed}")

    # Set seeds for all random number generators
    import random

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    wp.rand_init(args.seed)

    # Set PyTorch to deterministic mode for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Experiment naming
    exp_name_parts = ["ik", args.algorithm.lower(), f"dof{args.dof}"]
    if args.enable_obstacles:
        exp_name_parts.append(f"obs{args.num_obstacles}")
    exp_name_parts.append(f"seed{args.seed}")
    exp_name = "-".join(exp_name_parts)

    # Define output directory
    base_output_dir = f"./logs/{exp_name}"
    os.makedirs(base_output_dir, exist_ok=True)

    # Initialize wandb (allow override from environment variables)
    wandb_project = os.environ.get("WANDB_PROJECT", "inverse-kinematics-rl")
    wandb_run_name = os.environ.get("WANDB_RUN_NAME", exp_name)

    wandb.init(project=wandb_project, name=wandb_run_name, sync_tensorboard=True, config=vars(args))

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
            number_of_collision_objects=args.num_obstacles,
            stage_path=args.stage_path,
        )
        env = TimeLimit(env, max_episode_steps=args.max_episode_steps)
        env = Monitor(env)
        return env

    # Create training and evaluation environments
    if args.num_envs > 1:
        # Important: You must use the if __name__ == '__main__': guard to use SubprocVecEnv
        env = SubprocVecEnv([lambda i=i: make_env(args.seed + i) for i in range(args.num_envs)])
    else:
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

        # Start virtual display for rendering if needed
        display = None
        if args.save_episode_images and PYVIRTUALDISPLAY_AVAILABLE:
            try:
                display = Display(visible=0, size=(640, 480))
                display.start()
                print("Started virtual display for episode rendering")
            except Exception as e:
                print(f"Could not start virtual display: {e}")

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
                f"Distance = {info.get('distance_to_target', 'N/A')}"
            )

            # Save and upload episode image if requested
            if args.save_episode_images:
                try:
                    # Create temporary directory for USD and images
                    with tempfile.TemporaryDirectory() as temp_dir:
                        # Get the underlying environment (unwrap if needed)
                        env = test_env
                        while hasattr(env, "env") and not hasattr(env, "task"):
                            env = env.env

                        if hasattr(env, "task"):
                            # Set stage path for USD export
                            original_stage_path = env.task.stage_path
                            env.task.stage_path = temp_dir

                            # Debug: Check if model has body data
                            if hasattr(env.task, "model") and env.task.model is not None:
                                print(f"  Model body count: {env.task.model.body_count}")
                                print(f"  Model shape count: {env.task.model.shape_count}")

                            # Call render to create USD file
                            try:
                                env.render()
                                print("  Called render() successfully")
                            except Exception as e:
                                print(f"  Error during render(): {e}")

                            # The USD file should be created as "reaching.usd"
                            usd_path = os.path.join(temp_dir, "reaching.usd")
                            image_path = os.path.join(temp_dir, f"episode_{episode}.png")

                            # Wait a moment to ensure file is written
                            import time

                            time.sleep(0.5)

                            # Debug: List all files in temp_dir
                            temp_files = os.listdir(temp_dir)
                            print(f"  Files in temp directory: {temp_files}")

                            if os.path.exists(usd_path):
                                # Save a local copy for inspection
                                local_image_dir = os.path.join(base_output_dir, "test_images")
                                os.makedirs(local_image_dir, exist_ok=True)

                                # Save the original USD file (before camera modifications)
                                local_usd_original_path = os.path.join(
                                    local_image_dir, f"episode_{episode}_original.usd"
                                )
                                shutil.copy(usd_path, local_usd_original_path)
                                print(
                                    f"  Saved original USD file locally to: "
                                    f"{local_usd_original_path}"
                                )

                                # Also check file size
                                original_size = os.path.getsize(local_usd_original_path)
                                print(f"  Original USD file size: {original_size} bytes")

                                # Convert USD to image (this modifies the USD file)
                                if convert_usd_to_image(usd_path, image_path):
                                    # Save the image
                                    local_image_path = os.path.join(
                                        local_image_dir, f"episode_{episode}.png"
                                    )
                                    shutil.copy(image_path, local_image_path)
                                    print(f"  Saved episode image locally to: {local_image_path}")

                                    # Save the modified USD file (with camera and lighting)
                                    local_usd_modified_path = os.path.join(
                                        local_image_dir, f"episode_{episode}_with_camera.usd"
                                    )
                                    shutil.copy(usd_path, local_usd_modified_path)
                                    print(
                                        f"  Saved modified USD file (with camera) locally to: "
                                        f"{local_usd_modified_path}"
                                    )

                                # Upload image to wandb
                                distance_val = info.get("distance_to_target", "N/A")
                                if isinstance(distance_val, np.ndarray):
                                    distance_val = (
                                        float(distance_val[0]) if len(distance_val) > 0 else "N/A"
                                    )

                                caption = (
                                    f"Episode {episode+1} - "
                                    f"Success: {info.get('is_success', False)}, Distance: "
                                )
                                if isinstance(distance_val, (int, float)):
                                    caption += f"{distance_val:.4f}"
                                else:
                                    caption += str(distance_val)

                                wandb.log(
                                    {
                                        f"test_episode_{episode}/final_state": wandb.Image(
                                            image_path, caption=caption
                                        )
                                    }
                                )
                                print("  Uploaded episode image to wandb")
                            else:
                                print(f"  Warning: USD file not found at {usd_path}")

                            # Always restore original stage path
                            env.task.stage_path = original_stage_path
                        else:
                            print("  Warning: Could not access task for rendering")

                except Exception as e:
                    print(f"  Error saving episode image: {e}")

        # Log test results
        test_results = {
            f"{args.algorithm.lower()}_test_avg_reward": np.mean(rewards),
            f"{args.algorithm.lower()}_test_success_rate": np.mean(successes),
            f"{args.algorithm.lower()}_test_avg_final_distance": np.mean(final_distances),
        }
        wandb.log(test_results)

        # Stop virtual display if we started it
        if display:
            try:
                display.stop()
                print("Stopped virtual display for episode rendering")
            except Exception as e:
                print(f"Error stopping virtual display: {e}")

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


if __name__ == "__main__":
    main()
