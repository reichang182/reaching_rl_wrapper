import os
import sys

# Add co_design_task to path
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import gymnasium
import numpy as np
import warp as wp
import warp.sim
import warp.sim.render

# Import from co_design_task package
from co_design_task.context.inverse_kinematics_context_manager import (
    InverseKinematicsContextManager,
)
from gymnasium import spaces


@wp.kernel
def broadcast_joint_poses_kernel(
    dest_joint_poses: wp.array(dtype=wp.float32),
    src_action: wp.array(dtype=wp.float32),  # shape (dof,)
    num_envs: int,
    num_targets: int,
    dof: int,
):
    # This kernel is launched with dim=(num_envs * num_targets)
    # Each thread handles one robot
    tid = wp.tid()
    env_idx = tid // num_targets
    target_idx = tid % num_targets

    for dof_idx in range(dof):
        dest_idx = env_idx * num_targets * dof + target_idx * dof + dof_idx
        dest_joint_poses[dest_idx] = src_action[dof_idx]


class InverseKinematicsEnv(gymnasium.Env):
    """Gymnasium wrapper for the InverseKinematicsTask from co_design_task."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    # Observation normalization bounds
    OBS_BOUNDS = {
        "joint_poses": (-np.pi, np.pi),
        "link_lengths": (0.05, 0.3),
        "positions": (-1.0, 1.0),  # workspace bounds
        "distance": (0.0, 2.0),  # max expected distance
    }

    def __init__(
        self,
        random_seed=0,
        dof=3,
        number_of_targets=1,
        num_envs=1,
        stage_path=".",
        integrator_type="euler",
        fps=30,
        sim_substeps=5,
        grid_offset=1.0,
        robot_base_height=0.05,
        default_link_length=0.2,
        default_link_radius=0.02,
        end_effector_sphere_radius=0.02,
        max_episode_steps=200,
        terminate_on_collision=False,
        collision_penalty=10.0,
        target_threshold=0.05,
        machine_cost_weight=0.0,
        enable_obstacles=False,
        number_of_collision_objects=0,
    ):
        super().__init__()

        self.device = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"

        # Build config dict for InverseKinematicsContextManager
        self.config = {
            "general": {
                "stage_path": stage_path,
                "random_seed": random_seed,
                "device": self.device,
            },
            "simulation": {
                "integrator_type": integrator_type,
                "sim_substeps": sim_substeps,
                "fps": fps,
            },
            "task": {
                "degrees_of_freedom": dof,
                "number_of_targets": number_of_targets,
                "number_of_instances": num_envs,
                "grid_offset": grid_offset,
                "table_size_x": 1.0,
                "table_size_y": 0.01,
                "table_size_z": 1.0,
                "robot_base_height": robot_base_height,
                "default_link_length": default_link_length,
                "default_link_radius": default_link_radius,
                "end_effector_sphere_radius": end_effector_sphere_radius,
                "enable_obstacles": enable_obstacles,
                "number_of_collision_objects": number_of_collision_objects,
                "requires_grad": False,  # Not needed for RL
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
                "link_length_static_values": [default_link_length] * dof,
                "collision": True,
                "collision_weight": 1.0,
                # Fixes from review
                "add_joint_shapes": True,
                "collision_object_lower_bounds": [-0.3, 0.1, -0.3],
                "collision_object_upper_bounds": [0.3, 0.4, 0.3],
                "collision_object_shared_random_position": False,
                "static_dh_parameters": [],
                "link_length_lower_bounds": [0.05] * dof,
                "link_length_upper_bounds": [0.3] * dof,
            },
        }

        self.max_episode_steps = max_episode_steps
        self.terminate_on_collision = terminate_on_collision
        self.collision_penalty = collision_penalty
        self.target_threshold = target_threshold
        self.machine_cost_weight = machine_cost_weight
        self.dof = dof
        self.num_envs = num_envs

        # Create the task using the context manager
        self.task = InverseKinematicsContextManager.create(self.config)

        # GPU buffers for actions
        num_actions = self.num_envs * self.config["task"]["number_of_targets"] * self.dof
        self._joint_pose_gpu = wp.zeros(num_actions, dtype=wp.float32, device=self.device)
        self._action_gpu = wp.empty(self.dof, dtype=wp.float32, device=self.device)

        # Define action space: joint angles
        # InverseKinematicsTask expects joint poses as actions
        action_low = -np.pi * np.ones(dof, dtype=np.float32)
        action_high = np.pi * np.ones(dof, dtype=np.float32)
        self.action_space = spaces.Box(low=action_low, high=action_high, dtype=np.float32)

        # Define observation space
        # Observation includes: [joint_poses, link_lengths, ee_pos, target_pos, relative_pos, distance]
        obs_dim = (
            dof + dof + 3 + 3 + 3 + 1
        )  # joint_poses + link_lengths + ee_pos + target_pos + relative_pos + distance
        obs_low = -1e5 * np.ones(obs_dim, dtype=np.float32)
        obs_high = 1e5 * np.ones(obs_dim, dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        self.current_step = 0
        self.prev_distance = None

        # Warn about multi-environment limitation
        if self.num_envs > 1:
            print(
                f"Warning: Created with {self.num_envs} environments but currently only first environment is used"
            )
            self.num_envs = 1  # Force single environment for now

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.prev_distance = None

        # Reset the task
        self.task.reset()

        # Get initial observation
        obs = self._get_observation()

        # Store initial distance for progress reward
        feature_states = self.task.get_feature_states()
        ee_pos = feature_states["end_effector_position"].numpy()[0]  # First environment
        target_pos = feature_states["target_position"].numpy()[0]
        self.prev_distance = np.linalg.norm(ee_pos - target_pos)

        return obs, {}

    def step(self, action):
        self.current_step += 1

        # Clip action to be within the defined action space
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Copy action to GPU and broadcast to all environments/targets via a kernel
        wp.copy(self._action_gpu, wp.from_numpy(action.astype(np.float32), device=self.device))
        wp.launch(
            kernel=broadcast_joint_poses_kernel,
            dim=self.num_envs * self.config["task"]["number_of_targets"],
            inputs=[
                self._joint_pose_gpu,
                self._action_gpu,
                self.num_envs,
                self.config["task"]["number_of_targets"],
                self.dof,
            ],
            device=self.device,
        )

        # Set joint poses in the task's feature_states
        wp.copy(self.task.feature_states["joint_poses"], self._joint_pose_gpu)

        # Step the task
        self.task.step()

        # Get observation
        obs = self._get_observation()

        # Compute reward
        reward, info = self._compute_reward()

        # Check termination conditions
        terminated = False
        truncated = False

        # Check collision
        feature_states = self.task.get_feature_states()
        if "collisions" in feature_states:
            collision_count = np.sum(feature_states["collisions"].numpy())
            if self.terminate_on_collision and collision_count > 0:
                terminated = True

        # Check success
        distance = info["distance_to_target"]
        if distance < self.target_threshold:
            reward += 50.0  # Large success bonus
            terminated = True
            info["is_success"] = True
        else:
            info["is_success"] = False

        # Check max steps
        if self.current_step >= self.max_episode_steps:
            truncated = True

        # Update previous distance for next step
        self.prev_distance = distance

        return obs, reward, terminated, truncated, info

    def _get_observation(self):
        """Extract observation from task feature states."""
        feature_states = self.task.get_feature_states()

        # Validate required feature states exist
        required_features = [
            "joint_poses",
            "link_lengths",
            "end_effector_position",
            "target_position",
        ]
        for feature in required_features:
            if feature not in feature_states:
                raise KeyError(f"Required feature '{feature}' not found in task feature_states")

        # Get states for first environment (convert to numpy for indexing)
        joint_poses = feature_states["joint_poses"].numpy()[: self.dof]
        link_lengths = feature_states["link_lengths"].numpy()[: self.dof]
        ee_pos = feature_states["end_effector_position"].numpy()[0]
        target_pos = feature_states["target_position"].numpy()[0]

        # Compute relative position and distance
        relative_pos = target_pos - ee_pos
        distance = np.linalg.norm(relative_pos)

        # Normalize observations for stable learning
        norm_joint_poses = joint_poses / np.pi  # [-1, 1]
        norm_link_lengths = (link_lengths - 0.175) / 0.125  # Normalize around mean
        norm_ee_pos = np.clip(ee_pos / 0.5, -2.0, 2.0)  # Workspace normalization
        norm_target_pos = np.clip(target_pos / 0.5, -2.0, 2.0)
        norm_relative_pos = np.clip(relative_pos / 0.5, -2.0, 2.0)
        norm_distance = np.clip(distance / 0.5, 0.0, 2.0)  # Normalized distance

        # Concatenate normalized observations
        obs = np.concatenate(
            [
                norm_joint_poses,
                norm_link_lengths,
                norm_ee_pos,
                norm_target_pos,
                norm_relative_pos,
                [norm_distance],
            ]
        )

        return obs.astype(np.float32)

    def _compute_reward(self):
        """Compute reward from task state."""
        feature_states = self.task.get_feature_states()

        # Get states for first environment (convert to numpy for indexing)
        ee_pos = feature_states["end_effector_position"].numpy()[0]
        target_pos = feature_states["target_position"].numpy()[0]

        # Distance reward with better shaping
        distance = np.linalg.norm(ee_pos - target_pos)

        # Exponential distance reward for better guidance near target
        if distance < 0.2:  # Within 20cm
            # Strong exponential reward when close
            reward = 1.0 - distance * 5.0  # Range: [0, 1] at 20cm to 0cm
        else:
            # Linear decay, continuous with the exponential part
            reward = 0.2 - distance

        # Progress reward (increased weight)
        if self.prev_distance is not None:
            progress = self.prev_distance - distance
            reward += 5.0 * progress  # Significant progress bonus

        # Machine cost penalty - disabled for initial training
        # Can be re-enabled once basic reaching is learned
        machine_cost = feature_states["machine_cost"].numpy()[0]

        # Collision penalty
        if "collisions" in feature_states:
            collision_count = np.sum(feature_states["collisions"].numpy())
            if collision_count > 0:
                reward -= self.collision_penalty * float(collision_count)

        # Velocity penalty for smoother movements (if available)
        # Currently disabled - can add joint velocity tracking later
        smoothness_penalty = 0.0

        info = {
            "distance_to_target": distance,
            "machine_cost": machine_cost,
            "smoothness_penalty": smoothness_penalty,
        }

        return reward, info

    def render(self):
        """Render the environment."""
        # InverseKinematicsTask might have its own rendering method
        if hasattr(self.task, "render"):
            self.task.render()

    def close(self):
        """Clean up resources."""
        pass
