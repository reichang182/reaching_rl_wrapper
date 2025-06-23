import os
import sys

import gymnasium
import numpy as np
import warp as wp
import warp.sim
import warp.sim.render
from gymnasium import spaces

# Add co_design_task to path
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import from co_design_task package
from co_design_task.context.inverse_kinematics_context_manager import (
    InverseKinematicsContextManager,
)


class InverseKinematicsEnv(gymnasium.Env):
    """Gymnasium wrapper for the InverseKinematicsTask from co_design_task."""

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
        collision_penalty=1.0,
        target_threshold=0.03,
        machine_cost_weight=0.01,
        enable_obstacles=False,
        number_of_obstacles=0,
    ):
        super().__init__()

        # Build config dict for InverseKinematicsContextManager
        self.config = {
            "general": {
                "stage_path": stage_path,
                "random_seed": random_seed,
                "device": "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu",
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
                "number_of_obstacles": number_of_obstacles,
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

        # Apply action to the task
        # InverseKinematicsTask expects joint poses to be set in feature_states
        # For single target, just use the action directly
        # For multiple targets, repeat the action
        joint_poses_all = np.tile(action, self.config["task"]["number_of_targets"])

        # Set joint poses in the task's feature_states
        self.task.feature_states["joint_poses"].numpy()[:] = joint_poses_all

        # Step the task
        feature_states_dict = self.task.step()

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
            reward += 10.0  # Success bonus
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

        # Get states for first environment (convert to numpy for indexing)
        joint_poses = feature_states["joint_poses"].numpy()[: self.dof]
        link_lengths = feature_states["link_lengths"].numpy()[: self.dof]
        ee_pos = feature_states["end_effector_position"].numpy()[0]
        target_pos = feature_states["target_position"].numpy()[0]

        # Compute relative position and distance
        relative_pos = target_pos - ee_pos
        distance = np.linalg.norm(relative_pos)

        # Concatenate all observations
        obs = np.concatenate(
            [joint_poses, link_lengths, ee_pos, target_pos, relative_pos, [distance]]
        )

        return obs.astype(np.float32)

    def _compute_reward(self):
        """Compute reward from task state."""
        feature_states = self.task.get_feature_states()

        # Get states for first environment (convert to numpy for indexing)
        ee_pos = feature_states["end_effector_position"].numpy()[0]
        target_pos = feature_states["target_position"].numpy()[0]
        link_lengths = feature_states["link_lengths"].numpy()[: self.dof]

        # Distance reward (primary objective)
        distance = np.linalg.norm(ee_pos - target_pos)
        reward = -distance

        # Progress reward (optional)
        if self.prev_distance is not None:
            progress = self.prev_distance - distance
            reward += 0.1 * progress  # Small progress bonus

        # Machine cost penalty
        machine_cost = feature_states["machine_cost"].numpy()[0]
        reward -= self.machine_cost_weight * machine_cost

        # Collision penalty
        if "collisions" in feature_states:
            collision_count = np.sum(feature_states["collisions"].numpy())
            if collision_count > 0:
                reward -= self.collision_penalty * float(collision_count)

        # Link smoothness penalty - encourage gradual transitions
        link_diff = np.diff(link_lengths)
        smoothness_penalty = np.sum(link_diff**2) * 0.1
        reward -= smoothness_penalty

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
