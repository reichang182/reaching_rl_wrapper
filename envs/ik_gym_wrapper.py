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


def make_dh_table(dof: int, link_length: float) -> list[list[float]]:
    """Return DH parameters so every successive joint axis is orthogonal.

    Joint 0 keeps a vertical yaw axis; all later links alternate pitch/yaw
    automatically because a = 90° for every link.
    """
    dh: list[list[float]] = []
    for _ in range(dof):
        dh.append([0.0, 0.0, link_length, np.pi / 2])  # [d, θ, r=a, a]
    return dh


def configure_task_dict(
    cfg: dict,
    dof: int,
    link_length: float,
    link_length_low: float,
    link_length_high: float,
) -> None:
    """Configure DOF-dependent parameters in the task dictionary."""
    cfg["task"].update(
        {
            "static_dh_parameters": make_dh_table(dof, link_length),
            "joint_pose_lower_bounds": [-np.pi] * dof,
            "joint_pose_upper_bounds": [np.pi] * dof,
            "link_length_static_values": [link_length] * dof,
            "link_length_lower_bounds": [link_length_low] * dof,
            "link_length_upper_bounds": [link_length_high] * dof,
        }
    )
    reach = dof * link_length
    h_min = cfg["task"]["robot_base_height"]
    h_max = h_min + reach
    reach_sphere = dof * link_length_high * 0.8  # 0.9  (safe radius)

    cube_half_edge = reach_sphere / np.sqrt(3)  # ≈0.52

    cfg["task"]["target_lower_bounds"] = [-cube_half_edge, h_min, -cube_half_edge]
    cfg["task"]["target_upper_bounds"] = [cube_half_edge, h_max, cube_half_edge]


class InverseKinematicsEnv(gymnasium.Env):
    """Gymnasium wrapper for the InverseKinematicsTask from co_design_task."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    # Observation normalization bounds
    OBS_BOUNDS = {
        "joint_poses": (-np.pi, np.pi),
        "dh_r": (0.05, 0.3),
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
        collision_penalty=0,
        target_threshold=0.01,
        machine_cost_weight=0.0,
        enable_obstacles=False,
        number_of_collision_objects=0,
        link_length_low=0.01,
        link_length_high=0.30,
    ):
        super().__init__()

        if dof < 3:
            raise ValueError("dof must be ≥ 3 to reach arbitrary 3-D points")

        self.device = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"

        # Build config dict for InverseKinematicsContextManager
        self.config = {
            "general": {
                "stage_path": stage_path,
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
                "joint_reset_lower": -np.pi,
                "joint_reset_upper": np.pi,
                # Additional required parameters
                "target_shared_random_position": False,
                "link_length_reset_handler": "random",
                "collision": False,
                "collision_weight": 1.0,
                # Fixes from review
                "add_joint_shapes": False,  # visual only
                "collision_object_lower_bounds": [-0.5, 0.0, -0.5],
                "collision_object_upper_bounds": [0.5, 0.3, 0.5],
                "collision_object_shared_random_position": False,
                "machine_cost_handler": "penalty",
                "machine_cost_weight": 1.0,
                "machine_cost_penalty_factor": 1.0,
                "reaching_cost_handler": "absolute",
            },
        }

        # Configure DOF-dependent parameters
        configure_task_dict(
            self.config, dof, default_link_length, link_length_low, link_length_high
        )

        self.max_episode_steps = max_episode_steps
        self.terminate_on_collision = terminate_on_collision
        self.collision_penalty = collision_penalty
        self.target_threshold = target_threshold
        self.machine_cost_weight = machine_cost_weight
        self.dof = dof
        self.num_envs = num_envs
        self.num_targets = number_of_targets

        # Create the task using the context manager
        self.task = InverseKinematicsContextManager.create(self.config)
        assert "dh_r" in self.task.feature_states, (
            "InverseKinematicsTask did not initialise correctly; "
            "check that the task-config is complete."
        )

        # Define action space: joint angles + link lengths
        angle_low = -np.pi * np.ones(self.dof * self.num_targets, dtype=np.float32)
        angle_high = np.pi * np.ones_like(angle_low)

        length_low = link_length_low * np.ones(self.dof, dtype=np.float32)
        length_high = link_length_high * np.ones(self.dof, dtype=np.float32)

        self.action_space = spaces.Box(
            low=np.concatenate([angle_low, length_low]),
            high=np.concatenate([angle_high, length_high]),
            dtype=np.float32,
        )

        # Define observation space
        # Observation includes: [joint_poses, dh_r, ee_pos, target_pos, relative_pos, distance]
        obs_dim = self.dof * self.num_targets + self.dof + 10 * self.num_targets
        obs_low = -1e5 * np.ones(obs_dim, dtype=np.float32)
        obs_high = 1e5 * np.ones(obs_dim, dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        self.prev_distance = None

        # Warn about multi-environment limitation
        if self.num_envs > 1:
            print(
                f"Warning: Created with {self.num_envs} environments but currently only first environment is used"
            )
            self.num_envs = 1  # Force single environment for now

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.prev_distance = None

        # 2) Keep Warp's own CUDA RNG in sync (optional but nice)
        if seed is not None:
            wp.rand_init(seed)

        # 3) Forward Gym's generator to the task
        #
        #    self.np_random is a `numpy.random.Generator` that Gym
        #    created (or re-seeded) for us in the line above.
        self.task.reset(rng=self.np_random)

        # Get initial observation
        obs = self._get_observation()

        return obs, {}

    def step(self, action):
        # Clip action to be within the defined action space
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Split action into angles and lengths
        angles = action[: self.dof * self.num_targets]
        lengths = action[self.dof * self.num_targets :]

        # Apply angles to joint poses
        joint_poses_all = np.tile(angles, self.num_envs)
        wp.copy(
            self.task.feature_states["joint_poses"],
            wp.from_numpy(
                joint_poses_all.astype(np.float32),
                device=self.task.feature_states["joint_poses"].device,
            ),
        )

        # Apply lengths to DH parameters (dh_r)
        link_lengths_all = np.tile(lengths, self.num_envs)
        wp.copy(
            self.task.feature_states["dh_r"],
            wp.from_numpy(
                link_lengths_all.astype(np.float32),
                device=self.task.feature_states["dh_r"].device,
            ),
        )

        # Step the task
        self.task.feature_states["machine_cost"].zero_()
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
        if (distance < self.target_threshold).all():
            reward += 50.0  # Large success bonus
            terminated = True
            info["is_success"] = True
        else:
            info["is_success"] = False

        # Update previous distance for next step
        self.prev_distance = distance

        return obs, reward, terminated, truncated, info

    def _get_observation(self):
        """Extract observation from task feature states."""
        feature_states = self.task.get_feature_states()

        # Validate required feature states exist
        required_features = [
            "joint_poses",
            "dh_r",
            "end_effector_position",
            "target_position",
        ]
        for feature in required_features:
            if feature not in feature_states:
                raise KeyError(f"Required feature '{feature}' not found in task feature_states")

        # --- raw tensors ------------------------------------------------------
        joint_poses = feature_states["joint_poses"].numpy()[: self.dof * self.num_targets]
        link_lengths = feature_states["dh_r"].numpy()[: self.dof]  # shared
        ee_pos = feature_states["end_effector_position"].numpy()[: self.num_targets]
        target_pos = feature_states["target_position"].numpy()[: self.num_targets]

        # --- derived per-target values ---------------------------------------
        relative_pos = target_pos - ee_pos
        distance = np.linalg.norm(relative_pos, axis=1)

        # --- normalisation (unchanged math, now vectorised) ------------------
        norm_joint_poses = joint_poses / np.pi  # [-1, 1]

        # Dynamic normalization for link lengths
        link_length_high = self.config["task"]["link_length_upper_bounds"][0]
        link_length_low = self.config["task"]["link_length_lower_bounds"][0]
        mean_len = (link_length_high + link_length_low) / 2.0
        half_range = (link_length_high - link_length_low) / 2.0
        norm_link_lengths = (link_lengths - mean_len) / half_range

        scale = 0.5
        norm_ee_pos = np.clip(ee_pos / scale, -2, 2).reshape(-1)
        norm_target_pos = np.clip(target_pos / scale, -2, 2).reshape(-1)
        norm_relative_pos = np.clip(relative_pos / scale, -2, 2).reshape(-1)
        norm_distance = np.clip(distance / scale, 0, 2)  # Normalized distance

        # --- pack -------------------------------------------------------------
        obs = np.concatenate(
            [
                norm_joint_poses,
                norm_link_lengths,
                norm_ee_pos,
                norm_target_pos,
                norm_relative_pos,
                norm_distance,
            ]
        )

        return obs.astype(np.float32)

    def _compute_reward(self):
        """Compute reward from task state."""
        feature_states = self.task.get_feature_states()

        # Get states for first environment (convert to numpy for indexing)
        ee_pos = feature_states["end_effector_position"].numpy()[: self.num_targets]
        target_pos = feature_states["target_position"].numpy()[: self.num_targets]

        # Distance reward with better shaping
        distances = np.linalg.norm(ee_pos - target_pos, axis=1)

        # Exponential distance reward for better guidance near target
        reward = -distances.mean()

        # Machine cost penalty - disabled for initial training
        # Can be re-enabled once basic reaching is learned
        machine_cost = feature_states["machine_cost"].numpy()[0]
        reward -= self.machine_cost_weight * machine_cost

        # Collision penalty
        # if "collisions" in feature_states:
        #     collision_count = np.sum(feature_states["collisions"].numpy())
        #     if collision_count > 0:
        #         reward -= self.collision_penalty * float(collision_count)

        info = {
            "distance_to_target": distances,
            "machine_cost": machine_cost,
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
