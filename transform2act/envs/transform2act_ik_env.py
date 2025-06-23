import os
import sys
from typing import Any, Dict, List, Optional, Tuple

# Add parent directories to path
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Also add co_design_task parent directory
co_design_parent = os.path.dirname(parent_dir)
if co_design_parent not in sys.path:
    sys.path.insert(0, co_design_parent)

import numpy as np
import warp as wp

# Import from co_design_task package
from co_design_task.context.inverse_kinematics_context_manager import (
    InverseKinematicsContextManager,
)


class Transform2ActIKEnv:
    """Transform2Act-compatible wrapper for the InverseKinematicsTask.

    This environment supports dynamic morphology changes through:
    - Skeleton transformations (add/remove links)
    - Attribute transformations (modify link lengths and joint limits)
    - Control actions (set joint positions)
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
        random_seed: int = 0,
        init_xml: Optional[str] = None,
    ):
        """Initialize the Transform2Act IK environment.

        Args:
        ----
            cfg: Configuration dictionary with environment parameters
            random_seed: Random seed for reproducibility
            init_xml: Initial XML file (not used for IK task)

        """
        self.cfg = cfg
        self.random_seed = random_seed

        # Extract configuration parameters
        self.min_links = cfg.get("min_links", 2)
        self.max_links = cfg.get("max_links", 5)
        self.current_links = cfg.get("init_links", 3)
        self.base_link_length = cfg.get("base_link_length", 0.2)
        self.link_length_range = cfg.get("link_length_range", [0.1, 0.5])
        self.joint_limit_range = cfg.get("joint_limit_range", [-np.pi, np.pi])

        # Reward configuration
        self.distance_weight = cfg.get("distance_weight", 1.0)
        self.success_bonus = cfg.get("success_bonus", 10.0)
        self.collision_penalty = cfg.get("collision_penalty", 5.0)
        self.machine_cost_weight = cfg.get("machine_cost_weight", 0.01)
        self.smoothness_weight = cfg.get("smoothness_weight", 0.1)

        # Episode configuration
        self.max_episode_steps = cfg.get("max_episode_steps", 100)
        self.target_threshold = cfg.get("target_threshold", 0.05)
        self.terminate_on_collision = cfg.get("terminate_on_collision", False)

        # Transform2Act specific
        self.stage = "execution"  # Can be "skel_trans", "attr_trans", or "execution"
        self.use_transform_action = False
        self.transform_freq = cfg.get("transform_freq", 0.1)
        self.attr_transform_scale = cfg.get("attr_transform_scale", 0.05)

        # Initialize morphology parameters
        self.link_lengths = np.full(self.current_links, self.base_link_length)
        self.joint_limits = np.tile(self.joint_limit_range, (self.current_links, 1))

        # Create the IK task
        self._create_task()

        # Episode tracking
        self.current_step = 0
        self.prev_distance = None
        self.episode_reward = 0.0

        # Graph representation
        self.use_graph = cfg.get("use_graph", True)

    def _create_task(self):
        """Create or recreate the IK task with current morphology."""
        # Build config for InverseKinematicsContextManager
        self.task_config = {
            "general": {
                "stage_path": ".",
                "random_seed": self.random_seed,
                "device": "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu",
            },
            "simulation": {"integrator_type": "euler", "sim_substeps": 5, "fps": 30},
            "task": {
                "degrees_of_freedom": self.current_links,
                "number_of_targets": 1,
                "number_of_instances": 1,
                "grid_offset": 1.0,
                "table_size_x": 1.0,
                "table_size_y": 0.01,
                "table_size_z": 1.0,
                "robot_base_height": 0.05,
                "default_link_length": self.base_link_length,
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
                "target_shared_random_position": False,
                "joint_pose_lower_bounds": self.joint_limits[:, 0].tolist(),
                "joint_pose_upper_bounds": self.joint_limits[:, 1].tolist(),
                "link_length_reset_handler": "static",
                "link_length_static_values": self.link_lengths.tolist(),
                "collision": True,
                "collision_weight": 1.0,
            },
        }

        # Create the task
        self.task = InverseKinematicsContextManager.create(self.task_config)

    def reset_model(self):
        """Reset the environment to initial state."""
        self.current_step = 0
        self.prev_distance = None
        self.episode_reward = 0.0

        # Decide if we should allow transformations this episode
        if np.random.random() < self.transform_freq:
            self.use_transform_action = True
            self.stage = "skel_trans"
        else:
            self.use_transform_action = False
            self.stage = "execution"

        # Reset the task
        self.task.reset()

        # Get initial observation
        obs = self._get_obs()

        # Store initial distance
        feature_states = self.task.get_feature_states()
        ee_pos = feature_states["end_effector_position"].numpy()[0]
        target_pos = feature_states["target_position"].numpy()[0]
        self.prev_distance = np.linalg.norm(ee_pos - target_pos)

        return obs

    def step(self, action):
        """Take a step in the environment.

        Args:
        ----
            action: Can be skeleton transformation, attribute transformation, or control action

        Returns:
        -------
            obs: Next observation
            reward: Step reward
            done: Episode termination flag
            info: Additional information

        """
        self.current_step += 1

        # Apply action based on current stage
        if self.stage == "skel_trans":
            self._apply_skel_action(action)
            self.stage = "attr_trans"
        elif self.stage == "attr_trans":
            self._apply_attr_action(action)
            self.stage = "execution"
            self.use_transform_action = False
        else:  # execution stage
            self._apply_control_action(action)

        # Get observation
        obs = self._get_obs()

        # Compute reward
        reward, info = self._compute_reward()
        self.episode_reward += reward

        # Check termination
        done = self._check_termination(info)

        # Update previous distance
        self.prev_distance = info["distance_to_target"]

        return obs, reward, done, info

    def _get_obs(self) -> List[Any]:
        """Get observation in Transform2Act format.

        Returns
        -------
            List containing:
            - observation_tensor: Concatenated features
            - edges: Graph edges (if using graph)
            - use_transform_action: Whether in transformation stage
            - num_nodes: Number of nodes
            - body_ind: Body indices (optional)

        """
        # Get fixed attributes
        attr_fixed = self._get_attr_fixed()

        # Get simulation observations
        sim_obs = self._get_sim_obs()

        # Get design attributes
        attr_design = self._get_attr_design()

        # Concatenate all observations
        obs_tensor = np.concatenate([attr_fixed, sim_obs, attr_design])

        # Get graph edges if using graph representation
        if self.use_graph:
            edges = self._get_graph_edges()
            num_nodes = self.current_links + 1  # Links + base
        else:
            edges = None
            num_nodes = 1

        return [
            obs_tensor.astype(np.float32),
            edges,
            self.use_transform_action,
            num_nodes,
            None,  # body_ind not used for IK task
        ]

    def _get_attr_fixed(self) -> np.ndarray:
        """Get fixed attributes for current morphology."""
        attr_fixed = []

        # Base node (depth 0)
        attr_fixed.append(
            [
                0,  # depth
                0,  # is_joint (base has no parent joint)
                0,  # joint_type (not applicable)
                0,  # joint_range_low
                0,  # joint_range_high
            ]
        )

        # Link nodes
        for i in range(self.current_links):
            attr_fixed.append(
                [
                    i + 1,  # depth
                    1,  # is_joint
                    0,  # joint_type (0 for revolute)
                    self.joint_limits[i, 0],  # joint_range_low
                    self.joint_limits[i, 1],  # joint_range_high
                ]
            )

        return np.concatenate(attr_fixed)

    def _get_sim_obs(self) -> np.ndarray:
        """Get simulation state observations."""
        feature_states = self.task.get_feature_states()

        # Get joint positions and velocities
        joint_poses = feature_states["joint_poses"].numpy()[: self.current_links]
        # Approximate velocities (would need to track previous positions in practice)
        joint_velocities = np.zeros_like(joint_poses)

        # Get end effector and target positions
        ee_pos = feature_states["end_effector_position"].numpy()[0]
        target_pos = feature_states["target_position"].numpy()[0]

        # Compute relative position and distance
        relative_pos = target_pos - ee_pos
        distance = np.linalg.norm(relative_pos)

        # Concatenate observations
        sim_obs = np.concatenate(
            [joint_poses, joint_velocities, ee_pos, target_pos, relative_pos, [distance]]
        )

        return sim_obs

    def _get_attr_design(self) -> np.ndarray:
        """Get design attributes (morphology parameters)."""
        # Include link lengths and radii
        link_radii = np.full(self.current_links, 0.02)  # Fixed radius for now

        attr_design = np.concatenate([self.link_lengths, link_radii])

        return attr_design

    def _get_graph_edges(self) -> Optional[np.ndarray]:
        """Get graph edges representing robot structure."""
        if not self.use_graph:
            return None

        # Serial chain: each link connected to previous
        edges = []
        for i in range(self.current_links):
            edges.append([i, i + 1])  # Connect to next link

        return np.array(edges, dtype=np.int64) if edges else None

    def _apply_skel_action(self, action: int):
        """Apply skeleton transformation action.

        Args:
        ----
            action: 0 = no change, 1 = add link, 2 = remove link

        """
        if action == 1 and self.current_links < self.max_links:
            # Add a link
            self.current_links += 1
            self.link_lengths = np.append(self.link_lengths, self.base_link_length)
            new_limits = np.array([self.joint_limit_range])
            self.joint_limits = np.vstack([self.joint_limits, new_limits])

            # Recreate task with new morphology
            self._create_task()
            self.task.reset()

        elif action == 2 and self.current_links > self.min_links:
            # Remove a link
            self.current_links -= 1
            self.link_lengths = self.link_lengths[:-1]
            self.joint_limits = self.joint_limits[:-1]

            # Recreate task with new morphology
            self._create_task()
            self.task.reset()

    def _apply_attr_action(self, action: np.ndarray):
        """Apply attribute transformation action.

        Args:
        ----
            action: Continuous values for modifying link lengths and joint limits

        """
        # Extract link length changes (first half of action)
        link_length_changes = action[: self.current_links] * self.attr_transform_scale

        # Update link lengths
        self.link_lengths = np.clip(
            self.link_lengths + link_length_changes,
            self.link_length_range[0],
            self.link_length_range[1],
        )

        # Update task with new link lengths
        self.task.feature_states["link_lengths"].numpy()[: self.current_links] = self.link_lengths

    def _apply_control_action(self, action: np.ndarray):
        """Apply control action (joint positions).

        Args:
        ----
            action: Target joint positions

        """
        # Clip action to joint limits
        clipped_action = np.clip(
            action[: self.current_links], self.joint_limits[:, 0], self.joint_limits[:, 1]
        )

        # Set joint poses in the task
        self.task.feature_states["joint_poses"].numpy()[: self.current_links] = clipped_action

        # Step the simulation
        self.task.step()

    def _compute_reward(self) -> Tuple[float, Dict[str, Any]]:
        """Compute reward and info dictionary."""
        feature_states = self.task.get_feature_states()

        # Get current state
        ee_pos = feature_states["end_effector_position"].numpy()[0]
        target_pos = feature_states["target_position"].numpy()[0]

        # Distance reward
        distance = np.linalg.norm(ee_pos - target_pos)
        reward = -self.distance_weight * distance

        # Progress reward
        if self.prev_distance is not None:
            progress = self.prev_distance - distance
            reward += 0.1 * progress

        # Success bonus
        if distance < self.target_threshold:
            reward += self.success_bonus

        # Machine cost penalty
        machine_cost = feature_states["machine_cost"].numpy()[0]
        reward -= self.machine_cost_weight * machine_cost

        # Collision penalty
        collision_count = 0
        if "collisions" in feature_states:
            collision_count = np.sum(feature_states["collisions"].numpy())
            if collision_count > 0:
                reward -= self.collision_penalty * collision_count

        # Smoothness penalty
        link_diff = np.diff(self.link_lengths)
        smoothness_penalty = np.sum(link_diff**2) * self.smoothness_weight
        reward -= smoothness_penalty

        info = {
            "distance_to_target": distance,
            "machine_cost": machine_cost,
            "collision_count": collision_count,
            "smoothness_penalty": smoothness_penalty,
            "is_success": distance < self.target_threshold,
            "episode_reward": self.episode_reward + reward,
        }

        return reward, info

    def _check_termination(self, info: Dict[str, Any]) -> bool:
        """Check if episode should terminate."""
        # Success
        if info["is_success"]:
            return True

        # Collision
        if self.terminate_on_collision and info["collision_count"] > 0:
            return True

        # Max steps
        return self.current_step >= self.max_episode_steps

    def get_reward(self, info: Optional[Dict[str, Any]] = None) -> float:
        """Get reward (for compatibility with Transform2Act)."""
        if info is None:
            reward, _ = self._compute_reward()
            return reward
        return info.get("reward", 0.0)

    @property
    def action_space(self):
        """Get current action space based on stage."""
        if self.stage == "skel_trans":
            return 3  # Discrete: no change, add link, remove link
        elif self.stage == "attr_trans":
            return self.current_links * 2  # Link lengths + joint limits
        else:  # execution
            return self.current_links  # Joint positions

    @property
    def observation_space(self):
        """Get observation space size."""
        # Fixed attrs + sim obs + design attrs
        fixed_size = (self.current_links + 1) * 5
        sim_size = self.current_links * 2 + 3 + 3 + 3 + 1
        design_size = self.current_links * 2
        return fixed_size + sim_size + design_size
