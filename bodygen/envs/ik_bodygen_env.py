import numpy as np
from typing import List

# Import the RL wrapper for the co_design_task environment
# Local import of the RL wrapper for the co_design_task environment
from envs.ik_gym_wrapper import InverseKinematicsEnv


class IKBodyGenEnv:
    """Simplified BodyGen environment for the co_design_task wrapper."""

    def __init__(self, cfg, agent=None):
        self.cfg = cfg
        self.agent = agent

        env_specs = getattr(cfg, "env_specs", {})
        dof = env_specs.get("dof", 3)
        num_targets = env_specs.get("number_of_targets", 1)
        self.task_env = InverseKinematicsEnv(dof=dof, number_of_targets=num_targets)

        # Dimensions expected by BodyGen
        self.attr_fixed_dim = 0
        self.attr_design_dim = dof
        self.sim_obs_dim = self.task_env.observation_space.shape[0]
        self.control_action_dim = dof * num_targets
        self.skel_num_action = 1
        self.dof = dof

    # ------------------------------------------------------------------
    def reset(self):
        obs, _ = self.task_env.reset()
        return self._format_obs(obs)

    def seed(self, seed: int):
        # Forward seed to the underlying environment for reproducibility
        self.task_env.reset(seed=seed)

    def step(self, action: np.ndarray):
        # Forward the action directly to the wrapped environment
        obs, reward, term, trunc, info = self.task_env.step(action)
        done = term or trunc
        return self._format_obs(obs), reward, done, trunc, info

    # ------------------------------------------------------------------
    def _format_obs(self, obs: np.ndarray) -> List[np.ndarray]:
        """Return observation list expected by BodyGen."""
        num_nodes = self.dof + 1

        edges = np.zeros((2, 0), dtype=np.int64)
        body_ind = np.arange(num_nodes, dtype=np.int64)
        body_depths = np.zeros(num_nodes, dtype=np.float32)
        body_heights = np.zeros(num_nodes, dtype=np.float32)
        distances = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        lappe = np.zeros((num_nodes, 1), dtype=np.float32)
        use_transform_action = np.array([2], dtype=np.float32)
        num_nodes_arr = np.array([num_nodes], dtype=np.int64)

        return [
            obs.astype(np.float32),
            edges,
            use_transform_action,
            num_nodes_arr,
            body_ind,
            body_depths,
            body_heights,
            distances,
            lappe,
        ]
