"""BodyGen compatible environment wrapper."""

from typing import Any, ClassVar, Dict, Optional

import gymnasium as gym
import numpy as np

from transform2act.envs.transform2act_ik_env import Transform2ActIKEnv


class IGEnv(Transform2ActIKEnv, gym.Env):
    """Gymnasium wrapper around :class:`Transform2ActIKEnv` used by BodyGen."""

    metadata: ClassVar[dict[str, Any]] = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(
        self,
        cfg: Dict[str, Any],
        random_seed: int = 0,
        init_xml: Optional[str] = None,
    ) -> None:
        """Initialize the environment.

        Parameters
        ----------
        cfg:
            Configuration dictionary understood by :class:`Transform2ActIKEnv`.
        random_seed:
            Seed for reproducibility.
        init_xml:
            Optional XML string used by the base class (ignored for IK task).

        """
        super().__init__(cfg, random_seed=random_seed, init_xml=init_xml)

        # Expose additional dimensions required by BodyGen
        self.sim_obs_dim = self._get_sim_obs().shape[-1]
        self.attr_fixed_dim = self._get_attr_fixed().shape[-1]
        self.attr_design_dim = self._get_attr_design().shape[-1]
        self.control_action_dim = self.current_links
        self.skel_num_action = 3

    def _map_stage(self, stage: str) -> str:
        """Convert internal stage name to BodyGen's stage name."""
        mapping = {
            "skel_trans": "skeleton_transform",
            "attr_trans": "attribute_transform",
            "execution": "execution",
        }
        return mapping.get(stage, stage)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment and return the initial observation."""
        if seed is not None:
            np.random.seed(seed)
        obs = self.reset_model()
        info = {
            "use_transform_action": self.use_transform_action,
            "stage": self._map_stage(self.stage),
        }
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take a step using an action from the agent."""
        obs, reward, done, info = super().step(action)
        info.update(
            {
                "use_transform_action": self.use_transform_action,
                "stage": self._map_stage(self.stage),
            }
        )
        return obs, reward, done, False, info

    # ------------------------------------------------------------------
    # Convenience methods for BodyGen compatibility
    # ------------------------------------------------------------------
    def if_use_transform_action(self) -> int:
        """Return the index of the current stage used by BodyGen agents."""
        return ["skel_trans", "attr_trans", "execution"].index(self.stage)

    def get_attr_fixed(self) -> np.ndarray:
        """Return fixed attributes for each link."""
        return self._get_attr_fixed().reshape(self.current_links + 1, -1)

    def get_sim_obs(self) -> np.ndarray:
        """Return simulation observations for each link."""
        return self._get_sim_obs().reshape(self.current_links, -1)

    def get_attr_design(self) -> np.ndarray:
        """Return design attributes for each link."""
        return self._get_attr_design().reshape(self.current_links, -1)
