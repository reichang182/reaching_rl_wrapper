"""Default configuration values for InverseKinematicsTask"""

import numpy as np


def get_default_task_config():
    """Get default task configuration with all required parameters."""
    return {
        "degrees_of_freedom": 3,
        "number_of_targets": 1,
        "number_of_instances": 1,
        "grid_offset": 1.0,
        "table_size_x": 1.0,
        "table_size_y": 0.01,
        "table_size_z": 1.0,
        "robot_base_height": 0.05,
        "default_link_length": 0.15,
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
    }
