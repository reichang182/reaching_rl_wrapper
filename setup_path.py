"""Setup Python path to include co_design_task.

Import this at the beginning of scripts to ensure co_design_task is available.
"""

import os
import sys

# Get the parent directory that contains co_design_task
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
co_design_task_dir = os.path.join(parent_dir, "co_design_task")

# Add to Python path if not already there
if co_design_task_dir not in sys.path:
    sys.path.insert(0, co_design_task_dir)

# Also add the parent directory itself
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

print(f"Added to Python path: {co_design_task_dir}")
