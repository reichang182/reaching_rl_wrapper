#!/bin/bash
# Script to run RL training with proper Python path

# Add co_design_task to PYTHONPATH
export PYTHONPATH="/storage/chinjui/co_design_task:$PYTHONPATH"

# Run the training script
python scripts/train_rl.py "$@"
