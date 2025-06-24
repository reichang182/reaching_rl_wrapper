#!/bin/bash
# Optimized training script for inverse kinematics RL

# SAC training with optimized hyperparameters
echo "Starting optimized SAC training..."
python scripts/train_rl.py \
    --algorithm SAC \
    --total-timesteps 1000000 \
    --eval-freq 10000 \
    --n-eval-episodes 20 \
    --target_threshold 0.05 \
    --collision_penalty 10.0 \
    --machine_cost_weight 0.0 \
    --seed 42

# Alternative: PPO training (if SAC doesn't converge well)
# python scripts/train_rl.py \
#     --algorithm PPO \
#     --total-timesteps 2000000 \
#     --eval-freq 20000 \
#     --n-eval-episodes 20 \
#     --target_threshold 0.05 \
#     --collision_penalty 10.0 \
#     --machine_cost_weight 0.0 \
#     --seed 42
