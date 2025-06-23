#!/bin/bash
# Script to run training in background with proper logging

# Set PYTHONPATH
export PYTHONPATH=/storage/chinjui/co_design_task:$PYTHONPATH

# Create logs directory if it doesn't exist
mkdir -p training_logs

# Get current timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Parse algorithm from command line (default to SAC)
ALGO=${1:-SAC}
TIMESTEPS=${2:-1000000}

echo "Starting $ALGO training for $TIMESTEPS timesteps..."
echo "Logs will be saved to: training_logs/${ALGO}_${TIMESTAMP}.log"

# Run training in background with nohup
nohup python scripts/train_rl.py \
    --algorithm $ALGO \
    --total-timesteps $TIMESTEPS \
    --eval-freq 10000 \
    --n-eval-episodes 10 \
    --seed 42 \
    > training_logs/${ALGO}_${TIMESTAMP}.log 2>&1 &

# Get the process ID
PID=$!
echo "Training started with PID: $PID"
echo "You can monitor progress with: tail -f training_logs/${ALGO}_${TIMESTAMP}.log"
echo "To stop training: kill $PID"
