#!/bin/bash
set -e

# A Script to run multiple optuna automators at the same time. Can be run multiple times in order to start processes in batches.
# Do not use this in production. We are going to use a automator.py script.

PYTHONPATH="$(pwd)"
export PYTHONPATH

num_processes=$1
echo "Running '${num_processes}' processes..."

for ((i=1; i<=num_processes; i++))
do
    python modules/ml_optuna_1.py > "logs/process_${i}.log" 2>&1 &
    sleep 2
done

# Wait for all background jobs to complete
wait

echo "All parallel Python processes have completed."
