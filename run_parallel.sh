#!/bin/bash
set -e

# Do not use this in production. We are going to use a automator.py script.

for i in {1..4}
do
    python modules/ml_optuna_1.py > "iter_${i}.log" 2>&1 &
    sleep 2
done

# Wait for all background jobs to complete
wait

echo "All parallel Python scripts have completed."
