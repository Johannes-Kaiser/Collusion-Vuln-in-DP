#!/bin/bash

# Usage: ./run_budget_adv.sh total_start total_end num_jobs exp_yaml

set -e  # Exit immediately if a command exits with a non-zero status

export PYTHONUNBUFFERED=true

# Activate Python virtual environment
source ./.venv/bin/activate

# Read command-line arguments
total_start=$1
total_end=$2
num_jobs=$3
exp_yaml=$4

if [ -z "$total_start" ] || [ -z "$total_end" ] || [ -z "$num_jobs" ] || [ -z "$exp_yaml" ]; then
    echo "Usage: $0 total_start total_end num_jobs exp_yaml"
    exit 1
fi

# Compute chunk size (rounded up)
chunk_size=$(( (total_end - total_start + 1 + num_jobs - 1) / num_jobs ))

# Run Python scripts in parallel
for i in $(seq 0 $((num_jobs - 1))); do
    start=$(( total_start + i * chunk_size ))
    end=$(( start + chunk_size - 1 ))

    # Make sure the last chunk ends at total_end
    if [ $i -eq $((num_jobs - 1)) ]; then
        end=$total_end
    fi

    echo "Running indices $start to $end"
    /vol/miltank/users/kaiserj/Clipping_vs_Sampling/.venv/bin/python \
        scripts_experiments/mia/budget_control_adv.py \
        --exp_yaml "$exp_yaml" \
        --idx_start "$start" --idx_end "$end" &
done

# Wait for all background jobs to finish
wait

echo "All jobs finished."
