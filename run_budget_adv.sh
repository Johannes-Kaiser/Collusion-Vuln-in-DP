#!/bin/bash

#SBATCH --job-name=budg_adv
#SBATCH --output=./out/budg_adv-%A.out  # Standard output of the script (Can be absolute or relative path). %A adds the job id to the file name so you can launch the same script multiple times and get different logging files
#SBATCH --error=./out/budg_adv-%A.err   # Standard error of the script
#SBATCH --time=2-12:00:00          # Limit on the total run time (format: days-hours:minutes:seconds)
#SBATCH --gres=gpu:1               # Number of GPUs if needed
#SBATCH --cpus-per-task=36         # Number of CPUs (Don't use more than 24 per GPU)
#SBATCH --mem=80G                  # Memory in GB (Don't use more than 48GB per GPU unless you absolutely need it and know what you are doing)
#SBATCH --partition=universe
# #SBATCH --partition=eagle
# #SBATCH --account=eagle

export PYTHONUNBUFFERED=true

# load python module
source /vol/miltank/users/kaiserj/Indivdiual_Privacy_DPSGD_Evaluation/.venv/bin/activate

# Split the range between idx_start and idx_end into 10 chunks and run in parallel
total_start=$1
total_end=$2
num_jobs=$3
exp_yaml=$4

# Compute chunk size (rounded up)
chunk_size=$(( (total_end - total_start + 1 + num_jobs - 1) / num_jobs ))

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