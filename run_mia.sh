#!/bin/bash

#SBATCH --job-name=mia
#SBATCH --output=./out/mia-%A.out  # Standard output of the script (Can be absolute or relative path). %A adds the job id to the file name so you can launch the same script multiple times and get different logging files
#SBATCH --error=./out/mia-%A.err   # Standard error of the script
#SBATCH --time=1-22:00:00          # Limit on the total run time (format: days-hours:minutes:seconds)
#SBATCH --gres=gpu:1               # Number of GPUs if needed
#SBATCH --cpus-per-task=24         # Number of CPUs (Don't use more than 24 per GPU)
#SBATCH --mem=80G                  # Memory in GB (Don't use more than 48GB per GPU unless you absolutely need it and know what you are doing)
#SBATCH --partition=universe
#SBATCH --partition=eagle
#SBATCH --account=eagle

export PYTHONUNBUFFERED=true

# load python module
source /vol/miltank/users/kaiserj/Indivdiual_Privacy_DPSGD_Evaluation/.venv/bin/activate

MAX_PARALLEL=$3
pids=()

for seed in {1..5}; do
    /vol/miltank/users/kaiserj/Clipping_vs_Sampling/.venv/bin/python \
        scripts_experiments/mia/mia.py \
        --seed $seed --exp_yaml $1 --individualize $2 --name_ext $4 &

    pids+=($!)

    # If we have reached max parallel jobs, wait for them to finish
    if [ ${#pids[@]} -ge $MAX_PARALLEL ]; then
        wait -n   # wait for *any* to finish
        # remove finished PIDs from array
        new_pids=()
        for pid in "${pids[@]}"; do
            if kill -0 $pid 2>/dev/null; then
                new_pids+=($pid)
            fi
        done
        pids=("${new_pids[@]}")
    fi
done

# wait for all remaining
wait