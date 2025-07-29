#!/bin/bash

#SBATCH --job-name=budg_adv
#SBATCH --output=./out/budg_adv-%A.out  # Standard output of the script (Can be absolute or relative path). %A adds the job id to the file name so you can launch the same script multiple times and get different logging files
#SBATCH --error=./out/budg_adv-%A.err   # Standard error of the script
#SBATCH --time=2-12:00:00          # Limit on the total run time (format: days-hours:minutes:seconds)
#SBATCH --gres=gpu:1               # Number of GPUs if needed
#SBATCH --cpus-per-task=24         # Number of CPUs (Don't use more than 24 per GPU)
#SBATCH --mem=10G                  # Memory in GB (Don't use more than 48GB per GPU unless you absolutely need it and know what you are doing)
#SBATCH --partition=universe
# # SBATCH --partition=eagle
# # SBATCH --account=eagle

export PYTHONUNBUFFERED=true

# load python module
source /vol/miltank/users/kaiserj/Indivdiual_Privacy_DPSGD_Evaluation/.venv/bin/activate

/vol/miltank/users/kaiserj/Clipping_vs_Sampling/.venv/bin/python \
    scripts_experiments/mia/budget_control_adv.py \
    --exp_yaml ./scripts_experiments/mia/exp_yaml/mnist_4_budget_adv.yaml \
    --idx_start $1 --idx_end $2