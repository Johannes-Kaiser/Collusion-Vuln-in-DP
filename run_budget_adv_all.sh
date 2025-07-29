#!/bin/bash

for idx_start in $(seq 7 12); do
    idx_end=$((idx_start + 10))
    echo "sbatch run_budget_adv.sh $idx_start $idx_end"
    sbatch run_budget_adv.sh $idx_start $idx_end
done
