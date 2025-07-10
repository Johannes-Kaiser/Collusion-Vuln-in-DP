#!/bin/bash

for ((i=1; i<5; i++)); do
    portion1=$(awk "BEGIN {print $i / 5}")
    portion2=$(awk "BEGIN {print 1 - ($i / 5)}")
    echo "i: $i, portion: $portion1, portion2: $portion2"

    # Skip if either portion is 0.0
    if [[ "$portion1" == "0.0" || "$portion2" == "0.0" ]]; then
        continue
    fi
    for seed in $(seq 0 4); do
        echo "sbatch run_mia.sh --portions $portion1 $portion2 --args.seed $seed"
        sbatch run_mia.sh $portion1 $portion2 $seed
    done
done
