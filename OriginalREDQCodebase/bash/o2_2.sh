#!/bin/bash
for env in 'Humanoid'; do
    for seed in 0 42 1234; do
        python main_o2.py -info drq -env $env-v2 -seed $seed -eval_every 1000 -epochs 100 \
        -frames 100000 -eval_runs 10 -gpu_id 0 -updates_per_step 20 -method sac -layer_norm 1 -target_drop_rate 0.1
    done  
done