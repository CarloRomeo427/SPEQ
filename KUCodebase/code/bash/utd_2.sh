#!/bin/bash
for env in 'Hopper-v2'; do
    for seed in 0 42 1234 5678 9876; do
        for action_utd in 5 10; do
            python main_utd.py -info drq -env Hopper-v2 -seed $seed -eval_every 1000 -action_utd=$action_utd\
            -frames 100000 -eval_runs 10 -gpu_id 1 -updates_per_step 20 -method sac -target_entropy -1.0 -target_drop_rate 0.005 -layer_norm 1
        done
    done  
done