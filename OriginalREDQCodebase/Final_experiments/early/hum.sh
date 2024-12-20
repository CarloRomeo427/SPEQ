#!/bin/bash

for env in 'Humanoid'; do
    for seed in 0 42; do
        python main_early_speq.py -env $env-v2 -epochs 300 -target_drop_rate 0.1 -layer_norm 1 -offline_frequency 10000 \
            -offline_epochs 200000 -utd_ratio_online 1 -exp_name BiasValidation -gpu_id 0 -method sac -seed $seed -early_with_bias
    done  
done

