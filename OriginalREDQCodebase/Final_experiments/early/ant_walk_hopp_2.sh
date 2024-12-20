#!/bin/bash

for env in 'Hopper'; do
    for seed in 0 42; do
        python main_early_speq.py -env $env-v2 -epochs 300 -target_drop_rate 0.0001 -layer_norm 1 -offline_frequency 10000 \
            -offline_epochs 200000 -utd_ratio_online 1 -exp_name BiasValidation -gpu_id 1 -method sac -seed $seed -early_with_bias
    done  
done

for env in 'Ant'; do
    for seed in 0 42; do
        python main_early_speq.py -env $env-v2 -epochs 300 -target_drop_rate 0.01 -layer_norm 1 -offline_frequency 10000 \
            -offline_epochs 200000 -utd_ratio_online 1 -exp_name BiasValidation -gpu_id 1 -method sac -seed $seed -early_with_bias
    done  
done

for env in 'Walker2d'; do
    for seed in 0 42; do
        python main_early_speq.py -env $env-v2 -epochs 300 -target_drop_rate 0.005 -layer_norm 1 -offline_frequency 10000 \
            -offline_epochs 200000 -utd_ratio_online 1 -exp_name BiasValidation -gpu_id 1 -method sac -seed $seed -early_with_bias
    done  
done

