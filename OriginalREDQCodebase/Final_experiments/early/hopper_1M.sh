#!/bin/bash
for env in 'Hopper'; do
    for seed in 0 42 1234 23 666; do
        python main_early_speq.py -env $env-v2 -epochs 1000 -target_drop_rate 0.0001 -layer_norm 1 -offline_frequency 10000 \
            -offline_epochs 200000 -utd_ratio_online 1 -exp_name BiasValidation_1M_with_RB_Ext \
            -gpu_id 0 -method sac -seed $seed -early_with_bias -evaluate_bias
    done  
done


