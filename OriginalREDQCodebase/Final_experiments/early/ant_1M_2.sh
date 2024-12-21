#!/bin/bash
for env in 'Ant'; do
    for seed in 1234 23; do
        python main_early_speq.py -env $env-v2 -epochs 1000 -target_drop_rate 0.01 -layer_norm 1 -offline_frequency 10000 \
            -offline_epochs 200000 -utd_ratio_online 1 -exp_name BiasValidation_1M_with_RB_Ext -gpu_id 1 -method sac -seed $seed -early_with_bias -evaluate_bias
    done  
done


