#!/bin/bash
for env in 'Ant'; do
    for seed in 0 42 1234 23 666; do
        python main_o2.py -info lomo/$env/ -env $env-v2 -seed $seed -epochs 300 -gpu_id 0 -method sac -layer_norm 1 \
          -target_drop_rate 0.01 -exp_name 10K_200K_drop -offline_frequency 10000 -offline_epochs 200000 -expectile 0.5 -utd_ratio_online 1 -evaluate_bias
    done  
done
