#!/bin/bash
for env in 'Ant'; do
    for seed in 0 42 1234 23 666; do
        python main_o2.py -info lomo/$env/ -env $env-v2 -seed $seed -epochs 300 -gpu_id 0 -method sac -layer_norm 1 \
          -target_drop_rate 0.01 -exp_name 30K_20K_drop_bias_fix -offline_frequency 30000 -offline_epochs 20000 -expectile 0.5 -utd_ratio_online 1 -evaluate_bias
    done  
done
