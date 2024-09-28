#!/bin/bash
for env in 'Humanoid'; do
    for seed in 0 ; do
        python main_o2.py -info lomo/$env/ -env $env-v2 -seed $seed -epochs 300000 -gpu_id 1 -method sac -layer_norm 1 \
          -target_drop_rate 0.1 -exp_name td_error -offline_frequency 10000 -offline_epochs 75000  -expectile 0.5 -evaluate_bias -evaluate_td -utd_ratio_online 1
    done  
done
