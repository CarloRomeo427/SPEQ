#!/bin/bash
for env in 'Walker2d'; do
    for seed in 0 42 1234 23 666; do
        python main_o2.py -info lomo/$env/ -env $env-v2 -seed $seed -epochs 1000 -gpu_id 0 -method sac -layer_norm 0 \
          -target_drop_rate 0 -exp_name SAC -offline_frequency 0 -offline_epochs 0 -expectile 0.5 -utd_ratio_online 1 -evaluate_bias
    done  
done
