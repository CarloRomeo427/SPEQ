#!/bin/bash
for env in 'Walker2d'; do
    for seed in 0 42 1234; do
        python main_o2.py -info onlOff -env $env-v2 -seed $seed -epochs 300 -gpu_id 0 -method sac -layer_norm 1 \
          -target_drop_rate 0.005 -exp_name lomo -offline_frequency 10000 -offline_epochs 100000  -expectile 0.5 \
          -utd_ratio_offline 1 -policy_type None -offline_buffer full -utd_ratio_online 1
    done  
done
