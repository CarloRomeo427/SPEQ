#!/bin/bash
for env in 'Hopper'; do
    for seed in 0 42 1234 5678 9876; do
        python main_o2.py -info onlOff -env $env-v2 -seed $seed -epochs 100 -gpu_id 0 -method sac -layer_norm 1 \
          -target_drop_rate 0.0001 -exp_name noutdnopolicy_bclessOften -offline_frequency 5000 -offline_epochs 100 -offline_dimension 5000 -expectile 0.6 \
          -utd_ratio_offline 1 -policy_type bc
    done  
done
