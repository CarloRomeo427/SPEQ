#!/bin/bash



for env in 'Humanoid'; do
    ### pi vs Q
    for seed in 0; do
        echo "Running Policy: default"
        python main_o2.py -info lomo/$env/ -env $env-v2 -seed $seed -epochs 10 -gpu_id 0 -method sac -layer_norm 1 \
          -target_drop_rate 0.1 -exp_name DEBUG -offline_frequency 10000 -offline_epochs 75000 -offline_dimension 0 -expectile 0.5 \
          -utd_ratio_online 1 -evaluate_bias -policy_type default
    done
done