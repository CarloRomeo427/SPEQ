#!/bin/bash

for env in 'Humanoid'; do

    ### ablation F = 5, 50, 100
    for seed in 23 666; do
        for freq in 5000 50000 100000; do
            # epochs_str="${epochs%000}K"
            freq_str="${freq%000}K"

            exp_name="${freq_str}_75K_dropq_1M"
            python main_o2.py -info lomo/$env/ -env $env-v2 -seed $seed -epochs 1000 -gpu_id 0 -method sac -layer_norm 1 \
            -target_drop_rate 0.1 -exp_name $exp_name -offline_frequency $freq -offline_epochs 75000 -offline_dimension 0 -expectile 0.5 \
            -utd_ratio_online 1 -evaluate_bias
        done
    done 

    ### pi vs Q
    for seed in 0 42 1234; do
        python main_o2.py -info lomo/$env/ -env $env-v2 -seed $seed -epochs 1000 -gpu_id 0 -method sac -layer_norm 1 \
          -target_drop_rate 0.1 -exp_name abl_pi_1M -offline_frequency 10000 -offline_epochs 75000 -offline_dimension 0 -expectile 0.5 \
          -utd_ratio_online 1 -evaluate_bias -policy_type only

        python main_o2.py -info lomo/$env/ -env $env-v2 -seed $seed -epochs 1000 -gpu_id 0 -method sac -layer_norm 1 \
          -target_drop_rate 0.1 -exp_name abl_q_pi_1M -offline_frequency 10000 -offline_epochs 75000 -offline_dimension 0 -expectile 0.5 \
          -utd_ratio_online 1 -evaluate_bias -policy_type default
    done
    
done
