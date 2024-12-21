#!/bin/bash
for env in 'Hopper'; do
    for seed in 23 666; do
        python main_o2.py -info lomo/$env/ -env $env-v2 -seed $seed -epochs 1000 -gpu_id 0 -method redq -layer_norm 1 \
          -target_drop_rate 0 -exp_name vanilla_redQ_1M -offline_frequency -1 -offline_epochs 0 -offline_dimension 0 \
          -utd_ratio_online 20 -expectile 0.5 -evaluate_bias

        python main_o2.py -info lomo/$env/ -env $env-v2 -seed $seed -epochs 1000 -gpu_id 0 -method sac -layer_norm 1 \
          -target_drop_rate 0.0001 -exp_name vanilla_dropQ_1M -offline_frequency -1 -offline_epochs 0 -offline_dimension 0 \
          -utd_ratio_online 20 -expectile 0.5 -evaluate_bias

        python main_o2.py -info lomo/$env/ -env $env-v2 -seed $seed -epochs 1000 -gpu_id 0 -method sac -layer_norm 1 \
          -target_drop_rate 0.0001 -exp_name 10K_75K_dropq_1M -offline_frequency 10000 -offline_epochs 75000 -offline_dimension 0 \
          -utd_ratio_online 1 -expectile 0.5 -evaluate_bias

        python main_o2.py -info lomo/$env/ -env $env-v2 -seed $seed -epochs 1000 -gpu_id 0 -method sac -layer_norm 1 \
          -exp_name sac_1_vanilla_1M -offline_frequency -1 -offline_epochs 0 -offline_dimension 0 \
          -utd_ratio_online 1 -expectile 0.5 -evaluate_bias

    done  
done
