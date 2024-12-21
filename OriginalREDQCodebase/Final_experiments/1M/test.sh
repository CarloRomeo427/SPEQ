#!/bin/bash
for env in 'Ant'; do
    for seed in 23 666; do
        python main_o2.py -info lomo/$env/ -env $env-v2 -seed $seed -epochs 3 -gpu_id 0 -method redq -layer_norm 1 \
          -target_drop_rate 0 -exp_name test_red -offline_frequency -1 -offline_epochs 0 -offline_dimension 0 \
          -utd_ratio_online 20 -expectile 0.5 -evaluate_bias

        python main_o2.py -info lomo/$env/ -env $env-v2 -seed $seed -epochs 3 -gpu_id 0 -method sac -layer_norm 1 \
          -target_drop_rate 0.01 -exp_name test_drop -offline_frequency -1 -offline_epochs 0 -offline_dimension 0 \
          -utd_ratio_online 20 -expectile 0.5 -evaluate_bias

        python main_o2.py -info lomo/$env/ -env $env-v2 -seed $seed -epochs 3 -gpu_id 0 -method sac -layer_norm 1 \
          -target_drop_rate 0.01 -exp_name test_speq -offline_frequency 1000 -offline_epochs 750 -offline_dimension 0 \
          -utd_ratio_online 1 -expectile 0.5 -evaluate_bias -utd_ratio_offline 1

        python main_o2.py -info lomo/$env/ -env $env-v2 -seed $seed -epochs 3 -gpu_id 0 -method sac -layer_norm 1 \
          -exp_name test_sac -offline_frequency -1 -offline_epochs 0 -offline_dimension 0 \
          -utd_ratio_online 1 -expectile 0.5 -evaluate_bias

    done  
done
