#!/bin/bash
for env in 'Humanoid'; do
    for seed in 0 ; do
        python main_o2.py -info lomo/$env/ -env $env-v2 -seed $seed -epochs 300 -gpu_id 0 -method sac -layer_norm 1 \
          -target_drop_rate 0.01 -exp_name test_fast -offline_frequency 10000 -offline_epochs 75000 -offline_dimension 0 -expectile 0.5 -utd_ratio_online 1
    done  
done
