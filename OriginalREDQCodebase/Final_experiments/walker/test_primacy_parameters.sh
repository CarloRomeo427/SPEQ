#!/bin/bash
for env in 'Walker2d'; do
    for seed in 0 42 1234 23 666; do
        python main_o2.py -info lomo/$env/ -env $env-v2 -seed $seed -epochs 1000 -gpu_id 0 -method sac -layer_norm 0 \
          -target_drop_rate 0 -exp_name test_primacy -offline_frequency 200000 -offline_epochs 10000 -expectile 0.5 -reset_q -utd_ratio_online 1 -evaluate_bias
    done  
done
