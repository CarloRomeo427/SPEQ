#!/bin/bash
for env in 'HalfCheetah'; do
    for seed in 0 42 1234 23 666; do
        python main_o2.py -info lomo/$env/ -env $env-v2 -seed $seed -epochs 100 -gpu_id 0 -method redq -layer_norm 1 \
          -target_drop_rate 0 -exp_name redq_10K_75K -offline_frequency 10000 -offline_epochs 75000 -offline_dimension 0 -expectile 0.5 -evaluate_bias
    done  
done
