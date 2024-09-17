#!/bin/bash
for env in 'Ant'; do
    for freq in 50000 30000; do
        for epochs in 20000 10000 50000; do
            for seed in 0 42 1234 23 666; do
                # Convert numbers to strings with K notation
                epochs_str="${epochs%000}K"
                freq_str="${freq%000}K"

                # Create the exp_name
                exp_name="${freq_str}_${epochs_str}-redQ"

                # Run the training script
               python main_o2.py -info lomo/$env/ -env $env-v2 -seed $seed -epochs 300 -gpu_id 1 -method redq -layer_norm 1 \
              -target_drop_rate 0.0 -exp_name $exp_name -offline_frequency $freq -offline_epochs $epochs  -expectile 0.5 -utd_ratio_online 1
            done
        done
    done
done