#!/bin/bash
for env in 'Humanoid'; do
  for freq in  50000; do
    for seed in 0 42 1234; do
        python main_o2.py -info lomo/$env/ -env $env-v2 -seed $seed -epochs 300 -gpu_id 0 -method sac -layer_norm 1 \
          -target_drop_rate 0.1 -exp_name abl_freq_$freq-dropQ -offline_frequency $freq -offline_epochs 75000 -offline_dimension 0 -expectile 0.5 -utd_ratio_online 1 -evaluate_bias
    done
    done
done





