#!/bin/bash
for env in 'Walker2d'; do
  for utd in 1 2 3; do
    for seed in 0 42 1234 23 666; do
           python main_o2.py -info lomo/$env/ -env $env-v2 -seed $seed -epochs 300 -gpu_id 0 -method redq -layer_norm 1 \
          -target_drop_rate 0 -exp_name utd_$utd-redQ -offline_frequency -1 -offline_epochs 0 -offline_dimension 0 -expectile 0.5-utd_ratio_online $utd
    done
    done
done
