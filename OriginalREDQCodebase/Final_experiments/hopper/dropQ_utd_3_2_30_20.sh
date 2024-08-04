#!/bin/bash
env="Hopper"
  for utd in 2 3 1; do
    for seed in 0 42 1234 23 666; do
        python main_o2.py -info lomo/$env/ -env $env-v2 -seed $seed -epochs 100 -gpu_id 0 -method sac -layer_norm 1 \
          -target_drop_rate 0.0001 -exp_name utd_$utd-dropQ -offline_frequency -1 -offline_epochs 0 -offline_dimension 0 -expectile 0.5 -utd_ratio_online $utd
    done
done

for seed in 0 42 1234 23 666; do
python main_o2.py -info lomo/$env/ -env $env-v2 -seed $seed -epochs 100 -gpu_id 0 -method sac -layer_norm 1 \
 -target_drop_rate 0.0001 -exp_name 30K_20K -offline_frequency 30000 -offline_epochs 20000  -expectile 0.5 -utd_ratio_online 1
done
