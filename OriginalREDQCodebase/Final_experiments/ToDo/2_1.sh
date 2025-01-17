#!/bin/bash
for env in 'Walker2d'; do
    ### 10_75
    for seed in 0 42 1234; do
        python main_o2.py -info lomo/$env/ -env $env-v2 -seed $seed -epochs 1000 -gpu_id 1 -method sac -layer_norm 1 \
          -target_drop_rate 0.005 -exp_name 10K_75K_drop_1M -offline_frequency 10000 -offline_epochs 75000 -offline_dimension 0 -expectile 0.5 \
          -utd_ratio_online 1 -evaluate_bias
    done  

    ### ablation DropQ UTD = 9, 3, 2
    for seed in 0 42 1234; do
        for utd in 9 3 2; do
            exp_name="drop_1M_utd_${utd}"
            python main_o2.py -info lomo/$env/ -env $env-v2 -seed $seed -epochs 1000 -gpu_id 1 -method sac -layer_norm 1 \
            -target_drop_rate 0.005 -exp_name $exp_name -offline_frequency -1  -offline_epochs 0 -offline_dimension 0 -expectile 0.5 \
            -utd_ratio_online $utd -evaluate_bias
        done
    done
done

for env in 'Ant'; do
    ### 10_75
    for seed in 0 42 1234; do
        python main_o2.py -info lomo/$env/ -env $env-v2 -seed $seed -epochs 1000 -gpu_id 1 -method sac -layer_norm 1 \
          -target_drop_rate 0.01 -exp_name 10K_75K_drop_1M -offline_frequency 10000 -offline_epochs 75000 -offline_dimension 0 -expectile 0.5 \
          -utd_ratio_online 1 -evaluate_bias
    done  

    ### ablation DropQ UTD = 9, 3, 2
    for seed in 0 42 1234; do
        for utd in 9 3 2; do
            exp_name="drop_1M_utd_${utd}"
            python main_o2.py -info lomo/$env/ -env $env-v2 -seed $seed -epochs 1000 -gpu_id 1 -method sac -layer_norm 1 \
            -target_drop_rate 0.01 -exp_name $exp_name -offline_frequency -1  -offline_epochs 0 -offline_dimension 0 -expectile 0.5 \
            -utd_ratio_online $utd -evaluate_bias
        done
    done
done

for env in 'Humanoid'; do

    # ### pi vs Q
    # for seed in 0 42 1234; do
    #     python main_o2.py -info lomo/$env/ -env $env-v2 -seed $seed -epochs 1000 -gpu_id 1 -method sac -layer_norm 1 \
    #       -target_drop_rate 0.1 -exp_name abl_pi_1M -offline_frequency 10000 -offline_epochs 75000 -offline_dimension 0 -expectile 0.5 \
    #       -utd_ratio_online 1 -evaluate_bias -policy_type only

    #     python main_o2.py -info lomo/$env/ -env $env-v2 -seed $seed -epochs 1000 -gpu_id 1 -method sac -layer_norm 1 \
    #       -target_drop_rate 0.1 -exp_name abl_q_pi_1M -offline_frequency 10000 -offline_epochs 75000 -offline_dimension 0 -expectile 0.5 \
    #       -utd_ratio_online 1 -evaluate_bias -policy_type default
    # done
    
done
