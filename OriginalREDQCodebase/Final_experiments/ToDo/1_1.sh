#!/bin/bash
for env in 'Hopper'; do
    ### 10_75
    for seed in 0 42 1234; do
        python main_o2.py -info lomo/$env/ -env $env-v2 -seed $seed -epochs 1000 -gpu_id 0 -method sac -layer_norm 1 \
          -target_drop_rate 0.0001 -exp_name 10K_75K_drop_1M -offline_frequency 10000 -offline_epochs 75000 -offline_dimension 0 -expectile 0.5 \
          -utd_ratio_online 1 -evaluate_bias
    done  

    ### ablation DropQ UTD = 9, 3, 2
    for seed in 0 42 1234; do
        for utd in 9 3 2; do
            exp_name="drop_1M_utd_${utd}"
            python main_o2.py -info lomo/$env/ -env $env-v2 -seed $seed -epochs 1000 -gpu_id 0 -method sac -layer_norm 1 \
            -target_drop_rate 0.0001 -exp_name $exp_name -offline_frequency -1  -offline_epochs 0 -offline_dimension 0 -expectile 0.5 \
            -utd_ratio_online $utd -evaluate_bias
        done
    done

done

for env in 'Humanoid'; do
    ### 10_75
    for seed in 0 42 1234; do
        python main_o2.py -info lomo/$env/ -env $env-v2 -seed $seed -epochs 1000 -gpu_id 0 -method sac -layer_norm 1 \
          -target_drop_rate 0.1 -exp_name 10K_75K_drop_1M -offline_frequency 10000 -offline_epochs 75000 -offline_dimension 0 -expectile 0.5 \
          -utd_ratio_online 1 -evaluate_bias
    done  

    ### ablation DropQ UTD = 9, 3, 2
    for seed in 0 42 1234; do
        for utd in 9 3 2; do
            exp_name="drop_1M_utd_${utd}"
            python main_o2.py -info lomo/$env/ -env $env-v2 -seed $seed -epochs 1000 -gpu_id 0 -method sac -layer_norm 1 \
            -target_drop_rate 0.1 -exp_name $exp_name -offline_frequency -1  -offline_epochs 0 -offline_dimension 0 -expectile 0.5 \
            -utd_ratio_online $utd -evaluate_bias
        done
    done

done

for env in 'Humanoid'; do
    ### ablation K = 200, 100, 50, 20, 10
    for seed in 0 42 1234; do
        for epochs in 200000 100000 50000 20000 10000; do
            epochs_str="${epochs%000}K"
            # freq_str="${freq%000}K"

            exp_name="10K_${epochs_str}_dropq_1M"
            python main_o2.py -info lomo/$env/ -env $env-v2 -seed $seed -epochs 1000 -gpu_id 0 -method sac -layer_norm 1 \
            -target_drop_rate 0.1 -exp_name $exp_name -offline_frequency 10000 -offline_epochs $epochs -offline_dimension 0 -expectile 0.5 \
            -utd_ratio_online 1 -evaluate_bias
        done
    done 

    # ### ablation F = 5, 50, 100
    # for seed in 0 42 1234; do
    #     for freq in 5000 50000 100000; do
    #         # epochs_str="${epochs%000}K"
    #         freq_str="${freq%000}K"

    #         exp_name="${freq_str}_75K_dropq_1M"
    #         python main_o2.py -info lomo/$env/ -env $env-v2 -seed $seed -epochs 1000 -gpu_id 0 -method sac -layer_norm 1 \
    #         -target_drop_rate 0.1 -exp_name $exp_name -offline_frequency $freq -offline_epochs 75000 -offline_dimension 0 -expectile 0.5 \
    #         -utd_ratio_online 1 -evaluate_bias
    #     done
    # done 
    
done
