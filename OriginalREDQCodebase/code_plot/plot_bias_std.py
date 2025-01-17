import os.path
import numpy as np
import pandas as pd
import wandb
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('WebAgg')

# Initialize the wandb API
api = wandb.Api()

# Project is specified by <entity/project-name>
runs = api.runs("girolamomacaluso/lomo")
envs = ["Humanoid", "Ant", "Hopper", "Walker2d"]
durations = [300, 300, 300, 300]
save_dir = "/home/romeo/Projects/SPEQ/OriginalREDQCodebase/code_plot/plots/"
exp_name = "meta_std_bias"

def exponential_moving_average(data, alpha=0.05):
    ema = [data[0]]  # initialize with the first value
    for value in data[1:]:
        ema.append(ema[-1] * (1 - alpha) + alpha * value)
    return np.array(ema)

for j, env in enumerate(envs):
    if env not in ["Hopper", "HalfCheetah"]:
        eval_runs = [f'10K_75K_bias_dropQ_{env}-v2', f'vanilla_redQ_bias_{env}-v2']
    else:
        eval_runs = [f'10K_75K_bias_dropQ_300_{env}-v2', f'vanilla_redQ_bias_{env}-v2']

    labes = ["SPEQ (Ours)", "RedQ"]
    lables = dict(zip(eval_runs, labes))
    history_dict = dict(zip(eval_runs, [[] for _ in eval_runs]))
    std_dict = dict(zip(eval_runs, [[] for _ in eval_runs]))

    # Extract historical data from wandb runs
    for run in runs:
        if run.state == "finished" and run.name in eval_runs:
            print(run.name)
            rewards = run.history(keys=["normalized_bias_per_state"]).to_numpy()[:, 1]
            history_dict[run.name].append(rewards)

    # Compute mean and standard deviation
    for run in eval_runs:
        try:
            rewards_array = np.array(history_dict[run])
        except:
            a = history_dict[run]
            max_dim = max([arr.shape[0] for arr in a])
            for i in range(len(a)):
                if a[i].shape[0] < max_dim:
                    last_element = a[i][-1]
                    padding_length = max_dim - a[i].shape[0]
                    padding = np.full(padding_length, last_element)
                    a[i] = np.concatenate((a[i], padding))
            rewards_array = np.array(a)
        std_rewards = rewards_array.std(0)[5:durations[j]]  # Calculate standard deviation
        std_rewards = exponential_moving_average(std_rewards)
        std_dict[run] = std_rewards

    # Compute meta-STD
    std_values = []
    for run in eval_runs:
        if run in std_dict:
            std_values.append(std_dict[run])  # Collect STD values for each run

    std_values_array = np.array(std_values)

    # Plot Mean STD and Meta-STD for each run
    plt.figure(figsize=(12, 8), dpi=300)
    steps = np.arange(std_values_array.shape[1] * 1000, step=1000)

    for idx, run in enumerate(eval_runs):
        mean_std = std_values_array[idx]
        meta_std = std_values_array.std(axis=0)  # Compute meta-STD across all runs

        # Plot mean STD for the specific run
        plt.plot(steps, mean_std, label=f"{lables[run]}", linewidth=6.0)

        # Plot shaded region for the specific run
        plt.fill_between(
            steps,
            mean_std - meta_std,
            mean_std + meta_std,
            alpha=0.4,
            # label=f"{lables[run]} - Shaded Region"
        )

    # Plot details
    plt.xlabel('Environment Steps', fontsize=40)
    plt.ylabel('STD of Normalized Bias (with Meta-STD)', fontsize=30)
    plt.xticks(ticks=np.array([0, 50000, 150000, 250000]), fontsize=30)
    plt.yticks(fontsize=30)
    plt.grid(True)
    plt.legend(fontsize=20)
    plt.tight_layout()

    # Save the plot
    plt.savefig(os.path.join(save_dir, f'meta_std_{exp_name}_{env}.jpg'), bbox_inches='tight')
