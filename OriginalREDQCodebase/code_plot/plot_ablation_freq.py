import os.path

import numpy as np
import pandas as pd
import wandb
import matplotlib.pyplot as plt

# Initialize the wandb API
api = wandb.Api()

# Project is specified by <entity/project-name>
runs = api.runs("girolamomacaluso/lomo")
envs = ["Humanoid", ]
# durations = [300, 300, 300, 300, 300]  # ,
save_dir = "/home/ganjiro/PycharmProjects/dropRL/DropQ/OriginalREDQCodebase/plots"
exp_name = "alb_freq"


def exponential_moving_average(data, alpha=0.1):
    ema = [data[0]]  # initialize with the first value
    for value in data[1:]:
        ema.append(ema[-1] * (1 - alpha) + alpha * value)
    return np.array(ema)


for j, env in enumerate(envs):

    eval_runs = [f'abl_freq_5000-dropQ_{env}-v2',
                 f'10K_75K_bias_dropQ_{env}-v2',
                 f'abl_freq_50000-dropQ_{env}-v2',
                 f'abl_freq_100000-dropQ_{env}-v2',
                 ]
    labes = ["5K", "10K (Ours)", "50K", "100K"]
    colors = ['#2ca02c', '#1f77b4', '#d62728', '#9467bd', '#8c564b']
    lables = dict(zip(eval_runs, labes))
    colors = dict(zip(eval_runs, colors))
    history_dict = dict(zip(eval_runs, [[] for _ in eval_runs]))
    # To store the standard deviations
    std_dict = dict(zip(eval_runs, [[] for _ in eval_runs]))

    # Iterate through runs and extract relevant data
    for run in runs:
        if run.state == "finished" and run.name in eval_runs:
            print(run.name)
            # Extract historical data with steps and EvalReward
            rewards = run.history(keys=["EvalReward"], samples=750).to_numpy()[:, 1]
            history_dict[run.name].append(rewards)

    # Compute mean and standard deviation for each run
    for run in eval_runs:
        try:
            rewards_array = np.array(history_dict[run])
        except:
            a = history_dict[run]

            # Find the maximum dimension of arrays in `a`
            max_dim = max([arr.shape[0] for arr in a])

            # Pad arrays that are shorter than the maximum dimension
            for i in range(len(a)):
                if a[i].shape[0] < max_dim:
                    # Get the last element of the shorter array
                    last_element = a[i][-1]

                    # Calculate how many elements need to be added
                    padding_length = max_dim - a[i].shape[0]

                    # Create the padding array by repeating the last element
                    padding = np.full(padding_length, last_element)

                    # Concatenate the original array with the padding
                    a[i] = np.concatenate((a[i], padding))

            # Convert the list of arrays to a single numpy array
            rewards_array = np.array(a)
        mean_rewards = rewards_array.mean(0)[5:]
        std_rewards = rewards_array.std(0)[5:]  # Calculate standard deviation
        mean_rewards = exponential_moving_average(mean_rewards)
        std_rewards = exponential_moving_average(std_rewards)
        # Update the dictionaries
        history_dict[run] = mean_rewards
        std_dict[run] = std_rewards


    # Plot EvalReward mean over steps for each run

    # Plot EvalReward mean over steps for each run
    plt.figure(figsize=(12, 8), dpi=300)  # Increase dimensions and resolution
    for run in eval_runs:
        steps = np.arange(300_000, step=300_000 / len(history_dict[run]))
        plt.plot(steps, history_dict[run], label=lables[run], linewidth=6.0, color=colors[run])
        # Add variance (standard deviation) as shaded region
        plt.fill_between(steps, history_dict[run] - std_dict[run],
                         history_dict[run] + std_dict[run], alpha=0.4, color=colors[run])
    from matplotlib import rcParams

    rcParams.update({'figure.autolayout': True})

    # Add plot details
    # if env == "Walker2d":
    plt.xlabel('Environment Steps', fontsize=24)

    plt.ylabel('EvalReward', fontsize=24)

    # plt.title(, rotation='vertical', x=-0.16, y=0.4, fontsize=22, weight='bold')

    # if env == "Ant":
    leg = plt.legend(loc='upper left', fontsize=24)
    # change the line width for the legend
    for line in leg.get_lines():
        line.set_linewidth(6.0)
    import matplotlib as mpl

    mpl.rcParams['axes.linewidth'] = 2
    plt.xticks(ticks=np.array([0, 50000, 150_000, 250_000]), fontsize=24)
    plt.yticks(fontsize=20)
    plt.grid(True)
    # plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{exp_name}_{env}.pdf'), bbox_inches='tight')
