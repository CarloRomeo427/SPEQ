import os.path

import numpy as np
import pandas as pd
import wandb
import matplotlib.pyplot as plt

# Initialize the wandb API
api = wandb.Api()

# Project is specified by <entity/project-name>
runs = api.runs("girolamomacaluso/lomo")
envs = ["Hopper", "Humanoid", "Walker2d", "Ant", ]
durations = [300, 300, 300, 300, 300, ]

save_dir = "/home/ganjiro/PycharmProjects/dropRL/DropQ/OriginalREDQCodebase/plots"
exp_name = "teaser"


def exponential_moving_average(data, alpha=0.15):
    ema = [data[0]]  # initialize with the first value
    for value in data[1:]:
        ema.append(ema[-1] * (1 - alpha) + alpha * value)
    return np.array(ema)



eval_runs = [f'10K_75K_bias_dropQ_', f'SMC_sac_', f'SMC_redq_',
             f'sac_1_vanilla_',
             f'vanilla_redQ_',
             f'vanilla_dropQ_',
             ]

steps_runs = [3_000, 30_000, 2_020_000, 3_000, 401_000, 41_000]
labes = ["Ours", "SMR-SAC", "SMR-RedQ", "SAC", "RedQ", "DroQ"]
steps_runs = dict(zip(eval_runs, steps_runs))
labels = dict(zip(eval_runs, labes))

history_dict = dict(zip(eval_runs, [[] for _ in eval_runs]))
# To store the standard deviations
std_dict = dict(zip(eval_runs, [[] for _ in eval_runs]))

# Iterate through runs and extract relevant data
for run in runs:
    if run.state == "finished" and "HalfCheetah" not in run.name and any(
            substring in run.name for substring in eval_runs):
        print(run.name)
        # Extract historical data with steps and EvalReward
        rewards = run.history(keys=["EvalReward"]).to_numpy()[:, 1]
        run_corr = next((substring for substring in eval_runs if substring in run.name), None)
        history_dict[run_corr].append(rewards)

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
    mean_rewards = rewards_array.mean(0)[5:durations[0]]
    std_rewards = rewards_array.std(0)[5:durations[0]]  # Calculate standard deviation
    mean_rewards = exponential_moving_average(mean_rewards)
    std_rewards = exponential_moving_average(std_rewards)
    # Update the dictionaries
    history_dict[run] = mean_rewards
    std_dict[run] = std_rewards

# Convert history_dict to DataFrame for plotting means
history_df = pd.DataFrame.from_dict(history_dict)
# Plot EvalReward mean over steps for each run
from scipy.interpolate import interp1d

last_step = 0
# Plot EvalReward mean over steps for each run
plt.figure(figsize=(12, 8), dpi=300)  # Increase dimensions and resolution
for run in eval_runs:
    if '10K_75K_' not in run:
        steps = np.arange(len(history_dict[run]) * steps_runs[run], step=steps_runs[run])
    else:
        lomosteps = np.arange(len(history_dict[run]) * steps_runs[run], step=steps_runs[run])

        for i in range(int((len(history_dict[run]) - 5) / 10) - 1):
            lomosteps[(i + 1) * 10:] = lomosteps[(i + 1) * 10:] + 150000
        steps = np.array(lomosteps)
        last_step = steps[-1]

    plt.plot(steps, history_dict[run], label=labels[run], linewidth=6.0)

    # Add variance (standard deviation) as shaded region
    plt.fill_between(steps, history_dict[run] - std_dict[run],
                     history_dict[run] + std_dict[run], alpha=0.4)

# Add plot details
# if env == "Walker2d":
plt.xlabel('Gradient Steps', fontsize=24)
plt.ylabel('EvalReward', fontsize=24)
plt.xscale('log')

leg = plt.legend(loc='upper left', fontsize=30)

# change the line width for the legend
for line in leg.get_lines():
    line.set_linewidth(8.0)
# plt.xlim(0, last_step)
# plt.title(env)
# plt.legend(loc='upper left')
# plt.grid(True)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.grid(True, which="both", ls="-")
import matplotlib as mpl

mpl.rcParams['axes.linewidth'] = 3
plt.tight_layout()
plt.savefig(os.path.join(save_dir, f'{exp_name}.svg'), bbox_inches='tight', )
