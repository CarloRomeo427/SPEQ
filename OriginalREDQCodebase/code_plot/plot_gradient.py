import os.path

import numpy as np
import pandas as pd
import wandb
import matplotlib.pyplot as plt

# Initialize the wandb API
api = wandb.Api()

# Project is specified by <entity/project-name>
runs = api.runs("girolamomacaluso/lomo")
envs = ["Hopper", "Ant", ]
# durations = [300, 300, 300, 300, 300, ]

save_dir = "/home/ganjiro/PycharmProjects/dropRL/DropQ/OriginalREDQCodebase/plots"
exp_name = "gradient"


def exponential_moving_average(data, alpha=0.15):
    ema = [data[0]]  # initialize with the first value
    for value in data[1:]:
        ema.append(ema[-1] * (1 - alpha) + alpha * value)
    return np.array(ema)


for j, env in enumerate(envs):

    if env not in ["Hopper", "HalfCheetah"]:
        eval_runs = [f'10K_75K_drop_rw_{env}-v2', f'SMC_sac_{env}-v2', f'SMC_redq_{env}-v2',
                     f'sac_1_vanilla_{env}-v2',
                     f'vanilla_redQ_{env}-v2',
                     f'vanilla_dropQ_bias_{env}-v2',
                     ]
    else:
        eval_runs = [f'10K_75K_drop_rw_{env}-v2', f'SMC_sac_{env}-v2', f'SMC_redq_{env}-v2',
                     f'sac_1_vanilla_{env}-v2',
                     f'vanilla_redQ_300_{env}-v2',
                     f'vanilla_dropQ_bias_300_{env}-v2',
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

    # Convert history_dict to DataFrame for plotting means
    # history_df = pd.DataFrame.from_dict(history_dict)
    # Plot EvalReward mean over steps for each run
    from scipy.interpolate import interp1d

    last_step = 0
    # Plot EvalReward mean over steps for each run
    plt.figure(figsize=(12, 8), dpi=300)  # Increase dimensions and resolution
    for run in eval_runs:
        if "10K_75K" not in run:
            steps = np.arange(len(history_dict[run]) * steps_runs[run], step=steps_runs[run])
        else:
            # lomosteps = np.arange(len(history_dict[run]) * steps_runs[run], step=steps_runs[run])
            #
            # for i in range(int((len(history_dict[run]) - 5) / 10) - 1):
            #     lomosteps[(i + 1) * 10:] = lomosteps[(i + 1) * 10:] + 150000
            # steps = np.array(lomosteps)
            # last_step = steps[-1]

            steps = np.zeros(len(history_dict[run]))

            start = 5
            for i in range(5):
                steps[i:] += 3000
            # Continue until we reach the end of the array
            while start < len(steps):
                # Iterate over the next 10 elements
                for i in range(start, min(start + 15, len(steps))):
                    steps[i:] += 10000
                for i in range(start+15, min(start + 25, len(steps))):
                    steps[i:] += 3000
                # Skip 75 elements
                start += 25  # Move 10 forward and skip 75
            # steps = np.arange(5_400_000, step=5_400_000 / len(history_dict[run]))
            last_step = steps[-1]

        plt.plot(steps, history_dict[run], label=labels[run], linewidth=6.0)

        # Add variance (standard deviation) as shaded region
        plt.fill_between(steps, history_dict[run] - std_dict[run],
                         history_dict[run] + std_dict[run], alpha=0.4)

    # Add plot details
    if env == "Walker2d":
        plt.xlabel('Log Gradient Steps', fontsize=24)
    # plt.ylabel('EvalReward')

    print(last_step)
    plt.axvline(x=last_step, color='black', lw=4)
    plt.ylabel('EvalReward', fontsize=24)
    plt.xscale('log')
    # plt.xlim(0,last_step)
    # plt.title(env)
    # plt.legend(loc='upper left')
    # plt.grid(True)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.grid(True, which="both", ls="-")
    import matplotlib as mpl
    plt.ylim(bottom=0 )
    mpl.rcParams['axes.linewidth'] = 2
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{exp_name}_{env}.pdf'), bbox_inches='tight')
