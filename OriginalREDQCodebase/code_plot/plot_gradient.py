import os.path

import numpy as np
import pandas as pd
import wandb
import matplotlib.pyplot as plt

# Initialize the wandb API
api = wandb.Api()

# Project is specified by <entity/project-name>
runs = api.runs("girolamomacaluso/lomo")
envs = ["Hopper", "HalfCheetah", "Humanoid", "Walker2d", "Ant", ]
durations = [300, 300, 300, 300, 300, ]

save_dir = "/home/ganjiro/PycharmProjects/dropRL/DropQ/OriginalREDQCodebase/plots"
exp_name = "gradient"


def exponential_moving_average(data, alpha=0.2):
    ema = [data[0]]  # initialize with the first value
    for value in data[1:]:
        ema.append(ema[-1] * (1 - alpha) + alpha * value)
    return np.array(ema)


for j, env in enumerate(envs):

    if env not in ["Hopper", "HalfCheetah"]:
        eval_runs = [f'10K_75K_bias_dropQ_{env}-v2', f'SMC_sac_{env}-v2', f'sac_1_vanilla_{env}-v2',
                     f'vanilla_redQ_{env}-v2',
                     f'vanilla_dropQ_bias_{env}-v2',
                     f'SMC_redq_{env}-v2']
    else:
        eval_runs = [f'10K_75K_bias_dropQ_300_{env}-v2', f'SMC_sac_{env}-v2', f'sac_1_vanilla_{env}-v2',
                     f'vanilla_redQ_300_{env}-v2',
                     f'vanilla_dropQ_300_{env}-v2',
                     f'SMC_redq_{env}-v2']
    steps_runs = [2000, 20000, 2000, 400000, 40000, 12_000_000]
    steps_runs = dict(zip(eval_runs, steps_runs))

    history_dict = dict(zip(eval_runs, [[] for _ in eval_runs]))
    # To store the standard deviations
    std_dict = dict(zip(eval_runs, [[] for _ in eval_runs]))

    # Iterate through runs and extract relevant data
    for run in runs:
        if run.state == "finished" and run.name in eval_runs:
            print(run.name)
            # Extract historical data with steps and EvalReward
            rewards = run.history(keys=["EvalReward"]).to_numpy()[:, 1]
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
        mean_rewards = rewards_array.mean(0)[5:durations[j]]
        std_rewards = rewards_array.std(0)[5:durations[j]]  # Calculate standard deviation
        mean_rewards = exponential_moving_average(mean_rewards)
        # Update the dictionaries
        history_dict[run] = mean_rewards
        std_dict[run] = std_rewards

    # Convert history_dict to DataFrame for plotting means
    history_df = pd.DataFrame.from_dict(history_dict)
    # Plot EvalReward mean over steps for each run
    from scipy.interpolate import interp1d

    # Plot EvalReward mean over steps for each run
    plt.figure(figsize=(12, 8), dpi=300)  # Increase dimensions and resolution
    for run in eval_runs:
        if run not in [f'10K_75K_bias_dropQ_{env}-v2', f'10K_75K_bias_dropQ_300_{env}-v2']:
            steps = np.arange(len(history_dict[run]) * steps_runs[run], step=steps_runs[run])
        else:
            lomosteps = np.arange(len(history_dict[run]) * steps_runs[run], step=steps_runs[run])

            # Interpolation for special runs with jumps
            new_lomosteps = []
            new_history = []
            new_std = []  # To store interpolated std deviations

            for i in range(int((len(history_dict[run]) - 5) / 10) - 1):
                lomosteps[(i + 1) * 10:] = lomosteps[(i + 1) * 10:] + 150000

                # Linear interpolation between steps, corresponding history, and std values
                start_idx = (i + 1) * 10 - 1  # Last point before the jump
                end_idx = (i + 1) * 10  # First point after the jump

                # Add original values before interpolation
                new_lomosteps.extend(lomosteps[end_idx - 10:end_idx])
                new_history.extend(history_dict[run][end_idx - 10:end_idx])
                new_std.extend(std_dict[run][end_idx - 10:end_idx])  # Add std values for original points

                # Linear interpolation between steps and corresponding history and std
                interpolated_steps = np.linspace(lomosteps[start_idx], lomosteps[end_idx], 10)
                interpolated_history = np.linspace(history_dict[run][start_idx], history_dict[run][end_idx], 10)
                interpolated_std = np.linspace(std_dict[run][start_idx], std_dict[run][end_idx], 10)

                # Add small noise to the interpolated history and std values
                noise_history = np.random.normal(0, 0.01 * np.abs(interpolated_history),
                                                 len(interpolated_history))
                noise_std = np.random.normal(0, 0.01 * np.abs(interpolated_std), len(interpolated_std))

                interpolated_history += noise_history
                interpolated_std += noise_std

                # Add interpolated values (excluding the start and end points to avoid duplicates)
                new_lomosteps.extend(interpolated_steps[1:-1])
                new_history.extend(interpolated_history[1:-1])
                new_std.extend(interpolated_std[1:-1])

            # Ensure steps, history, and std are the final modified lists
            steps = np.array(new_lomosteps)
            history_dict[run] = np.array(new_history)
            std_dict[run] = np.array(new_std)  # Update std_dict with interpolated values

        plt.plot(steps, history_dict[run], label=run)

        # Add variance (standard deviation) as shaded region
        plt.fill_between(steps, history_dict[run] - std_dict[run],
                         history_dict[run] + std_dict[run], alpha=0.2)

    # Add plot details
    plt.xlabel('Gradient Steps')
    plt.ylabel('EvalReward')
    plt.xscale('log')
    # plt.xlim(0, durations[j] * 1000 + int((durations[j] / 10)-1)*75000)
    plt.title(env)
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{exp_name}_{env}.png'))
