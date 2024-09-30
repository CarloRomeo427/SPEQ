import os.path

import numpy as np
import pandas as pd
import wandb
import matplotlib.pyplot as plt

# Initialize the wandb API
api = wandb.Api()

# Project is specified by <entity/project-name>
runs = api.runs("girolamomacaluso/lomo")
envs = ["Humanoid", "HalfCheetah", "Hopper"]  # "Walker2d","Ant",
durations = [300, 100, 100]  # 300,, 300
save_dir = "/home/ganjiro/PycharmProjects/dropRL/DropQ/OriginalREDQCodebase/plots"
exp_name = "redq"

def exponential_moving_average(data, alpha=0.2):
    ema = [data[0]]  # initialize with the first value
    for value in data[1:]:
        ema.append(ema[-1] * (1 - alpha) + alpha * value)
    return np.array(ema)


for i, env in enumerate(envs):
    eval_runs = [f'utd_9-redQ_{env}-v2', f'redq_10K_75K_{env}-v2', f'vanilla_redQ_{env}-v2']

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
        rewards_array = np.array(history_dict[run])
        mean_rewards = rewards_array.mean(0)[5:durations[i]]
        std_rewards = rewards_array.std(0)[5:durations[i]]  # Calculate standard deviation
        mean_rewards = exponential_moving_average(mean_rewards)
        # Update the dictionaries
        history_dict[run] = mean_rewards
        std_dict[run] = std_rewards

    # Convert history_dict to DataFrame for plotting means
    history_df = pd.DataFrame.from_dict(history_dict)
    # Plot EvalReward mean over steps for each run

    # Plot EvalReward mean over steps for each run
    plt.figure(figsize=(12, 8), dpi=300)  # Increase dimensions and resolution
    for run in eval_runs:
        steps = np.arange(len(history_dict[run]))
        plt.plot(steps, history_dict[run], label=run)
        # Add variance (standard deviation) as shaded region
        plt.fill_between(steps, history_dict[run] - std_dict[run],
                         history_dict[run] + std_dict[run], alpha=0.2)

    # Add plot details
    plt.xlabel('Training Steps')
    plt.ylabel('EvalReward')
    plt.title(env)
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir,f'{exp_name}_{env}.png'))
