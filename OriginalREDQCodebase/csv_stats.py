import os
import numpy as np
import pandas as pd
import wandb

# Initialize the wandb API
api = wandb.Api()

# Project is specified by <entity/project-name>
runs = api.runs("girolamomacaluso/lomo")
envs = ["Humanoid", "Ant", "Hopper", "Walker2d"]
durations = [300, 300, 300, 300, 300]
save_dir = "/home/romeo/Projects/SPEQ/OriginalREDQCodebase/stats/"
exp_name = "all_stats"

def exponential_moving_average(data, alpha=0.1):
    ema = [data[0]]  # initialize with the first value
    for value in data[1:]:
        ema.append(ema[-1] * (1 - alpha) + alpha * value)
    return np.array(ema)

for j, env in enumerate(envs):
    eval_runs = [
        f'10K_75K_bias_dropQ_{env}-v2',
        f'vanilla_dropQ_bias_{env}-v2',
        f'vanilla_redQ_{env}-v2',
        f'sac_1_vanilla_bias_{env}-v2'
    ]

    labes = ["SPEQ (Ours)", "DroQ", "RedQ", "SAC"]

    lables = dict(zip(eval_runs, labes))
    history_dict = dict(zip(eval_runs, [[] for _ in eval_runs]))
    std_dict = dict(zip(eval_runs, [[] for _ in eval_runs]))
    mean_loss_q_dict = dict(zip(eval_runs, [[] for _ in eval_runs]))
    std_loss_q_dict = dict(zip(eval_runs, [[] for _ in eval_runs]))
    policy_loss_dict = dict(zip(eval_runs, [[] for _ in eval_runs]))
    std_policy_loss_dict = dict(zip(eval_runs, [[] for _ in eval_runs]))

    # Iterate through runs and extract relevant data
    for run in runs:
        if run.state == "finished" and run.name in eval_runs:
            print(run.name)
            # Extract historical data
            rewards = run.history(keys=["EvalReward"]).to_numpy()[:, 1]
            mean_loss_q = run.history(keys=["mean_loss_q"]).to_numpy()[:, 1]
            policy_loss = run.history(keys=["policy_loss"]).to_numpy()[:, 1]
            history_dict[run.name].append(rewards)
            mean_loss_q_dict[run.name].append(mean_loss_q)
            policy_loss_dict[run.name].append(policy_loss)

    # Compute mean and standard deviation for each run
    for run in eval_runs:
        try:
            rewards_array = np.array(history_dict[run])
            mean_loss_q_array = np.array(mean_loss_q_dict[run])
            policy_loss_array = np.array(policy_loss_dict[run])
        except:
            data_lists = [history_dict, mean_loss_q_dict, policy_loss_dict]
            arrays = []
            for data in data_lists:
                a = data[run]
                max_dim = max([arr.shape[0] for arr in a])
                for i in range(len(a)):
                    if a[i].shape[0] < max_dim:
                        last_element = a[i][-1]
                        padding_length = max_dim - a[i].shape[0]
                        padding = np.full(padding_length, last_element)
                        a[i] = np.concatenate((a[i], padding))
                arrays.append(np.array(a))
            rewards_array, mean_loss_q_array, policy_loss_array = arrays

        mean_rewards = rewards_array.mean(0)[5:durations[j]]
        std_rewards = rewards_array.std(0)[5:durations[j]]
        mean_loss_q = mean_loss_q_array.mean(0)[5:durations[j]]
        std_loss_q = mean_loss_q_array.std(0)[5:durations[j]]
        mean_policy_loss = policy_loss_array.mean(0)[5:durations[j]]
        std_policy_loss = policy_loss_array.std(0)[5:durations[j]]

        mean_rewards = exponential_moving_average(mean_rewards)
        std_rewards = exponential_moving_average(std_rewards)
        mean_loss_q = exponential_moving_average(mean_loss_q)
        std_loss_q = exponential_moving_average(std_loss_q)
        mean_policy_loss = exponential_moving_average(mean_policy_loss)
        std_policy_loss = exponential_moving_average(std_policy_loss)

        history_dict[run] = mean_rewards
        std_dict[run] = std_rewards
        mean_loss_q_dict[run] = mean_loss_q
        std_loss_q_dict[run] = std_loss_q
        policy_loss_dict[run] = mean_policy_loss
        std_policy_loss_dict[run] = std_policy_loss

    # Save history_dict, std_dict, mean_loss_q_dict, and policy_loss_dict to a CSV file
    combined_dict = {}
    for run in eval_runs:
        combined_dict[f'{run}-eval_return_mean'] = history_dict[run]
        combined_dict[f'{run}-eval_return_std'] = std_dict[run]
        combined_dict[f'{run}-mean_loss_q_mean'] = mean_loss_q_dict[run]
        combined_dict[f'{run}-mean_loss_q_std'] = std_loss_q_dict[run]
        combined_dict[f'{run}-policy_loss_mean'] = policy_loss_dict[run]
        combined_dict[f'{run}-policy_loss_std'] = std_policy_loss_dict[run]

    final_df = pd.DataFrame.from_dict(combined_dict, orient='index').transpose()
    csv_path = os.path.join(save_dir, f'{exp_name}_{env}_history_stats.csv')
    final_df.to_csv(csv_path, index=False)
    print(f"Saved history, std, mean_loss_q, and policy_loss stats to {csv_path}")
