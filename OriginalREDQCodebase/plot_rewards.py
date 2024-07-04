import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('WebAgg')


def load_progress_files(base_folder, seeds, experiment_name):
    dfs = []
    for seed in seeds:
        file_path = os.path.join(base_folder, experiment_name, f"{experiment_name}_s{seed}", "progress.txt")
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path, sep='\t')
                if 'AverageEpRet' in df.columns and 'TotalEnvInteracts' in df.columns:
                    df = df[['TotalEnvInteracts', 'AverageEpRet']]
                    dfs.append(df)
                    print(f"File {file_path} for seed {seed} loaded successfully with {len(df)} rows.")
                else:
                    print(f"File {file_path} does not contain required columns.")
            except pd.errors.EmptyDataError:
                print(f"File {file_path} is empty.")
        else:
            print(f"File {file_path} not found.")
    return dfs


def compute_common_average(dfs, max_interactions):
    if not dfs:
        return None

    # Truncate dataframes to the max_interactions
    truncated_dfs = [df[df['TotalEnvInteracts'] <= max_interactions] for df in dfs]

    # Concatenate dataframes and compute the mean and std
    concatenated_df = pd.concat(truncated_dfs).groupby('TotalEnvInteracts').agg(
        AverageEpRet=('AverageEpRet', 'mean'),
        StdEpRet=('AverageEpRet', 'std')
    ).reset_index()

    return concatenated_df


def plot_with_error_bars(df, alpha=0.3, label=None):
    if df is not None and 'AverageEpRet' in df.columns and 'StdEpRet' in df.columns:
        df['ExpMovingAvg'] = df['AverageEpRet'].ewm(alpha=alpha).mean()
        df['ExpMovingStd'] = df['StdEpRet'].ewm(alpha=alpha).mean()
        plt.plot(df['TotalEnvInteracts'], df['ExpMovingAvg'], label=label)
        plt.fill_between(df['TotalEnvInteracts'], df['ExpMovingAvg'] - df['ExpMovingStd'],
                         df['ExpMovingAvg'] + df['ExpMovingStd'], alpha=0.2)
        # plt.plot(df['TotalEnvInteracts'], df['AverageEpRet'], label=label)
        # plt.fill_between(df['TotalEnvInteracts'], df['AverageEpRet'] - df['StdEpRet'],
        #                  df['AverageEpRet'] + df['StdEpRet'], alpha=0.2)


def main():
    base_folder = 'runs/onlOff/'
    seeds = [0, 42, 1234, 5678, 9876]

    env = 'Hopper'

    # Load and process baseline progress files
    base_experiment_name = f"vanilla_{env}-v2"
    base_dfs = load_progress_files(base_folder, seeds, base_experiment_name)

    # Load and process o2 progress files
    experiment_name = "exp6"
    o2_experiment_name = f"{experiment_name}_{env}-v2"
    o2_dfs = load_progress_files(base_folder, seeds, o2_experiment_name)

    # Determine the maximum completed interactions for the o2 runs
    if o2_dfs:
        max_interactions = min(df['TotalEnvInteracts'].max() for df in o2_dfs)
    else:
        print("No valid o2 data found.")
        max_interactions = None

    if base_dfs:
        if max_interactions is not None:
            base_avg_df = compute_common_average(base_dfs, max_interactions)
        else:
            base_avg_df = compute_common_average(base_dfs, float('inf'))
        print('Baseline:', base_avg_df['AverageEpRet'][:-5].mean(), base_avg_df['StdEpRet'][:-5].mean())
    else:
        print("No valid baseline data found.")
        base_avg_df = None

    if o2_dfs and max_interactions is not None:
        o2_avg_df = compute_common_average(o2_dfs, max_interactions)
        print('O2:', o2_avg_df['AverageEpRet'][:-5].mean(), o2_avg_df['StdEpRet'][:-5].mean())
    else:
        o2_avg_df = None

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plot_with_error_bars(base_avg_df, label='Baseline')
    if o2_avg_df is not None:
        plot_with_error_bars(o2_avg_df, label=experiment_name)

    plt.xlabel('Total Environment Interactions')
    plt.ylabel('Average Episode Return')
    plt.title(f'{experiment_name} on {env}')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
