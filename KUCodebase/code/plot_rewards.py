import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib
matplotlib.use('WebAgg')

# Define the folder and list of files with their labels and seeds
folder = 'runs/drq/'
name_env = 'Hopper-v2'
seeds = [0, 42, 1234, 5678, 9876]
file_paths = [
    {'file': 'gt', 'label': 'Baseline'},
    {'file': 'utd/3', 'label': 'UTD 3'},
    {'file': 'utd/5', 'label': 'UTD 5'},
    {'file': 'utd/10', 'label': 'UTD 10'},
    {'file': 'utd/20', 'label': 'UTD 20'},
    # Add more runs here with their respective labels
]

# Set the visual style
sns.set(style="darkgrid")

# Initialize a dictionary to store dataframes for each label
data = {run['label']: [] for run in file_paths}
min_shared_steps = float('inf')

# Loop through each seed and file to read and stack the data
for seed in seeds:
    for run in file_paths:
        full_path = os.path.join(folder, name_env, str(seed), run['file'], "reward.csv")
        
        if os.path.exists(full_path):
            # Read the CSV file correctly
            df = pd.read_csv(full_path, header=None, names=["environment_interactions", "average_return", "extra"], usecols=[0, 1])
            
            # Filter out rows with rewards <= 0
            df = df[df["average_return"] > 0]

            # Update the minimum shared steps
            if not df.empty:
                min_shared_steps = min(min_shared_steps, df["environment_interactions"].max())
            
            # Append the dataframe to the respective label list
            data[run['label']].append(df)
        else:
            print(f"File not found: {full_path}")

# Compute mean and standard deviation for each label up to the minimum shared steps
means_stds = {}
for label, dfs in data.items():
    if dfs:
        # Concatenate all dataframes for the current label and trim to min_shared_steps
        combined_df = pd.concat(dfs)
        combined_df = combined_df[combined_df["environment_interactions"] <= min_shared_steps]
        
        # Calculate the mean and std deviation
        mean_df = combined_df.groupby("environment_interactions")["average_return"].mean().reset_index()
        std_df = combined_df.groupby("environment_interactions")["average_return"].std().reset_index()

        # Compute the exponentially weighted moving average (ewm)
        mean_df["ewm"] = mean_df["average_return"].ewm(span=50).mean()
        std_df["ewm"] = std_df["average_return"].ewm(span=50).mean()
        
        # Store the results
        means_stds[label] = {"mean": mean_df, "std": std_df}

# Create the plot
plt.figure(figsize=(10, 6))

# Plot the mean and standard deviation with the moving average
for label, dfs in means_stds.items():
    mean_df = dfs["mean"]
    std_df = dfs["std"]
    
    # Plot the mean with ewm
    sns.lineplot(x="environment_interactions", y="ewm", data=mean_df, label=f'{label} Mean')
    
    # Fill the area between mean ± std with ewm
    plt.fill_between(mean_df["environment_interactions"], mean_df["ewm"] - std_df["ewm"], mean_df["ewm"] + std_df["ewm"], alpha=0.3)

# Add labels and title
plt.xlabel("environment interactions (1e5)")
plt.ylabel("average return")
plt.title("Hopper-v2")

# Set x-axis scale
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x * 1e-5:.1f}'))

# Show the plot
plt.legend()
plt.show()
