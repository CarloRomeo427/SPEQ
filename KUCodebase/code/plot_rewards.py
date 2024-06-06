import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib
matplotlib.use('WebAgg')

# Define the folder and list of files with their labels
folder = 'runs/drq/'
file_paths = [
    {'file': 'Hopper-v2_2024-06-04T09:20:44.108523' + '/reward.csv', 'label': 'Baseline'},
    {'file': 'Hopper-v2_2024-06-06T09:44:36.632512' + '/reward.csv', 'label': 'MB'},
    # Add more runs here with their respective labels
]

# Set the visual style
sns.set(style="darkgrid")

# Create the plot
plt.figure(figsize=(10, 6))

# Loop through each file and plot the data
for run in file_paths:
    full_path = os.path.join(folder, run['file'])
    
    if os.path.exists(full_path):
        # Read the CSV file correctly
        df = pd.read_csv(full_path, header=None, names=["environment_interactions", "average_return", "extra"], usecols=[0, 1])

        # Verify the contents of the dataframe
        # print(f"Contents of {run['label']}:")
        # print(df.head())

        # Plot the data
        sns.lineplot(x="environment_interactions", y="average_return", data=df, label=run['label'])

    else:
        print(f"File not found: {full_path}")

# Add labels and title
plt.xlabel("environment interactions (1e5)")
plt.ylabel("average return")
plt.title("Hopper-v2")

# Set x-axis scale
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x * 1e-5:.1f}'))

# Show the plot
plt.legend()
plt.show()
