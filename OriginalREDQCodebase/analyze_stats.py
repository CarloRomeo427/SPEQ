import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('WebAgg')

# Load the dataset
environments = ["Walker2d", "Hopper", "Ant", "Humanoid"]

# Initialize a dictionary to store processed correlation matrices for each environment
first_row_correlation_matrices = {}
second_row_correlation_matrices = {}

for env in environments:
    file_path = f'/home/romeo/Projects/SPEQ/OriginalREDQCodebase/stats/all_stats_{env}_history_stats.csv'
    data = pd.read_csv(file_path)

    # Define relevant columns for each technique
    techniques = {
        "10k_75k": [
            f"10K_75K_bias_dropQ_{env}-v2-eval_return_mean",
            f"10K_75K_bias_dropQ_{env}-v2-mean_loss_q_mean",
            f"10K_75K_bias_dropQ_{env}-v2-policy_loss_mean",
        ],
        "vanilla_dropq": [
            f"vanilla_dropQ_bias_{env}-v2-eval_return_mean",
            f"vanilla_dropQ_bias_{env}-v2-mean_loss_q_mean",
            f"vanilla_dropQ_bias_{env}-v2-policy_loss_mean",
        ],
        "vanilla_redq": [
            f"vanilla_redQ_{env}-v2-eval_return_mean",
            f"vanilla_redQ_{env}-v2-mean_loss_q_mean",
            f"vanilla_redQ_{env}-v2-policy_loss_mean",
        ],
        "sac": [
            f"sac_1_vanilla_bias_{env}-v2-eval_return_mean",
            f"sac_1_vanilla_bias_{env}-v2-mean_loss_q_mean",
            f"sac_1_vanilla_bias_{env}-v2-policy_loss_mean",
        ],
    }

    # Compute the first row of the correlation matrix for each technique
    first_row = []
    second_row = []

    for technique, columns in techniques.items():
        technique_data = data[columns]
        technique_data.columns = ["eval_reward", "mean_loss_q", "policy_loss"]  # Rename for clarity
        corr_matrix = technique_data.corr()
        first_row.append(corr_matrix.iloc[0])  # Take the first row
        second_row.append(corr_matrix.iloc[1])  # Take the second row  

    # Stack all the rows vertically to create a unique correlation matrix
    first_row_correlation_matrix = pd.DataFrame(first_row, index=techniques.keys())
    first_row_correlation_matrices[env] = first_row_correlation_matrix

    second_row_correlation_matrix = pd.DataFrame(second_row, index=techniques.keys())
    second_row_correlation_matrices[env] = second_row_correlation_matrix

# Visualization: All stacked matrices in two rows in one figure
fig, axes = plt.subplots(2, len(environments), figsize=(50, 60), sharey=True)

# Plot first row correlation matrices
for i, (ax, (env, stacked_corr_matrix)) in enumerate(zip(axes[0], first_row_correlation_matrices.items())):
    sns.heatmap(
        stacked_corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", cbar=False, square=True, ax=ax,  # Remove color bar
        annot_kws={"size": 30},  # Increase annotation font size
        xticklabels=stacked_corr_matrix.columns, yticklabels=stacked_corr_matrix.index
    )
    ax.set_title(f"{env} - First Row", fontsize=30)
    ax.tick_params(axis='both', which='major', labelsize=30)
    ax.set_aspect('equal')  # Set aspect ratio to ensure all matrices have the same size

# Plot second row correlation matrices
for i, (ax, (env, stacked_corr_matrix)) in enumerate(zip(axes[1], second_row_correlation_matrices.items())):
    sns.heatmap(
        stacked_corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", cbar=False, square=True, ax=ax,  # Remove color bar
        annot_kws={"size": 30},  # Increase annotation font size
        xticklabels=stacked_corr_matrix.columns, yticklabels=stacked_corr_matrix.index
    )
    ax.set_title(f"{env} - Second Row", fontsize=30)
    ax.tick_params(axis='both', which='major', labelsize=30)
    ax.set_aspect('equal')  # Set aspect ratio to ensure all matrices have the same size

plt.tight_layout()
plt.show()


