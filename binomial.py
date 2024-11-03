import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import binom, norm

# Set parameters for the binomial distributions
n_trials = 10  # Number of trials for each binomial distribution
p_success1 = 0.5  # Probability of success for the first binomial
p_success2 = 0.6  # Probability of success for the second binomial
num_samples = 5000  # Number of samples

# Define a range of positive and negative correlation values
correlation_values = [-0.8, -0.5, -0.3, 0.8]  # Negative and positive correlations included

# Calculate theoretical means
mean1 = n_trials * p_success1
mean2 = n_trials * p_success2

# Translation amounts (if needed)
translation_x = -10
translation_y = -10

# Set up plot parameters for better visibility
sns.set(font_scale=1.2)  # Increase font size

# Create a subplot for each correlation value
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.flatten()

for i, corr in enumerate(correlation_values):
    # Generate correlated samples using copula with specified correlation
    normal_samples = np.random.multivariate_normal(
        [0, 0], [[1, corr], [corr, 1]], num_samples
    )
    u_samples = norm.cdf(normal_samples)

    # Generate binomial samples based on the copula-transformed data
    binom_samples1 = binom.ppf(u_samples[:, 0], n_trials, p_success1).astype(int)
    binom_samples2 = binom.ppf(u_samples[:, 1], n_trials, p_success2).astype(int)

    # Center the samples by subtracting the means
    binom_samples1_centered = binom_samples1 - mean1
    binom_samples2_centered = binom_samples2 - mean2

    # Create a 2D histogram for the joint distribution, with bins covering -10 to 10
    bins_range = range(-10, 11)  # Bins from -10 to 10 inclusive
    heatmap_data, x_edges, y_edges = np.histogram2d(
        binom_samples1_centered, binom_samples2_centered, bins=[bins_range, bins_range]
    )

    # Plot heatmap
    sns.heatmap(
        heatmap_data.T,
        cmap="Blues",
        cbar=True,
        annot=False,
        fmt="g",
        ax=axes[i],
        xticklabels=range(-10, 11),
        yticklabels=range(-10, 11),
    )
    axes[i].set_title(
        f"Heatmap of Centered Bivariate Binomial Distribution\n(Correlation={corr})",
        fontsize=16,
    )
    axes[i].set_xlabel("Centered Binomial Distribution 1", fontsize=14)
    axes[i].set_ylabel("Centered Binomial Distribution 2", fontsize=14)
    axes[i].set_xticks(np.arange(len(bins_range)) + 0.5)
    axes[i].set_yticks(np.arange(len(bins_range)) + 0.5)
    axes[i].invert_yaxis()  # To match the orientation of the heatmap with the axes

# Adjust layout for clear visualization
plt.tight_layout()
plt.show()