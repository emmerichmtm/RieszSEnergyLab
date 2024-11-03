import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import geom, norm
from mpl_toolkits.mplot3d import Axes3D

# Set parameters for the geometric distributions
p_success1 = 0.3  # Probability of success for the first geometric distribution
p_success2 = 0.3  # Probability of success for the second geometric distribution
num_samples = 5000  # Number of samples

# Define a range of positive and negative correlation values
correlation_values = [-0.8, -0.5, -0.3, 0.8]  # Negative and positive correlations included

# Set up plot parameters for better visibility
sns.set(font_scale=1.2)  # Increase font size

# Create a subplot for each correlation value with heatmap and 3D histogram
fig, axes = plt.subplots(2, 2, figsize=(15, 12), subplot_kw={'projection': None})
axes = axes.flatten()

fig_3d = plt.figure(figsize=(15, 12))  # Separate figure for 3D plots

for i, corr in enumerate(correlation_values):
    # Generate independent geometric samples Z11, Z12, Z21, and Z22
    Z11 = geom.rvs(p_success1, size=num_samples).astype(int)
    Z12 = geom.rvs(p_success1, size=num_samples).astype(int)
    Z21 = geom.rvs(p_success2, size=num_samples).astype(int)
    Z22 = geom.rvs(p_success2, size=num_samples).astype(int)

    # Calculate the differences X1 and X2
    X1 = Z11 - Z12
    X2 = Z21 - Z22

    # Generate correlated uniform samples using the copula
    normal_samples = np.random.multivariate_normal(
        [0, 0], [[1, corr], [corr, 1]], num_samples
    )
    u_samples_X1 = norm.cdf(normal_samples[:, 0])
    u_samples_X2 = norm.cdf(normal_samples[:, 1])

    # Rank-transform X1 and X2 to impose the copula correlation
    X1_sorted = np.sort(X1)
    X2_sorted = np.sort(X2)
    X1_correlated = X1_sorted[np.argsort(np.argsort(u_samples_X1))]
    X2_correlated = X2_sorted[np.argsort(np.argsort(u_samples_X2))]

    # Create a 2D histogram for the joint distribution, with bins covering -10 to 10
    bins_range = range(-10, 11)  # Bins from -10 to 10 inclusive
    heatmap_data, x_edges, y_edges = np.histogram2d(
        X1_correlated, X2_correlated, bins=[bins_range, bins_range]
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
        f"Heatmap of Correlated Geometric Differences\n(Correlation={corr})",
        fontsize=16,
    )
    axes[i].set_xlabel("Correlated Geometric Difference X1", fontsize=14)
    axes[i].set_ylabel("Correlated Geometric Difference X2", fontsize=14)
    axes[i].set_xticks(np.arange(len(bins_range)) + 0.5)
    axes[i].set_yticks(np.arange(len(bins_range)) + 0.5)
    axes[i].invert_yaxis()  # To match the orientation of the heatmap with the axes

    # 3D Histogram Plot
    ax3d = fig_3d.add_subplot(2, 2, i + 1, projection='3d')
    xpos, ypos = np.meshgrid(x_edges[:-1] + 0.5, y_edges[:-1] + 0.5, indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = np.zeros_like(xpos)

    # Flatten the histogram data for plotting
    dx = dy = 0.8 * np.ones_like(zpos)
    dz = heatmap_data.ravel()

    # Plot the 3D bars
    ax3d.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average', color="blue", alpha=0.6)
    ax3d.set_title(f"3D Histogram of Correlated Geometric Differences\n(Correlation={corr})", fontsize=10)
    ax3d.set_xlabel("Correlated Geometric Difference X1", fontsize=10)
    ax3d.set_ylabel("Correlated Geometric Difference X2", fontsize=10)
    ax3d.set_zlabel("Frequency", fontsize=10)

# Adjust layout for clear visualization
plt.tight_layout()
plt.show()
