import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting

# Simulate correlated Bernoulli process
def simulate_correlated_bernoulli(N, p1, p2, rho):
    mean = [0, 0]
    cov = [[1, rho], [rho, 1]]
    normal_samples = np.random.multivariate_normal(mean, cov, N)

    bernoulli_samples = np.zeros_like(normal_samples)
    bernoulli_samples[:, 0] = (normal_samples[:, 0] < np.percentile(normal_samples[:, 0], p1 * 100)).astype(int)
    bernoulli_samples[:, 1] = (normal_samples[:, 1] < np.percentile(normal_samples[:, 1], p2 * 100)).astype(int)

    df = pd.DataFrame(bernoulli_samples, columns=['Bernoulli_Variable_1', 'Bernoulli_Variable_2'])

    return df

# Compute the first success times c1 and c2
def compute_first_success_times(df):
    c1 = 0
    c2 = 0
    found_X1 = False
    found_X2 = False

    for i in range(len(df)):
        if df['Bernoulli_Variable_1'][i] == 1 and not found_X1:
            found_X1 = True
        else:
            if not found_X1:
                c1 += np.random.choice([-1, 1])

        if df['Bernoulli_Variable_2'][i] == 1 and not found_X2:
            found_X2 = True
        else:
            if not found_X2:
                c2 += np.random.choice([-1, 1])

    return c1, c2

# Main code to generate samples and plot 3D histogram and heatmap
def main():
    N = 100  # Number of steps in each Bernoulli walk
    p1 = 0.02  # Probability for Bernoulli_Variable_1
    p2 = 0.01  # Probability for Bernoulli_Variable_2
    rho = 0.2  # Correlation coefficient

    num_samples = 10000  # Number of repetitions
    c1_samples = []
    c2_samples = []

    for _ in range(num_samples):
        df_samples = simulate_correlated_bernoulli(N, p1, p2, rho)
        c1, c2 = compute_first_success_times(df_samples)
        c1_samples.append(c1)
        c2_samples.append(c2)

    # Convert samples to numpy arrays
    c1_samples = np.array(c1_samples)
    c2_samples = np.array(c2_samples)

    # Compute 2D histogram
    bins = [30, 30]  # Number of bins for c1 and c2
    hist, xedges, yedges = np.histogram2d(c1_samples, c2_samples, bins=bins)

    # Plotting the 3D histogram
    fig = plt.figure(figsize=(12, 6))

    # 3D Histogram
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')

    # Construct arrays for the anchor positions of the bars
    xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = np.zeros_like(xpos)

    # The width and depth of each bar
    dx = dy = (xedges[1] - xedges[0])
    dz = hist.ravel()

    # Normalize colors based on height
    max_height = np.max(dz)
    rgba = plt.cm.viridis(dz / max_height)

    ax1.bar3d(xpos, ypos, zpos, dx, dy, dz, color=rgba, shade=True)
    ax1.set_xlabel('c1 Values')
    ax1.set_ylabel('c2 Values')
    ax1.set_zlabel('Frequency')
    ax1.set_title('3D Histogram of c1 and c2 Samples')

    # Heatmap
    ax2 = fig.add_subplot(1, 2, 2)

    # Plotting the heatmap
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im = ax2.imshow(hist.T, origin='lower', extent=extent, aspect='auto', cmap='viridis')
    ax2.set_xlabel('c1 Values')
    ax2.set_ylabel('c2 Values')
    ax2.set_title('Heatmap of c1 and c2 Samples')
    plt.colorbar(im, ax=ax2, label='Frequency')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
