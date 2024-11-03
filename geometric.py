import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import geom

# Set the probability of success for the geometric distributions
p1 = 0.5  # Probability of success for the first geometric distribution
p2 = 0.5  # Probability of success for the second geometric distribution

# Number of samples to simulate
num_samples = 10000

# Simulate two independent geometric random variables
X = geom.rvs(p1, size=num_samples)
Y = geom.rvs(p2, size=num_samples)

# Compute the difference
D = X - Y

# Define the range for the histogram bins
min_D = np.min(D)
max_D = np.max(D)
bins = np.arange(min_D - 1, max_D + 2) - 0.5  # Offset bins by 0.5 for integer alignment

# Plot the histogram
plt.figure(figsize=(12, 6))
plt.hist(D, bins=bins, density=True, color='skyblue', edgecolor='black')

# Plot settings
plt.title('Histogram of the Difference of Two Geometric Distributions', fontsize=16)
plt.xlabel('Difference (X - Y)', fontsize=14)
plt.ylabel('Probability Density', fontsize=14)
plt.grid(True)
plt.xticks(range(int(min_D), int(max_D) + 1))
plt.show()
