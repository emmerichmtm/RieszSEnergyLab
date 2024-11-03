import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

# Define the vertices of the simplex (triangle in 2D)
vertices = np.array([
    [1, 0],  # (1, 0, 0)
    [0.5, np.sqrt(3) / 2],  # (0, 1, 0)
    [0, 0]   # (0, 0, 1)
])

# Create a flat list of barycentric coordinates within the triangle
num_points = 50
barycentric_coords = []
for i in range(num_points + 1):
    for j in range(num_points - i + 1):
        k = num_points - i - j
        barycentric_coords.append((i / num_points, j / num_points, k / num_points))

barycentric_coords = np.array(barycentric_coords)

# Map barycentric coordinates to 2D Cartesian coordinates for plotting
cartesian_coords = barycentric_coords @ vertices

# Choose a reference point, e.g., the centroid of the triangle
reference_point = np.mean(vertices, axis=0)

# Define a Riesz s-energy function with respect to a reference point
def riesz_s_energy(x, y, ref_point, s=2):
    """Compute the Riesz s-energy of a point (x, y) relative to ref_point for the given s."""
    # Calculate the distance from the reference point
    distance = np.sqrt((x - ref_point[0])**2 + (y - ref_point[1])**2)
    if distance < 1e-10:
        return 1e10  # Large finite value for very small distances
    return np.log(1+1 / (distance**s))

# Calculate the Riesz s-energy for each point in the Cartesian grid
s = 2  # Define s parameter for Riesz energy
energy_values = np.array([riesz_s_energy(x, y, reference_point, s) for x, y in cartesian_coords])

# Normalize the energy values to enhance visualization
energy_values = (energy_values - energy_values.min()) / (energy_values.max() - energy_values.min())

# Triangulate the grid for smooth plotting
tri = Delaunay(cartesian_coords)

# Plot the heatmap over the simplex
plt.figure(figsize=(8, 7))
plt.tricontourf(cartesian_coords[:, 0], cartesian_coords[:, 1], tri.simplices, energy_values, levels=100, cmap="viridis")
plt.colorbar(label=f"Normalized Riesz s-energy (s={s})")
plt.title(f"Normalized Riesz s-energy Heatmap over Unit Simplex (s={s})")
plt.scatter(vertices[:, 0], vertices[:, 1], color="red")  # Mark vertices
for i, txt in enumerate(["(1,0,0)", "(0,1,0)", "(0,0,1)"]):
    plt.annotate(txt, (vertices[i, 0], vertices[i, 1]), ha="center", fontsize=12, color="white")
plt.xlim(-0.1, 1.1)
plt.ylim(-0.1, np.sqrt(3)/2 + 0.1)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
