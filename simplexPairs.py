import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

# Define the vertices of the 2D simplex (triangle)
v1 = np.array([0.0, 0.0])
v2 = np.array([1.0, 0.0])
v3 = np.array([0.5, np.sqrt(3) / 2])
vertices = np.array([v1, v2, v3])
s = 2.0  # Riesz s-energy parameter

# Function to compute Riesz s-energy for a set of points
def riesz_energy(points, s):
    energy = 0.0
    for i, j in combinations(range(len(points)), 2):
        dist = np.linalg.norm(points[i] - points[j])
        energy += 1 / dist**s if dist != 0 else 0
    return energy

# Generate interior points within the triangle using barycentric coordinates
num_points_per_side = 10  # Define sampling granularity
interior_points = []

for i in range(num_points_per_side + 1):
    for j in range(num_points_per_side + 1 - i):
        k = num_points_per_side - i - j
        weights = np.array([i, j, k]) / num_points_per_side
        # Compute interior point as a convex combination of vertices
        interior_point = weights[0] * v1 + weights[1] * v2 + weights[2] * v3
        # Exclude points that are too close to vertices
        if not any(np.allclose(interior_point, vertex) for vertex in vertices):
            interior_points.append(interior_point)

interior_points = np.array(interior_points)

# Compute Riesz s-energy for each pair of interior points combined with the vertices
results = []
for idx, (p1, p2) in enumerate(combinations(interior_points, 2)):
    # Form the set including vertices and the pair
    points_set = np.vstack([vertices, p1, p2])
    # Calculate the Riesz s-energy for this configuration
    energy = riesz_energy(points_set, s)
    # Store the pair and its corresponding energy
    results.append(((p1, p2), energy))

# Sort the results based on energy and select the lowest-energy pairs
results = sorted(results, key=lambda x: x[1])[:3]

# Visualization
plt.figure(figsize=(8, 7))

# Plot the triangle
triangle = plt.Polygon(vertices, fill=None, edgecolor='black', linewidth=1.5)
plt.gca().add_patch(triangle)

# Plot all interior points
plt.scatter(interior_points[:, 0], interior_points[:, 1], color='gray', alpha=0.5, label='Interior Points')

# Highlight the minimum energy pairs
colors = plt.cm.rainbow(np.linspace(0, 1, len(results)))
for idx, ((p1, p2), energy) in enumerate(results):
    x_values = [p1[0], p2[0]]
    y_values = [p1[1], p2[1]]
    plt.plot(x_values, y_values, color=colors[idx], marker='o', markersize=5,
             linestyle='-', linewidth=2, label=f"Pair {idx + 1}: Energy={energy:.2f}")

# Plot the vertices
plt.scatter(vertices[:, 0], vertices[:, 1], color='red', s=100, label='Vertices')

# Label the vertices
vertex_labels = ['v1 (0,0)', 'v2 (1,0)', 'v3 (0.5, √3/2)']
#for i, txt in enumerate(vertex_labels):
#    plt.annotate(txt, (vertices[i, 0], vertices[i, 1]), textcoords="offset points",
#                 xytext=(0,10), ha='center', fontsize=12, color='blue')

plt.title('Visualization of Lowest Riesz s-Energy Pairs on a 2D Simplex')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
plt.gca().set_aspect('equal', adjustable='box')
plt.grid(True)
plt.tight_layout()
plt.show()
