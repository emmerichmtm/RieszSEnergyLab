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

# Compute Riesz s-energy for each triple of interior points combined with the vertices
results = []
for idx, (p1, p2, p3) in enumerate(combinations(interior_points, 3)):
    # Form the set including vertices and the triple
    points_set = np.vstack([vertices, p1, p2, p3])
    # Calculate the Riesz s-energy for this configuration
    energy = riesz_energy(points_set, s)
    # Store the triple and its corresponding energy
    results.append(((p1, p2, p3), energy))

# Sort the results based on energy and select the lowest-energy triples
results = sorted(results, key=lambda x: x[1])[:3]

# Visualization
plt.figure(figsize=(8, 7))

# Plot the triangle
triangle = plt.Polygon(vertices, fill=None, edgecolor='black', linewidth=1.5)
plt.gca().add_patch(triangle)

# Plot all interior points
plt.scatter(interior_points[:, 0], interior_points[:, 1], color='gray', alpha=0.5, label='Interior Points')

# Highlight the minimum energy triples
colors = plt.cm.rainbow(np.linspace(0, 1, len(results)))
for idx, ((p1, p2, p3), energy) in enumerate(results):
    triangle_points = np.array([p1, p2, p3])
    polygon = plt.Polygon(triangle_points, fill=None, edgecolor=colors[idx],
                          linestyle='-', linewidth=2,
                          label=f"Triple {idx + 1}: Energy={energy:.2f}")
    plt.gca().add_patch(polygon)
    # Plot the points
    plt.scatter(triangle_points[:, 0], triangle_points[:, 1], color=colors[idx], s=50)

# Plot the vertices
plt.scatter(vertices[:, 0], vertices[:, 1], color='red', s=100, label='Vertices')

# Label the vertices
vertex_labels = ['v1 (0,0)', 'v2 (1,0)', 'v3 (0.5, √3/2)']
# Uncomment the following lines if you wish to label the vertices
# for i, txt in enumerate(vertex_labels):
#     plt.annotate(txt, (vertices[i, 0], vertices[i, 1]), textcoords="offset points",
#                  xytext=(0,10), ha='center', fontsize=12, color='blue')

plt.title('Visualization of Lowest Riesz s-Energy Triples on a 2D Simplex'1
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
plt.gca().set_aspect('equal', adjustable='box')
plt.grid(True)
plt.tight_layout()
plt.show()
