import pygame
import numpy as np

# Initialize Pygame
pygame.init()

# Screen dimensions and settings
width, height = 800, 400
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Minimizing Riesz s-Energy with Rescaled Target Space")

# Colors
white = (255, 255, 255)
black = (0, 0, 0)
blue = (0, 0, 255)
red = (255, 0, 0)
gray = (200, 200, 200)

# Parameters
num_points = 6  # k=6 points
radius = 5
padding = 50
learning_rate = 0.005  # Reduced learning rate for smoother adjustments
s = 2  # Exponent for Riesz s-Energy
repulsive_strength = 0.1  # Small repulsive force to prevent clumping

# Initialize random points in the range [0, 1] for (x1, x2)
x_points = np.random.uniform(0, 1, (num_points, 2))

# Define transformation functions
def f1(p1, p2):
    return p1 ** 2 + p2 ** 2

def f2(p1, p2):
    return (p1 - 1) ** 2 + (p2 - 1) ** 2

# Define the Jacobian matrix of the transformation
def jacobian_matrix(p1, p2):
    df1_dp1 = 2 * p1
    df1_dp2 = 2 * p2
    df2_dp1 = 2 * (p1 - 1)
    df2_dp2 = 2 * (p2 - 1)
    return np.array([[df1_dp1, df1_dp2], [df2_dp1, df2_dp2]])

# Compute Riesz s-Energy in (y1, y2) space
def riesz_s_energy(y_points, s):
    energy = 0
    for i in range(len(y_points)):
        for j in range(i + 1, len(y_points)):
            dist = np.linalg.norm(y_points[i] - y_points[j])
            if dist != 0:
                energy += 1 / (dist ** s)
    return energy

# Compute the negative gradient of Riesz s-Energy for each point in (y1, y2)
def riesz_s_energy_gradient(y_points, s):
    grad_y = np.zeros_like(y_points)
    for i in range(len(y_points)):
        for j in range(len(y_points)):
            if i != j:
                diff = y_points[i] - y_points[j]
                dist = np.linalg.norm(diff)
                if dist != 0:
                    # Negative gradient for minimization, plus a small repulsive force
                    grad_y[i] -= (s * diff / (dist ** (s + 2))) - repulsive_strength * diff / (dist ** 2)
    return grad_y

# Project gradient from (y1, y2) to (x1, x2) using the Jacobian
def project_gradient_to_x(points, grad_y):
    grad_x = np.zeros_like(points)
    for i, (p1, p2) in enumerate(points):
        jacobian = jacobian_matrix(p1, p2)
        try:
            inv_jacobian = np.linalg.inv(jacobian)
            grad_x[i] = inv_jacobian @ grad_y[i]  # Map to (x1, x2)
        except np.linalg.LinAlgError:
            grad_x[i] = np.zeros_like(grad_y[i])  # Handle singular Jacobian by skipping
    return grad_x

# Scaling functions to fit the unit square [0,1] to screen coordinates
def scale_x(x, left=True):
    if left:
        return int(padding + x * (width / 4 - padding))
    else:
        return int(width / 2 + padding + x * (width / 4 - padding))

def scale_y(y):
    return int(height - (padding + y * (height - 2 * padding)))

# Dynamic rescaling for the y1, y2 space
def dynamic_rescale_y(y_points):
    # Find min and max values for y1 and y2 to scale within bounds
    min_y = np.min(y_points, axis=0)
    max_y = np.max(y_points, axis=0)
    y_range = max_y - min_y
    y_range[y_range == 0] = 1  # Avoid division by zero in case of singular range

    # Scale y points to fit within [0, 1] for display
    return (y_points - min_y) / y_range

# Main loop
running = True
clock = pygame.time.Clock()
while running:
    # Map points to (y1, y2) space
    y_points = np.array([[f1(p[0], p[1]), f2(p[0], p[1])] for p in x_points])

    # Compute the negative gradient of the Riesz s-Energy in (y1, y2) space
    grad_y = riesz_s_energy_gradient(y_points, s)

    # Project the negative gradient from (y1, y2) to (x1, x2) using the Jacobian
    grad_x = project_gradient_to_x(x_points, grad_y)

    # Update points based on the negative projected gradient
    x_points -= learning_rate * grad_x  # Move points in the negative gradient direction

    # Ensure points stay within bounds [0, 1] for both x1 and x2
    x_points = np.clip(x_points, 0, 1)

    # Rescale y_points dynamically for display
    scaled_y_points = dynamic_rescale_y(y_points)

    # Drawing
    screen.fill(white)

    # Draw coordinate axes for the left plot (x1, x2)
    pygame.draw.line(screen, gray, (padding, height - padding), (width / 2 - padding, height - padding), 1)
    pygame.draw.line(screen, gray, (padding, padding), (padding, height - padding), 1)

    # Draw coordinate axes for the right plot (y1, y2)
    pygame.draw.line(screen, gray, (width / 2 + padding, height - padding), (width - padding, height - padding), 1)
    pygame.draw.line(screen, gray, (width / 2 + padding, padding), (width / 2 + padding, height - padding), 1)

    # Draw original points (x1, x2) on the left side
    for i, (p1, p2) in enumerate(x_points):
        x_pos = scale_x(p1, left=True)
        y_pos = scale_y(p2)
        pygame.draw.circle(screen, blue, (x_pos, y_pos), radius)

    # Draw mapped points (y1, y2) on the right side after rescaling
    for i, (q1, q2) in enumerate(scaled_y_points):
        x_pos = scale_x(q1, left=False)
        y_pos = scale_y(q2)
        pygame.draw.circle(screen, red, (x_pos, y_pos), radius)

    # Draw lines connecting each (p1, p2) to (q1, q2)
    for i in range(num_points):
        start_pos = (scale_x(x_points[i, 0], left=True), scale_y(x_points[i, 1]))
        end_pos = (scale_x(scaled_y_points[i, 0], left=False), scale_y(scaled_y_points[i, 1]))
        pygame.draw.line(screen, black, start_pos, end_pos, 1)

    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    pygame.display.flip()
    clock.tick(30)  # Control the frame rate

# Quit Pygame
pygame.quit()
