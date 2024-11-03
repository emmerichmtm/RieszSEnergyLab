import pygame
import numpy as np

# Initialize Pygame
pygame.init()

# Screen dimensions and settings
width, height = 800, 400
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Minimizing Riesz s-Energy with Normalized Axis Scaling")

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
s = 1  # Exponent for Riesz s-Energy
repulsive_strength = 0.1  # Small repulsive force to prevent clumping

# Initialize random points in the range [0, 1] for (x1, x2)
x_points = np.random.uniform(0, 1, (num_points, 2))

# Font for axis labels
pygame.font.init()
font = pygame.font.SysFont("Arial", 12)


# Define the identity transformation functions
def f1(p1, p2):
    return p1


def f2(p1, p2):
    return p2


# Define the Jacobian matrix for the identity transformation
def jacobian_matrix(p1, p2):
    return np.array([[1, 0], [0, 1]])


# Compute Riesz s-Energy in (y1, y2) space
def riesz_s_energy(y_points, s):
    energy = 0
    for i in range(len(y_points)):
        for j in range(i + 1, len(y_points)):
            dist = np.linalg.norm(y_points[i] - y_points[j])
            if dist != 0:
                energy += 1 / (dist ** s)
    return energy


# Compute the normalized negative gradient of Riesz s-Energy for each point in (y1, y2)
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

    # Normalize the gradients
    norms = np.linalg.norm(grad_y, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero for zero gradients
    grad_y_normalized = grad_y / norms
    return grad_y_normalized


# Project gradient from (y1, y2) to (x1, x2) using the Jacobian
def project_gradient_to_x(points, grad_y):
    grad_x = np.zeros_like(points)
    for i, (p1, p2) in enumerate(points):
        jacobian = jacobian_matrix(p1, p2)
        grad_x[i] = jacobian @ grad_y[i]  # Apply the Jacobian (identity here)
    return grad_x


# Scaling functions to fit the unit square [0,1] to screen coordinates
def scale_x(x, left=True):
    if left:
        return int(padding + x * (width / 2 - 2 * padding))
    else:
        return int(width / 2 + padding + x * (width / 2 - 2 * padding))


def scale_y(y):
    return int(height - (padding + y * (height - 2 * padding)))


# Function to render axis labels
def draw_axis_labels():
    # Labels for x1, x2 space
    for i in range(5):
        label_value = i * 0.25
        label = font.render(f"{label_value:.2f}", True, black)
        x = scale_x(label_value, left=True)
        y = height - padding + 5
        screen.blit(label, (x - 5, y))

        label_y = font.render(f"{label_value:.2f}", True, black)
        x = padding - 25
        y = scale_y(label_value)
        screen.blit(label_y, (x, y - 5))

    # Labels for y1, y2 space
    for i in range(5):
        label_value = i * 0.25
        label = font.render(f"{label_value:.2f}", True, black)
        x = scale_x(label_value, left=False)
        y = height - padding + 5
        screen.blit(label, (x - 5, y))

        label_y = font.render(f"{label_value:.2f}", True, black)
        x = width // 2 + padding - 25
        y = scale_y(label_value)
        screen.blit(label_y, (x, y - 5))


# Main loop
running = True
clock = pygame.time.Clock()
while running:
    # Map points to (y1, y2) space using the identity functions
    y_points = np.array([[f1(p[0], p[1]), f2(p[0], p[1])] for p in x_points])

    # Compute the normalized negative gradient of the Riesz s-Energy in (y1, y2) space
    grad_y = riesz_s_energy_gradient(y_points, s)

    # Project the negative gradient from (y1, y2) to (x1, x2) using the Jacobian
    grad_x = project_gradient_to_x(x_points, grad_y)

    # Update points based on the normalized negative projected gradient
    x_points -= learning_rate * grad_x  # Move points in the negative gradient direction

    # Ensure points stay within bounds [0, 1] for both x1 and x2
    x_points = np.clip(x_points, 0, 1)

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

    # Draw mapped points (y1, y2) on the right side
    for i, (q1, q2) in enumerate(y_points):
        x_pos = scale_x(q1, left=False)
        y_pos = scale_y(q2)
        pygame.draw.circle(screen, red, (x_pos, y_pos), radius)

    # Draw lines connecting each (p1, p2) to (q1, q2)
    for i in range(num_points):
        start_pos = (scale_x(x_points[i, 0], left=True), scale_y(x_points[i, 1]))
        end_pos = (scale_x(y_points[i, 0], left=False), scale_y(y_points[i, 1]))
        pygame.draw.line(screen, black, start_pos, end_pos, 1)

    # Draw axis labels
    draw_axis_labels()

    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    pygame.display.flip()
    clock.tick(30)  # Control the frame rate

# Quit Pygame
pygame.quit()
