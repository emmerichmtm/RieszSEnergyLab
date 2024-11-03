import numpy as np
import pygame

# Initialize Pygame
pygame.init()

# Parameters
width, height = 800, 400
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Riesz S-Energy Gradient Descent")

# Colors
white = (255, 255, 255)
black = (0, 0, 0)
blue = (0, 0, 255)

# Parameters for Riesz s-Energy
s = 2  # Exponent in the Riesz s-Energy
num_points = 6
learning_rate = 0.005
padding = 50  # Padding for coordinate system boundaries

# Generate random points in the unit square
points = np.random.uniform(0, 1, (num_points, 2))


# Function to calculate Riesz s-Energy
def riesz_s_energy(points, s):
    energy = 0
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dist = np.linalg.norm(points[i] - points[j])
            if dist != 0:
                energy += 1 / (dist ** s)
    return energy


# Function to compute the gradient of Riesz s-Energy
def riesz_s_energy_gradient(points, s):
    grad = np.zeros_like(points)
    for i in range(len(points)):
        for j in range(len(points)):
            if i != j:
                diff = points[i] - points[j]
                dist = np.linalg.norm(diff)
                if dist != 0:
                    grad[i] += (s * diff) / (dist ** (s + 2))
    return grad


# Function to compute the Jacobian for the set (simplified as identity matrix in this case)
def compute_jacobian(points):
    # Assuming an identity Jacobian for simplicity here; customize as needed for specific mappings
    return np.eye(2)


# Main loop
running = True
clock = pygame.time.Clock()
while running:
    # Compute the gradient of the Riesz s-Energy
    gradient = riesz_s_energy_gradient(points, s)

    # Apply the Jacobian to the gradient (here, assuming an identity Jacobian for simplicity)
    for i in range(num_points):
        jacobian = compute_jacobian(points[i])
        movement = jacobian @ gradient[i]  # Matrix multiplication with the Jacobian
        points[i] -= learning_rate * movement  # Move point based on gradient

    # Keep points within the bounds [0, 1]
    points = np.clip(points, 0, 1)

    # Draw everything
    screen.fill(white)
    for i, point in enumerate(points):
        x_pos = int(padding + point[0] * (width / 2 - padding))
        y_pos = int(height - (padding + point[1] * (height - 2 * padding)))
        pygame.draw.circle(screen, blue, (x_pos, y_pos), 5)

    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    pygame.display.flip()
    clock.tick(30)

pygame.quit()
