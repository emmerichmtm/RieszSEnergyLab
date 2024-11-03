import pygame
import numpy as np

# Initialize Pygame
pygame.init()

# Screen dimensions and settings
width, height = 800, 400
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Gradient-based Random Walk")

# Colors
white = (255, 255, 255)
black = (0, 0, 0)
blue = (0, 0, 255)
red = (255, 0, 0)
gray = (200, 200, 200)

# Parameters
num_points = 6
radius = 5
padding = 50
learning_rate = 0.01  # Step size for random walk

# Initialize random points in the range [0, 1] for (x1, x2)
x_points = np.random.uniform(0, 1, (num_points, 2))


# Define transformation functions
def f1(p1, p2):
    return p1 ** 2 + p2 ** 2


def f2(p1, p2):
    return (p1 - 1) ** 2 + (p2 - 1) ** 2


# Define gradient functions
def grad_f1(p1, p2):
    return np.array([2 * p1, 2 * p2])


def grad_f2(p1, p2):
    return np.array([2 * (p1 - 1), 2 * (p2 - 1)])


# Scaling functions to fit the unit square [0,1] to screen coordinates
def scale_x(x, left=True):
    if left:
        return int(padding + x * (width / 4 - padding))
    else:
        return int(width / 2 + padding + x * (width / 4 - padding))


def scale_y(y):
    return int(height - (padding + y * (height - 2 * padding)))


# Main loop
running = True
clock = pygame.time.Clock()
while running:
    # Randomly change weights (w1, w2) over time
    w1, w2 = np.random.uniform(0.5, 1.5, 2)

    # Update each point based on the gradient
    for i in range(num_points):
        p1, p2 = x_points[i]

        # Compute the weighted gradient of w1*f1 + w2*f2
        grad = w1 * grad_f1(p1, p2) + w2 * grad_f2(p1, p2)

        # Update point position with a small step in the direction of the gradient
        x_points[i] = x_points[i] - learning_rate * grad

        # Ensure points stay within bounds [0, 1] (clipping)
        x_points[i] = np.clip(x_points[i], 0, 1)

    # Map updated points using f1 and f2
    y_points = np.array([[f1(p[0], p[1]), f2(p[0], p[1])] for p in x_points])

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
        p1, p2 = x_points[i]
        q1, q2 = y_points[i]

        start_pos = (scale_x(p1, left=True), scale_y(p2))
        end_pos = (scale_x(q1, left=False), scale_y(q2))

        pygame.draw.line(screen, black, start_pos, end_pos, 1)

    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    pygame.display.flip()
    clock.tick(30)  # Control the frame rate

# Quit Pygame
pygame.quit()
