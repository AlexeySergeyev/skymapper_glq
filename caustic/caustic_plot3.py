import numpy as np
import pygame
import sys
from pygame.locals import QUIT, MOUSEBUTTONDOWN, MOUSEMOTION

# Define the 2D Gaussian Function
def gaussian_2d(x, y, x0, y0, sigma):
    """
    Computes the 2D Gaussian function value at (x, y).

    Parameters:
    - x (float or np.ndarray): X-coordinate(s).
    - y (float or np.ndarray): Y-coordinate(s).
    - x0 (float): X-coordinate of the Gaussian center.
    - y0 (float): Y-coordinate of the Gaussian center.
    - sigma (float): Standard deviation of the Gaussian.

    Returns:
    - value (float or np.ndarray): Gaussian function value(s) at the given coordinate(s).
    """
    return np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

# Normalization Function (similar to Matplotlib's simple_norm)
def normalize_image(image, stretch='linear', percent=99.9):
    """
    Normalizes the image data to the range [0, 255].

    Parameters:
    - image (np.ndarray): 2D array of image data.
    - stretch (str): Type of stretch ('linear').
    - percent (float): Percentile for normalization.

    Returns:
    - norm_image (np.ndarray): Normalized image scaled to [0, 255].
    """
    if stretch == 'linear':
        v_min = np.percentile(image, 100 - percent)
        v_max = np.percentile(image, percent)
        norm_image = (image - v_min) / (v_max - v_min)
        norm_image = np.clip(norm_image, 0, 1)
    else:
        raise NotImplementedError("Only 'linear' stretch is implemented.")
    norm_image = (norm_image * 255).astype(np.uint8)
    return norm_image

# Initialize Pygame
def init_pygame(window_size):
    pygame.init()
    screen = pygame.display.set_mode(window_size)
    pygame.display.set_caption("2D Gaussian Interactive Map")
    return screen

# Convert Image Data to Pygame Surface
def image_to_surface(image):
    """
    Converts a 2D numpy array to a Pygame Surface.

    Parameters:
    - image (np.ndarray): 2D array of normalized image data (0-255).

    Returns:
    - surface (pygame.Surface): Pygame Surface object.
    """
    # Create an RGB image by stacking the grayscale image
    rgb_image = np.stack((image,)*3, axis=-1)
    # Convert to Pygame Surface
    surface = pygame.surfarray.make_surface(rgb_image)
    return surface

# Main Function
def main():
    # Define Parameters
    theta_E = 2.0
    kappa = 0.2 / theta_E
    gamma = 0.3
    size = 500  # Reduced size for better performance
    mu = 1.0 / abs((1.0 - kappa)**2 - gamma**2)
    print(f"mu: {mu}")
    range_x = 4 * mu * abs(1.0 - kappa + gamma)
    range_y = 4 * mu * abs(1.0 - kappa - gamma)
    source_size = 500  # Reduced source size to match image size
    source_range_x = 3
    source_range_y = 3
    print(f"range_x: {range_x}, range_y: {range_y}")

    image_range_x = (-range_x, range_x)
    image_range_y = (-range_y, range_y)

    theta_x = np.linspace(*image_range_x, size)
    theta_y = np.linspace(*image_range_y, size)
    theta_x, theta_y = np.meshgrid(theta_x, theta_y)
    theta_square = theta_x**2 + theta_y**2

    # Avoid division by zero
    epsilon = 1e-8
    theta_square = np.where(theta_square == 0, epsilon, theta_square)

    beta_x = theta_x - ((theta_E**2 * theta_x) / theta_square + gamma * theta_x)
    beta_y = theta_y - ((theta_E**2 * theta_y) / theta_square - gamma * theta_y)

    # Compute magnification_map using histogram2d
    magnification_map, x_edges, y_edges = np.histogram2d(
        beta_x.flatten(), beta_y.flatten(),
        bins=(source_size, source_size),
        range=((-source_range_x, source_range_x), (-source_range_y, source_range_y))
    )

    # Initial Gaussian Parameters
    xs, ys = 0.4, 0.2  # Initial positions
    sigma = 0.05

    # # Normalize magnification_map
    # magn_norm = normalize_image(magnification_map, stretch='linear', percent=99.9)

    # Function to Compute and Update Image
    def compute_image(xs, ys):
        gauss_image = gaussian_2d(beta_x.flatten(), beta_y.flatten(),
                                  xs, ys, sigma=sigma)
        gauss_image = 100 * gauss_image.reshape(size, size)
        image = magnification_map + gauss_image
        norm_image = normalize_image(image, stretch='linear', percent=99.5)
        return norm_image

    # Compute Initial Image
    image = compute_image(xs, ys)
    surface = image_to_surface(image)

    # Initialize Pygame
    window_size = (size, size)
    screen = init_pygame(window_size)

    # Define Colors
    RED = (255, 0, 0)
    YELLOW = (255, 255, 0)
    WHITE = (255, 255, 255)

    # Draw Initial Image
    screen.blit(surface, (0, 0))

    # Draw Ellipse (circle with width=theta_E, height=theta_E)
    # Scale ellipse size to pixels
    # Map theta_E to image coordinates
    ellipse_width = (theta_E / (image_range_x[1] - image_range_x[0])) * size
    ellipse_height = (theta_E / (image_range_y[1] - image_range_y[0])) * size
    ellipse_rect = pygame.Rect(
        size//2 - ellipse_width//2,
        size//2 - ellipse_height//2,
        ellipse_width,
        ellipse_height
    )
    pygame.draw.ellipse(screen, RED, ellipse_rect, 2)  # 2 pixels thick

    # Draw Initial Scatter Point
    # Convert (xs, ys) to pixel coordinates
    def coord_to_pixel(x, y):
        pixel_x = int(((x - image_range_x[0]) / (image_range_x[1] - image_range_x[0])) * size)
        pixel_y = int(((image_range_y[1] - y) / (image_range_y[1] - image_range_y[0])) * size)
        return pixel_x, pixel_y

    scatter_pos = coord_to_pixel(xs, ys)
    pygame.draw.circle(screen, YELLOW, scatter_pos, 5)

    pygame.display.flip()

    # Main Loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                break
            if event.type == QUIT:
                running = False
                break
            # elif event.type == MOUSEBUTTONDOWN and event.button == 1:  # Left click
            elif event.type == MOUSEMOTION:
                mouse_x, mouse_y = event.pos
                # Convert pixel to image coordinates
                x = image_range_x[0] + (mouse_x / size) * (image_range_x[1] - image_range_x[0])
                y = image_range_y[1] - (mouse_y / size) * (image_range_y[1] - image_range_y[0])
                xs, ys = x, y
                print(f"Selected Coordinates: xs={xs:.3f}, ys={ys:.3f}")
                # Recompute Image
                image = compute_image(xs, ys)
                surface = image_to_surface(image)
                # Update Screen
                screen.blit(surface, (0, 0))
                # Redraw Ellipse
                pygame.draw.ellipse(screen, RED, ellipse_rect, 2)
                # Redraw Scatter Point
                scatter_pos = coord_to_pixel(xs, ys)
                pygame.draw.circle(screen, YELLOW, scatter_pos, 5)
                pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()