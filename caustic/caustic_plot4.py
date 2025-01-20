import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
from matplotlib.patches import Ellipse

def lens_equation(theta, theta_E, gamma, beta):
    """
    Defines the lens equation for a point mass with external shear.

    Parameters:
    - theta: Array-like, [theta_x, theta_y], image position.
    - theta_E: Einstein radius.
    - gamma: External shear parameter.
    - beta: Array-like, [beta_x, beta_y], source position.

    Returns:
    - Array-like, residuals of the lens equation.
    """
    theta_x, theta_y = theta
    beta_x, beta_y = beta

    # Avoid division by zero
    theta_sq = theta_x**2 + theta_y**2
    if theta_sq == 0:
        theta_sq = 1e-8

    # Deflection by point mass
    alpha_x = theta_E**2 * theta_x / theta_sq
    alpha_y = theta_E**2 * theta_y / theta_sq

    # External shear (assuming shear aligned with x-axis)
    alpha_shear_x = gamma * theta_x
    alpha_shear_y = -gamma * theta_y

    # Total deflection
    alpha_total_x = alpha_x + alpha_shear_x
    alpha_total_y = alpha_y + alpha_shear_y

    # Lens equation
    return [theta_x - alpha_total_x - beta_x,
            theta_y - alpha_total_y - beta_y]

def find_images(beta, theta_E, gamma, initial_guesses):
    """
    Finds all image positions for a given source position using multiple initial guesses.

    Parameters:
    - beta: Array-like, [beta_x, beta_y], source position.
    - theta_E: Einstein radius.
    - gamma: External shear parameter.
    - initial_guesses: List of initial guess positions for image finding.

    Returns:
    - List of image positions [ [theta_x1, theta_y1], [theta_x2, theta_y2], ... ]
    """
    images = []
    for guess in initial_guesses:
        sol = root(lens_equation, guess, args=(theta_E, gamma, beta), method='hybr')
        if sol.success:
            theta = sol.x
            # Check for duplicates within a tolerance
            if not any(np.allclose(theta, img, atol=1e-4) for img in images):
                images.append(theta)
    return images

def compute_caustics_ray_tracing(theta_E=1.0, gamma=0.3, grid_size=200, grid_range=3.0):
    """
    Computes the caustic map using ray-tracing by identifying source positions
    where the number of images changes (i.e., caustic crossings).

    Parameters:
    - theta_E: Einstein radius.
    - gamma: External shear parameter.
    - grid_size: Number of points along each dimension in the source plane grid.
    - grid_range: Spatial range for beta_x and beta_y (-range to +range).

    Returns:
    - beta_x_caustic, beta_y_caustic: Arrays of caustic source positions.
    """
    # Create a fine grid of source positions
    beta_x = np.linspace(-grid_range, grid_range, grid_size)
    beta_y = np.linspace(-grid_range, grid_range, grid_size)
    Beta_X, Beta_Y = np.meshgrid(beta_x, beta_y)
    Beta_X_flat = Beta_X.flatten()
    Beta_Y_flat = Beta_Y.flatten()

    # Initialize lists to store caustic points
    caustic_points_x = []
    caustic_points_y = []

    # Define initial guesses for image positions
    # Based on symmetric assumptions, initial guesses can be around +/- theta_E
    initial_guesses = [
        [theta_E, 0.0],
        [-theta_E, 0.0],
        [0.0, theta_E],
        [0.0, -theta_E]
    ]

    # Previous number of images
    prev_num_images = None

    # Iterate over source grid
    for i in range(len(Beta_X_flat)):
        beta = [Beta_X_flat[i], Beta_Y_flat[i]]
        images = find_images(beta, theta_E, gamma, initial_guesses)
        num_images = len(images)

        if prev_num_images is not None and num_images != prev_num_images:
            # Source position is near a caustic
            caustic_points_x.append(beta[0])
            caustic_points_y.append(beta[1])

        prev_num_images = num_images

    return np.array(caustic_points_x), np.array(caustic_points_y)

def plot_caustics_ray_tracing(caustic_x, caustic_y, theta_E=1.0, gamma=0.3):
    """
    Plots the caustic map obtained via ray-tracing.

    Parameters:
    - caustic_x, caustic_y: Arrays of caustic source positions.
    - theta_E: Einstein radius.
    - gamma: External shear parameter.
    """
    plt.figure(figsize=(8, 8))
    plt.scatter(caustic_x, caustic_y, s=1, color='blue', label='Caustics')
    plt.title(f'Caustic Map via Ray-Tracing (γ = {gamma})')
    plt.xlabel(r'$\beta_x$')
    plt.ylabel(r'$\beta_y$')
    plt.axhline(0, color='k', lw=0.5)
    plt.axvline(0, color='k', lw=0.5)
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.show()

def generate_sample_sources(theta_E=1.0, gamma=0.3, caustic_x=None, caustic_y=None, num_samples=5):
    """
    Generates and plots sample source positions near the caustics.

    Parameters:
    - theta_E: Einstein radius.
    - gamma: External shear parameter.
    - caustic_x, caustic_y: Arrays of caustic source positions.
    - num_samples: Number of sample sources to generate around the caustic.
    """
    if caustic_x is None or caustic_y is None:
        print("Caustic points are required to generate sample sources.")
        return

    # Select random points near the caustic
    idx = np.random.choice(len(caustic_x), size=num_samples, replace=False)
    sample_sources = np.vstack((caustic_x[idx], caustic_y[idx])).T

    # Plot the caustic map and sample sources
    plt.figure(figsize=(8, 8))
    plt.scatter(caustic_x, caustic_y, s=1, color='lightblue', label='Caustics')
    plt.scatter(sample_sources[:,0], sample_sources[:,1], color='red', marker='x', s=100, label='Sample Sources')
    plt.title(f'Sample Sources Near Caustics (γ = {gamma})')
    plt.xlabel(r'$\beta_x$')
    plt.ylabel(r'$\beta_y$')
    plt.axhline(0, color='k', lw=0.5)
    plt.axvline(0, color='k', lw=0.5)
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.show()

def compute_caustics_direct(theta_E=1.0, gamma=0.3, grid_size=500, grid_range=3.0):
    """
    An alternative method to compute caustics by directly mapping critical curves.

    Parameters:
    - theta_E: Einstein radius.
    - gamma: External shear parameter.
    - grid_size: Number of points to parametrize the critical curve.
    - grid_range: Spatial range for theta_x and theta_y.

    Returns:
    - beta_x_caustic, beta_y_caustic: Arrays of caustic source positions.
    """
    # Parametrize critical curve in image plane
    angles = np.linspace(0, 2*np.pi, grid_size)
    # For simplicity, assume critical curve is roughly circular; refine based on lens model
    theta_x = theta_E * np.cos(angles)
    theta_y = theta_E * np.sin(angles)

    # Adjust for shear
    # In reality, critical curves are distorted by shear; for a simple model, apply linear approximation
    theta_x *= (1 + gamma)
    theta_y *= (1 - gamma)

    # Map critical curve to source plane via lens equation
    beta_x = theta_x - (theta_E**2 * theta_x) / (theta_x**2 + theta_y**2) - gamma * theta_x
    beta_y = theta_y - (theta_E**2 * theta_y) / (theta_x**2 + theta_y**2) + gamma * theta_y

    return beta_x, beta_y

def plot_caustics_direct(beta_x, beta_y, theta_E=1.0, gamma=0.3):
    """
    Plots the caustic map obtained via direct critical curve mapping.

    Parameters:
    - beta_x, beta_y: Arrays of caustic source positions.
    - theta_E: Einstein radius.
    - gamma: External shear parameter.
    """
    plt.figure(figsize=(8, 8))
    plt.plot(beta_x, beta_y, color='red', lw=2, label='Caustic Curve')
    plt.title(f'Caustic Map via Direct Mapping (γ = {gamma})')
    plt.xlabel(r'$\beta_x$')
    plt.ylabel(r'$\beta_y$')
    plt.axhline(0, color='k', lw=0.5)
    plt.axvline(0, color='k', lw=0.5)
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.show()

def main():
    # Define lens parameters
    theta_E = 1.0  # Einstein radius
    gamma = 0.3    # External shear

    # Compute caustics via ray-tracing
    print("Computing caustics via ray-tracing...")
    caustic_x, caustic_y = compute_caustics_ray_tracing(theta_E=theta_E, gamma=gamma, grid_size=200, grid_range=3.0)

    # Plot caustic map
    print("Plotting caustic map via ray-tracing...")
    plot_caustics_ray_tracing(caustic_x, caustic_y, theta_E=theta_E, gamma=gamma)

    # Optionally, plot sample sources near caustics
    print("Generating sample sources near caustics...")
    generate_sample_sources(theta_E=theta_E, gamma=gamma, caustic_x=caustic_x, caustic_y=caustic_y, num_samples=10)

    # Compute caustics via direct critical curve mapping
    print("Computing caustics via direct mapping...")
    beta_x_direct, beta_y_direct = compute_caustics_direct(theta_E=theta_E, gamma=gamma, grid_size=1000, grid_range=3.0)

    # Plot caustic map via direct mapping
    print("Plotting caustic map via direct mapping...")
    plot_caustics_direct(beta_x_direct, beta_y_direct, theta_E=theta_E, gamma=gamma)

if __name__ == "__main__":
    main()