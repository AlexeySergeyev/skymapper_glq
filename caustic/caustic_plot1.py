import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.path import Path
from matplotlib.collections import LineCollection

def compute_caustics(theta_E=1.0, gamma=0.1, grid_size=1000, grid_range=3.0):
    """
    Compute the caustic map for a point-like lens with external shear.

    Parameters:
    - theta_E: Einstein radius of the lens.
    - gamma: External shear magnitude (0 <= gamma < 1).
    - grid_size: Number of points in each dimension for the grid.
    - grid_range: Range of theta_x and theta_y (-grid_range to grid_range).

    Returns:
    - beta_x_caustic, beta_y_caustic: Coordinates of the caustic curves in source plane.
    """

    # Define grid in image plane
    theta_x = np.linspace(-grid_range, grid_range, grid_size)
    theta_y = np.linspace(-grid_range, grid_range, grid_size)
    theta_X, theta_Y = np.meshgrid(theta_x, theta_y)

    # Compute |theta|^2
    theta_squared = theta_X**2 + theta_Y**2

    # Avoid division by zero
    epsilon = 1e-8
    theta_squared = np.where(theta_squared == 0, epsilon, theta_squared)

    # Lens equation: beta = theta - (theta_E^2 / theta^2) * theta - gamma * theta
    # Assuming shear is aligned with x-axis for simplicity
    beta_X = theta_X - (theta_E**2 * theta_X / theta_squared) - gamma * theta_X
    beta_Y = theta_Y - (theta_E**2 * theta_Y / theta_squared) + gamma * theta_Y

    # Compute derivatives for Jacobian
    d_beta_X_d_theta_X = 1 - theta_E**2 * (theta_squared - 2 * theta_X**2) / (theta_squared**2) - gamma
    d_beta_X_d_theta_Y = 2 * theta_E**2 * theta_X * theta_Y / (theta_squared**2)
    d_beta_Y_d_theta_X = 2 * theta_E**2 * theta_X * theta_Y / (theta_squared**2)
    d_beta_Y_d_theta_Y = 1 - theta_E**2 * (theta_squared - 2 * theta_Y**2) / (theta_squared**2) + gamma

    # Compute Jacobian determinant
    det_J = (d_beta_X_d_theta_X * d_beta_Y_d_theta_Y) - (d_beta_X_d_theta_Y * d_beta_Y_d_theta_X)

    # Find critical curves where det_J = 0
    # Use contour to find zero contour
    plt.figure(figsize=(8, 6))
    CS = plt.contour(theta_X, theta_Y, det_J, levels=[0], colors='none')
    plt.close()  # Close the plot as we don't need to display it here

    # Extract the paths of the critical curves
    critical_curves = []
    for collection in CS.collections:
        for path in collection.get_paths():
            v = path.vertices
            critical_curves.append(v)

    # Map critical curves to source plane using lens equation
    caustic_curves = []
    for curve in critical_curves:
        theta_x_curve = curve[:,0]
        theta_y_curve = curve[:,1]
        beta_x_curve = theta_x_curve - (theta_E**2 * theta_x_curve / (theta_x_curve**2 + theta_y_curve**2 + epsilon)) - gamma * theta_x_curve
        beta_y_curve = theta_y_curve - (theta_E**2 * theta_y_curve / (theta_x_curve**2 + theta_y_curve**2 + epsilon)) + gamma * theta_y_curve
        caustic_curves.append(np.vstack((beta_x_curve, beta_y_curve)).T)

    return caustic_curves

def plot_caustics(caustic_curves, title='Caustic Map of a Point Lens with Shear'):
    """
    Plot the caustic curves in the source plane.

    Parameters:
    - caustic_curves: List of arrays containing caustic curve coordinates.
    - title: Title of the plot.
    """
    plt.figure(figsize=(8, 8))
    for curve in caustic_curves:
        plt.plot(curve[:,0], curve[:,1], 'b-', lw=2)

    plt.title(title)
    plt.xlabel(r'$\beta_x$')
    plt.ylabel(r'$\beta_y$')
    plt.axhline(0, color='k', lw=0.5)
    plt.axvline(0, color='k', lw=0.5)
    plt.axis('equal')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Parameters
    theta_E = 1.0  # Einstein radius
    gamma = 0.3     # External shear (0 < gamma < 1)

    # Compute caustics
    caustic_curves = compute_caustics(theta_E=theta_E, gamma=gamma, grid_size=2000, grid_range=3.0)

    # Plot caustics
    plot_caustics(caustic_curves, title=f'Caustic Map (gamma={gamma})')