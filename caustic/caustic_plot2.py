import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d
from scipy.optimize import root
from astropy.visualization import simple_norm

def compute_magnification_map(theta_E=1.0, gamma=0.3, image_grid_size=2000, 
                              image_grid_range=2.0, source_grid_size=500, 
                              source_grid_range=2.0):
    """
    Compute the magnification map in the source plane for a point mass lens with external shear.
    
    Parameters:
    - theta_E: Einstein radius of the lens.
    - gamma: External shear magnitude (0 <= gamma < 1).
    - image_grid_size: Number of points along each dimension in the image plane grid.
    - image_grid_range: Range of theta_x and theta_y in the image plane (-range to +range).
    - source_grid_size: Number of points along each dimension in the source plane grid.
    - source_grid_range: Range of beta_x and beta_y in the source plane (-range to +range).
    
    Returns:
    - source_x_edges, source_y_edges: Bin edges for the source plane grid.
    - magnification_map: 2D array of magnification values.
    - theta_X_flat, theta_Y_flat: Flattened image plane coordinates.
    - magnification: Flattened magnification values corresponding to image plane points.
    """
    
    # 1. Define Image Plane Grid
    theta_x = np.linspace(-image_grid_range, image_grid_range, image_grid_size)
    theta_y = np.linspace(-image_grid_range, image_grid_range, image_grid_size)
    theta_X, theta_Y = np.meshgrid(theta_x, theta_y)
    
    # Flatten the grid for vectorized computations
    theta_X_flat = theta_X.flatten()
    theta_Y_flat = theta_Y.flatten()
    
    # 2. Compute Source Plane Coordinates using Lens Equation
    theta_squared = theta_X_flat**2 + theta_Y_flat**2
    epsilon = 1e-4  # Prevent division by zero
    theta_squared = np.where(theta_squared == 0, epsilon, theta_squared)
    
    # Deflection due to mass
    alpha_mass_x = theta_E**2 * theta_X_flat / theta_squared
    alpha_mass_y = theta_E**2 * theta_Y_flat / theta_squared
    
    # Deflection due to shear (aligned with x-axis)
    alpha_shear_x = gamma * theta_X_flat
    alpha_shear_y = -gamma * theta_Y_flat
    
    # Total deflection
    alpha_x = alpha_mass_x + alpha_shear_x
    alpha_y = alpha_mass_y + alpha_shear_y
    
    # Lens Equation: beta = theta - alpha(theta)
    beta_x = theta_X_flat - alpha_x
    beta_y = theta_Y_flat - alpha_y
    
    # 3. Compute Jacobian Determinant
    # Partial derivatives
    # d(beta_x)/d(theta_x)
    dbeta_x_dtheta_x = 1 - (theta_E**2 * (theta_squared - 2 * theta_X_flat**2)) / (theta_squared**2) - gamma
    # d(beta_x)/d(theta_y)
    dbeta_x_dtheta_y = (2 * theta_E**2 * theta_X_flat * theta_Y_flat) / (theta_squared**2)
    
    # d(beta_y)/d(theta_x)
    dbeta_y_dtheta_x = (2 * theta_E**2 * theta_X_flat * theta_Y_flat) / (theta_squared**2)
    # d(beta_y)/d(theta_y)
    dbeta_y_dtheta_y = 1 - (theta_E**2 * (theta_squared - 2 * theta_Y_flat**2)) / (theta_squared**2) + gamma
    
    # Jacobian determinant
    det_J = (dbeta_x_dtheta_x * dbeta_y_dtheta_y) - (dbeta_x_dtheta_y * dbeta_y_dtheta_x)
    
    # Magnification
    # To avoid division by zero, set a lower limit on |det_J|
    det_J = np.where(det_J == 0, epsilon, det_J)
    magnification = 1.0 / np.abs(det_J)
    
    # 4. Define Source Plane Grid for Binning
    source_bins = [source_grid_size, source_grid_size]
    source_range = [[-source_grid_range, source_grid_range], 
                   [-source_grid_range, source_grid_range]]
    
    # 5. Bin Magnification onto Source Plane Grid
    # Using 2D binning with sum as the statistic
    magnification_map, x_edges, y_edges, _ = binned_statistic_2d(
        beta_x, beta_y, magnification, statistic='sum', bins=source_bins, 
        range=source_range, expand_binnumbers=True)
    
    # Handle NaN values (no images mapped to these source positions)
    magnification_map = np.nan_to_num(magnification_map)
    
    return x_edges, y_edges, magnification_map, theta_X_flat, theta_Y_flat, magnification

def plot_magnification_map(x_edges, y_edges, magnification_map, 
                          title='Caustic Magnification Map', 
                          cmap='inferno', log_scale=True):
    """
    Plot the magnification map.
    
    Parameters:
    - x_edges, y_edges: Bin edges for the source plane grid.
    - magnification_map: 2D array of magnification values.
    - title: Title of the plot.
    - cmap: Colormap to use.
    - log_scale: Whether to apply logarithmic scaling to magnification.
    """
    # Compute bin centers
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    
    # Create meshgrid for plotting
    X, Y = np.meshgrid(x_centers, y_centers)
    
    # Plotting
    plt.figure(figsize=(8, 6))
    if log_scale:
        # To handle zero magnification, add a small constant
        plt.pcolormesh(X, Y, np.log10(magnification_map + 1e-3), 
                       shading='auto', cmap=cmap)
        cbar = plt.colorbar()
        cbar.set_label(r'$\log_{10}(\mu)$')
    else:
        plt.pcolormesh(X, Y, magnification_map, shading='auto', cmap=cmap)
        cbar = plt.colorbar()
        cbar.set_label(r'$\mu$')
    
    plt.title(title)
    plt.xlabel(r'$\beta_x$')
    plt.ylabel(r'$\beta_y$')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

def generate_2d_gaussian_source(beta_x0=0.0, beta_y0=0.0, sigma_x=0.2, sigma_y=0.2, amplitude=1.0, grid_size=500, grid_range=3.0):
    """
    Generate a 2D Gaussian source.
    
    Parameters:
    - beta_x0, beta_y0: Center position of the Gaussian.
    - sigma_x, sigma_y: Standard deviations along x and y axes.
    - amplitude: Peak amplitude of the Gaussian.
    - grid_size: Number of points along each dimension.
    - grid_range: Range of beta_x and beta_y (-range to +range).
    
    Returns:
    - beta_X, beta_Y: Meshgrid coordinates.
    - source: 2D array representing the Gaussian source.
    """
    beta_x = np.linspace(-grid_range, grid_range, grid_size)
    beta_y = np.linspace(-grid_range, grid_range, grid_size)
    beta_X, beta_Y = np.meshgrid(beta_x, beta_y)
    
    exponent = -(((beta_X - beta_x0)**2) / (2 * sigma_x**2) + ((beta_Y - beta_y0)**2) / (2 * sigma_y**2))
    source = amplitude * np.exp(exponent)
    
    return beta_X, beta_Y, source

def lens_transform_source(beta_X, beta_Y, source, theta_E=1.0, gamma=0.3):
    """
    Transform the source plane to image plane using the lens equation.
    
    Parameters:
    - beta_X, beta_Y: Meshgrid coordinates of the source.
    - source: 2D array representing the Gaussian source.
    - theta_E: Einstein radius of the lens.
    - gamma: External shear magnitude.
    
    Returns:
    - image_X, image_Y: Meshgrid coordinates of the image.
    - lensed_image: 2D array representing the lensed image.
    """
    # Flatten the source grid
    beta_x_flat = beta_X.flatten()
    beta_y_flat = beta_Y.flatten()
    source_flat = source.flatten()
    
    # Initialize lists to store image positions and magnifications
    image_positions_x = []
    image_positions_y = []
    image_magnifications = []
    
    # Define a function to solve the lens equation for theta given beta
    def lens_eq(theta, beta_x, beta_y, theta_E, gamma):
        theta_x, theta_y = theta
        theta_sq = theta_x**2 + theta_y**2
        if theta_sq == 0:
            theta_sq = 1e-8  # Avoid division by zero
        alpha_x = (theta_E**2 * theta_x) / theta_sq + gamma * theta_x
        alpha_y = (theta_E**2 * theta_y) / theta_sq - gamma * theta_y
        return [beta_x - (theta_x - alpha_x), beta_y - (theta_y - alpha_y)]
    
    # Iterate over all source points
    for i in range(len(beta_x_flat)):
        beta_x_i = beta_x_flat[i]
        beta_y_i = beta_y_flat[i]
        source_flux = source_flat[i]
        
        # Initial guesses for image positions (simple approximation)
        initial_guesses = [
            [beta_x_i + theta_E, beta_y_i],
            [beta_x_i - theta_E, beta_y_i],
            [beta_x_i, beta_y_i + theta_E],
            [beta_x_i, beta_y_i - theta_E]
        ]
        
        # Solve lens equation to find image positions
        solutions = []
        for guess in initial_guesses:
            sol = root(lens_eq, guess, args=(beta_x_i, beta_y_i, theta_E, gamma))
            if sol.success:
                theta_x_sol, theta_y_sol = sol.x
                # Check if the solution is already found (within a tolerance)
                duplicate = False
                for existing_sol in solutions:
                    if np.hypot(theta_x_sol - existing_sol[0], theta_y_sol - existing_sol[1]) < 1e-4:
                        duplicate = True
                        break
                if not duplicate:
                    solutions.append(sol.x)
        
        # For each solution, compute magnification and accumulate flux
        for sol in solutions:
            theta_x, theta_y = sol
            theta_sq = theta_x**2 + theta_y**2
            if theta_sq == 0:
                theta_sq = 1e-8  # Avoid division by zero
            
            # Compute partial derivatives for Jacobian
            dbeta_x_dtheta_x = 1 - (theta_E**2 * (theta_sq - 2 * theta_x**2)) / (theta_sq**2) - gamma
            dbeta_x_dtheta_y = (2 * theta_E**2 * theta_x * theta_y) / (theta_sq**2)
            dbeta_y_dtheta_x = (2 * theta_E**2 * theta_x * theta_y) / (theta_sq**2)
            dbeta_y_dtheta_y = 1 - (theta_E**2 * (theta_sq - 2 * theta_y**2)) / (theta_sq**2) + gamma
            
            # Compute Jacobian determinant
            det_J = (dbeta_x_dtheta_x * dbeta_y_dtheta_y) - (dbeta_x_dtheta_y * dbeta_y_dtheta_x)
            if det_J == 0:
                det_J = 1e-8  # Avoid division by zero
            magnification = 1.0 / np.abs(det_J)
            
            # Accumulate image properties
            image_positions_x.append(theta_x)
            image_positions_y.append(theta_y)
            image_magnifications.append(magnification * source_flux)
    
    # Convert to arrays
    image_positions_x = np.array(image_positions_x)
    image_positions_y = np.array(image_positions_y)
    image_magnifications = np.array(image_magnifications)
    
    # Define Image Plane Grid for Binning
    image_bins = [500, 500]
    image_range = [[-3.0, 3.0], [-3.0, 3.0]]
    
    # Bin magnifications onto image plane grid
    lensed_image, x_edges, y_edges, _ = binned_statistic_2d(
        image_positions_x, image_positions_y, image_magnifications, statistic='sum', 
        bins=image_bins, range=image_range, expand_binnumbers=True)
    
    # Handle NaN values
    lensed_image = np.nan_to_num(lensed_image)
    
    # Compute bin centers
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    image_X, image_Y = np.meshgrid(x_centers, y_centers)
    
    return image_X, image_Y, lensed_image

def plot_source_and_lensed_image(beta_X, beta_Y, source, 
                                 image_X, image_Y, lensed_image,
                                 magnification_map, 
                                 x_edges, y_edges, 
                                 title_source='2D Gaussian Source',
                                 title_lensed='Lensed Image',
                                 cmap='plasma', log_scale=True):
    """
    Plot the original Gaussian source and its lensed image.
    
    Parameters:
    - beta_X, beta_Y: Meshgrid coordinates of the source.
    - source: 2D array representing the Gaussian source.
    - image_X, image_Y: Meshgrid coordinates of the image.
    - lensed_image: 2D array representing the lensed image.
    - magnification_map: 2D array of magnification values.
    - x_edges, y_edges: Bin edges for the source plane grid.
    - title_source: Title for the source plot.
    - title_lensed: Title for the lensed image plot.
    - cmap: Colormap to use.
    - log_scale: Whether to apply logarithmic scaling to magnification.
    """
    fig, axs = plt.subplots(1, 3, figsize=(13, 4))
    
    # Plot Source
    im0 = axs[0].imshow(source, extent=(beta_X.min(), beta_X.max(), beta_Y.min(), beta_Y.max()), 
                        origin='lower', cmap='plasma')
    axs[0].set_title(title_source)
    axs[0].set_xlabel(r'$\beta_x$')
    axs[0].set_ylabel(r'$\beta_y$')
    axs[0].axis('equal')
    cbar0 = fig.colorbar(im0, ax=axs[0])
    cbar0.set_label('Brightness')
    
    # Plot Magnification Map
    # Compute bin centers
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    X_map, Y_map = np.meshgrid(x_centers, y_centers)
    
    if log_scale:
        im1 = axs[1].pcolormesh(X_map, Y_map, np.log10(magnification_map + 1e-3), 
                                shading='auto', cmap=cmap)
        cbar1 = fig.colorbar(im1, ax=axs[1])
        cbar1.set_label(r'$\log_{10}(\mu)$')
    else:
        im1 = axs[1].pcolormesh(X_map, Y_map, magnification_map, shading='auto', cmap=cmap)
        cbar1 = fig.colorbar(im1, ax=axs[1])
        cbar1.set_label(r'$\mu$')
    
    axs[1].set_title('Magnification Map')
    axs[1].set_xlabel(r'$\beta_x$')
    axs[1].set_ylabel(r'$\beta_y$')
    axs[1].axis('equal')
    
    # Plot Lensed Image
    norm = simple_norm(lensed_image, 'log', percent=99)
    im2 = axs[2].imshow(lensed_image, extent=(image_X.min(), image_X.max(), image_Y.min(), image_Y.max()), 
                        origin='lower', cmap='plasma',
                        norm=norm)
    axs[2].set_title(title_lensed)
    axs[2].set_xlabel(r'$\theta_x$')
    axs[2].set_ylabel(r'$\theta_y$')
    axs[2].axis('equal')
    cbar2 = fig.colorbar(im2, ax=axs[2])
    cbar2.set_label('Lensed Brightness')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Parameters
    theta_E = 1.0      # Einstein radius
    gamma = 0.3        # External shear (0 <= gamma < 1)
    image_grid_size = 500   # Resolution of image plane grid
    image_grid_range = 3.0    # Range of image plane coordinates
    source_grid_size = 500    # Resolution of source plane grid
    source_grid_range = 3.0    # Range of source plane coordinates
    
    # Compute Magnification Map
    x_edges, y_edges, magnification_map, theta_X_flat, theta_Y_flat, magnification = compute_magnification_map(
        theta_E=theta_E, 
        gamma=gamma, 
        image_grid_size=image_grid_size, 
        image_grid_range=image_grid_range, 
        source_grid_size=source_grid_size, 
        source_grid_range=source_grid_range
    )
    
    # Plot Magnification Map
    # plot_magnification_map(
    #     x_edges, 
    #     y_edges, 
    #     magnification_map, 
    #     title=f'Caustic Magnification Map (γ = {gamma})',
    #     cmap='inferno',
    #     log_scale=True
    # )
    
    # Generate 2D Gaussian Source
    beta_X, beta_Y, source = generate_2d_gaussian_source(
        beta_x0=0.0,    # Centered at origin
        beta_y0=0.0,
        sigma_x=0.1,     # Width of the Gaussian
        sigma_y=0.1,
        amplitude=1.0,
        grid_size=500,
        grid_range=2.0
    )
    
    # Transform Source to Image Plane
    image_X, image_Y, lensed_image = lens_transform_source(
        beta_X, beta_Y, source, 
        theta_E=theta_E, 
        gamma=gamma
    )
    
    # Plot Source and Lensed Image
    plot_source_and_lensed_image(
        beta_X, beta_Y, source, 
        image_X, image_Y, lensed_image, 
        magnification_map, 
        x_edges, y_edges, 
        title_source='2D Gaussian Source',
        title_lensed='Lensed Image',
        cmap='plasma',
        log_scale=True
    )