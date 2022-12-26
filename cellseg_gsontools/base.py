from numba import jit
import numpy as np

@jit
def gauss2d(dx, dy, points, sx=150, sy=150):

    """
    Convert coordinates to normalized density map

    Args:
    ---------
        dx,dy: slide shape
        points: coordinate array
        sx, sy: covariance (smoothing)

    Returns:
    ---------
        np.ndarray: Shape (H, W)
    """

    x = np.linspace(0, dx, dx)
    y = np.linspace(0, dy, dy)
    x, y = np.meshgrid(x, y)
    grid = np.empty_like(x)

    a = 1.0/(2.0 * np.pi * sx * sy)

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            for point in points:
                v = a*np.exp(
                    -((point[0]-x[i,j]) ** 2.0 / (2.0 * sx**2.0) + (point[1]-y[i,j]) ** 2.0 / (2.0 * sy**2.0))
                )
                grid[i, j] += v

    return grid / np.sum(grid)
