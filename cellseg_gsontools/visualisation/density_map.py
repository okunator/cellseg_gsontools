import numpy as np
from numba import jit


@jit
def density_map(
    dx: int, dy: int, points: list[tuple[float, float]], sx: int = 25, sy: int = 25
) -> np.array:
    """Convert cell coordinates to normalized density map.

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
    X, Y = np.meshgrid(x, y)
    grid = np.zeros_like(X)

    a = 1.0 / (2.0 * np.pi * sx * sy)

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            for point in points:
                v = a * np.exp(
                    -(
                        (point[0] - X[i, j]) ** 2.0 / (2.0 * sx**2.0)
                        + (point[1] - Y[i, j]) ** 2.0 / (2.0 * sy**2.0)
                    )
                )
                grid[i, j] += v

    return grid / np.sum(grid)
