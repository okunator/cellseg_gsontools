import numpy as np
import shapely
from distribution import gauss2d
from geometry import alpha_shape
from helper import create_grid


def get_gradient(pdf: np.ndarray) -> np.ndarray:
    """Convert distribution into gradient distribution.

    Args:
    ---------
        pdf: 2D cell distribution

    Returns:
    ---------
        np.ndarray: Shape (H, W) with gradient lengths
    """
    dx, dy = np.gradient(np.array(pdf.T))

    # Length for each vector
    gradient_flat = np.sqrt(dx**2 + dy**2)

    return gradient_flat


def total_gradient(pdf: np.ndarray) -> float:
    """Sum gradient of distribution.

    Args:
    ---------
        pdf: 2D cell distribution

    Returns:
    ---------
        np.ndarray: Shape (H, W) with gradient lengths
    """
    return np.sum(get_gradient(pdf))


def differentiation(
    z: np.ndarray, points: list[tuple[float, float]], dx: int, dy: int, n: int = 100
) -> float:
    """Calculate differentioation metric for cells.

    Args:
    ---------
        pdf: 2D cell distribution
        points: array of cell coordinates
        dx, dy: Distribution size
        n: sampling frequency

    Returns:
    ---------
        float: metric
    """
    X, Y = create_grid(dx, dy)
    ddx, ddy = np.gradient(z)

    connective_hull, edge_points = alpha_shape(points, alpha=0.001)
    center_area = shapely.affinity.scale(
        connective_hull, xfact=0.8, yfact=0.8, zfact=0.8, origin="center"
    )

    X = X[::n, ::n]
    Y = Y[::n, ::n]
    ddx = ddx[::n, ::n]
    ddy = ddy[::n, ::n]

    for i in range(0, len(X)):
        for j in range(0, len(Y[0])):
            point = shapely.geometry.Point(X[i, j], Y[i, j])

            if not center_area.contains(point):
                ddx[i, j] = 0
                ddy[i, j] = 0

    return n**2 * np.sum(np.sqrt(ddx**2 + ddy**2))


def gradient_bootstrap(
    points: list[tuple[float, float]], H: int, W: int, size: int, length: int
) -> list[tuple[float, float]]:
    """Bootstrap cells to approximate gradient of the area.

    Args:
    ---------
        points: array of cell coordinates
        H, W: distribution size
        size: amount to sample

    Returns:
    ---------
        np.ndarray: Shape (H, W) with gradient lengths
    """
    sample = []
    for i in range(length):
        spoints = np.array(points)[np.random.randint(np.shape(points)[0], size=size), :]
        z = gauss2d(H, W, spoints)
        val = differentiation(z, spoints, H, W)
        sample.append(val)

    return sample
