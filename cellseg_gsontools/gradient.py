import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from helper import create_grid
import shapely

def get_gradient(pdf):

    """
    Convert distribution into gradient distribution

    Args:
    ---------
        pdf: 2D cell distribution

    Returns:
    ---------
        np.ndarray: Shape (H, W) with gradient lengths
    """

    grad = np.gradient(np.array(pdf.T))

    #Length for each vector
    gradient_flat = np.sqrt(grad[0]**2 + grad[1]**2)

    return gradient_flat

def development_metric(pdf):

    """
    Measures tissue differentiation based on cell distribution

    Args:
    ---------
        pdf: 2D cell distribution

    Returns:
    ---------
        float: measure of development
    """

    return np.sum(np.abs(get_gradient(pdf)))

def arrow_plot(z, points, dx, dy):

    """
        3D plot of cell distribution

    Args:
    ---------
        z: 2D cell distribution
        points: coordinate array
        dx,dy: slide shape

    Returns:
    ---------
        None
    """

    X, Y = create_grid(dx, dy)

    ddx, ddy = np.gradient(z)

    fig, ax = plt.subplots()

    n=100
    ax.quiver(X[::n,::n],Y[::n,::n],ddx[::n,::n], -ddy[::n,::n],
            units="xy", width = 10)

    #Cell area
    polygon = shapely.geometry.Polygon(points)
    convex_hull = polygon.convex_hull
    x, y = convex_hull.exterior.xy
    ax.plot(x, y, zorder=2.5, color="red")

    plt.gca().invert_yaxis()
    plt.show()

def multidim_plot(z, points, dx, dy):

    """
        Arrow plot of cell distribution gradient

    Args:
    ---------
        z: 2D cell distribution
        points: coordinate array
        dx,dy: slide shape

    Returns:
    ---------
        None
    """

    hf = plt.figure()
    ha = hf.add_subplot(111, projection='3d')

    X, Y = create_grid(dx, dy)

    ha.plot_surface(X,Y,z, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)

    #Cell area
    polygon = shapely.geometry.Polygon(points)
    convex_hull = polygon.convex_hull
    x, y = convex_hull.exterior.xy
    ha.plot(x, y, zorder=2.5, color="red")

    plt.show()
