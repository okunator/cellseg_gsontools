import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import shapely
from descartes import PolygonPatch
from helper import create_grid
from geometry import alpha_shape
from matplotlib import cm


def arrow_plot(
    z: np.array,
    points: list[shapely.geometry.Point],
    dx: int,
    dy: int,
    ax: plt.axes,
    region: bool = False,
    n: int = 100,
) -> plt.axes:
    """Plot distribution gradient.

    Args:
    ---------
        z: 2D cell distribution
        points: coordinate array
        dx,dy: slide shape
    Returns:
    ---------
        axis
    """
    X, Y = create_grid(dx, dy)
    ddy, ddx = np.gradient(z)

    connective_hull, edge_points = alpha_shape(points, alpha=0.003)
    scale = 1
    center_area = shapely.affinity.scale(
        connective_hull, xfact=scale, yfact=scale, zfact=scale, origin="center"
    )

    x, y = center_area.exterior.xy
    ax.plot(x, y, zorder=2.5, color="red")

    X = X[::n, ::n]
    Y = Y[::n, ::n]
    ddx = ddx[::n, ::n]
    ddy = ddy[::n, ::n]

    if region:
        for i in range(0, len(X)):
            for j in range(0, len(Y[1])):
                point = shapely.geometry.Point(X[i, j], Y[i, j])

                if not center_area.contains(point):
                    ddx[i, j] = 0
                    ddy[i, j] = 0

    return ax.quiver(X, Y, -ddx, ddy, units="xy", width=10, color="black")


def multidim_plot(
    z: np.array, points: list[shapely.geometry.Point], dx: int, dy: int
) -> None:
    """Plot cell distribution.

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
    ha = hf.add_subplot(111, projection="3d")

    X, Y = create_grid(dx, dy)

    ha.plot_surface(X, Y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    # Cell area
    polygon = shapely.geometry.Polygon(points)
    convex_hull = polygon.convex_hull
    x, y = convex_hull.exterior.xy
    ha.plot(x, y, zorder=2.5, color="red")
    plt.gca().invert_xaxis()

    plt.show()


def plot_celltype(cells: gpd.GeoDataFrame, slide: np.array, axis: plt.axes) -> plt.axes:
    """Plot cells on slide.

    Args:
    ---------
        cells: Geodataframe of cells
        slide: image
        axis: plt.subplots object

    Returns:
    ---------
        axis
    """
    axis.imshow(slide)
    cells.plot(ax=axis, color="yellow", markersize=5)

    points = []
    for i in range(0, len(cells["geometry"])):
        point = cells["geometry"].iloc[i].coords[:][0]
        points.append(point)

    connective_hull, edge_points = alpha_shape(points, alpha=0.003)
    center_area = shapely.affinity.scale(connective_hull, origin="center")

    x, y = center_area.exterior.xy
    axis.plot(x, y, zorder=2.5, color="red")

    return axis


def plot_polygon(polygon, color: str, axis: plt.axes, margin: int = 0) -> plt.axes:
    """Plot polygon.

    Args:
    ---------
        polygon
        color
        axis
        margin:

    Returns:
    ---------
        axis
    """
    x_min, y_min, x_max, y_max = polygon.bounds

    axis.set_xlim([x_min - margin, x_max + margin])
    axis.set_ylim([y_min - margin, y_max + margin])

    patch = PolygonPatch(polygon, color=color, alpha=0.25)

    axis.add_patch(patch)

    return axis
