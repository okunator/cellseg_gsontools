import math
from typing import Sequence, Union

import numpy as np
from scipy.spatial import Delaunay
from shapely.geometry import MultiLineString, MultiPoint, Polygon
from shapely.ops import polygonize, unary_union

__all__ = ["alpha_shape"]


def alpha_shape(polygon: Union[Polygon, Sequence], alpha: float = 0.1) -> Polygon:
    """Compute the alpha shape (concave hull) of a polygon.

    Parameters
    ----------
        polygon : Polygon or Sequence
            Input shapely polygon object or a sequence of points (ndarray/list).
        alpha : float, default=0.1
            Alpha value to influence the gooeyness of the border. Smaller
            numbers don't fall inward as much as larger numbers. Too large,
            and you lose everything.

    Returns
    -------
        Polygon:
            The alpha shape as a shapely polygon.
    """
    if isinstance(polygon, Polygon):
        points = list(polygon.exterior.coords)
    elif isinstance(polygon, (np.ndarray, list)):
        points = list(points)

    if len(points) < 4:
        # When you have a triangle, there is no sense in computing an alpha shape.
        return MultiPoint(points).convex_hull

    def add_edge(edges, edge_points, coords, i, j):
        if (i, j) in edges or (j, i) in edges:
            # already added
            return
        edges.add((i, j))
        edge_points.append(coords[[i, j]])

    # loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    coords = np.array(points)
    tri = Delaunay(coords)
    edges = set()
    edge_points = []
    for ia, ib, ic in tri.simplices:
        pa = coords[ia]
        pb = coords[ib]
        pc = coords[ic]

        # Lengths of sides of triangle
        a = math.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = math.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = math.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)

        # Semiperimeter of triangle
        s = (a + b + c) / 2.0

        # Area of triangle by Heron's formula
        area = math.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)

        # Here's the radius filter.
        if circum_r < 1.0 / alpha:
            add_edge(edges, edge_points, coords, ia, ib)
            add_edge(edges, edge_points, coords, ib, ic)
            add_edge(edges, edge_points, coords, ic, ia)

    m = MultiLineString(edge_points)
    triangles = list(polygonize(m))
    union = unary_union(triangles)

    return union
