from typing import Tuple

import cv2
import numpy as np
from shapely.geometry import Polygon
from skimage.draw import polygon2mask
from skimage.morphology import erosion

__all__ = ["inscribing_circle", "circumscribing_circle"]


def inscribing_circle(polygon: Polygon) -> Tuple[float, float, float]:
    """Get the inscribing circle of a polygon.

    NOTE: sometimes for small polygons the center coords can be shifted
    but that does not affect the radius.

    Parameters
    ----------
        polygon : Polygon
            A shapely polygon object.

    Returns
    -------
        Tuple[float, float, float]:
            The centroid x, y- coords and the radius of the circle.
    """
    # get convex hull
    coords = polygon.exterior.coords

    # get hull bounds
    minx, miny, maxx, maxy = polygon.bounds
    w = maxx - minx
    h = maxy - miny
    shape = (int(h) + 1, int(w) + 1)

    # shift to origin for skimage
    points = np.array(coords)  # xy
    points[:, 0] -= minx
    points[:, 1] -= miny

    # convert poly to mask
    points = np.flip(points, 1)  # flip to yx for skimage
    m = np.pad(polygon2mask(shape, points), 1)  # add padding for dist transform
    m = erosion(m)  # erode

    # distance transform
    dist_map = cv2.distanceTransform(
        m.astype("uint8"), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )
    dist_map = dist_map[1:-1, 1:-1]  # undo padding

    # find radius (max distance), and center (argmax)
    _, r, _, center = cv2.minMaxLoc(dist_map)

    # shift back to original coords
    x = center[0] + minx
    y = center[1] + miny

    return x, y, r


def circumscribing_circle(polygon: Polygon) -> Tuple[float, float, float]:
    """Get the circumscribing circle of a a polygon.

    Parameters
    ----------
        polygon : Polygon
            A shapely polygon object.

    Returns
    -------
        Tuple[float, float, float]:
            The centroid x, y- coords and the radius of the circle.
    """
    coords = polygon.exterior.coords
    center, r = cv2.minEnclosingCircle(np.array(coords).astype("int"))

    x = center[0]
    y = center[1]

    return x, y, r
