from typing import Tuple, Union

import numpy as np
from shapely.geometry import Point, Polygon

__all__ = ["axis_len", "axis_angle", "_dist", "_azimuth"]


def _dist(p1: Point, p2: Point) -> float:
    """Compute distance between two points."""
    return p1.distance(p2)


def _azimuth(p1: Point, p2: Point) -> float:
    """Azimuth between 2 points (interval 0 - 180)."""
    angle = np.arctan2(p2[0] - p1[0], p2[1] - p1[1])
    return np.degrees(angle) if angle > 0 else np.degrees(angle) + 180


def axis_len(
    mrr: Polygon, which: str = "both", **kwargs
) -> Union[float, Tuple[float, float]]:
    """Compute major and minor axis from minimum rotated rectangle.

    Parameters
    ----------
        mrr : Polygon
            Input minimum rotated rectangle of a shapely polygon object.
        which : str, default="both"
            One of ("both", "major", "minor").

    Returns
    -------
        Union[float, Tuple[float, float]]
            The major and/or minor axis lengths.
    """
    allowed = ("both", "major", "minor")
    if which not in allowed:
        raise ValueError(f"Illegal arg `which`. Got: {which}. Allowed: {allowed}")

    axis1 = _dist(Point(mrr[0]), Point(mrr[3]))
    axis2 = _dist(Point(mrr[0]), Point(mrr[1]))

    minoraxis_len = min([axis1, axis2])
    majoraxis_len = max([axis1, axis2])

    if which == "major":
        return majoraxis_len
    elif which == "minor":
        return minoraxis_len
    else:
        return majoraxis_len, minoraxis_len


def axis_angle(
    mrr: Polygon, which: str = "both", **kwargs
) -> Union[float, Tuple[float, float]]:
    """Compute major and minor axis angles from minimum rotated rectangle.

    Parameters
    ----------
        polygon : Polygon
            Input minimum rotated rectangle polygon object.
        which : str, default="both"
            One of ("both", "major", "minor").

    Returns
    -------
        Union[float, Tuple[float, float]]
            The major and/or minor axis angles in degrees.
    """
    allowed = ("both", "major", "minor")
    if which not in allowed:
        raise ValueError(f"Illegal arg `which`. Got: {which}. Allowed: {allowed}")

    axis1 = _dist(Point(mrr[0]), Point(mrr[3]))
    axis2 = _dist(Point(mrr[0]), Point(mrr[1]))

    if axis1 <= axis2:
        minoraxis_ang = _azimuth(mrr[0], mrr[3])
        majoraxis_ang = _azimuth(mrr[0], mrr[1])
    else:
        minoraxis_ang = _azimuth(mrr[0], mrr[1])
        majoraxis_ang = _azimuth(mrr[0], mrr[3])

    if which == "major":
        return majoraxis_ang
    elif which == "minor":
        return minoraxis_ang
    else:
        return majoraxis_ang, minoraxis_ang
