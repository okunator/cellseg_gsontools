from collections import defaultdict
from functools import partial
from typing import Any, Tuple

import geopandas as gpd
import numpy as np
import shapely
from libpysal.weights import W
from scipy.spatial import Voronoi
from shapely import vectorized

from cellseg_gsontools.apply import gdf_apply
from cellseg_gsontools.geometry import major_axis_len

__all__ = [
    "perpendicular_line",
    "equal_interval_points",
    "voronoi_medial",
    "line_branches",
    "medial_lines",
    "perpendicular_lines",
]


def perpendicular_line(
    line: shapely.LineString, seg_length: float
) -> shapely.LineString:
    """Create a perpendicular line from a line segment.

    Note:
        Returns an empty line if perpendicular line is not possible from the input.

    Parameters:
        line (shapely.LineString):
            Line segment to create a perpendicular line from.
        seg_length (float):
            Length of the perpendicular line.

    Returns:
        shapely.LineString:
            Perpendicular line to the input line of length `seg_length`.
    """
    left = line.parallel_offset(seg_length / 2, "left").centroid
    right = line.parallel_offset(seg_length / 2, "right").centroid

    if left.is_empty or right.is_empty:
        return shapely.LineString()

    return shapely.LineString([left, right])


def equal_interval_points(obj: Any, n: int = None, delta: float = None):
    """Resample the points of a shapely object at equal intervals.

    Parameters:
        obj (Any):
            Any shapely object that has length property.
        n (int):
            Number of points, defaults to None
        delta (float):
            Distance between points, defaults to None

    Returns:
        points (numpy.ndarray):
            Array of points at equal intervals along the input object.
    """
    length = obj.length

    if n is None:
        if delta is None:
            delta = obj.length / 1000
        n = round(length / delta)

    distances = np.linspace(0, length, n)
    points = [obj.interpolate(distance) for distance in distances]
    points = np.array([(p.x, p.y) for p in points])

    return points


# Adapted from
# https://github.com/mikedh/trimesh/blob/main/trimesh/path/polygons.py
def voronoi_medial(
    polygon: shapely.Polygon, n: int = None, delta: float = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the medial lines of a polygon using voronoi diagram.

    Parameters:
        polygon (shapely.geometry.Polygon):
            Polygon to compute the medial lines of.
        n (int):
            Number of resampled points in the input polygon, defaults to None
        delta (float):
            Distance between resampled polygon points, defaults to None. Ignored
            if n is not None.

    Returns:
        vertices (numpy.ndarray):
            Array of vertices of the voronoi diagram.
        edges (numpy.ndarray):
            Array of edges of the voronoi diagram.
    """
    points = equal_interval_points(polygon.exterior, n=n, delta=delta)

    # # create the voronoi diagram on 2D points
    voronoi = Voronoi(points)

    # which voronoi vertices are contained inside the polygon
    contains = vectorized.contains(polygon, *voronoi.vertices.T)

    # ridge vertices of -1 are outside, make sure they are False
    contains = np.append(contains, False)

    # make sure ridge vertices is numpy array
    ridge = np.asanyarray(voronoi.ridge_vertices, dtype=np.int64)

    # only take ridges where every vertex is contained
    edges = ridge[contains[ridge].all(axis=1)]

    # now we need to remove uncontained vertices
    contained = np.unique(edges)
    mask = np.zeros(len(voronoi.vertices), dtype=np.int64)
    mask[contained] = np.arange(len(contained))

    # mask voronoi vertices
    vertices = voronoi.vertices[contained]

    # re-index edges
    return vertices, mask[edges]


def line_branches(edges: np.ndarray) -> np.ndarray:
    """Get the branch points of a line graph.

    Note:
        Helps to get rid of the random branches of the voronoi medial lines.

    Parameters:
        edges (numpy.ndarray):
            Array of edges of the line graph.

    Returns:
        numpy.ndarray:
            Array of edges of the line graph with the branches removed.
    """
    # create a graph from the edges
    neighbors = defaultdict(list)
    for line in edges:
        start_id, end_id = line
        neighbors[start_id].append(end_id)
        neighbors[end_id].append(start_id)

    w = W(dict(sorted(neighbors.items())))

    # get the branch points
    branch_points = [k for k, c in w.cardinalities.items() if c > 2]

    # get the paths from the branch points
    paths = []
    stack = [(bp, None, [bp]) for bp in branch_points]
    while stack:
        cur, prev, path = stack.pop()

        if len(w.neighbors[cur]) == 1 or (prev and cur in branch_points):
            paths.append(path)
            continue

        for neighbor in w.neighbors[cur]:
            if neighbor != prev:
                stack.append((neighbor, cur, path + [neighbor]))

    return paths


def medial_lines(
    polygon: shapely.Polygon,
    n: int = None,
    delta: float = None,
    rm_branches: bool = False,
) -> gpd.GeoDataFrame:
    """Get the medial lines of a polygon using a voronoi diagram.

    Parameters:
        polygon (shapely.Polygon):
            Polygon to compute the medial lines of.
        n (int):
            Number of resampled points in the input polygon, defaults to None
        delta (float):
            Distance between resampled polygon points, defaults to None. Ignored
            if n is not None.
        rm_branches (bool):
            Whether to remove the branches of the medial lines, defaults to False.

    Returns:
        gpd.GeoDataFrame:
            GeoDataFrame of the medial lines.

    Examples:
        >>> import geopandas as gpd
        >>> import shapely
        >>> from cellseg_gsontools.lines import medial_lines
        >>> from cellseg_gsontools.data import cervix_tissue
        >>> tissues = cervix_tissue()
        >>> tumor = tissues[tissues["class_name"] == "area_cin"]
        >>> tumor_poly = tumor.geometry.iloc[0]  # get the first tumor polygon
        >>> polygon = shapely.Polygon(tumor_poly.exterior)
        >>> med_lines = medial_lines(polygon, delta=500, rm_branches=False)
        >>> med_lines.head()
        geometry
            0  LINESTRING (10789.887 49746.299, 10910.622 493...
            1  LINESTRING (10789.887 49746.299, 10926.865 498...
            2  LINESTRING (10924.971 48929.809, 10829.145 492...
            3  LINESTRING (10910.622 49332.471, 10829.145 492...
            4  LINESTRING (10926.865 49843.003, 10794.602 502...
    """
    # get the medial lines
    vertices, edges = voronoi_medial(polygon, n=n, delta=delta)

    # remove lone branches of the medial lines
    if rm_branches:
        # get the line paths and branches of the medial lines
        paths = line_branches(edges)
        edges = np.vstack(
            [
                np.array(list(zip(branch, branch[1:])))
                for branch in paths
                if len(branch) > 2
            ]
        )

    med_lines = gpd.GeoDataFrame(
        [shapely.LineString(vertices[line]) for line in edges], columns=["geometry"]
    )
    # clip the medial lines to the polygon
    med_lines = med_lines.loc[med_lines.within(polygon)]

    return med_lines


def perpendicular_lines(
    lines: gpd.GeoDataFrame, polygon: shapely.Polygon = None
) -> gpd.GeoDataFrame:
    """Get perpendicular lines to the input lines starting from the line midpoints.

    Parameters:
        lines (gpd.GeoDataFrame):
            GeoDataFrame of the input lines.
        polygon (shapely.Polygon):
            Polygon to clip the perpendicular lines to.

    Returns:
        gpd.GeoDataFrame:
            GeoDataFrame of the perpendicular lines.
    """
    # create perpendicular lines to the medial lines
    if polygon is None:
        polygon = lines.unary_union.convex_hull

    seg_len = major_axis_len(polygon)
    func = partial(perpendicular_line, seg_length=seg_len)
    perp_lines = gdf_apply(lines, func, columns=["geometry"])

    # clip the perpendicular lines to the polygon
    perp_lines = gpd.GeoDataFrame(perp_lines, columns=["geometry"]).clip(polygon)

    # explode perpendicular lines & take only the ones that intersect w/ medial lines
    perp_lines = perp_lines.explode(index_parts=False).reset_index(drop=True)

    # drop the perpendicular lines that are too short or too long
    # since these are likely artefacts
    perp_lines["len"] = perp_lines.geometry.length
    low, high = perp_lines.len.quantile([0.05, 0.85])
    perp_lines = perp_lines.query(f"{low}<len<{high}")

    return perp_lines
