from typing import List, Tuple

import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon
from shapely.ops import unary_union

__all__ = ["validate_and_simplify", "get_overlapping_objects", "merge_overlaps"]


def validate_and_simplify(
    poly: Polygon, buffer: float = 1.0, simplify: bool = False, level: float = 0.3
) -> Polygon:
    """Validate and simplify a shapely polygon.

    Parameters:
        poly (Polygon):
            The polygon to validate and simplify.
        buffer (float):
            The size of the buffer.
        simplify (bool):
            Whether to simplify the polygon or not.
        level (float):
            The simplification level.

    Returns:
        Polygon: The validated and simplified polygon.
    """
    # workaround to handle invalid polygons
    if not poly.is_valid:
        poly = poly.buffer(0)

    # do some simplifying
    if simplify:
        poly = (
            poly.buffer(10.0, join_style=1)
            .buffer(-10.0 + buffer, join_style=1)
            .simplify(level)
        )

    return poly


def get_overlapping_objects(gdf: gpd.GeoDataFrame) -> List[Tuple[int, int]]:
    """Get index pairs of overlapping objects in a gdf.

    Parameters:
        gdf (gpd.GeoDataFrame):
            The gdf to check for overlapping objects.

    Returns:
        (List[Tuple[int, int]]):
            A list of index pairs of overlapping objects.

    Examples:
        >>> import geopandas as gpd
        >>> from cellseg_gsontools.utils import read_gdf
        >>> gdf = read_gdf("/path/to/cells.json")
        >>> get_overlapping_objects(gdf)
        [(9, 34), (9, 7), (38, 39), (9, 34, 7), (32, 29, 31)]
    """
    overlap_inds = []
    for i in range(len(gdf)):
        overlaps = gdf.overlaps(gdf.geometry.loc[i])
        inds = set(i for i in overlaps[overlaps].index.tolist())
        inds.add(i)
        overlap_inds.append(inds)

    overlap_inds = list(set(frozenset(item) for item in overlap_inds if len(item) > 1))
    overlap_inds = [tuple(item) for item in overlap_inds]

    return overlap_inds


def merge_overlaps(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Merge overlapping objects in a gdf.

    Parameters:
        gdf (gpd.GeoDataFrame):
            The gdf containing overlapping objects.

    Returns:
        (gpd.GeoDataFrame):
            A gdf with merged overlapping objects.

    Examples:
        >>> import geopandas as gpd
        >>> from cellseg_gsontools.utils import read_gdf
        >>> gdf = read_gdf("/path/to/cells.json")
        >>> merge_overlaps(gdf)
    """
    # get overlapping objects
    gdf = gdf.copy()
    overlaps = get_overlapping_objects(gdf)

    # merge the overlapping objects
    new_polys = []
    new_classes = []
    for o in overlaps:
        # merge the overlapping objects and majority vote for the class
        # skip if the objects have already been merged
        if all([i in gdf.index for i in o]):
            data = gdf.loc[list(o)]
            new_polys.append(unary_union(data.geometry.tolist()))
            new_classes.append(data.class_name.value_counts().idxmax())

        # drop the objects from the orig gdf
        gdf = gdf.drop([i for i in o if i in gdf.index])

    overlap_gdf = gpd.GeoDataFrame({"geometry": new_polys, "class_name": new_classes})
    return pd.concat([gdf, overlap_gdf])
