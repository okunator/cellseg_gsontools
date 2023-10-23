from typing import Tuple

import geopandas as gpd
from shapely.geometry import Polygon, box

__all__ = ["bounding_box", "get_grid", "grid_overlay"]


def grid_overlay(
    gdf: gpd.GeoDataFrame,
    patch_size: Tuple[int, int],
    stride: Tuple[int, int],
    pad: int = None,
    predicate: str = "intersects",
) -> gpd.GeoDataFrame:
    """Overlay a grid to the given areas of a GeoDataFrame.

    NOTE: returns None if the gdf is empty.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame to fit the grid to. Uses the bounding box of the GeoDataFrame.
        to fit the grid.
    patch_size : Tuple[int, int]
        Patch size of the grid.
    stride : Tuple[int, int]
        Stride of the sliding window in the grid.
    pad : int, optional
        Pad the bounding box with the given number of pixels, by default None.
    predicate : str, optional
        Predicate to use for the spatial join, by default "intersects".
        Allowed values are "intersects" and "within".

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with the grid fitted to the given GeoDataFrame.
    """
    if gdf.empty or gdf is None:
        return

    allowed = ["intersects", "within"]
    if predicate not in allowed:
        raise ValueError(f"predicate must be one of {allowed}. Got {predicate}")
    grid = get_grid(gdf, patch_size, stride, pad=pad)
    grid.set_crs(epsg=4328, inplace=True, allow_override=True)
    grid = grid.sjoin(gdf, predicate=predicate)

    return grid


def bounding_box(gdf: gpd.GeoDataFrame, pad: int = 0) -> gpd.GeoDataFrame:
    """Get the bounding box of a GeoDataFrame.

    NOTE: returns None if the gdf is empty.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        The input GeoDataFrame.
    pad : int, default=0
        The padding to add to the bounding box.

    Returns
    -------
    gpd.GeoDataFrame:
        The bounding box as a GeoDataFrame.
    """
    if gdf.empty or gdf is None:
        return

    xmin, ymin, xmax, ymax = gdf.total_bounds
    bbox = box(xmin - pad, ymin - pad, xmax + pad, ymax + pad)
    return gpd.GeoDataFrame({"geometry": bbox}, index=[0])


def _get_margins(
    first_endpoint: int, size: int, stride: int, pad: int = None
) -> Tuple[int, int]:
    """Get the number of slices needed for one direction and the overlap.

    Parameters
    ----------
    first_endpoint : int
        The first coordinate of the patch.
    size : int
        The size of the input.
    stride : int
        The stride of the sliding window.
    pad : int, default=None
        The padding to add to the patch

    Returns
    -------
    Tuple[int, int]:
        The number of patches needed for one direction and the overlap.
    """
    pad = int(pad) if pad is not None else 20  # at least some padding needed
    size += pad

    n = 1
    mod = 0
    end = first_endpoint
    while True:
        n += 1
        end += stride

        if end > size:
            mod = end - size
            break
        elif end == size:
            break

    return n, mod + pad


def get_grid(
    gdf: gpd.GeoDataFrame,
    patch_size: Tuple[int, int],
    stride: Tuple[int, int],
    pad: int = None,
) -> gpd.GeoDataFrame:
    """Get a grid of patches from a GeoDataFrame.

    NOTE: returns None if the gdf is empty.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        The input GeoDataFrame.
    patch_size : Tuple[int, int]
        The size of the patch.
    stride : Tuple[int, int]
        The stride of the sliding window.
    pad : int, default=None
        The padding to add to the patch

    Returns
    -------
    gpd.GeoDataFrame:
        The grid of patches.
    """
    if gdf.empty or gdf is None:
        return

    # add some default padding
    if pad is None:
        pad = 20

    bbox: gpd.GeoDataFrame = bounding_box(gdf, pad=pad)
    minx, miny, maxx, maxy = bbox.geometry.bounds.values[0]
    width = maxx - minx
    height = maxy - miny

    total_size = (height, width)

    y_end, x_end = patch_size
    nrows, _ = _get_margins(y_end, total_size[0], stride[0], pad=pad)
    ncols, _ = _get_margins(x_end, total_size[1], stride[1], pad=pad)

    grid_rects = []
    for row in range(nrows):
        for col in range(ncols):
            y_start = row * stride[0] + miny
            y_end = y_start + patch_size[0]
            x_start = col * stride[1] + minx
            x_end = x_start + patch_size[1]
            rect = Polygon(
                [(x_start, y_start), (x_end, y_start), (x_end, y_end), (x_start, y_end)]
            )
            grid_rects.append(rect)

    return gpd.GeoDataFrame({"geometry": grid_rects})
