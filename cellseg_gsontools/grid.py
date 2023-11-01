from functools import partial
from typing import Any, Callable, Tuple, Union

import geopandas as gpd
from shapely.geometry import Polygon, box

from cellseg_gsontools.apply import gdf_apply

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

    return grid.drop_duplicates("geometry")


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


def get_rect_metric(
    rect, objs: gpd.GeoDataFrame, metric_func: Callable, predicate: str
) -> Any:
    """Get the metric of the given rectangle.

    Parameters
    ----------
    rect : Polygon
        The rectangle to get the metric of.
    objs : gpd.GeoDataFrame
        The objects to use for the metric.
    metric_func : Callable
        The metric function to use.
    predicate : str
        The predicate to use for the spatial join. Allowed values are "intersects"
        and "within".

    Returns
    -------
    Any:
        The metric of the rectangle.
    """
    if predicate == "intersects":
        sub_objs: gpd.GeoDataFrame = objs[objs.geometry.intersects(rect)]
    elif predicate == "within":
        sub_objs: gpd.GeoDataFrame = objs[objs.geometry.within(rect)]
    else:
        raise ValueError(
            f"Illegal predicate: {predicate}. Allowed: 'intersects', 'within'"
        )

    return metric_func(sub_objs)


def grid_classify(
    grid: gpd.GeoDataFrame,
    objs: gpd.GeoDataFrame,
    metric_func: Callable,
    predicate: str,
    new_col_names: Union[Tuple[str, ...], str],
    parallel: bool = True,
    num_processes: int = -1,
    pbar: bool = False,
    **kwargs,
) -> gpd.GeoDataFrame:
    """Classify the grid based on objs inside.

    Parameters
    ----------
    grid : gpd.GeoDataFrame
        The grid of rectangles to classify.
    objs : gpd.GeoDataFrame
        The objects to use for classification.
    metric_func : Callable
        The metric/heuristic function to use for classification.
    predicate : str
        The predicate to use for the spatial join. Allowed values are "intersects"
        and "within".
    new_col_names : Union[Tuple[str, ...], str]
        The name of the new column(s) in the grid gdf.
    parallel : bool, default=True
        Whether to use parallel processing.
    num_processes : int, default=-1
        The number of processes to use. If -1, uses all available cores.
        Ignored if parallel=False.
    pbar : bool, default=False
        Whether to show a progress bar. Ignored if parallel=False.

    Example
    -------
    Get the number of immune cells in each grid cell at the tumor stroma interface:

    >>> import geopandas as gpd
    >>> from cellseg_gsontools.grid import grid_classify
    >>> from cellseg_gsontools.apply import gdf_apply
    >>> from shapely.geometry import Polygon
    >>> from functools import partial
    >>> from cellseg_gsontools.context import InterfaceContext
    >>> from cellseg_gsontools.grid import grid_overlay

    >>> # Define a heuristic function to get the number of immune cells
    >>> def get_immune_cell_cnt(gdf: gpd.GeoDataFrame, **kwargs) -> int:
    >>>     try:
    >>>         cnt = gdf.class_name.value_counts()["inflammatory"]
    >>>     except KeyError:
    >>>         cnt = 0

    >>>     return int(cnt)

    >>> # Read in the tissue areas and cells
    >>> area_gdf = gpd.read_file("path/to/area.geojson")
    >>> cell_gdf = gpd.read_file("path/to/cell.geojson")

    >>> # Fit a tumor-stroma interface
    >>> tumor_stroma_iface = InterfaceContext(
    >>>     area_gdf=area_gdf,
    >>>     cell_gdf=cell_gdf,
    >>>     top_labels="area_cin",
    >>>     bottom_labels="areastroma",
    >>>     buffer_dist=250,
    >>>     graph_type="distband",
    >>>     dist_thresh=75,
    >>>     patch_size=(128, 128),
    >>>     stride=(128, 128),
    >>>     min_area_size=50000
    >>> )
    >>> tumor_stroma_iface.fit(parallel=False)

    >>> # Get the grid and the cells at the interface
    >>> iface_grid = grid_overlay(
    ...     tumor_stroma_iface.context2gdf("interface_area"),
    ...     patch_size=(128, 128),
    ...     stride=(128, 128)
    ... )
    >>> cells = tumor_stroma_iface.context2gdf("interface_cells")

    >>> # Classify the grid
    >>> iface_grid = grid_classify(
    >>>     grid=iface_grid,
    >>>     objs=cells,
    >>>     metric_func=get_immune_cnt,
    >>>     predicate="intersects",
    >>>     new_col_name="immune_cnt",
    >>>     parallel=True,
    >>>     pbar=True,
    >>>     num_processes=-1
    >>> )
    >>> iface_grid
                                                 geometry  immune_cnt
    28  POLYGON ((20032.00000 54098.50000, 20160.00000...               15
    29  POLYGON ((20160.00000 54098.50000, 20288.00000...               3
    """
    allowed = ["intersects", "within"]
    if predicate not in allowed:
        raise ValueError(f"predicate must be one of {allowed}. Got {predicate}")

    if isinstance(new_col_names, str):
        new_col_names = [new_col_names]

    func = partial(
        get_rect_metric, objs=objs, predicate=predicate, metric_func=metric_func
    )
    grid.loc[:, list(new_col_names)] = gdf_apply(
        grid,
        func=func,
        parallel=parallel,
        pbar=pbar,
        num_processes=num_processes,
        columns=["geometry"],
    )

    return grid
