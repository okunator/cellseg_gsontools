from functools import partial
from typing import Any, Callable, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import Polygon, box, mapping

from cellseg_gsontools.apply import gdf_apply
from cellseg_gsontools.utils import get_holes, lonlat_to_xy, xy_to_lonlat

try:
    import h3

    _has_h3 = True
except ImportError:
    _has_h3 = False

__all__ = [
    "bounding_box",
    "get_grid",
    "grid_overlay",
    "hexgrid_overlay",
    "fit_spatial_grid",
]


def fit_spatial_grid(
    gdf: gpd.GeoDataFrame, grid_type: str = "square", **kwargs
) -> gpd.GeoDataFrame:
    """Quick wrapper to fit either a hex or square grid to a `geopandas.GeoDataFrame`.

    Note:
        - Hexagonal grid requires the `h3` package to be installed.
        - Hexagonal grid only works for a gdf containing one single polygon.

    Parameters:
        gdf (gpd.GeoDataFrame):
            GeoDataFrame to fit grid to.
        grid_type (str):
            Type of grid to fit, by default "square".
        **kwargs (Dict[str, Any]):
            Keyword arguments to pass to grid fitting functions.

    Returns:
        gpd.GeoDataFrame:
            Fitted grid.

    Raises:
        ValueError: If grid_type is not one of "square" or "hex".
        ImportError: If grid_type is "hex" and the `h3` package is not installed.

    Examples:
        Fit a hexagonal grid to a gdf:
        >>> from cellseg_gsontools import read_gdf
        >>> from cellseg_gsontools.grid import fit_spatial_grid
        >>> # Read in the tissue areas
        >>> area_gdf = gpd.read_file("path/to/area.geojson")
        >>> # Fit the grid
        >>> hex_grid = fit_spatial_grid(area_gdf, grid_type="hex", resolution=9)
        >>> hex_grid
        gpd.GeoDataFrame

        Fit a square grid to a gdf:
        >>> from cellseg_gsontools import read_gdf
        >>> from cellseg_gsontools.grid import fit_spatial_grid
        >>> # Read in the tissue areas
        >>> area_gdf = gpd.read_file("path/to/area.geojson")
        >>> # Fit the grid
        >>> sq_grid = fit_spatial_grid(
        ...     area_gdf, grid_type="square", patch_size=(256, 256), stride=(256, 256)
        ... )
        >>> sq_grid
        gpd.GeoDataFrame
    """
    allowed = ["square", "hex"]
    if grid_type not in allowed:
        raise ValueError(f"grid_type must be one of {allowed}, got {grid_type}")

    if grid_type == "square":
        grid = grid_overlay(gdf, **kwargs)
    else:
        if not _has_h3:
            raise ImportError("h3 package not installed. Install with `pip install h3`")
        grid = hexgrid_overlay(gdf, **kwargs)

    return grid


def hexgrid_overlay(
    gdf: gpd.GeoDataFrame, resolution: int = 9, to_lonlat: bool = True
) -> gpd.GeoDataFrame:
    """Fit a `h3` hexagonal grid on top of a `geopandas.GeoDataFrame`.

    Parameters:
        gdf (gpd.GeoDataFrame):
            GeoDataFrame to fit grid to.
        resolution (int):
            H3 resolution, by default 9.
        to_lonlat (bool):
            Whether to convert to lonlat coordinates, by default True.

    Returns:
        gpd.GeoDataFrame:
            Fitted h3 hex grid.

    Examples:
        Fit a hexagonal grid to a gdf:
        >>> from cellseg_gsontools import read_gdf
        >>> from cellseg_gsontools.grid import hexgrid_overlay
        >>> # Read in the tissue areas
        >>> area_gdf = gpd.read_file("path/to/area.geojson")
        >>> # Fit the grid
        >>> hex_grid = hexgrid_overlay(area_gdf, resolution=9)
        >>> hex_grid
        gpd.GeoDataFrame
    """
    if gdf.empty or gdf is None:
        return

    # drop invalid geometries if there are any after buffer
    gdf.geometry = gdf.geometry.buffer(0)
    gdf = gdf[gdf.is_valid]

    orig_crs = gdf.crs

    poly = shapely.force_2d(gdf.unary_union)
    if isinstance(poly, Polygon):
        hexagons = poly2hexgrid(poly, resolution=resolution, to_lonlat=to_lonlat)
    else:
        output = []
        for geom in poly.geoms:
            hexes = poly2hexgrid(geom, resolution=resolution, to_lonlat=to_lonlat)
            output.append(hexes)
        hexagons = pd.concat(output)

    return hexagons.set_crs(
        orig_crs, inplace=True, allow_override=True
    ).drop_duplicates("geometry")


def grid_overlay(
    gdf: gpd.GeoDataFrame,
    patch_size: Tuple[int, int] = (256, 256),
    stride: Tuple[int, int] = (256, 256),
    pad: int = 20,
    predicate: str = "intersects",
) -> gpd.GeoDataFrame:
    """Overlay a square grid to the given areas of a `geopandas.GeoDataFrame`.

    Note:
        Returns None if the gdf is empty.

    Parameters:
        gdf (gpd.GeoDataFrame):
            GeoDataFrame to fit the grid to. Uses the bounding box of the GeoDataFrame
            to fit the grid.
        patch_size (Tuple[int, int]):
            Patch size of the grid.
        stride (Tuple[int, int]):
            Stride of the sliding window in the grid.
        pad (int):
            Pad the bounding box with the given number of pixels, by default None.
        predicate (str):
            Predicate to use for the spatial join, by default "intersects".
            Allowed values are "intersects" and "within".

    Returns:
        gpd.GeoDataFrame:
            GeoDataFrame with the grid fitted to the given GeoDataFrame.

    Raises:
        ValueError: If predicate is not one of "intersects" or "within".

    Examples:
        Fit a square grid to a gdf:
        >>> from cellseg_gsontools import read_gdf
        >>> from cellseg_gsontools.grid import grid_overlay
        >>> # Read in the tissue areas
        >>> area_gdf = gpd.read_file("path/to/area.geojson")
        >>> # Fit the grid
        >>> sq_grid = grid_overlay(area_gdf, patch_size=(256, 256), stride=(256, 256))
        >>> sq_grid
        gpd.GeoDataFrame
    """
    if gdf.empty or gdf is None:
        return

    allowed = ["intersects", "within"]
    if predicate not in allowed:
        raise ValueError(f"predicate must be one of {allowed}. Got {predicate}")
    grid = get_grid(gdf, patch_size, stride, pad=pad)
    grid.set_crs(epsg=4328, inplace=True, allow_override=True)
    _, grid_inds = grid.sindex.query(gdf.geometry, predicate=predicate)
    grid = grid.iloc[np.unique(grid_inds)]
    # grid = grid.sjoin(gdf, predicate=predicate)

    return grid.drop_duplicates("geometry")


def poly2hexgrid(
    poly: Polygon, resolution: int = 9, to_lonlat: bool = True
) -> gpd.GeoDataFrame:
    """Convert a shapely Polygon to a h3 hexagon grid.

    Parameters:
        poly (Polygon):
            Polygon to convert.
        resolution (int, optional):
            H3 resolution, by default 9.
        to_lonlat (bool, optional):
            Whether to convert to lonlat coordinates, by default True.

    Returns:
        gpd.GeoDataFrame:
            GeoDataFrame of h3 hexagons.
    """
    x, y = poly.exterior.coords.xy
    if to_lonlat:
        x, y = xy_to_lonlat(x, y)
    holes = get_holes(poly, to_lonlat=to_lonlat)

    poly = Polygon(list(zip(x, y)), holes=holes)
    hexs = h3.polyfill(mapping(poly), resolution, geo_json_conformant=True)

    # to gdf
    hex_polys = gpd.GeoSeries(list(map(polygonise, hexs)), index=hexs, crs=4326)
    hex_polys = gpd.GeoDataFrame(geometry=hex_polys)

    return hex_polys


def polygonise(hex_id: str, to_cartesian: bool = True) -> Polygon:
    """Polygonise a h3 hexagon.

    Parameters:
        hex_id (str):
            H3 hexagon id.
        to_cartesian (bool, optional):
            Whether to convert to cartesian coordinates, by default True.

    Returns:
        Polygon:
            Polygonised h3 hexagon.
    """
    poly = Polygon(h3.h3_to_geo_boundary(hex_id, geo_json=True))

    if to_cartesian:
        lon, lat = poly.exterior.coords.xy
        x, y = lonlat_to_xy(lon, lat)
        poly = Polygon(list(zip(x, y)))

    return poly


def bounding_box(gdf: gpd.GeoDataFrame, pad: int = 0) -> gpd.GeoDataFrame:
    """Get the bounding box of a GeoDataFrame.

    Note:
        returns None if the gdf is empty.

    Parameters:
        gdf (gpd.GeoDataFrame):
            The input GeoDataFrame.
        pad (int):
            The padding to add to the bounding box.

    Returns:
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

    Parameters:
        first_endpoint (int):
            The first coordinate of the patch.
        size (int):
            The size of the input.
        stride (int):
            The stride of the sliding window.
        pad (int):
            The padding to add to the patch

    Returns:
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
    patch_size: Tuple[int, int] = (256, 256),
    stride: Tuple[int, int] = (256, 256),
    pad: int = 20,
) -> gpd.GeoDataFrame:
    """Get a grid of patches from a GeoDataFrame.

    Note:
        returns None if the gdf is empty.

    Parameters:
        gdf (gpd.GeoDataFrame):
            The input GeoDataFrame.
        patch_size (Tuple[int, int]):
            The size of the patch.
        stride (Tuple[int, int]):
            The stride of the sliding window.
        pad (int):
            The padding to add to the patch

    Returns:
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

    Parameters:
        rect (Polygon):
            The rectangle to get the metric of.
        objs (gpd.GeoDataFrame):
            The objects to use for the metric.
        metric_func (Callable):
            The metric function to use.
        predicate (str):
            The predicate to use for the spatial join. Allowed values are "intersects"
            and "within".

    Returns:
        Any:
            The metric of the rectangle.
    """
    allowed = ["intersects", "within"]
    if predicate not in allowed:
        raise ValueError(f"predicate must be one of {allowed}. Got {predicate}.")

    sub_objs = objs.iloc[objs.geometry.sindex.query(rect, predicate=predicate)]

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
    """Classify the grid based on objs inside the grid cells.

    Parameters:
        grid (gpd.GeoDataFrame):
            The grid of rectangles to classify.
        objs (gpd.GeoDataFrame):
            The objects to use for classification.
        metric_func (Callable):
            The metric/heuristic function to use for classification.
        predicate (str):
            The predicate to use for the spatial join. Allowed values are "intersects"
            and "within".
        new_col_names (Union[Tuple[str, ...], str]):
            The name of the new column(s) in the grid gdf.
        parallel (bool):
            Whether to use parallel processing.
        num_processes (int):
            The number of processes to use. If -1, uses all available cores.
            Ignored if parallel=False.
        pbar (bool):
            Whether to show a progress bar. Ignored if parallel=False.

    Returns:
        gpd.GeoDataFrame:
            The grid with the new columns added.

    Raises:
        ValueError: If predicate is not one of "intersects" or "within".

    Examples:
        Get the number of immune cells in each grid cell at the tumor stroma interface:
        >>> import geopandas as gpd
        >>> from shapely.geometry import Polygon
        >>> from functools import partial
        >>> from cellseg_gsontools import gdf_apply, read_gdf
        >>> from cellseg_gsontools.grid import grid_classify, grid_overlay
        >>> from cellseg_gsontools.context import InterfaceContext
        >>> # Define a heuristic function to get the number of immune cells
        >>> def get_immune_cell_cnt(gdf: gpd.GeoDataFrame, **kwargs) -> int:
        ...     try:
        ...         cnt = gdf.class_name.value_counts()["inflammatory"]
        ...     except KeyError:
        ...         cnt = 0
        ...     return int(cnt)
        >>> # Read in the tissue areas and cells
        >>> area_gdf = gpd.read_file("path/to/area.geojson")
        >>> cell_gdf = gpd.read_file("path/to/cell.geojson")
        >>> # Fit a tumor-stroma interface
        >>> tumor_stroma_iface = InterfaceContext(
        ...     area_gdf=area_gdf,
        ...     cell_gdf=cell_gdf,
        ...     top_labels="area_cin",
        ...     bottom_labels="areastroma",
        ...     buffer_dist=250,
        ...     graph_type="distband",
        ...     dist_thresh=75,
        ...     patch_size=(128, 128),
        ...     stride=(128, 128),
        ...     min_area_size=50000,
        ... )
        >>> tumor_stroma_iface.fit(verbose=False)
        >>> # Get the grid and the cells at the interface
        >>> iface_grid = grid_overlay(
        ...     tumor_stroma_iface.context2gdf("interface_area"),
        ...     patch_size=(128, 128),
        ...     stride=(128, 128),
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
        28  POLYGON ((20032.00000 54098.50000, 20160.00000... 15
        29  POLYGON ((20160.00000 54098.50000, 20288.00000... 3
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
