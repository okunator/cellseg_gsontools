from typing import Callable

import geopandas as gpd
from pandarallel import pandarallel

__all__ = ["gdf_apply"]


def gdf_apply(
    gdf: gpd.GeoDataFrame,
    func: Callable,
    col: str = "geometry",
    parallel: bool = False,
    pbar: bool = False,
    **kwargs,
) -> gpd.GeoSeries:
    """Apply or parallel apply a function to a GeoDataFrame.geometry.

    Parameters
    ----------
        gdf : gpd.GeoDataFrame
            Input GeoDataFrame.
        func : Callable
            A callable function.
        col : str, default="geometry"
            The name of the column of the gdf that is used as the input
            to apply operation.
        parallel : bool, default=False
            Flag, whether to parallelize the operation with pandarallel
        pbar : bool, default=False
            Show progress bar when executing in parallel mode. Ignored if
            `parallel=False`
        **kwargs:
            Arbitrary keyword args for the `func` callable.

    Returns
    -------
        gpd.GeoSeries
            A GeoSeries object containing the metrics computed for each polygon in the
            input gdf.

    Examples
    --------
        >>> # Get the compactness of the polygons in a gdf
        >>> gdf["compactness"] = gdf_apply(gdf, compactness, parallel=True)
    """
    if not parallel:
        res = gdf[col].apply(func, **kwargs)
    else:
        pandarallel.initialize(verbose=1, progress_bar=pbar)
        res = gdf[col].parallel_apply(func, **kwargs)

    return res
