from typing import Callable

import geopandas as gpd
from pandarallel import pandarallel

__all__ = ["gdf_apply"]


def gdf_apply(
    gdf: gpd.GeoDataFrame,
    func: Callable,
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
        parallel : bool, default=False
            Flag, whether to parallelize the operation with pandarallel
        pbar : bool, default=False
            Show progress bar when executing in parallel mode. Ignored if
            `parallel=False`

    Returns
    -------
        gpd.GeoSeries
            A GeoSeries object containing the metrics computed for each polygon in the
            input gdf.
    """
    if not parallel:
        res = gdf.geometry.apply(func, **kwargs)
    else:
        pandarallel.initialize(verbose=1, progress_bar=pbar)
        res = gdf.geometry.parallel_apply(func, **kwargs)

    return res
