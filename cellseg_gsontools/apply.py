from typing import Callable, Optional, Tuple

import geopandas as gpd
import psutil
from pandarallel import pandarallel

__all__ = ["gdf_apply"]


def gdf_apply(
    gdf: gpd.GeoDataFrame,
    func: Callable,
    axis: int = 1,
    parallel: bool = True,
    num_processes: Optional[int] = -1,
    pbar: bool = False,
    columns: Optional[Tuple[str, ...]] = None,
    **kwargs,
) -> gpd.GeoSeries:
    """Apply or parallel apply a function to any col or row of a GeoDataFrame.

    Parameters:
        gdf (gpd.GeoDataFrame):
            Input GeoDataFrame.
        func (Callable):
            A callable function.
        axis (int, default=1):
            The gdf axis to apply the function on.axis=1 means rowise. axis=0
            means columnwise.
        parallel (bool, default=True):
            Flag, whether to parallelize the operation with `pandarallel`.
        num_processes (int, default=-1):
            The number of processes to use when parallel=True. If -1,
            this will use all available cores.
        pbar (bool, default=False):
            Show progress bar when executing in parallel mode. Ignored if
            `parallel=False`.
        columns (Optional[Tuple[str, ...]], default=None):
            A tuple of column names to apply the function on. If None,
            this will apply the function to all columns.
        **kwargs (Dict[str, Any]): Arbitrary keyword args for the `func` callable.

    Returns:
        gpd.GeoSeries:
            A GeoSeries object containing the computed values for each
            row or col in the input gdf.

    Examples:
        Get the compactness of the polygons in a gdf
        >>> from cellseg_gsontools import gdf_apply
        >>> gdf["compactness"] = gdf_apply(
        ...     gdf, compactness, columns=["geometry"], parallel=True
        ... )
    """
    if columns is not None:
        if not isinstance(columns, (tuple, list)):
            raise ValueError(f"columns must be a tuple or list, got {type(columns)}")
        gdf = gdf[columns]

    if not parallel:
        res = gdf.apply(lambda x: func(*x, **kwargs), axis=axis)
    else:
        cpus = psutil.cpu_count(logical=False) if num_processes == -1 else num_processes
        pandarallel.initialize(verbose=1, progress_bar=pbar, nb_workers=cpus)
        res = gdf.parallel_apply(lambda x: func(*x, **kwargs), axis=axis)

    return res
