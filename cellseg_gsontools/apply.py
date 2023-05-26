from typing import Callable, Optional

import geopandas as gpd
from pandarallel import pandarallel

__all__ = ["gdf_apply"]


def gdf_apply(
    gdf: gpd.GeoDataFrame,
    func: Callable,
    col: Optional[str] = "geometry",
    extra_col: Optional[str] = None,
    parallel: bool = False,
    pbar: bool = False,
    **kwargs,
) -> gpd.GeoSeries:
    """Apply or parallel apply a function to any col of a GeoDataFrame.

    Parameters
    ----------
        gdf : gpd.GeoDataFrame
            Input GeoDataFrame.
        func : Callable
            A callable function.
        col : str, optional, default="geometry"
            The name of the column of the gdf that is used as the input
            to apply operation.
        extra_col : str, optional
            An extra column that can be used in the apply operation.
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
    Get the compactness of the polygons in a gdf
    >>> from cellseg_gsontools.utils import gdf_apply
    >>> gdf["compactness"] = gdf_apply(gdf, compactness, parallel=True)

    Get a list of QuPath readable geojson dict objects from a gdf
    >>> from cellseg_gsontools.utils import gdf_apply
    >>> from cellseg_gsontools.merging.save_utils import row_to_qupath
    >>> gdf_apply(gdf, row_to_qupath, col=None, axis=1, parallel=True).tolist()
    """
    if not parallel:
        if col is None:
            res = gdf.apply(func, **kwargs)
        else:
            if extra_col is None:
                res = gdf[col].apply(func, **kwargs)
            else:
                res = gdf[[col, extra_col]].apply(lambda x: func(*x, **kwargs), axis=1)
    else:
        pandarallel.initialize(verbose=1, progress_bar=pbar)
        if col is None:
            res = gdf.parallel_apply(func, **kwargs)
        else:
            if extra_col is None:
                res = gdf[col].parallel_apply(func, **kwargs)
            else:
                res = gdf[[col, extra_col]].parallel_apply(
                    lambda x: func(*x, **kwargs), axis=1
                )

    return res
