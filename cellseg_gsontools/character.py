from functools import partial
from typing import Optional, Sequence, Tuple, Union

import geopandas as gpd
import numpy as np
from libpysal.weights import W

from cellseg_gsontools.apply import gdf_apply
from cellseg_gsontools.neighbors import neighborhood, nhood_dists, nhood_vals
from cellseg_gsontools.utils import set_uid

__all__ = ["reduce", "local_character", "local_distances"]


def reduce(
    x: Sequence[Union[int, float]],
    areas: Optional[Sequence[float]] = None,
    how: str = "sum",
) -> float:
    """Reduce a numeric sequence.

    Note:
        Optionally can weight the input values based on area.

    Parameters:
        x (Sequence):
            The input value-vector. Shape (n, )
        areas (Sequence, optional):
            The areas of the spatial objects. This is for weighting. Optional.
        how (str, default="sum"):
            The reduction method for the neighborhood. One of:
            "sum", "mean", "median", "min", "max", "std".

    Raises:
        ValueError:
            If an illegal reduction method is given.

    Returns:
        float:
            The reduced value of the input array.
    """
    w = 1.0
    if areas is not None:
        w = areas / (np.sum(areas) + 1e-8)

    res = 0
    if how == "sum":
        res = np.sum(x * w)
    elif how == "mean":
        res = np.mean(x * w)
    elif how == "median":
        res = np.median(x * w)
    elif how == "max":
        res = np.max(x * w)
    elif how == "min":
        res = np.min(x * w)
    elif how == "std":
        res = np.std(x * w)
    else:
        allowed = ("sum", "mean", "median", "min", "max", "std")
        ValueError(f"Illegal param `how`. Got: {how}, Allowed: {allowed}")

    return float(res)


def local_character(
    gdf: gpd.GeoDataFrame,
    spatial_weights: W,
    val_col: Union[str, Tuple[str, ...]],
    id_col: str = None,
    reductions: Tuple[str, ...] = ("sum",),
    weight_by_area: bool = False,
    parallel: bool = True,
    num_processes: int = -1,
    rm_nhood_cols: bool = True,
    col_prefix: str = None,
    create_copy: bool = True,
) -> gpd.GeoDataFrame:
    """Compute the local sum/mean/median/min/max/std of a specified metric for each
    neighborhood of geometry objects in a gdf.

    Note:
        Option to weight the nhood values by their area before reductions.

    Parameters:
        gdf (gpd.GeoDataFrame):
            The input GeoDataFrame.
        spatial_weights (libysal.weights.W):
            Libpysal spatial weights object.
        val_col (Union[str, Tuple[str, ...]]):
            The name of the column in the gdf for which the reduction is computed.
            If a tuple, the reduction is computed for each column.
        id_col (str):
            The unique id column in the gdf. If None, this uses `set_uid` to set it.
            Defaults to None.
        reductions (Tuple[str, ...]):
            A list of reduction methods for the neighborhood. One of
            "sum", "mean", "median", "min", "max", "std". Defaults to ("sum", ).
        weight_by_area (bool):
            Flag whether to weight the neighborhood values by the area of the object.
            Defaults to False.
        parallel (bool):
            Flag whether to use parallel apply operations when computing the character.
            Defaults to True.
        num_processes (int, default=-1):
            The number of processes to use when parallel=True. If -1,
            this will use all available cores.
        rm_nhood_cols (bool):
            Flag, whether to remove the extra neighborhood columns from the result gdf.
            Defaults to True.
        col_prefix (str):
            Prefix for the new column names.
        create_copy (bool):
            Flag whether to create a copy of the input gdf and return that.
            Defaults to True.

    Returns:
        gpd.GeoDataFrame:
            The input geodataframe with computed character column added.

    Examples:
        Compute the mean of eccentricity values for each cell neighborhood
        >>> from cellseg_gsontools.character import local_character
        >>> from cellseg_gsontools.graphs import fit_graph
        >>> w = fit_graph(gdf, type="distband", thres=75.0)
        >>> local_character(
        ...     gdf,
        ...     spatial_weights=w,
        ...     val_col=["eccentricity", "area"],
        ...     reduction=["mean", "median"],
        ...     weight_by_area=True,
        ... )
    """
    allowed = ("sum", "mean", "median", "min", "max", "std")
    if not all(r in allowed for r in reductions):
        raise ValueError(
            f"Illegal reduction in `reductions`. Got: {reductions}. "
            f"Allowed reductions: {allowed}."
        )

    if create_copy:
        gdf = gdf.copy()

    # set uid
    if id_col is None:
        id_col = "uid"
        gdf = set_uid(gdf)

    # Get the immediate node neighborhood
    func = partial(neighborhood, spatial_weights=spatial_weights)
    gdf["nhood"] = gdf_apply(
        gdf,
        func,
        columns=[id_col],
        axis=1,
        parallel=parallel,
        num_processes=num_processes,
    )

    # get areas
    area_col = None
    if weight_by_area:
        area_col = "nhood_areas"
        func = partial(nhood_vals, values=gdf.geometry.area)
        gdf[area_col] = gdf_apply(
            gdf,
            func,
            columns=["nhood"],
            axis=1,
            parallel=parallel,
            num_processes=num_processes,
        )

    if isinstance(val_col, str):
        val_col = (val_col,)

    # get character values
    # Compute the neighborhood characters
    col_prefix = "" if col_prefix is None else col_prefix
    for col in val_col:
        values = gdf[col]
        char_col = f"{col}_nhood_vals"
        func = partial(nhood_vals, values=values)
        gdf[char_col] = gdf_apply(
            gdf,
            func,
            columns=["nhood"],
            axis=1,
            parallel=parallel,
            num_processes=num_processes,
        )

        # loop over the reduction methods
        for r in reductions:
            columns = [char_col]
            new_col = f"{col_prefix}{col}_nhood_{r}"
            if area_col in gdf.columns:
                columns.append(area_col)
                new_col = f"{col_prefix}{col}_nhood_{r}_area_weighted"

            func = partial(reduce, how=r)
            gdf[new_col] = gdf_apply(
                gdf,
                func,
                columns=columns,
                axis=1,
                parallel=parallel,
                num_processes=num_processes,
            )

        if rm_nhood_cols:
            gdf = gdf.drop(labels=[char_col], axis=1)

    if rm_nhood_cols:
        labs = ["nhood"]
        if weight_by_area:
            labs.append(area_col)
        gdf = gdf.drop(labels=labs, axis=1)

    return gdf


def local_distances(
    gdf: gpd.GeoDataFrame,
    spatial_weights: W,
    id_col: str = None,
    reductions: Tuple[str, ...] = ("mean",),
    weight_by_area: bool = False,
    invert: bool = False,
    parallel: bool = True,
    num_processes: int = -1,
    rm_nhood_cols: bool = True,
    col_prefix: str = None,
    create_copy: bool = True,
) -> gpd.GeoDataFrame:
    """Compute the local sum/mean/median/min/max/std distance of the neighborhood
    distances for each geometry object in a gdf.

    Note:
        Option to weight the nhood values by their area before reductions.

    Parameters:
        gdf (gpd.GeoDataFrame):
            The input GeoDataFrame.
        spatial_weights (libysal.weights.W):
            Libpysal spatial weights object.
        id_col (str):
            The unique id column in the gdf. If None, this uses `set_uid` to set it.
            Defaults to None.
        reductions (Tuple[str, ...]):
            A list of reduction methods for the neighborhood. One of "sum", "mean",
            "median", "min", "max", "std". Defaults to ("sum", ).
        weight_by_area (bool):
            Flag whether to weight the neighborhood values by the area of the object.
            Defaults to False.
        invert (bool):
            Flag whether to invert the distances. Defaults to False.
        parallel (bool):
            Flag whether to use parallel apply operations when computing the character.
            Defaults to True.
        num_processes (int, default=-1):
            The number of processes to use when parallel=True. If -1,
            this will use all available cores.
        rm_nhood_cols (bool):
            Flag, whether to remove the extra neighborhood columns from the result gdf.
            Defaults to True.
        col_prefix (str):
            Prefix for the new column names.
        create_copy (bool):
            Flag whether to create a copy of the input gdf and return that.
            Defaults to True.

    Returns:
        gpd.GeoDataFrame:
            The input geodataframe with computed distances column added.

    Examples:
        Compute the mean of eccentricity values for each neighborhood
        >>> from cellseg_gsontools.character import local_distances
        >>> from cellseg_gsontools.graphs import fit_graph
        >>> w = fit_graph(gdf, type="distband", thres=75.0)
        >>> local_distances(gdf, spatial_weights=w, reduction=["mean"], weight_by_area=True)
    """
    allowed = ("sum", "mean", "median", "min", "max", "std")
    if not all(r in allowed for r in reductions):
        raise ValueError(
            f"Illegal reduction in `reductions`. Got: {reductions}. "
            f"Allowed reductions: {allowed}."
        )

    if create_copy:
        gdf = gdf.copy()

    # set uid
    if id_col is None:
        id_col = "uid"
        gdf = set_uid(gdf)

    # get the immediate node neighborhood
    func = partial(neighborhood, spatial_weights=spatial_weights)
    gdf["nhood"] = gdf_apply(
        gdf,
        func,
        columns=[id_col],
        axis=1,
        parallel=parallel,
        num_processes=num_processes,
    )

    # get areas
    area_col = None
    if weight_by_area:
        func = partial(nhood_vals, values=gdf.geometry.area)
        gdf[area_col] = gdf_apply(
            gdf,
            func,
            columns=["nhood"],
            axis=1,
            parallel=parallel,
            num_processes=num_processes,
        )

    # get distances
    func = partial(nhood_dists, centroids=gdf.centroid, invert=invert)
    gdf["nhood_dists"] = gdf_apply(
        gdf,
        func,
        columns=["nhood"],
        axis=1,
        parallel=parallel,
        num_processes=num_processes,
    )

    col_prefix = "" if col_prefix is None else col_prefix

    # loop over the reduction methods
    for r in reductions:
        columns = ["nhood_dists"]
        new_col = f"{col_prefix}nhood_dists_{r}"
        if area_col in gdf.columns:
            columns.append(area_col)
            new_col = f"{col_prefix}nhood_dists_{r}_area_weighted"

        func = partial(reduce, how=r)
        gdf[new_col] = gdf_apply(
            gdf,
            func,
            columns=columns,
            axis=1,
            parallel=parallel,
            num_processes=num_processes,
        )

    if rm_nhood_cols:
        labs = ["nhood", "nhood_dists"]
        if weight_by_area:
            labs.append(area_col)
        gdf = gdf.drop(labels=labs, axis=1)

    return gdf
