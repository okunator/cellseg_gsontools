from typing import Optional, Sequence, Tuple, Union

import geopandas as gpd
import numpy as np
from libpysal.weights import W

from cellseg_gsontools.apply import gdf_apply
from cellseg_gsontools.neighbors import neighborhood, nhood_vals
from cellseg_gsontools.utils import set_uid

__all__ = ["reduce", "local_character"]


def reduce(
    x: Sequence[Union[int, float]],
    areas: Optional[Sequence[float]] = None,
    how: str = "sum",
) -> float:
    """Reduce a numeric sequence.

    NOTE: Optionally can weight the input values based on area.

    Parameters
    ----------
        x : Sequence
            The input value-vector. Shape (n, )
        areas : Sequence, optional
            The areas of the spatial objects. This is for weighting. Optional.
        how : str, default="sum"
            The reduction method for the neighborhood. One of "sum", "mean", "median".

    Raises
    ------
        ValueError: If an illegal reduction method is given.

    Returns
    -------
        float:
            The mean, sum or median of the input array.
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
    else:
        allowed = ("sum", "mean", "median")
        ValueError(f"Illegal param `how`. Got: {how}, Allowed: {allowed}")

    return float(res)


def local_character(
    gdf: gpd.GeoDataFrame,
    spatial_weights: W,
    val_col: Union[str, Tuple[str, ...]],
    id_col: str = None,
    reductions: Tuple[str, ...] = ("sum",),
    weight_by_area: bool = False,
    parallel: bool = False,
    rm_nhood_cols: bool = True,
    col_prefix: str = None,
    create_copy: bool = True,
) -> gpd.GeoDataFrame:
    """Compute the local sum/mean/median of a specified metric for each row in a gdf.

    Local character: The sum/mean/median of the immediate neighborhood of a cell.

    NOTE: Option to weight the nhood values by their area before reductions.

    Parameters
    ----------
        gdf : gpd.GeoDataFrame
            The input GeoDataFrame.
        spatial_weights : libysal.weights.W
            Libpysal spatial weights object.
        val_col: Union[str, Tuple[str, ...]]
            The name of the column in the gdf for which the reduction is computed.
            If a tuple, the reduction is computed for each column.
        id_col : str, default=None
            The unique id column in the gdf. If None, this uses `set_uid` to set it.
        reductions : Tuple[str, ...], default=("sum", )
            A list of reduction methods for the neighborhood. One of "sum", "mean",
            "median".
        weight_by_area : bool, default=False
            Flag wheter to weight the neighborhood values by the area of the object.
        parallel : bool, default=False
            Flag whether to use parallel apply operations when computing the character
        rm_nhood_cols : bool, default=True
            Flag, whether to remove the extra neighborhood columns from the result gdf.
        col_prefix : str, optional
            Prefix for the new column names.
        create_copy : bool, default=True
            Flag whether to create a copy of the input gdf and return that.

    Returns
    -------
        gpd.GeoDataFrame:
            The input geodataframe with computed character column added.

    Examples
    --------
    Compute the mean of eccentricity values for each neighborhood
    >>> from libpysal.weights import DistanceBand
    >>> from cellseg_gsontools.character import local_character

    >>> w_dist = DistanceBand.from_dataframe(gdf, threshold=55.0, alpha=-1.0)
    >>> local_character(
    ...     gdf,
    ...     spatial_weights=w_dist,
    ...     val_col="eccentricity",
    ...     reduction=["mean"],
    ...     weight_by_area=True
    ... )
    """
    allowed = ("mean", "median", "sum")
    if not all(r in allowed for r in reductions):
        raise ValueError(
            f"Illegal reduction in `reductions`. Got: {reductions}. "
            f"Allowed reductions: {allowed}."
        )

    if create_copy:
        data = gdf.copy()
    else:
        data = gdf

    # set uid
    if id_col is None:
        id_col = "uid"
        data = set_uid(data)

    # Get the immediate node neighborhood
    data["nhood"] = gdf_apply(
        data, neighborhood, col="uid", spatial_weights=spatial_weights, parallel=False
    )

    # get areas
    area_col = None
    if weight_by_area:
        area_col = f"{val_col}_nhood_areas"
        data[area_col] = gdf_apply(
            data, nhood_vals, col="nhood", values=data.geometry.area, parallel=False
        )

    if isinstance(val_col, str):
        val_col = (val_col,)

    # get character values
    for col in val_col:
        values = data[col]
        char_col = f"{col}_nhood_vals"
        data[char_col] = gdf_apply(
            data, nhood_vals, col="nhood", values=values, parallel=False
        )

        # Compute the neighborhood characters
        col_prefix = "" if col_prefix is None else col_prefix
        # loop over the reduction methods
        for r in reductions:
            data[f"{col_prefix}{col}_nhood_{r}"] = gdf_apply(
                data,
                reduce,
                col=char_col,
                parallel=parallel,
                how=r,
                extra_col=area_col,
            )

        if rm_nhood_cols:
            data = data.drop(labels=[char_col], axis=1)

    if rm_nhood_cols:
        labs = ["nhood"]
        if weight_by_area:
            labs.append(area_col)
        data = data.drop(labels=labs, axis=1)

    return data
