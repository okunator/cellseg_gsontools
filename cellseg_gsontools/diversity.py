from typing import Sequence, Tuple

import geopandas as gpd
import mapclassify
import numpy as np
from libpysal.weights import W

from .apply import gdf_apply
from .neighbors import neighborhood, nhood_counts, nhood_vals
from .utils import set_uid

__all__ = ["simpson_index", "shannon_index", "gini_index", "local_diversity"]


def simpson_index(counts: Sequence) -> float:
    """Compute the simpson diversity index on a count vector.

    Simpson diversity index: 1 - sum((species_count / total_count)^2)

    Parameters
    ----------
        counts : Sequence
            A count vector/list of shape (C, ).

    Returns
    -------
        float:
            The computed simpson diversity index.
    """
    N = np.sum(counts)
    return 1 - np.sum([(n / N) ** 2 for n in counts if n != 0])


def shannon_index(counts: Sequence) -> float:
    """Compute the shannon index/entropy on a count vector.

    Shannon index: -sum(p_i * ln(p_i))

    Parameters
    ----------
        counts : Sequence
            A count vector/list of shape (C, ).

    Returns
    -------
        float:
            The computed shannon diversity index.
    """
    N = np.sum(counts)
    probs = [float(n) / N for n in counts]

    return -np.sum([p * np.log(p) for p in probs if p != 0])


def gini_index(x: Sequence) -> float:
    """Compute the gini coefficient of inequality for species.

    This is based on
    http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm.

    Parameters
    ----------
        x : Sequence
            The input value-vector. Shape (n, )

    Raises
    ------
        ValueError: If there are negative input values.

    Returns
    -------
        float:
            The computed Gini coefficient.
    """
    if np.min(x) < 0:
        raise ValueError("Input values need to be positive for Gini coeff")

    n = len(x)
    s = np.sum(x)
    nx = n * s

    rx = (2.0 * np.arange(1, n + 1) * x[np.argsort(x)]).sum()
    return (rx - nx - s) / nx


def theil_index(x: Sequence) -> float:
    """Compute the Theil index of inequality for species.

    Parameters
    ----------
        x : Sequence
            The input value-vector. Shape (n, )

    Returns
    -------
        float:
            The computed Theil index.
    """
    SMALL = np.finfo("float").tiny

    n = len(x)
    x = x + SMALL * (x == 0)  # can't have 0 values
    xt = np.sum(x, axis=0)
    s = x / (xt * 1.0)
    lns = np.log(n * s)
    slns = s * lns
    t = np.sum(slns)

    return t


DIVERSITY_LOOKUP = {
    "simpson_index": simpson_index,
    "shannon_index": shannon_index,
    "gini_index": gini_index,
    "theil_index": theil_index,
}


def local_diversity(
    gdf: gpd.GeoDataFrame,
    spatial_weights: W,
    val_col: str,
    metrics: Tuple[str, ...] = ("simpson_index",),
    categorical: bool = False,
    parallel: bool = False,
    rm_nhood_cols: bool = True,
) -> gpd.GeoDataFrame:
    """Compute the local diversity/heterogenity metric for each row in a gdf.

    Local diversity: The diversity metric is computed from the immediate
        neighborhood of the node/cell.

    Parameters
    ----------
        gdf : gpd.GeoDataFrame
            The input GeoDataFrame.
        spatial_weights : libysal.weights.W
            Libpysal spatial weights object.
        val_col: str
            The name of the column in the gdf for which the diversity is computed
        metrics : Tuple[str, ...]
            A Tuple/List of diversity metrics. Allowed metrics: "shannon_entropy",
            "simpson_index", "gini_index", "theil_index"
        categorical : bool, default=False
            A flag, signalling whether the `val_col` column of the gdf is categorical.
        parallel : bool, default=False
            Flag whether to use parallel apply operations when computing the diversities
        rm_nhood_cols : bool, default=True
            Flag, whether to remove the extra neighborhood columns from the result gdf.

    Raises
    ------
        ValueError: If an illegal metric is given.

    Returns
    -------
        gpd.GeoDataFrame:
            The input geodataframe with computed diversity metric columns added.

    Examples
    --------
    Compute the simpson diversity of eccentricity values for each neighborhood

        >>> from libpysal.weights import DistanceBand
        >>> from cellseg_gsontools.diversity import get_diversity

        >>> w_dist = DistanceBand.from_dataframe(gdf, threshold=55.0, alpha=-1.0)
        >>> diversity(
                gdf,
                spatial_weights=w_dist,
                val_col="eccentricity",
                metrics=["simpson_index"]
            )
    """
    allowed = list(DIVERSITY_LOOKUP.keys())
    if not all(m in allowed for m in metrics):
        raise ValueError(
            f"Illegal metric in `metrics`. Got: {metrics}. Allowed metrics: {allowed}."
        )

    # If shannon or simpson index in metrics, counts are needed
    ret_counts = False
    if any([m in metrics for m in ("simpson_index", "shannon_index")]):
        ret_counts = True

    # If Gini is in metrics, neighboring values are needed
    ret_vals = False
    if any([m in metrics for m in ("gini_index", "theil_index")]):
        ret_vals = True

    # set uid
    data = set_uid(gdf)
    data = data.set_index("uid", drop=False)

    # Get the immediate node neighborhood
    data["nhood"] = gdf_apply(
        data,
        neighborhood,
        col="uid",
        spatial_weights=spatial_weights,
    )

    # Get bins if data not categorical
    values = data[val_col]
    if not categorical:
        bins = mapclassify.classify(values, scheme="HeadTailBreaks").bins
    else:
        bins = None

    # Get the counts of the binned metric inside the neighborhoods
    if ret_counts:
        data[f"{val_col}_nhood_counts"] = gdf_apply(
            data,
            nhood_counts,
            col="nhood",
            values=values,
            bins=bins,
            categorical=categorical,
        )

    if ret_vals:
        data[f"{val_col}_nhood_vals"] = gdf_apply(
            data,
            nhood_vals,
            col="nhood",
            values=values,
        )

    # Compute the diversity metrics for the neighborhood counts
    for metric in metrics:
        colname = (
            f"{val_col}_nhood_counts"
            if metric not in ("gini_index", "theil_index")
            else f"{val_col}_nhood_vals"
        )

        data[f"{val_col}_{metric}"] = gdf_apply(
            data,
            DIVERSITY_LOOKUP[metric],
            col=colname,
            parallel=parallel,
        )

    if rm_nhood_cols:
        data = data.drop(labels=[colname, "nhood"], axis=1)

    return data
