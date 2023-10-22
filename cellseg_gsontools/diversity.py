"""Indices adapted from: https://github.com/pysal/inequality.

BSD 3-Clause License

Copyright (c) 2018, pysal-inequality developers
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from functools import partial
from typing import Sequence, Tuple, Union

import geopandas as gpd
import mapclassify
import numpy as np
from libpysal.weights import W

from .apply import gdf_apply
from .neighbors import neighborhood, nhood_counts, nhood_vals
from .utils import is_categorical, set_uid

__all__ = [
    "simpson_index",
    "shannon_index",
    "gini_index",
    "local_diversity",
]

SMALL = np.finfo("float").tiny


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
    N = np.sum(counts) + SMALL
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
    N = np.sum(counts) + SMALL
    probs = [float(n) / N for n in counts]

    entropy = -np.sum([p * np.log(p) for p in probs if p != 0])

    if entropy == 0:
        return 0.0

    return entropy


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
    nx = n * s + SMALL

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
    n = len(x)
    x = x + SMALL * (x == 0)  # can't have 0 values
    xt = np.sum(x, axis=0) + SMALL
    s = x / (xt * 1.0)
    lns = np.log(n * s)
    slns = s * lns
    t = np.sum(slns)

    return t


def theil_between_group(x: Sequence, partition: Sequence) -> float:
    """Between group Theil index.

    Parameters
    ----------
        x : Sequence
            The input value-vector. Shape (n, )
        partition : Sequence
            The groups for each x value. Shape (n, ).

    Returns
    -------
        float:
            The computed between group Theil index.
    """
    groups = np.unique(partition)
    x_total = x.sum(0) + SMALL

    # group totals
    g_total = np.array([x[partition == gid].sum(axis=0) for gid in groups])

    if x_total.size == 1:  # y is 1-d
        sg = g_total / (x_total * 1.0)
        sg.shape = (sg.size, 1)
    else:
        sg = np.dot(g_total, np.diag(1.0 / x_total))

    ng = np.array([np.sum(partition == gid) for gid in groups])
    ng.shape = (ng.size,)  # ensure ng is 1-d
    n = x.shape[0]

    # between group inequality
    sg = sg + (sg == 0)  # handle case when a partition has 0 for sum
    bg = np.multiply(sg, np.log(np.dot(np.diag(n * 1.0 / ng), sg))).sum(axis=0)

    return float(bg)


def theil_within_group(x: Sequence, partition: Sequence) -> float:
    """Within group Theil index.

    Parameters
    ----------
        x : Sequence
            The input value-vector. Shape (n, )
        partition : Sequence
            The groups for each x value. Shape (n, ).

    Returns
    -------
        float:
            The computed within group Theil index.
    """
    theil = theil_index(x)
    theil_bg = theil_between_group(x, partition)

    return float(theil - theil_bg)


DIVERSITY_LOOKUP = {
    "simpson_index": simpson_index,
    "shannon_index": shannon_index,
    "gini_index": gini_index,
    "theil_index": theil_index,
}


GROUP_DIVERSITY_LOOKUP = {
    "theil_between_group": theil_between_group,
    "theil_within_group": theil_within_group,
}


def local_diversity(
    gdf: gpd.GeoDataFrame,
    spatial_weights: W,
    val_col: Union[str, Tuple[str, ...]],
    id_col: str = None,
    metrics: Tuple[str, ...] = ("simpson_index",),
    scheme: str = "FisherJenks",
    parallel: bool = True,
    rm_nhood_cols: bool = True,
    col_prefix: str = None,
    create_copy: bool = True,
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
        val_col : Union[str, Tuple[str, ...]]
            The name of the column in the gdf for which the diversity is computed.
            You can also pass in a list of columns, in which case the diversity is
            computed for each column.
        id_col : str, default=None
            The unique id column in the gdf. If None, this uses `set_uid` to set it.
        metrics : Tuple[str, ...]
            A Tuple/List of diversity metrics. Allowed metrics: "shannon_index",
            "simpson_index", "gini_index", "theil_index"
        scheme : str, default="HeadTailBreaks"
            `pysal.mapclassify` classification scheme.
        parallel : bool, default=True
            Flag whether to use parallel apply operations when computing the diversities
        rm_nhood_cols : bool, default=True
            Flag, whether to remove the extra neighborhood columns from the result gdf.
        col_prefix : str, optional
            Prefix for the new column names.
        create_copy : bool, default=True
            Flag whether to create a copy of the input gdf or not.

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
    >>> from cellseg_gsontools.diversity import local_diversity

    >>> w_dist = DistanceBand.from_dataframe(gdf, threshold=55.0, alpha=-1.0)
    >>> local_diversity(
    ...     gdf,
    ...     spatial_weights=w_dist,
    ...     val_col="eccentricity",
    ...     metrics=["simpson_index"]
    ... )
    """
    allowed = list(DIVERSITY_LOOKUP.keys())
    if not all(m in allowed for m in metrics):
        raise ValueError(
            f"Illegal metric in `metrics`. Got: {metrics}. Allowed metrics: {allowed}."
        )

    if create_copy:
        gdf = gdf.copy()

    # set uid
    if id_col is None:
        id_col = "uid"
        gdf = set_uid(gdf)

    # If shannon or simpson index in metrics, counts are needed
    ret_counts = False
    if any([m in metrics for m in ("simpson_index", "shannon_index")]):
        ret_counts = True

    # If Gini is in metrics, neighboring values are needed
    gt = ("gini_index", "theil_index")
    ret_vals = False
    if any([m in metrics for m in gt]):
        ret_vals = True

    # Get the immediate node neighborhood
    func = partial(neighborhood, spatial_weights=spatial_weights)
    gdf["nhood"] = gdf_apply(gdf, func, columns=[id_col], axis=1, parallel=parallel)

    if isinstance(val_col, str):
        val_col = (val_col,)

    for col in val_col:
        values = gdf[col]

        # Get bins if data not categorical
        if not is_categorical(values):
            bins = mapclassify.classify(values, scheme=scheme).bins
        else:
            bins = None

        # Get the counts of the binned metric inside the neighborhoods
        if ret_counts:
            func = partial(nhood_counts, values=values, bins=bins)
            gdf[f"{col}_nhood_counts"] = gdf_apply(
                gdf,
                func,
                columns=["nhood"],
                axis=1,
                parallel=parallel,
            )

        if ret_vals:
            func = partial(nhood_vals, values=values)
            gdf[f"{col}_nhood_vals"] = gdf_apply(
                gdf,
                func,
                columns=["nhood"],
                axis=1,
                parallel=parallel,
            )

        # Compute the diversity metrics for the neighborhood counts
        for metric in metrics:
            colname = f"{col}_nhood_counts" if metric not in gt else f"{col}_nhood_vals"

            col_prefix = "" if col_prefix is None else col_prefix
            gdf[f"{col_prefix}{col}_{metric}"] = gdf_apply(
                gdf,
                DIVERSITY_LOOKUP[metric],
                columns=[colname],
                parallel=parallel,
            )

        if rm_nhood_cols:
            gdf = gdf.drop(labels=[colname], axis=1)

    if rm_nhood_cols:
        gdf = gdf.drop(labels=["nhood"], axis=1)

    return gdf
