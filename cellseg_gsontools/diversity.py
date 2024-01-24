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
    """Compute the Simpson diversity index on a count vector.

    Note:
        Simpson diversity index is a quantitative measure that reflects how many
        different types (such as species) there are in a dataset (a community). It
        is a probability measure, when it is low, the greater the probability that
        two randomly selected individuals will be the same species.
        - [A. Wilson, N. Gownaris](https://bio.libretexts.org/Courses/Gettysburg_College/01%3A_Ecology_for_All/22%3A_Biodiversity/22.02%3A_Diversity_Indices)


    **Simpson index:**
    $$
    D = 1 - \\sum_{i=1}^n \\left(\\frac{n_i}{N}\\right)^2
    $$

    where $n_i$ is the count of species $i$ and $N$ is the total count of species.

    Parameters:
        counts (Sequence):
            A count vector/list of shape (C, ).

    Returns:
        float:
            The computed Simpson diversity index.
    """
    N = np.sum(counts) + SMALL
    return 1 - np.sum([(n / N) ** 2 for n in counts if n != 0])


def shannon_index(counts: Sequence) -> float:
    """Compute the Shannon Weiner index/entropy on a count vector.

    Note:
        "*The Shannon index is related to the concept of uncertainty. If for example,
        a community has very low diversity, we can be fairly certain of the identity of
        an organism we might choose by random (high certainty or low uncertainty). If a
        community is highly diverse and we choose an organism by random, we have a
        greater uncertainty of which species we will choose (low certainty or high
        uncertainty).*"
        - [A. Wilson, N. Gownaris](https://bio.libretexts.org/Courses/Gettysburg_College/01%3A_Ecology_for_All/22%3A_Biodiversity/22.02%3A_Diversity_Indices)

    **Shannon index:**
    $$
    H^{\\prime} = -\\sum_{i=1}^n p_i \\ln(p_i)
    $$

    where $p_i$ is the proportion of species $i$ and $n$ is the total count of species.

    Parameters:
        counts (Sequence):
            A count vector/list of shape (C, ).

    Returns:
        float:
            The computed Shannon diversity index.
    """
    N = np.sum(counts) + SMALL
    probs = [float(n) / N for n in counts]

    entropy = -np.sum([p * np.log(p) for p in probs if p != 0])

    if entropy == 0:
        return 0.0

    return entropy


def gini_index(x: Sequence) -> float:
    """Compute the gini coefficient of inequality for species.

    Note:
        This is based on
        http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm.

    **Gini-index:**
    $$
    G = \\frac{\\sum_{i=1}^n (2i - n - 1)x_i} {n \\sum_{i=1}^n x_i}
    $$

    where $x_i$ is the count of species $i$ and $n$ is the total count of species.

    Parameters:
        x (Sequence):
            The input value-vector. Shape (n, )

    Raises:
        ValueError:
            If there are negative input values.

    Returns:
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

    **Theil-index:**
    $$
    T = \\sum_{i=1}^n \\left(
        \\frac{y_i}{\\sum_{i=1}^n y_i} \\ln \\left[
            N \\frac{y_i} {\\sum_{i=1}^n y_i}
        \\right]\\right)
    $$

    where $y_i$ is the count of species $i$ and $N$ is the total count of species.

    Parameters:
        x (Sequence):
            The input value-vector. Shape (n, )

    Returns:
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
    """Compute the between group Theil index.

    Parameters:
        x (Sequence):
            The input value-vector. Shape (n, )
        partition (Sequence):
            The groups for each x value. Shape (n, ).

    Returns:
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
    """Compute the within group Theil index.

    Parameters:
        x (Sequence):
            The input value-vector. Shape (n, )
        partition (Sequence):
            The groups for each x value. Shape (n, ).

    Returns:
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
    num_processes: int = -1,
    rm_nhood_cols: bool = True,
    col_prefix: str = None,
    create_copy: bool = True,
) -> gpd.GeoDataFrame:
    """Compute the local diversity/heterogenity metric for cell neighborhood.

    Note:
        Allowed diversity metrics:

        - `simpson_index` - for both categorical and real valued neighborhoods
        - `shannon_index` - for both categorical and real valued neighborhoods
        - `gini_index` - for only real valued neighborhoods
        - `theil_index` - for only real valued neighborhoods

    Note:
        If `val_col` is not categorical, the values are binned using `mapclassify`.
        The bins are then used to compute the diversity metrics. If `val_col` is
        categorical, the values are used directly.

    Parameters:
        gdf (gpd.GeoDataFrame):
            The input GeoDataFrame.
        spatial_weights (libysal.weights.W):
            Libpysal spatial weights object.
        val_col (Union[str, Tuple[str, ...]]):
            The name of the column in the gdf for which the diversity is computed.
            You can also pass in a list of columns, in which case the diversity is
            computed for each column.
        id_col (str):
            The unique id column in the gdf. If None, this uses `set_uid` to set it.
            Defaults to None.
        metrics (Tuple[str, ...]):
            A Tuple/List of diversity metrics. Allowed metrics: "shannon_index",
            "simpson_index", "gini_index", "theil_index". Defaults to None.
        scheme (str):
            `mapclassify` classification scheme. Defaults to "FisherJenks".
        parallel (bool):
            Flag whether to use parallel apply operations when computing the diversities.
            Defaults to True.
        num_processes (int, default=-1):
            The number of processes to use when parallel=True. If -1,
            this will use all available cores.
        rm_nhood_cols (bool):
            Flag, whether to remove the extra neighborhood columns from the result gdf.
            Defaults to True.
        col_prefix (str):
            Prefix for the new column names. Defaults to None.
        create_copy (bool):
            Flag whether to create a copy of the input gdf or not. Defaults to True.

    Raises:
        ValueError:
            If an illegal metric is given.

    Returns:
        gpd.GeoDataFrame:
            The input geodataframe with computed diversity metric columns added.

    Examples:
        Compute the simpson diversity of eccentricity values for each cell neighborhood
        >>> from cellseg_gsontools.diversity import local_diversity
        >>> from cellseg_gsontools.graphs import fit_graph
        >>> w = fit_graph(gdf, type="distband", thres=75.0)
        >>> local_diversity(
        ...     gdf,
        ...     spatial_weights=w_dist,
        ...     val_col="eccentricity",
        ...     metrics=["simpson_index"],
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
    gdf["nhood"] = gdf_apply(
        gdf,
        func,
        columns=[id_col],
        axis=1,
        parallel=parallel,
        num_processes=num_processes,
    )

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
                num_processes=num_processes,
            )

        if ret_vals:
            func = partial(nhood_vals, values=values)
            gdf[f"{col}_nhood_vals"] = gdf_apply(
                gdf,
                func,
                columns=["nhood"],
                axis=1,
                parallel=parallel,
                num_processes=num_processes,
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
                num_processes=num_processes,
            )

        if rm_nhood_cols:
            gdf = gdf.drop(labels=[colname], axis=1)

    if rm_nhood_cols:
        gdf = gdf.drop(labels=["nhood"], axis=1)

    return gdf
