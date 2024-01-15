from typing import List, Sequence, Union

import mapclassify
import numpy as np
import pandas as pd
from libpysal.weights import W

from .geometry.axis import _dist
from .utils import is_categorical

__all__ = [
    "neighborhood",
    "nhood_counts",
    "nhood_vals",
    "nhood_type_count",
    "nhood_dists",
]


def neighborhood(
    node: Union[int, pd.Series],
    spatial_weights: W,
    include_self: bool = True,
    ret_n_neighbors: bool = False,
) -> Union[List[int], int]:
    """Get immediate neighborhood of a node given the spatial weights obj.

    Note:
        This function is designed to be used with the `gdf_apply` function.
        See the example.

    Note:
        The neighborhood contains the given node itself by default.

    Parameters:
        node (int or pd.Series):
            Input node uid.
        spatial_weights (libysal.weights.W):
            Libpysal spatial weights object.
        include_self (bool):
            Flag, whether to include the node itself in the neighborhood.
            Defaults to True.
        ret_n_neighbors (bool):
            If True, instead of returning a sequence of the neighbor node uids
            returns just the number of neighbors. Defaults to False.

    Returns:
        List[int] or int:
            A list of the neighboring node uids. E.g. [1, 4, 19].
            or the number of neighbors if `ret_n_neighbors=True`.

    Examples:
        Use `gdf_apply` to extract the neighboring nodes for each node/cell
        >>> from functools import partial
        >>> from cellseg_gsontools.data import gland_cells
        >>> from cellseg_gsontools.graphs import fit_graph
        >>> from cellseg_gsontools.utils import set_uid
        >>> from cellseg_gsontools.apply import gdf_apply
        >>> from cellseg_gsontools.neighbors import neighborhood
        >>> gc = gland_cells()
        >>> # To fit the delaunay graph, we need to set a unique id for each cell first
        >>> gc = set_uid(gc, id_col="uid")
        >>> w = fit_graph(gc, type="delaunay", thresh=100, id_col="uid")
        >>> # Get the neihgboring nodes of the graph
        >>> func = partial(neighborhood, spatial_weights=w)
        >>> gc["nhood"] = gdf_apply(gc, func, columns=["uid"])
        >>> gc["nhood"].head(5)
                uid
        0       [0, 1, 3, 4, 483, 484]
        1     [1, 0, 4, 482, 483, 487]
        2       [2, 3, 5, 6, 484, 493]
        3      [3, 0, 2, 4, 5, 7, 484]
        4    [4, 0, 1, 3, 7, 487, 488]
        Name: nhood, dtype: object
    """
    if isinstance(node, pd.Series):
        node = node.iloc[0]  # assume that the series is a row

    nhood = np.nan
    if ret_n_neighbors:
        nhood = spatial_weights.cardinalities[node]
    elif node in spatial_weights.neighbors.keys():
        # get spatial neighborhood
        nhood = spatial_weights.neighbors[node]
        if include_self:
            nhood = [node] + list(nhood)

    return nhood


def nhood_vals(nhood: Sequence[int], values: pd.Series, **kwargs) -> np.ndarray:
    """Get the values of objects in the neighboring nodes.

    Note:
        This function is designed to be used with the `gdf_apply` function.
        See the example.

    Parameters:
        nhood (Sequence[int]):
            A list or array of neighboring node uids.
        values (pd.Series):
            A value column-vector of shape (N, ).
        **kwargs (Dict[str, Any]):
            Additional keyword arguments. Not used.

    Returns:
        np.ndarray:
            The counts vector of the given values vector. Shape (n_classes, )

    Examples:
        Use `gdf_apply` to get the neighborhood values for each area of a cell
        >>> from functools import partial
        >>> from cellseg_gsontools.data import gland_cells
        >>> from cellseg_gsontools.graphs import fit_graph
        >>> from cellseg_gsontools.utils import set_uid
        >>> from cellseg_gsontools.apply import gdf_apply
        >>> from cellseg_gsontools.neighbors import neighborhood, nhood_vals
        >>> gc = gland_cells()
        >>> # To fit the delaunay graph, we need to set a unique id for each cell first
        >>> gc = set_uid(gc, id_col="uid")
        >>> w = fit_graph(gc, type="delaunay", thresh=100, id_col="uid")
        >>> # Get the neihgboring nodes of the graph
        >>> func = partial(neighborhood, spatial_weights=w)
        >>> gc["nhood"] = gdf_apply(gc, func, columns=["uid"])
        >>> # get the area values of the neighbors
        >>> func = partial(nhood_vals, values=gc.area.round(2))
        >>> gc["neighbor_areas"] = gdf_apply(
        ...     gc,
        ...     func=func,
        ...     parallel=True,
        ...     columns=["nhood"],
        ... )
        >>> gc["neighbor_areas"].head(5)
            uid
        0     [520.24, 565.58, 435.91, 302.26, 241.85, 418.02]
        1     [565.58, 520.24, 302.26, 318.15, 241.85, 485.71]
        2      [721.5, 435.91, 556.05, 466.96, 418.02, 678.35]
        3    [435.91, 520.24, 721.5, 302.26, 556.05, 655.42...
        4    [302.26, 520.24, 565.58, 435.91, 655.42, 485.7...
        Name: neighbor_areas, dtype: object
    """
    if isinstance(nhood, pd.Series):
        nhood = nhood.iloc[0]  # assume that the series is a row

    nhood_vals = np.array([0])
    if nhood not in (None, np.nan) and isinstance(nhood, (Sequence, np.ndarray)):
        nhood_vals = values.loc[nhood].to_numpy()

    return nhood_vals


def nhood_counts(
    nhood: Sequence[int], values: pd.Series, bins: Sequence, **kwargs
) -> np.ndarray:
    """Get the counts of objects that belong to bins/classes in the neighborhood.

    Note:
        This function is designed to be used with the `gdf_apply` function.
        See the example.

    Parameters:
        nhood (Sequence[int]):
            A list or array of neighboring node uids.
        values (pd.Series):
            A value column-vector of shape (N, ).
        bins (Sequence):
            The bins of any value vector. Shape (n_bins, ).
        return_vals (bool, optional):
            If True, also, the values the values are. Defaults to False.
        **kwargs (Dict[str, Any]):
            Additional keyword arguments. Not used.

    Returns:
        np.ndarray:
            The counts vector of the given values vector. Shape (n_classes, )

    Examples:
        Use `gdf_apply` to compute the neighborhood counts for each areal bin
        >>> import mapclassify
        >>> from functools import partial
        >>> from cellseg_gsontools.data import gland_cells
        >>> from cellseg_gsontools.graphs import fit_graph
        >>> from cellseg_gsontools.utils import set_uid
        >>> from cellseg_gsontools.apply import gdf_apply
        >>> from cellseg_gsontools.neighbors import neighborhood, nhood_vals, nhood_counts
        >>> gc = gland_cells()
        >>> # To fit the delaunay graph, we need to set a unique id for each cell first
        >>> gc = set_uid(gc, id_col="uid")
        >>> w = fit_graph(gc, type="delaunay", thresh=100, id_col="uid")
        >>> # Get the neihgboring nodes of the graph
        >>> func = partial(neighborhood, spatial_weights=w)
        >>> gc["nhood"] = gdf_apply(gc, func, columns=["uid"])
        >>> # get the area values of the neighbors
        >>> func = partial(nhood_vals, values=gc.area.round(2))
        >>> gc["neighbor_areas"] = gdf_apply(
        ...     gc,
        ...     func=func,
        ...     parallel=True,
        ...     columns=["nhood"],
        ... )
        >>> bins = mapclassify.Quantiles(gc.area, k=5)
        >>> func = partial(nhood_counts, values=gc.area, bins=bins.bins)
        >>> gc["area_bins"] = gdf_apply(
        ...     gc,
        ...     func,
        ...     columns=["nhood"],
        ... )
        >>> gc["area_bins"].head(5)
        uid
        0    [0, 2, 0, 3, 1]
        1    [0, 2, 1, 2, 1]
        2    [0, 0, 0, 3, 3]
        3    [0, 1, 0, 3, 3]
        4    [0, 1, 0, 3, 3]
        Name: area_bins, dtype: object
    """
    if isinstance(nhood, pd.Series):
        nhood = nhood.iloc[0]  # assume that the series is a row

    counts = np.array([0])
    if nhood not in (None, np.nan) and isinstance(nhood, (Sequence, np.ndarray)):
        nhood_vals = values.loc[nhood]

        if is_categorical(nhood_vals):
            counts = nhood_vals.value_counts().values
        else:
            sample_bins = mapclassify.UserDefined(nhood_vals, bins)
            counts = sample_bins.counts

    return counts


def nhood_type_count(
    cls_neighbors: Sequence, cls: Union[int, str], frac: bool = True, **kwargs
) -> float:
    """Get the number of nodes of a specific category in a neighborhood of a node.

    Note:
        This function is designed to be used with the `gdf_apply` function.
        See the example.

    Parameters:
        cls_neihbors (Sequence):
            A array/list (int or str) containing a category for each value in the data.
        cls (int or str):
            The specific category.
        frac (bool, optional):
            Flag, whether to return the fraction instead of the count. Defaults to True.
        **kwargs (Dict[str, Any])]):
            Additional keyword arguments. Not used.
    Returns:
        float:
            The count or fraction of a node of specific category in a neighborhood.

    Raises:
        TypeError:
            If the `cls_neighbors` is not categorical.

    Examples:
        Use `gdf_apply` to get the neighborhood fractions of immune cells for each node
        >>> from functools import partial
        >>> from cellseg_gsontools.data import gland_cells
        >>> from cellseg_gsontools.graphs import fit_graph
        >>> from cellseg_gsontools.utils import set_uid
        >>> from cellseg_gsontools.apply import gdf_apply
        >>> from cellseg_gsontools.neighbors import (
        ...     neighborhood,
        ...     nhood_vals,
        ...     hnood_type_count,
        ... )
        >>> gc = gland_cells()
        >>> # To fit the delaunay graph, we need to set a unique id for each cell first
        >>> gc = set_uid(gc, id_col="uid")
        >>> w = fit_graph(gc, type="delaunay", thresh=100, id_col="uid")
        >>> # Get the neihgboring nodes of the graph
        >>> func = partial(neighborhood, spatial_weights=w)
        >>> gc["nhood"] = gdf_apply(gc, func, columns=["uid"])
        >>> # get the classes of the neighbors
        >>> func = partial(nhood_vals, values=gc.class_name)
        >>> gc["neighbor_classes"] = gdf_apply(
        ...     gc,
        ...     func=func,
        ...     parallel=True,
        ...     columns=["nhood"],
        ... )
        >>> func = partial(nhood_type_count, cls="inflammatory", frac=True)
        >>> gc["n_immune_neighbors"] = gdf_apply(
        ...    gc,
        ...    func=func,
        ...    parallel=True,
        ...    columns=["neighbor_classes"],
        >>> )
        >>> gc[gc["n_immune_neighbors"] > 0]["n_immune_neighbors"].head(5)
        uid
        39    0.333333
        40    0.111111
        42    0.166667
        44    0.375000
        48    0.166667
        Name: n_immune_neighbors, dtype: float64
    """
    if isinstance(cls_neighbors, pd.Series):
        cls_neighbors = cls_neighbors.iloc[0]  # assume that the series is a row

    ret = 0
    if isinstance(cls_neighbors, (Sequence, np.ndarray)):
        if len(cls_neighbors) > 0:
            if not isinstance(cls_neighbors[0], (int, str)):
                raise TypeError("cls_neighbors must contain int of str values.")

        t, c = np.unique(cls_neighbors, return_counts=True)

        ret = 0.0
        if cls in t:
            ix = np.where(t == cls)
            ret = c[ix][0]
            if frac:
                ret = ret / np.sum(c)

    return ret


def nhood_dists(
    nhood: Sequence[int],
    centroids: pd.Series,
    ids: pd.Series = None,
    invert: bool = False,
) -> np.ndarray:
    """Compute the neighborhood distances between the center node.

    Note:
        This function is designed to be used with the `gdf_apply` function.
        See the example.

    Note:
        It is assumed that the center node is the first index in the `nhood`
        array. Use `include_self=True` in `neighborhood` to include the center.

    Parameters:
        nhood (Sequence[int]):
            An array containing neighbor indices. The first index is assumed to be
            the center node.
        centroids (pd.Series):
            A pd.Series array containing the centroid Points of the full gdf.
        ids (pd.Series):
            A pd.Series array containing the ids of the full gdf.
        invert (bool):
            Flag, whether to invert the distances. E.g. 1/dists. Defaults to False.

    Returns:
        np.ndarray:
            An array containing the distances between the center node and its
            neighborhood.

    Examples:
        Use `gdf_apply` to extract the neighboring node distances for each node/cell
        >>> from functools import partial
        >>> from cellseg_gsontools.data import gland_cells
        >>> from cellseg_gsontools.graphs import fit_graph
        >>> from cellseg_gsontools.utils import set_uid
        >>> from cellseg_gsontools.apply import gdf_apply
        >>> from cellseg_gsontools.neighbors import neighborhood, nhood_dists
        >>> gc = gland_cells()
        >>> # To fit the delaunay graph, we need to set a unique id for each cell first
        >>> gc = set_uid(gc, id_col="uid")
        >>> w = fit_graph(gc, type="delaunay", thresh=100, id_col="uid")
        >>> # Get the neihgboring nodes of the graph
        >>> func = partial(neighborhood, spatial_weights=w)
        >>> gc["nhood"] = gdf_apply(gc, func, columns=["uid"])
        >>> func = partial(nhood_dists, centroids=gc.centroid)
        >>> gc["nhood_dists"] = gdf_apply(
        ...     gc,
        ...     func,
        ...     columns=["nhood"],
        ... )
        >>> gc["nhood_dists"].head(5)
        uid
        0        [0.0, 26.675, 24.786, 30.068, 30.228, 41.284]
        1        [0.0, 26.675, 23.428, 42.962, 39.039, 23.949]
        2        [0.0, 25.577, 39.348, 46.097, 34.309, 29.478]
        3    [0.0, 24.786, 25.577, 39.574, 37.829, 47.16, 3...
        4    [0.0, 30.068, 23.428, 39.574, 29.225, 36.337, ...
        Name: nhood_dists, dtype: object
    """
    if isinstance(nhood, pd.Series):
        nhood = nhood.iloc[0]  # assume that the series is a row

    nhood_dists = np.array([0])
    if nhood not in (None, np.nan) and isinstance(nhood, (Sequence, np.ndarray)):
        if ids is not None:
            nhood = ids[ids.isin(nhood)].index

        node = nhood[0]
        center_node = centroids.loc[node]
        nhood_nodes = centroids.loc[nhood].to_numpy()
        nhood_dists = np.array(
            [np.round(_dist(center_node, c), 3) for c in nhood_nodes]
        )
        if invert:
            nhood_dists = np.round(np.reciprocal(nhood_dists, where=nhood_dists > 0), 3)

    return nhood_dists
