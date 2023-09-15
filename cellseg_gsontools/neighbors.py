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
    node: int,
    spatial_weights: W,
    include_self: bool = True,
    ret_n_neighbors: bool = False,
) -> Union[List[int], int]:
    """Get immediate neighborhood of a node given the spatial weights obj.

    NOTE: The neighborhood contains the given node itself.

    Parameters
    ----------
        node : int
            Input node uid.
        spatial_weights : libysal.weights.W
            Libpysal spatial weights object.
        include_self : bool, default=True
            Flag, whether to include the node itself in the neighborhood.
        ret_n_neighbors : bool, default=False
            If True, instead of returnig a sequence of the neigbor node uids
            returns just the number of neighbors.

    Returns
    -------
        List[int] or int:
            A list of the neighboring node uids. E.g. [1, 4, 19].
            or the number of neighbors if `ret_n_neighbors=True.`

    Examples
    --------
    Use `gdf_apply` to extract the neighboring nodes for each node/cell
    >>> from cellseg_gsontools.apply import gdf_apply
    >>> from cellseg_gsontools.neighbors import neighborhood
    >>> from cellseg_gsontools.utils import set_uid

    >>> # Set uid to the gdf
    >>> data = set_uid(gdf)

    >>> # Get spatial weights
    >>> w_dist = DistanceBand.from_dataframe(gdf, threshold=55.0, alpha=-1.0)

    >>> # Get the neihgboring nodes of the graph
    >>> gdf_apply(data, neighborhood, col="uid", spatial_weights=w_dist)
    0                            [0]
    1                     [1, 9, 19]
    2                            [2]
    3                      [3, 4, 6]
    4            [4, 3, 5, 6, 8, 14]
                    ...
    361              [360, 336, 338]
    362    [361, 331, 348, 350, 363]
    363         [362, 339, 345, 365]
    364    [363, 331, 348, 350, 361]
    365    [364, 340, 341, 349, 352]
    Name: uid, Length: 365, dtype: object

    """
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

    Parameters
    ----------
        nhood : Sequence[int]
            A list or array of neighboring node uids.
        values : pd.Series
            A value column-vector of shape (N, ).

    Returns
    -------
        np.ndarray:
            The counts vector of the given values vector. Shape (n_classes, )

    Examples
    --------
    Use `gdf_apply` to get the neihgborhood values for each node/cell given a metric
    >>> from libpysal.weights import DistanceBand
    >>> from cellseg_gsontools.apply import gdf_apply
    >>> from cellseg_gsontools.neighbors import neighborhood, nhood_vals
    >>> from cellseg_gsontools.utils import set_uid

    >>> # Set uid to the gdf
    >>> data = set_uid(gdf)

    >>> # Get spatial weights
    >>> w_dist = DistanceBand.from_dataframe(gdf, threshold=55.0, alpha=-1.0)

    >>> # Get the neihgboring nodes of the graph
    >>> data["nhood"] = gdf_apply(
            data, neighborhood, col="uid", spatial_weights=w_dist
        )

    >>> # Define the gdf column of interest
    >>> val_col = "eccentricity"
    >>> values = data.set_index("uid")[val_col]

    >>> # get the neighborhood metric values
    >>> gdf_apply(
            data,
            nhood_vals,
            col="nhood",
            values=values,
        )

    0                                   [0.42]
    1                       [0.92, 0.83, 0.68]
    2                                    [0.8]
    3                        [0.81, 0.4, 0.74]
    4      [0.4, 0.81, 0.59, 0.74, 0.46, 0.44]
                        ...
    361                      [0.53, 0.82, 0.5]
    362         [0.26, 0.31, 0.93, 0.58, 0.29]
    363                 [0.7, 0.69, 0.5, 0.36]
    364         [0.29, 0.31, 0.93, 0.58, 0.26]
    365         [0.25, 0.28, 0.44, 0.59, 0.42]
    """
    nhood_vals = np.array([0])
    if nhood not in (None, np.nan):
        nhood_vals = values.loc[nhood].to_numpy()

    return nhood_vals


def nhood_counts(
    nhood: Sequence[int], values: pd.Series, bins: Sequence, **kwargs
) -> np.ndarray:
    """Get the counts of objects that belong to bins/classes in the neighborhood.

    Parameters
    ----------
        nhood : Sequence[int]
            A list or array of neighboring node uids.
        values : pd.Series
            A value column-vector of shape (N, ).
        bins : Sequence
            The bins of any value vector. Shape (n_bins, ).
        return_vals : bool, default=False
            If True, also, the values the values are

    Returns
    -------
        np.ndarray:
            The counts vector of the given values vector. Shape (n_classes, )

    Examples
    --------
    Use `gdf_apply` to compute the neihgborhood counts for each node/cell given a metric
    >>> from cellseg_gsontools.apply import gdf_apply
    >>> from cellseg_gsontools.neighbors import neighborhood, nhood_counts
    >>> from cellseg_gsontools.utils import set_uid

    >>> # Set uid to the gdf
    >>> data = set_uid(gdf)

    >>> # Get spatial weights
    >>> w_dist = DistanceBand.from_dataframe(gdf, threshold=55.0, alpha=-1.0)

    >>> # Get the neihgboring nodes of the graph
    >>> data["nhood"] = gdf_apply(
            data, neighborhood, col="uid", spatial_weights=w_dist
        )

    >>> # Define the gdf column that will be binned
    >>> val_col = "eccentricity"
    >>> values = data.set_index("uid")[val_col]

    >>> # compute the counts of the bins inside the neighborhood
    >>> gdf_apply(
            data,
            nhood_counts,
            col="nhood",
            values=values,
            bins=bins,
        )

    0      [1, 0, 0, 0, 0, 0, 0]
    1      [0, 1, 1, 1, 0, 0, 0]
    2      [0, 0, 1, 0, 0, 0, 0]
    3      [1, 1, 1, 0, 0, 0, 0]
    4      [4, 1, 1, 0, 0, 0, 0]
                ...
    361    [2, 0, 1, 0, 0, 0, 0]
    362    [4, 0, 0, 1, 0, 0, 0]
    363    [2, 2, 0, 0, 0, 0, 0]
    364    [4, 0, 0, 1, 0, 0, 0]
    365    [5, 0, 0, 0, 0, 0, 0]
    Name: nhood, Length: 365, dtype: object
    """
    counts = np.array([0])
    if nhood not in (None, np.nan):
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

    Parameters
    ----------
        cls_neihbors : Sequence
            A array/list (int or str) containing a category for each value in the data.
        cls : int or str
            The specific category
        frac : bool, default=True
            Flag, whether to return the fraction instead of the count

    Returns
    -------
        float:
            The count or fraction of a node of specific category in a neighborhood.

    Raises
    ------
        TypeError: If the `cls_neighbors` is not cadtegorical.

    Example
    -------
        Use `gdf_apply` to get the neihgborhood fractions of immune cells for each node
        >>> from cellseg_gsontools.apply import gdf_apply
        >>> from cellseg_gsontools.neighbors import neighborhood, nhood_vals
        >>> from cellseg_gsontools.utils import set_uid

        >>> # Set uid to the gdf
        >>> data = set_uid(gdf)

        >>> # Get spatial weights
        >>> w_dist = DistanceBand.from_dataframe(gdf, threshold=55.0, alpha=-1.0)

        >>> # Get the neihgboring nodes of the graph
        >>> data["nhood"] = gdf_apply(
                data, neighborhood, col="uid", spatial_weights=w_dist
            )

        >>> # Define the class name column
        >>> val_col = "class_name"
        >>> values = data.set_index("uid")[val_col]

        >>> # get the neighborhood classes
        >>> data[f"{val_col}_nhood_vals"] = gdf_apply(
                data,
                nhood_vals,
                col="nhood",
                values=values,
            )

        >>> data["local_infiltration_fraction"] = gdf_apply(
                data,
                nhood_type_count,
                col=f"{val_col}_nhood_vals",
                cls="inflammatory",
                parallel=True
            ).head(14)
        uid
        1     0.000000
        2     0.500000
        3     0.000000
        4     0.250000
        5     0.000000
        6     1.000000
        7     0.000000
        8     0.250000
        9     0.333333
        10    0.333333
        11    0.000000
        12    0.250000
        13    0.333333
        14    0.000000
        Name: class_name_nhood_vals, dtype: float64
    """
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
    nhood: Sequence[int], centroids: pd.Series, ids: pd.Series = None
) -> np.ndarray:
    """Compute the neihborhood distances between the center node.

    NOTE: It is assumed that the center node is the first index in the `nhood`
    array. To get the neighborhoods use for example:
    `gdf_apply(d, neighborhood, col="uid", spatial_weights=w_dist, include_self=True)`

    Parameters
    ----------
        nhood : Sequence[int]
            An array containig neighbor indices. The first index is assumed to be
            the center node.
        centroids : pd.Series
            A pd.Series array containing the centroid Points of the full gdf.
        ids : pd.Series, optional
            A pd.Series array containing the ids of the full gdf.

    Returns
    -------
        np.ndarray:
            An array containing the distances between the center node and it's nhood.

    Example
    -------
    >>> from libpysal.weights import Delaney
    >>> from cellseg_gsontools.apply import gdf_apply
    >>> from cellseg_gsontools.neighbors import neighborhood

    >>> id_col = "iid"
    >>> gdf[id_col] = range(len(gdf))
    >>> gdf = gdf.set_index(id_col, drop=False)
    >>> w = Delaunay.from_dataframe(gdf.centroid)

    >>> gdf["nhood"] = gdf_apply(gdf, neighborhood, col="iid", spatial_weights=w)

    >>> gdf["nhood_dists"] = gdf_apply(
            gdf,
            nhood_dists,
            col="nhood",
            centroids=gdf.centroid,
        )
    """
    nhood_dists = np.array([0])
    if nhood not in (None, np.nan):
        if ids is not None:
            nhood = ids[ids.isin(nhood)].index

        node = nhood[0]
        center_node = centroids.loc[node]
        nhood_nodes = centroids.loc[nhood].to_numpy()
        nhood_dists = np.array(
            [np.round(_dist(center_node, c), 2) for c in nhood_nodes]
        )

    return nhood_dists
