from typing import List, Sequence, Union

import mapclassify
import numpy as np
import pandas as pd
from libpysal.weights import W

__all__ = ["neighborhood", "nhood_counts", "nhood_vals"]


def neighborhood(
    node: int, spatial_weights: W, ret_n_neighbors: bool = False
) -> Union[List[int], int]:
    """Get immediate neighborhood of a node given the spatial weights obj.

    NOTE: The neighborhood contains the given node itself.

    Parameters
    ----------
        node : int
            Input node uid.
        spatial_weights : libysal.weights.W
            Libpysal spatial weights object.
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
        nhood = [node] + spatial_weights.neighbors[node]

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

        >>> # Define the gdf column that will be binned
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
    nhood: Sequence[int],
    values: pd.Series,
    bins: Sequence,
    categorical: bool = False,
    **kwargs
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
        categorical : bool, default=False
            A flag to signal, whether the value vector values are categorical.
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
        >>> categorical = False # the column-values are not categorical
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

        if categorical:
            counts = nhood_vals.value_counts().values
        else:
            sample_bins = mapclassify.UserDefined(nhood_vals, bins)
            counts = sample_bins.counts

    return counts
