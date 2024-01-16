import warnings
from typing import Optional, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
from libpysal.weights import KNN, Delaunay, DistanceBand, Relative_Neighborhood, W

__all__ = [
    "fit_graph",
    "dist_thresh_weights_sequential",
    "get_border_crosser_links",
    "_drop_neighbors",
]


def fit_graph(
    gdf: gpd.GeoDataFrame,
    type: str,
    id_col: Optional[str] = None,
    thresh: Optional[float] = None,
    silence_warnings: bool = True,
    **kwargs,
) -> W:
    """Fit a `libpysal` spatial weights graph to a gdf.

    Optionally, a distance threshold can be set for edges that are too long.

    This is a wrapper to fit `libpysal` graph with additional distance threshing.

    Note:
        Allowed graph fitting methods:

        - `delaunay`
        - `knn`
        - `distband`
        - `relative_nhood`

    Parameters:
        gdf (gpd.GeoDataFrame):
            The input geodataframe.
        type (str):
            The type of the libpysal graph. Allowed: "delaunay", "knn", "distband",
            "relative_nhood"
        id_col (str):
            The unique id column in the gdf. If None, this uses `set_uid` to set it.
        thresh (float):
            A distance threshold for too long edges.
        silence_warnings (bool):
            Flag to silence the warnings.
        **kwargs (Dict[str, Any]):
            Arbitrary keyword arguments for the Graph init functions.

    Returns:
        libpysal.weights.W or None:
            A libpysal spatial weights object, containing the neighbor graph data.
            Returns None if the input gdf is empty.

    Examples:
        Fit a DistanceBand to a gdf with a dist threshold of 120.0.
        >>> from cellseg_gsontools.graphs import fit_graph
        >>> w = fit_graph(gdf, type="distband", thresh=120)

        Fit a delaunay graph to a gdf without a dist threshold.
        >>> from cellseg_gsontools.graphs import fit_graph
        >>> w = fit_graph(gdf, type="delaunay", thresh=None)
    """
    allowed = (
        "delaunay",
        "knn",
        "distband",
        "relative_nhood",
    )
    if type not in allowed:
        raise ValueError(f"Illegal graph type given. Got: {type}. Allowed: {allowed}.")

    if gdf is None or gdf.empty:
        return

    # warn if id_col is not provided
    if not silence_warnings:
        _graph_warn(type, id_col)

    # can't fit delaunay or relative nhood graphs with less than 4 points
    if type in ("delaunay", "relative_nhood"):
        if len(gdf) < 4:
            return
    if type == "delaunay":
        # NOTE: neighbor keys start from 0
        w = Delaunay.from_dataframe(
            gdf.centroid,
            silence_warnings=True,
            use_index=True,
            ids=gdf[id_col],
            **kwargs,
        )
    elif type == "relative_nhood":
        # NOTE: neighbor indices start from 0
        w = Relative_Neighborhood.from_dataframe(
            gdf.centroid, silence_warnings=True, **kwargs
        )
    elif type == "knn":
        if "ids" in list(kwargs.keys()):  # drop ids kwarg since it fails
            kwargs.pop("ids")

        # NOTE: neighbor indices equal gdf[`ìd_col`]
        w = KNN.from_dataframe(gdf, silence_warnings=True, ids=id_col, **kwargs)
    elif type == "distband":
        if thresh is None:
            raise ValueError("DistBand requires `thresh` param. Not provided.")

        if "ids" in list(kwargs.keys()):  # drop ids kwarg since it fails
            kwargs.pop("ids")

        # NOTE: neighbor indices equal gdf[`ìd_col`]
        w = DistanceBand.from_dataframe(
            gdf,
            threshold=thresh,
            alpha=-1.0,
            ids=id_col,
            silence_warnings=True,
            **kwargs,
        )

    # convert graph indices to global ids
    if type in ("delaunay", "relative_nhood"):
        w = _graph_to_index(w, gdf)
        if id_col is not None:
            w = _graph_to_global_ids(w, gdf, id_col)

    # # Threshold edges based on distance to center node.
    if thresh is not None and type != "distband":
        w = dist_thresh_weights_sequential(gdf, w, thresh, id_col=id_col)

    return w


def dist_thresh_weights_sequential(
    gdf: gpd.GeoDataFrame,
    w: W,
    thresh: float,
    id_col: Optional[str] = None,
    include_self: bool = False,
) -> W:
    """Threshold edges based on distance to center node.

    Parameters:
        gdf (gpd.GeoDataFrame):
            The input geodataframe.
        w (libpysal.weights.W):
            The input spatial weights object.
        thresh (float):
            The distance threshold.
        id_col (str, optional):
            The unique id column in the gdf. If None, this uses `set_uid` to set it.
        include_self (bool, default=True):
            Whether to include self-loops in the neighbors.

    Returns:
        libpysal.weights.W:
            A libpysal spatial weights object, containing the neighbor graph data.
    """
    gdf = gdf.copy()

    # drop duplicate rows
    gdf = gdf.drop_duplicates(subset=[id_col], keep="first")
    gdf["nhood"] = pd.Series(list(w.neighbors.values()), index=gdf.index)

    new_neighbors = []
    for _, row in gdf.iterrows():
        neighbor_rows = gdf[gdf.loc[:, id_col].isin(row.nhood)]
        nhood_dist = [
            np.round(row.geometry.centroid.distance(ngh), 2)
            for ngh in neighbor_rows.geometry.centroid
        ]

        new_neighbors.append(
            _drop_neighbors(row.nhood, nhood_dist, thresh, include_self=include_self)
        )

    gdf["new_neighbors"] = new_neighbors
    return W(dict(zip(gdf[id_col], gdf["new_neighbors"])), silence_warnings=True)


def get_border_crosser_links(
    union_weights: W,
    roi_weights: W,
    iface_weights: W,
    only_border_crossers: bool = True,
) -> W:
    """Get the links that cross the border between the ROI and interface cells.

    Parameters:
        union_weights (W):
            The union of the ROI and interface weights. NOTE: contains links between ROI
            & interface cells.
        roi_weights (W):
            The ROI weights. NOTE: contains only links between ROI cells.
        iface_weights (W):
            The interface weights. NOTE: contains only links between interface cells.
        only_border_crossers (bool, optional):
            Whether to return only the links that cross the border between the ROI and
            interface cells or all neighbors of the node that has a border crossing link.
            This includes also the liks that do not cross the border. By default True.

    Returns:
        W:
            The links that cross the border between the ROI and interface cells.
    """
    # loop the nodes in the union graph and check if they are border crossing links
    graph = {}
    for node, neigh in union_weights.neighbors.items():
        is_roi_node = node in roi_weights.neighbors.keys()
        is_iface_node = node in iface_weights.neighbors.keys()

        neighbors = []
        for n in neigh:
            is_roi_neigh = n in roi_weights.neighbors.keys()
            is_iface_neigh = n in iface_weights.neighbors.keys()
            if (is_roi_node and is_iface_neigh) or (is_iface_node and is_roi_neigh):
                # return all neighbors if requested
                if not only_border_crossers:
                    neighbors.extend(neigh)
                    break
                else:
                    neighbors.append(n)

        graph[node] = neighbors

    return W(graph, silence_warnings=True)


def _drop_neighbors(
    nhood: Sequence[int],
    nhood_dist: Sequence[float],
    thresh: float,
    include_self: bool = False,
) -> np.ndarray:
    """Drop neighbors based on distance to center node."""
    if len(nhood) != len(nhood_dist):
        raise ValueError(
            f"Both inputs should be the same length. Got: `len(nhood) = {len(nhood)}`"
            f"and `len(nhood_dist) = {len(nhood_dist)}`."
        )

    nhood = np.array(nhood)
    nhood_dist = np.array(nhood_dist)

    # drop below thresh
    nhood = nhood[nhood_dist < thresh]

    # drop self loop
    if include_self:
        center = nhood[0]
        nhood = nhood[nhood != center]

    return nhood


def _graph_to_global_ids(w: W, in_gdf: gpd.GeoDataFrame, id_col: str) -> W:
    """Convert the graph ids to global ids i.e. to the ids of the id_col``."""
    new_neighbors = {}

    for node, neighbors in w.neighbors.items():
        new_id = in_gdf.loc[node, id_col]
        nghs = [in_gdf.loc[ngh, id_col] for ngh in neighbors]
        new_neighbors[new_id] = nghs

    return W(new_neighbors, id_order=sorted(new_neighbors.keys()))


def _graph_to_index(w: W, in_gdf: gpd.GeoDataFrame) -> W:
    new_neighbors = {}

    for node, neighbors in w.neighbors.items():
        new_id = in_gdf.iloc[node].name
        nghs = [in_gdf.iloc[ngh].name for ngh in neighbors]
        new_neighbors[new_id] = nghs

    return W(new_neighbors, id_order=sorted(new_neighbors.keys()))


def _graph_warn(type: str, id_col: str):
    if type in ("delaunay", "relative_nhood"):
        if id_col is None:
            warnings.warn(
                f"For graphs of type: {type}, if `id_col` is not provided "
                "The neighbors object will have keys starting from 0. "
                "These keys are likely not the same as the ids in the gdf."
            )
    elif type in ("knn", "distband"):
        if id_col is None:
            warnings.warn(
                f"For graphs of type: {type}, if `id_col` is not provided "
                "The neighbors object will have keys equalling the gdf.index. "
            )
