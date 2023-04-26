from typing import Dict, Optional, Sequence, Tuple

import geopandas as gpd
import numpy as np
import shapely
from libpysal.weights import (
    KNN,
    Delaunay,
    DistanceBand,
    Gabriel,
    Kernel,
    Relative_Neighborhood,
    Voronoi,
    W,
)

from .apply import gdf_apply
from .neighbors import neighborhood, nhood_dists
from .utils import set_uid

__all__ = [
    "fit_graph",
    "dist_thresh_weights",
    "_drop_neighbors",
    "graph_pos",
    "neighborhood_density",
]


def fit_graph(
    gdf: gpd.GeoDataFrame,
    type: str,
    id_col: Optional[str] = None,
    thresh: Optional[float] = None,
    **kwargs,
) -> W:
    """Fit a libpysal spatial weights graph to a gdf.

    Optionally, a distance threshold can be set for edges that are too long.

    Basically, a wrapper to fit libpysal graph with additional distance threshing.

    Parameters
    ----------
        gdf : gpd.GeoDataFrame
            The input geodataframe.
        type : str
            The type of the libpysal graph. Allowed: "delaunay", "gabriel", "knn",
            "distband", "voronoi", "relative_nhood", "kernel"
        id_col : str, optional
            The unique id column in the gdf. If None, this uses `set_uid` to set it.
        thresh : float, optional
            A distance threshold for too long edges.
        **kwargs:
            Arbitrary keyword arguments for the Graph init functions.

    Returns
    -------
        libpysal.weights.W:
            A libpysal spatial weights object, containing the neighbor graph data.

    Examples
    --------
    Fit a DistanceBand to a gdf with a dist threshold of 120.0.
    >>> from cellseg_gsontools.graphs import fit_graph
    >>> w = fit_graph(gdf, type="distband", thresh=120)

    Fit a kernel graph to a gdf with k=5, and a triangular non-fixed kernel.
    >>> from cellseg_gsontools.graphs import fit_graph
    >>> w = fit_graph(
            gdf, type="kernel", thresh=None, k=5, fixed=False, function="triangular"
        )

    Fit a delaunay graph to a gdf without a dist threshold.
    >>> from cellseg_gsontools.graphs import fit_graph
    >>> w = fit_graph(gdf, type="delaunay", thresh=None)
    """

    def shift_indices(w):
        neighbors = {k + 1: [n + 1 for n in ngh] for k, ngh in w.neighbors.items()}
        weights = {k + 1: it for k, it in w.weights.items()}

        return W(neighbors, weights)

    allowed = (
        "delaunay",
        "gabriel",
        "knn",
        "distband",
        "voronoi",
        "relative_nhood",
        "kernel",
    )
    if type not in allowed:
        raise ValueError(f"Illegal graph type given. Got: {type}. Allowed: {allowed}.")

    if id_col is None:
        id_col = "uid"
        gdf = set_uid(gdf, drop=False)

    if type == "delaunay":
        w = Delaunay.from_dataframe(gdf.centroid, silence_warnings=True, **kwargs)
        if id_col == "uid":
            w = shift_indices(
                w
            )  # delaunay does not work with ids that are manually set
    elif type == "gabriel":
        w = Gabriel.from_dataframe(gdf.centroid, silence_warnings=True, **kwargs)
        if id_col == "uid":
            w = shift_indices(w)  # gabriel does not work with ids that are manually set
    elif type == "knn":
        w = KNN.from_dataframe(gdf, silence_warnings=True, **kwargs)
    elif type == "distband":
        if thresh is None:
            raise ValueError("DistBand requires `thresh` param. Not provided.")

        w = DistanceBand.from_dataframe(
            gdf, threshold=thresh, alpha=-1.0, silence_warnings=True, **kwargs
        )
    elif type == "voronoi":
        w = Voronoi(
            np.array(list(zip(gdf.centroid.x, gdf.centroid.y))),
            silence_warnings=True,
            **kwargs,
        )
        if id_col == "uid":
            w = shift_indices(w)  # voronoi does not work with ids that are manually set
    elif type == "relative_nhood":
        w = Relative_Neighborhood.from_dataframe(
            gdf.centroid, silence_warnings=True, **kwargs
        )
        if id_col == "uid":
            w = shift_indices(
                w
            )  # relative_nhood does not work with ids that are manually set
    elif type == "kernel":
        w = Kernel.from_dataframe(
            gdf, silence_warnings=True, **kwargs
        )  # includes self-loop in neighbors

    # Threshold edges based on distance to center node.
    if thresh is not None and type != "distband":
        w = dist_thresh_weights(gdf, w, thresh, id_col=id_col)

    return w


def dist_thresh_weights(
    gdf: gpd.GeoDataFrame,
    w: W,
    thresh: float = 120.0,
    id_col: str = "uid",
    includes_self: bool = True,
) -> W:
    """Drop edges from the spatial weights graph that are longer than `thresh`.

    Parameters
    ----------
        gdf : gpd.GeoDataFrame
            Input geodataframe.
        w : libpysal.weights.W
            Spatial weights object. Needs to be created from the `gdf`.
        thresh : float, default=120.0
            Distance threshold value.
        id_col : str, default="uid"
            Column in the `gdf` indicating the unique id.
        includes_self : bool, default=False
            Sometimes the weights object includes a self-loop in the neighbors.
            I.e. {0: [0, 1, 2]} (weights.Kernel). This is a flag, whether that is
            the case. This needs to be considered when using `neighborhood`.

    Returns
    -------
        libpysal.weights.W:
            A new libpysal spatial weights object with long edges dropped.

    Example
    -------
    Create a Delaunay graph and drop too long edges away.
    >>> from libpysal.weights import Delaunay
    >>> from cellseg_gsontools.graphs import dist_thresh_weights

    >>> id_col = "iid"
    >>> gdf[id_col] = range(len(gdf))
    >>> gdf = gdf.set_index(id_col, drop=False)
    >>> ids = list(gdf.index.values)
    >>> w = Delaunay.from_dataframe(gdf.centroid, id_order=ids, ids=ids)

    >>> # drop the edges
    >>> w = dist_thresh_weights(gdf, w, thresh=120.0)
    >>> ax = gdf.plot(edgecolor='grey', facecolor='w', figsize=(25, 25))
    >>> f, ax = w.plot(
            gdf,
            ax=ax,
            edge_kws=dict(color='r', linestyle=':', linewidth=1),
            node_kws=dict(marker='')
        )
    >>> ax.set_axis_off()
    """
    gdf["nhood"] = gdf_apply(
        gdf, neighborhood, col=id_col, spatial_weights=w, include_self=includes_self
    )

    gdf["nhood_dists"] = gdf_apply(
        gdf,
        nhood_dists,
        col="nhood",
        centroids=gdf.centroid,
    )

    gdf["new_neighbors"] = gdf_apply(
        gdf, _drop_neighbors, col="nhood", extra_col="nhood_dists", thresh=thresh
    )

    return W(gdf["new_neighbors"].to_dict())


def _drop_neighbors(
    nhood: Sequence[int], nhood_dists: Sequence[float], thresh: float
) -> np.ndarray:
    """Drop neihgbors that are further than `thresh` from the center node.

    NOTE: center node here is `nhood[0]`.
    This function should be used with `gdf_apply`.

    Parameters
    ----------
        nhood : Sequence[int]
            A list or array of neighboring node uids.
        nhood_dists : Sequence[float]
            A list or array of distances to neighboring nodes.

    Returns
    -------
        np.ndarray:
            A new set of neighbor where some of the neighbors are dropped.
    """
    if len(nhood) != len(nhood_dists):
        raise ValueError(
            f"Both inputs should be the same length. Got: `len(nhood) = {len(nhood)}`"
            f"and `len(nhood_dists) = {len(nhood_dists)}`."
        )

    nhood = np.array(nhood)
    nhood_dists = np.array(nhood_dists)

    # drop below thresh
    nhood = nhood[nhood_dists < thresh]
    # drop self loop
    center = nhood[0]
    nhood = nhood[nhood != center]
    return nhood


def graph_pos(gdf: gpd.GeoDataFrame) -> Dict[int, Tuple[int, int]]:
    """Create pos object from gdf for graph computations.

    Parameters
    ----------
        gdf : gpd.GeoDataFrame
            Input geodataframe.

    Returns
    -------
        Dict[int, Tuple[int, int]]:
            Dict with cell coordinates.
    """
    coords = [(c.x, c.y) for c in gdf["geometry"]]
    pos = [i for i in zip(gdf.index, coords)]
    return dict(pos)


def neighborhood_density(gdf: gpd.GeoDataFrame, k: int = 4) -> gpd.GeoDataFrame:
    """Calculate neighborhood density for points with sum of distances to KNN.

    Parameters
    ----------
        gdf : gpd.GeoDataFrame
            Input geodataframe.
        k : int, default=4
            Number of neirest neighbours in KNN.

    Returns
    -------
        gpd.GeoDataFrame:
            Input gdf with a weights column added.
    """
    G = KNN.from_dataframe(gdf, k=k)

    weights = []
    pos = graph_pos(gdf)

    for i in gdf.index:
        weight = 0

        for edge in G.neighbors[i]:
            cell = shapely.ops.Point(pos[i])
            neighbor = shapely.ops.Point(pos[edge])
            weight += cell.distance(neighbor)

        weights.append(weight)

    gdf["weights"] = weights

    return gdf
