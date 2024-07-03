from functools import partial
from typing import List, Tuple

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from libpysal.weights import W, lag_spatial, w_subset
from sklearn.cluster import DBSCAN, HDBSCAN, OPTICS

from cellseg_gsontools.apply import gdf_apply
from cellseg_gsontools.graphs import fit_graph
from cellseg_gsontools.neighbors import neighborhood, nhood_type_count, nhood_vals

__all__ = [
    "cluster_points",
    "find_lisa_clusters",
    "cluster_cells",
    "get_connected_components",
    "label_connected_components",
    "moran_hot_cold_spots",
]


def cluster_points(
    gdf: gpd.GeoDataFrame,
    eps: float = 350.0,
    min_samples: int = 30,
    method: str = "dbscan",
    n_jobs: int = -1,
    **kwargs,
) -> gpd.GeoDataFrame:
    """Apply a clustering to centroids in a gdf.

    This is just a quick wrapper for a few clustering algos adapted
    to geodataframes.

    Note:
        Allowed clustering methods are:

        - `dbscan` (sklearn.cluster.DBSCAN)
        - `hdbscan` (sklearn.cluster.HDBSCAN)
        - `optics` (sklearn.cluster.OPTICS)
        - `adbscan` (esda.adbscan.ADBSCAN)

    Parameters:
        gdf (gpd.GeoDataFrame):
            Input geo dataframe with a properly set geometry column.
        eps (float):
            The maximum distance between two samples for one to be considered as in the
            neighborhood of the other. This is not a maximum bound on the distances of
            gdf within a cluster.
        min_samples (int):
            The number of samples (or total weight) in a neighborhood for a point to be
            considered as a core point. This includes the point itself.
        method (str):
            The clustering method to be used. Allowed: ("dbscan", "adbscan", "optics").
        n_jobs (int):
            The number of parallel jobs to run. None means 1. -1 means using all
            processors.
        **kwargs (Dict[str, Any]):
            Arbitrary key-word arguments passed to the clustering methods.

    Raises:
        ValueError:
            If illegal method is given or input `gdf` is of wrong type.

    Returns:
        gpd.GeoDataFrame:
            The input gdf with a new "labels" columns of the clusters.

    Examples:
        Cluster immune cell centroids in a gdf using dbscan.

        >>> from cellseg_gsontools.clustering import cluster_points
        >>> gdf = read_gdf("cells.json")
        >>> gdf = cluster_points(
        ...     gdf[gdf["class_name"] == "immune"],
        ...     method="dbscan",
        ...     eps=350.0,
        ...     min_samples=30,
        ... )
    """
    allowed = ("dbscan", "adbscan", "optics", "hdbscan")
    if method not in allowed:
        raise ValueError(
            f"Illegal clustering method was given. Got: {method}, allowed: {allowed}"
        )

    if isinstance(gdf, gpd.GeoDataFrame):
        xy = np.vstack([gdf.centroid.x, gdf.centroid.y]).T
    else:
        raise ValueError(
            "The input `gdf` needs to be a gpd.GeoDataFrame with geometry col "
            f"Got: {type(gdf)}"
        )

    if method == "adbscan":
        try:
            from esda.adbscan import ADBSCAN
        except ImportError:
            raise ImportError(
                "The adbscan method requires the esda package to be installed."
                "Install it with: pip install esda"
            )
        xy = pd.DataFrame({"X": xy[:, 0], "Y": xy[:, 1]})
        clusterer = ADBSCAN(eps=eps, min_samples=min_samples, n_jobs=n_jobs, **kwargs)
    elif method == "dbscan":
        clusterer = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=n_jobs, **kwargs)
    elif method == "hdbscan":
        clusterer = HDBSCAN(min_samples=min_samples, n_jobs=n_jobs, **kwargs)
    elif method == "optics":
        clusterer = OPTICS(
            max_eps=eps, min_samples=min_samples, n_jobs=n_jobs, **kwargs
        )

    labels = clusterer.fit(xy).labels_
    gdf["labels"] = labels

    return gdf


def moran_hot_cold_spots(moran_loc, p: int = 0.05) -> np.ndarray:
    """Get the hot and cold spots of the Moran_Local analysis.

    Parameters:
        moran_loc (esda.Moran_Local):
            The Moran_Local object.
        p (int):
            The p-value threshold to use.

    Returns:
        cluster (np.ndarray):
            The cluster labels of the objects.
    """
    sig = 1 * (moran_loc.p_sim < p)
    HH = 1 * (sig * moran_loc.q == 1)
    LL = 3 * (sig * moran_loc.q == 3)
    LH = 2 * (sig * moran_loc.q == 2)
    HL = 4 * (sig * moran_loc.q == 4)
    cluster = HH + LL + LH + HL

    return cluster


def find_lisa_clusters(
    gdf: gpd.GeoDataFrame,
    label: str,
    graph_type: str = "distband",
    dist_thresh: int = 100,
    permutations: int = 100,
    seed: int = 42,
    spatial_weights: W = None,
) -> Tuple[List[int], W]:
    """Calculate LISA clusters of objects with `class_name=label`.

    Note:
        LISA is short for local indicator of spatial association. You can read more,
        for example, from:
        - https://geodacenter.github.io/workbook/6a_local_auto/lab6a.html#lisa-principle.
        The LISA clusters are calculated using the local Moran analysis. The cluster
        labels are set to HH, LL, LH, HL.

    Note:
        In this function, the local statistic used to form the clusters is the fraction
        of objects of type `label` in the neighborhood times the absolute number of the
        objects of type `label` in the neighborhood. Due to the stochastic nature of the
        LISA analysis, the clustering results may wary marginally between runs if seed is
        changed. This is due to the random selection of the permutations in the
        local Moran analysis.

    Parameters:
        gdf (gpd.GeoDataFrame):
            The GeoDataFrame with the objects to calculate the LISA clusters of.
        label (str):
            The class name to calculate the LISA clusters of.
        graph_type (str):
            The type of graph to fit. Options are "delaunay", "knn" and "distband".
        dist_thresh (int):
            The distance threshold to use for the graph.
        permutations (int):
            The number of permutations to use in the Moran_Local analysis.
        seed (int):
            The random seed to use in the Moran_Local analysis.
        spatial_weights (W):
            The spatial weights object to use in the analysis.
            If None, the spatial weights are calculated.

    Returns:
        labels (List[int]):
            The cluster labels of the objects.
        w (W):
            The spatial weights object used in the analysis.

    Examples:
        Find the LISA clusters of inflammatory cells in a GeoDataFrame.
        >>> from cellseg_gsontools.clustering import find_lisa_clusters
        >>> from cellseg_gsontools.utils import read_gdf
        >>> cells = read_gdf("cells.geojson")
        >>> labels, w = find_lisa_clusters(cells, label="inflammatory", seed=42)
    """
    try:
        import esda
    except ImportError:
        raise ImportError(
            "This function requires the esda package to be installed."
            "Install it with: pip install esda"
        )

    if spatial_weights is not None:
        w = spatial_weights
    else:
        # Fit the distband
        w = fit_graph(
            gdf,
            type=graph_type,
            id_col="uid",
            thresh=dist_thresh,
        )

        # Row-standardized weights
        w.transform = "R"

    # Get the neihgboring nodes of the graph
    func = partial(neighborhood, spatial_weights=w)
    gdf["nhood"] = gdf_apply(gdf, func, columns=["uid"])

    # Get the classes of the neighboring nodes
    func = partial(nhood_vals, values=gdf["class_name"])
    gdf["nhood_classes"] = gdf_apply(
        gdf,
        func=func,
        parallel=True,
        columns=["nhood"],
    )

    # Get the number of inflammatory gdf in the neighborhood
    func = partial(nhood_type_count, cls=label, frac=False)
    gdf[f"{label}_cnt"] = gdf_apply(
        gdf,
        func=func,
        parallel=True,
        columns=["nhood_classes"],
    )

    # Get the fraction of objs of type `label` gdf in the neighborhood
    func = partial(nhood_type_count, cls=label, frac=True)
    gdf[f"{label}_frac"] = gdf_apply(
        gdf,
        func=func,
        parallel=True,
        columns=["nhood_classes"],
    )

    # This will smooth the extremes (e.g. if there is only one cell of type label in the
    # neighborhood, the fraction will be 1)
    gdf[f"{label}_index"] = gdf[f"{label}_frac"] * gdf[f"{label}_cnt"]

    # Standardize the index
    gdf[f"{label}_index_normed"] = gdf[f"{label}_index"] - gdf[f"{label}_index"].mean()

    # Find lisa clusters
    gdf[gdf[f"{label}_index_normed"] > 0][f"{label}_cnt"].value_counts(sort=False)

    gdf[f"{label}_index_lag"] = lag_spatial(w, gdf[f"{label}_index_normed"].values)

    lisa = esda.Moran_Local(
        gdf[f"{label}_index_normed"],
        w,
        island_weight=np.nan,
        seed=seed,
        permutations=permutations,
    )

    # Classify the gdf to HH, LL, LH, HL
    clusters = moran_hot_cold_spots(lisa)

    cluster_labels = ["ns", "HH", "LH", "LL", "HL"]
    labels = [cluster_labels[i] for i in clusters]

    return labels, w


def get_connected_components(gdf: gpd.GeoDataFrame, w: W) -> List[W]:
    """Get the connected components of a spatial weights object.

    Parameters:
        gdf (gpd.GeoDataFrame):
            The GeoDataFrame with the objects to get the connected components of.
        w (W):
            The spatial weights object.

    Returns:
        sub_graphs (List[W]):
            The connected components of the graph.

    Examples:
        Get the connected components of a GeoDataFrame.
        >>> from cellseg_gsontools.clustering import get_connected_components
        >>> from cellseg_gsontools.graphs import fit_graph
        >>> from cellseg_gsontools.utils import read_gdf, set_uid
        >>> cells = read_gdf("cells.geojson")
        >>> cells = cells[cells["class_name"] == "inflammatory"]
        >>> cells = set_uid(cells)
        >>> w = fit_graph(cells, type="distband", id_col="uid", thresh=100)
        >>> sub_graphs = get_connected_components(cells, w)
    """
    w_sub = w_subset(w, gdf.index.to_list(), silence_warnings=True)

    G = w_sub.to_networkx()
    sub_graphs = [
        W(nx.to_dict_of_lists(G.subgraph(c).copy()), silence_warnings=True)
        for c in nx.connected_components(G)
    ]

    return sub_graphs


def label_connected_components(
    gdf: gpd.GeoDataFrame, sub_graphs: List[W], label_col: str, min_size: int = 10
) -> gpd.GeoDataFrame:
    """Assign cluster labels to the objects in the GeoDataFrame.

    The cluster labels are assigned based on the connected components of the graph.

    Parameters:
        gdf (gpd.GeoDataFrame):
            The GeoDataFrame with the objects to assign cluster labels to.
        sub_graphs (List[W]):
            The connected components of the graph.
        label_col (str):
            The column name to assign the cluster labels to.
        min_size (int):
            The minimum size of the cluster to assign a label.

    Returns:
        gdf (gpd.GeoDataFrame):
            The GeoDataFrame with the assigned cluster labels.

    Examples:
        Assign cluster labels to the objects in a GeoDataFrame.
        >>> from cellseg_gsontools.clustering import label_connected_components
        >>> from cellseg_gsontools.graphs import fit_graph
        >>> from cellseg_gsontools.utils import read_gdf, set_uid
        >>> cells = read_gdf("cells.geojson")
        >>> cells = cells[cells["class_name"] == "inflammatory"]
        >>> cells = set_uid(cells)
        >>> w = fit_graph(cells, type="distband", id_col="uid", thresh=100)
        >>> sub_graphs = get_connected_components(cells, w)
        >>> labeled_cells = label_connected_components(
        ...     cells, sub_graphs, "label", min_size=10
        ... )
    """
    i = 0
    for ww in sub_graphs:
        idxs = list(ww.neighbors.keys())
        if len(idxs) < min_size:
            continue

        gdf.iloc[idxs, gdf.columns.get_loc(label_col)] = i
        i += 1

    return gdf


def cluster_cells(
    cells: gpd.GeoDataFrame,
    cell_type: str = "inflammatory",
    graph_type: str = "distband",
    dist_thresh: int = 100,
    min_size: int = 10,
    seed: int = 42,
    spatial_weights: W = None,
) -> gpd.GeoDataFrame:
    """Cluster the cells of the given type.

    Uses Local Moran analysis to find the LISA clusters of the cells.

    Note:
        LISA is short for local indicator of spatial association. You can read more,
        for example, from:
        - https://geodacenter.github.io/workbook/6a_local_auto/lab6a.html#lisa-principle.
        The LISA clusters are calculated using the local Moran analysis. The cluster
        labels are set to HH, LL, LH, HL.

    Note:
        In this function, the local statistic used to form the clusters is the fraction
        of objects of type `label` in the neighborhood times the absolute number of the
        objects of type `label` in the neighborhood. Due to the stochastic nature of the
        LISA analysis, the clustering results may wary marginally between runs if seed is
        changed. This is due to the random selection of the permutations in the
        local Moran analysis.

    Parameters:
        cells (gpd.GeoDataFrame):
            The GeoDataFrame with the cells.
        cell_type (str):
            The class name of the cells to cluster.
        graph_type (str):
            The type of graph to fit. Options are "delaunay", "knn" and "distband".
        dist_thresh (int):
            The distance threshold to use for the graph.
        min_size (int):
            The minimum size of the cluster to assign a label.
        seed (int):
            The random seed to use in the Moran_Local analysis.
        spatial_weights (W):
            The spatial weights object to use in the analysis.
            If None, the spatial weights are calculated.

    Returns:
        clustered_cells (gpd.GeoDataFrame):
            The GeoDataFrame with the clustered cells.

    Examples:
        Cluster the inflammatory cells in a GeoDataFrame.

        >>> from cellseg_gsontools.clustering import cluster_cells
        >>> from cellseg_gsontools.utils import read_gdf
        >>> cells = read_gdf("cells.geojson")
        >>> clustered_cells = cluster_cells(cells, cell_type="inflammatory", seed=42)
            class_name    geometry                            lisa_label    label
        uid
        0    inflammatory  POLYGON ((64.00 115.020, 69.010 ...  HH            0
        1    inflammatory  POLYGON ((65.00 15.020, 61.010 ...   HH            0
        2   inflammatory  POLYGON ((66.00 110.020, 69.010 ...   HH            2

    """
    # Find the LISA clusters
    lisa_labels, w = find_lisa_clusters(
        cells,
        label=cell_type,
        graph_type=graph_type,
        dist_thresh=dist_thresh,
        seed=seed,
        spatial_weights=spatial_weights,
    )
    cells["lisa_label"] = lisa_labels

    # Select the HH clusters
    clustered_cells = cells.loc[
        (cells["class_name"] == cell_type) & (cells["lisa_label"] == "HH")
    ]
    clustered_cells = clustered_cells.assign(label=-1)

    # Get the connected components
    sub_graphs = get_connected_components(clustered_cells, w)
    clustered_cells = label_connected_components(
        clustered_cells, sub_graphs, "label", min_size=min_size
    )
    clustered_cells.set_crs(4328, inplace=True, allow_override=True)

    # drop too small clusters
    clustered_cells = clustered_cells.loc[clustered_cells["label"] != -1]

    return clustered_cells
