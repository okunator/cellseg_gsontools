import geopandas as gpd
import numpy as np
import pandas as pd
from esda.adbscan import ADBSCAN
from sklearn.cluster import DBSCAN, OPTICS

__all__ = ["cluster_points"]


def cluster_points(
    gdf: gpd.GeoDataFrame,
    eps: float = 350.0,
    min_samples: int = 30,
    method: str = "dbscan",
    n_jobs: int = -1,
) -> gpd.GeoDataFrame:
    """Apply a clustering to centroids in a gdf.

    NOTE: this is just a quick wrapper for a few clustering algos adapted
    to geodataframes.

    For now dbscan and its variants can be used as the clustering method.
    (dbscan, optics, adbscan). This means that label -1 equals to noise!

    Parameters
    ----------
        gdf : np.ndarray or gpd.GeoDataFrame
            Input geo dataframe with a properly set geometry column.
        eps : float, default=0.5
            The maximum distance between two samples for one to be considered as in the
            neighborhood of the other. This is not a maximum bound on the distances of
            gdf within a cluster.
        min_samples : int, default=35
            The number of samples (or total weight) in a neighborhood for a point to be
            considered as a core point. This includes the point itself.
        method : str, default="dbscan"
            The clustering method to be used. Allowed: ("dbscan", "adbscan", "optics").
        n_jobs : int, default=-1
            The number of parallel jobs to run. None means 1. -1 means using all
            processors.

    Raises
    ------
        ValueError if illegal method is given or input `gdf` is of wrong type.

    Returns
    -------
        gpd.GeoDataFrame:
            The input gdf with a new "labels" columns of the clusters.
    """
    allowed = ("dbscan", "adbscan", "optics")
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
        xy = pd.DataFrame({"X": xy[:, 0], "Y": xy[:, 1]})
        clusterer = ADBSCAN(eps=eps, min_samples=min_samples, n_jobs=n_jobs)
    elif method == "dbscan":
        clusterer = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=n_jobs)
    elif method == "optics":
        clusterer = OPTICS(max_eps=eps, min_samples=min_samples, n_jobs=n_jobs)

    labels = clusterer.fit(xy).labels_
    gdf["labels"] = labels

    return gdf