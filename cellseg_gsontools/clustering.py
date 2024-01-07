import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, HDBSCAN, OPTICS

__all__ = ["cluster_points"]


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
