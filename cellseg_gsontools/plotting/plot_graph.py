from typing import Dict, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from libpysal.weights import W

__all__ = ["plot_graph"]


# This is the W.plot() class-methsd from PySAL, only added figsize param.
# https://github.com/pysal/libpysal/blob/main/libpysal/weights/weights.py
def plot_graph(
    gdf: gpd.GeoDataFrame,
    w: W,
    indexed_on: str = None,
    ax: plt.Axes = None,
    node_kws: Dict = None,
    edge_kws: Dict = None,
    figsize: Tuple[float, float] = (10, 10),
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot a graph from a GeoDataFrame and a PySAL W object.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        The GeoDataFrame to plot.
    w : W
        The PySAL W object that defines the graph.
    indexed_on : str, optional
        The column in `gdf` to use as the index for the graph, by default None
    ax : plt.Axes, optional
        The axes to plot on, by default None
    node_kws : Dict, optional
        Keyword arguments to pass to the node plotting function, by default None
    edge_kws : Dict, optional
        Keyword arguments to pass to the edge plotting function, by default None
    figsize : Tuple[float, float], optional
        The size of the figure to plot, by default (10, 10).

    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        The figure and axes used for plotting.

    Examples
    --------
    >>> import geopandas as gpd
    >>> import libpysal
    """
    if ax is None:
        f = plt.figure(figsize=figsize)
        ax = plt.gca()
    else:
        f = plt.gcf()

    color = "k"
    if node_kws is None:
        node_kws = dict(color=color)
    if edge_kws is None:
        edge_kws = dict(color=color)

    for idx, neighbors in w.neighbors.items():
        # skip islands
        if idx in w.islands:
            continue

        if indexed_on is not None:
            neighbors = gdf[gdf[indexed_on].isin(neighbors)].index.tolist()
            idx = gdf[gdf[indexed_on] == idx].index.tolist()[0]

        centroids = gdf.loc[neighbors].centroid.apply(lambda p: (p.x, p.y))
        centroids = np.vstack(centroids.values)
        focal = np.hstack(gdf.loc[idx].geometry.centroid.xy)

        seen = set()
        for nidx, neighbor in zip(neighbors, centroids):
            if (idx, nidx) in seen:
                continue
            ax.plot(*list(zip(focal, neighbor)), marker=None, **edge_kws)
            seen.update((idx, nidx))
            seen.update((nidx, idx))

    ax.scatter(
        gdf.centroid.apply(lambda p: p.x),
        gdf.centroid.apply(lambda p: p.y),
        **node_kws,
    )

    return f, ax
