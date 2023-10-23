from typing import Any, Dict, Tuple

import geopandas as gpd
from matplotlib import pyplot as plt

__all__ = ["plot_all"]


def plot_all(
    area_gdf: gpd.GeoDataFrame,
    cell_gdf: gpd.GeoDataFrame,
    context_gdf: gpd.GeoDataFrame = None,
    network_gdf: gpd.GeoDataFrame = None,
    grid_gdf: gpd.GeoDataFrame = None,
    show_legends: bool = True,
    color: str = None,
    figsize: Tuple[int, int] = (12, 12),
    edge_kws: Dict[str, Any] = None,
    **kwargs,
) -> plt.Axes:
    """Plot the slide with areas, cells, and interface areas highlighted.

    Parameters
    ----------
    area_gdf : gpd.GeoDataFrame
        GeoDataFrame containing the areas of interest.
    cell_gdf : gpd.GeoDataFrame
        GeoDataFrame containing the cells.
    context_gdf : gpd.GeoDataFrame, optional
        GeoDataFrame containing the context, by default None
    network_gdf : gpd.GeoDataFrame, optional
        GeoDataFrame containing the network, by default None
    grid_gdf : gpd.GeoDataFrame, optional
        GeoDataFrame containing the grid, by default None
    show_legends : bool, default=True
        Flag, whether to include legends for each in the plot.
    color : str, optional
        A color for the interfaces or rois, Ignored if `show_legends=True`.
    figsize : Tuple[int, int], default=(12, 12)
        Size of the figure.
    **kwargs
        Extra keyword arguments passed to the `plot` method of the
        geodataframes.

    Returns
    -------
        AxesSubplot: The axes used for plotting.
    """
    _, ax = plt.subplots(figsize=figsize)

    ax = area_gdf.plot(
        ax=ax,
        column="class_name",
        categorical=True,
        legend=show_legends,
        alpha=0.1,
        legend_kwds={
            "loc": "upper center",
        },
        aspect=None,
        **kwargs,
    )
    leg1 = ax.legend_

    ax = cell_gdf.plot(
        ax=ax,
        column="class_name",
        categorical=True,
        legend=show_legends,
        legend_kwds={
            "loc": "upper right",
        },
        aspect=None,
        **kwargs,
    )
    leg2 = ax.legend_

    if context_gdf is not None:
        ax = context_gdf.plot(
            ax=ax,
            color=color,
            column="label",
            alpha=0.7,
            legend=False,
            categorical=True,
            aspect=None,
            **kwargs,
        )

    if grid_gdf is not None:
        grid_gdf.geometry = grid_gdf.boundary
        ax = grid_gdf.plot(
            ax=ax,
            color=color,
            alpha=0.7,
            # column="class_name",
            # cmap="jet",
        )

    if network_gdf is not None:
        edge_kws = edge_kws or {}
        ax = network_gdf.plot(
            ax=ax,
            column="class_name",
            categorical=True,
            legend=show_legends,
            legend_kwds={"loc": "upper left"},
            cmap="Paired",
            **edge_kws,
        )
        leg3 = ax.legend_

    if show_legends:
        ax.add_artist(leg1)
        ax.add_artist(leg2)
        ax.add_artist(leg3)

    return ax
