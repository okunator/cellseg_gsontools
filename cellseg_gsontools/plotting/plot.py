from typing import Any, Dict, List, Tuple

import geopandas as gpd
from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap

__all__ = ["plot_all", "plot_gdf"]


def replace_legend_items(legend, mapping):
    for txt in legend.texts:
        for k, v in mapping.items():
            if txt.get_text() == str(k):
                txt.set_text(v)


def plot_gdf(
    gdf: gpd.GeoDataFrame,
    col: str,
    ax: plt.Axes = None,
    cmap: str = None,
    bin_legends: List[str] = None,
    n_bins: int = None,
    show_legend: bool = True,
    loc: str = "upper right",
    figsize: tuple = (10, 10),
    **kwargs,
) -> plt.Axes:
    """Plot one gdf wrapper.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        The gdf to plot.
    col : str
        The column to highlight.
    ax : plt.Axes, optional
        The axes to plot on, by default None
    cmap : str, optional
        The colormap to use, by default None
    bin_legends : List[str], optional
        The bins to use, by default None
    n_bins : int, optional
        The number of bins to use, by default None
    show_legend : bool, optional
        Whether to show the legend, by default True
    loc : str, optional
        The location of the legend, by default "upper right"
    figsize : tuple, optional
        The size of the figure, by default (10, 10)
    **kwargs
        Extra keyword arguments passed to the `plot` method of the

    Returns
    -------
    plt.Axes
        The axes used for plotting.
    """
    if cmap is not None:
        cmap = get_cmap(cmap, n_bins)

    ax = gdf.plot(
        ax=ax,
        column=col,
        cmap=cmap,
        categorical=True,
        legend=show_legend,
        legend_kwds={"loc": loc},
        figsize=figsize,
        **kwargs,
    )
    if show_legend:
        leg = ax.legend_
        ax.add_artist(leg)

    if cmap is not None and show_legend and bin_legends is not None:
        mapping = dict([(i, s) for i, s in enumerate(bin_legends)])
        replace_legend_items(ax.get_legend(), mapping)

    return ax


def plot_all(
    area_gdf: gpd.GeoDataFrame,
    cell_gdf: gpd.GeoDataFrame,
    context_gdf: gpd.GeoDataFrame = None,
    network_gdf: gpd.GeoDataFrame = None,
    grid_gdf: gpd.GeoDataFrame = None,
    show_legends: bool = True,
    color: str = None,
    figsize: Tuple[int, int] = (12, 12),
    grid_cmap: str = None,
    grid_col: str = "class_name",
    grid_n_bins: int = None,
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
    grid_cmap : str, default="jet"
        The colormap to use for the grid. If `grid_gdf` is None, this is ignored.
    grid_col : str, default="class_name"
        The column to use for the grid. If `grid_gdf` is None, this is ignored.
    grid_n_bins : int, default=None
        The number of bins to use for the grid. If `grid_gdf` is None, this is ignored.
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
    if show_legends:
        leg1 = ax.legend_
        ax.add_artist(leg1)

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
    if show_legends:
        leg2 = ax.legend_
        ax.add_artist(leg2)

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
        if show_legends:
            leg3 = ax.legend_
            ax.add_artist(leg3)

    if grid_gdf is not None:
        grid_gdf.geometry = grid_gdf.boundary

        cmap = None
        if color is not None or grid_cmap is not None:
            if grid_n_bins is not None:
                cmap = get_cmap(grid_cmap, grid_n_bins)
            else:
                cmap = get_cmap(grid_cmap)

        ax = grid_gdf.plot(
            ax=ax,
            column=grid_col,
            color=color,
            alpha=0.7,
            cmap=cmap,
            categorical=True,
            legend=show_legends,
            legend_kwds={"loc": "lower left"},
        )
        if show_legends:
            leg4 = ax.legend_
            ax.add_artist(leg4)

    return ax
