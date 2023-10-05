from typing import Dict, Tuple, Union

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from libpysal.weights import W, w_subset, w_union

from ..graphs import fit_graph
from ..plotting import plot_graph
from ..utils import set_uid
from .ops import get_objs

__all__ = ["_SpatialContext"]


class _SpatialContext:
    def __init__(
        self,
        area_gdf: gpd.GeoDataFrame,
        cell_gdf: gpd.GeoDataFrame,
        label: str,
        min_area_size: Union[float, str] = None,
        q: float = 25.0,
        graph_type: str = "delaunay",
        dist_thresh: float = 100.0,
        predicate: str = "intersects",
        silence_warnings: bool = True,
    ) -> None:
        """Create a base class for spatial context."""
        if "class_name" not in area_gdf.columns:
            raise ValueError(
                "'class_name' column not found. `area_gdf` must contain a 'class_name' "
                f"column. Got: {list(area_gdf.columns)}"
            )

        if "class_name" not in cell_gdf.columns:
            raise ValueError(
                "'class_name' column not found. `cell_gdf` must contain a 'class_name' "
                f"column. Got: {list(cell_gdf.columns)}"
            )

        self.dist_thresh = dist_thresh
        self.graph_type = graph_type
        self.silence_warnings = silence_warnings
        self.label = label
        self.predicate = predicate

        # set up cell gdf
        self.cell_gdf = set_uid(cell_gdf, id_col="global_id")

        # set up area gdf and filter small areas
        area_gdf.set_crs(epsg=4328, inplace=True, allow_override=True)
        self.area_gdf = area_gdf
        thresh = self._get_thresh(
            area_gdf[area_gdf["class_name"] == label], min_area_size, q
        )
        self.context_area = self.filter_above_thresh(area_gdf, label, thresh)
        self.context_area = set_uid(
            self.context_area, id_col="global_id"
        )  # set global uid (parent gdf), starts from 1

        # set to geocentric cartesian crs. (unit is metre not degree as by default)
        # helps to avoid warning flood
        self.cell_gdf.set_crs(epsg=4328, inplace=True, allow_override=True)
        self.context_area.set_crs(epsg=4328, inplace=True, allow_override=True)

    @staticmethod
    def filter_above_thresh(
        gdf: gpd.GeoDataFrame,
        label: str,
        thresh: float = None,
    ) -> gpd.GeoDataFrame:
        """Filter areas or objects that are above a threshold.

        NOTE: threshold by default is the mean of the area.
        """
        gdf = gdf.loc[gdf["class_name"] == label].copy()
        gdf.loc[:, "area"] = np.round(gdf.area)

        if thresh is not None:
            gdf = gdf.loc[gdf.area >= thresh]

        # drop the area column to avoid confusion
        gdf = gdf.drop(columns=["area"])

        return gdf

    def roi(self, ix) -> gpd.GeoDataFrame:
        """Get a roi area of index `ix`.

        Parameters
        ----------
            ix : int
                The index of the roi area. I.e., the ith roi area.

        Returns
        -------
            gpd.GeoDataFrame:
                The ith roi area.
        """
        row: gpd.GeoSeries = self.context_area.loc[ix]
        roi_area = gpd.GeoDataFrame([row], crs=self.context_area.crs)

        return roi_area

    def roi_cells(
        self,
        ix: int = None,
        roi_area: gpd.GeoDataFrame = None,
        predicate: str = "within",
    ) -> gpd.GeoDataFrame:
        """Get the cells within the roi area.

        Parameters
        ----------
            ix : int, optional
                The index of the roi area. I.e., the ith roi area. If None, the
                `roi_area` must be given.
            roi_area : gpd.GeoDataFrame, optional
                The roi area. If None, the roi area is extracted from the context with
                the `ix` key.
            predicate : str, default="within"
                The predicate to use for the spatial join. See `geopandas.tools.sjoin`

        Returns
        -------
            gpd.GeoDataFrame:
                The cells within the roi area.
        """
        # check to not compute roi_area if already computed
        if not isinstance(roi_area, gpd.GeoDataFrame):
            if (roi_area, ix) == (None, None):
                raise ValueError("Either `ix` or `roi_area` must be given.")
            roi_area: gpd.GeoDataFrame = self.roi(ix)

        if roi_area is None or roi_area.empty:
            return

        return self.get_objs_within(roi_area, self.cell_gdf, predicate=predicate)

    def merge_weights(self, key: str) -> W:
        """Merge libpysal spatial weights of the context.

        Parameters
        ----------
            key : str
                The key of the context dictionary that contains the spatial
                weights to be merged. One of "roi_network", "full_network",
                "interface_network", "border_network"

        Returns
        -------
            libpysal.weights.W:
                A spatial weights object containing all the distinct networks
                in the context.
        """
        allowed = ("roi_network", "full_network", "interface_network", "border_network")
        if key not in allowed:
            raise ValueError(f"Illegal key. Got: {key}. Allowed: {allowed}")

        cxs = list(self.context.items())
        wout = W({0: [0]})
        for _, c in cxs:
            w = c[key]
            if isinstance(w, W):
                wout = w_union(wout, w, silence_warnings=True)

        # remove self loops
        wout = w_subset(wout, list(wout.neighbors.keys())[1:], silence_warnings=True)

        return wout

    def context2gdf(self, key: str) -> gpd.GeoDataFrame:
        """Convert the context of type `key` into a geodataframe.

        Parameters
        ----------
            key : str
                The key of the context dictionary that contains the data to be converted
                to gdf. One of "roi_area", "roi_cells", "interface_area",
                "interface_cells", "roi_interface_cells"

        Returns
        -------
            gpd.GeoDataFrame:
                Geo dataframe containing all the objects
        """
        con = []
        for i in self.context.keys():
            if self.context[i][key] is not None:
                if isinstance(self.context[i][key], tuple):
                    con.append(self.context[i][key][0])
                else:
                    con.append(self.context[i][key])

        gdf = pd.concat(
            con,
            keys=[i for i in self.context.keys() if self.context[i][key] is not None],
        )
        gdf = gdf.explode(ignore_index=True)

        return gdf.reset_index(level=0, names="label")

    def context2weights(self, key: str, **kwargs) -> W:
        """Fit a network on the cells inside the context `key`.

        Parameters
        ----------
            key : str
                The key of the context dictionary that contains the data to be converted
                to gdf. One of "roi_cells", "interface_cells"
        """
        allowed = ("roi_cells", "interface_cells")
        if key not in allowed:
            raise ValueError(
                "Illegal key. Note that network can be only fitted to cell gdfs. "
                f"Got: {key}. Allowed: {allowed}"
            )

        cells = self.context2gdf(key)
        cells = cells.drop_duplicates(subset=["global_id"], keep="first")
        w = fit_graph(
            cells,
            type=self.graph_type,
            id_col="global_id",
            thresh=self.dist_thresh,
        )

        return w

    def get_objs_within(
        self,
        area: gpd.GeoDataFrame,
        objects: gpd.GeoDataFrame,
        predicate: str = "within",
    ) -> gpd.GeoDataFrame:
        """Get the objects within the area.

        Parameters
        ----------
            area : gpd.GeoDataFrame
                The area of interest in GeoDataFrame.
            objects : gpd.GeoDataFrame
                The objects (cells) of interest.
            predicate : str, default="within"
                The predicate to use for the spatial join. See `geopandas.tools.sjoin`

        Returns
        -------
            gpd.GeoDataFrame:
                The objects (cells) within the area gdf.
        """
        objs_within = get_objs(
            area, objects, silence_warnings=self.silence_warnings, predicate=predicate
        )

        if objs_within is None or objs_within.empty:
            return

        # rename spatial join columns
        objs_within = objs_within.rename(
            columns={
                "index_right": "spatial_context_id",
                "global_id_left": "global_id",
                "class_name_left": "class_name",
                "class_name_right": "spatial_context",
            },
            inplace=False,
        )

        # drop unnecessary columns and return
        objs_within.drop(columns=["global_id_right"], inplace=True)

        return objs_within

    def _get_thresh(self, area_gdf, min_area_size, q=None) -> float:
        """Get the threshold value for filtering by area."""
        if isinstance(min_area_size, str):
            allowed = ("mean", "median", "quantile")
            if min_area_size not in allowed:
                raise ValueError(
                    f"Got illegal `min_area_size`. Got: {min_area_size}. "
                    f"Allowed values are floats or these options: {allowed}."
                )
            if min_area_size == "mean":
                thresh = area_gdf.area.mean()
            elif min_area_size == "median":
                thresh = area_gdf.area.median()
            elif min_area_size == "quantile":
                thresh = np.nanpercentile(area_gdf.area, q)
        elif isinstance(min_area_size, (int, float)):
            thresh = float(min_area_size)
        elif min_area_size is None:
            thresh = None
        else:
            raise ValueError(
                f"Got illegal `min_area_size`. Got: {min_area_size}. "
                f"Allowed values are floats or these options: {allowed}."
            )

        return thresh

    def plot(
        self,
        key: str,
        network_key: str = None,
        show_area: bool = True,
        show_cells: bool = True,
        show_legends: bool = True,
        color: str = None,
        figsize: Tuple[int, int] = (12, 12),
        **kwargs,
    ) -> plt.Axes:
        """Plot the slide with areas, cells, and interface areas highlighted.

        Parameters
        ----------
            key : str
                The key of the context dictionary that contains the data to be plotted.
                One of "roi_area", "roi_cells",
            network_key : str, optional
                The key of the context dictionary that contains the spatial weights to
                be plotted. One of "roi_network", "full_network", "interface_network",
                "border_network"
            show_area : bool, default=True
                Flag, whether to include the tissue areas in the plot.
            show_cells : bool, default=True
                Flag, whether to include the cells in the plot.

            show_legends : bool, default=True
                Flag, whether to include legends for each in the plot.
            color : str, optional
                A color for the interfaces, Ignored if `show_legends=True`.
            figsize : Tuple[int, int], default=(12, 12)
                Size of the figure.
            **kwargs
                Extra keyword arguments passed to the `plot` method of the
                geodataframes.

        Returns
        -------
            AxesSubplot

        Examples
        --------
        Plot the slide with cluster areas and cells highlighted
        >>> from cellseg_gsontools.spatial_context import PointClusterContext

        >>> cells = read_gdf("cells.feather")
        >>> clusters = PointClusterContext(
        ...     cell_gdf=cells,
        ...     label="inflammatory",
        ...     cluster_method="optics",
        ... )

        >>> clusters.fit(verbose=False)
        >>> clusters.plot("roi_area", show_legends=True, aspect=1)
        <AxesSubplot: >
        """
        allowed = ("roi_area", "interface_area")
        if key not in allowed:
            raise ValueError(f"Illegal key. Got: {key}. Allowed: {allowed}")

        _, ax = plt.subplots(figsize=figsize)

        if show_area:
            ax = self.area_gdf.plot(
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

        if show_cells:
            ax = self.cell_gdf.plot(
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

        context = self.context2gdf(key)
        ax = context.plot(
            ax=ax,
            color=color,
            column="label",
            alpha=0.7,
            legend=show_legends,
            categorical=True,
            legend_kwds={"loc": "upper left"},
            aspect=None,
            **kwargs,
        )
        if show_legends:
            if show_area:
                ax.add_artist(leg1)
            if show_cells:
                ax.add_artist(leg2)

        if network_key is not None:
            _, ax = self.plot_weights(
                network_key,
                ax=ax,
                ix=-1,
                edge_kws=dict(color="r", linestyle=":", linewidth=1),
                node_kws=dict(marker=""),
            )

        return ax

    def plot_weights(
        self,
        key: str,
        ax: plt.Axes,
        ix: int = -1,
        id_col: str = "global_id",
        node_kws: Dict = None,
        edge_kws: Dict = None,
        figsize: Tuple[float, float] = (10, 10),
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot the spatial weights graph.

        Parameters
        ----------
            key : str
                The key of the context dictionary that contains the spatial weights
                to be plotted. One of "roi_network", "full_network",
                "interface_network", "border_network"
            ax : plt.Axes
                The axes to plot on. If None, a new figure is created.
            ix : int, default=-1
                The index of the context network. I.e., the ith context network of
                type `key`. If -1 or None, the merged context network is plotted.
            id_col : str, default="global_id"
                The unique id column in the gdf.
            node_kws : dict, optional
                Keyword arguments passed to the `ax.scatter` method.
            edge_kws : dict, optional
                Keyword arguments passed to the `ax.plot` method.
            figsize : Tuple[float, float], optional
                The size of the figure to plot, by default (10, 10).

        Returns
        -------
            plt.Figure, plt.Axes

        Examples
        --------
        Plot the spatial weights graph of the cells inside immune clusters.
        >>> import matplotlib.pyplot as plt
        >>> from cellseg_gsontools.spatial_context import PointClusterContext

        >>> clusters = PointClusterContext(
        ...     cell_gdf=cells,
        ...     label="inflammatory",
        ...     cluster_method="optics",
        ... )

        >>> clusters.fit(verbose=False)

        >>> f, ax = plt.subplots(figsize=(10, 10))

        >>> ix = 2
        >>> clusters.context[ix]["roi_area"].plot(ax=ax)
        >>> cells.plot(
        ...     ax=ax,
        ...     column="class_name",
        ...     categorical=True,
        ...     aspect=1
        ... )

        >>> clusters.plot_weights(
        ...     "roi_network",
        ...     ix,
        ...     ax=ax,
        ...     edge_kws=dict(color='r', linestyle=':', linewidth=1),
        ...     node_kws=dict(marker='')
        ... )
        <AxesSubplot: >

        Plot tumor-stroma interface border netwroks of the cells inside the interfaces
        >>> from cellseg_gsontools.spatial_context import InterfaceContext
        >>> import geopandas as gpd
        >>> import matplotlib.pyplot as plt

        >>> cells = gpd.read_feather(cells.feather)
        >>> areas = gpd.read_feather(areas.feather)

        >>> iface_context = InterfaceContext(
        ...     area_gdf=areas,
        ...     cell_gdf=cells,
        ...     label1="area_cin",
        ...     label2="areastroma",
        ...     silence_warnings=True,
        ...     min_area_size=100000.0,
        ... )
        >>> iface_context.fit()

        >>> f, ax = plt.subplots(figsize=(20, 20))

        >>> iface_context.context2gdf("interface_area").plot(ax=ax, alpha=0.5)
        >>> cells.plot(
        ...     ax=ax,
        ...     column="class_name",
        ...     categorical=True,
        ...     aspect=1
        ... )

        >>> iface_context.plot_weights(
        ...     "border_network",
        ...     ix=-1,
        ...     ax=ax,
        ...     edge_kws=dict(color='r', linestyle=':', linewidth=1),
        ...     node_kws=dict(marker='')
        ... )
        <AxesSubplot: >
        """
        allowed = (
            "roi_network",
            "full_network",
            "interface_network",
            "border_network",
        )
        if key not in allowed:
            raise ValueError(f"Illegal key. Got: {key}. Allowed: {allowed}")

        gdf = self.cell_gdf

        # merge weights if ix not given
        if ix == -1 or ix is None:
            w = self.merge_weights(key)
        else:
            w = self.context[ix][key]

        if w is None:
            raise ValueError(
                f"Got None for the spatial weights of type `{key}`. "
                "Make sure you have run `fit` before calling `plot_weights`. "
                f"If you have run `fit`, then the spatial weights obj of type `{key}` "
                "is None."
            )

        f, ax = plot_graph(
            gdf,
            w,
            indexed_on=id_col,
            ax=ax,
            node_kws=node_kws,
            edge_kws=edge_kws,
            figsize=figsize,
        )

        return f, ax
