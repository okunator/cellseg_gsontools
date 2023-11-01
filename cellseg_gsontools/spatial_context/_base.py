from typing import Any, Dict, Tuple, Union

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from libpysal.weights import W, w_subset, w_union

from ..grid import grid_overlay
from ..links import weights2gdf
from ..plotting import plot_all
from ..utils import set_uid
from .ops import get_objs

__all__ = ["_SpatialContext"]


class _SpatialContext:
    def __init__(
        self,
        area_gdf: gpd.GeoDataFrame,
        cell_gdf: gpd.GeoDataFrame,
        labels: str,
        min_area_size: Union[float, str] = None,
        graph_type: str = "delaunay",
        dist_thresh: float = 100.0,
        patch_size: Tuple[int, int] = (256, 256),
        stride: Tuple[int, int] = (256, 256),
        pad: int = None,
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
        self.patch_size = patch_size
        self.stride = stride
        self.pad = pad
        self.silence_warnings = silence_warnings
        self.labels = labels
        self.predicate = predicate

        # set up cell gdf
        self.cell_gdf = set_uid(cell_gdf, id_col="global_id")

        # set up area gdf and filter small areas
        area_gdf.set_crs(epsg=4328, inplace=True, allow_override=True)
        self.area_gdf = area_gdf

        # get the areas that have type in labels
        if isinstance(labels, str):
            # a little faster than .isin
            self.context_area = area_gdf[area_gdf["class_name"] == labels]
        else:
            if len(labels) == 1:
                self.context_area = area_gdf[area_gdf["class_name"] == labels[0]]
            else:
                self.context_area = area_gdf[area_gdf["class_name"].isin(labels)]

        # drop areas smaller than min_area_size
        if min_area_size is not None:
            self.context_area = self.context_area.loc[
                self.context_area.area >= min_area_size
            ]
        self.context_area = set_uid(self.context_area, id_col="global_id")

        # set to geocentric cartesian crs. (unit is metre not degree as by default)
        # helps to avoid warning flood
        self.cell_gdf.set_crs(epsg=4328, inplace=True, allow_override=True)
        self.context_area.set_crs(epsg=4328, inplace=True, allow_override=True)

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

    def context2weights(self, key: str) -> W:
        """Merge the networks of type `key` in the context into one spatial weights obj.

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
        """Merge the GeoDataFrames of type `key` in the context into one geodataframe.

        NOTE: Returns None if no data is found.

        Parameters
        ----------
            key : str
                The key of the context dictionary that contains the data to be converted
                to gdf. One of "roi_area", "roi_cells", "interface_area", "roi_grid",
                "interface_grid", "interface_cells", "roi_interface_cells"

        Returns
        -------
            gpd.GeoDataFrame:
                Geo dataframe containing all the objects
        """
        allowed = (
            "roi_area",
            "roi_cells",
            "interface_area",
            "roi_grid",
            "interface_grid",
            "interface_cells",
            "roi_interface_cells",
        )
        if key not in allowed:
            raise ValueError(f"Illegal key. Got: {key}. Allowed: {allowed}")

        con = []
        for i in self.context.keys():
            if self.context[i][key] is not None:
                if isinstance(self.context[i][key], tuple):
                    con.append(self.context[i][key][0])
                else:
                    con.append(self.context[i][key])

        if not con:
            return

        gdf = pd.concat(
            con,
            keys=[i for i in self.context.keys() if self.context[i][key] is not None],
        )
        gdf = gdf.explode(ignore_index=True)

        return gdf.reset_index(level=0, names="label").drop_duplicates("geometry")

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

    def plot(
        self,
        key: str,
        network_key: str = None,
        grid_key: str = None,
        show_legends: bool = True,
        color: str = None,
        figsize: Tuple[int, int] = (12, 12),
        edge_kws: Dict[str, Any] = None,
        **kwargs,
    ) -> plt.Axes:
        """Plot the slide with areas, cells, and interface areas highlighted.

        Parameters
        ----------
            key : str
                The key of the context dictionary that contains the data to be plotted.
                One of "roi_area", "interface_area",
            network_key : str, optional
                The key of the context dictionary that contains the spatial weights to
                be plotted. One of "roi_network", "full_network", "interface_network",
                "border_network"
            grid_key : str, optional
                The key of the context dictionary that contains the grid to be plotted.
                One of "roi_grid", "interface_grid"
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

        context_gdf = self.context2gdf(key)

        grid_gdf = None
        if grid_key is not None:
            grid_gdf = grid_overlay(
                context_gdf,
                self.patch_size,
                self.stride,
                self.pad,
                self.predicate,
            )
            if grid_gdf is not None:
                grid_gdf = grid_gdf.drop_duplicates("geometry")

        network_gdf = None
        if network_key is not None:
            edge_kws = edge_kws or {}
            w = self.context2weights(network_key)
            network_gdf = weights2gdf(self.cell_gdf, w)

        return plot_all(
            cell_gdf=self.cell_gdf,
            area_gdf=self.area_gdf,
            context_gdf=context_gdf,
            grid_gdf=grid_gdf,
            network_gdf=network_gdf,
            show_legends=show_legends,
            color=color,
            figsize=figsize,
            edge_kws=edge_kws,
            **kwargs,
        )
