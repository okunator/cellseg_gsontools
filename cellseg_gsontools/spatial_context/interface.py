import warnings
from typing import List, Tuple, Union

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from libpysal.weights import DistanceBand, w_subset
from shapely.geometry import Polygon
from tqdm import tqdm

from ..utils import set_uid
from ._base import _SpatialContext

__all__ = ["get_interface_zones", "InterfaceContext"]


def get_interface_zones(
    roi: Union[Polygon, gpd.GeoDataFrame], gdf: gpd.GeoDataFrame, buffer_dist: int = 200
) -> gpd.GeoDataFrame:
    """Get the interfaces b/w the polygons defined in a gdf and a given a roi/area.

    Interface is the region around the border of two touching polygons. The interface
    radius is determined by the `buffer_dist` parameter.

    Applies a buffer to the roi and finds the intersection between the buffer and
    the polygons in a gdf.

    Useful for example, when you want to extract the interface of two distinct tissues
    like stroma and cancer.

    Parameters
    ----------
        roi : Polygon or gpd.GeoDataFrame
            The area or region of interest whichse borders are buffered.
        gdf : gpd.GeoDataFrame
            A geodataframe containing polygons that might intersect with the roi.
        buffer_dist : int, default=200
            The radius of the buffer.

    Raises
    ------
        TypeError: If the input `roi` has an illegal type.

    Returns
    -------
        gpd.GeoDataFrame:
            A geodataframe containing the intersecting polygons including the buffer.

    """
    if isinstance(roi, Polygon):
        roi = gpd.GeoDataFrame({"geometry": [roi]})  # convert to gdf
    elif isinstance(roi, pd.Series):
        roi = gpd.GeoDataFrame([roi])
    elif not isinstance(roi, gpd.GeoDataFrame):
        raise TypeError(
            "`roi` has to be either a Polygon, pd.Series or GeoDataFrame object. "
            f"Got: {type(roi)}"
        )

    buffer_zone = gpd.GeoDataFrame({"geometry": list(roi.buffer(buffer_dist))})
    inter = gdf.overlay(buffer_zone, how="intersection")

    # if the intersecting area is covered totally by any polygon in the gdf
    # take the difference of the intresecting area and the orig roi to discard
    # the roi from the interface 'sheet'
    if not inter.empty:
        if gdf.covers(inter.geometry.loc[0]).any():
            inter = inter.overlay(roi, how="difference")

    return inter


class InterfaceContext(_SpatialContext):
    def __init__(
        self,
        area_gdf: gpd.GeoDataFrame,
        cell_gdf: gpd.GeoDataFrame,
        label1: str,
        label2: str,
        min_area_size: Union[float, str] = None,
        q: float = 25.0,
        buffer_dist: int = 200,
        verbose: bool = False,
        silence_warnings: bool = True,
    ) -> None:
        """Handle & extract interface regions from the `cell_gdf` and `area_gdf`.

        Interface is the band-like area b/w the areas of type `label1` and `label2`.
        The bredth of the interface is given by the `buffer_dist` param.

        Parameters
        ----------
            area_gdf : gpd.GeoDataFrame
                A geo dataframe that contains large polygons enclosing smaller objs.
            cell_gdf : gpd.GeoDataFrame
                A geo dataframe that contains smaller cell objs enclosed in larger areas
            label1 : str
                The class name of the areas of interest. E.g. "cancer". These areas are
                buffered on top of the area of type `label2`
            label2 : str
                The class name of the area on top of which the buffering is applied.
            min_area_size : float or str, optional
                The minimum area of the objects that are kept. All the objects in the
                `area_gdf` that are larger are kept. Can be either a float or one of:
                "mean", "median", "quantile"
            q : float, default=25.0
                The quantile. This is only used if `min_area_size = "quantile"`.
            buffer_dist : int, default=200
                The radius of the buffer.
            verbose : bool, default=False
                Flag, whether to use tqdm pbar.
            silence_warnings : bool, default=True
                Flag, whether to silence all the warnings.

        Attributes
        ----------
            context : Dict[int, Dict[str, Union[gpd.GeoDataFrame, DistanceBand]]]
                A nested dict that contains dicts for each index of the distinct areas
                of type `label1`. Each of the inner dicts contain the keys: 'roi_area',
                'interface_area', 'interface_cells', 'roi_cells', 'roi_network',
                'interface_network', 'full_network', and 'border_network'. The 'area'
                and 'cells' keys contain gpd.GeoDataFrame of the roi and the interfacing
                area and cells. The 'network' keys contain DistanceBands fitted to the
                cells inside the roi interface, union of the roi and interface and a
                network between cells at the border of the roi and interface areas.

        Raises
        ------
            ValueError if `area_gdf` or `cell_gdf` don't contain 'class_name' column.

        Examples
        --------
        Define an interface context and plot the cells inside the interface area.

        >>> from cellseg_gsontools.spatial_context import InterfaceContext

        >>> area_gdf = read_gdf("area.json")
        >>> cell_gdf = pre_proc_gdf(read_gdf("cells.json"))
        >>> interface_context = InterfaceContext(
                area_gdf=area_gdf,
                cell_gdf=cell_gdf,
                label1="area_cin",
                label2="area_stroma",
                buffer_dist=250.0,
                silence_warnings=True,
                verbose=True,
                min_area_size=100000.0
            )

        >>> ix = 1
        >>> ax = interface_context.context[ix]["interface_area"].plot(
                figsize=(10, 10), alpha=0.5
            )
        >>> interface_context.context[ix]["interface_cells"].plot(
                ax=ax, column="class_name", categorical=True, legend=True
            )
        """
        super().__init__(
            area_gdf=area_gdf,
            cell_gdf=cell_gdf,
            label=label1,
            min_area_size=min_area_size,
            q=q,
            verbose=verbose,
            silence_warnings=silence_warnings,
        )
        self.buffer_dist = buffer_dist

        thresh = self._get_thresh(
            area_gdf[area_gdf["class_name"] == label2], min_area_size, q
        )
        self.context_area2 = self.filter_above_thresh(area_gdf, label2, thresh)

        self.context = {}
        pbar = (
            tqdm(self.context_area.index, total=self.context_area.shape[0])
            if self.verbose
            else self.context_area.index
        )
        for ix in pbar:
            if verbose:
                pbar.set_description(f"Processing interface area: {ix}")

            self.context[ix] = {"roi_area": self.roi(ix)}
            self.context[ix]["interface_area"] = self.interface(ix)
            self.context[ix]["interface_cells"] = self.interface_cells(ix)
            self.context[ix]["roi_cells"] = self.roi_cells(ix)
            self.context[ix]["roi_interface_cells"] = self.roi_interface_cells(ix)

            union_net, inter_net, roi_net = self.cell_neighbors(ix, which="all")
            self.context[ix]["roi_network"] = roi_net
            self.context[ix]["interface_network"] = inter_net
            self.context[ix]["full_network"] = union_net
            self.context[ix]["border_network"] = self.border_neighbors(ix)

    def interface(self, ix: int) -> gpd.GeoDataFrame:
        """Get an interface area of index `ix`."""
        # Get the intersection of the area1 zone and the interface area (Multi)Polygon
        try:
            iface = self.context[ix]["interface_area"]
        except KeyError:
            roi = self.roi(ix)
            iface = get_interface_zones(
                roi=roi, gdf=self.context_area2, buffer_dist=self.buffer_dist
            )
            if len(iface) > 1:
                aggfuncs = {c: "first" for c in iface.columns if c != "geometry"}
                aggfuncs["area"] = "sum"
                iface = iface.dissolve(aggfunc=aggfuncs).set_geometry("geometry")

        return iface

    def interface_cells(self, ix: int) -> gpd.GeoDataFrame:
        """Get the cells inside the interface area."""
        try:
            iface_cells = self.context[ix]["interface_cells"]
        except KeyError:
            interface = self.interface(ix)

            if interface.empty:
                if not self.silence_warnings:
                    warnings.warn(
                        "`self.interface` resulted in an empty GeoDataFrame. Make sure "
                        "to set a valid `label2` and `ix`. Returning None "
                        " from`interface_cells`",
                        RuntimeWarning,
                    )

                return None
            iface_cells = self.cell_gdf[self.cell_gdf.within(interface.geometry.loc[0])]

        return iface_cells

    def roi_interface_cells(self, ix: int) -> Tuple[gpd.GeoDataFrame, int, int]:
        """Concatenate the cells of the interface and roi to one gdf.

        NOTE: returns also the lengths of the two frames that are concatenated.
        """
        try:
            roi_iface_cells = self.context[ix]["roi_interface_cells"]
        except KeyError:
            roi = self.roi_cells(ix)
            iface = self.interface_cells(ix)
            if iface is not None and roi is not None:
                roi_iface_cells = (
                    set_uid(pd.concat([roi, iface]), id_col="id"),
                    len(roi),
                    len(iface),
                )
            elif iface is None and roi is not None:
                roi_iface_cells = (set_uid(roi, id_col="id"), len(roi), 0)
            elif iface is not None and roi is None:
                roi_iface_cells = (set_uid(iface, id_col="id"), 0, len(iface))
            else:
                roi_iface_cells = None

        return roi_iface_cells

    def cell_neighbors(
        self,
        ix: int,
        thresh: float = 75.0,
        which: str = "union",
    ) -> Union[DistanceBand, Tuple[DistanceBand, DistanceBand, DistanceBand]]:
        """Create a distance network of the cells.

        NOTE: option to subset the netwrok based on the context. I.e.
        subsets of the full network containing cells from the interface and roi.

        Parameters
        ----------
            ix : int
                The index of the ROI geo-object. Starts from one.
            thresh : float, default=75.0
                Distance threshold for the network.
            which : str, default="union"
                Flag whether to fit the network on the cells of the 'roi', 'interface'
                or the 'union' of the roi and interface. If `which == 'all`, all the
                options are fitted and returned.
        Returns
        -------
            Union[DistanceBand, Tuple[DistanceBand, DistanceBand, DistanceBand]]:
                Either one distance network or all three possible distance networks.
        """

        def get_subset(subset_ids: List[int], cells: gpd.GeoDataFrame):
            ids = []
            for id in subset_ids:
                uid = cells["uid"].loc[id]
                if uid in w.neighbors.keys():
                    ids.extend([uid] + w.neighbors[uid])
            return ids

        allowed = ("union", "all", "roi", "interface")
        if which not in allowed:
            raise ValueError(f"Illegal arg: `which`. Got: {which}. Allowed: {allowed}.")

        try:
            return (
                self.context[ix]["full_network"],
                self.context[ix]["interface_network"],
                self.context[ix]["roi_network"],
            )
        except KeyError:
            cells, len_roi, len_iface = self.roi_interface_cells(ix)
            if cells is None:
                return None

            w = DistanceBand.from_dataframe(
                cells, threshold=thresh, ids="uid", alpha=-1.0, silence_warnings=True
            )

            if which == "union":
                return w
            elif which == "interface":
                subset_ids = list(range(len_roi + 1, len_iface + len_roi + 1))
            elif which == "roi":
                subset_ids = list(range(1, len_roi + 1))
            else:
                subset_ids1 = list(range(len_roi + 1, len_iface + len_roi + 1))
                subset_ids2 = list(range(1, len_roi + 1))
                ids1 = get_subset(subset_ids1, cells)
                ids2 = get_subset(subset_ids2, cells)
                return (
                    w,
                    w_subset(w, set(ids1), silence_warnings=True),
                    w_subset(w, set(ids2), silence_warnings=True),
                )

            ids = get_subset(subset_ids, cells)

            return w_subset(w, set(ids), silence_warnings=True)

    def border_neighbors(self, ix: int) -> DistanceBand:
        """Get the neighbors that have links to the other side of the border."""
        w, iface_w, roi_w = self.cell_neighbors(ix, which="all")

        return w_subset(
            w,
            set(roi_w.neighbors.keys()) & set(iface_w.neighbors.keys()),
            silence_warnings=True,
        )

    def plot(
        self,
        show_area: bool = True,
        show_cells: bool = True,
        show_legends: bool = True,
        color: str = None,
        figsize: Tuple[int, int] = (12, 12),
    ) -> None:
        """Plot the slide with areas, cells, and interface areas highlighted.

        Parameters
        ----------
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

        Returns
        -------
            AxesSubplot
        """
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
            )
            leg2 = ax.legend_

        ifaces = self.context2gdf("interface_area")
        ax = ifaces.plot(
            ax=ax,
            color=color,
            column="label",
            alpha=0.7,
            legend=show_legends,
            categorical=True,
            legend_kwds={"loc": "upper left"},
        )
        if show_legends:
            if show_area:
                ax.add_artist(leg1)
            if show_cells:
                ax.add_artist(leg2)

        return ax
