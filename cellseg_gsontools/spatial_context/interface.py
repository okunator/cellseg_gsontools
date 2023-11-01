from functools import partial
from typing import Any, Dict, Optional, Tuple, Union

import geopandas as gpd
import pandas as pd
from libpysal.weights import W, w_subset
from tqdm import tqdm

from ..apply import gdf_apply
from ..graphs import fit_graph, get_border_crosser_links
from ..grid import grid_overlay
from ..utils import set_uid
from ._base import _SpatialContext
from .ops import get_interface_zones

__all__ = ["InterfaceContext"]


class InterfaceContext(_SpatialContext):
    def __init__(
        self,
        area_gdf: gpd.GeoDataFrame,
        cell_gdf: gpd.GeoDataFrame,
        top_labels: Union[Tuple[str, ...], str],
        bottom_labels: Union[Tuple[str, ...], str],
        min_area_size: Union[float, str] = None,
        buffer_dist: int = 200,
        graph_type: str = "distband",
        dist_thresh: float = 50.0,
        patch_size: Tuple[int, int] = (256, 256),
        stride: Tuple[int, int] = (256, 256),
        pad: int = None,
        roi_cell_type: Optional[str] = None,
        iface_cell_type: Optional[str] = None,
        predicate: str = "intersects",
        silence_warnings: bool = True,
    ) -> None:
        """Handle & extract interface regions from the `cell_gdf` and `area_gdf`.

        Interfaces are created by buffering areas of type `top_label` on top of the
        areas of type `bottom_labels` and taking the intersection of the buffered area
        and the `top_label` area. The result interface is a band-like area b/w
        `top_label` and `bottom_label` areas. The bredth of the interface is given by
        the `buffer_dist` param.

        NOTE: `area_gdf` and `cell_gdf` have to contain a column named 'class_name'
            specifying the class of the area/cell.

        Parameters
        ----------
        area_gdf : gpd.GeoDataFrame
            A geo dataframe that contains large tissue area polygons enclosing
            smaller cellular objects in `cell_gdf`.
        cell_gdf : gpd.GeoDataFrame
            A geo dataframe that contains small cellular objectss that are enclosed
            by larger tissue areas in `area_gdf`.
        top_labels : Union[Tuple[str, ...], str]
            The class name(s) of the areas of interest. E.g. "tumor". These areas are
            buffered on top of the areas that have type in `bottom_labels`.For example,
            buffering the tumor area on top of the stroma will get the tumor-stroma
            interface.
        bottom_labels : Union[Tuple[str, ...], str]
            The class name(s) of the area(s) on top of which the buffering is applied.
            Typically you want to buffer at least on top of the stromal area to get e.g.
            tumor-stroma interface. Other options are ofc possible.
        min_area_size : float, optional
            The minimum area of the objects that are kept. All the objects in the
            `area_gdf` that are larger are kept than `min_area_size`. If None, all
            areas are kept.
        buffer_dist : int, default=200
            The radius of the buffer.
        graph_type : str, default="distband"
            The type of the graph to be fitted to the cells inside interfaces. One of:
            "delaunay", "distband", "relative_nhood", "knn"
        dist_thresh : float, default=50.0
            Distance threshold for the length of the network links.
        patch_size : Tuple[int, int], default=(256, 256)
            The size of the grid patches to be fitted on the context.
        stride : Tuple[int, int], default=(256, 256)
            The stride of the sliding window for grid patching.
        pad : int, default=None
            The padding to add to the bounding box on the grid.
        roi_cell_type : str, optional
            The cell type in the roi area. If None, all cells are returned.
        iface_cell_type : str, optional
            The cell type in the interface area. If None, all cells are returned.
        predicate : str, default="intersects"
            The predicate to use for the spatial join when extracting the ROI cells.
            See `geopandas.tools.sjoin`
        silence_warnings : bool, default=True
            Flag, whether to silence all the warnings related to creating the graphs.

        Attributes
        ----------
        context : Dict[int, Dict[str, Union[gpd.GeoDataFrame, libpysal.weights.W]]]
            A nested dict that contains dicts for each of the distinct interface
            area. The keys of the outer dict are the indices of these areas.

            The inner dicts contain the unique interfaces and have the keys:
            - 'roi_area' - gpd.GeoDataFrame of the roi area. Roi area is the tissue
                area of type `label1` that is buffered on top of the area of type
                `label2` to get the interface.
            - 'interface_area', gpd.GeoDataFrame of the interface area. Interface
                area is the area that is the intersection of the roi (`label1`) and
                the area of `label2`.
            - 'roi_cells' - gpd.GeoDataFrame of the cells that are contained inside
                the roi area.
            - 'interface_cells' - gpd.GeoDataFrame of the cells that are contained
                inside the interface area.
            - 'roi_network' - libpysal.weights.W spatial weights network of the
                cells inside the roi area. This can be used to extract graph
                features inside the roi.
            - 'interface_network' - libpysal.weights.W spatial weights network of
                the cells inside the interface area. This can be used to extract
                graph features inside the interface.
            - 'border_network' - libpysal.weights.W spatial weights network of the
                cells at the border of the roi and interface areas. This can be
                used to extract graph features at the border of the roi and
                interface.
            - 'full_network' - libpysal.weights.W spatial weights network of the
                cells inside the union of the roi and interface areas. This can be
                used to extract graph features inside the union of the roi and
                interface.

        Raises
        ------
            ValueError if `area_gdf` or `cell_gdf` don't contain 'class_name' column.

        Examples
        --------
        Define an tumor-stroma interface context and plot the cells inside the
        interface area.
        >>> from cellseg_gsontools.spatial_context import InterfaceContext

        >>> area_gdf = read_gdf("area.json")
        >>> cell_gdf = read_gdf("cells.json")
        >>> interface_context = InterfaceContext(
                area_gdf=area_gdf,
                cell_gdf=cell_gdf,
                top_labels=["area_cin"],
                bottom_labels=["area_stroma"],
                buffer_dist=250.0,
                graph_type="delaunay",
                silence_warnings=True,
                min_area_size=100000.0
            )
        >>> interface_context.fit(parallel=False)
        >>> interface_context.plot("interface_area", show_legends=True)
        <AxesSubplot: >
        """
        super().__init__(
            area_gdf=area_gdf,
            cell_gdf=cell_gdf,
            labels=top_labels,
            min_area_size=min_area_size,
            silence_warnings=silence_warnings,
            dist_thresh=dist_thresh,
            graph_type=graph_type,
            predicate=predicate,
            patch_size=patch_size,
            stride=stride,
            pad=pad,
        )
        self.roi_cell_type = roi_cell_type
        self.interface_cell_type = iface_cell_type
        self.buffer_dist = buffer_dist

        # Get the bottom_label areas
        # (top label areas are in base-class self.context_area)
        if isinstance(bottom_labels, str):
            # a little faster than .isin
            self.context_area2 = area_gdf[area_gdf["class_name"] == bottom_labels]
        else:
            if len(bottom_labels) == 1:
                self.context_area2 = area_gdf[
                    area_gdf["class_name"] == bottom_labels[0]
                ]
            else:
                self.context_area2 = area_gdf[
                    area_gdf["class_name"].isin(bottom_labels)
                ]

        # drop areas smaller than min_area_size
        if min_area_size is not None:
            self.context_area2 = self.context_area2.loc[
                self.context_area2.area >= min_area_size
            ]

        # set global_id and crs
        self.context_area2 = set_uid(self.context_area2, id_col="global_id")
        self.context_area2.set_crs(epsg=4328, inplace=True, allow_override=True)

    def fit(
        self,
        verbose: bool = True,
        fit_graph: bool = True,
        fit_grid: bool = True,
        parallel: bool = False,
        num_processes: int = -1,
    ) -> None:
        """Fit the interface context.

        This sets the `self.context` class attribute.

        NOTE: parallel=True is recommended for only very large gdfs.
        For small gdfs, parallel=False is usually faster.

        Parameters
        ----------
            verbose : bool, default=True
                Flag, whether to use tqdm pbar when creating the interfaces.
            fit_graph : bool, default=True
                Flag, whether to fit the spatial weights networks for the context.
            fit_grid : bool, default=True
                Flag, whether to fit the a grid on the contextes.
            parallel : bool, default=False
                Flag, whether to create the interfaces in parallel.
            num_processes : int, default=-1
                The number of processes to use when parallel=True. If -1, this will use
                all available cores.

        Created Attributes
        -------------------
            context : Dict[int, Dict[str, Union[gpd.GeoDataFrame, libpysal.weights.W]]]
                A nested dict that contains dicts for each of the distinct interface
                area. The keys of the outer dict are the indices of these areas.

                The inner dicts contain the unique interfaces and have the keys:
                - 'roi_area' - gpd.GeoDataFrame of the roi area. Roi area is the tissue
                    area of type `label1` that is buffered on top of the area of type
                    `label2` to get the interface.
                - 'interface_area', gpd.GeoDataFrame of the interface area. Interface
                    area is the area that is the intersection of the roi (`label1`) and
                    the area of `label2`.
                - 'roi_cells' - gpd.GeoDataFrame of the cells that are contained inside
                    the roi area.
                - 'interface_cells' - gpd.GeoDataFrame of the cells that are contained
                    inside the interface area.
                - 'roi_network' - libpysal.weights.W spatial weights network of the
                    cells inside the roi area. This can be used to extract graph
                    features inside the roi.
                - 'interface_network' - libpysal.weights.W spatial weights network of
                    the cells inside the interface area. This can be used to extract
                    graph features inside the interface.
                - 'border_network' - libpysal.weights.W spatial weights network of the
                    cells at the border of the roi and interface areas. This can be
                    used to extract graph features at the border of the roi and
                    interface.
                - 'full_network' - libpysal.weights.W spatial weights network of the
                    cells inside the union of the roi and interface areas. This can be
                    used to extract graph features inside the union of the roi and
                    interface.
                - 'roi_grid' - gpd.GeoDataFrame of the grid fitted on the roi area.
                    This can be used to extract grid features inside the roi.
                - 'interface_grid' - gpd.GeoDataFrame of the grid fitted on the
                    interface area. This can be used to extract grid features inside
                    the interface.
        """
        if not parallel:
            context_dict = {}
            pbar = (
                tqdm(self.context_area.index, total=self.context_area.shape[0])
                if verbose
                else self.context_area.index
            )
            for ix in pbar:
                if verbose:
                    pbar.set_description(f"Processing interface area: {ix}")

                context_dict[ix] = InterfaceContext._get_context(
                    ix=ix,
                    spatial_context=self,
                    fit_graph=fit_graph,
                    fit_grid=fit_grid,
                )
        else:
            func = partial(InterfaceContext._get_context, spatial_context=self)
            context_dict = gdf_apply(
                self.context_area,
                func=func,
                columns=["global_id"],
                parallel=True,
                pbar=verbose,
                num_processes=num_processes,
            ).to_dict()

        self.context = context_dict

    @staticmethod
    def _get_context(
        ix: int,
        spatial_context: _SpatialContext,
        fit_graph: bool = True,
        fit_grid: bool = True,
    ) -> Dict[int, Any]:
        """Get the context dict of the given index."""
        roi_area = spatial_context.roi(ix)
        roi_cells = spatial_context.roi_cells(
            roi_area=roi_area, predicate=spatial_context.predicate
        )
        context_dict = {"roi_area": roi_area}
        context_dict["roi_cells"] = roi_cells

        # interface context
        iface_area = spatial_context.interface(roi_area=roi_area)
        iface_cells = spatial_context.interface_cells(
            iface_area=iface_area, predicate=spatial_context.predicate
        )
        context_dict["interface_area"] = iface_area
        context_dict["interface_cells"] = iface_cells

        # context networks
        if fit_graph:
            net_u, net_r, net_i, net_b = spatial_context.cell_neighbors(
                roi_cells=roi_cells,
                iface_cells=iface_cells,
                graph_type=spatial_context.graph_type,
                thresh=spatial_context.dist_thresh,
                roi_cell_type=spatial_context.roi_cell_type,
                iface_cell_type=spatial_context.interface_cell_type,
                predicate=spatial_context.predicate,
            )
            context_dict["roi_network"] = net_r
            context_dict["interface_network"] = net_i
            context_dict["full_network"] = net_u
            context_dict["border_network"] = net_b

        if fit_grid:
            context_dict["roi_grid"] = grid_overlay(
                gdf=roi_area,
                patch_size=spatial_context.patch_size,
                stride=spatial_context.stride,
                pad=spatial_context.pad,
                predicate=spatial_context.predicate,
            )
            context_dict["interface_grid"] = grid_overlay(
                gdf=iface_area,
                patch_size=spatial_context.patch_size,
                stride=spatial_context.stride,
                pad=spatial_context.pad,
                predicate=spatial_context.predicate,
            )

        return context_dict

    def interface(
        self, ix: int = None, roi_area: gpd.GeoDataFrame = None
    ) -> gpd.GeoDataFrame:
        """Get an interface area of index `ix`.

        Parameters
        ----------
            ix : int, default=None
                The index of the interface area. I.e., the ith interface area.
                If None, `roi_area` must be given.
            roi_area : gpd.GeoDataFrame, default=None
                The roi area. If None, `ix` must be given.
        """
        # check to not compute the roi area twice
        if not isinstance(roi_area, gpd.GeoDataFrame):
            if (roi_area, ix) == (None, None):
                raise ValueError("Either `ix` or `roi_area` must be given.")
            roi_area: gpd.GeoDataFrame = self.roi(ix)

        # Get the intersection of the roi and the area of type `label2`
        iface = get_interface_zones(
            buffer_area=roi_area,
            areas=self.context_area2,
            buffer_dist=self.buffer_dist,
        )

        # If there are many interfaces, dissolve them into one
        if len(iface) > 1:
            iface = iface.dissolve().set_geometry("geometry")

        return iface

    def interface_cells(
        self,
        ix: int = None,
        iface_area: gpd.GeoDataFrame = None,
        predicate: str = "intersects",
    ) -> gpd.GeoDataFrame:
        """Get the cells within the interface area.

        Parameters
        ----------
            ix : int, default=None
                The index of the interface area. I.e., the ith interface area.
                If None, `iface_area` must be given.
            iface_area : gpd.GeoDataFrame, default=None
                The interface area. If None, `ix` must be given.
            predicate : str, default="within"
                The predicate to use for the spatial join when extracting the ROI cells.

        Returns
        -------
            gpd.GeoDataFrame:
                The cells within the interface area.
        """
        # check to not compute the iface area twice
        if not isinstance(iface_area, gpd.GeoDataFrame):
            if (iface_area, ix) == (None, None):
                raise ValueError("Either `ix` or `iface_area` must be given.")
            iface_area: gpd.GeoDataFrame = self.interface(ix)

        if iface_area is None or iface_area.empty:
            return

        return self.get_objs_within(iface_area, self.cell_gdf, predicate=predicate)

    def cell_neighbors(
        self,
        ix: int = None,
        roi_cells: gpd.GeoDataFrame = None,
        iface_cells: gpd.GeoDataFrame = None,
        thresh: float = 100.0,
        graph_type: str = "delaunay",
        roi_cell_type: Optional[str] = None,
        iface_cell_type: Optional[str] = None,
        only_border_crossers: bool = True,
        predicate: str = "intersects",
    ) -> Tuple[W, W, W, W]:
        """Get the spatial weight objects of the cells for different spatial contexts.

        Gets the spatial weights of the cells:
            - inside the roi
            - inside the interface areas.
            - inside the union of the roi and interface areas.
            - crossing the border of the roi and interface areas.

        Parameters
        ----------
            ix : int
                The index of the interface area. I.e., the ith interface area.
            thresh : float, default=75.0
                The distance threshold for the spatial weights.
            graph_type : str, default="delaunay"
                The type of the graph used to get the spatial weights.
            roi_cell_type : str, optional
                The cell type in the roi area. If None, all cells are returned.
            iface_cell_type : str, optional
                The cell type in the interface area. If None, all cells are returned.
            only_border_crossers : bool, default=True
                Whether to return only the links that cross the border between the ROI
                and interface areas.
            predicate : str, default="intersects"
                The predicate to use for the spatial join when extracting the ROI and
                interface cells.

        Returns
        -------
            Tuple[W, W, W, W]:
            The spatial weights of the cells:
            - inside the union of the roi and interface areas.
            - inside the roi
            - inside the interface areas.
            - crossing the border of the roi and interface areas.
        """
        # get roi cells
        rcells = roi_cells
        if not isinstance(roi_cells, gpd.GeoDataFrame):
            if ix is not None:
                rcells = self.roi_cells(ix, predicate=predicate)

        # get interface cells
        icells = iface_cells
        if not isinstance(iface_cells, gpd.GeoDataFrame):
            if ix is not None:
                icells = self.interface_cells(ix, predicate=predicate)

        # return None if neither roi or iface gdf has no cells
        if (icells is None or icells.empty) or (rcells is None or rcells.empty):
            return None, None, None, None

        # subset only specific cell types if needed
        if roi_cell_type is not None:
            rcells = rcells[rcells["class_name"] == roi_cell_type]
        if iface_cell_type is not None:
            icells = icells[icells["class_name"] == iface_cell_type]

        # merge the gdfs to compute union weights
        cells = pd.concat([rcells, icells], sort=False)

        # fit the union graph
        union_weights = fit_graph(
            cells,
            type=graph_type,
            id_col="global_id",
            thresh=thresh,
            use_index=False,
        )

        # Get the weight subsets
        roi_weights = w_subset(
            union_weights, sorted(set(rcells.global_id)), silence_warnings=True
        )
        iface_weights = w_subset(
            union_weights, sorted(set(icells.global_id)), silence_warnings=True
        )

        # get the weights for the nodes that have links crossing the interface border
        border_weights = get_border_crosser_links(
            union_weights, roi_weights, iface_weights, only_border_crossers
        )

        return union_weights, roi_weights, iface_weights, border_weights
