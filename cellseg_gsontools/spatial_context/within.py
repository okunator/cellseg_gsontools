from functools import partial
from typing import Any, Dict, Tuple, Union

import geopandas as gpd
from libpysal.weights import W
from tqdm import tqdm

from ..apply import gdf_apply
from ..graphs import fit_graph
from ..grid import grid_overlay
from ._base import _SpatialContext

__all__ = ["WithinContext"]


class WithinContext(_SpatialContext):
    def __init__(
        self,
        area_gdf: gpd.GeoDataFrame,
        cell_gdf: gpd.GeoDataFrame,
        labels: Union[Tuple[str, ...], str],
        min_area_size: Union[float, str] = None,
        graph_type: str = "delaunay",
        dist_thresh: float = 100.0,
        patch_size: Tuple[int, int] = (256, 256),
        stride: Tuple[int, int] = (256, 256),
        pad: int = None,
        predicate: str = "intersects",
        silence_warnings: bool = True,
    ) -> None:
        """Handle & extract cells from the `cell_gdf` within areas in `area_gdf`.

        Within context is simply a spatial context where the cells are within the areas
        of type `label`. I.e. You can manage objects easily within distinct areas of
        type `label`.

        Parameters
        ----------
        area_gdf : gpd.GeoDataFrame
            A geo dataframe that contains large tissue area polygons enclosing
            smaller cellular objects in `cell_gdf`.
        cell_gdf : gpd.GeoDataFrame
            A geo dataframe that contains small cellular objectss that are enclosed
            by larger tissue areas in `area_gdf`.
        labels : Union[Tuple[str, ...], str]
            The class name(s) of the areas of interest. The objects within these areas
            are extracted. E.g. "cancer" or "stroma".
        min_area_size : float or str, optional
            The minimum area of the objects that are kept. All the objects in the
            `area_gdf` that are larger are kept than `min_area_size`. If None, all the
            areas are kept.
        graph_type : str, default="delaunay"
            The type of the graph to be fitted to the cells inside interfaces. One of:
            "delaunay", "distband", "relative_nhood", "knn"
        dist_thresh : float, default=100.0
            Distance threshold for the length of the network links.
        patch_size : Tuple[int, int], default=(256, 256)
            The size of the grid patches to be fitted on the context.
        stride : Tuple[int, int], default=(256, 256)
            The stride of the sliding window for grid patching.
        pad : int, default=None
            The padding to add to the bounding box on the grid.
        predicate : str, default="intersects"
            The predicate to use for the spatial join when extracting the ROI cells.
            See `geopandas.tools.sjoin`
        silence_warnings : bool, default=True
            Flag, whether to silence all the warnings.

        Attributes
        ----------
        context : Dict[int, Dict[str, Union[gpd.GeoDataFrame, DistanceBand]]]
            A nested dict that contains dicts for each of the distinct areas of type
            `label`. The keys of the outer dict are the indices of these areas.

            The inner dicts contain the unique areas of type `label` and have the keys:
            - 'roi_area' - gpd.GeoDataFrame of the roi area. Roi area is the tissue
                area of type `label`.
            - 'roi_cells' - gpd.GeoDataFrame of the cells that are contained inside
                the roi area.
            - 'roi_network' - libpysal.weights.W spatial weights network of the
                cells inside the roi area. This can be used to extract graph
                features inside the roi.
            - 'roi_grid' - gpd.GeoDataFrame of the grid fitted on the roi area.
                This can be used to extract grid features inside the roi.

        Raises
        ------
            ValueError if `area_gdf` or `cell_gdf` don't contain 'class_name' column.

        Examples
        --------
        Define a within context and plot the cells inside a specific roi area.

        >>> from cellseg_gsontools.spatial_context import WithinContext

        >>> area_gdf = read_gdf("area.json")
        >>> cell_gdf = pre_proc_gdf(read_gdf("cells.json"))
        >>> within_context = WithinContext(
                area_gdf=area_gdf,
                cell_gdf=cell_gdf,
                labels=["area_cin"],
                silence_warnings=True,
                min_area_size=100000.0
            )
        >>> within_context.fit(parallel=False)
        >>> >>> within_context.plot("roi_area", show_legends=True)
        <AxesSubplot: >
        """
        super().__init__(
            area_gdf=area_gdf,
            cell_gdf=cell_gdf,
            labels=labels,
            min_area_size=min_area_size,
            silence_warnings=silence_warnings,
            dist_thresh=dist_thresh,
            predicate=predicate,
            graph_type=graph_type,
            patch_size=patch_size,
            stride=stride,
            pad=pad,
        )

    def fit(
        self,
        verbose: bool = True,
        fit_graph: bool = True,
        fit_grid: bool = True,
        parallel: bool = False,
        num_processes: int = -1,
    ) -> None:
        """Fit the context.

        This sets the `self.context` attribute.

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
                Flag, whether to parallelize the context fitting with pandarallel
                package.
            num_processes : int, default=-1
                The number of processes to use when parallel=True. If -1, this will use
                all the available cores.

        Created Attributes
        ------------------
            context : Dict[int, Dict[str, Union[gpd.GeoDataFrame, libpysal.weights.W]]]
            A nested dict that contains dicts for each of the distinct regions of
            interest areas. The keys of the outer dict are the indices of these
            areas.

            The inner dicts contain the unique interfaces and have the keys:
            - 'roi_area' - gpd.GeoDataFrame of the roi area. Roi area is the tissue
                area of type `label`
            - 'roi_cells' - gpd.GeoDataFrame of the cells that are contained inside
                the roi area.
            - 'roi_network' - libpysal.weights.W spatial weights network of the
                cells inside the roi area. This can be used to extract graph
                features inside the roi.
            - 'roi_grid' - gpd.GeoDataFrame of the grid fitted on the roi area.
                    This can be used to extract grid features inside the roi.
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
                    pbar.set_description(f"Processing roi area: {ix}")

                context_dict[ix] = WithinContext._get_context(
                    ix=ix,
                    spatial_context=self,
                    fit_graph=fit_graph,
                    fit_grid=fit_grid,
                )
        else:
            func = partial(WithinContext._get_context, spatial_context=self)
            context_dict = gdf_apply(
                self.context_area,
                func=func,
                columns=["global_id"],
                parallel=True,
                pbar=verbose,
                num_processes=num_processes,
            ).to_dict()

        self.context = context_dict

    def _get_context(
        ix: int,
        spatial_context: _SpatialContext,
        fit_graph: bool = True,
        fit_grid: bool = True,
    ) -> Dict[str, Any]:
        """Get the context dict of the given index."""
        roi_area = spatial_context.roi(ix)
        roi_cells = spatial_context.roi_cells(
            ix, roi_area, predicate=spatial_context.predicate
        )
        context_dict = {"roi_area": roi_area}
        context_dict["roi_cells"] = roi_cells

        if fit_graph:
            roi_net: W = spatial_context.cell_neighbors(
                roi_cells=roi_cells,
                graph_type=spatial_context.graph_type,
                thresh=spatial_context.dist_thresh,
                predicate=spatial_context.predicate,
            )
            context_dict["roi_network"] = roi_net

        if fit_grid:
            context_dict["roi_grid"] = grid_overlay(
                gdf=roi_area,
                patch_size=spatial_context.patch_size,
                stride=spatial_context.stride,
                pad=spatial_context.pad,
                predicate=spatial_context.predicate,
            )

        return context_dict

    def cell_neighbors(
        self,
        ix: int = None,
        roi_cells: gpd.GeoDataFrame = None,
        thresh: float = 75.0,
        graph_type: str = "delaunay",
        predicate: str = "intersects",
    ) -> W:
        """Create a distance network of the cells.

        Parameters
        ----------
            ix : int, default=None
                The index of the ROI geo-object. Starts from one. If None, `cells`
                must be given.
            roi_cells :  gpd.GeoDataFrame, default=None
                The cells to create the network from. If None, `ix` must be given.
            thresh : float, default=75.0
                Distance threshold for the network.
        Returns
        -------
            W:
                A spatial weights network of the cells.
        """
        # check to not compute the roi cells thing twice
        rcells = roi_cells
        if not isinstance(roi_cells, gpd.GeoDataFrame):
            if ix is not None:
                rcells = self.roi_cells(ix, predicate=predicate)

        if rcells is None or rcells.empty:
            return

        w = fit_graph(
            rcells, type=graph_type, id_col="global_id", thresh=thresh, use_index=False
        )

        return w
