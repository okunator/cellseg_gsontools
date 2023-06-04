from typing import Union

import geopandas as gpd
from libpysal.weights import DistanceBand
from tqdm import tqdm

from ..graphs import fit_graph
from ._base import _SpatialContext

__all__ = ["WithinContext"]


class WithinContext(_SpatialContext):
    def __init__(
        self,
        area_gdf: gpd.GeoDataFrame,
        cell_gdf: gpd.GeoDataFrame,
        label: str,
        min_area_size: Union[float, str] = None,
        q: float = 25.0,
        graph_type: str = "delaunay",
        dist_thresh: float = 100.0,
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
        label : str
            The class name of the areas of interest. E.g. "cancer".
        min_area_size : float or str, optional
            The minimum area of the objects that are kept. All the objects in the
            `area_gdf` that are larger are kept than `min_area_size`. Can be either
            a float or one of: "mean", "median", "quantile" None. If None, all the
            areas are kept.
        q : float, default=25.0
            The quantile. I.e. areas smaller than `q` in the `area_gdf` are dropped.
            This is only used if `min_area_size = "quantile"`, ignored otherwise.
        graph_type : str, default="delaunay"
            The type of the graph to be fitted to the cells inside interfaces. One of:
            "delaunay", "distband", "relative_nhood", "knn"
        dist_thresh : float, default=100.0
            Distance threshold for the length of the network links.
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
                label="area_cin",
                silence_warnings=True,
                min_area_size=100000.0
            )

        >>> ix = 1
        >>> ax = within_context.context_area.plot(figsize=(10, 10), alpha=0.5)
        >>> within_context.context[ix]["roi_cells"].plot(
                ax=ax, column="class_name", categorical=True, legend=True
            )
        <AxesSubplot: >
        """
        super().__init__(
            area_gdf=area_gdf,
            cell_gdf=cell_gdf,
            label=label,
            min_area_size=min_area_size,
            q=q,
            silence_warnings=silence_warnings,
            dist_thresh=dist_thresh,
            graph_type=graph_type,
        )

    def fit(self, verbose: bool = True) -> None:
        """Fit the within regions.

        NOTE: This only sets the `context_dict` attribute.

        Parameters
        ----------
            verbose : bool, default=True
                Flag, whether to use tqdm pbar when creating the interfaces.

        Returns
        -------
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
        """
        context_dict = {}
        pbar = (
            tqdm(self.context_area.index, total=self.context_area.shape[0])
            if verbose
            else self.context_area.index
        )

        for ix in pbar:
            if verbose:
                pbar.set_description(f"Processing roi area: {ix}")
            context_dict[ix] = {"roi_area": self.roi(ix)}
            context_dict[ix]["roi_cells"] = self.roi_cells(ix)
            context_dict[ix]["roi_network"] = self.cell_neighbors(
                ix, graph_type=self.graph_type, thresh=self.dist_thresh
            )

        self.context = context_dict

    def cell_neighbors(
        self,
        ix: int,
        thresh: float = 75.0,
        graph_type: str = "delaunay",
    ) -> DistanceBand:
        """Create a distance network of the cells.

        Parameters
        ----------
            ix : int
                The index of the ROI geo-object. Starts from one.
            thresh : float, default=75.0
                Distance threshold for the network.
        Returns
        -------
            DistanceBand:
                Either one distance network or all three possible distance networks.
        """
        cells = self.roi_cells(ix)
        if cells.empty or cells is None:
            return None

        w = fit_graph(
            cells,
            type=graph_type,
            id_col="global_id",
            thresh=thresh,
        )

        return w
