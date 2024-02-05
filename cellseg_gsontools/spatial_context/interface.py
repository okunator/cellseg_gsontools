from functools import partial
from typing import Any, Dict, Tuple, Union

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import psutil
from libpysal.weights import W, w_subset, w_union
from tqdm import tqdm

from cellseg_gsontools.apply import gdf_apply
from cellseg_gsontools.graphs import fit_graph, get_border_crosser_links
from cellseg_gsontools.grid import fit_spatial_grid
from cellseg_gsontools.links import weights2gdf
from cellseg_gsontools.plotting import plot_all
from cellseg_gsontools.spatial_context.backend import _SpatialContextGP
from cellseg_gsontools.utils import set_uid

__all__ = ["InterfaceContext"]


class InterfaceContext:
    """Handle & extract interface regions from the `cell_gdf` and `area_gdf`.

    Interfaces are created by buffering areas of type `top_label` on top of the
    areas of type `bottom_labels` and taking the intersection of the buffered area
    and the `top_label` area. The result interface is a band-like area b/w
    `top_label` and `bottom_label` areas. The bredth of the interface is given by
    the `buffer_dist` param.

    Note:
        `area_gdf` and `cell_gdf` have to contain a column named 'class_name'

    Parameters:
        area_gdf (gpd.GeoDataFrame):
            A geo dataframe that contains large tissue area polygons enclosing
            the smaller cellular objects in `cell_gdf`.
        cell_gdf (gpd.GeoDataFrame):
            A geo dataframe that contains small cellular objects that are
            enclosed by larger tissue areas in `area_gdf`.
        top_labels (Union[Tuple[str, ...], str]):
            The class name(s) of the areas of interest. E.g. "tumor". These areas
            are buffered on top of the areas that have type in `bottom_labels`. For
            example, buffering the tumor area on top of the stroma will get the
            tumor-stroma interface.
        bottom_labels (Union[Tuple[str, ...], str]):
            The class name of the area on top of which the buffering is applied.
            Typically you want to buffer at least on top of the stromal area to get
            e.g. tumor-stroma interface. Other options are ofc possible.
        min_area_size (float or str, optional):
            The minimum area of the objects that are kept. All the objects in
            the `area_gdf` that are larger are kept than `min_area_size`. If
            None, all the areas are kept. Defaults to None.
        graph_type (str):
            The type of the graph to be fitted to the cells inside interfaces.
            One of: "delaunay", "distband", "relative_nhood", "knn".
        dist_thresh (float):
            Distance threshold for the length of the network links.
        grid_type (str):
            The type of the grid to be fitted on the roi areas. One of:
            "square", "hex".
        patch_size (Tuple[int, int]):
            The size of the grid patches to be fitted on the context. This is
            used when `grid_type='square'`.
        stride (Tuple[int, int]):
            The stride of the sliding window for grid patching. This is used
            when `grid_type='square'`.
        pad (int):
            The padding to add to the bounding box on the grid. This is used
            when `grid_type='square'`.
        resolution (int):
            The resolution of the h3 hex grid. This is used when
            `grid_type='hex'`.
        predicate (str):
            The predicate to use for the spatial join when extracting the ROI
            cells. See `geopandas.tools.sjoin`
        silence_warnings (bool):
            Flag, whether to silence all the warnings.
        parallel (bool):
            Flag, whether to parallelize the context fitting. If
            `backend == "geopandas"`, the parallelization is implemented with
            `pandarallel` package. If `backend == "spatialpandas"`, or
            `backend == "dask-geopandas"` the parallelization is implemented
            with Dask library.
        num_processes (int):
            The number of processes to use when parallel=True. If -1, this
            will use all the available cores.
        backend (str):
            The backend to use for the spatial context. One of "geopandas",
            "spatialpandas" "dask-geopandas". "spatialpandas" or
            "dask-geopandas" is recommended for gdfs that may contain huge
            polygons.

    Attributes:
        context (Dict[int, Dict[str, Union[gpd.GeoDataFrame, libpysal.weights.W]]]):
            A nested dict that contains dicts for each of the distinct ROIs
            of type `top_labels` and the interfaces b/w areas of type `bottom_labels`.
            The keys of the outer dict are the indices of these areas.
            The inner dicts contain the keys:

            - `roi_area`- `gpd.GeoDataFrame`: of the roi area. Roi area is the tissue
                    area(s) of type `top_labels` that is buffered on top of the area
                    of type `bottom_labels` to get the interface.
            - `roi_cells` - `gpd.GeoDataFrame`: of the cells that are contained
                    inside the `roi_area`.
            - `roi_network` - `libpysal.weights.W`: spatial weights network of
                    the cells inside the `roi_area`. This can be used to extract
                    graph features inside the `roi_area`.
            - `roi_grid` - `gpd.GeoDataFrame`: of the grid fitted on the `roi_area`.
                    This can be used to extract grid features inside the `roi_area`.
            - `interface_area` - `gpd.GeoDataFrame`:the interface area. Interface
                    area is the area that is the intersection of the buffered
                    `roi_area` (`top_labels`) and the area of type `bottom_labels`.
            - `interface_network` - `libpysal.weights.W`: spatial weights network of
                    the cells inside the `interface_area`.
            - `border_network` - `libpysal.weights.W`: spatial weights network of the
                    cells at the border of the roi and interface areas.
            - `full_network` - `libpysal.weights.W`: spatial weights network of the
                    cells inside the union of the roi and interface areas.

    Raises:
        ValueError: if `area_gdf` or `cell_gdf` don't contain 'class_name' column.
    """

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
        grid_type: str = "square",
        patch_size: Tuple[int, int] = (256, 256),
        stride: Tuple[int, int] = (256, 256),
        pad: int = None,
        resolution: int = 9,
        predicate: str = "intersects",
        silence_warnings: bool = True,
        parallel: bool = False,
        num_processes: int = -1,
        backend: str = "geopandas",
    ) -> None:
        self.backend_name = backend
        if backend == "geopandas":
            self.backend = _SpatialContextGP()
        # elif backend == "spatialpandas":
        #     self.backend = _SpatialContextSP()
        # elif backend == "dask-geopandas":
        #     self.backend = _SpatialContextDGP()
        else:
            raise ValueError(
                f"Unknown backend: {backend}. "
                "Allowed: 'spatialpandas', 'geopandas', 'dask-geopandas'"
            )

        # check if the 'class_name' column is present
        self.backend.check_columns(area_gdf, cell_gdf)

        # set up the attributes
        self.buffer_dist = buffer_dist
        self.min_area_size = min_area_size
        self.dist_thresh = dist_thresh
        self.graph_type = graph_type
        self.patch_size = patch_size
        self.stride = stride
        self.pad = pad
        self.silence_warnings = silence_warnings
        self.top_labels = top_labels
        self.bottom_labels = bottom_labels
        self.predicate = predicate
        self.parallel = parallel
        self.num_processes = num_processes
        self.grid_type = grid_type
        self.resolution = resolution

        # set to geocentric cartesian crs. (unit is metre not degree as by default)
        # helps to avoid warning flooding
        self.cell_gdf = set_uid(cell_gdf, id_col="global_id")
        self.cell_gdf.set_crs(epsg=4328, inplace=True, allow_override=True)

        # cache the full area gdf for plotting
        self.area_gdf = area_gdf
        self.area_gdf.set_crs(epsg=4328, inplace=True, allow_override=True)

        # filter small areas and tissue types for the top and bottom labels
        self.context_area = self.backend.filter_areas(
            self.area_gdf, top_labels, min_area_size
        )
        self.context_area = set_uid(self.context_area, id_col="global_id")

        self.context_area2 = self.backend.filter_areas(
            self.area_gdf, bottom_labels, min_area_size
        )
        self.context_area2 = set_uid(self.context_area2, id_col="global_id")

        # set up cpu count
        if parallel:
            self.cpus = (
                psutil.cpu_count(logical=False)
                if self.num_processes == -1 or self.num_processes is None
                else self.num_processes
            )
        else:
            self.cpus = 1

        # convert the gdfs to the backend format
        self.context_area = self.backend.convert_area_gdf(self.context_area)
        self.context_area2 = self.backend.convert_area_gdf(self.context_area2)

        self.cell_gdf = self.backend.convert_cell_gdf(
            self.cell_gdf, parallel=parallel, n_partitions=self.cpus
        )

    def __getattr__(self, name):
        """Get attribute."""
        return self.backend.__getattribute__(name)

    def fit(
        self,
        verbose: bool = True,
        fit_graph: bool = True,
        fit_grid: bool = True,
    ) -> None:
        """Fit the interface context.

        This sets the `self.context` class attribute.

        Parameters:
            verbose (bool):
                Flag, whether to use tqdm pbar when creating the interfaces.
            fit_graph (bool):
                Flag, whether to fit the spatial weights networks for the
                context.
            fit_grid (bool):
                Flag, whether to fit the a grid on the contextes.

        Examples:
            Define an tumor-stroma interface context and plot the cells inside the
            interface area.

            >>> from cellseg_gsontools.backend import InterfaceContext
            >>> area_gdf = read_gdf("area.json")
            >>> cell_gdf = read_gdf("cells.json")
            >>> interface_context = InterfaceContext(
            ...     area_gdf=area_gdf,
            ...     cell_gdf=cell_gdf,
            ...     top_labels=["area_cin"],
            ...     bottom_labels=["area_stroma"],
            ...     buffer_dist=250.0,
            ...     graph_type="delaunay",
            ...     silence_warnings=True,
            ...     min_area_size=100000.0,
            ... )
            >>> interface_context.fit(parallel=False)
            >>> interface_context.plot("interface_area", show_legends=True)
            <AxesSubplot: >
        """
        get_context_func = partial(
            InterfaceContext._get_context,
            backend=self.backend,
            context_area=self.context_area,
            context_area2=self.context_area2,
            cell_gdf=self.cell_gdf,
            fit_network=fit_graph,
            fit_grid=fit_grid,
            grid_type=self.grid_type,
            resolution=self.resolution,
            predicate=self.predicate,
            buffer_dist=self.buffer_dist,
            silence_warnings=self.silence_warnings,
            graph_type=self.graph_type,
            dist_thresh=self.dist_thresh,
            patch_size=self.patch_size,
            stride=self.stride,
            pad=self.pad,
            parallel=self.parallel,
            num_processes=self.cpus,
        )

        if self.backend_name == "geopandas" and self.parallel:
            # run in parallel
            context_dict = gdf_apply(
                self.context_area,
                func=get_context_func,
                columns=["global_id"],
                parallel=True,
                pbar=verbose,
                num_processes=self.cpus,
            ).to_dict()
        else:
            context_dict = {}
            pbar = (
                tqdm(self.context_area.index, total=self.context_area.shape[0])
                if verbose
                else self.context_area.index
            )

            for ix in pbar:
                if verbose:
                    pbar.set_description(f"Processing roi area: {ix}")

                if self.backend_name == "dask-geopandas" and self.parallel:
                    get_context_func = partial(
                        get_context_func, cell_gdf_dgp=self.backend.cell_gdf_dgp
                    )

                context_dict[ix] = get_context_func(ix=ix)

        self.context = context_dict

    @staticmethod
    def _get_context(
        ix: int,
        backend,
        context_area: gpd.GeoDataFrame,
        context_area2: gpd.GeoDataFrame,
        cell_gdf: gpd.GeoDataFrame,
        buffer_dist: int = 200,
        fit_network: bool = True,
        fit_grid: bool = True,
        grid_type: str = "square",
        resolution: int = 9,
        predicate: str = "intersects",
        silence_warnings: bool = True,
        graph_type: str = "distband",
        dist_thresh: float = 75.0,
        patch_size: Tuple[int, int] = (256, 256),
        stride: Tuple[int, int] = (256, 256),
        pad: int = None,
        parallel: bool = False,
        num_processes: int = None,
        **kwargs,
    ) -> Dict[int, Any]:
        """Get the context dict of the given index."""
        roi_area: gpd.GeoDataFrame = backend.roi(ix=ix, context_area=context_area)
        roi_cells: gpd.GeoDataFrame = backend.roi_cells(
            roi_area=roi_area,
            cell_gdf=cell_gdf,
            predicate=predicate,
            silence_warnings=silence_warnings,
            parallel=parallel,
            num_processes=num_processes,
            **kwargs,
        )
        context_dict = {"roi_area": roi_area}
        context_dict["roi_cells"] = roi_cells

        # interface context
        iface_area: gpd.GeoDataFrame = backend.interface(
            top_roi_area=roi_area, bottom_gdf=context_area2, buffer_dist=buffer_dist
        )
        iface_cells: gpd.GeoDataFrame = backend.roi_cells(
            roi_area=iface_area,
            cell_gdf=cell_gdf,
            predicate=predicate,
            silence_warnings=silence_warnings,
            parallel=parallel,
            num_processes=num_processes,
            **kwargs,
        )
        context_dict["interface_area"] = iface_area
        context_dict["interface_cells"] = iface_cells

        # context networks
        if fit_network:
            if (iface_cells is None or iface_cells.empty) or (
                roi_cells is None or roi_cells.empty
            ):
                context_dict["full_network"] = None
                context_dict["roi_network"] = None
                context_dict["interface_network"] = None
                context_dict["border_network"] = None
            else:
                # merge the gdfs to compute union weights
                cells = pd.concat([roi_cells, iface_cells], sort=False)

                # fit the union graph
                context_dict["full_network"] = fit_graph(
                    cells,
                    type=graph_type,
                    id_col="global_id",
                    thresh=dist_thresh,
                    use_index=False,
                )

                # Get the weight subsets
                context_dict["roi_network"] = w_subset(
                    context_dict["full_network"],
                    sorted(set(roi_cells.global_id)),
                    silence_warnings=silence_warnings,
                )
                context_dict["interface_network"] = w_subset(
                    context_dict["full_network"],
                    sorted(set(iface_cells.global_id)),
                    silence_warnings=silence_warnings,
                )

                # get the weights 4 the nodes that have links crossing the iface border
                context_dict["border_network"] = get_border_crosser_links(
                    union_weights=context_dict["full_network"],
                    roi_weights=context_dict["roi_network"],
                    iface_weights=context_dict["interface_network"],
                    only_border_crossers=True,
                )

        if fit_grid:
            if grid_type == "hex":
                kwargs = {"resolution": resolution}
            else:
                kwargs = {
                    "patch_size": patch_size,
                    "stride": stride,
                    "pad": pad,
                    "predicate": predicate,
                }

            context_dict["roi_grid"] = fit_spatial_grid(
                gdf=roi_area, grid_type=grid_type, **kwargs
            )
            context_dict["interface_grid"] = fit_spatial_grid(
                gdf=iface_area, grid_type=grid_type, **kwargs
            )

        return context_dict

    def context2weights(self, key: str) -> W:
        """Merge the networks of type `key` into one spatial weights obj.

        Parameters:
            key (str):
                The key of the context dictionary that contains the spatial
                weights to be merged. One of "roi_network", "full_network",
                "interface_network", "border_network"

        Returns:
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
        """Merge the GeoDataFrames of type `key` into one geodataframe.

        Note:
            Returns None if no data is found.

        Parameters:
            key (str):
                The key of the context dictionary that contains the data to be converted
                to gdf. One of "roi_area", "roi_cells", "interface_area", "roi_grid",
                "interface_grid", "interface_cells", "roi_interface_cells"

        Returns:
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

        return (
            gdf.reset_index(level=0, names="label")
            .drop_duplicates("geometry")
            .set_geometry("geometry")
        )

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
        """Plot the context with areas, cells, and interface areas highlighted.

        Parameters:
            key (str):
                The key of the context dictionary that contains the data to be plotted.
                One of "roi_area",
            network_key (str):
                The key of the context dictionary that contains the spatial weights to
                be plotted. One of "roi_network"
            grid_key (str):
                The key of the context dictionary that contains the grid to be plotted.
                One of "roi_grid"
            show_legends (bool):
                Flag, whether to include legends for each in the plot.
            color (str):
                A color for the interfaces or rois, Ignored if `show_legends=True`.
            figsize (Tuple[int, int]):
                Size of the figure.
            **kwargs (Dict[str, Any])]):
                Extra keyword arguments passed to the `plot` method of the
                geodataframes.

        Returns:
            AxesSubplot

        Examples:
            Plot the tumor-stroma areas.

            >>> from cellseg_gsontools.spatial_context import InterfaceContext
            >>> cells = read_gdf("cells.feather")
            >>> areas = read_gdf("areas.feather")
            >>> ts_iface = InterfaceContext(
            ...     area_gdf=areas,
            ...     cell_gdf=cells,
            ...     top_labels="tumor",
            ...     bottom_labels="stroma",
            ... )
            >>> ts_iface.fit(verbose=False)
            >>> ts_iface.plot("interface_area", show_legends=True)
            <AxesSubplot: >
        """
        allowed = ("roi_area", "interface_area")
        if key not in allowed:
            raise ValueError(f"Illegal key. Got: {key}. Allowed: {allowed}")

        context_gdf = self.context2gdf(key)

        grid_gdf = None
        if grid_key is not None:
            grid_gdf = self.context2gdf(grid_key)

        network_gdf = None
        if network_key is not None:
            edge_kws = edge_kws or {}
            w = self.context2weights(network_key)
            network_gdf = weights2gdf(self.cell_gdf, w)

        return plot_all(
            cell_gdf=self.cell_gdf.set_geometry("geometry"),
            area_gdf=self.area_gdf.set_geometry("geometry"),
            context_gdf=context_gdf,
            grid_gdf=grid_gdf,
            network_gdf=network_gdf,
            show_legends=show_legends,
            color=color,
            figsize=figsize,
            edge_kws=edge_kws,
            **kwargs,
        )
