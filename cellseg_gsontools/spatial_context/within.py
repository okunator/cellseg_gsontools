from functools import partial
from typing import Any, Dict, Tuple, Union

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import psutil
from libpysal.weights import W, w_subset, w_union
from tqdm import tqdm

from cellseg_gsontools.apply import gdf_apply
from cellseg_gsontools.graphs import fit_graph
from cellseg_gsontools.grid import grid_overlay
from cellseg_gsontools.links import weights2gdf
from cellseg_gsontools.plotting import plot_all
from cellseg_gsontools.spatial_context.backend import (
    _SpatialContextDGP,
    _SpatialContextGP,
    _SpatialContextSP,
)
from cellseg_gsontools.utils import set_uid

__all__ = ["WithinContext"]


class WithinContext:
    def __init__(
        self,
        area_gdf: gpd.GeoDataFrame,
        cell_gdf: gpd.GeoDataFrame,
        labels: Union[Tuple[str, ...], str],
        min_area_size: Union[float, str] = None,
        graph_type: str = "distband",
        dist_thresh: float = 100.0,
        patch_size: Tuple[int, int] = (256, 256),
        stride: Tuple[int, int] = (256, 256),
        pad: int = None,
        predicate: str = "intersects",
        silence_warnings: bool = True,
        parallel: bool = False,
        num_processes: int = -1,
        backend: str = "geopandas",
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
        graph_type : str, default="distband"
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
        parallel : bool, default=False
            Flag, whether to parallelize the context fitting. If backend == "geopandas",
            the parallelization is implemented with pandarallel package.
            If backend == "spatialpandas", the parallelization is implemented with Dask
        num_processes : int, default=-1
            The number of processes to use when parallel=True. If -1, this will use
            all the available cores.
        backend : str, default="geopandas"
            The backend to use for the spatial context. One of "geopandas",
            "spatialpandas" "dask-geopandas". "spatialpandas" or "dask-geopandas" is
            recommended for large gdfs.

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

        >>> from cellseg_gsontools.spatial_context import WithinContextSP

        >>> area_gdf = read_gdf("area.json")
        >>> cell_gdf = read_gdf("cells.json")
        >>> within_context = WithinContextSP(
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
        self.backend_name = backend
        if backend == "spatialpandas":
            self.backend = _SpatialContextSP()
        elif backend == "geopandas":
            self.backend = _SpatialContextGP()
        elif backend == "dask-geopandas":
            self.backend = _SpatialContextDGP()
        else:
            raise ValueError(
                f"Unknown backend: {backend}. "
                "Allowed: 'spatialpandas', 'geopandas', 'dask-geopandas'"
            )

        # check if the 'class_name' column is present
        self.backend.check_columns(area_gdf, cell_gdf)

        # set up the attributes
        self.min_area_size = min_area_size
        self.dist_thresh = dist_thresh
        self.graph_type = graph_type
        self.patch_size = patch_size
        self.stride = stride
        self.pad = pad
        self.silence_warnings = silence_warnings
        self.labels = labels
        self.predicate = predicate
        self.parallel = parallel
        self.num_processes = num_processes

        # set to geocentric cartesian crs. (unit is metre not degree as by default)
        # helps to avoid warning flooding
        self.cell_gdf = set_uid(cell_gdf, id_col="global_id")
        self.cell_gdf.set_crs(epsg=4328, inplace=True, allow_override=True)

        # cache the full area gdf for plotting
        self.area_gdf = area_gdf
        self.area_gdf.set_crs(epsg=4328, inplace=True, allow_override=True)

        # filter small areas and tissue types of interest for the tissue context gdf
        self.context_area = self.backend.filter_areas(
            self.area_gdf, labels, min_area_size
        )
        self.context_area = set_uid(self.context_area, id_col="global_id")

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
        """Fit the context.

        This sets the `self.context` attribute.

        Parameters
        ----------
        verbose : bool, default=True
            Flag, whether to use tqdm pbar when creating the interfaces.
        fit_graph : bool, default=True
            Flag, whether to fit the spatial weights networks for the context.
        fit_grid : bool, default=True
            Flag, whether to fit the a grid on the contextes.

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
        get_context_func = partial(
            WithinContext._get_context,
            backend=self.backend,
            context_area=self.context_area,
            cell_gdf=self.cell_gdf,
            fit_network=fit_graph,
            fit_grid=fit_grid,
            predicate=self.predicate,
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
        cell_gdf: gpd.GeoDataFrame,
        fit_network: bool = True,
        fit_grid: bool = True,
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
    ) -> Dict[str, Any]:
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

        if fit_network:
            roi_net: W = fit_graph(
                gdf=roi_cells,
                type=graph_type,
                id_col="global_id",
                thresh=dist_thresh,
                use_index=False,
            )
            context_dict["roi_network"] = roi_net

        if fit_grid:
            context_dict["roi_grid"] = grid_overlay(
                gdf=roi_area,
                patch_size=patch_size,
                stride=stride,
                pad=pad,
                predicate=predicate,
            )

        return context_dict

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
        """Plot the slide with areas, cells, and interface areas highlighted.

        Parameters
        ----------
            key : str
                The key of the context dictionary that contains the data to be plotted.
                One of "roi_area",
            network_key : str, optional
                The key of the context dictionary that contains the spatial weights to
                be plotted. One of "roi_network"
            grid_key : str, optional
                The key of the context dictionary that contains the grid to be plotted.
                One of "roi_grid"
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
        allowed = "roi_area"
        if key != allowed:
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
