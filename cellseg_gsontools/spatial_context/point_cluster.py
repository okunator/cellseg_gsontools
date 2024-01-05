from typing import Tuple, Union

import geopandas as gpd
import numpy as np
from libpysal.cg import alpha_shape_auto

from ..clustering import cluster_points
from .within import WithinContext

__all__ = ["PointClusterContext"]


class PointClusterContext(WithinContext):
    """Handle & extract dense point clusters from `cell_gdf`.

    Point-clusters are dense regions of points. This context is useful when you
    want to extract dense regions of points of type `label` from `cell_gdf` such as
    immune-cell clusters. The clusters are extracted using one of the clustering
    methods: "dbscan", "adbscan", "optics" after which the clusters are converted
    to polygons/areas using alpha-shapes.

    Note:
        This class inherits from `WithinContext` and thus has all the methods
        and attributes of that class.

    Note:
        `cell_gdf` has to contain a column named 'class_name'

    Parameters:
        cell_gdf (gpd.GeoDataFrame):
            A geo dataframe that contains small cellular objects that are
            enclosed by larger tissue areas in `area_gdf`.
        labels (Union[Tuple[str, ...], str]):
            The class name(s) of the areas of interest. The objects within
            these areas are extracted. E.g. "cancer" or "stroma".
        cluster_method (str):
            The clustering method used to extract the point-clusters. One of:
            "dbscan", "adbscan", "optics"
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
        **kwargs (Dict[str, Any]):
            Arbitrary key-word arguments passed to the clustering methods.

    Attributes:
        context (Dict[int, Dict[str, Union[gpd.GeoDataFrame, libpysal.weights.W]]]):
            A nested dict that contains dicts for each of the distinct clusters.
            The keys of the outer dict are the indices of these areas. The inner dicts
            contain the keys:

            - `roi_area`- `gpd.GeoDataFrame`:  the alpha shape of a cell cluster.
            - `roi_cells` - `gpd.GeoDataFrame`: the cells that are contained
                    inside the `roi_area`.
            - `roi_network` - `libpysal.weights.W`: spatial weights network of
                    the cells inside the `roi_area`. This can be used to extract
                    graph features inside the clusters.
            - `roi_grid` - `gpd.GeoDataFrame`: of the grid fitted on the `roi_area`.
                    This can be used to extract grid features inside the clusters.

    Raises:
        ValueError:
            if `cell_gdf` don't contain 'class_name' column.

    Examples:
        Create a point cluster context and plot the cells inside one cluster area.
        >>> from cellseg_gsontools.spatial_context import ClusterContext
        >>> cell_gdf = pre_proc_gdf(read_gdf("cells.json"))
        >>> cluster_context = PointClusterContext(
        ...     cell_gdf=cell_gdf,
        ...     labels=["inflammatory"],
        ...     cluster_method="adbscan",
        ...     silence_warnings=True,
        ... )
        >>> cluster_context.fit(parallel=False, fit_graph=False)
        >>> cluster_context.plot("roi_area", show_legends=True)
        <AxesSubplot: >
    """

    def __init__(
        self,
        cell_gdf: gpd.GeoDataFrame,
        labels: Union[Tuple[str, ...], str],
        cluster_method: str = "dbscan",
        min_area_size: Union[float, str] = None,
        graph_type: str = "distband",
        dist_thresh: float = 100.0,
        grid_type: str = "square",
        patch_size: Tuple[int, int] = (256, 256),
        stride: Tuple[int, int] = (256, 256),
        pad: int = None,
        resolution: int = 9,
        predicate: str = "intersects",
        silence_warnings: bool = True,
        n_jobs: int = -1,
        parallel: bool = False,
        num_processes: int = -1,
        backend: str = "geopandas",
        **kwargs,
    ) -> None:
        cell_gdf.set_crs(epsg=4328, inplace=True, allow_override=True)
        area_gdf = self.run_clustering(
            cell_gdf, labels, cluster_method, n_jobs, **kwargs
        )

        if isinstance(labels, (list, tuple)):
            if len(labels) == 1:
                labels = labels[0]
            else:
                labels = "-".join(labels)

        super().__init__(
            area_gdf=area_gdf,
            cell_gdf=cell_gdf,
            labels=labels,
            min_area_size=min_area_size,
            dist_thresh=dist_thresh,
            graph_type=graph_type,
            patch_size=patch_size,
            stride=stride,
            pad=pad,
            predicate=predicate,
            silence_warnings=silence_warnings,
            parallel=parallel,
            num_processes=num_processes,
            backend=backend,
            grid_type=grid_type,
            resolution=resolution,
        )

    def run_clustering(
        self,
        cell_gdf: gpd.GeoDataFrame,
        labels: Union[Tuple[str, ...], str],
        cluster_method: str,
        n_jobs: int,
        **kwargs,
    ) -> gpd.GeoDataFrame:
        """Run clustering on the cells and convert the clusters to areas.

        Parameters:
            cell_gdf (gpd.GeoDataFrame):
                A geo dataframe that contains the cell/nuclei objects.
            labels (Union[Tuple[str, ...], str]):
                The class name(s of the objects of interest. E.g. "cancer", "immune".
            cluster_method (str):
                The clustering method used to extract the point-clusters. One of:
                "dbscan", "adbscan", "optics"
            n_jobs (int):
                The number of processes to use in clustering.
            **kwargs (Dict[str, Any]):
                Arbitrary key-word arguments passed to the clustering methods.

        Returns:
            gpd.GeoDataFrame:
                A gdf containing the areas of the clusters.
        """
        # cluster the cells
        if isinstance(labels, str):
            # a little faster than .isin
            cells = cell_gdf[cell_gdf["class_name"] == labels].copy()
        else:
            if len(labels) == 1:
                cells = cell_gdf[cell_gdf["class_name"] == labels[0]].copy()
            else:
                cells = cell_gdf[cell_gdf["class_name"].isin(labels)].copy()

        cells = cluster_points(cells, method=cluster_method, n_jobs=n_jobs, **kwargs)

        # convert the clusters to areas with alpha-shapes
        labs = cells["labels"].unique()
        area_data = {"geometry": []}
        for lab in labs:
            if lab == str(-1) or lab == int(-1):
                continue

            if isinstance(lab, str):
                lab = str(lab)

            c = cells[cells["labels"] == lab]
            coords = np.vstack([c.centroid.x, c.centroid.y]).T
            alpha_shape = alpha_shape_auto(coords, step=15)
            area_data["geometry"].append(alpha_shape.buffer(10.0).buffer(-20.0))

        area_gdf = gpd.GeoDataFrame(area_data)

        if isinstance(labels, (list, tuple)):
            if len(labels) == 1:
                labels = labels[0]
            else:
                labels = "-".join(labels)

        area_gdf["class_name"] = [labels] * len(area_gdf)

        return area_gdf
