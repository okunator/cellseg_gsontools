from typing import Union

import geopandas as gpd
import numpy as np
from libpysal.cg import alpha_shape_auto

from ..clustering import cluster_points
from .within import WithinContext

__all__ = ["PointClusterContext"]


class PointClusterContext(WithinContext):
    def __init__(
        self,
        cell_gdf: gpd.GeoDataFrame,
        label: str,
        cluster_method: str = "dbscan",
        min_area_size: Union[float, str] = None,
        q: float = 25.0,
        graph_type: str = "delaunay",
        dist_thresh: float = 100.0,
        predicate: str = "intersects",
        silence_warnings: bool = True,
        n_jobs: int = -1,
        **kwargs,
    ) -> None:
        """Handle & extract dense point clusters from `cell_gdf`.

        Point-clusters are dense regions of points. This context is useful when you
        want to extract dense regions of points of type `label` from `cell_gdf` such as
        immune-cell clusters. The clusters are extracted using one of the clustering
        methods: "dbscan", "adbscan", "optics" after which the clusters are converted
        to polygons/areas using alpha-shapes.

        Parameters
        ----------
        cell_gdf : gpd.GeoDataFrame
            A geo dataframe that contains cell objects.
        label : str
            The class name of the objects of interest. E.g. "cancer", "immune".
        cluster_method : str, default="dbscan"
            The clustering method used to extract the point-clusters. One of:
            "dbscan", "adbscan", "optics"
        min_area_size : float or str, optional
            The minimum area of the clusters that are kept. All the clusters in the
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
        predicate : str, default="within"
            The predicate to use for the spatial join when extracting the ROI cells.
            See `geopandas.tools.sjoin`
        silence_warnings : bool, default=True
            Flag, whether to silence all the warnings.
        n_jobs : int,default=-1
            Number of jobs used when clustering. None=1, and -1 means all available.
        **kwargs:
            Arbitrary key-word arguments passed to the clustering methods.

        Attributes
        ----------
        context : Dict[int, Dict[str, Union[gpd.GeoDataFrame, DistanceBand]]]
            A nested dict that contains dicts for each of the distinct point-cluster
            areas. The keys of the outer dict are the indices of these areas.

            The inner dicts contain the unique point-clusters and have the keys:
            - 'roi_area' - gpd.GeoDataFrame of the cluster areas.
            - 'roi_cells' - gpd.GeoDataFrame of the cells that are contained inside
                the clusters.
            - 'roi_network' - libpysal.weights.W spatial weights network of the
                cells inside the cluster areas. This can be used to extract graph
                features inside the clusters.

        Raises
        ------
            ValueError if `cell_gdf` doesn't contain 'class_name' column.

        Examples
        --------
        Create a point cluster context and plot the cells inside one cluster area.

        >>> from cellseg_gsontools.spatial_context import ClusterContext

        >>> cell_gdf = pre_proc_gdf(read_gdf("cells.json"))
        >>> cluster_context = PointClusterContext(
                cell_gdf=cell_gdf,
                label="inflammatory",
                cluster_method="adbscan",
                silence_warnings=True,
            )
        >>> cluster_context.fit()

        >>> ix = 1
        >>> ax = cluster_context.context[ix]["roi_area"].plot(
                figsize=(10, 10), alpha=0.5
            )
        >>> cluster_context.context[ix]["roi_cells"].plot(
                ax=ax, column="class_name", categorical=True, legend=True
            )
        <AxesSubplot: >
        """
        cell_gdf.set_crs(epsg=4328, inplace=True, allow_override=True)
        area_gdf = self.run_clustering(
            cell_gdf, label, cluster_method, n_jobs, **kwargs
        )

        super().__init__(
            area_gdf=area_gdf,
            cell_gdf=cell_gdf,
            label=label,
            min_area_size=min_area_size,
            q=q,
            dist_thresh=dist_thresh,
            graph_type=graph_type,
            predicate=predicate,
            silence_warnings=silence_warnings,
        )

    def run_clustering(
        self,
        cell_gdf: gpd.GeoDataFrame,
        label: str,
        cluster_method: str,
        n_jobs: int,
        **kwargs,
    ) -> gpd.GeoDataFrame:
        """Run clustering on the cells and convert the clusters to areas.

        Parameters
        ----------
        label : str
            The class name of the objects of interest. E.g. "cancer", "immune".
        cluster_method : str, default="dbscan"
            The clustering method used to extract the point-clusters. One of:
            "dbscan", "adbscan", "optics"

        Returns
        -------
        gpd.GeoDataFrame:
            A gdf containing the areas of the clusters.
        """
        # cluster the cells
        cells = cell_gdf[cell_gdf["class_name"] == label].copy()
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
        area_gdf["class_name"] = [label] * len(area_gdf)

        return area_gdf
