from typing import Tuple, Union

import geopandas as gpd
import matplotlib.pyplot as plt
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
        verbose: bool = False,
        silence_warnings: bool = True,
        n_jobs: int = -1,
    ) -> None:
        """Handle & extract dense point clusters from `cell_gdf`.

        I.e. Cluster points/centroids of type `label` and enclose those points with
        alpha shape.

        Parameters
        ----------
            cell_gdf : gpd.GeoDataFrame
                A geo dataframe that contains smaller cell objs.
            label : str
                The class name of the objects of interest. E.g. "cancer", "immune".
            cluster_method : str, default="dbscan"
                The clustering method. One of "dbscan", "adbscan", "optics"
            min_area_size : float or str, optional
                The minimum area of the cluster areas that are kept.
            q : float, default=25.0
                The quantile. This is only used if `min_area_size = "quantile"`.
            verbose : bool, default=False
                Flag, whether to use tqdm pbar.
            silence_warnings : bool, default=True
                Flag, whether to silence all the warnings.
            n_jobs : int,default=-1
                Number of jobs used when clustering. None=1, and -1 means all available.

        Attributes
        ----------
            context : Dict[int, Dict[str, Union[gpd.GeoDataFrame, DistanceBand]]]
                A nested dict that contains dicts for each index of the distinct areas
                of type `label`. Each of the inner dicts contain the keys: 'roi_area',
                'roi_cells', 'roi_network'. The 'area' and 'cells' keys contain
                gpd.GeoDataFrame of the roi area and cells. The 'network' key contains
                a DistanceBand fitted to the cells inside the roi.

        Raises
        ------
            ValueError if `area_gdf` or `cell_gdf` don't contain 'class_name' column.

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
                verbose=True,
            )

        >>> ix = 1
        >>> ax = cluster_context.context[ix]["roi_area"].plot(
                figsize=(10, 10), alpha=0.5
            )
        >>> cluster_context.context[ix]["roi_cells"].plot(
                ax=ax, column="class_name", categorical=True, legend=True
            )

        """
        cells = cell_gdf[cell_gdf["class_name"] == label].copy()
        cells = cluster_points(cells, method=cluster_method, n_jobs=n_jobs)

        labs = cells["labels"].unique()
        area_data = {"geometry": []}
        for lab in labs:
            if lab == str(-1) or lab == int(-1):
                continue

            if isinstance(lab, str):
                lab = str(lab)

            c = cells[cells["labels"] == lab]
            coords = np.vstack([c.centroid.x, c.centroid.y]).T
            alpha_shape = alpha_shape_auto(coords, step=10)
            area_data["geometry"].append(alpha_shape)

        area_gdf = gpd.GeoDataFrame(area_data)
        area_gdf["class_name"] = [label] * len(area_gdf)

        super().__init__(
            area_gdf=area_gdf,
            cell_gdf=cell_gdf,
            label=label,
            min_area_size=min_area_size,
            q=q,
            verbose=verbose,
            silence_warnings=silence_warnings,
        )

    def plot(
        self,
        show_area: bool = True,
        show_cells: bool = True,
        show_legends: bool = True,
        color: str = None,
        figsize: Tuple[int, int] = (12, 12),
    ) -> None:
        """Plot the slide with areas, cells, and clustered areas highlighted.

        Parameters
        ----------
            show_area : bool, default=True
                Flag, whether to include the tissue areas in the plot.
            show_cells : bool, default=True
                Flag, whether to include the cells in the plot.
            show_legends : bool, default=True
                Flag, whether to include legends for each category in the plot.
            color : str, optional
                A color for the cluster areas, Ignored if `show_legends=True`.
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

        roi = self.context2gdf("roi_area")
        ax = roi.plot(
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
