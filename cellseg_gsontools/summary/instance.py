from typing import Optional, Tuple

import geopandas as gpd
import pandas as pd
from libpysal.weights import W

from ..diversity import DIVERSITY_LOOKUP, GROUP_DIVERSITY_LOOKUP, local_diversity
from ..geometry import SHAPE_LOOKUP, shape_metric
from ._base import Summary

__all__ = ["InstanceSummary"]


class InstanceSummary(Summary):
    def __init__(
        self,
        cell_gdf: gpd.GeoDataFrame,
        metrics: Tuple[str, ...],
        groups: Optional[Tuple[str, ...]] = None,
        spatial_weights: Optional[W] = None,
        prefix: str = None,
        **kwargs,
    ) -> None:
        """Create a summary object for nuclei object instances.

        Parameters
        ----------
            cell_gdf : gpd.GeoDataFrame
                A geo dattaframe containing nuclei instance segmentation objects.
            metrics : Tuple[str, ...]
                A collection of metrics to be computed from the polygon objects.
            groups : Tuple[str, ...], optional
                A list of catergorical groups. These group-names have to be found in the
                columns of the `gdf`. These groups are used in a groupby operation and
                the metric summaries are computed for each of the groups and sub-groups.
                Ignored if set to None.
            spatial_weights : libysal.weights.W
                Libpysal spatial weights object.
            prefix : str, optional
                A prefix for the named indices.

        Attributes
        ----------
            summary : pd.Series
                The summary vector after running `summarize()`

        Examples
        --------
        Compute a summary of the areas of the cells in different immune cell clusters
        of a gdf. The different immune cell clusters have unique labels in the column
        `label`. Assumes that the `cell_gdf` contains a column `class_name` with
        `inflammatory` values.

        >>> from cellseg_gsontools.spatial_context import PointClusterContext
        >>> from cellseg_gsontools.summary import InstanceSummary

        >>> # Get the immune point clusters from the gdf
        >>> cluster_context = PointClusterContext(
                cell_gdf=cell_gdf,
                label="inflammatory",
                cluster_method="adbscan",
                silence_warnings=True,
                verbose=True,
                min_area_size="mean",
                n_jobs=1
            )
        >>> cluster_context.fit(verbose=True)

        >>> immune_cluster_cells = cluster_context.context2gdf("roi_cells")
        >>> immune_clust_summary = InstanceSummary(
                immune_cluster_cells,
                metrics=["area"],
                groups=["label"],
                prefix="immune-cluster-cells-"
            )
        >>> immune_clust_summary.summarize()
        Processing roi area: 2: 100%|██████████| 2/2 [00:00<00:00,  3.99it/s]
        immune-cluster-cells-1-count         1275.000
        immune-cluster-cells-2-count         3052.000
        immune-cluster-cells-total-count    4327.000
        immune-cluster-cells-1-area-mean      351.390
        immune-cluster-cells-2-area-mean      334.253
        dtype: float64
        """
        self.metrics = metrics
        self.cell_gdf = cell_gdf
        self.groups = groups
        self.spatial_weights = spatial_weights
        self.prefix = prefix

    def summarize(
        self,
        return_counts: bool = True,
        return_quantiles: bool = False,
        return_std: bool = False,
        parallel: bool = True,
        filter_pattern: Optional[str] = None,
        id_col: str = None,
    ) -> pd.Series:
        """Summarize the instance segmentation objects.

        Parameters
        ----------
            return_count : bool, default=True
                Flag, whether to return count data.
            return_quantiles : bool, default=False
                Flag, whether to return qunatile data.
            return_std : bool, default=False
                Flag, whether to return standard deviation of the data.
            parallel : bool, default=True
                Flag, whether the dataframe operations are run in parallel
                with pandarallel package.
            filter_pattern : str, optional
                A string pattern. All off the values containing this pattern
                in the result pd.Series are filtered out.
            id_col : str, optional
                A column name of the `self.cell_gdf` that contains unique identifiers

        Returns
        -------
            pd.Series:
                A summary vector containing summary features of the instance
                segmentation objects found in the `area_gdf`.
        """
        # TODO: clean up the code
        # compute metrics
        shape_mets = [m for m in self.metrics if m in SHAPE_LOOKUP.keys()]
        if shape_mets:
            self.cell_gdf = shape_metric(
                self.cell_gdf.copy(), metrics=shape_mets, parallel=parallel
            )

        div_mets = [m for m in self.metrics if m in DIVERSITY_LOOKUP.keys()]
        if div_mets and self.spatial_weights is not None:
            self.cell_gdf = local_diversity(
                self.cell_gdf.copy(),
                val_col="class_name",
                spatial_weights=self.spatial_weights,
                metrics=div_mets,
                categorical=True,
                parallel=parallel,
                scheme="FisherJenks",
                id_col=id_col,
            )
            # modify the metrics to match the new column names
            for i, m in enumerate(self.metrics):
                if m in m in DIVERSITY_LOOKUP.keys():
                    self.metrics[i] = f"class_name_{m}"
        elif div_mets and self.spatial_weights is None:
            raise ValueError(
                "To run diversity metrics a spatial weights object is needed."
            )

        # get the summary vec
        self.summary = self.gen_metric_summary(
            self.cell_gdf,
            [m for m in self.metrics if m not in GROUP_DIVERSITY_LOOKUP.keys()],
            self.groups,
            self.prefix,
        )
        group_div_mets = [m for m in self.metrics if m in GROUP_DIVERSITY_LOOKUP.keys()]

        # TODO: fix this
        if group_div_mets is not None:
            for m in group_div_mets:
                if self.groups is None:
                    raise ValueError(
                        f"Group diversity metric: {m} require a group, but "
                        "`self.groups` was set to None."
                    )
                for met in self.metrics:
                    if met not in GROUP_DIVERSITY_LOOKUP.keys():
                        for group in self.groups:
                            self.summary[
                                f"{self.prefix}{m}-{group}-{met}"
                            ] = GROUP_DIVERSITY_LOOKUP[m](
                                self.cell_gdf[met], self.cell_gdf[group]
                            )

        # filter
        if not return_counts:
            self.summary = self.summary.loc[~self.summary.index.str.contains("count")]

        if not return_quantiles:
            pat = "min|max|25%|50%|75%"
            self.summary = self.summary.loc[~self.summary.index.str.contains(pat)]

        if not return_std:
            self.summary = self.summary.loc[~self.summary.index.str.contains("std")]

        if filter_pattern is not None:
            self.summary = self.summary.loc[
                ~self.summary.index.str.contains(filter_pattern)
            ]

        return self.summary
