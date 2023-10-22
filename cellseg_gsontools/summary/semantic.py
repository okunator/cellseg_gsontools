from typing import Optional, Tuple

import geopandas as gpd
import pandas as pd

from ..diversity import GROUP_DIVERSITY_LOOKUP
from ..geometry import shape_metric
from ..geometry.shape_metrics import SHAPE_LOOKUP
from ._base import Summary

__all__ = ["SemanticSummary"]


class SemanticSummary(Summary):
    def __init__(
        self,
        area_gdf: gpd.GeoDataFrame,
        metrics: Tuple[str, ...],
        groups: Optional[Tuple[str, ...]] = None,
        prefix: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Create a summary object for semantic tissue areas.

        Parameters
        ----------
            area_gdf : gpd.GeoDataFrame
                A geo dattaframe containing tissue area segmentation objects.
            metrics : Tuple[str, ...]
                A collection of shape metrics to be computed from the polygon objects.
                Only shape metrics allowed.
            groups : Tuple[str, ...], optional
                A list of catergorical groups. These group-names have to be found in the
                columns of the `gdf`. These groups are used in a groupby operation and
                the metric summaries are computed for each of the groups and sub-groups.
                Ignored if set to None.
            prefix : str, optional
                A prefix for the named indices.

        Attributes
        ----------
            summary : pd.Series
                The summary vector after running `summarize()`

        Examples
        --------
        Compute a summary of the areas of different immune cell clusters in a gdf.
        Namely, the number of them and the mean area of them.

        >>> from cellseg_gsontools.spatial_context import PointClusterContext
        >>> from cellseg_gsontools.summary import SemanticSummary

        >>> cluster_context = PointClusterContext(
                cell_gdf=cell_gdf,
                labels="inflammatory",
                cluster_method="adbscan",
                silence_warnings=True,
                verbose=True,
                min_area_size=50000.0,
                n_jobs=1
            )
        >>> cluster_context.fit(verbose=True)

        >>> immune_cluster_areas = cluster_context.context2gdf("roi_area")
        >>> immune_areas = SemanticSummary(
                immune_cluster_areas,
                ["area"],
                prefix="immune-cluster-"
            )
        >>> immune_areas.summarize()
        Processing roi area: 3: 100%|██████████| 3/3 [00:00<00:00,  4.82it/s]
        immune-cluster-total-count                          3.000
        immune-cluster-area-mean                      2756128.667
        Name: area, dtype: float64
        """
        self.metrics = metrics
        self.area_gdf = area_gdf
        self.groups = groups
        self.prefix = prefix

    def summarize(
        self,
        return_counts: bool = True,
        return_quantiles: bool = False,
        return_std: bool = False,
        parallel: bool = True,
        filter_pattern: Optional[str] = None,
        **kwargs,
    ) -> pd.Series:
        """Summarize the semantic segmentation areas.

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

        Returns
        -------
            pd.Series:
                A summary vector containing summary features of the semantic
                segmentation objects found in the `area_gdf`.
        """
        shape_mets = [m for m in self.metrics if m in SHAPE_LOOKUP.keys()]
        if shape_mets:
            self.area_gdf = shape_metric(
                self.area_gdf.copy(), metrics=shape_mets, parallel=parallel
            )

        # get the summary vec
        self.summary = self.gen_metric_summary(
            self.area_gdf, shape_mets, self.groups, self.prefix
        )

        group_div_mets = [m for m in self.metrics if m in GROUP_DIVERSITY_LOOKUP.keys()]
        if group_div_mets is not None:
            for m in group_div_mets:
                if self.groups is None:
                    raise ValueError(
                        f"Group diversity metric: {m} require a group, but "
                        "`self.groups` was set to None."
                    )
                for met in shape_mets:
                    for group in self.groups:
                        self.summary[
                            f"{self.prefix}{m}-{group}-{met}"
                        ] = GROUP_DIVERSITY_LOOKUP[m](
                            self.area_gdf[met], self.area_gdf[group]
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
