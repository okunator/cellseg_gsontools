from typing import List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd

from ._base import Summary

__all__ = ["DistanceSummary"]


class DistanceSummary(Summary):
    def __init__(
        self,
        gdf1: gpd.GeoDataFrame,
        gdf2: gpd.GeoDataFrame,
        groups: Optional[Tuple[str, ...]] = None,
        prefix: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Create a summary object for summarizing distances between two gdfs.

        Parameters
        ----------
            gdf1 : gpd.GeoDataFrame
                Input geo dataframe 1. The objects in this frame are buffered until they
                hit the objects in gdf2.
            gdf2 : gpd.GeoDataFrame
                Input geo dataframe 2. The objects in this frame are used as references.
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
        Get the proximities of immune clusters to the neoplastic areas. Assumes that the
        `cell_gdf` contains a column `class_name` with `inflammatory` values and that
        `area_gdf` contains a column `class_name` with `area_neoplastic` values.

        >>> from cellseg_gsontools.summary import DistanceSummary
        >>> form cellseg_gsontools.spatial_context import (
                WithinContext, PointClusterContext
            )

        >>> # Get the neoplastic areas in the gdf
        >>> within_context = WithinContext(
                area_gdf=area_gdf,
                cell_gdf=cell_gdf,
                label="area_neoplastic",
                silence_warnings=True,
                verbose=True,
                min_area_size=100000.0
            )
        >>> within_context.fit()

        >>> # Get the immune clusters from the gdf
        >>> cluster_context = PointClusterContext(
                cell_gdf=cell_gdf,
                label="inflammatory",
                cluster_method="adbscan",
                silence_warnings=True,
                verbose=True,
                min_area_size=50000.0,
                n_jobs=1
            )
        >>> cluster_context.fit()

        >>> # compute the summary counts
        >>> neoplastic_areas = within_context.context2gdf("roi_area")
        >>> immune_cluster_areas = cluster_context.context2gdf("roi_area")
        >>> immune_proximities = DistanceSummary(
                immune_cluster_areas,
                lesion_areas,
                prefix="close_to_lesion-"
            )
        >>> immune_proximities.summarize(thresh_dist=300.0)
        close_to_lesion-0-count    2.0
        close_to_lesion-1-count    7.0
        dtype: float64
        """
        self.gdf1 = gdf1
        self.gdf2 = gdf2
        self.groups = groups
        self.prefix = prefix

    @staticmethod
    def dist2area(
        gdf1: gpd.GeoDataFrame,
        gdf2: gpd.GeoDataFrame,
        thresh_dist: int = 350,
        step: int = 5,
    ) -> Tuple[List[bool], List[int]]:
        """Add buffer to polygons in `gdf1` iteratively & check if it intersects `gdf2`.

        Parameters
        ----------
            gdf1 : gpd.GeoDataFrame
                A GeodataFrame that is buffered until intersection
            gdf2 : gpd.GeoDataFrame
                A geodataframe whose objects are checked for intersection
            thresh_dist : int, default=350
                A distance threshold indicating max distance that is considered to be
                small enough that we can say that an area is close to another.
            step : int, default=5
                Amount of buffering added at each iteration.

        Returns
        -------
            Tuple[List[bool], List[int]]:
                List: a list booleans indicating whether area intersected `gdf2`
                List: a list of integers indicating the amount of buffer needed for
                    intersection with precision of `step`.
        """
        is_close = [False] * len(gdf1)
        dist = [0] * len(gdf1)
        for buf in np.arange(0, thresh_dist, step):
            for i, ix in enumerate(gdf1.index):
                if gdf2.intersects(gdf1.buffer(buf).geometry.loc[ix]).any():
                    is_close[i] = True
                else:
                    is_close[i] = False
                    dist[i] = buf

            if all(is_close):
                break

        return is_close, dist

    def summarize(
        self,
        thresh_dist: int = 350,
        step: int = 5,
        filter_pattern: Optional[str] = None,
        **kwargs,
    ) -> pd.Series:
        """Count the objs that are less than `thresh_dist` away from the objs in `gdf2`.

        Parameters
        ----------
            thresh_dist : int, default=350
                A distance threshold indicating max distance that is considered to be
                small enough that we can say that an area is close to another.
            step : int, default=5
                Amount of buffering added at each iteration.

        Returns
        -------
            pd.Series:
                A summary vector containing counts on how many objects are close
                to the objects in `gdf2`
        """
        is_close, _ = self.dist2area(self.gdf1, self.gdf2, thresh_dist, step)

        isclose_col = "is_close"
        self.gdf1[isclose_col] = np.array(is_close, dtype="uint8")

        if self.groups is not None:
            self.groups.append(isclose_col)
        else:
            self.groups = [isclose_col]

        self.summary = self.gen_metric_summary(
            self.gdf1, (isclose_col,), self.groups, self.prefix
        )

        # filter
        pat = "min|max|25%|50%|75%|total-count|mean|std"
        self.summary = self.summary.loc[~self.summary.index.str.contains(pat)]

        if filter_pattern is not None:
            self.summary = self.summary.loc[
                ~self.summary.index.str.contains(filter_pattern)
            ]

        return self.summary
