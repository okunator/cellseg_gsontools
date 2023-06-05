from typing import Optional, Tuple

import geopandas as gpd
import pandas as pd
from libpysal.weights import W

from ..links import link_counts
from ..summary._base import Summary

__all__ = ["SpatialWeightSummary"]


class SpatialWeightSummary(Summary):
    def __init__(
        self,
        spatial_weights: W,
        gdf: gpd.GeoDataFrame,
        classes: Tuple[str, ...],
        prefix: str = None,
    ) -> None:
        """Create a summary object for the cell networks in a _SpatialContext obj.

        Parameters
        ----------
            spatial_weights : W
                A libpysal spatial weights object.
            gdf : gpd.GeoDataFrame
                The input geodataframe.
            classes : Tuple[str, ...], optional
                The cell type classes found in the data. Optional.
            prefix : str, optional
                A prefix for the named indices.

        Attributes
        ----------
            summary : pd.Series
                The summary vector after running `summarize()`
            link_counts : pd.DataFrame
                A dataframe containing link-counts for each context area.
            link_props : pd.DataFrame
                A dataframe containing link-proportions for each context area

        Example
        -------
        Compute the link-counts in the tumor-stroma interface of a slide.

        >>> from cellseg_gsontools.spatial_context import InterfaceContext
        >>> iface_context = InterfaceContext(
        ...     area_gdf=areas,
        ...     cell_gdf=cells,
        ...     label1="area_cin",
        ...     label2="areastroma",
        ...     silence_warnings=True,
        ...     verbose=True,
        ...     min_area_size=100000.0
        ... )
        >>> iface_context.fit(verbose=False)

        >>> classes = [
        ...     "inflammatory",
        ...     "connective",
        ...     "glandular_epithel",
        ...     "squamous_epithel",
        ...     "neoplastic",
        ... ]

        >>> ss = SpatialWeightSummary(
        ...     iface_context.merge_weights("border_network"),
        ...     iface_context.cell_gdf,
        ...     classes=classes,
        ...     prefix="n-"
        ... )

        >>> ss.summarize()
        n-inflammatory-inflammatory               31
        n-inflammatory-connective                 89
        n-inflammatory-glandular_epithel           0
        n-inflammatory-squamous_epithel            0
        n-inflammatory-neoplastic                 86
        n-connective-connective                  131
        n-connective-glandular_epithel             0
        n-connective-squamous_epithel              0
        n-connective-neoplastic                  284
        n-glandular_epithel-glandular_epithel      0
        n-glandular_epithel-squamous_epithel       0
        n-glandular_epithel-neoplastic             0
        n-squamous_epithel-squamous_epithel        0
        n-squamous_epithel-neoplastic              0
        n-neoplastic-neoplastic                  236
        dtype: int64
        """
        self.prefix = prefix
        self.classes = classes
        self.spatial_weights = spatial_weights
        self.gdf = gdf

    @staticmethod
    def get_link_counts(
        gdf: gpd.GeoDataFrame,
        spatial_weights: W,
        classes: Optional[Tuple[str, ...]] = None,
    ) -> pd.Series:
        """Compute the link counts given a gdf and a spatial weights object W.

        Parameters
        ----------
            gdf : gpd.GeoDataFrame
                The input geodataframe.
            spatial_weights : W
                Libpysal spatial weightsobject fitted from the `gdf`.
            classes : Tuple[str, ...]
                The classes of the dataset.

        Returns
        -------
            pd.Series:
                A named ps.Series object containing the link-counts per class.
        """
        if classes is not None:
            classes = classes
        else:
            classes = list(gdf["class_name"].unique())

        sum_vec = pd.Series(link_counts(gdf, spatial_weights, classes))

        return sum_vec

    @property
    def link_props(self) -> pd.DataFrame:
        """Return link proportions instead of counts."""
        return self.link_counts.div(self.link_counts.sum(axis=1), axis=0)

    def summarize(self, filter_pattern: Optional[str] = None) -> pd.Series:
        """Summarize the cell networks.

        Parameters
        ----------
            filter_pattern : str, optional
                A string pattern. All off the values containing this pattern
                in the result pd.Series are filtered out.

        Raises
        ------
            ValueError: If illegal key is given.

        Returns
        -------
            pd.Series:
                A summary vector containing summary features of the cell network
                objects found in the `spatial_context`.
        """
        counts: pd.Series = self.get_link_counts(
            self.gdf, self.spatial_weights, classes=self.classes
        )
        df = pd.DataFrame(counts, columns=["count"])
        link_summary = df.loc[~(df == 0).all(axis=1)]

        if filter_pattern is not None:
            link_summary = link_summary.loc[
                ~link_summary.index.str.contains(filter_pattern)
            ]

        if self.prefix is not None:
            link_summary = link_summary.set_index(
                self.prefix + link_summary.index.astype(str)
            )

        self.summary = link_summary.squeeze()

        return self.summary
