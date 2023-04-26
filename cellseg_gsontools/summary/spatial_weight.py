from typing import Optional, Tuple

import geopandas as gpd
import pandas as pd
from libpysal.weights import W

from ..links import link_counts
from ..spatial_context._base import _SpatialContext
from ..summary._base import Summary

__all__ = ["SpatialWeightSummary"]


class SpatialWeightSummary(Summary):
    def __init__(
        self,
        spatial_context: _SpatialContext,
        classes: Optional[Tuple[str, ...]] = None,
        prefix: str = None,
    ) -> None:
        """Create a summary object for the cell networks in a _SpatialContext obj.

        Parameters
        ----------
            spatial_context : _SpatialContext
                A spatial context object. E.g `InterfaceContext`, or `WithinContext`.
            classes : Tuple[str, ...], optional
                The cell type classes found in the data. Optional.
            prefix : str, optional
                A prefix for the named indices.

        Attributes
        ----------
            link_counts : pd.DataFrame
                A dataframe containing link-counts for each context area.
            link_props : pd.DataFrame
                A dataframe containing link-proportions for each context area

        Raises
        ------
            ValueError if input is not a _SpatiaContext object

        Example
        -------
            Compute the link-counts in the tumor-stroma interface of a slide.

            >>> from cellseg_gsontools.spatial_context import InterfaceContext
            >>> iface_context = InterfaceContext(
                    area_gdf=areas,
                    cell_gdf=cells,
                    label1="area_cin",
                    label2="areastroma",
                    silence_warnings=True,
                    verbose=True,
                    min_area_size=100000.0
                )

            >>> t = [
                    "inflammatory",
                    "connective",
                    "glandular_epithel",
                    "squamous_epithel",
                    "neoplastic"
                ]

            >>> s = SpatialWeightSummary(iface_context, t, prefix="n-")
            >>> s.summarize("border_network")
            Processing interface area: 4: 100%|██████████| 4/4 [00:01<00:00,  2.21it/s]
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
        self.spatial_context = spatial_context

        if not isinstance(spatial_context, _SpatialContext):
            raise ValueError(
                "Input needs to be a class ingerited from the `_SpatialContext` object."
                f" Got: {type(spatial_context)}."
            )

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

    def summarize(self, key: str) -> pd.Series:
        """Summarize the cell networks.

        Parameters
        ----------
            key : str
                The network name to summarize. One of: "full_network", "border_network",
                "interface_network", "roi_network".

        Raises
        ------
            ValueError: I illegal  key is given.

        Returns
        -------
            pd.Series:
                A summary vector containing summary features of the cell network
                objects found in the `spatial_context`.
        """
        allowed = ("full_network", "border_network", "interface_network", "roi_network")
        if key not in allowed:
            raise ValueError(f"Illegal key. Got: {key}. Allowed: {allowed}")

        prefix = self.prefix if self.prefix is not None else ""
        counts = []
        for i, con in self.spatial_context.context.items():
            datum = pd.Series({"label": i})
            link_c = self.get_link_counts(
                self.spatial_context.cell_gdf, con[key], self.classes
            )
            datum = pd.concat([datum, link_c.add_prefix(f"{prefix}")])
            counts.append(datum)

        df = pd.DataFrame(counts).set_index("label")
        self.link_counts = df.loc[~(df == 0).all(axis=1)]

        return self.link_counts.sum(axis=0)
