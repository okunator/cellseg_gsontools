import warnings
from typing import Union

import geopandas as gpd
import numpy as np
import pandas as pd
from libpysal.weights import DistanceBand, W, w_subset, w_union

from cellseg_gsontools.utils import set_uid

__all__ = ["_SpatialContext"]


class _SpatialContext:
    def __init__(
        self,
        area_gdf: gpd.GeoDataFrame,
        cell_gdf: gpd.GeoDataFrame,
        label: str,
        min_area_size: Union[float, str] = None,
        q: float = 25.0,
        verbose: bool = False,
        silence_warnings: bool = True,
    ) -> None:
        """Create a base class for spatial context."""
        if "class_name" not in area_gdf.columns:
            raise ValueError(
                "'class_name' column not found. `area_gdf` must contain a 'class_name' "
                f"column. Got: {list(area_gdf.columns)}"
            )

        if "class_name" not in cell_gdf.columns:
            raise ValueError(
                "'class_name' column not found. `cell_gdf` must contain a 'class_name' "
                f"column. Got: {list(cell_gdf.columns)}"
            )

        self.verbose = verbose
        self.silence_warnings = silence_warnings
        self.label = label
        self.cell_gdf = set_uid(cell_gdf)
        self.area_gdf = area_gdf
        thresh = self._get_thresh(
            area_gdf[area_gdf["class_name"] == label], min_area_size, q
        )
        self.context_area = self.filter_above_thresh(area_gdf, label, thresh)

    @staticmethod
    def filter_above_thresh(
        gdf: gpd.GeoDataFrame,
        label: str,
        thresh: float = None,
    ) -> gpd.GeoDataFrame:
        """Filter areas or objects that are above a threshold.

        NOTE: threshold by default is the mean of the area.
        """
        gdf = gdf.loc[gdf["class_name"] == label].copy()
        gdf.loc[:, "area"] = np.round(gdf.area)

        if thresh is not None:
            gdf = gdf.loc[gdf.area >= thresh]

        return set_uid(gdf)

    def roi(self, ix: int) -> gpd.GeoDataFrame:
        """Get the roi/area of interest.

        Parameters
        ----------
            ix : int
                The index of the ROI geo-object. Starts from one.
        """
        try:
            roi = self.context[ix]["roi_area"]
        except KeyError:
            roi = gpd.GeoDataFrame([self.context_area.loc[ix]])

        return roi

    def roi_cells(self, ix: int) -> gpd.GeoDataFrame:
        """Get the cells inside the roi."""
        try:
            roi_c = self.context[ix]["roi_cells"]
        except KeyError:
            roi = self.roi(ix)
            if roi.empty:
                if not self.silence_warnings:
                    warnings.warn(
                        "`self.roi` resulted in an empty GeoDataFrame. Make sure to set"
                        " a valid `label` and `ix`. Returning None from `roi_cells`",
                        RuntimeWarning,
                    )
                return None

            roi_c = self.cell_gdf[self.cell_gdf.within(roi.geometry[ix])]

        return roi_c

    def merge_weights(self, key: str) -> W:
        """Merge libpysal spatial weights of the context.

        Parameters
        ----------
            key : str
                The key of the context dictionary that contains the spatial
                weights to be merged. One of "roi_network", "ful_network",
                "interface_network", "border_network"

        Returns
        -------
            libpysal.weights.W:
                A spatial weights object containing all the distinct networks
                in the context.
        """
        allowed = ("roi_network", "ful_network", "interface_network", "border_network")
        if key not in allowed:
            raise ValueError(f"Illegal key. Got: {key}. Allowed: {allowed}")

        cxs = list(self.context.items())
        wout = W({0: [0]})
        for _, c in cxs:
            w = c[key]
            wout = w_union(wout, w, silence_warnings=True)
        wout = w_subset(wout, list(wout.neighbors.keys())[1:], silence_warnings=True)

        return wout

    def context2gdf(self, key: str) -> gpd.GeoDataFrame:
        """Convert the context of type `key` into a geodataframe.

        Parameters
        ----------
            key : str
                The key of the context dictionary that contains the data to be converted
                to gdf. One of "roi_area", "roi_cells", "interface_area",
                "interface_cells", "roi_interface_cells"

        Returns
        -------
            gpd.GeoDataFrame:
                Geo dataframe containing all the objects
        """
        con = []
        for i in self.context.keys():
            if self.context[i][key] is not None:
                if isinstance(self.context[i][key], tuple):
                    con.append(self.context[i][key][0])
                else:
                    con.append(self.context[i][key])

        gdf = pd.concat(
            con,
            keys=[i for i in self.context.keys() if self.context[i][key] is not None],
        )
        # return set_uid(gdf.reset_index(level=0, names="label"))
        return gdf.reset_index(level=0, names="label")

    def context2weights(self, key: str, **kwargs) -> W:
        """Fit a network on the cells inside the context.

        Parameters
        ----------
            key : str
                The key of the context dictionary that contains the data to be converted
                to gdf. One of "roi_area", "roi_cells", "interface_area",
                "interface_cells", "roi_interface_cells"
        """
        allowed = ("roi_cells", "interface_cells", "roi_interface_cells")
        if key not in allowed:
            raise ValueError(
                "Illegal key. Note that network can be only fitted to cell gdfs. "
                f"Got: {key}. Allowed: {allowed}"
            )

        cells = self.context2gdf(key)
        w = DistanceBand.from_dataframe(cells, silence_warnings=True, **kwargs)

        return w

    def _get_thresh(self, area_gdf, min_area_size, q=None) -> float:
        """Get the threshold value for filtering by area."""
        if isinstance(min_area_size, str):
            allowed = ("mean", "median", "quantile")
            if min_area_size not in allowed:
                raise ValueError(
                    f"Got illegal `min_area_size`. Got: {min_area_size}. "
                    f"Allowed values are floats or these options: {allowed}."
                )
            if min_area_size == "mean":
                thresh = area_gdf.area.mean()
            elif min_area_size == "median":
                thresh = area_gdf.area.median()
            elif min_area_size == "quantile":
                thresh = np.nanpercentile(area_gdf.area, q)
        elif isinstance(min_area_size, (int, float)):
            thresh = float(min_area_size)
        elif min_area_size is None:
            thresh = None
        else:
            raise ValueError(
                f"Got illegal `min_area_size`. Got: {min_area_size}. "
                f"Allowed values are floats or these options: {allowed}."
            )

        return thresh
