from abc import ABC, abstractmethod
from typing import List, Union

import geopandas as gpd

__all__ = ["_SpatialBackend"]


class _SpatialBackend(ABC):
    @abstractmethod
    def convert_area_gdf(self):
        """Convert an area gdf to another gdf format."""
        raise NotImplementedError

    @abstractmethod
    def convert_cell_gdf(self):
        """Convert an area gdf to another gdf format."""
        raise NotImplementedError

    @abstractmethod
    def to_geopandas(self):
        """Convert a gdf to a geopandas dataframe."""
        raise NotImplementedError

    @abstractmethod
    def roi(self):
        """Get the region of interest."""
        raise NotImplementedError

    @abstractmethod
    def roi_cells(self):
        """Get the cells in the region of interest."""
        raise NotImplementedError

    @abstractmethod
    def interface(self):
        """Get the interface of two regions."""
        raise NotImplementedError

    def check_columns(
        self, area_gdf: gpd.GeoDataFrame, cell_gdf: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        """Check if the input gdfs have the required columns.

        Parameters
        ----------
        area_gdf : gpd.GeoDataFrame
            A geodataframe containing the areas.
        cell_gdf : gpd.GeoDataFrame
            A geodataframe containing the cells.

        Returns
        -------
        gpd.GeoDataFrame:
            The gdf with the required columns.
        """
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

    def filter_areas(
        self,
        area_gdf: gpd.GeoDataFrame,
        labels: Union[str, List[str]],
        min_area_size: float = None,
    ) -> gpd.GeoDataFrame:
        """Filter areas by tissue type and size.

        Parameters
        ----------
        area_gdf : gpd.GeoDataFrame
            A geodataframe containing the areas.
        labels : Union[str, List[str]]
            The labels to filter the areas by.
        min_area_size : float, optional
            The minimum area size to keep.

        Returns
        -------
        gpd.GeoDataFrame
            A geodataframe containing the filtered areas.
        """
        # get the areas that have type in labels
        if isinstance(labels, str):
            area_gdf = area_gdf[area_gdf["class_name"] == labels]
        else:
            if len(labels) == 1:
                area_gdf = area_gdf[area_gdf["class_name"] == labels[0]]
            else:
                area_gdf = area_gdf[area_gdf["class_name"].isin(labels)]

        # drop areas smaller than min_area_size
        if min_area_size is not None:
            area_gdf = area_gdf.loc[area_gdf.geometry.area >= min_area_size]

        return area_gdf
