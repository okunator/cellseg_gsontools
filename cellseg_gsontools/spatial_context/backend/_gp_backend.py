import geopandas as gpd

from cellseg_gsontools.spatial_context.ops import get_interface_zones, get_objs

from ._base_backend import _SpatialBackend

__all__ = ["_SpatialContextGP"]


class _SpatialContextGP(_SpatialBackend):
    def __init__(self) -> None:
        """Create a geopandas spatial context class."""
        self.backend_name = "geopandas"

    def roi(self, context_area: gpd.GeoDataFrame, ix: int) -> gpd.GeoDataFrame:
        """Get a roi area of index `ix`.

        Parameters
        ----------
        context_area : gpd.GeoDataFrame
            The context area gdf.
        ix : int
            The index of the roi area. I.e., the ith roi area.

        Returns
        -------
            gpd.GeoDataFrame:
                The ith roi area.
        """
        row: gpd.GeoSeries = context_area.loc[ix]
        roi_area = gpd.GeoDataFrame([row], crs=context_area.crs)

        return roi_area

    def roi_cells(
        self,
        roi_area: gpd.GeoDataFrame,
        cell_gdf: gpd.GeoDataFrame,
        predicate: str = "intersects",
        silence_warnings: bool = True,
        **kwargs,
    ) -> gpd.GeoDataFrame:
        """Get the cells within the roi area.

        Parameters
        ----------
        roi_area : gpd.GeoDataFrame
            The roi area.
        cell_gdf : gpd.GeoDataFrame
            The cell gdf.
        predicate : str, default="intersects"
            The predicate to use for the spatial join. See `geopandas.tools.sjoin`
        silence_warnings : bool, default=True
            Flag, whether to silence all the geopandas related warnings.

        Returns
        -------
            gpd.GeoDataFrame:
                The cells within the roi area.
        """
        if roi_area is None or roi_area.empty:
            return

        objs_within: gpd.GeoDataFrame = get_objs(
            roi_area,
            cell_gdf,
            silence_warnings=silence_warnings,
            predicate=predicate,
        )

        if objs_within is None or objs_within.empty:
            return

        return objs_within.set_geometry("geometry")

    def interface(
        self,
        top_roi_area: gpd.GeoDataFrame,
        bottom_gdf: gpd.GeoDataFrame,
        buffer_dist: int,
    ) -> gpd.GeoDataFrame:
        """Get an interface area of index `ix`.

        Parameters
        ----------
            top_roi_area : gpd.GeoDataFrame
                The top roi area.
            bottom_gdf : gpd.GeoDataFrame
                The bottom gdf on top of which top_roi_area is buffered.
        """
        # Get the intersection of the roi and the area of type `label2`
        iface = get_interface_zones(
            buffer_area=top_roi_area,
            areas=bottom_gdf,
            buffer_dist=buffer_dist,
        )

        # If there are many interfaces, dissolve them into one
        if len(iface) > 1:
            iface = iface.dissolve().set_geometry("geometry")

        return iface

    def to_geopandas(self, gdf) -> gpd.GeoDataFrame:
        """Convenience identity function to match the backend API."""
        return gdf

    def convert_area_gdf(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Convenience identity function to match the backend API.."""
        return gdf

    def convert_cell_gdf(self, gdf: gpd.GeoDataFrame, **kwargs) -> gpd.GeoDataFrame:
        """Convenience identity function to match the backend API.."""
        return gdf
