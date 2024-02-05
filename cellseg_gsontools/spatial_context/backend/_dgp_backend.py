# try:
#     import dask_geopandas

#     _has_dask_geopandas = True
# except ImportError:
#     _has_dask_geopandas = False


# import geopandas as gpd

# from cellseg_gsontools.spatial_context.backend import _SpatialBackend
# from cellseg_gsontools.spatial_context.ops import (
#     get_interface_zones,
#     get_objs,
#     get_objs_dgp,
# )

# __all__ = ["_SpatialContextDGP"]


# class _SpatialContextDGP(_SpatialBackend):
#     def __init__(self) -> None:
#         """Create a dask_geopandas spatial context class."""
#         if not _has_dask_geopandas:
#             raise ImportError(
#                 "dask_geopandas is required to use the spatial context backend."
#                 "Install it with `pip install dask_geopandas`."
#             )
#         self.backend_name = "dask_geopandas"

#     def roi(self, context_area: gpd.GeoDataFrame, ix: int) -> gpd.GeoDataFrame:
#         """Get a roi area of index `ix`.

#         Parameters
#         ----------
#         context_area : gpd.GeoDataFrame
#             The context area gdf.
#         ix : int
#             The index of the roi area. I.e., the ith roi area.

#         Returns
#         -------
#             gpd.GeoDataFrame:
#                 The ith roi area.
#         """
#         row: gpd.GeoSeries = context_area.loc[ix]
#         roi_area = gpd.GeoDataFrame([row], crs=context_area.crs)

#         return roi_area

#     def roi_cells(
#         self,
#         roi_area: gpd.GeoDataFrame,
#         cell_gdf: gpd.GeoDataFrame,
#         cell_gdf_dgp: dask_geopandas.GeoDataFrame = None,
#         predicate: str = "intersects",
#         silence_warnings: bool = True,
#         parallel: bool = False,
#         **kwargs,
#     ) -> gpd.GeoDataFrame:
#         """Get the cells within the roi area.

#         Parameters
#         ----------
#         roi_area : gpd.GeoDataFrame
#             The roi area.
#         cell_gdf : gpd.GeoDataFrame
#             The cell gdf.
#         cell_gdf_dgp : dask_geopandas.GeoDataFrame, optional
#             The cell gdf as a dask_geopandas.GeoDataFrame. If None, then the geopandas
#             cell_gdf is used.
#         predicate : str, default="intersects"
#             The predicate to use for the spatial join. See `geopandas.tools.sjoin`
#         silence_warnings : bool, default=True
#             Flag, whether to silence all the geopandas related warnings.
#         parallel : bool, default=False
#             Flag, whether to parallelize the spatial join.

#         Returns
#         -------
#             gpd.GeoDataFrame:
#                 The cells within the roi area.

#         Returns
#         -------
#             gpd.GeoDataFrame:
#                 The cells within the roi area.
#         """
#         if roi_area is None or roi_area.empty:
#             return

#         # get the cells within the roi area
#         # if area is large, use dask-geopandas
#         if (
#             roi_area.geometry.area.values[0] > 1e8
#             and parallel
#             and cell_gdf_dgp is not None
#         ):
#             objs_within: gpd.GeoDataFrame = get_objs_dgp(
#                 roi_area,
#                 cell_gdf_dgp,
#                 silence_warnings=silence_warnings,
#                 predicate=predicate,
#             )
#         else:
#             objs_within: gpd.GeoDataFrame = get_objs(
#                 roi_area,
#                 cell_gdf,
#                 silence_warnings=silence_warnings,
#                 predicate=predicate,
#             )

#         if objs_within is None or objs_within.empty:
#             return

#         # rename spatial join columns
#         objs_within = objs_within.rename(
#             columns={
#                 "index_right": "spatial_context_id",
#                 "global_id_left": "global_id",
#                 "class_name_left": "class_name",
#                 "class_name_right": "spatial_context",
#             },
#         )

#         # drop unnecessary columns and return
#         objs_within.drop(columns=["global_id_right"], inplace=True)

#         # convert to geopandas if dask-geopandas
#         if objs_within.index.name == "hilbert_distance":
#             objs_within = self.to_geopandas(objs_within)

#         return objs_within.set_geometry("geometry")

#     def interface(
#         self,
#         top_roi_area: gpd.GeoDataFrame,
#         bottom_gdf: gpd.GeoDataFrame,
#         buffer_dist: int,
#     ) -> gpd.GeoDataFrame:
#         """Get an interface area of index `ix`.

#         Parameters
#         ----------
#             top_roi_area : gpd.GeoDataFrame
#                 The top roi area.
#             bottom_gdf : gpd.GeoDataFrame
#                 The bottom gdf on top of which top_roi_area is buffered.
#         """
#         # Get the intersection of the roi and the area of type `label2`
#         iface = get_interface_zones(
#             buffer_area=top_roi_area,
#             areas=bottom_gdf,
#             buffer_dist=buffer_dist,
#         )

#         # If there are many interfaces, dissolve them into one
#         if len(iface) > 1:
#             iface = iface.dissolve().set_geometry("geometry")

#         return iface

#     def to_geopandas(self, gdf):
#         """Convert gdf to right SpatialContext gdf format."""
#         return gdf.sort_values("global_id").set_index("global_id", drop=False)

#     def convert_area_gdf(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
#         """Convenience identity function to match the backend API.."""
#         return gdf

#     def convert_cell_gdf(
#         self, gdf: gpd.GeoDataFrame, n_partitions: int = None, **kwargs
#     ) -> dask_geopandas.GeoDataFrame:
#         """Set up a dask_geopandas.GeoDataFrame attribute for parallel spatial ops.

#         Parameters
#         ----------
#         gdf : gpd.GeoDataFrame
#             The cell gdf to convert.
#         n_partitions : int, optional
#             The number of processes/partitions to use for the DaskGeoDataFrame. If None,
#             the number of partitions is set to the number of cores.

#         Attributes
#         ----------
#         cell_gdf_dgp : DaskGeoDataFrame
#             The cell gdf as a dask_geopandas.GeoDataFrame.

#         Returns
#         -------
#         gpd.GeoDataFrame:
#             Return the original gpd.GeoDataFrame.
#         """
#         if _has_dask_geopandas:
#             print(
#                 f"partitioning cell_gdf into {n_partitions} partitions and shuffling."
#             )
#             self.cell_gdf_dgp = dask_geopandas.from_geopandas(
#                 gdf, npartitions=n_partitions
#             )
#             self.cell_gdf_dgp = self.cell_gdf_dgp.spatial_shuffle()
#         else:
#             raise ImportError(
#                 "dask-geopandas is required to use the spatial context "
#                 "backend. with backend == 'dask-geopandas'."
#                 "Install with `pip install dask-geopandas`."
#             )

#         return gdf
