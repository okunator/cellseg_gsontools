# from typing import Union

# try:
#     from spatialpandas import GeoDataFrame
#     from spatialpandas.dask import DaskGeoDataFrame

#     _has_spatialpandas = True
# except ImportError:
#     _has_spatialpandas = False

# try:
#     import dask
#     import dask.dataframe as dd
#     from dask.distributed import Client, LocalCluster

#     _has_dask = True
# except ImportError:
#     _has_dask = False

# import geopandas as gpd

# from cellseg_gsontools.spatial_context.ops import get_interface_zones, get_objs_sp

# from ._base_backend import _SpatialBackend

# if _has_dask:
#     dask.config.set({"logging.distributed": "error"})

# __all__ = ["_SpatialContextSP"]


# class _SpatialContextSP(_SpatialBackend):
#     def __init__(self) -> None:
#         """Create a spatialpandas spatial context class."""
#         if not _has_spatialpandas:
#             raise ImportError(
#                 "spatialpandas is required to use the spatial context backend."
#                 "Install it with `pip install spatialpandas`."
#             )
#         self.backend_name = "spatialpandas"

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
#         roi_area = context_area.loc[[ix]]

#         return self.to_geopandas(roi_area)

#     def roi_cells(
#         self,
#         roi_area: gpd.GeoDataFrame,
#         cell_gdf: gpd.GeoDataFrame,
#         predicate: str = "intersects",
#         silence_warnings: bool = True,
#         parallel: bool = False,
#         num_processes: int = None,
#         **kwargs,
#     ) -> gpd.GeoDataFrame:
#         """Get the cells within the roi area.

#         Parameters
#         ----------
#         roi_area : gpd.GeoDataFrame
#             The roi area.
#         cell_gdf : GeoDataFrame
#             The cell gdf.
#         predicate : str, default="intersects"
#             The predicate to use for the spatial join.
#         silence_warnings : bool, default=True
#             Flag, whether to silence all the geopandas related warnings.
#         parallel : bool, default=False
#             Whether to use parallel processing.
#         num_processes : int, optional
#             The number of processes/partitions to use for the DaskGeoDataFrame. If None,
#             the number of partitions is set to the number of cores.

#         Returns
#         -------
#             gpd.GeoDataFrame:
#                 The cells within the roi area.
#         """
#         # check to not compute roi_area if already computed
#         if roi_area is None or roi_area.empty:
#             return

#         # distributing sjoin to many workers results in speedup only for large areas
#         if roi_area.geometry.area.values[0] > 1e8 and parallel:
#             with LocalCluster(
#                 n_workers=int(num_processes),
#                 processes=True,
#                 threads_per_worker=1,
#             ) as cluster, Client(cluster) as client:  # noqa
#                 objs_within: GeoDataFrame = get_objs_sp(
#                     GeoDataFrame(roi_area),
#                     cell_gdf,
#                     silence_warnings=silence_warnings,
#                     predicate=predicate,
#                 )
#         else:
#             objs_within: GeoDataFrame = get_objs_sp(
#                 GeoDataFrame(roi_area),
#                 cell_gdf,
#                 silence_warnings=silence_warnings,
#                 predicate=predicate,
#             )

#         if objs_within is None or len(objs_within.index) == 0:
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
#         objs_within = objs_within.drop(columns=["global_id_right"])

#         return self.to_geopandas(objs_within)

#     def interface(
#         self,
#         top_roi_area: gpd.GeoDataFrame,
#         bottom_gdf: GeoDataFrame,
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
#             areas=bottom_gdf.to_geopandas().set_crs(
#                 epsg=4328, inplace=True, allow_override=True
#             ),
#             buffer_dist=buffer_dist,
#         )

#         # If there are many interfaces, dissolve them into one
#         if len(iface) > 1:
#             iface = iface.dissolve().set_geometry("geometry")

#         return iface

#     def to_geopandas(self, gdf) -> gpd.GeoDataFrame:
#         """Convert a spatialpandas GeoDataFrame to a geopandas GeoDataFrame."""
#         if isinstance(gdf, GeoDataFrame):
#             gdf = (
#                 gdf.to_geopandas()
#                 .set_geometry("geometry")
#                 .set_crs(epsg=4328, inplace=True, allow_override=True)
#             )
#         return gdf

#     def convert_area_gdf(self, gdf: gpd.GeoDataFrame) -> GeoDataFrame:
#         """Convert area gdf to spatialpandas gdf."""
#         if _has_spatialpandas:
#             if isinstance(gdf, gpd.GeoDataFrame):
#                 gdf = GeoDataFrame(gdf)
#             else:
#                 raise TypeError(
#                     f"input gdf has to be a geopandas.GeoDataFrame. Got: {type(gdf)}"
#                 )
#         else:
#             raise ImportError(
#                 "spatialpandas is required to use the spatial context backend."
#                 "with backend == 'spatialpandas'. Install it with "
#                 "`pip install spatialpandas`."
#             )

#         return gdf

#     def convert_cell_gdf(
#         self, gdf: gpd.GeoDataFrame, parallel: bool, n_partitions: int = None
#     ) -> Union[GeoDataFrame, DaskGeoDataFrame]:
#         """Convert cell gdf to spatialpandas gdf.

#         Parameters
#         ----------
#         gdf : gpd.GeoDataFrame
#             The cell gdf to convert.
#         parallel : bool
#             Whether to convert to a DaskGeoDataFrame.
#         n_partitions : int, optional
#             The number of processes/partitions to use for the DaskGeoDataFrame. If None,
#             the number of partitions is set to the number of cores.

#         Returns
#         -------
#         Union[GeoDataFrame, DaskGeoDataFrame]:
#             The converted cell gdf.
#         """
#         if _has_spatialpandas:
#             if isinstance(gdf, gpd.GeoDataFrame):
#                 # spatialpandas can use only Point arrays in sjoin
#                 gdf["centroid"] = gdf.centroid
#                 gdf = gdf.set_geometry("centroid")
#                 gdf = GeoDataFrame(gdf)
#             else:
#                 raise TypeError(
#                     f"input gdf has to a geopandas.GeoDataFrame. Got: {type(gdf)}"
#                 )
#         else:
#             raise ImportError(
#                 "spatialpandas is required to use the spatial context backend."
#                 "with backend == 'spatialpandas'. Install it with "
#                 "`pip install spatialpandas`"
#             )

#         if parallel:
#             if _has_dask and _has_spatialpandas:
#                 print(
#                     f"partitioning cell_gdf into {n_partitions} partitions. This can "
#                     "take a few minutes...",
#                 )
#                 gdf = dd.from_pandas(gdf, npartitions=n_partitions).persist()
#                 print("partitioning done.")

#                 # Precompute the partition-level spatial index
#                 gdf.partition_sindex
#             else:
#                 raise ImportError(
#                     "spatialpandas and dask is required to use the spatial context "
#                     "backend. with backend == 'spatialpandas' and parallel==True. "
#                     "Install with `pip install spatialpandas` and `pip install dask`."
#                 )
#         return gdf
