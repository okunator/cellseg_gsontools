import warnings
from typing import Union

import geopandas as gpd
import numpy as np

__all__ = ["get_interface_zones", "get_objs"]


def get_objs(
    area: gpd.GeoDataFrame,
    objects: gpd.GeoDataFrame,
    silence_warnings: bool = True,
    predicate: str = "within",
) -> Union[gpd.GeoDataFrame, None]:
    """Get the objects within the area.

    Parameters:
        area (gpd.GeoDataFrame):
            The area of interest in GeoDataFrame.
        objects (gpd.GeoDataFrame):
            The objects (cells) of interest.
        silence_warnings (bool):
            Flag, whether to suppress warnings.
        predicate (str):
            The predicate to use for the spatial join. See `geopandas.tools.sjoin`

    Returns:
        gpd.GeoDataFrame:
            The objects (cells) within the area gdf.
    """
    # NOTE, gdfs need to have same crs, otherwise warning flood.
    inds = objects.geometry.sindex.query(area.iloc[0].geometry, predicate=predicate)
    objs_within: gpd.GeoDataFrame = objects.iloc[np.unique(inds)]

    if objs_within.empty and not silence_warnings:
        warnings.warn(
            """`get_objs_within` resulted in an empty GeoDataFrame. No objects were
            found within the area. Returning None from `get_objs_within`.,
            """,
            RuntimeWarning,
        )
        return

    return objs_within


def get_interface_zones(
    buffer_area: gpd.GeoDataFrame, areas: gpd.GeoDataFrame, buffer_dist: int = 200
) -> gpd.GeoDataFrame:
    """Get the interfaces b/w the polygons defined in a `areas `gdf and `buffer_area`.

    Interface is the region around the border of two touching polygons. The interface
    radius is determined by the `buffer_dist` parameter.

    Applies a buffer to the `buffer_area` and finds the intersection between the buffer
    and the polygons in `areas` gdf.

    Useful for example, when you want to extract the interface of two distinct tissues
    like stroma and cancer.

    Parameters:
        buffer_area (gpd.GeoDataFrame):
            The area or region of interest that is buffered on top of polygons in gdf.
        areas (gpd.GeoDataFrame):
            A geodataframe containing polygons (tissue areas) that might intersect
            with the `buffer_area`.
        buffer_dist (int):
            The radius of the buffer.

    Returns:
        gpd.GeoDataFrame:
            A geodataframe containing the intersecting polygons including the buffer.
    """
    buffer_zone = gpd.GeoDataFrame(
        {"geometry": list(buffer_area.buffer(buffer_dist))},
        crs=buffer_area.crs,
    )
    inter = areas.overlay(buffer_zone, how="intersection")

    # if the intersecting area is covered totally by any polygon in the `areas` gdf
    # take the difference of the intresecting area and the orig roi to discard
    # the roi from the interface 'sheet'
    if not inter.empty:
        if areas.covers(inter.geometry.loc[0]).any():  # len(inter) == 1
            inter = inter.overlay(buffer_area, how="difference", keep_geom_type=True)

    return inter


# try:
#     import dask_geopandas

#     _has_dask_geopandas = True
# except ImportError:
#     _has_dask_geopandas = False


# try:
#     from spatialpandas import GeoDataFrame
#     from spatialpandas import sjoin as sp_sjoin
#     from spatialpandas.dask import DaskGeoDataFrame

#     _has_spatialpandas = True
# except ImportError:
#     _has_spatialpandas = False

# try:
#     import dask.distributed  # noqa

#     _has_dask = True
# except ImportError:
#     _has_dask = False
# def get_objs_sp(
#     area: GeoDataFrame,
#     objects: GeoDataFrame,
#     silence_warnings: bool = False,
#     predicate: str = "intersects",
# ) -> GeoDataFrame:
#     """Get objects within the given area.

#     Note:
#         This function is a wrapper around the spatialpandas sjoin function.
#     Note:
#         Optionally parallel execution with Dask.

#     Parameters:
#         area (gpd.GeoDataFrame):
#             The area of interest.
#         objects (gpd.GeoDataFrame):
#             The objects to filter.
#         silence_warnings (bool):
#             Whether to silence warnings.
#         predicate (str):
#             The spatial predicate to use for the spatial join. One of "intersects"

#     Returns:
#         spatialpandas.GeoDataFrame:
#             The objects within the given area. NOTE: This is a spatialpandas GeoDataFrame.
#     """
#     if not _has_spatialpandas:
#         raise ImportError(
#             "spatialpandas not installed. Please install spatialpandas to use this "
#             "function. `pip install spatialpandas`"
#         )

#     if isinstance(objects, DaskGeoDataFrame) and not _has_dask:
#         raise ImportError(
#             "Dask not installed. Please install Dask to use this function in parallel. "
#             '`python -m pip install "dask[distributed]"`'
#         )

#     # run spatialpandas sjoin
#     objs_within = sp_sjoin(objects, area, how="inner", op=predicate)

#     # if dask, compute
#     if isinstance(objects, DaskGeoDataFrame):
#         objs_within = objs_within.compute()

#     if len(objs_within.index) == 0 and not silence_warnings:
#         warnings.warn(
#             """`get_objs_sp` resulted in an empty GeoDataFrame. No objects were
#             found within the area. Returning None from `get_objs_sp`.,
#             """,
#             RuntimeWarning,
#         )
#         return

#     return objs_within


# def get_objs_dgp(
#     area: gpd.GeoDataFrame,
#     objects: dask_geopandas.core.GeoDataFrame,
#     silence_warnings: bool = False,
#     predicate: str = "intersects",
# ) -> GeoDataFrame:
#     """Get objects within the given area.

#     Note:
#         This function is a wrapper around the dask_geopandas sjoin function.

#     Parameters:
#         area (gpd.GeoDataFrame):
#             The area of interest.
#         objects (dask_geopandas.core.GeoDataFrame):
#             The objects to filter.
#         silence_warnings (bool):
#             Whether to silence warnings.
#         predicate (str):
#             The spatial predicate to use for the spatial join. One of "intersects"

#     Returns:
#         gpd.GeoDataFrame:
#             The objects within the given area. NOTE: This is a spatialpandas GeoDataFrame.
#     """
#     if not _has_dask_geopandas:
#         raise ImportError(
#             "dask-geopandas not installed. Please install spatialpandas to use this "
#             "function. `pip install dask-geopandas`"
#         )

#     # run spatialpandas sjoin
#     objs_within = dask_geopandas.sjoin(objects, area, how="inner", predicate=predicate)
#     objs_within: gpd.GeoDataFrame = objs_within.compute()

#     if len(objs_within.index) == 0 and not silence_warnings:
#         warnings.warn(
#             """`get_objs_sp` resulted in an empty GeoDataFrame. No objects were
#             found within the area. Returning None from `get_objs_sp`.,
#             """,
#             RuntimeWarning,
#         )
#         return

#     return objs_within
