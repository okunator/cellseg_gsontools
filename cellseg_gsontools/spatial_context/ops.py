import warnings
from typing import Union

import geopandas as gpd
from geopandas.tools import sjoin

__all__ = ["get_interface_zones", "get_objs"]


def get_objs(
    area: gpd.GeoDataFrame,
    objects: gpd.GeoDataFrame,
    silence_warnings: bool = True,
    predicate: str = "within",
) -> Union[gpd.GeoDataFrame, None]:
    """Get the objects within the area.

    Parameters
    ----------
        area : gpd.GeoDataFrame
            The area of interest in GeoDataFrame.
        objects : gpd.GeoDataFrame
            The objects (cells) of interest.
        silence_warnings : bool, default=True
            Flag, whether to suppress warnings.
        predicate : str, default="within"
            The predicate to use for the spatial join. See `geopandas.tools.sjoin`

    Returns
    -------
        gpd.GeoDataFrame:
            The objects (cells) within the area gdf.
    """
    # NOTE, gdfs need to have same crs, otherwise warning flood.
    objs_within: gpd.GeoDataFrame = sjoin(
        right_df=area, left_df=objects, how="inner", predicate=predicate
    )

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

    Parameters
    ----------
        buffer_area : gpd.GeoDataFrame
            The area or region of interest that is buffered on top of polygons in gdf.
        areas : gpd.GeoDataFrame
            A geodataframe containing polygons (tissue areas) that might intersect
            with the `buffer_area`.
        buffer_dist : int, default=200
            The radius of the buffer.

    Returns
    -------
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
            inter = inter.overlay(buffer_area, how="difference")

    return inter
