import warnings

import geopandas as gpd
import pandas as pd
import shapely

__all__ = ["set_uid", "read_gdf", "pre_proc_gdf"]


def set_uid(
    gdf: gpd.GeoDataFrame, id_col: str = "uid", drop: bool = False
) -> gpd.GeoDataFrame:
    """Set a unique identifier column to gdf.

    NOTE: by default sets a running index column to gdf as the uid.

    Parameters
    ----------
        gdf : gpd.GeoDataFrame
            Input Geodataframe
        id_col : str, default="uid"
            The name of the column that will be used or set to the id.
        drop : bool, default=False
            Drop the column after it is added to index.

    Returns
    -------
        gpd.GeoDataFrame:
            The inputr gdf with a "uid" column added to it
    """
    allowed = list(gdf.columns) + ["uid", "id"]
    if id_col not in allowed:
        raise ValueError(f"Illegal `id_col`. Got: {id_col}. Allowed: {allowed}.")

    gdf = gdf.copy()
    if id_col in ("uid", "id"):
        gdf[id_col] = range(1, len(gdf) + 1)

    gdf = gdf.set_index(id_col, drop=drop)

    return gdf


def read_gdf(fname: str) -> gpd.GeoDataFrame:
    """Read a gson file into geodataframe."""
    gdf = pd.read_json(fname)
    gdf["geometry"] = gdf["geometry"].apply(shapely.geometry.shape)
    gdf = gpd.GeoDataFrame(gdf).set_geometry("geometry")

    # add class name column
    gdf["class_name"] = gdf["properties"].apply(lambda x: x["classification"]["name"])

    return gdf


def pre_proc_gdf(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Apply some light pre-processing of a geodataframe.

    Namely, remove invalid polygons, empty geometries and add bounds to the gdf.
    """
    # drop invalid geometries if there are any after buffer
    gdf.geometry = gdf.geometry.buffer(0)
    gdf = gdf[gdf.is_valid]

    # drop empty geometries
    gdf = gdf[~gdf.is_empty]

    # drop geometries that are not polygons
    gdf = gdf[gdf.geom_type == "Polygon"]

    try:
        # add bounding box coords of the polygons to the gdfs
        # and correct for the max coords
        gdf["xmin"] = gdf.bounds["minx"].astype(int)
        gdf["ymin"] = gdf.bounds["miny"].astype(int)
        gdf["ymax"] = gdf.bounds["maxy"].astype(int) + 1
        gdf["xmax"] = gdf.bounds["maxx"].astype(int) + 1
    except Exception:
        warnings.warn("Could not create bounds cols to gdf.", RuntimeWarning)

    return gdf
