import warnings
from pathlib import Path
from typing import Optional, Tuple, Union

import geopandas as gpd
import pandas as pd
import shapely
from shapely.geometry import Polygon

__all__ = ["set_uid", "read_gdf", "pre_proc_gdf", "clip_gdf"]


def set_uid(
    gdf: gpd.GeoDataFrame, start_ix: int = 1, id_col: str = "uid", drop: bool = False
) -> gpd.GeoDataFrame:
    """Set a unique identifier column to gdf.

    NOTE: by default sets a running index column to gdf as the uid.

    Parameters
    ----------
        gdf : gpd.GeoDataFrame
            Input Geodataframe.
        start_ix : int, default=1
            The starting index of the id column.
        id_col : str, default="uid"
            The name of the column that will be used or set to the id.
        drop : bool, default=False
            Drop the column after it is added to index.

    Returns
    -------
        gpd.GeoDataFrame:
            The inputr gdf with a "uid" column added to it
    """
    gdf = gdf.copy()
    if id_col not in gdf.columns:
        gdf[id_col] = range(start_ix, len(gdf) + start_ix)

    gdf = gdf.set_index(id_col, drop=drop)

    return gdf


def read_gdf(
    fname: Union[Path, str],
    qupath_format: Optional[str] = None,
    suppress_warnigns: bool = True,
) -> gpd.GeoDataFrame:
    """Read a file into geodataframe.

    Allowed formats:
    ".json", ".geojson", ".feather", ".parquet"

    Parameters
    ----------
        fname : Union[Path, str]
            The filename of the gson file.
        qupath_format : str, optional
            One of: "old", "latest", None.

    Raises
    ------
        ValueError: If suffix is not one of ".json", ".geojson", ".feather", ".parquet".
        ValueError: If `qupath_format` is not one of "old", "latest", None.

    Returns
    -------
        gpd.GeoDataFrame:
            The geodataframe.

    Examples
    --------
    Read a geojson file that is QuPath-readable.
        >>> from cellseg_gsontools.utils import read_gdf
        >>> gdf = read_gdf(
        ...    "path/to/file.json", qupath_format="latest"
        ... )

    """
    fname = Path(fname)
    format = fname.suffix
    allowed_formats = (".json", ".geojson", ".feather", ".parquet")
    if format not in allowed_formats:
        raise ValueError(
            f"Illegal `format`. Got: {format}. Allowed: {allowed_formats}."
        )

    allowed_qupath_formats = ("old", "latest", None)
    if qupath_format not in allowed_qupath_formats:
        raise ValueError(
            f"Illegal `qupath_format`. Got: {qupath_format}. "
            f"Allowed: {allowed_qupath_formats}."
        )

    if format in (".geojson", ".json"):
        if qupath_format == "old":
            gdf = pd.read_json(fname)

            if gdf.empty or gdf is None:
                if not suppress_warnigns:
                    warnings.warn(
                        f"Empty geojson file: {fname.name}. Returning empy gdf."
                    )
                return gpd.GeoDataFrame()

            gdf["geometry"] = gdf["geometry"].apply(shapely.geometry.shape)
            gdf = gpd.GeoDataFrame(gdf).set_geometry("geometry")

            # add class name column
            gdf["class_name"] = gdf["properties"].apply(
                lambda x: x["classification"]["name"]
            )
        elif qupath_format == "latest":
            gdf = gpd.read_file(fname)
            gdf["geometry"] = gdf["geometry"].apply(shapely.geometry.shape)
            gdf = gpd.GeoDataFrame(gdf).set_geometry("geometry")
            gdf["class_name"] = gdf["classification"].apply(lambda x: x["name"])
        else:
            gdf = gpd.read_file(fname)
    elif format == ".feather":
        gdf = gpd.read_feather(fname)
    elif format == ".parquet":
        gdf = gpd.read_parquet(fname)

    if gdf.empty:
        warnings.warn(f"Empty geojson file: {fname.name}. Returning empty gdf.")

    return gdf


def pre_proc_gdf(
    gdf: gpd.GeoDataFrame, min_size: int = None
) -> Union[gpd.GeoDataFrame, None]:
    """Apply some light pre-processing of a geodataframe.

    Namely, remove invalid polygons, empty geometries and add bounds to the gdf.

    Parameters
    ----------
        gdf : gpd.GeoDataFrame
            Input geodataframe.
        min_size : int, optional
            The minimum size of the polygons in pixels.

    Returns
    -------
        gpd.GeoDataFrame:
            The pre-processed gdf or None if input gdf is empty or None.
    """
    if gdf.empty or gdf is None:
        return gdf

    # drop invalid geometries if there are any after buffer
    gdf.geometry = gdf.geometry.buffer(0)
    gdf = gdf[gdf.is_valid]

    # drop empty geometries
    gdf = gdf[~gdf.is_empty]

    # if there are multipolygon geometries, explode them
    if "MultiPolygon" in list(gdf["geometry"].geom_type):
        gdf = gdf.explode(index_parts=False).reset_index(drop=True)

    # drop geometries that are less than min_size pixels
    if min_size is not None:
        gdf = gdf[gdf.area > min_size]

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


def clip_gdf(
    gdf: gpd.GeoDataFrame, bbox: Tuple[int, int, int, int]
) -> gpd.GeoDataFrame:
    """Clip a gdf to a bounding box.

    Parameters
    ----------
        gdf : gpd.GeoDataFrame
            Input geodataframe.
        bbox : Tuple[int, int, int, int]
            Bounding box to clip to. Format: (xmin, ymin, xmax, ymax).

    Returns
    -------
        gpd.GeoDataFrame:
            Clipped gdf.
    """
    xmin = bbox[0]
    ymin = bbox[1]
    xmax = bbox[2]
    ymax = bbox[3]

    crop = Polygon(
        [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin), (xmin, ymin)]
    )

    return gdf.clip(crop)
