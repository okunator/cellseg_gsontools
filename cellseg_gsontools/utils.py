import warnings
from pathlib import Path
from typing import Dict, Sequence, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from pandas.api.types import (
    is_bool_dtype,
    is_categorical_dtype,
    is_object_dtype,
    is_string_dtype,
)
from pyproj import CRS, Transformer
from shapely.geometry import Polygon

from .apply import gdf_apply

__all__ = [
    "set_uid",
    "read_gdf",
    "pre_proc_gdf",
    "clip_gdf",
    "is_categorical",
    "get_holes",
    "xy_to_lonlat",
    "lonlat_to_xy",
]


def is_categorical(col: pd.Series) -> bool:
    """Check if a column is categorical."""
    return (
        is_categorical_dtype(col)
        or is_string_dtype(col)
        or is_object_dtype(col)
        or is_bool_dtype(col)
    )


def set_uid(
    gdf: gpd.GeoDataFrame, start_ix: int = 0, id_col: str = "uid", drop: bool = False
) -> gpd.GeoDataFrame:
    """Set a unique identifier column to gdf.

    Note:
        by default sets a running index column to gdf as the uid.

    Parameters:
        gdf (gpd.GeoDataFrame):
            Input Geodataframe.
        start_ix (int, default=0):
            The starting index of the id column.
        id_col (str, default="uid"):
            The name of the column that will be used or set to the id.
        drop (bool, default=False):
            Drop the column after it is added to index.

    Returns:
        gpd.GeoDataFrame:
            The input gdf with a "uid" column added to it.

    Examples:
        >>> from cellseg_gsontools import set_uid
        >>> gdf = set_uid(gdf, drop=True)
    """
    gdf = gdf.copy()
    if id_col not in gdf.columns:
        gdf[id_col] = range(start_ix, len(gdf) + start_ix)

    gdf = gdf.set_index(id_col, drop=drop)

    return gdf


def _set_cols(property_dict: Dict):
    return property_dict["properties"]


def _set_geom(property_dict: Dict):
    return property_dict["geometry"]


def _set_gdf(gdf):
    gdf["geometry"] = gdf_apply(gdf, _set_geom, columns=["features"], parallel=False)
    gdf["properties"] = gdf_apply(gdf, _set_cols, columns=["features"], parallel=False)
    return gdf


def _get_class(property_dict: Dict) -> str:
    """Return the class of the gdf."""
    try:
        if "classification" in property_dict.keys():
            return property_dict["classification"]["name"]
        return property_dict["name"]
    except Exception as e:
        warnings.warn(f"Could not set a class to annotation due to a Error: {e}.")
        return


def _get_prob(property_dict: Dict) -> str:
    """Return the class probabilities of the gdf."""
    if "classification" in property_dict.keys():
        return property_dict["classification"]["probabilities"]
    return property_dict["probabilities"]


def read_gdf(
    fname: Union[Path, str],
    silence_warnigns: bool = True,
) -> gpd.GeoDataFrame:
    """Read a file into a geodataframe.

    This is a wrapper around `geopandas` I/O that adds some extra
    functionality.

    Note:
        Allowed formats:

        - `.json`,
        - `.geojson`,
        - `.feather`,
        - `.parquet`

    Parameters:
        fname (Union[Path, str]):
            The filename of the gson file.
        silence_warnigns (bool):
            Whether to silence warnings, by default True.

    Raises:
        ValueError:
            If suffix is not one of ".json", ".geojson", ".feather", ".parquet".

    Returns:
        gpd.GeoDataFrame:
            The geodataframe.

    Examples:
        Read a geojson file that is QuPath-readable.
        >>> from cellseg_gsontools.utils import read_gdf
        >>> gdf = read_gdf("path/to/file.json")
    """
    fname = Path(fname)
    format = fname.suffix
    allowed_formats = (".json", ".geojson", ".feather", ".parquet")
    if format not in allowed_formats:
        raise ValueError(
            f"Illegal `format`. Got: {format}. Allowed: {allowed_formats}."
        )

    if format == ".json":
        df = pd.read_json(fname)
    elif format == ".geojson":
        try:
            df = gpd.read_file(fname)
        except Exception:
            df = pd.read_json(fname)
            df = _set_gdf(df)
    elif format == ".feather":
        df = gpd.read_feather(fname)
    elif format == ".parquet":
        df = gpd.read_parquet(fname)

    if df.empty:
        if not silence_warnigns:
            warnings.warn(f"Empty geojson file: {fname.name}. Returning empty gdf.")
        return df

    property_col = "properties" if "properties" in df.columns else "classification"

    if "class_name" not in df.columns:
        try:
            df["class_name"] = gdf_apply(df, _get_class, columns=[property_col])
        except Exception:
            if not silence_warnigns:
                warnings.warn(
                    f"Could not find 'name' key in {property_col} column."
                    "Can't set the `class_name` column to the output gdf."
                )

    if "class_probs" not in df.columns:
        try:
            df["class_probs"] = gdf_apply(df, _get_prob, columns=[property_col])
        except Exception:
            if not silence_warnigns:
                warnings.warn(
                    f"Could not find 'probabilities' key in {property_col} column. "
                    "Can't set the `class_probs` column to the output gdf."
                )

    df["geometry"] = gdf_apply(df, shapely.geometry.shape, columns=["geometry"])
    return gpd.GeoDataFrame(df).set_geometry("geometry")


def pre_proc_gdf(
    gdf: gpd.GeoDataFrame, min_size: int = None
) -> Union[gpd.GeoDataFrame, None]:
    """Apply some light pre-processing of a geodataframe.

    Namely, remove invalid polygons, empty geometries and add bounds to the gdf.

    Parameters:
        gdf (gpd.GeoDataFrame):
            Input geodataframe.
        min_size (int, optional):
            The minimum size of the polygons in pixels.

    Returns:
        gpd.GeoDataFrame:
            The pre-processed gdf or None if input gdf is empty or None.

    Examples:
        >>> from cellseg_gsontools import pre_proc_gdf
        >>> gdf = pre_proc_gdf(gdf, min_size=100)
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

    Parameters:
        gdf (gpd.GeoDataFrame):
            Input geodataframe.
        bbox (Tuple[int, int, int, int]):
            Bounding box to clip to. Format: (xmin, ymin, xmax, ymax).

    Returns:
        gpd.GeoDataFrame:
            The Clipped gdf.

    Examples:
        >>> from cellseg_gsontools import clip_gdf
        >>> gdf = clip_gdf(gdf, bbox=(0, 0, 100, 100))
    """
    xmin = bbox[0]
    ymin = bbox[1]
    xmax = bbox[2]
    ymax = bbox[3]

    crop = Polygon(
        [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin), (xmin, ymin)]
    )

    return gdf.clip(crop)


def xy_to_lonlat(
    x: Union[float, Sequence], y: Union[float, Sequence]
) -> Tuple[Union[float, Sequence], Union[float, Sequence]]:
    """Converts x, y coordinates to lon, lat coordinates.

    Parameters:
        x (Union[float, Sequence]):
            x coordinate(s).
        y (Union[float, Sequence]):
            y coordinate(s).

    Returns:
        Tuple[Union[float, Sequence], Union[float, Sequence]]:
            lon, lat coordinates.

    Examples:
        >>> from cellseg_gsontools import xy_to_lonlat
        >>> from shapely.geometry import Polygon
        >>> poly = Polygon([(0, 0), (0, 1), (1, 1)])
        >>> x, y = poly.exterior.coords.xy
        >>> lon, lat = xy_to_lonlat(x, y)
        >>> lon, lat
        (array('d', [10.511, 10.511, 10.511, 10.511]),
         array('d', [0.0, 9.019e-06, 9.019e-06, 0.0]))
    """
    crs_utm = CRS(proj="utm", zone=33, ellps="WGS84")
    crs_latlon = CRS(proj="latlong", zone=33, ellps="WGS84")
    transformer = Transformer.from_crs(crs_utm, crs_latlon, always_xy=True)
    lonlat = transformer.transform(x, y)

    return lonlat[0], lonlat[1]


def lonlat_to_xy(
    lon: Union[float, Sequence], lat: Union[float, Sequence]
) -> Tuple[Union[float, Sequence], Union[float, Sequence]]:
    """Converts lon, lat coordinates to x, y coordinates.

    Parameters:
        lon (Union[float, Sequence]):
            Longitude coordinate(s).
        lat (Union[float, Sequence]):
            Latitude coordinate(s).

    Returns:
        Tuple[Union[float, Sequence], Union[float, Sequence]]:
            x, y coordinates.

    Examples:
        >>> from shapely.geometry import Polygon
        >>> from cellseg_gsontools import lonlat_to_xy
        >>> poly = Polygon([(10, 10), (10, 0), (20, 10)])
        >>> lon, lat = poly.exterior.coords.xy
        >>> x, y = lonlat_to_xy(lon, lat)
        >>> x, y
        (array('d', [-48636.648, -57087.120, 1048636.648, -48636.648]),
         array('d', [1109577.311, 0.0, 1109577.311, 1109577.311]))
    """
    crs_utm = CRS(proj="utm", zone=33, ellps="WGS84")
    crs_latlon = CRS(proj="latlong", zone=33, ellps="WGS84")
    transformer = Transformer.from_crs(crs_latlon, crs_utm, always_xy=True)
    xy = transformer.transform(lon, lat)

    return xy[0], xy[1]


def get_holes(poly: Polygon, to_lonlat: bool = True) -> Sequence[Sequence[float]]:
    """Get holes from a shapely Polygon.

    Parameters:
        poly (Polygon):
            Polygon to get holes from.
        to_lonlat (bool, optional):
            Whether to convert to lonlat coordinates, by default True.

    Returns:
        Sequence[Sequence[float]]:
            A list of xy coordinate tuples.

    Examples:
        >>> from cellseg_gsontools import get_holes
        >>> holes = get_holes(poly)
    """
    holes = []
    for interior in poly.interiors:
        x, y = interior.coords.xy
        if to_lonlat:
            x, y = xy_to_lonlat(x, y)
        holes.append(list(zip(x, y)))

    return holes


def gdf_slide_query(
    gdf: gpd.GeoDataFrame,
    slide_path: Union[Path, str],
    width: int = None,
    height: int = None,
    level: int = 0,
) -> np.ndarray:
    """Queries a slide based on the total bound coordinates of the input gdf.

    Parameters:
        gdf (gpd.GeoDataFrame):
            GeoDataFrame with a "class_name" column.
        slide_path (Union[Path, str]):
            Path to the slide file.
        width (int):
            Width of the query area. If None, the width will be calculated from the input gdf.
        height (int):
            Height of the query area. If None, the height will be calculated from the input gdf.
        level (int):
            Level of the slide to query.

    Returns:
        np.ndarray:
            Image of the query area.
    """
    try:
        from histoprep import SlideReader
    except ImportError:
        raise ImportError(
            "The histoprep package is required for this function. "
            "Please install it via `pip install histoprep`. Requires openslide"
        )
    reader = SlideReader(Path(slide_path))

    # get slide bounds since there is possible offset
    offset_x, offset_y, _, _ = reader.data_bounds
    xmin, ymin, xmax, ymax = gdf.total_bounds
    xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)

    if width is None or height is None:
        width = xmax - xmin
        height = ymax - ymin

    im = reader.read_region(
        (xmin + offset_x, ymin + offset_y, width, height), level=level
    )

    return im
