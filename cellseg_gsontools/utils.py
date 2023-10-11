import warnings
from pathlib import Path
from typing import Dict, Tuple, Union

import geopandas as gpd
import pandas as pd
import shapely
from pandas.api.types import (
    is_bool_dtype,
    is_categorical_dtype,
    is_object_dtype,
    is_string_dtype,
)
from shapely.geometry import Polygon

from .apply import gdf_apply

__all__ = ["set_uid", "read_gdf", "pre_proc_gdf", "clip_gdf", "is_categorical"]


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

    NOTE: by default sets a running index column to gdf as the uid.

    Parameters
    ----------
        gdf : gpd.GeoDataFrame
            Input Geodataframe.
        start_ix : int, default=0
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


def _get_class(property_dict: Dict) -> str:
    """Return the class of the gdf."""
    if "classification" in property_dict.keys():
        return property_dict["classification"]["name"]
    return property_dict["name"]


def _get_prob(property_dict: Dict) -> str:
    """Return the class probabilities of the gdf."""
    if "classification" in property_dict.keys():
        return property_dict["classification"]["probabilities"]
    return property_dict["probabilities"]


def read_gdf(
    fname: Union[Path, str],
    silence_warnigns: bool = True,
) -> gpd.GeoDataFrame:
    """Read a file into geodataframe.

    Allowed formats:
    ".json", ".geojson", ".feather", ".parquet"

    Parameters
    ----------
        fname : Union[Path, str]
            The filename of the gson file.
        silence_warnings : bool, optional
            Whether to silence warnings, by default True.

    Raises
    ------
        ValueError: If suffix is not one of ".json", ".geojson", ".feather", ".parquet".

    Returns
    -------
        gpd.GeoDataFrame:
            The geodataframe.

    Examples
    --------
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
        df = gpd.read_file(fname)
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
            df["class_name"] = gdf_apply(df, _get_class, col=property_col)
        except KeyError:
            if not silence_warnigns:
                warnings.warn(
                    f"Could not find 'name' key in {property_col} column."
                    "Can't set the `class_name` column to the output gdf."
                )

    if "class_probs" not in df.columns:
        try:
            df["class_probs"] = gdf_apply(df, _get_prob, col=property_col)
        except KeyError:
            if not silence_warnigns:
                warnings.warn(
                    f"Could not find 'probabilities' key in {property_col} column. "
                    "Can't set the `class_probs` column to the output gdf."
                )

    df["geometry"] = gdf_apply(df, shapely.geometry.shape, col="geometry")
    return gpd.GeoDataFrame(df).set_geometry("geometry")


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
