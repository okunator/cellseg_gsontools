import re
from pathlib import Path
from typing import List, Tuple, Union

import geopandas as gpd

from cellseg_gsontools.apply import gdf_apply

__all__ = [
    "check_format",
    "get_xy_coords",
    "get_file_from_coords",
    "gdf_to_file",
]


def _add_qupath_classification(class_name: str) -> gpd.GeoSeries:
    """Add QuPath properties to a row of a gdf."""
    props = {"name": class_name, "color": None}
    return props


def gdf_to_file(
    gdf: gpd.GeoDataFrame,
    out_fn: Union[str, Path],
    format: str = ".feather",
) -> None:
    """Write a geojson/feather/parquet file from a gdf.

    This is wrapper around `geopandas.GeoDataFrame` I/O methods
    that adds some extra functionality.

    Parameters:
        gdf (gpd.GeoDataFrame):
            The input gdf.
        out_fn (Union[str, Path]):
            The output filename.
        format (str):
            The output format. One of ".feather", ".parquet", ".geojson".

    Raises:
        ValueError: If `format` is not one of ".feather", ".geojson", ".parquet".
        ValueError: If the input gdf does not have a "class_name" column.

    Examples:
        Write a geojson file.
        >>> from cellseg_gsontools import gdf_to_file
        >>> gdf_to_file(gdf, "out.geojson")
    """
    out_fn = Path(out_fn)
    if format not in (".feather", ".parquet", ".geojson", None):
        raise ValueError(
            f"Invalid format. Got: {format}. Allowed: .feather, .parquet, .geojson"
        )

    if "class_name" not in gdf.columns:
        raise ValueError("The input gdf needs to have a 'class_name' column.")

    # add objectType col (QuPath)
    if "objectType" not in gdf.columns:
        gdf["objectType"] = "annotation"

    # add classification col (QuPath)
    if "classification" not in gdf.columns:
        gdf["classification"] = gdf_apply(
            gdf, _add_qupath_classification, axis=1, columns=["class_name"]
        )

    if format == ".feather":
        gdf.to_feather(out_fn.with_suffix(".feather"))
    elif format == ".parquet":
        gdf.to_parquet(out_fn.with_suffix(".parquet"))
    elif format == ".geojson":
        gdf.to_file(out_fn.with_suffix(".geojson"), driver="GeoJSON", index=False)


def check_format(fname: Union[Path, str]) -> None:
    """Check if the input file has the correct format.

    Parameters
        fname (str):
            The filename.

    Raises:
        ValueError: If not all coordinates were found in filename.
        ValueError: If both x and y coordinates are not present in filename.
    """
    fn = Path(fname)
    if fn.suffix not in (".json", ".geojson", ".png", ".feather", ".parquet"):
        raise ValueError(
            f"Input file {fn} has wrong format. Expected '.png', '.json' or '.geojson'."
        )

    has_x = False
    has_y = False

    # get the x and y coordinates from the filename
    # NOTE: fname needs to contain x & y-coordinates in x_[coord1]_y_[coord2]-format
    # or x-[coord1]_y-[coord2]-format. The order of x and y can be any.
    xy_str: List[str] = re.findall(r"x\d+|y\d+|x_\d+|y_\d+|x-\d+|y-\d+", fn.as_posix())

    try:
        for s in xy_str:
            if "x" in s:
                has_x = True
            elif "y" in s:
                has_y = True
    except IndexError:
        raise ValueError(
            "Not all coordinates were found in filename. "
            f"Filename has to be in 'x-[coord1]_y-[coord2]'-format, Got: {fn.name}"
        )

    if not has_x or not has_y:
        raise ValueError(
            "Both x and y coordinates have to be present in filename. "
            f"Got: {xy_str}. Filename has to be in 'x-[coord1]_y-[coord2]'-format."
        )

    return


def get_xy_coords(fname: Union[Path, str]) -> Tuple[int, int]:
    """Get the x and y-coordinates from a filename.

    Note:
        The filename needs to contain x & y-coordinates in
        "x-[coord1]_y-[coord2]"-format

    Parameters:
        fname (str):
            The filename. Has to contain x & y-coordinates

    Raises:
        ValueError: If not the delimeter of x and y- coord is not '_' or '-'.

    Returns:
        Tuple[int, int]: The x and y-coordinates in this order.
    """
    check_format(fname)

    if isinstance(fname, Path):
        fname = fname.as_posix()

    xy_str: List[str] = re.findall(r"x\d+|y\d+|x_\d+|y_\d+|x-\d+|y-\d+", fname)
    xy: List[int] = [0, 0]
    for s in xy_str:
        if "x" in s:
            if "_" in s:
                xy[0] = int(s.split("_")[1])
            elif "-" in s:
                xy[0] = int(s.split("-")[1])
            elif "x" in s and "-" not in s and "_" not in s:
                xy[0] = int(s.split("x")[1])
            else:
                raise ValueError(
                    "The fname needs to contain x & y-coordinates in "
                    f"'x-[coord1]_y-[coord2]'-format. Got: {fname}"
                )
        elif "y" in s:
            if "_" in s:
                xy[1] = int(s.split("_")[1])
            elif "-" in s:
                xy[1] = int(s.split("-")[1])
            elif "y" in s and "-" not in s and "_" not in s:
                xy[1] = int(s.split("y")[1])
            else:
                raise ValueError(
                    "The fname needs to contain x & y-coordinates in "
                    f"'x-[coord1]_y-[coord2]'-format. Got: {fname}"
                )

    return xy[0], xy[1]


def get_file_from_coords(
    files: List[Union[Path, str]], x: int, y: int
) -> Union[str, None]:
    """Get the correct file name from a list of filenames given x and y-coords.

    Note:
        fname needs to contain x & y-coordinates in  "x-[coord1]_y-[coord2]"-format.

    Parameters:
        files (List[Union[Path, str]]): A list of paths to the files.
        x (int): x-coord in pixels.
        y (int): y-coord in pixels.

    Returns:
        str or None: returns the file name if it exists in the given input dir, else None.
    """
    for f in files:
        if get_xy_coords(f) == (x, y):
            return f

    return
