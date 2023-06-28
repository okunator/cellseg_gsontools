import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import geopandas as gpd

from ..apply import gdf_apply

__all__ = [
    "save_geojson",
    "check_format",
    "get_xy_coords",
    "get_file_from_coords",
    "row_to_qupath",
    "add_qupath_props",
    "add_qupath_classification",
    "gdf_to_file",
]


def save_geojson(annotations: List[Dict[str, Any]], fname: Union[Path, str]) -> None:
    """Save a list of geojson dict objects to a '.json' file.

    Parameters
    ----------
        annotations : List[Dict[str, Any]]
            A list of geojson dict objects.
        fname : Union[Path, str]
            The filename.
    """
    try:
        import geojson
    except ImportError:
        raise ImportError(
            "You need to install `geojson` to use `save_geojson` function. "
            "Try `pip install geojson`."
        )

    fname = Path(fname).with_suffix(".json")
    with fname.open("w") as out:
        geojson.dump(annotations, out)


def gdf_to_file(
    gdf: gpd.GeoDataFrame,
    out_fn: Union[Path, str],
    format: str,
    qupath_format: Optional[str] = None,
) -> None:
    """Save a GeoDataFrame to a file.

    Parameters
    ----------
        out_fn : Union[Path, str]
            Filename for the output file.
        format : str
            The format of the output file. If None, the merged gdf is saved
            to the class attribute `self.annots` only. One of: "feather",
            "parquet", "geojson", "json" or None.
        qupath_format : str, optional, default="old"
            If this is not None, additional metadata is added to the geojson file to
            make it properly readable by QuPath. One of: "old", "latest", None
            NOTE: `old` works for versions less than 0.3.0. `latest` works for
            newer versions.

    Raises
    ------
        ValueError: If `format` is not one of "feather", "geojson", "parquet".
    """
    allowed = ("feather", "parquet", "geojson", None)
    if format not in allowed:
        raise ValueError(f"param `format` must be one of {allowed}. Got {format}.")

    allowed = ("old", "latest", None)
    if qupath_format not in allowed:
        raise ValueError(
            f"param `qupath_format` must be one of {allowed}. Got {qupath_format}."
        )

    out_fn = Path(out_fn)
    if format == "feather":
        gdf.to_feather(out_fn.with_suffix(".feather"))
    elif format == "parquet":
        gdf.to_parquet(out_fn.with_suffix(".parquet"))
    elif format == "geojson":
        if qupath_format == "old":
            # Extract the gson objs from the gdf
            gson_objs = gdf_apply(
                gdf, row_to_qupath, col=None, axis=1, parallel=True
            ).tolist()
            # save to a '.json' file
            save_geojson(gson_objs, out_fn)
        elif qupath_format == "latest":
            gdf["objectType"] = "annotation"
            gdf["classification"] = gdf_apply(
                gdf, add_qupath_classification, col=None, axis=1, parallel=True
            )
            gdf = gdf.drop("class_name", axis=1)
            gdf.to_file(out_fn.with_suffix(".geojson"), driver="GeoJSON", index=False)
        else:
            gdf.to_file(out_fn.with_suffix(".geojson"), driver="GeoJSON", index=True)


def check_format(fname: Union[Path, str]) -> None:
    """Check if the input file has the correct format.

    Parameters
    ----------
        fname : str
            The filename.

    Raises
    ------
        ValueError: If not all coordinates were found in filename.
        ValueError: If both x and y coordinates are not present in filename.
    """
    fn = Path(fname)
    if fn.suffix not in (".json", ".geojson", ".png"):
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

    NOTE: The filename needs to contain x & y-coordinates in
        "x-[coord1]_y-[coord2]"-format

    Parameters
    ----------
        fname : str
            The filename. Has to contain x & y-coordinates

    Raises
    ------
        ValueError: If not the delimeter of x and y- coord is not '_' or '-'.

    Returns
    -------
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

    NOTE: fname needs to contain x & y-coordinates in  "x-[coord1]_y-[coord2]"-format.

    Parameters
    ----------
        files : List[Union[Path, str]]
            A list of paths to the files.
        x : int
            x-coord in pixels.
        y : int
            y-coord in pixels.

    Returns
    -------
        str or None:
            returns the file name if it exists in the given input dir, else None.
    """
    for f in files:
        if get_xy_coords(f) == (x, y):
            return f

    return


def qupath_gsonobj(id: str = "PathCellDetection") -> Dict[str, Any]:
    """Initialize a empty QuPath readable geojson dict object for a cell annotation.

    Parameters
    ----------
        id : str, default="PathCellDetection"
            The id of the geojson object for QuPath. One of "PathCellDetection",
            "PathCellAnnotation", "PathDetectionObject".
    """
    assert id in ("PathCellDetection", "PathCellAnnotation", "PathDetectionObject")

    geo_obj = {}
    geo_obj.setdefault("type", "Feature")

    geo_obj.setdefault("id", id)
    geo_obj.setdefault("geometry", {"type": "Polygon", "coordinates": None})
    geo_obj.setdefault(
        "properties",
        {"isLocked": "false", "measurements": [], "classification": {"name": None}},
    )

    return geo_obj


def add_qupath_props(row: gpd.GeoSeries) -> gpd.GeoSeries:
    """Add QuPath properties to a row of a gdf."""
    c = row["class_name"]
    props = {"isLocked": "false", "measurements": [], "classification": {"name": c}}
    return props


def add_qupath_classification(row: gpd.GeoSeries) -> gpd.GeoSeries:
    """Add QuPath properties to a row of a gdf."""
    c = row["class_name"]
    props = {"name": c, "color": None}
    return props


def row_to_qupath(row: gpd.GeoSeries) -> Dict[str, Any]:
    """Convert a row of a gdf to a QuPath<0.3 readable geojson obj.

    NOTE: This format is readable only in older version of QuPath (<0.3).

    Paramaters
    ----------
        row : gpd.GeoSeries
            A row of a gdf.

    Returns
    -------
        Dict[str, Any]: A QuPath readable geojson dictionary.

    Examples
    --------
    >>> from cellseg.utils import read_gdf
    >>> from cellseg_gsontools.merging.save_utils import row_to_qupath
    >>> gdf = gpd.read_file("path/to/file.json")
    >>> gdf.apply(row_to_qupath, axis=1).tolist()
    [{'type': 'Feature',
    'id': 'PathCellDetection',
    'geometry': {'type': 'Polygon',
    'coordinates': [[47001.01188288047, 113010.00360463238],
        [47006.00676531105, 113005.0101250035],
        [47003.006765311016, 113007.01012500352],
        [47001.01188288047, 113010.00360463238]]},
    'properties': {'isLocked': 'false',
    'measurements': [],
    'classification': {'name': 'glandular_epithel'}}},
    {'type': 'Feature',
    'id': 'PathCellDetection',
    'geometry': {'type': 'Polygon',
    'coordinates': [[48841.01332475732, 112177.0],
        [47587.00676531103, 113024.98987499649],
        [47589.00360463241, 113025.98811711954],
        [47595.99510422325, 113025.9881805494],
        [48841.01332475732, 112177.0]]},
    'properties': {'isLocked': 'false',
    'measurements': [],
    'classification': {'name': 'connective'}}},
    ...]
    """
    geom = row["geometry"]
    class_name = row["class_name"]

    # get QuPath readable geojson obj
    gson_obj = qupath_gsonobj(id="PathCellAnnotation")
    gson_obj["geometry"]["coordinates"] = [
        list(list(coord) for coord in geom.exterior.coords[:])
    ]
    # gson_obj["geometry"]["coordinates"] = list(
    #     list(coord) for coord in geom.exterior.coords[:]
    # )

    gson_obj["properties"]["classification"]["name"] = class_name

    return gson_obj
