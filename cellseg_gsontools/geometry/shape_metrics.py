from typing import Tuple

import geopandas as gpd
import numpy as np
import shapely
from shapely.geometry import Polygon

from ..apply import gdf_apply
from .axis import axis_angle, axis_len
from .circle import inscribing_circle

__all__ = [
    "major_axis_len",
    "minor_axis_len",
    "major_axis_angle",
    "minor_axis_angle",
    "compactness",
    "circularity",
    "convexity",
    "solidity",
    "elongation",
    "eccentricity",
    "fractal_dimension",
    "sphericity",
    "shape_index",
    "rectangularity",
    "squareness",
    "equivalent_rectangular_index",
    "shape_metric",
    "SHAPE_LOOKUP",
]


def major_axis_len(polygon: Polygon) -> float:
    """Compute the major axis length of a polygon.

    Parameters:
        polygon (Polygon):
            Input shapely polygon object.

    Returns:
        float:
            The length of the major axis.
    """
    mrr = polygon.minimum_rotated_rectangle.exterior.coords
    return axis_len(mrr, "major")


def minor_axis_len(polygon: Polygon) -> float:
    """Compute the minor axis length of a polygon.

    Parameters:
        polygon (Polygon):
            Input shapely polygon object.

    Returns:
        float:
            The length of the minor axis.
    """
    mrr = polygon.minimum_rotated_rectangle.exterior.coords
    return axis_len(mrr, "minor")


def major_axis_angle(polygon: Polygon) -> float:
    """Compute the major axis angle of a polygon.

    Parameters:
        polygon (Polygon):
            Input shapely polygon object.

    Returns:
        float:
            The angle of the major axis in degrees.
    """
    mrr = polygon.minimum_rotated_rectangle.exterior.coords
    return axis_angle(mrr, "major")


def minor_axis_angle(polygon: Polygon) -> float:
    """Compute the minor axis angle of a polygon.

    Parameters:
        polygon (Polygon):
            Input shapely polygon object.

    Returns:
        float:
            The angle of the minor axis in **degrees**.
    """
    mrr = polygon.minimum_rotated_rectangle.exterior.coords
    return axis_angle(mrr, "minor")


def compactness(polygon: Polygon, **kwargs) -> float:
    """Compute the compactness of a polygon.

    **Compactness:**
    $$
    \\frac{4\\pi A_{poly}}{P_{poly}^2}
    $$

    where $A_{poly}$ is the area of the polygon and $P_{poly}$ is the perimeter of
    the polygon.

    Parameters:
        polygon (Polygon):
            Input shapely polygon object.

    Returns:
        float:
            The compactness value of a polygon between 0-1.
    """
    perimeter = polygon.length
    area = polygon.area

    compactness = (4 * np.pi * area) / perimeter**2

    return compactness


def circularity(polygon: Polygon, **kwargs) -> float:
    """Compute the circularity of a polygon.

    **Circularity:**
    $$
    \\frac{4 \\times \\pi A_{poly}}{P_{convex}^2}
    $$

    where $A_{poly}$ is the area of the polygon and $P_{convex}$ is the perimeter of
    the convex hull.

    Parameters:
        polygon (Polygon):
            Input shapely polygon object.

    Returns:
        float:
            The circularity value of a polygon between 0-1.
    """
    convex_perimeter = polygon.convex_hull.length
    area = polygon.area

    circularity = (4 * np.pi * area) / convex_perimeter**2

    return circularity


def convexity(polygon: Polygon, **kwargs) -> float:
    """Compute the convexity of a polygon.

    **Convexity:**
    $$
    \\frac{P_{convex}}{P_{poly}}
    $$

    where $P_{convex}$ is the perimeter of the convex hull and $P_{poly}$ is the
    perimeter of the polygon.

    Note:
        Object is convex if convexity = 1.

    Parameters:
        polygon (Polygon):
            Input shapely polygon object.

    Returns:
        float:
            The convexity value of a polygon between 0-1.
    """
    convex_perimeter = polygon.convex_hull.length
    perimeter = polygon.length

    convexity = convex_perimeter / perimeter

    return convexity


def solidity(polygon: Polygon, **kwargs) -> float:
    """Compute the solidity of a polygon.

    **Solidity:**
    $$
    \\frac{A_{poly}}{A_{convex}}
    $$

    where $A_{poly}$ is the area of the polygon and $A_{convex}$ is the area of the
    convex hull.

    Note:
        Object is solid if solidity = 1.

    Parameters:
        polygon (Polygon):
            Input shapely polygon object.

    Returns:
        float:
            The solidity value of a polygon between 0-1.
    """
    convex_area = polygon.convex_hull.area
    area = polygon.area

    return area / convex_area


def elongation(polygon: Polygon, **kwargs) -> float:
    """Compute the elongation of a polygon.

    **Elongation:**
    $$
    \\frac{\\text{bbox width}}{\\text{bbox height}}
    $$

    Parameters:
        polygon (Polygon):
            Input shapely polygon object.

    Returns:
        float:
            The elongation value of a polygon between 0-1.
    """
    minx, miny, maxx, maxy = polygon.bounds

    width = maxx - minx
    height = maxy - miny

    if width <= height:
        elongation = width / height
    else:
        elongation = height / width

    return elongation


def eccentricity(polygon: Polygon, **kwargs) -> float:
    """Compute the eccentricity of a polygon.

    **Eccentricity:**
    $$
    \\sqrt{1 - \\frac{\\text{minor axis}^2}{\\text{major axis}^2}}
    $$

    Parameters:
        polygon (Polygon):
            Input shapely polygon object.

    Returns:
        float:
            The eccentricity value of a polygon between 0-1.
    """
    mrr = polygon.minimum_rotated_rectangle.exterior.coords
    major_ax, minor_ax = axis_len(mrr)
    eccentricity = np.sqrt(1 - (minor_ax**2 / major_ax**2))
    return eccentricity


def fractal_dimension(polygon: Polygon, **kwargs) -> float:
    """Compute the fractal dimension of a polygon.

    **Fractal dimension:**
    $$
    2 \\times \\frac{\\log(\\frac{P_{poly}}{4})}{\\log(A_{poly})}
    $$

    where $P_{poly}$ is the perimeter of the polygon and $A_{poly}$ is the area of the
    polygon.

    'Fractal dimension is a measure of how "complicated" a self-similar
    figure is. In a rough sense, it measures "how many points" lie in a given set.'
    - first google search results

    Parameters:
        polygon (Polygon):
            Input shapely polygon object.

    Returns:
        float:
            The fractal dimension value of a polygon.
    """
    perimeter = polygon.length
    area = polygon.area

    return (2 * np.log(perimeter / 4)) / np.log(area)


def sphericity(polygon: Polygon, **kwargs) -> float:
    """Compute the sphericity of a polygon.

    **Sphericity:**
    $$
    \\frac{\\text{MIR}}{\\text{MBR}}
    $$

    where $\\text{MIR}$ is the radius of the minimum inscribing circle radius and
    $\\text{MBR}$ is the radius of the minimum bounding radius.

    Parameters:
        polygon (Polygon):
            Input shapely polygon object.

    Returns:
        float:
            The sphericity value of a polygon.
    """
    _, _, ri = inscribing_circle(polygon)
    rc = shapely.minimum_bounding_radius(polygon)

    return ri / rc


def shape_index(polygon: Polygon, **kwargs) -> float:
    """Compute the shape index of a polygon.

    **Shape Index:**
    $$
    \\frac{\\sqrt{\\frac{A_{poly}}{\\pi}}}{\\text{MBR}}
    $$

    where $A_{poly}$ is the area of the polygon and $\\text{MBR}$ is the radius of the
    minimum bounding radius.

    Note:
        $\\text{MBR}$ = radius of the minimum circumscribing circle.

    Parameters:
        polygon (Polygon):
            Input shapely polygon object.

    Returns:
        float:
            The shape index value of a polygon.
    """
    r = shapely.minimum_bounding_radius(polygon)
    area = polygon.area

    return np.sqrt(area / np.pi) / r


def squareness(polygon: Polygon, **kwargs) -> float:
    """Compute the squareness of a polygon.

    **Squareness:**
    $$
    \\left(\\frac{4*\\sqrt{A_{poly}}}{P_{poly}}\\right)^2
    $$

    where $A_{poly}$ is the area of the polygon and $P_{poly}$ is the perimeter of
    the polygon.

    Note:
        For irregular shapes, squareness is close to zero and for circular shapes close
        to 1.3. For squares, equals 1

    Parameters:
        polygon (Polygon):
            Input shapely polygon object.

    Returns:
        float:
            The squareness value of a polygon.
    """
    area = polygon.area
    perimeter = polygon.length

    return ((np.sqrt(area) * 4) / perimeter) ** 2


def rectangularity(polygon: Polygon, **kwargs) -> float:
    """Compute the rectangularity of a polygon.

    **Rectangularity:**
    $$
    \\frac{A_{poly}}{A_{MRR}}
    $$

    where $A_{poly}$ is the area of the polygon and $A_{MRR}$ is the area of the
    minimum rotated rectangle.

    Parameters:
        polygon (Polygon):
            Input shapely polygon object.

    Returns:
        float:
            The rectangularity value of a polygon between 0-1.
    """
    mrr = polygon.minimum_rotated_rectangle

    return polygon.area / mrr.area


def equivalent_rectangular_index(polygon: Polygon) -> float:
    """Compute the equivalent rectangular index.

    I.e. the deviation of a polygon from an equivalent rectangle

    **ERI:**
    $$
    \\frac{\\sqrt{A_{poly}}}{A_{MRR}}
    \\times
    \\frac{P_{MRR}}{P_{poly}}
    $$

    where $A_{poly}$ is the area of the polygon, $A_{MRR}$ is the area of the
    minimum rotated rectangle, $P_{MRR}$ is the perimeter of the minimum rotated
    rectangle and $P_{poly}$ is the perimeter of the polygon.

    Parameters:
        polygon (Polygon):
            Input shapely polygon object.

    Returns:
        float:
            The ERI value of a polygon between 0-1.
    """
    mrr = polygon.minimum_rotated_rectangle

    return np.sqrt(polygon.area / mrr.area) / (mrr.length / polygon.length)


SHAPE_LOOKUP = {
    "major_axis_len": major_axis_len,
    "minor_axis_len": minor_axis_len,
    "major_axis_angle": major_axis_angle,
    "minor_axis_angle": minor_axis_angle,
    "compactness": compactness,
    "circularity": circularity,
    "convexity": convexity,
    "solidity": solidity,
    "elongation": elongation,
    "eccentricity": eccentricity,
    "fractal_dimension": eccentricity,
    "sphericity": sphericity,
    "shape_index": shape_index,
    "rectangularity": rectangularity,
    "squareness": squareness,
    "equivalent_rectangular_index": equivalent_rectangular_index,
    "area": None,
}


def shape_metric(
    gdf: gpd.GeoDataFrame,
    metrics: Tuple[str, ...],
    parallel: bool = True,
    col_prefix: str = None,
    create_copy: bool = True,
) -> gpd.GeoDataFrame:
    """Compute a set of shape metrics for every row of the gdf.

    Parameters:
        gdf (gpd.GeoDataFrame):
            The input GeoDataFrame.
        metrics (Tuple[str, ...]):
            A Tuple/List of shape metrics.
        parallel (bool):
            Flag whether to use parallel apply operations when computing the diversities.
        col_prefix (str):
            Prefix for the new column names.
        create_copy (bool):
            Flag whether to create a copy of the input gdf or not.

    Note:
        Allowed shape metrics are:

        - `area`
        - `major_axis_len`
        - `minor_axis_len`
        - `major_axis_angle`
        - `minor_axis_angle`
        - `compactness`
        - `circularity`
        - `convexity`
        - `solidity`
        - `elongation`
        - `eccentricity`
        - `fractal_dimension`
        - `sphericity`
        - `shape_index`
        - `rectangularity`
        - `squareness`
        - `equivalent_rectangular_index`

    Raises:
        ValueError:
            If an illegal metric is given.

    Returns:
        gpd.GeoDataFrame:
            The input geodataframe with computed shape metric columns added.

    Examples:
        Compute the eccentricity and solidity for each polygon in gdf.
        >>> from cellseg_gsontools.geometry import shape_metric
        >>> shape_metric(gdf, metrics=["eccentricity", "solidity"], parallel=True)
    """
    if not isinstance(metrics, (list, tuple)):
        raise ValueError(f"`metrics` must be a list or tuple. Got: {type(metrics)}.")

    allowed = list(SHAPE_LOOKUP.keys())
    if not all(m in allowed for m in metrics):
        raise ValueError(
            f"Illegal metric in `metrics`. Got: {metrics}. Allowed metrics: {allowed}."
        )

    if create_copy:
        gdf = gdf.copy()

    if col_prefix is None:
        col_prefix = ""
    else:
        col_prefix += "_"

    met = list(metrics)
    if "area" in metrics:
        gdf[f"{col_prefix}area"] = gdf.area
        met.remove("area")

    for metric in met:
        gdf[metric] = gdf_apply(
            gdf,
            SHAPE_LOOKUP[metric],
            columns=["geometry"],
            parallel=parallel,
        )

    return gdf
