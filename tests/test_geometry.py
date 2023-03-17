import pytest

from cellseg_gsontools.apply import gdf_apply
from cellseg_gsontools.geometry import (
    alpha_shape,
    circularity,
    compactness,
    convexity,
    eccentricity,
    elongation,
    equivalent_rectangular_index,
    fractal_dimension,
    major_axis_angle,
    major_axis_len,
    minor_axis_angle,
    minor_axis_len,
    rectangularity,
    shape_index,
    shape_metric,
    solidity,
    sphericity,
    squareness,
)


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize(
    "func",
    [
        major_axis_len,
        minor_axis_len,
        major_axis_angle,
        minor_axis_angle,
        compactness,
        circularity,
        convexity,
        solidity,
        elongation,
        eccentricity,
        fractal_dimension,
        sphericity,
        shape_index,
        rectangularity,
        squareness,
        equivalent_rectangular_index,
    ],
)
def test_shape_metrics(cell_gson, func, parallel):
    gdf = cell_gson
    gdf["m"] = gdf_apply(gdf, func, parallel=parallel)

    assert gdf["m"].mean() >= 0


def test_alpha_shape(cell_gson):
    ash = alpha_shape(cell_gson.geometry.loc[0])

    assert len(list(ash.exterior.coords)) > 3


def test_shape_met(cell_gson):
    shape_metric(cell_gson, ["solidity", "squareness"])
