from pathlib import Path
from typing import Tuple

import geopandas as gpd
import pandas as pd
import pytest
import shapely


@pytest.fixture(scope="package")
def cell_gson() -> gpd.GeoDataFrame:
    """Return a path to directory with a few test images."""
    path = Path().resolve()
    path = path / "tests/data/test_cells.json"

    df = pd.read_json(path)
    df["geometry"] = df["geometry"].apply(shapely.geometry.shape)
    df = gpd.GeoDataFrame(df).set_geometry("geometry")
    df["class_name"] = df["properties"].apply(lambda x: x["classification"]["name"])

    return df


@pytest.fixture(scope="package")
def cells_and_areas() -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    path = Path().resolve()
    path_cells = path / "tests/data/test_cells.feather"
    path_areas = path / "tests/data/test_area.feather"

    cells = gpd.read_feather(path_cells)
    areas = gpd.read_feather(path_areas)

    return cells, areas


@pytest.fixture(scope="package")
def merge_data_cell() -> Path:
    path = Path().resolve()
    path = path / "tests/data/merge_data/cell"

    return path


@pytest.fixture(scope="package")
def merge_data_area() -> Path:
    path = Path().resolve()
    path = path / "tests/data/merge_data/area"

    return path
