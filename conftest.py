from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
import shapely


@pytest.fixture(scope="package")
def cell_gson() -> Path:
    """Return a path to directory with a few test images."""
    path = Path().resolve()
    path = path / "tests/data/test_cells.json"

    df = pd.read_json(path)
    df["geometry"] = df["geometry"].apply(shapely.geometry.shape)
    df = gpd.GeoDataFrame(df).set_geometry("geometry")

    return df
