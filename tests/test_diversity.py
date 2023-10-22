import pandas as pd
import pytest
from libpysal.weights import DistanceBand

from cellseg_gsontools.apply import gdf_apply
from cellseg_gsontools.diversity import local_diversity
from cellseg_gsontools.geometry import eccentricity
from cellseg_gsontools.utils import set_uid


@pytest.mark.parametrize(
    "index",
    [
        "shannon_index",
        "simpson_index",
        "gini_index",
        "theil_index",
    ],
)
@pytest.mark.parametrize("parallel", [True, False])
def test_diversity(cell_gson, index, parallel):
    gdf = cell_gson
    gdf["eccentricity"] = gdf_apply(
        gdf, eccentricity, columns=["geometry"], parallel=True
    )

    data = set_uid(gdf)
    w_dist = DistanceBand.from_dataframe(data, ids="uid", threshold=55.0, alpha=-1.0)

    data = local_diversity(
        data,
        w_dist,
        metrics=[index],
        val_col="eccentricity",
        parallel=parallel,
        rm_nhood_cols=True,
    )
