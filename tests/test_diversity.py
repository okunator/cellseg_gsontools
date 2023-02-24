import numpy as np
import pytest
from libpysal.weights import DistanceBand

from cellseg_gsontools.apply import gdf_apply
from cellseg_gsontools.diversity import local_diversity
from cellseg_gsontools.geometry import eccentricity
from cellseg_gsontools.neighbors import neighborhood
from cellseg_gsontools.utils import set_uid


@pytest.mark.parametrize("ret_n", [True, False])
def test_neighbors(cell_gson, ret_n):
    gdf = cell_gson
    data = set_uid(gdf)

    # Get spatial weights
    w_dist = DistanceBand.from_dataframe(data, ids="uid", threshold=55.0, alpha=-1.0)
    data["neighbors"] = gdf_apply(
        data, neighborhood, col="uid", spatial_weights=w_dist, ret_n_neighbors=ret_n
    )

    datum = data["neighbors"].loc[1]
    if isinstance(datum, (int, float, np.int64)):
        assert datum >= 0.0
    elif isinstance(datum, (list, np.ndarray)):
        assert len(datum) >= 0
    else:
        assert datum == 0


@pytest.mark.parametrize(
    "index", ["shannon_index", "simpson_index", "gini_index", "theil_index"]
)
@pytest.mark.parametrize("parallel", [True, False])
def test_diversity(cell_gson, index, parallel):
    gdf = cell_gson
    gdf["eccentricity"] = gdf_apply(gdf, eccentricity, parallel=True)

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
