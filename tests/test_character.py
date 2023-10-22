import pytest
from libpysal.weights import DistanceBand

from cellseg_gsontools.apply import gdf_apply
from cellseg_gsontools.character import local_character, local_distances
from cellseg_gsontools.geometry import eccentricity
from cellseg_gsontools.utils import set_uid


@pytest.mark.parametrize("reduction", ["sum", "mean", "median"])
@pytest.mark.parametrize("weight", [True, False])
@pytest.mark.parametrize("parallel", [True, False])
def test_character(cell_gson, reduction, weight, parallel):
    gdf = cell_gson
    gdf["eccentricity"] = gdf_apply(
        gdf, eccentricity, columns=["geometry"], parallel=True
    )

    data = set_uid(gdf)
    w_dist = DistanceBand.from_dataframe(data, ids="uid", threshold=55.0, alpha=-1.0)

    data = local_character(
        data,
        w_dist,
        reductions=[reduction],
        val_col="eccentricity",
        weight_by_area=weight,
        parallel=parallel,
        rm_nhood_cols=True,
    )


@pytest.mark.parametrize("reduction", ["sum", "mean", "median"])
@pytest.mark.parametrize("weight", [True, False])
@pytest.mark.parametrize("parallel", [True, False])
def test_dists(cell_gson, reduction, weight, parallel):
    gdf = cell_gson

    data = set_uid(gdf)
    w_dist = DistanceBand.from_dataframe(data, ids="uid", threshold=55.0, alpha=-1.0)

    data = local_distances(
        data,
        w_dist,
        reductions=[reduction],
        weight_by_area=weight,
        parallel=parallel,
        rm_nhood_cols=True,
    )
