import numpy as np
import pytest
from libpysal.weights import Delaunay, DistanceBand

from cellseg_gsontools.apply import gdf_apply
from cellseg_gsontools.graphs import dist_thresh_weights
from cellseg_gsontools.neighbors import neighborhood, nhood_type_count, nhood_vals
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


@pytest.mark.parametrize("frac", [True, False])
def test_nhood_type_count(cell_gson, frac):
    data = set_uid(cell_gson)

    # Get spatial weights
    w_dist = DistanceBand.from_dataframe(data, threshold=55.0, alpha=-1.0)

    # Get the neihgboring nodes of the graph
    data["nhood"] = gdf_apply(data, neighborhood, col="uid", spatial_weights=w_dist)

    # Define the class name column
    val_col = "class_name"
    values = data.set_index("uid")[val_col]

    # get the neighborhood classes
    data[f"{val_col}_nhood_vals"] = gdf_apply(
        data,
        nhood_vals,
        col="nhood",
        values=values,
    )

    data["local_infiltration_fraction"] = gdf_apply(
        data,
        nhood_type_count,
        col=f"{val_col}_nhood_vals",
        cls="inflammatory",
        frac=frac,
    )


def test_dist_thresh(cell_gson):
    gdf = cell_gson

    id_col = "iid"
    gdf[id_col] = range(len(gdf))
    gdf = gdf.set_index(id_col, drop=False)
    ids = list(gdf.index.values)
    w = Delaunay.from_dataframe(gdf.centroid, id_order=ids, ids=ids)

    # drop the edges
    w = dist_thresh_weights(gdf, w, thresh=120.0)
