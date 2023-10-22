from functools import partial

import numpy as np
import pytest
from libpysal.weights import Delaunay, DistanceBand

from cellseg_gsontools.apply import gdf_apply
from cellseg_gsontools.neighbors import neighborhood, nhood_type_count, nhood_vals
from cellseg_gsontools.utils import set_uid


@pytest.mark.parametrize("ret_n", [True, False])
def test_neighbors(cell_gson, ret_n):
    gdf = cell_gson
    data = set_uid(gdf)

    # Get spatial weights
    w_dist = DistanceBand.from_dataframe(data, ids="uid", threshold=55.0, alpha=-1.0)
    func = partial(neighborhood, spatial_weights=w_dist, ret_n_neighbors=ret_n)
    data["nhood"] = gdf_apply(data, func, columns=["uid"])

    datum = data["nhood"].loc[1]
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
    func = partial(neighborhood, spatial_weights=w_dist)
    data["nhood"] = gdf_apply(data, func, columns=["uid"])

    # Define the class name column
    val_col = "class_name"
    values = data.set_index("uid")[val_col]

    # get the neighborhood classes
    func = partial(nhood_vals, values=values)
    data[f"{val_col}_nhood_vals"] = gdf_apply(
        data,
        func,
        columns=["nhood"],
    )

    func = partial(nhood_type_count, cls="inflammatory", frac=frac)
    data["local_infiltration_fraction"] = gdf_apply(
        data,
        func,
        columns=[f"{val_col}_nhood_vals"],
    )
