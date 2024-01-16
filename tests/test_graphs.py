import pytest

from cellseg_gsontools.graphs import fit_graph
from cellseg_gsontools.utils import set_uid


@pytest.mark.parametrize(
    "type",
    ["delaunay", "knn", "distband", "relative_nhood"],
)
@pytest.mark.parametrize("thresh", [150.0, None])
def test_fit_graph(cell_gson, type, thresh):
    if type == "distband" and thresh is None:
        thresh = 20
    w = fit_graph(
        set_uid(cell_gson, start_ix=0), type=type, thresh=thresh, id_col="uid"
    )

    assert w.n == len(cell_gson)
