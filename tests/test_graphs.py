import pytest

from cellseg_gsontools.graphs import fit_graph


@pytest.mark.parametrize(
    "type",
    ["delaunay", "gabriel", "knn", "distband", "voronoi", "relative_nhood", "kernel"],
)
@pytest.mark.parametrize("thresh", [150.0, None])
def test_fit_graph(cell_gson, type, thresh):
    if type == "distband" and thresh is None:
        thresh = 20
    w = fit_graph(cell_gson, type=type, thresh=thresh)

    assert 0 not in list(w.neighbors.keys())
