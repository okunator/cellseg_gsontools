import pytest

from cellseg_gsontools.spatial_context import (
    InterfaceContext,
    PointClusterContext,
    WithinContext,
)


@pytest.mark.parametrize("clust_method", ["dbscan", "adbscan", "optics", "hdbscan"])
@pytest.mark.parametrize("labels", ["inflammatory", ["inflammatory", "connective"]])
@pytest.mark.parametrize("backend", ["geopandas", "dask-geopandas", "spatialpandas"])
def test_cluster_context(cells_and_areas, clust_method, labels, backend):
    cells = cells_and_areas[0]

    cluster_context = PointClusterContext(
        cell_gdf=cells,
        labels=labels,
        cluster_method=clust_method,
        silence_warnings=True,
        min_area_size=50,
        graph_type="distband",
        backend=backend,
    )
    cluster_context.fit()

    assert len(cluster_context.context[0]["roi_area"]) == 1


@pytest.mark.parametrize("toplabels", ["areagland", ["areagland", "area_cin"]])
@pytest.mark.parametrize("bottomlabels", ["areastroma", ["areastroma"]])
@pytest.mark.parametrize("backend", ["geopandas", "dask-geopandas", "spatialpandas"])
def test_interface_context(cells_and_areas, toplabels, bottomlabels, backend):
    cells = cells_and_areas[0]
    areas = cells_and_areas[1]

    interface_context = InterfaceContext(
        area_gdf=areas,
        cell_gdf=cells,
        top_labels=toplabels,
        bottom_labels=bottomlabels,
        silence_warnings=True,
        min_area_size=50,
        graph_type="distband",
        parallel=False,
        backend=backend,
    )
    interface_context.fit()

    interface_context.context2gdf("interface_area")
    assert len(interface_context.context[1]["roi_area"]) == 1


@pytest.mark.parametrize("labels", ["area_cin", ["area_cin", "areagland"]])
@pytest.mark.parametrize("backend", ["geopandas", "dask-geopandas", "spatialpandas"])
@pytest.mark.parametrize("parallel", [True, False])
def test_within_context(cells_and_areas, labels, backend, parallel):
    cells = cells_and_areas[0]
    areas = cells_and_areas[1]

    within_context = WithinContext(
        cell_gdf=cells,
        area_gdf=areas,
        labels=labels,
        silence_warnings=True,
        min_area_size=50,
        graph_type="distband",
        parallel=parallel,
        num_processes=1,
        backend=backend,
    )
    within_context.fit()

    assert len(within_context.context[1]["roi_area"]) == 1
