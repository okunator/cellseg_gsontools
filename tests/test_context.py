import pytest

from cellseg_gsontools.spatial_context import (
    InterfaceContext,
    PointClusterContext,
    WithinContext,
)


@pytest.mark.parametrize("clust_method", ["dbscan", "adbscan", "optics"])
def test_cluster_context(cells_and_areas, clust_method):
    cells = cells_and_areas[0]

    cluster_context = PointClusterContext(
        cell_gdf=cells,
        label="inflammatory",
        cluster_method=clust_method,
        silence_warnings=True,
        min_area_size="mean",
    )
    cluster_context.fit()

    assert len(cluster_context.context[1]["roi_area"]) == 1


def test_interface_context(cells_and_areas):
    cells = cells_and_areas[0]
    areas = cells_and_areas[1]

    interface_context = InterfaceContext(
        area_gdf=areas,
        cell_gdf=cells,
        label1="areagland",
        label2="areastroma",
        silence_warnings=True,
        min_area_size="median",
    )
    interface_context.fit()

    interface_context.context2gdf("interface_area")
    assert len(interface_context.context[1]["roi_area"]) == 1


def test_within_context(cells_and_areas):
    cells = cells_and_areas[0]
    areas = cells_and_areas[1]

    within_context = WithinContext(
        cell_gdf=cells,
        area_gdf=areas,
        label="area_cin",
        silence_warnings=True,
        min_area_size="quantile",
        q=45.0,
    )
    within_context.fit()

    assert len(within_context.context[1]["roi_area"]) == 1
