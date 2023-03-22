import pytest

from cellseg_gsontools.spatial_context import PointClusterContext, WithinContext
from cellseg_gsontools.summary import DistanceSummary, InstanceSummary, SemanticSummary


@pytest.mark.parametrize("groups", [["label", "class_name"], None])
def test_instance_summary(cells_and_areas, groups):
    cells = cells_and_areas[0]

    cluster_context = PointClusterContext(
        cell_gdf=cells,
        label="inflammatory",
        cluster_method="optics",
        min_area_size=50000.0,
        min_samples=10,
    )

    metrics = ["area"]
    if groups is not None:
        metrics += ["theil_within_group"]

    immune_cluster_cells = cluster_context.context2gdf("roi_cells")
    immune_clust_summary = InstanceSummary(
        immune_cluster_cells[immune_cluster_cells["class_name"] == "inflammatory"],
        metrics=metrics,
        groups=groups,
        prefix="icc-",
    )

    immune_clust_summary.summarize()

    # assert "icc-theil_within_group-label-area" in ic_cells.index


def test_semantic_summary(cells_and_areas):
    cells = cells_and_areas[0]
    areas = cells_and_areas[1]
    within_context = WithinContext(area_gdf=areas, cell_gdf=cells, label="area_cin")

    cin_area = within_context.context2gdf("roi_area")
    cin_area_summary = SemanticSummary(
        cin_area,
        metrics=["area", "theil_within_group"],
        groups=["label"],
        prefix="sem-",
    )

    summ = cin_area_summary.summarize(return_counts=False, filter_pattern="mean")

    assert "sem-theil_within_group-label-area" in summ.index


def test_distance_summary(cells_and_areas):
    cells = cells_and_areas[0]
    areas = cells_and_areas[1]

    within_context = WithinContext(
        area_gdf=areas, cell_gdf=cells, label="area_cin", min_area_size=100000.0
    )
    lesion_areas = within_context.context2gdf("roi_area")

    cluster_context = PointClusterContext(
        cell_gdf=cells,
        label="inflammatory",
        cluster_method="optics",
        min_area_size=50000.0,
        min_samples=10,
    )

    immune_cluster_areas = cluster_context.context2gdf("roi_area")
    immune_proximities = DistanceSummary(
        immune_cluster_areas,
        lesion_areas,
        groups=None,
        prefix="icc-close2lesion-",
    )
    ic_dists = immune_proximities.summarize()

    assert "icc-close2lesion-1-count" in ic_dists.index
