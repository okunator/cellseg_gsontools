from cellseg_gsontools.links import link_counts
from cellseg_gsontools.spatial_context import InterfaceContext


def test_link_counts(cells_and_areas):
    cells = cells_and_areas[0]
    areas = cells_and_areas[1]
    iface_context = InterfaceContext(
        area_gdf=areas,
        cell_gdf=cells,
        top_labels="area_cin",
        bottom_labels="areastroma",
        silence_warnings=True,
        min_area_size=100000.0,
    )
    iface_context.fit()

    classes = [
        "inflammatory",
        "connective",
        "glandular_epithel",
        "squamous_epithel",
        "neoplastic",
    ]

    wout = iface_context.merge_weights("border_network")
    link_counts(cells, wout, classes)
