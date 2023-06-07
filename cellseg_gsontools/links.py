from itertools import combinations_with_replacement
from typing import Dict, List, Tuple

import geopandas as gpd
from libpysal.weights import W

__all__ = ["link_counts", "get_link_combinations"]


def get_link_combinations(classes: Tuple[str, ...]) -> List[str]:
    """Return a list of link combinations between the classes in `classes`."""
    combos = ["-".join(t) for t in list(combinations_with_replacement(classes, 2))]

    return combos


def link_counts(
    gdf: gpd.GeoDataFrame, w: W, classes: Tuple[str, ...]
) -> Dict[str, int]:
    """Get the link-type counts of a geodataframe given a spatial weight object `w`.

    Parameters
    ----------
        gdf : gpd.GeoDataFrame
            Input geodataframe. Has to have a `class_name` column.
        w : libysal.weights.W
            Libpysal spatial weights object of the gdf.
        classes : Tuple[str, ...]
            A list/tuple containing the classes of your dataset.

    Returns
    -------
        Dict[str, int]: A contigency dictionary.

    Example
    -------
    Get the link types of the tumor-stroma interfaces in a slide.
    >>> from cellseg_gsontools.spatial_context import InterfaceContext
    >>> from cellseg_gsontools.links import link_counts

    >>> iface_context = InterfaceContext(
            area_gdf=areas,
            cell_gdf=cells,
            label1="area_cin",
            label2="areastroma",
            silence_warnings=True,
            verbose=True,
            min_area_size=100000.0
        )
    >>> classes = [
            "inflammatory",
            "connective",
            "glandular_epithel",
            "squamous_epithel",
            "neoplastic"
        ]

    >>> w = iface_context.merge_weights("border_network")
    >>> link_counts(cells, w, t)
    Processing interface area: 4: 100%|██████████| 4/4 [00:01<00:00,  2.58it/s]
    {'inflammatory-inflammatory': 31,
    'inflammatory-connective': 89,
    'inflammatory-glandular_epithel': 0,
    'inflammatory-squamous_epithel': 0,
    'inflammatory-neoplastic': 86,
    'connective-connective': 131,
    'connective-glandular_epithel': 0,
    'connective-squamous_epithel': 0,
    'connective-neoplastic': 284,
    'glandular_epithel-glandular_epithel': 0,
    'glandular_epithel-squamous_epithel': 0,
    'glandular_epithel-neoplastic': 0,
    'squamous_epithel-squamous_epithel': 0,
    'squamous_epithel-neoplastic': 0,
    'neoplastic-neoplastic': 236}
    """
    combos = get_link_combinations(classes)
    link_cnt = {combo: 0 for combo in combos}

    all_node_pairs = []
    for node, neighbors in w.neighbors.items():
        ncls = gdf.loc[node, "class_name"]
        if neighbors:
            neighbors_cls = gdf.loc[neighbors, "class_name"]
            node_pairs = [set((node, n)) for n in neighbors]

            for ngh_ix, ngh_cls in zip(neighbors, neighbors_cls):
                node_ids = set((node, ngh_ix))
                if node_ids not in all_node_pairs:
                    for combo in combos:
                        types = set(combo.split("-"))
                        if ncls in types and ngh_cls in types and ngh_cls != ncls:
                            link_cnt[combo] += 1
                        elif ncls in types and ngh_cls in types and ngh_cls == ncls:
                            if len(types) == 1:
                                link_cnt[combo] += 1

            all_node_pairs.extend(node_pairs)

    return link_cnt
