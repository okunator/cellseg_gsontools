from functools import partial
from itertools import combinations_with_replacement
from typing import Dict, List, Tuple

import geopandas as gpd
from libpysal.weights import W
from shapely.geometry import LineString, Point

from .apply import gdf_apply

__all__ = ["weights2gdf", "link_counts", "get_link_combinations"]


def get_link_combinations(classes: Tuple[str, ...]) -> List[str]:
    """Return a list of link combinations between the classes in `classes`.

    Parameters:
        classes (Tuple[str, ...]):
            A list/tuple containing the classes of your dataset.
    """
    combos = ["-".join(t) for t in list(combinations_with_replacement(classes, 2))]

    return combos


def _create_link(focal: Point, neighbor: Point) -> LineString:
    """Create a LineString between two centroids.

    Parameters:
        focal (Point):
            Focal centroid.
        neighbor (Point):
            Neighbor centroid.

    Returns:
        LineString:
            LineString between the two centroids.
    """
    return LineString([focal, neighbor])


def _get_link_class(
    focal_class: str, neighbor_class: str, link_combos: List[str]
) -> str:
    """Get link class based on focal and neighbor class.

    Parameters:
        focal_class (str):
            Focal class name.
        neighbor_class (str):
            Neighbor class name.
        link_combos (List[str]):
            List of all possible link class combinations.

    Returns:
        str:
            Link class name.
    """
    for link_class in link_combos:
        class1, class2 = link_class.split("-")
        if (focal_class == class1 and neighbor_class == class2) or (
            focal_class == class2 and neighbor_class == class1
        ):
            return link_class
    return None


def weights2gdf(
    gdf: gpd.GeoDataFrame, w: W, parallel: bool = False
) -> gpd.GeoDataFrame:
    """Convert a `libpysal` weights object to a `geopandas.GeoDataFrame`.

    Add class names and node centroids to the dataframe.

    Note:
        if `w.neighbors` is empty, this will return None.

    Parameters:
        gdf (gpd.GeoDataFrame):
            GeoDataFrame of the nodes.
        w (W):
            PySAL weights object.
        parallel (bool, default=False):
            Whether to use parallel processing.

    Returns:
        gpd.GeoDataFrame:
            GeoDataFrame of the links.

    Examples:
        Convert `libpysal` weights from `InterfaceContext` to `geopandas.GeoDataFrame`.
        >>> from cellseg_gsontools.spatial_context import InterfaceContext
        >>> from cellseg_gsontools.links import weights2gdf
        >>> iface_context = InterfaceContext(
        ...     area_gdf=areas,
        ...     cell_gdf=cells,
        ...     top_labels="area_cin",
        ...     bottom_labels="areastroma",
        ...     silence_warnings=True,
        ...     min_area_size=100000.0,
        ... )
        >>> iface_context.fit(verbose=False)
        >>> w = iface_context.context2weights("border_network")
        >>> link_gdf = weights2gdf(cells, w)
    """
    if not w.neighbors:
        return

    # get all possible link class combinations
    classes = sorted(gdf.class_name.unique().tolist())
    link_combos = get_link_combinations(classes)

    # init link gdf
    link_gdf = w.to_adjlist(remove_symmetric=True, drop_islands=True).reset_index()

    # add centroids and class names
    link_gdf.loc[:, "focal_centroid"] = gdf.loc[link_gdf.focal].centroid.to_list()
    link_gdf.loc[:, "neighbor_centroid"] = gdf.loc[link_gdf.neighbor].centroid.to_list()
    link_gdf.loc[:, "focal_class_name"] = gdf.loc[link_gdf.focal].class_name.to_list()
    link_gdf.loc[:, "neighbor_class_name"] = gdf.loc[
        link_gdf.neighbor
    ].class_name.to_list()

    func = partial(_get_link_class, link_combos=link_combos)
    link_gdf["class_name"] = gdf_apply(
        link_gdf,
        func=func,
        columns=["focal_class_name", "neighbor_class_name"],
        axis=1,
        parallel=parallel,
    )

    link_gdf["geometry"] = gdf_apply(
        link_gdf,
        func=_create_link,
        columns=["focal_centroid", "neighbor_centroid"],
        axis=1,
        parallel=parallel,
    )
    link_gdf = link_gdf.set_geometry("geometry")

    return link_gdf


def link_counts(
    gdf: gpd.GeoDataFrame, w: W, classes: Tuple[str, ...]
) -> Dict[str, int]:
    """Get the link-type counts of a geodataframe given a spatial weights object `w`.

    Parameters:
        gdf (gpd.GeoDataFrame):
            Input geodataframe. Has to have a `class_name` column.
        w (libysal.weights.W):
            Libpysal spatial weights object of the gdf.
        classes (Tuple[str, ...]):
            A list/tuple containing the classes of your dataset.

    Returns:
        Dict[str, int]:
            A contigency dictionary.

    Examples:
        Get the link types of the tumor-stroma interfaces.
        >>> from cellseg_gsontools.spatial_context import InterfaceContext
        >>> from cellseg_gsontools.links import link_counts
        >>> iface_context = InterfaceContext(
        ...     area_gdf=areas,
        ...     cell_gdf=cells,
        ...     top_labels="area_cin",
        ...     bottom_labels="areastroma",
        ...     silence_warnings=True,
        ...     min_area_size=100000.0,
        ... )
        >>> classes = [
        ...     "inflammatory",
        ...     "connective",
        ...     "glandular_epithel",
        ...     "squamous_epithel",
        ...     "neoplastic",
        ... ]
        >>> iface_context.fit(verbose=False)
        >>> w = iface_context.context2weights("border_network")
        >>> link_counts(cells, w, classes)
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
