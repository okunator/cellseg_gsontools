import geopandas as gpd
import libpysal
import shapely

__all__ = ["graph_pos", "neighborhood_density"]


def graph_pos(cells: gpd.geodataframe):
    """Create pos object from gdf for graph computations.

    Args:
    ---------
        cells: Geodataframe
    Returns:
    ---------
        Dict with cell coordinates
    """
    coords = [(c.x, c.y) for c in cells["geometry"]]
    pos = [i for i in zip(cells.index, coords)]
    return dict(pos)


def neighborhood_density(cells: gpd.geodataframe, k: int = 4) -> gpd.geodataframe:
    """Calculate neighborhood density for points with sum of distances to KNN.

    Args:
    ---------
        cells: Geodataframe of cells
        k: parameter for KNN

    Returns:
    ---------
        np.array of weights
    """
    G = libpysal.weights.KNN.from_dataframe(cells, k=k)

    weights = []
    pos = graph_pos(cells)

    for i in cells.index:
        weight = 0

        for edge in G.neighbors[i]:
            cell = shapely.ops.Point(pos[i])
            neighbor = shapely.ops.Point(pos[edge])
            weight += cell.distance(neighbor)

        weights.append(weight)

    cells["weights"] = weights

    return cells
