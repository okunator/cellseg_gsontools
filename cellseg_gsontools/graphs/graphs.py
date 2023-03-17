import geopandas as gpd
import libpysal
import matplotlib.pyplot as plt
import networkx as nx
import shapely


def points_to_coords(points):
    """Convert array of point object into array of coordinates.

    Args:
    ---------
        points: list of point objects for cells
    Returns:
    ---------
        array of cell coordinates
    """
    return [p.coords[0] for p in points]


def graph_pos(cells: gpd.geodataframe):
    """Get cell coordinate object from Geodataframe.

    Args:
    ---------
        cells: Geodataframe
    Returns:
    ---------
        Library with cell coordinates
    """
    positions = [i for i in zip(cells.index, points_to_coords(cells["geometry"]))]

    graph_pos = {}
    for position in positions:
        graph_pos[position[0]] = position[1]

    return graph_pos


def list_to_pos(cells: gpd.geodataframe):
    """Convert graph_pos to array of cell coordinates.

    Args:
    ---------
        cells: Geodataframe
    Returns:
    ---------
        list of cell coordinates
    """
    return list(graph_pos(cells).values())


def graph_weights(cells: gpd.geodataframe, G: nx.graph):
    """Calculate weights for graph nodes based on connecting edge lengths.

    Args:
    ---------
        cells: Geodataframe of cells
            G: nx.Graph object

    Returns:
    ---------
        array of weights for graph nodes
    """
    weights = []
    pos = list(graph_pos(cells).values())
    for node in range(0, len(cells.index)):
        weight = 0
        for edge in G.edges(node, data=True):
            point1 = shapely.ops.Point(pos[node])
            point2 = shapely.ops.Point(pos[edge[1]])
            weight += point1.distance(point2)
        weights.append(weight / 4)

    return weights


def create_knn(cells: gpd.geodataframe, k=3):
    """Construct KNN graph from geodataframe.

    Args:
    ---------
        cells: Geodataframe of cells
        k: parameter for KNN

    Returns:
    ---------
        nx.Graph object and array of weights
    """
    G = libpysal.weights.KNN.from_dataframe(cells, k=k).to_networkx()
    G_weights = graph_weights(cells, G)

    return G, G_weights


def draw_graph(G: nx.graph, pos, weights):
    """Draw graph with preset graphics.

    Args:
    ---------
        G: nx.graph object
        pos: list of cell coordinates
        weights: list of graph node weights

    Returns:
    ---------
        none
    """
    return nx.draw(G, node_size=20, pos=pos, node_color=weights, cmap=plt.cm.bwr)
