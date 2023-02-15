import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely


def import_slide(slide_path: str):
    """Import image into matrix objects.

    Args:
    ---------
        slide_path

    Returns:
    ---------
        np.array of image
    """
    return cv2.cvtColor(cv2.imread(slide_path), cv2.COLOR_BGR2RGB)


def get_size(slide_path: str) -> tuple[int, int]:
    """Get image size from path.

    Args:
    ---------
        slide_path

    Returns:
    ---------
        Image size (H,W)
    """
    x = slide_path.split("x-")[1].split("_")[0]
    y = slide_path.split("y-")[1].split(".")[0]
    return (int(x), int(y))


def import_json(json_path: str):
    """Read geojson file into Geodataframe object.

    Args:
    ---------
        json_path

    Returns:
    ---------
        gpd.Geodataframe
    """
    df = pd.read_json(json_path)
    df["geometry"] = df["geometry"].apply(shapely.geometry.shape)
    return gpd.GeoDataFrame(df).set_geometry("geometry")


def prep_json(gdf: gpd.geodataframe) -> gpd.geodataframe:
    """Clean imported geodataframe.

    Args:
    ---------
        Geodataframe

    Returns:
    ---------
        Cleaned Geodataframe
    """
    # drop invalid geometries if there are any after buffer
    gdf.geometry = gdf.geometry.buffer(0)
    gdf = gdf[gdf.is_valid]

    # drop empty geometries
    gdf = gdf[~gdf.is_empty]

    # drop geometries that are not polygons
    gdf = gdf[gdf.geom_type == "Polygon"]

    # add bounding box coords of the polygons to the gdfs
    # and correct for the max coords
    gdf["xmin"] = gdf.bounds["minx"].astype(int)
    gdf["ymin"] = gdf.bounds["miny"].astype(int)
    gdf["ymax"] = gdf.bounds["maxy"].astype(int) + 1
    gdf["xmax"] = gdf.bounds["maxx"].astype(int) + 1

    gdf["class_name"] = gdf["properties"].apply(lambda x: x["classification"]["name"])

    return gdf


def get_cells(gdf: gpd.geodataframe, offset: tuple[int, int]):
    """Retrieve cell centroid.

    Args:
    ---------
        cells: Geodataframe of cells

    Returns:
    ---------
        cells dataframe with geometry replaced by centroid
    """
    cells = gdf
    cells.geometry = cells.geometry.centroid.translate(
        yoff=-int(offset[1]), xoff=-int(offset[0])
    )
    return cells


def get_coords(cells):
    """Retrieve points objects from geodataframe.

    Args:
    ---------
        cells: Geodataframe of cells

    Returns:
    ---------
        cells: array of shapley.ops.Points
    """
    points = []
    for i in range(0, len(cells["geometry"])):
        point = cells["geometry"].iloc[i]
        points.append(point)
    return points


def get_points(cells):
    """Retrieve cell coordinates from geodataframe.

    Args:
    ---------
        cells: Geodataframe of cells

    Returns:
    ---------
        cells: array of coordinates
    """
    points = []
    for i in range(0, len(cells["geometry"])):
        point = cells["geometry"].iloc[i].coords[:][0]
        points.append(point)
    return points


def create_grid(dx: int, dy: int, n: int = 1) -> np.meshgrid:
    """Create an image matrix of size (dx,dy).

    Args:
    ---------
        dy: int
        dx: int
        n: scale

    Returns:
    ---------
        np.meshrid of size (dx, dy)
    """
    X = np.linspace(0, dx, int(dx / n))
    Y = np.linspace(0, dy, int(dy / n))
    return np.meshgrid(X, Y)


def filter_outliers(cells, dist: np.array):
    """Filter least meaningful cells based on a distribution.

    Args:
    ---------
        cells: Geodataframe of cells
        dist: Heatmap for cells
    Returns:
    ---------
        Cells picked based on magnitude (least significant cells).
    """
    coords = cells["geometry"]
    cells["weight"] = [dist[int(p.coords[0][1])][int(p.coords[0][0])] for p in coords]
    top = cells.sort_values("weight", ascending=True)

    worst = list(top.iterrows())[0]
    th = (
        int(np.format_float_scientific(worst[1].weight, exp_digits=1, precision=0)[-1])
        - 1
    )

    return cells[cells["weight"] > 10 ** (-th)]


def sample_table(cells, k=2500):
    """Take random sample of cells.

    Args:
    ---------
        cells: Geodataframe of cells

    Returns:
    ---------
        sample: Geodataframe of cells
    """
    index = np.random.choice(np.array(cells.index), k)
    sample = cells.loc[index]

    return sample