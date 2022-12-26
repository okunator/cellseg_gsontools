import cv2
import pandas as pd
import geopandas as gpd
import numpy as np
import shapely

def import_slide(slide_path):
    return cv2.cvtColor(cv2.imread(slide_path), cv2.COLOR_BGR2RGB)

def get_size(slide_path):
    x = slide_path.split('x-')[1].split('_')[0]
    y = slide_path.split('y-')[1].split('.')[0]
    return (x,y)

def import_json(json_path):
    df = pd.read_json(json_path)
    df["geometry"] = df["geometry"].apply(shapely.geometry.shape)
    return gpd.GeoDataFrame(df).set_geometry("geometry")

def prep_json(gdf):
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

def get_cells(gdf, offset):
    cells = gdf
    cells.geometry = cells.geometry.centroid.translate(yoff=-int(offset[1]), xoff=-int(offset[0]))
    return cells

def create_grid(dx, dy):
    X = np.linspace(0, dx, dx)
    Y = np.linspace(0, dy, dy)
    return np.meshgrid(X,Y)
