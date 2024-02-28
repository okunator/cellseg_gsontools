from typing import Dict

import geopandas as gpd
import numpy as np
import rasterio
import shapely
from rasterio.features import geometry_mask

__all__ = ["gdf2instance", "gdf2semantic"]


def gdf2instance(gdf: gpd.GeoDataFrame, reset_index: bool = False) -> gpd.GeoDataFrame:
    """Converts a GeoDataFrame to an instance segmentation mask.

    Parameters:
        gdf (gpd.GeoDataFrame):
            GeoDataFrame with a "class_name" column.

    Returns:
        np.ndarray:
            Instance segmentation mask of the input gdf.
    """
    xmin, ymin, xmax, ymax = gdf.total_bounds
    width = xmax - xmin
    height = ymax - ymin
    image_shape = (int(height), int(width))
    out_mask = np.zeros(image_shape, dtype=np.int32)
    for i, (ix, row) in enumerate(gdf.iterrows()):
        mask = geometry_mask(
            [shapely.affinity.translate(row.geometry, -xmin, -ymin)],
            out_shape=image_shape,
            transform=rasterio.Affine(1, 0, 0, 0, 1, 0),
            invert=True,
            all_touched=True,
        )
        if reset_index:
            out_mask[mask] = int(i + 1)
        else:
            out_mask[mask] = int(ix)

    return out_mask


def gdf2semantic(
    gdf: gpd.GeoDataFrame, class_dict: Dict[str, int] = None
) -> np.ndarray:
    """Converts a GeoDataFrame to a semantic segmentation mask.

    Parameters:
        gdf (gpd.GeoDataFrame):
            GeoDataFrame with a "class_name" column.
        class_dict (Dict[str, int]):
            Dictionary mapping class names to integers. If None, the classes
            will be mapped to integers in the order they appear in the GeoDataFrame.

    Returns:
        np.ndarray:
            Semantic segmentation mask of the input gdf.
    """
    xmin, ymin, xmax, ymax = gdf.total_bounds
    width = xmax - xmin
    height = ymax - ymin
    image_shape = (int(height), int(width))
    out_mask = np.zeros(image_shape, dtype=np.int32)
    for i, (cl, gdf) in enumerate(gdf.explode(index_parts=True).groupby("class_name")):
        mask = geometry_mask(
            gdf.geometry.translate(-xmin, -ymin),
            out_shape=image_shape,
            transform=rasterio.Affine(1, 0, 0, 0, 1, 0),
            invert=True,
            all_touched=True,
        )
        if class_dict is None:
            out_mask[mask] = int(i + 1)
        else:
            out_mask[mask] = class_dict[cl]

    return out_mask
