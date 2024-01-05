from pathlib import Path
from typing import Tuple, Union

import geopandas as gpd

from ..utils import pre_proc_gdf, read_gdf
from .save_utils import get_xy_coords

__all__ = ["GSONTile"]


class GSONTile:
    def __init__(
        self,
        filename: Union[Path, str],
        tile_size: Tuple[int, int] = (1000, 1000),
        min_size: int = 30,
    ) -> None:
        """Handle geojson annotations of a tile.

        Allows easy extraction of border and non-border annotations.

        Parameters:
            filename (Union[Path, str]):
                Name of the input geojson file.
            tile_size (Tuple[int, int]):
                Height and width of the tile in pixels.
            min_size (int):
                Minimum size of the annotations in pixels.

        Examples:
            Get non-border annotations of the tile.
            >>> from cellseg_gsontools.tile import GSONTile
            >>> tile = GSONTile("path/to/tile.json")
            >>> tile.non_border_annots
        """
        self.filename = Path(filename)
        self.min_size = min_size
        self.xmin, self.ymin = get_xy_coords(filename)
        self.xmax = self.xmin + tile_size[0]
        self.ymax = self.ymin + tile_size[1]

        self.gdf = pre_proc_gdf(read_gdf(self.filename), min_size=min_size)

    def __len__(self):
        """Return the length of the gson obj."""
        return len(self.gdf)

    def __repr__(self):
        """Return the representation of the gson obj."""
        return f"{self.__class__.__name__}({self.filename})"

    @property
    def non_border_annots(self) -> gpd.GeoDataFrame:
        """Get all the annotations/polygons that do not touch any edges of the tile.

        Note: Resets the index of the returned gdf.
        Note: Origin in the top-left corner of the image/tile. In geoformat, the origin
                is in the bottom-left corner.
        """
        not_r = self.gdf["xmax"] != self.xmax
        not_l = self.gdf["xmin"] != self.xmin
        not_b = self.gdf["ymax"] != self.ymax
        not_t = self.gdf["ymin"] != self.ymin
        non_border_annots = self.gdf[not_r & not_l & not_b & not_t].copy()
        non_border_annots = non_border_annots.reset_index(drop=True)
        return non_border_annots

    @property
    def right_border_annots(self) -> gpd.GeoDataFrame:
        """Get all the annotations/polygons that touch the right edge of the tile.

        Note: Resets the index of the returned gdf.
        Note: Origin in the top-left corner of the image/tile. In geoformat, the origin
                is in the bottom-left corner.
        """
        r_border_anns = self.gdf[self.gdf["xmax"] == self.xmax].copy()

        # translate one unit right
        translated_coords = r_border_anns.translate(xoff=1.0)
        r_border_anns["geometry"] = translated_coords
        r_border_anns = r_border_anns.reset_index(drop=True)

        return r_border_anns

    @property
    def left_border_annots(self) -> gpd.GeoDataFrame:
        """Get all the annotations/polygons that touch the left edge of the tile.

        Note: Resets the index of the returned gdf.
        Note: Origin in the top-left corner of the image/tile. In geoformat, the origin
                is in the bottom-left corner.
        """
        l_border_anns = self.gdf[self.gdf["xmin"] == self.xmin].copy()
        l_border_anns = l_border_anns.reset_index(drop=True)

        return l_border_anns

    @property
    def bottom_border_annots(self) -> gpd.GeoDataFrame:
        """Get all the annotations/polygons that touch the bottom edge of the tile.

        Note: Resets the index of the returned gdf.
        Note: Origin in the top-left corner of the image/tile. In geoformat, the origin
                is in the bottom-left corner.
        """
        b_border_anns = self.gdf[self.gdf["ymax"] == self.ymax].copy()

        # translate 1-unit down
        translated_coords = b_border_anns.translate(yoff=1.0)
        b_border_anns["geometry"] = translated_coords
        b_border_anns = b_border_anns.reset_index(drop=True)

        return b_border_anns

    @property
    def bottom_left_border_annots(self) -> gpd.GeoDataFrame:
        """Get all the annotations/polygons that touch the bottom edge of the tile.

        Note: Resets the index of the returned gdf.
        Note: Origin in the top-left corner of the image/tile. In geoformat, the origin
                is in the bottom-left corner.
        """
        b = (self.gdf["ymax"] == self.ymax) & (self.gdf["xmin"] == self.xmin)
        bl_border_anns = self.gdf[b].copy()

        # translate 1-unit down
        translated_coords = bl_border_anns.translate(yoff=1.0)
        bl_border_anns["geometry"] = translated_coords
        bl_border_anns = bl_border_anns.reset_index(drop=True)

        return bl_border_anns

    @property
    def bottom_right_border_annots(self) -> gpd.GeoDataFrame:
        """Get all the annotations/polygons that touch the bottom edge of the tile."""
        b = (self.gdf["ymax"] == self.ymax) & (self.gdf["xmax"] == self.xmax)
        br_border_anns = self.gdf[b].copy()

        # translate 1-unit down and right
        translated_coords = br_border_anns.translate(yoff=1.0, xoff=1.0)
        br_border_anns["geometry"] = translated_coords
        br_border_anns = br_border_anns.reset_index(drop=True)

        return br_border_anns

    @property
    def top_border_annots(self) -> gpd.GeoDataFrame:
        """Get all the annotations/polygons that touch the top edge of the tile.

        Note: Resets the index of the returned gdf.
        Note: Origin in the top-left corner of the image/tile. In geoformat, the origin
                is in the bottom-left corner.
        """
        t_border_anns = self.gdf[self.gdf["ymin"] == self.ymin].copy()
        t_border_anns = t_border_anns.reset_index(drop=True)

        return t_border_anns

    @property
    def top_right_border_annots(self) -> gpd.GeoDataFrame:
        """Get all the annotations/polygons that touch the top edge of the tile.

        Note: Resets the index of the returned gdf.
        Note: Origin in the top-left corner of the image/tile. In geoformat, the origin
                is in the bottom-left corner.
        """
        b = (self.gdf["ymin"] == self.ymin) & (self.gdf["xmax"] == self.xmax)
        tr_border_anns = self.gdf[b].copy()
        translated_coords = tr_border_anns.translate(xoff=1.0)
        tr_border_anns["geometry"] = translated_coords
        tr_border_anns = tr_border_anns.reset_index(drop=True)

        return tr_border_anns

    @property
    def top_left_border_annots(self) -> gpd.GeoDataFrame:
        """Get all the annotations/polygons that touch the top edge of the tile."""
        b = (self.gdf["ymin"] == self.ymin) & (self.gdf["xmin"] == self.xmin)
        tl_border_anns = self.gdf[b].copy()
        tl_border_anns = tl_border_anns.reset_index(drop=True)

        return tl_border_anns
