from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import geopandas as gpd
import pandas as pd
from shapely.geometry import MultiPolygon
from shapely.ops import unary_union
from tqdm import tqdm

from ..utils import set_uid
from ._base_merger import BaseGSONMerger
from .gson_tile import GSONTile
from .poly_utils import merge_overlaps, validate_and_simplify
from .save_utils import gdf_to_file

__all__ = ["CellMerger"]


class CellMerger(BaseGSONMerger):
    """Merge adjascent cell annotation files to one file.

    Note:
        Assumes:

        - Input files contain nuclei/cell instance segmentation annotations.
        - Files have start x and y coords embedded in filename e.g. `x-[coord]_y-[coord]`
        - Input Tiles are the same size.
        - Allowed input file-formats `.json`, `.geojson`, `.feather`, `.parquet`

    Parameters:
        in_dir (Union[Path, str]):
            Path to the directory containing the annotation files of tiles.
        tile_size (Tuple[int, int]):
            Height and width of the tiles in pixels.

    Attributes:
        border_annots (gpd.GeoDataFrame):
            A gdf of the merged border annotations. Available after merging.
        non_border_annots (gpd.GeoDataFrame):
            A gdf of the merged non-border annotations. Available after merging.
        annots (gpd.GeoDataFrame):
            A gdf of the resulting annotations. Available after merging.
    """

    def __init__(
        self, in_dir: Union[Path, str], tile_size: Tuple[int, int] = (1000, 1000)
    ) -> None:
        super().__init__(in_dir, tile_size)

        # Lookup to manage the relations between the main and adjacent tiles
        # main tile is the current tile and adj is the adjacent tile
        self.neighbor_relations = {
            "right": {"main": "right", "adj": "left"},
            "left": {"main": "left", "adj": "right"},
            "top": {"main": "top", "adj": "bottom"},
            "bottom": {"main": "bottom", "adj": "top"},
        }

        # Lookup for already visited neighbors
        self.visited = {
            f.name: {
                "left": None,
                "right": None,
                "top": None,
                "bottom": None,
                "non_border": None,
                "top_right": None,
                "top_left": None,
                "bottom_left": None,
                "bottom_right": None,
            }
            for f in self.files
        }

    def merge_dir(
        self,
        out_fn: Optional[Union[Path, str]] = None,
        format: Optional[str] = None,
        verbose: bool = True,
    ) -> None:
        """Merge all the instance segmentation files in the input directory.

        Parameters:
            out_fn (Union[Path, str]):
                Filename for the output file. If None, the merged gdf is saved to the
                class attribute `self.annots` only.
            format (str):
                The format of the output geojson file. One of: ".feather", ".parquet",
                ".geojson", None. This is ignored if `out_fn` is None.
            verbose (bool):
                Whether to show a progress bar or not.

        Examples:
            Write geojson files to a standard '.geojson' file.
            >>> from cellseg_gsontools.merging import CellMerger
            >>> merger = CellMerger("/path/to/geojsons/", tile_size=(1000, 1000))
            >>> merger.merge_dir("/path/to/output.json", format="geojson")

            Write input geojson files to feather file.
            >>> from cellseg_gsontools.merging import CellMerger
            >>> merger = CellMerger("/path/to/geojsons/", tile_size=(1000, 1000))
            >>> merger.merge_dir("/path/to/output.feather", format="feather")

            Write input geojson files to parquet file.
            >>> from cellseg_gsontools.merging import CellMerger
            >>> merger = CellMerger("/path/to/geojsons/", tile_size=(1000, 1000))
            >>> merger.merge_dir("/path/to/output.parquet", format="parquet")
        """
        if out_fn is not None:
            out_fn = Path(out_fn)

        if format not in (".feather", ".parquet", ".geojson", None):
            raise ValueError(
                f"Invalid format. Got: {format}. Allowed: .feather, .parquet, .geojson"
            )

        # merge the tiles
        self.annots = self._merge(verbose=verbose)
        if verbose:
            msg = f"{format}-format" if out_fn is not None else "`self.annots`"
            print(f"Saving the merged geojson file: {out_fn} to {msg}")

        if out_fn is not None:
            # save the merged geojson
            gdf_to_file(self.annots, out_fn, format)

    def _get_non_border_polygons(self, gson: GSONTile) -> List[Dict[str, Any]]:
        """Get all the polygons that do not touch any edges of the tile.

        Parameters
        ----------
            gson : GSONTile
                GSONTile obj of a geojson tile.

        Returns
        -------
            Tuple[List[Polygon], List[str]]:
                A list of cell polygon objects and a list of the corresponding classes.
        """
        nb_annots = gson.non_border_annots

        classes = []
        new_polys = []
        if not nb_annots.empty:
            for poly, c in zip(nb_annots.geometry, nb_annots.class_name):
                poly = validate_and_simplify(poly, simplify=True, buffer=0.0)
                new_polys.append(poly)
                classes.append(c)

        return new_polys, classes

    def _merge_adj_ploygons(
        self, gson: GSONTile, gson_adj: GSONTile, adj_pos: str
    ) -> List[Dict[str, Any]]:
        """Merge adjascent polygons in two adjacsent geojsons.

        This concatenates the cells that are split at the image borders.

        Parameters
        ----------
            gson : GSONTile
                GSONTile obj of the geojson of the main tile.
            gson_adj : GSONTile
                GSONTile obj of the geojson of the adjascnet tile.
            adj_pos : str
                The postition of the adjascent tile relative to the main tile.
                One of: "left", "right", "bottom", "bottomleft", "bottomright", "top",
                "topleft", "topright"

        Returns
        -------
            Tuple[List[Polygon], List[str]]:
                A list of cell polygon objects and a list of the corresponding classes.
        """
        # Get the polygons that end/start at the image border
        if adj_pos == "right":
            border_annots_main = gson.right_border_annots
            border_annots_adj = gson_adj.left_border_annots
        elif adj_pos == "left":
            border_annots_main = gson.left_border_annots
            border_annots_adj = gson_adj.right_border_annots
        elif adj_pos == "bottom":
            border_annots_main = gson.bottom_border_annots
            border_annots_adj = gson_adj.top_border_annots
        elif adj_pos == "top":
            border_annots_main = gson.top_border_annots
            border_annots_adj = gson_adj.bottom_border_annots

        # combine polygons that intersect/touch between two image tiles
        # (cells that are split in two between two image tiles)
        new_classes = []
        new_polys = []
        if not border_annots_main.empty and not border_annots_adj.empty:
            for main_poly, c1 in zip(
                border_annots_main.geometry, border_annots_main.class_name
            ):
                main_poly = validate_and_simplify(main_poly, buffer=0.0)
                for adj_poly, c2 in zip(
                    border_annots_adj.geometry, border_annots_adj.class_name
                ):
                    adj_poly = validate_and_simplify(adj_poly, buffer=0.0)

                    # combine the polygons if they intersect
                    if main_poly.intersects(adj_poly):
                        new_poly = unary_union([main_poly, adj_poly])

                        # do some simplifying
                        new_poly = validate_and_simplify(
                            new_poly, simplify=True, buffer=0.0
                        )

                        if isinstance(new_poly, MultiPolygon):
                            print("has multipoly")

                        # take the class of the larger object
                        if adj_poly.area >= main_poly.area:
                            new_class = c2
                        else:
                            new_class = c1

                        new_polys.append(new_poly)
                        new_classes.append(new_class)

        return new_polys, new_classes

    def _merge(self, verbose: bool = True) -> gpd.GeoDataFrame:
        """Merge all annotations in the files to one.

        Handles the split cells at the image borders.

        NOTE: This does not handle corners
        """
        non_border_annots = []
        non_border_classes = []
        border_annots = []
        border_classes = []
        pbar = tqdm(self.files) if verbose else self.files
        for f in pbar:
            if verbose:
                pbar.set_description(f"Processing file: {f.name}")

            # get adjascent tiles
            adj = self._get_adjascent_tiles(f.name)

            # Init GSONTile obj
            gson = GSONTile(f, tile_size=self.tile_size)
            if gson.gdf is not None and not gson.gdf.empty:
                # add the non border polygons
                if self.visited[f.name]["non_border"] is None:
                    non_border_polygons, non_border_cls = self._get_non_border_polygons(
                        gson
                    )
                    non_border_annots.extend(non_border_polygons)
                    non_border_classes.extend(non_border_cls)
                    self.visited[f.name]["non_border"] = f.name

                # loop the adjascent tiles and add the border polygons
                for pos, f_adj in adj.items():
                    if f_adj is not None:
                        if self.visited[f.name][pos] is None:
                            gson_adj = GSONTile(f_adj, tile_size=self.tile_size)

                            if gson_adj.gdf is not None and not gson_adj.gdf.empty:
                                border_polygons, border_cls = self._merge_adj_ploygons(
                                    gson, gson_adj, pos
                                )
                                border_annots.extend(border_polygons)
                                border_classes.extend(border_cls)

                                # update lookup
                                main_pos = self.neighbor_relations[pos]["main"]
                                adj_pos = self.neighbor_relations[pos]["adj"]
                                self.visited[f_adj.name][adj_pos] = f.name
                                self.visited[f.name][main_pos] = f_adj.name

        # save the annotations to class attributes
        self.non_border_annots = gpd.GeoDataFrame(
            {"geometry": non_border_annots, "class_name": non_border_classes}
        )
        border_annots = gpd.GeoDataFrame(
            {"geometry": border_annots, "class_name": border_classes}
        )
        self.border_annots = merge_overlaps(border_annots)  # merge overlapping objects
        annots = set_uid(
            pd.concat([self.non_border_annots, self.border_annots]),
            0,
            id_col="id",
            drop=False,
        )
        return annots[annots.class_name != "background"]
