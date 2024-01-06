from functools import partial
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import geopandas as gpd
import networkx as nx
import pandas as pd
from libpysal.weights import W, fuzzy_contiguity
from shapely.ops import unary_union
from tqdm import tqdm

from ..multiproc import run_pool
from ..utils import read_gdf, set_uid
from ._base_merger import BaseGSONMerger
from .poly_utils import validate_and_simplify
from .save_utils import gdf_to_file

__all__ = ["AreaMerger"]


class AreaMerger(BaseGSONMerger):
    """Merge the area/tissue annotation files of the tiles to one file.

    Note:
        Assumes:

        - Input files contain area/tissue semantic segmentation annotations.
        - Files have start x and y coords embedded in filename e.g. `x-[coord]_y-[coord]`
        - Tiles are the same size.
        - Allowed input file-formats `.json`, `.geojson`, `.feather`, `.parquet`

    Parameters:
        in_dir (Union[Path, str]):
            Path to the directory containing the annotation files of tiles.

    Attributes:
        annots (gpd.GeoDataFrame):
            A gdf of the resulting annotations. Available after merging.
    """

    def __init__(self, in_dir: Union[Path, str]) -> None:
        super().__init__(in_dir, None)

    def merge_dir(
        self,
        out_fn: Optional[Union[Path, str]] = None,
        format: Optional[str] = None,
        verbose: bool = True,
        parallel: bool = False,
    ) -> None:
        """Merge all the semantic segmentation files in the input directory into one.

        Note:
            Unlike the CellMerger.merge_dir() -method, this can be parallelized with
            `parallel=True` -argument.

        Parameters:
            out_fn (Union[Path, str]):
                Filename for the output file. If None, the merged gdf is saved to the
                class attribute `self.annots` only.
            format (str):
                The format of the output geojson file. One of: "feather", "parquet",
                "geojson", None. This is ignored if `out_fn` is None.
            verbose (bool):
                Whether to show a progress bar or not.
            parallel (bool):
                Whether to use parallel processing or not.

        Examples:
            Write feather files to a '.geojson' file.
            >>> from cellseg_gsontools.merging import AreaMerger
            >>> merger = AreaMerger("/path/to/feather_files/")
            >>> merger.merge_dir("/path/to/output.json", format="geojson")

            Write input geojson files to feather file.
            >>> from cellseg_gsontools.merging import AreaMerger
            >>> merger = AreaMerger("/path/to/geojsons/")
            >>> merger.merge_dir("/path/to/output.feather", format="feather")

            Write input parquet files to parquet file.
            >>> from cellseg_gsontools.merging import AreaMerger
            >>> merger = AreaMerger("/path/to/parquet_files/")
            >>> merger.merge_dir("/path/to/output.parquet", format="parquet")
        """
        if out_fn is not None:
            out_fn = Path(out_fn)

        if format not in (".feather", ".parquet", ".geojson", None):
            raise ValueError(
                f"Invalid format. Got: {format}. Allowed: .feather, .parquet, .geojson"
            )

        # merge the tiles
        self.annots = self._merge(verbose=verbose, parallel=parallel)

        if verbose:
            msg = f"{format}-format" if out_fn is not None else "`self.annots`"
            print(f"Saving the merged geojson file: {out_fn} to {msg}")

        if out_fn is not None:
            # save the merged geojson
            gdf_to_file(self.annots, out_fn, format)

    def _read_files_to_gdf(
        self, files: List[Path], verbose: bool = True
    ) -> gpd.GeoDataFrame:
        """Read in the input files to a gdf."""
        cols = None
        rows = []
        pbar = tqdm(files, total=len(files)) if verbose else files
        for file in pbar:
            gdf = read_gdf(file)
            if gdf is not None and not gdf.empty:
                cols = gdf.columns
                for _, row in gdf.iterrows():
                    rows.append(row)

        return gpd.GeoDataFrame(rows, columns=cols)

    def _merge_adjascent_polygons(
        self, gdf: gpd.GeoDataFrame, w: W, verbose: bool = True, cl: str = None
    ) -> gpd.GeoDataFrame:
        """Divide spatial weights into subgraphs & merge the polygons in each."""
        # Get all disconnected subgraphs.
        G = w.to_networkx()
        sub_graphs = [
            W(nx.to_dict_of_lists(G.subgraph(c).copy()))
            for c in nx.connected_components(G)
        ]

        # loop over the subgraphs
        pbar = tqdm(sub_graphs, total=len(sub_graphs)) if verbose else sub_graphs
        result_polygons = []
        for sub_w in pbar:
            if verbose:
                pbar.set_description(f"Processing {cl} connected regions:")

            # init a visited lookup table for nodes
            visited = {node: False for node in sub_w.neighbors.keys()}

            # loop over the nodes/polygons in the subgraph
            polygons_to_merge = []
            for node, neighs in sub_w.neighbors.items():
                # if an island, buffer the polygon
                if not neighs:
                    poly = gdf.loc[node].geometry.buffer(2)
                    result_polygons.append(
                        validate_and_simplify(poly, simplify=True, buffer=0.0)
                    )
                    continue

                # if not visited, check if it intersects with any of its neighbors
                # and add it to the list of polygons to merge
                if not visited[node]:
                    poly = gdf.loc[node].geometry.buffer(2)
                    # poly = validate_and_simplify(poly, simplify=True, buffer=0.0)
                    inter = [poly]
                    for neigh in neighs:
                        if not visited[neigh]:
                            neigh_poly = gdf.loc[neigh].geometry.buffer(2)
                            neigh_poly = validate_and_simplify(
                                poly, simplify=True, buffer=0.0
                            )
                            if poly.intersects(neigh_poly):
                                inter.append(neigh_poly)
                                visited[neigh_poly] = True

                    visited[node] = True
                    polygons_to_merge.extend(inter)

            # don't merge if there are no polygons to merge
            if polygons_to_merge:
                result_polygons.append(unary_union(polygons_to_merge))

        # return the merged polygons as gdf
        out = gpd.GeoDataFrame({"geometry": result_polygons, "class_name": cl})
        return out[~out.is_empty].explode(index_parts=False).reset_index(drop=True)

    def _merge_one(
        self, in_gdf: gpd.GeoDataFrame, cl: str, verbose: bool = True
    ) -> gpd.GeoDataFrame:
        """Merge the polygons in one class."""
        in_gdf = set_uid(in_gdf)

        w = fuzzy_contiguity(
            in_gdf,
            buffering=True,
            buffer=2,
            predicate="intersects",
            silence_warnings=True,
        )

        return self._merge_adjascent_polygons(in_gdf, w, verbose=verbose, cl=cl)

    def _merge_one_wrap(self, args: Tuple[Any], verbose: bool = True):
        return self._merge_one(*args, verbose=verbose)

    def _merge(self, verbose: bool = True, parallel: bool = True) -> gpd.GeoDataFrame:
        """Merge the polygons in by class."""
        gdf = self._read_files_to_gdf(self.files)
        classes = gdf["class_name"].unique()

        if not parallel:
            polys = []
            for cl in classes:
                in_gdf = gdf[gdf["class_name"] == cl]
                polys.append(self._merge_one(in_gdf, cl, verbose=verbose))
        else:
            merge_func = partial(self._merge_one_wrap, verbose=False)

            polys = run_pool(
                merge_func,
                [(gdf[gdf["class_name"] == cl], cl) for cl in classes],
                n_jobs=len(classes),
                pbar=verbose,
                pooltype="thread",
                maptype="uimap",
            )

        return set_uid(pd.concat(polys, ignore_index=True))
