from pathlib import Path
from typing import Optional, Tuple, Union

import geopandas as gpd
from shapely.geometry import MultiPolygon
from shapely.ops import unary_union
from shapely.strtree import STRtree
from tqdm import tqdm

from cellseg_gsontools.utils import pre_proc_gdf, read_gdf

from ._base_merger import BaseGSONMerger
from .poly_utils import validate_and_simplify
from .save_utils import gdf_to_file


class AreaMerger(BaseGSONMerger):
    def __init__(
        self, in_dir: Union[Path, str], tile_size: Tuple[int, int] = (1000, 1000)
    ) -> None:
        """Merge the area/tissue annotation files of the tiles to one file.

        NOTE: Assumes
        - The input files contain area/tissue semantic segmentation annotations.
        - the tiles are named as "x-[coord]_y-[coord](.json|.geojson|.feather|.parquet)"
        - the tiles are the same size.

        Parameters
        ----------
            in_dir : Union[Path, str]
                Path to the directory containing the annotation files of tiles.
            tile_size : Tuple[int, int], default=(1000, 1000)
                Height and width of the tiles in pixels.

        Attributes
        ----------
            annots : gpd.GeoDataFrame
                A gdf of the merged annotations. Available after merging.

        Examples
        --------
        Merge the annotations of the QuPath-formatted tiles in a directory.
        >>> from cellseg_gsontools.merging import AreaMerger
        >>> merger = AreaMerger("/path/to/geojsons/", tile_size=(1000, 1000))
        >>> merger.merge_dir(out.geojson, format="geojson", in_qupath_format="latest")
        """
        super().__init__(in_dir, tile_size)

    def merge_dir(
        self,
        out_fn: Optional[Union[Path, str]] = None,
        format: Optional[str] = None,
        in_qupath_format: Optional[str] = None,
        out_qupath_format: Optional[str] = None,
        verbose: bool = True,
    ) -> None:
        """Merge all the semantic segmentation files in the input directory into one.

        NOTE Assumes:
        - The input files contain area/tissue semantic segmentation annotations.
        - the tiles are named as "x-[coord]_y-[coord](.json|.geojson|.feather|.parquet)"
        - the tiles are the same size.

        Parameters
        ----------
            out_fn : Union[Path, str], optional
                Filename for the output file. If None, the merged gdf is saved to the
                class attribute `self.annots` only.
            format : str, optional
                The format of the output geojson file. One of: "feather", "parquet",
                "geojson", None. This is ignored if `out_fn` is None.
            in_qupath_format : str, optional
                This specifies the qupath format of the input files. If they are not in
                QuPath-readable format set this to None. One of: "old", "latest",
                NOTE: `old` works for versions less than 0.3.0. `latest` works for
                newer versions. This is ignored if `out_fn` is None or format is not
                "geojson".
            out_qupath_format : str, optional
                If this is not None, some additional metadata is added to the geojson
                file to make it properly readable by QuPath when the file is written.
                One of: "old", "latest",
                NOTE: `old` works for versions less than 0.3.0. `latest` works for
                newer versions. This is ignored if `out_fn` is None or format is not
                "geojson".
            verbose : bool, default=True
                Whether to show a progress bar or not.

        Examples
        --------
        Write standard formatted geojson files to a QuPath-readable '.geojson' file.
        >>> from cellseg_gsontools.merging import AreaMerger
        >>> merger = AreaMerger("/path/to/geojsons/", tile_size=(1000, 1000))
        >>> merger.merge_dir(
        ...     "/path/to/output.json", format="geojson", out_qupath_format="latest"
        ... )

        Write input geojson files to feather file.
        >>> from cellseg_gsontools.merging import AreaMerger
        >>> merger = AreaMerger("/path/to/geojsons/", tile_size=(1000, 1000))
        >>> merger.merge_dir("/path/to/output.feather", format="feather")

        Write input geojson files to parquet file.
        >>> from cellseg_gsontools.merging import AreaMerger
        >>> merger = AreaMerger("/path/to/geojsons/", tile_size=(1000, 1000))
        >>> merger.merge_dir("/path/to/output.parquet", format="parquet")
        """
        # merge the tiles
        self._merge(qupath_format=in_qupath_format, verbose=verbose)

        if out_fn is not None:
            msg = f"{format}-format" if format is not None else "`self.annots`"
            qmsg = (
                f"{out_qupath_format} QuPath-readable"
                if out_qupath_format is not None
                else ""
            )
            print(f"Saving the merged geojson file to {qmsg} {msg}")

            # save the merged geojson
            gdf_to_file(self.annots, out_fn, format, out_qupath_format)

    def _merge(
        self,
        qupath_format: Optional[str] = None,
        verbose: bool = True,
    ) -> None:
        """Merge all the files in the input directory into one."""
        # get all the polygons in all of the geojson files
        cols = None
        rows = []
        for file in self.files:
            gdf = pre_proc_gdf(
                read_gdf(file, qupath_format=qupath_format), min_size=1000
            )
            if gdf is not None and not gdf.empty:
                cols = gdf.columns
                for _, row in gdf.iterrows():
                    rows.append(row)

        gdf = gpd.GeoDataFrame(rows, columns=cols).dissolve(
            "class_name", as_index=False, sort=False
        )

        merged_polys = []
        classes = []
        pbar = tqdm(gdf.iterrows(), total=len(gdf)) if verbose else gdf.iterrows()
        for _, row in pbar:
            geo = row.geometry
            c = row.class_name
            classes.append(c)

            if verbose:
                pbar.set_description(f"Processing {c}-annots")

            # merge the given polygons if they intersect and have same class
            if isinstance(geo, MultiPolygon):
                new_coords = []
                tree = STRtree([poly.buffer(1.0) for poly in list(geo.geoms)])

                # convert multipolygons to polygons
                for poly in list(geo.geoms):
                    poly = poly.buffer(1.0)
                    inter = [
                        tree.geometries.take(p)
                        for p in tree.query(poly)
                        if tree.geometries.take(p).intersects(poly)
                    ]
                    merged = unary_union(inter)
                    new_coords.append(merged)

                coords = unary_union(new_coords)
                merged_polys.append(
                    validate_and_simplify(coords, buffer=1.0, simplify=True, level=0.1)
                )
            else:
                merged_polys.append(
                    validate_and_simplify(geo, buffer=1.0, simplify=True, level=0.1)
                )

        self.annots = (
            gpd.GeoDataFrame({"geometry": merged_polys, "class_name": classes})
            .explode(index_parts=False)
            .reset_index(drop=True)
        )
