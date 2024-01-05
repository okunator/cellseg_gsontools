from pathlib import Path
from typing import Dict, Tuple, Union

from .save_utils import check_format, get_file_from_coords, get_xy_coords

__all__ = ["BaseGSONMerger"]


class BaseGSONMerger:
    def __init__(
        self, in_dir: Union[Path, str], tile_size: Tuple[int, int] = (1000, 1000)
    ) -> None:
        """Create a base class for geojson merger classes.

        Implements methods to access the adjascent tiles of a given tile.

        Parameters:
            in_dir (Union[Path, str]):
                Path to the directory containing the geojson files.
            tile_size (Tuple[int, int]):
                Height and width of the tile in pixels.
        """
        self.in_dir = Path(in_dir)
        self.tile_size = tile_size
        self.files = sorted(Path(in_dir).glob("*"))

        # Check that all the files are in the correct format
        if self.files:
            for f in self.files:
                check_format(f)
        else:
            raise ValueError(f"No files found in {in_dir}.")

    def _get_right_neighbor(self, fname: str) -> Path:
        x, y = get_xy_coords(fname)
        x += self.tile_size[0]
        return get_file_from_coords(self.files, x, y)

    def _get_left_neighbor(self, fname: str) -> Path:
        x, y = get_xy_coords(fname)
        x -= self.tile_size[0]
        return get_file_from_coords(self.files, x, y)

    def _get_bottom_neighbor(self, fname: str) -> Path:
        x, y = get_xy_coords(fname)
        y += self.tile_size[1]
        return get_file_from_coords(self.files, x, y)

    def _get_bottom_right_neighbor(self, fname: str) -> Path:
        x, y = get_xy_coords(fname)
        x += self.tile_size[0]
        y += self.tile_size[1]
        return get_file_from_coords(x, y)

    def _get_bottom_left_neighbor(self, fname: str) -> Path:
        x, y = get_xy_coords(fname)
        x -= self.tile_size[0]
        y += self.tile_size[1]
        return get_file_from_coords(self.files, x, y)

    def _get_top_neighbor(self, fname: str) -> Path:
        x, y = get_xy_coords(fname)
        y -= self.tile_size[1]
        return get_file_from_coords(self.files, x, y)

    def _get_top_right_neighbor(self, fname: str) -> Path:
        x, y = get_xy_coords(fname)
        x += self.tile_size[0]
        y -= self.tile_size[1]
        return get_file_from_coords(self.files, x, y)

    def _get_top_left_neighbor(self, fname: str) -> Path:
        x, y = get_xy_coords(fname)
        x -= self.tile_size[0]
        y -= self.tile_size[1]
        return get_file_from_coords(self.files, x, y)

    def _get_adjascent_tiles(self, fname: str) -> Dict[str, Path]:
        adj = {}
        adj["left"] = self._get_left_neighbor(fname)
        adj["right"] = self._get_right_neighbor(fname)
        adj["bottom"] = self._get_bottom_neighbor(fname)
        adj["top"] = self._get_top_neighbor(fname)

        return adj
