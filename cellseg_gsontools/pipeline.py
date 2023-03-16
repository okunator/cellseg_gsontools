from abc import ABC, abstractmethod
from functools import partial
from pathlib import Path
from typing import Any, List, Tuple, Union

import geopandas as gpd
import pandas as pd

from cellseg_gsontools.multiproc import run_pool
from cellseg_gsontools.utils import pre_proc_gdf, read_gdf


class Pipeline(ABC):
    def __init__(
        self,
        in_path_cells: Union[str, Path] = None,
        in_path_areas: Union[str, Path] = None,
        parallel_df: bool = True,
        parallel_sample: bool = False,
    ) -> None:
        """Create a base abstract class for any pipeline object.

        Parameters
        ----------
            in_path_cells : str or Path
                The path to the folder containing all the input cell geodata files
                Allowed formats: .json, .feather, .parquet
            in_path_areas : str or Path
                The path to the folder containing all the input area geodata files
                Allowed formats: .json, .feather, .parquet
            parallel_df : bool, default=True
                Flag, whether to parallelize the pipeline over the dataframe rows
                via pandarallel.
            parallel_sample : bool, default=False
                Flag, whether to parallelize the pipeline over the input files via
                multiprocessing.

        Returns
        -------
            pd.DataFrame:
                A summary table of the slides containing the features of interest
                The features of interest are computed for every geodata file.
        """
        if in_path_areas is None and in_path_cells is None:
            raise ValueError(
                "Both `in_path_cells` & `in_path_areas` can't be None simultaneously."
            )

        self.in_path_cells = in_path_cells
        self.in_files_cells = None
        if self.in_path_cells is not None:
            self.in_files_cells = sorted(Path(in_path_cells).glob("*"))

        self.in_path_areas = in_path_areas
        self.in_files_areas = None
        if self.in_path_areas is not None:
            self.in_files_areas = sorted(Path(in_path_areas).glob("*"))

        self.parallel_df = parallel_df
        self.parallel_sample = parallel_sample

    @staticmethod
    def read_input(fname: str, preproc: bool = False) -> gpd.GeoDataFrame:
        """Read input geodataframe and filter out ."""
        gdf = read_gdf(fname)
        if preproc:
            gdf = pre_proc_gdf(gdf)

        return gdf

    def _pipe_unpack(self, args: List[Tuple[Any, ...]]) -> Any:
        """Uunpack tuple args to pipeline."""
        return self.pipeline(*args)

    @abstractmethod
    def pipeline(
        self, fn_cell_gdf: Path = None, fn_area_gdf: Path = None, parallel: bool = True
    ) -> pd.Series:
        """Pipeline method to be overridden.

        This method is the pipeline for one sample.
        """
        raise NotImplementedError(
            "`pipeline`-method not implemented. Every object inheriting `Pipeline` has"
            " to implement and override the `pipeline` method."
        )

    def __call__(self) -> pd.DataFrame:
        """Run the pipeline.

        Returns
        -------
            pd.DataFrame or None
        """
        if not self.parallel_sample:
            res = []
            if self.in_path_cells is None and self.in_path_areas is not None:
                for fn in self.in_files_areas:
                    res.append(self.pipeline(fn_aree_gdf=fn, parallel=self.parallel_df))
            elif self.in_path_cells is not None and self.in_path_areas is None:
                for fn in self.in_files_cells:
                    res.append(
                        self.pipeline(fn_cells_gdf=fn, parallel=self.parallel_df)
                    )
            elif self.in_path_cells is not None and self.in_path_areas is not None:
                for fn1, fn2 in zip(self.in_files_cells, self.in_files_areas):
                    res.append(self.pipeline(fn1, fn2, parallel=self.parallel_df))
            else:
                raise ValueError(
                    "The pipeline method has to take in one of the arguments "
                    "with names `fn_cell_gdf`, or `fn_area_gdf`. Override the method "
                    "such that these argument names are in the function parameters."
                )
        else:
            if self.in_path_cells is None and self.in_path_areas is not None:
                self.pipeline = partial(self.pipeline, parallel=False, fn_cell_gdf=None)
                args = self.in_files_areas
            elif self.in_path_cells is not None and self.in_path_areas is None:
                self.pipeline = partial(self.pipeline, parallel=False, fn_area_gdf=None)
                args = self.in_files_cells
            elif self.in_path_cells is not None and self.in_path_areas is not None:
                self.pipeline = partial(self.pipeline, parallel=False)
                args = list(zip(self.in_files_cells, self.in_files_areas))
            else:
                raise ValueError(
                    "The pipeline method has to take in one of the arguments "
                    "with names `fn_cell_gdf`, or `fn_area_gdf`. Override the method "
                    "such that these argument names are in the function parameters."
                )

            res = run_pool(self._pipe_unpack, args=args)

        return res
