from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List, Tuple, Union

import geopandas as gpd
import pandas as pd

from cellseg_gsontools.multiproc import run_pool
from cellseg_gsontools.utils import pre_proc_gdf, read_gdf

__all__ = ["Pipeline"]


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
                "If you wish to inherit `Pipeline` class, the class constructor must "
                "have atleast one of the parameters: `in_path_cells` or `in_path_areas`"
                ". These indicate the paths to the folders that contain geojson files. "
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
    def read_input(fname: str, preproc: bool = False, **kwargs) -> gpd.GeoDataFrame:
        """Read input geodataframe and filter out ."""
        gdf = read_gdf(fname, **kwargs)
        if preproc:
            gdf = pre_proc_gdf(gdf)

        return gdf

    def _pipe_unpack(self, args: List[Tuple[Any, ...]]) -> Any:
        """Unpack tuple args to pipeline."""
        if isinstance(args, tuple):
            return self.pipeline(*args)
        else:
            return self.pipeline(args)

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

    def collect(self, res_list: List[Any]) -> pd.DataFrame:
        """Collect the individual results and convert them to a dataframe."""
        if isinstance(res_list[0], pd.Series):
            res_df = pd.DataFrame(res_list)

        return res_df

    def __call__(
        self,
        pbar: bool = True,
        n_jobs: int = -1,
        pooltype: str = "thread",
        maptype: str = "imap",
    ) -> pd.DataFrame:
        """Run the pipeline.

        Parameters
        ----------
            verbose : bool, default=True
                Flag, whether to print progress bar.
            n_jobs : int, default=-1
                Number of jobs to run in parallel.
            pooltype : str, default='thread'
                The type of pool to use for multiprocessing.
                Allowed values: 'thread', 'process', 'serial'
            maptype : str, default='imap'
                The type of map to use for multiprocessing.
                Allowed values: 'imap', 'map', 'uimap', 'amap'

        Returns
        -------
            pd.DataFrame:
                Slide summaries in a df.
        """
        if not self.parallel_sample:
            res = []
            index = []
            if self.in_path_cells is None and self.in_path_areas is not None:
                for fn in self.in_files_areas:
                    res.append(self.pipeline(fn_area_gdf=fn))
                    index.append(fn.with_suffix("").name)
            elif self.in_path_cells is not None and self.in_path_areas is None:
                for fn in self.in_files_cells:
                    res.append(self.pipeline(fn_cell_gdf=fn))
                    index.append(fn.with_suffix("").name)
            elif self.in_path_cells is not None and self.in_path_areas is not None:
                for fn1, fn2 in zip(self.in_files_cells, self.in_files_areas):
                    res.append(self.pipeline(fn1, fn2))
                    index.append(fn1.with_suffix("").name)
            else:
                raise ValueError(
                    "The pipeline method has to take in one of the arguments "
                    "with names `fn_cell_gdf`, or `fn_area_gdf`. Override the method "
                    "such that these argument names are in the function parameters."
                )
        else:
            if self.in_path_cells is None and self.in_path_areas is not None:
                args = self.in_files_areas
            elif self.in_path_cells is not None and self.in_path_areas is None:
                args = self.in_files_cells
            elif self.in_path_cells is not None and self.in_path_areas is not None:
                args = list(zip(self.in_files_cells, self.in_files_areas))
            else:
                raise ValueError(
                    "The pipeline method has to take in one of the arguments "
                    "with names `fn_cell_gdf`, or `fn_area_gdf`. Override the method "
                    "such that these argument names are in the function parameters."
                )
            if isinstance(args[0], tuple):
                index = [fn1.with_suffix("").name for fn1, _ in args]
            else:
                index = [fn.with_suffix("").name for fn in args]

            res = run_pool(
                self._pipe_unpack,
                args=args,
                pooltype=pooltype,
                maptype=maptype,
                n_jobs=n_jobs,
                pbar=pbar,
            )

        res = self.collect(res)
        res.index = index

        return res
