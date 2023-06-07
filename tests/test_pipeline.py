from pathlib import Path
from typing import Union

import pandas as pd
import pytest

from cellseg_gsontools.geometry import shape_metric
from cellseg_gsontools.pipeline import Pipeline


class TestPipeline(Pipeline):
    def __init__(
        self,
        in_path_cells: Union[str, Path] = None,
        in_path_areas: Union[str, Path] = None,
        parallel_df: bool = True,
        parallel_sample: bool = False,
    ) -> None:
        """Create a cervix CIN2 WSI nalysis pipeline

        An example pipeline to summarize human-readable features from CIN2-graded
        WSI images.
        """
        super().__init__(in_path_cells, in_path_areas, parallel_df, parallel_sample)

    def pipeline(
        self,
        fn_cell_gdf: Path = None,
        fn_area_gdf: Path = None,
    ) -> pd.Series:
        """Pipeline for one set of area and cell annotations."""
        fn_cell_gdf = Path(fn_cell_gdf)
        fn_area_gdf = Path(fn_area_gdf)

        cell_gdf = self.read_input(fn_cell_gdf, preproc=True, qupath_format="old")
        # area_gdf = self.read_input(fn_area_gdf, preproc=False, qupath_format="old")

        cell_gdf = shape_metric(
            cell_gdf, ["area", "eccentricity"], parallel=self.parallel_df
        )
        return pd.Series(
            {
                "cell_area_mean": cell_gdf["area"].mean(),
                "cell_ecc_mean": cell_gdf["eccentricity"].mean(),
            }
        )


@pytest.mark.parametrize("parallel_df", [True, False])
@pytest.mark.parametrize("parallel_sample", [True, False])
def test_pipeline(pipeline_areas, pipeline_cells, parallel_df, parallel_sample):
    pipe = TestPipeline(
        pipeline_cells,
        pipeline_areas,
        parallel_df=parallel_df,
        parallel_sample=parallel_sample,
    )
    pipe(pooltype="thread", maptype="uimap")
