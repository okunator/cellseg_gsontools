from typing import Tuple, Union

import geopandas as gpd
import matplotlib.pyplot as plt
from libpysal.weights import DistanceBand
from tqdm import tqdm

from ._base import _SpatialContext

__all__ = ["WithinContext"]


class WithinContext(_SpatialContext):
    def __init__(
        self,
        area_gdf: gpd.GeoDataFrame,
        cell_gdf: gpd.GeoDataFrame,
        label: str,
        min_area_size: Union[float, str] = None,
        q: float = 25.0,
        verbose: bool = False,
        silence_warnings: bool = True,
    ) -> None:
        """Handle & extract cells from the `cell_gdf` within areas in `area_gdf`.

        I.e. Get objects within an area of type `label`.

        Parameters
        ----------
            area_gdf : gpd.GeoDataFrame
                A geo dataframe that contains large polygons enclosing smaller objs.
            cell_gdf : gpd.GeoDataFrame
                A geo dataframe that contains smaller cell objs enclosed in larger areas
            label : str
                The class name of the areas of interest. E.g. "cancer".
            min_area_size : float or str, optional
                The minimum area of the objects that are kept. All the objects in the
                `area_gdf` that are larger are kept. Can be either a float or one of:
                "mean", "median", "quantile"
            q : float, default=25.0
                The quantile. This is only used if `min_area_size = "quantile"`.
            verbose : bool, default=False
                Flag, whether to use tqdm pbar.
            silence_warnings : bool, default=True
                Flag, whether to silence all the warnings.

        Attributes
        ----------
            context : Dict[int, Dict[str, Union[gpd.GeoDataFrame, DistanceBand]]]
                A nested dict that contains dicts for each index of the distinct areas
                of type `label`. Each of the inner dicts contain the keys: 'roi_area',
                'roi_cells', 'roi_network'. The 'area' and 'cells' keys contain
                gpd.GeoDataFrame of the roi area and cells. The 'network' key contains
                a DistanceBand fitted to the cells inside the roi.

        Raises
        ------
            ValueError if `area_gdf` or `cell_gdf` don't contain 'class_name' column.

        Examples
        --------
        Define a within context and plot the cells inside a roi area.

        >>> from cellseg_gsontools.spatial_context import WithinContext

        >>> area_gdf = read_gdf("area.json")
        >>> cell_gdf = pre_proc_gdf(read_gdf("cells.json"))
        >>> within_context = WithinContext(
                area_gdf=area_gdf,
                cell_gdf=cell_gdf,
                label="area_cin",
                silence_warnings=True,
                verbose=True,
                min_area_size=100000.0
            )

        >>> ix = 1
        >>> ax = within_context.context_area.plot(figsize=(10, 10), alpha=0.5)
        >>> within_context.context[ix]["roi_cells"].plot(
                ax=ax, column="class_name", categorical=True, legend=True
            )
        """
        super().__init__(
            area_gdf=area_gdf,
            cell_gdf=cell_gdf,
            label=label,
            min_area_size=min_area_size,
            q=q,
            verbose=verbose,
            silence_warnings=silence_warnings,
        )

        # create context
        self.context = {}
        pbar = (
            tqdm(self.context_area.index, total=self.context_area.shape[0])
            if self.verbose
            else self.context_area.index
        )
        for ix in pbar:
            if self.verbose:
                pbar.set_description(f"Processing roi area: {ix}")
            self.context[ix] = {"roi_area": self.roi(ix)}
            self.context[ix]["roi_cells"] = self.roi_cells(ix)
            self.context[ix]["roi_network"] = self.cell_neighbors(ix)

    def cell_neighbors(
        self,
        ix: int,
        thresh: float = 75.0,
    ) -> DistanceBand:
        """Create a distance network of the cells.

        Parameters
        ----------
            ix : int
                The index of the ROI geo-object. Starts from one.
            thresh : float, default=75.0
                Distance threshold for the network.
        Returns
        -------
            DistanceBand:
                Either one distance network or all three possible distance networks.
        """
        cells = self.roi_cells(ix)
        if cells is None:
            return None

        w = DistanceBand.from_dataframe(
            cells, threshold=thresh, ids="uid", alpha=-1.0, silence_warnings=True
        )

        return w

    def plot(
        self,
        show_area: bool = True,
        show_cells: bool = True,
        show_legends: bool = True,
        color: str = None,
        figsize: Tuple[int, int] = (12, 12),
    ) -> None:
        """Plot the slide with areas, cells, and interface areas highlighted.

        Parameters
        ----------
            show_area : bool, default=True
                Flag, whether to include the tissue areas in the plot.
            show_cells : bool, default=True
                Flag, whether to include the cells in the plot.
            show_legends : bool, default=True
                Flag, whether to include legends for each in the plot.
            color : str, optional
                A color for the interfaces, Ignored if `show_legends=True`.
            figsize : Tuple[int, int], default=(12, 12)
                Size of the figure.

        Returns
        -------
            AxesSubplot
        """
        _, ax = plt.subplots(figsize=figsize)

        if show_area:
            ax = self.area_gdf.plot(
                ax=ax,
                column="class_name",
                categorical=True,
                legend=show_legends,
                alpha=0.1,
                legend_kwds={
                    "loc": "upper center",
                },
            )
            leg1 = ax.legend_

        if show_cells:
            ax = self.cell_gdf.plot(
                ax=ax,
                column="class_name",
                categorical=True,
                legend=show_legends,
                legend_kwds={
                    "loc": "upper right",
                },
            )
            leg2 = ax.legend_

        areas = self.context2gdf("roi_area")
        ax = areas.plot(
            ax=ax,
            color=color,
            column="label",
            alpha=0.5,
            legend=show_legends,
            categorical=True,
            legend_kwds={"loc": "upper left"},
        )
        if show_legends:
            if show_area:
                ax.add_artist(leg1)
            if show_cells:
                ax.add_artist(leg2)

        return ax
