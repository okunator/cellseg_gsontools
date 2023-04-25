from abc import ABC, abstractmethod
from typing import Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd

from cellseg_gsontools.diversity import DIVERSITY_LOOKUP
from cellseg_gsontools.geometry.shape_metrics import SHAPE_LOOKUP

METRICS = {**DIVERSITY_LOOKUP, **SHAPE_LOOKUP}


__all__ = ["Summary"]


class Summary(ABC):
    @abstractmethod
    def summarize(self) -> pd.Series:
        """Summarize method to be overridden.

        Takes in a gpd.GeoDataFrame & returns a pd.Series.
        """
        raise NotImplementedError(
            "`summarize`-method not implemented. Every object inheriting `Summary` has"
            " to implement and override the `summarize` method."
        )

    @staticmethod
    def gen_metric_summary(
        gdf: gpd.GeoDataFrame,
        metrics: Tuple[str, ...],
        groups: Optional[Tuple[str, ...]] = None,
        prefix: Optional[str] = None,
    ) -> pd.Series:
        """Generate a feature vector from the computed metrics objects in a gdf.

        Parameters
        ----------
            gdf : gpd.GeoDataFrame
                Input geo dataframe.
            metrics : Tuple[str, ...]
                A list of metrics that will be included in the feature vector. These
                metrics have to be found in the columns of the `gdf`
            groups : Tuple[str, ...], optional
                A list of catergorical groups. These group-names have to be found in the
                columns of the `gdf`. These groups are used in a groupby operation and
                the metric summaries are computed for each of the groups and sub-groups.
                Ignored if set to None.
            prefix : str, optional
                A prefix for the named indices.

        Raises
        ------
            ValueError: If more than 2 groups is given.

        Returns
        -------
            pd.Series:
                A series/named-vector of the computed metric summaries.
        """
        mets = list(metrics)

        if groups is not None:
            groups = list(groups)
            summary = gdf.groupby(groups)[mets].describe()
        else:
            summary = gdf[mets].describe()

        if groups is not None:
            if len(groups) == 1:
                sum_vec = Summary._gen_sum_vec_l1(summary, prefix)
            elif len(groups) == 2:
                sum_vec = Summary._gen_sum_vec_l2(summary, prefix)
            else:
                raise ValueError(
                    "too many values in `groups`. Max 2 groups allowed for now. "
                    f"Got: {len(groups)}"
                )
        else:
            sum_vec = Summary._gen_sum_vec_l0(summary, prefix)

        # de-duplicate counts
        counts = sum_vec[sum_vec.index.str.contains("count")]
        counts = counts[counts.index.str.contains(metrics[0])]
        counts.index = counts.index.str.replace(f"{metrics[0]}-", "")

        prefix = prefix if prefix is not None else ""
        if groups is not None:
            counts = pd.concat(
                [counts, pd.Series(counts.sum(), index=[f"{prefix}total-count"])]
            )
        else:
            counts.index = counts.index.str.replace("count", f"{prefix}total-count")

        sum_vec = sum_vec[~sum_vec.index.str.contains("count")]
        sum_vec = pd.concat([counts, np.round(sum_vec, 3)])

        return sum_vec

    @staticmethod
    def get_counts(
        summary: pd.Series,
        thresh: int,
        rule: str = "above",
        prefix: str = None,
        pat: str = None,
    ) -> pd.Series:
        """Get the counts of objects that are under or above some threshold.

        Parameters
        ----------
            summary : pd.Series
                A summary vector with named indices.
            thresh : int
                The threshold value.
            rule : str, default="above"
                One of: "above", "under", "equal"
            prefix : str, optional
                Prefix to the result series indices.
            pat : str, optional
                A pattern to filter out values based whose index does not contain this
                pattern.

        Returns
        -------
            pd.Series:
                A count vector of negative and positive cases.
        """
        allowed = ("above", "under", "equal")
        if rule not in allowed:
            raise ValueError(f"Illegal rule. Got: {rule}. Allowed: {allowed}")

        if pat is not None:
            summary = summary[summary.index.str.contains(pat)]

        if rule == "above":
            vals = summary > thresh
        elif rule == "under":
            vals = summary < thresh
        if rule == "equal":
            vals = summary == thresh

        count_thresh = pd.value_counts(vals).astype(int)
        if prefix is not None:
            count_thresh.index = prefix + count_thresh.index.astype(str)

        return count_thresh

    @staticmethod
    def _gen_sum_vec(
        s: pd.Series,
        metric: str,
        level_values: Tuple[str, ...],
        prefix: str = None,
    ) -> pd.Series:
        """Generate a summary series with easy to understand names."""
        s = s.copy()
        if level_values is not None:
            values = "-".join([str(v) for v in level_values]) + "-"
        else:
            values = ""

        if prefix is None:
            prefix = ""

        s.index = prefix + values + f"{metric}-" + s.index.astype(str)

        return s

    @staticmethod
    def _gen_sum_vec_l2(summary: pd.DataFrame, prefix: str = None) -> pd.Series:
        """Generate a summary vector from all the metrics. Two groups."""
        all_feats = []
        for l1, l2 in summary.index:
            s = summary.loc[l1, l2]
            metrics = tuple(s.index.unique(0))
            for met in metrics:
                all_feats.append(Summary._gen_sum_vec(s[met], met, (l1, l2), prefix))

        return pd.concat(all_feats)

    @staticmethod
    def _gen_sum_vec_l1(summary: pd.DataFrame, prefix: str = None) -> pd.Series:
        """Generate a summary vector from all the metrics. One group."""
        all_feats = []
        for l1 in summary.index:
            s = summary.loc[l1]
            metrics = tuple(s.index.unique(0))
            for met in metrics:
                all_feats.append(Summary._gen_sum_vec(s[met], met, (l1,), prefix))

        return pd.concat(all_feats)

    @staticmethod
    def _gen_sum_vec_l0(summary: pd.DataFrame, prefix: str = None) -> pd.Series:
        """Generate a summary vector from all the metrics. No grouping."""
        all_feats = []
        metrics = tuple(summary.columns)
        for met in metrics:
            all_feats.append(Summary._gen_sum_vec(summary[met], met, None, prefix))
        return pd.concat(all_feats)
