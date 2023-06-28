<a id='changelog-0.1.0'></a>
# 0.1.0 — 2023-06-28

## First release

## Features
- Morphological, graph, and diversity feature extraction methods.
- Parallelized dataframe operations via `pandarallel`.
- Merging of adjacsent geojson annotations via the `_BaseMerger` classes.
- Spatial context interface with `_SpatialContext`-classes to subset regions of the tissues with sophistication.
- Summary features over the whole data via `_Summary` classes. Run summaries of the extracted features.
- Simple pipeline interface with `Pipeline`-class to run analysis in parallel over samples.
