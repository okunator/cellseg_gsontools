
<a id='changelog-0.1.0-alpha.2'></a>
# 0.1.0-alpha.2 — 2023-09-15

## Fixes

- Set crs in `_SpatialContext` to avoid warnings

- Unify the `local_character` and `local_diversity` function api.

## Features

- Add option to add multiple columns to `local_diversity` and `local_character`
- add `is_categorical` helper func
- Add hbdscan clustering method.
- Add option to not fit graphs in `.fit()` method of `_SpatialContext`-classes

## Chore
- add scikit-learn (1.3.0) dependency
- Update to latest geopandas (0.13) and shapely (> 2.0)

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
