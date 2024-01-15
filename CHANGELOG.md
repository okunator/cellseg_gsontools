
<a id='changelog-0.1.3'></a>
# 0.1.3 — 2023-12-22

## Breaking changes

- The SpatialContext API now takes params: `parallel` and `num_processes` in the class constructor instead of the `fit()`-method.

## Features

- Added support for new backends (python libraries) to compute the spatial joins in the `SpatialContext`-classes. This allows for more efficient computation of the spatial joins especially for very large tissue areas.
- Backends:
    - dask-geopandas
    - spatialpandas

- Add `h3` hexgrid support in grid fitting for spatial context classes

## Performance
- sjoin operations replaced with sindex ops. This makes subsetting 10-100x faster.

## Docs
New documentation website: https://okunator.github.io/cellseg_gsontools/

<a id='changelog-0.1.2'></a>
# 0.1.2 — 2023-11-01

## Deprecated

- the `Pipeline` class was removed.

## Features

- Add `local_distances` function to compute the mean/median/sum of the distances between nodes in local neighborhoods

- Add `weights2gdf` function to convert `libpysal.weights.W` into a `gpd.GeoDataFrame` with `geometry` column consisting of `shapely.geometry.LineStrings`
- Add grid fitting to `_SpatialContext` classes. Allows a grid of patches to be overlayed on top of different context areas. The patch sizes and strides can be user defined.
- Better looking links in `_SpatialContext.plot()`. Different link classes now represented with different colors.
- Add `grid_classify` method to classify grids based on heuristics.

## Performance
- Support for parallel spatial context fitting: `_SpatialContext.fit(parallel=True)`

## Fixes

- Drop duplicates in `context2gdf`-class method

- clarify the `apply_gdf` function api.
- clarify the `_SpatialContext` api.

<a id='changelog-0.1.1'></a>
# 0.1.1 — 2023-10-13

## Features

- Add join predicate param for `sjoin` operation in `get_objs_within` function

## Performance

- Optimize spatial subsetting operations by getting rid of redundant operations in spatial context classes.

## Perf

- Parallelize and memory optimize `AreaMerger`.

## Fixes

- Simplify `border_network` computation and return only the node-node links that go accross the border.
- Simplify weights plotting interfacee

- Fix bug in interface and roi network fitting.

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
