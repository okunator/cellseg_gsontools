<div align="center">

# Cellseg_gsontools

[![Github Test](https://img.shields.io/github/actions/workflow/status/okunator/cellseg_gsontools/tests.yml?label=tests)](https://github.com/okunator/cellseg_models.pytorch/actions/workflows/tests.yml) [![Generic badge](https://img.shields.io/github/license/okunator/cellseg_gsontools
)](https://github.com/okunator/cellseg_gsontools/blob/master/LICENSE) [![Python - Version](https://img.shields.io/pypi/pyversions/cellseg_gsontools
)](https://www.python.org/) [![Package - Version](https://img.shields.io/pypi/v/cellseg_gsontools)
](https://pypi.org/project/cellseg-gsontools/)

Localized quantification of cell and tissue segmentation maps.

</div>

## Introduction

**Cellseg_gsontools** is a Python toolset designed to analyze and summarize large cell and tissue segmentation maps created from Whole Slide Images (WSI). It provides a range of metrics and algorithms out of the box, while also allowing users to define their own functions to meet specific needs. The library is built on top of [`geopandas`](https://geopandas.org/en/stable/index.html) and heavily utilizes the `GeoDataFrame` data structure or `gdf` for short. In other words, the library is built to process `geospatial` data with `GeoJSON`-interface. The library is synergetic with the [cellseg_models.pytorch](https://github.com/okunator/cellseg_models.pytorch) segmentation library which enables you to segment your WSI into `GeoJSON` format.

**NOTE** The toolset is still in alpha-phase and under constant development.

## Installation

```shell
pip install cellseg-gsontools
```

To add some extra capabilities, like, support for arrow files or abdscan clustering use:

```shell
pip install cellseg-gsontools[all]
```

## Usage

The idea of `cellseg_gsontools` is to provide an easy-to-use API to extract features from `GeoJSON`-formatted cell/nuclei/tissue segmentation maps. This can be done via different spatial-analysis methods including:

* Methods for computing morphological metrics.
* Methods for extracting neighborhoods metrics.
* Methods for computing diversity metrics.
* Subsetting cells with tissue areas for more localized feature extraction
* Spatial point clustering methods.
* Utilities for pretty visualization of the spatial data.

Specifically:
* **Functional API** helps to quickly compute object-level metrics in a `GeoDataFrame`.
* **Spatial Context classes** help handling and combining cell and tissue segmentations for more localized and spatially contextualized feature extraction. These classes include algorithms to subset different spatial contexts with methods like spatial joins, graph networks, and clustering.
* **Summary classes** can be used to reduce context objects into summarised tabular form, if you for some reason happen to be too lazy for `geopandas`-based data-wrangling. These classes include `InstanceSummary`, `DistanceSummary`, `SemanticSummary`, `SpatialWeightSummary`

**NOTE**: The input `GeoDataFrame`s always need to contain a column called `class_name` or otherwise nothing will work. The column should contain the class or category of the geo-object, e.g. the cell type, or tissue type. This restriction might loosen in the future.

## Code examples

### Function API

**Good to know:** These functions can be parallelized through [`pandarallel`](https://github.com/nalepae/pandarallel) by passing in the argument `parallel=True`.

#### Geometry

Geometrical (or sometimes morphological) features can be computed for individual nuclear-, cell-, or any type of polygon-objects over the whole `GeoDataFrame` (`gdf`). They can be also computed using the summary-object showcased later. These functions take in a `gdf` and a list of geometrical features and return the same `gdf` with the computed features for each object.

```python
from cellseg_gsontools.utils import read_gdf
from cellseg_gsontools.geometry import shape_metric

path = "/path/to/cells.json"
gdf = read_gdf(path)
gdf = set_uid(gdf, id_col="uid") # set a running index id column 'uid'

shape_metric(
    gdf,
    metrics = [
        "area",
        "major_axis_len",
        "circularity",
        "eccentricity",
        "squareness"
    ],
    parallel=True
)
```
| **uid** | **geometry**       | **class_name** |   **area** | **major_axis_len** | **circularity** | **eccentricity** | **squareness** |
|--------:|-------------------:|---------------:|-----------:|-------------------:|----------------:|-----------------:|---------------:|
|       1 | Polygon((x, y)...) |   inflammatory | 161.314012 |          15.390584 |        0.942791 |         0.834633 |       1.198831 |
|       2 | Polygon((x, y)...) |     connective | 401.877306 |          26.137359 |        0.948243 |         0.783638 |       1.205882 |
|       3 | Polygon((x, y)...) |     connective | 406.584839 |          29.674783 |        0.877915 |         0.594909 |       1.117796 |
|       4 | Polygon((x, y)...) |     connective | 281.779998 |          24.017928 |        0.885816 |         0.617262 |       1.127856 |
|       5 | Polygon((x, y)...) |     connective | 257.550056 |          17.988285 |        0.891481 |         0.999339 |       1.131054 |
|     ... |                    |            ... |        ... |                ... |             ... |              ... |            ... |

#### Spatial Neighborhood and Neighborhood metrics

**Spatial neighborhoods** are sets of nodes in large graph networks. The nodes are cell or object-centroids and the links between a node to neighboring nodes define the immediate neighborhood of any object. To extract spatial neighborhoods, a graph needs to be fitted to a `gdf`. The `fit_graph`-method can be used to fit a graph network also known as `spatial weights` object in geospatial analysis terms. Allowed spatial weights fitting methods are `["delaunay", "knn", "distband", "relative_nhood"]`.

Now if you want to extract features from the immediate neighborhood of objects, you can use the `local_character`-function. It takes in a `gdf` and a spatial weights object. The function reduces the neighborhood values to either `mean`, `median` or `sum`. **NOTE** you can also input a list of columns to the function to compute reductions over many columns.

```python
from cellseg_gsontools.utils import read_gdf, set_uid
from cellseg_gsontools.graphs import fit_graph
from cellseg_gsontools.character import local_character
from cellseg_gsontools.geometry import shape_metric

path = "/path/to/cells.json"
gdf = read_gdf(path)
gdf = set_uid(gdf, id_col="uid") # set a running index id column 'uid'

# compute the eccentricity of the cells
gdf = shape_metric(gdf, metrics = ["eccentricity"], parallel=True)

# fit a spatial weights object
w = fit_graph(gdf, type="delaunay", thresh=150, id_col="uid")

# compute the mean eccentricity of each cell's neighborhood
local_character(
    gdf,
    spatial_weights=w,
    val_col="eccentricity",
    reductions=["mean"], # mean, median, sum,
    weight_by_area=True, # weight the values by the object areas
    parallel=True
)
```
| **uid** | **geometry**       | **class_name** |   **eccentricity** | **eccentricity_nhood_mean** |
|--------:|-------------------:|---------------:|-------------------:|----------------------------:|
|       1 | Polygon((x, y)...) |   inflammatory |           0.556404 |               0.090223      |
|       2 | Polygon((x, y)...) |     connective |           0.474120 |               0.121348      |
|       3 | Polygon((x, y)...) |     connective |           0.712500 |               0.128495      |
|       4 | Polygon((x, y)...) |     connective |           0.325301 |               0.671348      |
|       5 | Polygon((x, y)...) |     connective |           0.285714 |               0.000000      |
|     ... |                    |            ... |                ... |                    ...      |

#### Neighborhood Diversity

**Neighborhood-diversity** metrics calculate how diverse (or sometimes heterogenous) the immediate neighborhood of an object is. The diversity is computed with respect to a feature (e.g. nuclei area, eccentricity, or class name) of the neighboring objects. Note that the features can also be categorical, like the cell-type class etc. The `neighborhood diversities` can be computed with the `local_diversity`-function. The available diversity methods to compute are `["simpson_index", "shannon_index", "gini_index", "theil_index"]`. **NOTE:** you can also input a list of columns to the function to compute diversity metrics over many columns.

```python
from cellseg_gsontools.utils import read_gdf, set_uid
from cellseg_gsontools.graphs import fit_graph
from cellseg_gsontools.geometry import shape_metric
from cellseg_gsontools.diversity import local_diversity

path = "/path/to/cells.json"
gdf = read_gdf(path)
gdf = set_uid(gdf, id_col="cell_id") # set a running index id column 'cell_id'

# compute the eccentricity of the cells
gdf = shape_metric(gdf, metrics = ["area"], parallel=True)

# fit a spatial weights object
w = fit_graph(gdf, type="delaunay", thresh=150, id_col="cell_id")

# compute the heterogeneity of the neighborhood areas
local_diversity(
    gdf,
    spatial_weights=w,
    val_col="area",
    metrics=["simpson_index"],
)
```

| **uid** | **geometry**       | **class_name** |   **area** | **area_simpson_index** |
|--------:|-------------------:|---------------:|-----------:|-----------------------:|
|       1 | Polygon((x, y)...) |   inflammatory | 161.314012 |               0.000000 |
|       2 | Polygon((x, y)...) |     connective | 401.877306 |               0.000000 |
|       3 | Polygon((x, y)...) |     connective | 406.584839 |               0.444444 |
|       4 | Polygon((x, y)...) |     connective | 281.779998 |               0.500000 |
|       5 | Polygon((x, y)...) |     connective | 257.550056 |               0.000000 |
|     ... |                    |            ... |        ... |                    ... |

## Spatial Contexts

Spatial Context classes combine cell segmentation maps with tissue area segmentation maps which helps to extract cells under different spatial contexts. The specific Spatial Context classes are the `InterfaceContext` `WithinContext`, and `PointClusterContext`. All the context-classes include a `.fit()`-method that builds the context. The `.plot()`-method can be used to plot figures where the different spatial context areas are highlighted.

**Good to know:** The `.fit()`-method can be parallelized through pandarallel by passing in the argument `.fit(parallel=True)`.

**Context class methods**
- `.fit()` - fits the contexts
- `.plot()` - plots the contexts
- `.context2gdf(key="some_context")` - converts the distinct contexts in to one single `gdf`.
- `.context2weights(key="some_network")` - converts the distinct spatial weights objects of the context-class into one graph.

#### WithinContext

The `WithinContext` extracts the cells from the `cell_gdf` within the areas in `area_gdf` that have the types specified in the `labels` argument.

`context2gdf` accepts the following values for the `key`-argument:
- `'roi_area'` - returns the roi areas of type `labels`
- `'roi_cells'` - returns the cells inside the roi areas.
- `'roi_grid'` - returns the grids fitted on top of the roi areas of type `labels`.

`context2weights` accepts the following values for the `key`-argument
- `'roi_network'` - returns the network fitted on the `roi_cells`.

**Example**
Extract the cells within the tumor areas. and plot the tumor regions and grids fitted over of them.

```python
from cellseg_gsontools.spatial_context import WithinContext
from cellseg_gsontools.utils import read_gdf

area_gdf = read_gdf("area.feather")
cell_gdf = read_gdf("cells.feather")

within_context = WithinContext(
    area_gdf = area_gdf,
    cell_gdf = cell_gdf,
    labels = "area_cin", # Extract the cells that are within tissue areas of this type
    min_area_size = 100000.0 # discard areas smaller than this
)

within_context.fit(parallel=False, fit_graph=False)
within_context.plot(key="roi_area", grid_key="roi_grid")
```
Processing roi area: 3: 100%|██████████| 4/4 [00:17<00:00,  4.48s/it]
![border_network.png](/images/within_context.png)

#### InterfaceContext

The `InterfaceContext` returns the border regions between given areas. It works by buffering regions of type `top_labels` on top of regions of type `bottom_labels` and taking the intersection.

`context2gdf` accepts the following values for the `key`-argument:
- `'roi_area'` - returns the roi areas of type `top_labels`
- `'roi_cells'` - returns the cells inside the roi areas.
- `'roi_grid'` - returns the grids fitted on top of the roi areas of type `top_labels`.
- `'interface_area'` - returns the interface areas between `top_labels` and `bottom_labels` areas.
- `'interface_cells'` - returns the cells inside the interface areas.
- `'interface_grid'` - returns the grids fitted on top of the interface areas.

`context2weights` accepts the following values for the `key`-argument
- `'roi_network'` - returns the network fitted on the `roi_cells`.
- `'interface_network'` - returns the network fitted on top of the `interface_cells`
- `'border_network'` - returns the network fitted on top of the cells that cross the interface border

**Example**
Extract and plot the tumor-stroma interface and the grid fitted on top of it and the links of the `border_network`.

```python
from cellseg_gsontools.spatial_context import InterfaceContext
from cellseg_gsontools.utils import read_gdf

area_gdf = read_gdf("area.feather")
cell_gdf = read_gdf("cells.feather")

interface_context = InterfaceContext(
    area_gdf = area_gdf,
    cell_gdf = cell_gdf,
    top_labels = "area_cin",
    bottom_labels = "areastroma",
    buffer_dist = 250.0, # breadth of the interface (in pixels)
    min_area_size = 100000.0 # discard areas smaller than this
)
interface_context.fit(parallel=False, fit_grid=True)
interface_context.plot(key="interface_area", grid_key="interface_grid")
```
Processing interface area: 3: 100%|██████████| 4/4 [00:01<00:00,  2.76it/s]

![border_network.png](/images/interface_context.png)


**PointClusterContext**

The `PointClusterContext` clusters the cells of type `labels` from the `cell_gdf` and draws a border around the spatial point clusters.

`context2gdf` accepts the following values for the `key`-argument:
- `'roi_area'` - returns the clustered areas that contains clusters of cells of type `labels`
- `'roi_cells'` - returns the cells inside the clustered areas.
- `'roi_grid'` - returns the grids fitted on top of the clustered areas.

`context2weights` accepts the following values for the `key`-argument
- `'roi_network'` - returns the network fitted on the `roi_cells`.

**Example**
Cluster the immune cells and plot the cluster regions and the different links between the cells.

```python
from cellseg_gsontools.spatial_context import PointClusterContext
from cellseg_gsontools.utils import read_gdf

cell_gdf = read_gdf("cells.feather")

cluster_context = PointClusterContext(
    cell_gdf = cell_gdf,
    labels = "inflammatory", # cells of this type will be clustered
    cluster_method = "dbscan", # clustering algorithm. One of: dbscan, adbscan, hbdscan, optics
    min_area_size = 5000.0, # dbscan param
    min_samples = 70, # dbscan param
    graph_type="delaunay", # fit a delaunay graph over the clustered cellsm
    dist_thresh=75.0, # drop links over 75 pixels long
)

cluster_context.fit(parallel=False, fit_graph=True, fit_grid=False)
cluster_context.plot(key="roi_area", network_key="roi_network")
```
Processing roi area: 2: 100%|██████████| 3/3 [00:21<00:00,  7.14s/it]
![border_network.png](/images/cluster_context.png)
