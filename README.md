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

For latest pre-release, use
```shell
pip install cellseg-gsontools==0.1.0a2
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


**WithinContext**

Extracts cells from the `cell_gdf` within areas in `area_gdf`. Call `context2gdf`-method to retrieve the cells in a gdf. The different context that can be accessed with the `WithinContext`-class are `["roi_area", "roi_cells", "roi_network"]`.

```python
from cellseg_gsontools.spatial_context import WithinContext

within_context = WithinContext(
    area_gdf = area_gdf,
    cell_gdf = cell_gdf,
    label = "area_cin", # Extract the cells that are within tissue areas of this type
    min_area_size = 100000.0 # discard areas smaller than this
)

within_context.fit()
within_context.context2gdf("roi_cells")
```

**InterfaceContext**

Returns border region between two types of area given. The different context that can be accessed with the `InterfaceContext`-class are:  `["roi_area", "roi_cells", "roi_network", "interface_cells", "interface_area", "interface_network", "border_network", "full_network"]`.

```python
from cellseg_gsontools.spatial_context import InterfaceContext

interface_context = InterfaceContext(
    area_gdf = area_gdf,
    cell_gdf = cell_gdf,
    label1 = "area_cin",
    label2 = "areastroma",
    min_area_size = 100000.0 # discard areas smaller than this
)
interface_context.fit()
interface_context.plot(key = "interface_area")
```
![border_network.png](/images/border_network.png)

Here we pick the border area between the neoplastic lesion and the stroma to study for example the immune cells on the border.

**PointClusterContext**

Uses a given clustering algorithm to cluster cells of the given type. This can help to extract spatial clusters of cells.

```python
from cellseg_gsontools.spatial_context import PointClusterContext

cluster_context = PointClusterContext(
    cell_gdf = cell_gdf,
    label = "inflammatory", # cells of this type will be clustered
    cluster_method = "dbscan",
    min_area_size = 5000.0, # dbscan param
    min_samples = 70 # dbscan param
)

cluster_context.fit()
cluster_context.plot_weights("roi_network")
```
![cluster_network.png](/images/inf_network.png)

Here we clustered the immune cells on the slide and fitted a network on the cells that were within the alpha shape of the cluster.

### Summary

Summarize cells, areas, contexts, and intermediates of a slide into a tabular format for further analysis. `Summarise`-method must be called before using summary. The summaries can be grouped by any metadata or annotation column in the gdf. You can also use `filter_pattern` argument to choose the statistics (`mean`, `count` etc.) or groups used in summary output.

**InstanceSummary**

Easy way to calculate nuclei and area `metrics` for different classes of cells over the neoplastic areas of the tissue.

```python
neoplastic_areas = within_context.context2gdf("roi_cells")
spatial_weights =  within_context.context2weights("interface_cells", threshold=75.0)

lesion_summary = InstanceSummary(
    neoplastic_areas,
    metrics = ["area"],
    groups = ["class_name"],
    spatial_weights = spatial_weights,
    prefix = "lesion-cells-"
)
lesion_summary.summarize()
```

|                                         | **sample_cells** |
|----------------------------------------:|-----------------:|
|     **lesion-cells-inflammatory-count** |           118.00 |
|       **lesion-cells-neoplastic-count** |          4536.00 |
|            **lesion-cells-total-count** |          4787.00 |
| **lesion-cells-inflammatory-area-mean** |           241.17 |
|   **lesion-cells-neoplastic-area-mean** |           532.79 |


**SemanticSummary**

Summarizes tissues areas. Here we summarize the areas of immune clusters in the while tissue.

```python
immune_cluster_areas = cluster_context.context2gdf("roi_area")
immune_areas = SemanticSummary(
    immune_cluster_areas,
    metrics = ["area"],
    prefix = "immune-clusters-"
)
immune_areas.summarize()
```
|                                                 | **sample_cells** |
|------------------------------------------------:|-----------------:|
| **immune-clusters-immune-clusters-total-count** |            42.00 |
|                   **immune-clusters-area-mean** |       2558514.25 |


**SpatialWeightSummary**

Summarizes cell networks by counting edges between neighboring cells. Here we compute the cell-cell connections over the tumor-stroma interface. The cell-cell connections are defined by the spatial weights graph.

```python
interface_summary = SpatialWeightSummary(
    iface_context.merge_weights("border_network"),
    iface_context.cell_gdf,
    classes= ["inflammatory", "neoplastic"],
    prefix = "interface-"
)
interface_summary.summarize()
```
|                                         | **sample_cells** |
|----------------------------------------:|-----------------:|
| **interface-inflammatory-inflammatory** |               19 |
|   **interface-inflammatory-neoplastic** |              153 |
|     **interface-neoplastic-neoplastic** |              363 |


**DistanceSummary**

Summarizes distances between different areas. For example how many immune clusters are close to a neoplastic lesion.

```python
immune_proximities = DistanceSummary(
    immune_cluster_areas,
    neoplastic_areas,
    groups = None,
    prefix = "icc-close2lesion-",
)
immune_proximities.summarize()
```
|                              | **sample_cells** |
|-----------------------------:|-----------------:|
| **icc-close2lesion-0-count** |               34 |
| **icc-close2lesion-1-count** |                8 |
