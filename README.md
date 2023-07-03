## Introduction

**Cellseg_gsontools** is a toolset to analyse and summarize cell and tissue segmentations into interpretable features. Widely used metrics are provided out of the box but the pipeline infrastructure is amenable to self-defined functions to suit specific needs.  

## Installation

```shell
pip install cellseg-gsontools
```

## Usage

1. Create a **pipeline-class**

* **Context-classes** help handling and combining cell and tissue segmentations. Classes include algorithms and methods e.g. clustering.

* **Summary-classes** reduce context objects into summarised tabular form.

2. Run the **pipeline** on segmented cell and area gson-files.


## Code examples

### Geometry

Geometrical features of nuclei provided are listed in the example code below. They can be computed for single polygons directly or for a whole dataframe using the summary-object showcased later.

```python
from from cellseg_gsontools.geometry import shape_metric

shape_metric(
    gdf,
    metrics = [
     "area",
     "major_axis_len",
     "circularity",
     "eccentricity",
     "squareness"
    ]
)
```
![table of features](/images/table1.png)


### Entropy and diversity

Entropy can be calculated in any summary-object by passing one of `["simpson_index", "shannon_index", "gini_index", "theil_index"]` as a metric. Spatial_weights must be passed in the call.

**Local-diversity** calculates a feature's heterogeneity in a cell's neighborhood.

```python
from cellseg_gsontools.diversity import local_diversity
from libpysal.weights import DistanceBand

weights = DistanceBand.from_dataframe(gdf, threshold=55.0, alpha=-1.0)
local_diversity(
     gdf,
     spatial_weights = weights,
     val_col = "area",              #columns for which to compute the metric
     metrics = ["simpson_index"]
)
```

![table of features](/images/table2.png)

### Spatial-context

Combine cell-segmentation with area-segmentation for spatial information. `Fit`-method must be called before using context.

**WithinContext**

Extracts cells from the `cell_gdf` within areas in `area_gdf`. Call `context2gdf`-method to retrieve the cells in a gdf. `context2weights` can be used to fit a graphical network on the cells.

```python
from cellseg_gsontools.summary import (
    InstanceSummary,
    SemanticSummary,
    DistanceSummary
)

within_context = WithinContext(
    area_gdf = area_gdf,
    cell_gdf = cell_gdf,
    label = "area_cin",
    min_area_size = 100000.0
)

within_context.fit()
within_context.context2gdf("roi_cells")
```

**InterfaceContext**

Returns border region between two types of area given.

```python
interface_context = InterfaceContext(
    area_gdf = area_gdf,
    cell_gdf = cell_gdf,
    label1 = "area_cin",
    label2 = "areastroma",
    min_area_size = 100000.0
)
interface_context.fit()
interface_context.plot(key = "interface_area")
```
![border_network.png](/images/border_network.png)
Here we pick the border area between the neoplastic lesion and the stroma to study for example immune cells on the border.

**PointClusterContext**

Uses chosen clustering algorithm to cluster cells.

```python
cluster_context = PointClusterContext(
    cell_gdf = cell_gdf,
    label = "inflammatory",
    cluster_method = "dbscan",
    min_area_size = 5000.0,
    min_samples = 70
)

cluster_context.fit()
cluster_context.context2weights("roi_cells")
cluster_context.plot_weights("roi_network")
```
![cluster_network.png](/images/inf_network.png)
Here we clustered immune cells on the slide and fitted a network on a cluster.

### Summary

Summarize cells, areas, contexts, and intermediates into a tabular format for further analysis. `Summarise`-method must be called before using summary. Use `filter_pattern` argument to choose groups used in summary.

 **InstanceSummary**
 
Easy way to calculate nuclei and area `metrics` for `groups`
 
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
![table.png](/images/table3.png)


**SemanticSummary**

Summarizes tissues areas.

```python
immune_cluster_areas = cluster_context.context2gdf("roi_area")
immune_areas = SemanticSummary(
    immune_cluster_areas,
    metrics = ["area"],
    prefix = "immune-clusters-"
)
immune_areas.summarize()
```
![table.png](/images/table4.png)


**SpatialWeightSummary**

Summarizes cell networks.

```python
interface_summary = SpatialWeightSummary(
    iface_context.merge_weights("border_network"),
    iface_context.cell_gdf,
    classes= ["inflammatory", "neoplastic"],
    prefix = "interface-"
)
interface_summary.summarize()
```

**DistanceSummary**

Summarizes distances between contexts.

```python
immune_proximities = DistanceSummary(
    immune_cluster_areas,
    neoplastic_areas,
    groups = None,
    prefix = "icc-close2lesion-",
)
immune_proximities.summarize()
```
![table.png](/images/table5.png)


### Pipeline

Infrastructure for bulk analysis of gson-files. Computation is defined in the `pipeline`-method.

```python
from cellseg_gsontools.pipeline import Pipeline
from pathlib import Path
from typing import Union
import pandas as pd

class ExamplePipeline(Pipeline):
    def __init__(
        self,
        in_path_cells: Union[str, Path] = None,
        in_path_areas: Union[str, Path] = None,
        parallel_df: bool = True,
        parallel_sample: bool = False
    ) -> None:

    super().__init__(in_path_cells, in_path_areas, parallel_df, parallel_sample)
        
    def pipeline(
        self,
        fn_cell_gdf: Path = None,
        fn_area_gdf: Path = None,
    ) -> None:

        cell_gdf = self.read_input(fn_cell_gdf, preproc=True, qupath_format="old")
        cell_gdf = set_uid(cell_gdf)
        area_gdf = self.read_input(fn_area_gdf, preproc=False, qupath_format="old")

        # Define the neoplastic lesion as context
        within_context = WithinContext(
            area_gdf = area_gdf,
            cell_gdf = cell_gdf,
            label = "area_cin",
            min_area_size = 100000.0
        )
        within_context.fit()

        # Retrieve geometrical metrics for cells inside the context
        neoplastic_areas = within_context.context2gdf("roi_cells")
        lesion_summary = InstanceSummary(
            neoplastic_areas,
            metrics = ["area"],
            groups = ["class_name"],
            prefix = "lesion-cells-"
        )

        # Filter everything but neoplastic and inflammatory cells from the summary. Also include cell counts and metric quantiles
        fpat = "connective|glandular_epithel|dead|squamous_epithel|background|inflammatory"
        return pd.concat(
            [
                lesion_summary.summarize(filter_pattern = fpat,
                                         return_counts = True,
                                         return_quantiles = True)
            ]
        )

pipe = ExamplePipeline(
    "/path_to_data/cells",
    "/path_to_data/areas",
    parallel_df = False,
    parallel_sample = True
)

res = pipe()
res.to_csv("result.csv")

```
![table.png](/images/table6.png)


## Whats next

* More examples are provided in jupyter notebooks
* Documentation contains more detailed explanations for classes and functions

