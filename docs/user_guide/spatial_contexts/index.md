# Spatial Context Classes

Usually tissue slides contain several interesting areas or regions of interest (ROI).
However, at a whole-slide level, there are usually even more regions that are not of interest at all. Typically, the interesting regions are some tissue specific locations on the slide, like the tumor or stroma, or the interface between the tumor and stroma. If a cell appears in an interesting region, it is likely an interesting one and we want to be able to efficiently subset only interesting cells.

**Bring in the spatial context classes!**

`cellseg_gsontools` provides tools to extract and analyze cells from different spatial contexts with the spatial context classes. The spatial context classes can be used to categorize cells based on their spatial context. For example, a lymphocyte within a tumor can be considered as a tumor infiltrating lymphocyte (TIL) whereas a lymphocyte cell in the stroma can be considered as a stromal lymphocyte. Although, both are lymphocytes, they appear in a different spatial context and thus they should be treated differently. These classes provide tools to focus only on the cells that are of interest in a specific spatial context.

All the context-classes include the following methods:

- `.fit()` - fits the ROIs
- `.plot()` - plots the ROIs
- `.context2gdf(key="some_context")` - converts the distinct ROIs (if there are many) in
  to one single `gpd.GeoDataFrame`.
- `.context2weights(key="some_network")` - converts the distinct spatial
  weights objects of the ROIs into one `libpysal` graph.

When the spatial context classes are fit, the following is done:

- Extraction of the unique ROIs of a given tissue type or types such as stroma, or tumor etc.
- Extraction of cells from within the ROIs.
- Fitting of a graph on the cells within each ROI so that the cell neighborhoods can be investigated.
- Fitting of a grid on the ROIs so that different density heuristics can be computed on the ROIs. For example, the immune cell density can be computed on the ROIs with by applying the grid to the ROIs.

The resulting ROIs are saved in a class attribute called `.context`. This is a nested dictionary containg each unique ROI and the associated cells, graph and grid.
