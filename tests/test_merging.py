from cellseg_gsontools.merging import AreaMerger, CellMerger


def test_cell_merging(merge_data_cell):
    merger = CellMerger(merge_data_cell, tile_size=(1000, 1000))
    merger.merge_dir(verbose=False)


def test_area_merging(merge_data_area):
    merger = AreaMerger(merge_data_area)
    merger.merge_dir(verbose=False)
