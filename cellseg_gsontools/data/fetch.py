from pathlib import Path

BASE_PATH = Path(__file__).parent.resolve()


def _load(f):
    """Load a gdf file located in the data directory.

    Parameters
    ----------
    f : string
        File name.

    Returns
    -------
    gpd.GeoDataFrame
        A gdf loaded from f.
    """
    from cellseg_gsontools.utils import read_gdf

    return read_gdf(f)


def cervix_tissue():
    """A GeoDataframe containing the cervix tissue areas."""
    return _load(BASE_PATH / "cervix_tissue.feather")


def cervix_cells():
    """A GeoDataframe containing the cells of the cervix tissue."""
    return _load(BASE_PATH / "cervix_cells.feather")


def gland_cells():
    """A GeoDataframe containing the cells of the cervix tissue."""
    return _load(BASE_PATH / "gland_cells.feather")


def gland_tissue():
    """A GeoDataframe containing the cells of the cervix tissue."""
    return _load(BASE_PATH / "gland_areas.feather")


def tumor_stroma_intreface_cells():
    """A GeoDataframe containing the cells of the cervix tissue."""
    return _load(BASE_PATH / "ts_iface_cells.feather")


def tumor_stroma_intreface_tissue():
    """A GeoDataframe containing the cells of the cervix tissue."""
    return _load(BASE_PATH / "ts_iface_areas.feather")


def tissue_merge_dir():
    """Return the path to the directory containing the adjascent tissue patches."""
    return BASE_PATH / "merge_data" / "area"


def cell_merge_dir():
    """Return the path to the directory containing the adjascent cell patches."""
    return BASE_PATH / "merge_data" / "cell"
