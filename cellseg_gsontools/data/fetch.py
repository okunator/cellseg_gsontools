from pathlib import Path

BASE_PATH = Path(__file__).parent.resolve()


def _load(f):
    """Load a gdf file located in the data directory.

    Parameters:
        f (str):
            File name.

    Returns:
        gpd.GeoDataFrame:
            A gdf loaded from f.
    """
    from cellseg_gsontools.utils import read_gdf

    return read_gdf(f)


def cervix_tissue():
    """A GeoDataframe containing cervical tissue areas.

    Examples:
        >>> from cellseg_gsontools.data import cervix_tissue
        >>> cervix_tissue().head(3)
                type                                           geometry  class_name
        uid
        1    Feature  POLYGON ((1852.953 51003.603, 1853.023 51009.1...  areastroma
        2    Feature  POLYGON ((4122.334 48001.899, 4122.994 48014.8...   areagland
        3    Feature  POLYGON ((3075.002 48189.068, 3075.001 48218.8...   areagland
    """
    return _load(BASE_PATH / "cervix_tissue.feather")


def cervix_cells():
    """A GeoDataframe containing cells of the cervical tissue.

    Examples:
        >>> from cellseg_gsontools.data import cervix_cells
        >>> cervix_cells().head(3)
                type                                           geometry    class_name
        uid
        1    Feature  POLYGON ((-10.988 48446.005, -10.988 48453.996...  inflammatory
        2    Feature  POLYGON ((-20.988 48477.996, -19.990 48479.993...    connective
        3    Feature  POLYGON ((-14.988 48767.995, -11.993 48770.990...  inflammatory
    """
    return _load(BASE_PATH / "cervix_cells.feather")


def gland_cells():
    """A GeoDataframe containing cells of cervical gland tissue.

    Examples:
        >>> from cellseg_gsontools.data import gland_cells
        >>> gland_cells().plot()
        plt.Axes
    """
    return _load(BASE_PATH / "gland_cells.feather")


def gland_tissue():
    """A GeoDataframe containing cervical gland tissue.

    Examples:
        >>> from cellseg_gsontools.data import gland_tissue
        >>> gland_tissue().plot()
        plt.Axes
    """
    return _load(BASE_PATH / "gland_areas.feather")


def tumor_stroma_intreface_cells():
    """A GeoDataframe containing cells of a TS-interface.

    Examples:
        >>> from cellseg_gsontools.data import tumor_stroma_intreface_cells
        >>> tumor_stroma_intreface_cells().plot()
        plt.Axes
    """
    return _load(BASE_PATH / "ts_iface_cells.feather")


def tumor_stroma_intreface_tissue():
    """A GeoDataframe containing tissues of a TS-interface.

    Examples:
        >>> from cellseg_gsontools.data import tumor_stroma_intreface_tissue
        >>> tumor_stroma_intreface_tissue().plot()
        plt.Axes
    """
    return _load(BASE_PATH / "ts_iface_areas.feather")


def tissue_merge_dir():
    """Return the path to the directory containing the adjascent tissue patches."""
    return BASE_PATH / "merge_data" / "area"


def cell_merge_dir():
    """Return the path to the directory containing the adjascent cell patches."""
    return BASE_PATH / "merge_data" / "cell"
