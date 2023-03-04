import geopandas as gpd


def set_uid(
    gdf: gpd.GeoDataFrame, id_col: str = "uid", drop: bool = False
) -> gpd.GeoDataFrame:
    """Set a unique identifier column to gdf.

    NOTE: by default sets a running index column to gdf as the uid.

    Parameters
    ----------
        gdf : gpd.GeoDataFrame
            Input Geodataframe
        id_col : str, default="uid"
            The name of the column that will be used or set to the id.
        drop : bool, default=True
            Drop the column after it is added to index.

    Returns
    -------
        gpd.GeoDataFrame:
            The inputr gdf with a "uid" column added to it
    """
    allowed = list(gdf.columns) + ["uid", "id"]
    if id_col not in allowed:
        raise ValueError(f"Illegal `id_col`. Got: {id_col}. Allowed: {allowed}.")

    gdf = gdf.copy()
    if id_col in ("uid", "id"):
        gdf[id_col] = range(1, len(gdf) + 1)

    gdf = gdf.set_index(id_col, drop=drop)

    return gdf
