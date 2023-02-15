import geopandas as gpd


def set_uid(gdf: gpd.GeoDataFrame, id_col: str = "uid") -> gpd.GeoDataFrame:
    """Set a unique identifier column to gdf.

    NOTE: by default sets a running index column to gdf as the uid.

    Parameters
    ----------
        gdf : gpd.GeoDataFrame
            Input Geodataframe

    Returns
    -------
        gpd.GeoDataFrame:
            The inputr gdf with a "uid" column added to it
    """
    allowed = list(gdf.columns) + ["uid"]
    if id_col not in allowed:
        raise ValueError(f"Illegal `id_col`. Got: {id_col}. Allowed: {allowed}.")

    if id_col == "uid":
        gdf[id_col] = range(1, len(gdf) + 1)
    else:
        gdf["uid"] = gdf[id_col]

    return gdf
