"""
Shapefile (polygones + HAUTEUR) → carte de hauteur + masque de validité.
Adapté de pipeline_batiments_dagster/core/instances.py::rasterize_polygons.
"""
from __future__ import annotations
import numpy as np
import rasterio.features
import geopandas as gpd


def rasterize_hauteur(gdf: gpd.GeoDataFrame, transform, shape_hw: tuple[int, int]):
    """Peint HAUTEUR dans les pixels bâtiment. Petits polygones peints en
    premier pour ne pas être écrasés par un grand voisin qui chevauche."""
    height_map = np.zeros(shape_hw, dtype=np.float32)
    valid_mask = np.zeros(shape_hw, dtype=np.uint8)

    if gdf is None or len(gdf) == 0:
        return height_map, valid_mask

    gdf_clean = gdf[gdf.geometry.is_valid & (gdf["HAUTEUR"] > 0)]
    rows = sorted(
        zip(gdf_clean.geometry, gdf_clean["HAUTEUR"]),
        key=lambda item: item[0].area,
    )

    for geom, hauteur in rows:
        layer = rasterio.features.rasterize(
            [(geom, 1)], out_shape=shape_hw, transform=transform, fill=0, dtype="uint8",
        )
        unpainted = (layer > 0) & (valid_mask == 0)
        height_map[unpainted] = hauteur
        valid_mask[unpainted] = 1

    return height_map, valid_mask