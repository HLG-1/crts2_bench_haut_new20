"""
Utilitaires géospatiaux : lecture fenêtrée, vérification CRS/alignement.
Repris et adapté de pipeline_batiments_dagster/core/geo_io.py.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
import geopandas as gpd


def raster_meta(path: str | Path) -> dict:
    """Lit uniquement les métadonnées du GeoTIFF, sans charger les pixels."""
    with rasterio.open(path) as src:
        return {
            "path": str(path),
            "crs": src.crs,
            "transform": src.transform,
            "bounds": src.bounds,
            "width": src.width,
            "height": src.height,
            "n_bands": src.count,
            "stem": Path(path).stem,
        }


def read_window_rgb(path: str | Path, window: Window) -> np.ndarray:
    """Lit exactement la fenêtre demandée, en RGB uint8."""
    with rasterio.open(path) as src:
        data = src.read([1, 2, 3], window=window)
    return np.transpose(data, (1, 2, 0)).astype(np.uint8)


def window_transform(path: str | Path, window: Window):
    with rasterio.open(path) as src:
        return rasterio.windows.transform(window, src.transform)


def safe_crs(crs_value) -> str:
    """Retourne toujours un CRS valide pour GeoPandas."""
    if crs_value is None:
        return "EPSG:4326"
    try:
        s = crs_value.to_string() if hasattr(crs_value, "to_string") else str(crs_value).strip()
    except Exception:
        s = ""
    if not s or s.lower() in ("none", "null", "unknown"):
        return "EPSG:4326"
    try:
        from pyproj import CRS
        CRS.from_user_input(s)
        return s
    except Exception:
        return "EPSG:4326"


def auditer_zone(nom: str, shp_path: str, tif_path: str) -> tuple[gpd.GeoDataFrame, bool]:
    """Vérifie CRS + chevauchement vecteur/raster, sans charger les pixels."""
    gdf = gpd.read_file(shp_path)
    meta = raster_meta(tif_path)
    raster_crs = safe_crs(meta["crs"])

    gdf_reproj = gdf.to_crs(raster_crs) if str(gdf.crs) != raster_crs else gdf
    vec_bounds = gdf_reproj.total_bounds
    rb = meta["bounds"]

    chevauchement = not (
        vec_bounds[2] < rb.left or vec_bounds[0] > rb.right or
        vec_bounds[3] < rb.bottom or vec_bounds[1] > rb.top
    )

    print(f"[{nom}] {len(gdf)} polygones | chevauchement={chevauchement} | "
          f"HAUTEUR min/max/moy = {gdf['HAUTEUR'].min()}/{gdf['HAUTEUR'].max()}/{gdf['HAUTEUR'].mean():.1f}")

    return gdf_reproj, chevauchement