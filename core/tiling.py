"""
Découpage en tuiles chevauchantes AVEC rasterisation de la hauteur
directement par fenêtre — évite de construire une carte pleine échelle
(qui fait plusieurs Go et fait planter la machine).
"""
from __future__ import annotations
import os
import numpy as np
from PIL import Image
from shapely.geometry import box
import rasterio
from rasterio.windows import Window
from rasterio.features import rasterize


def tile_positions(height: int, width: int, tile_size: int, overlap: int):
    stride = tile_size - overlap
    rows = list(range(0, max(height - tile_size, 0) + 1, stride))
    cols = list(range(0, max(width - tile_size, 0) + 1, stride))
    if not rows or rows[-1] + tile_size < height:
        rows.append(max(0, height - tile_size))
    if not cols or cols[-1] + tile_size < width:
        cols.append(max(0, width - tile_size))
    return sorted(set(rows)), sorted(set(cols))


def decouper_en_patches(
    tif_path: str,
    gdf_clean,                 # GeoDataFrame déjà filtré (géométries valides, HAUTEUR > 0)
    out_dir: str,
    zone_name: str,
    tile_size: int = 512,
    overlap: int = 64,
    min_pixels_batiment: int = 100,
):
    os.makedirs(out_dir, exist_ok=True)
    sindex = gdf_clean.sindex

    with rasterio.open(tif_path) as src:
        H, W, full_transform = src.height, src.width, src.transform
        rows, cols = tile_positions(H, W, tile_size, overlap)

        manifest = []
        compteur = 0
        for r in rows:
            for c in cols:
                window = Window(c, r, tile_size, tile_size)
                win_transform = rasterio.windows.transform(window, full_transform)
                bounds = rasterio.windows.bounds(window, full_transform)

                # Ne garder que les polygones qui touchent cette fenêtre (index spatial)
                candidats = list(sindex.intersection(bounds))
                if not candidats:
                    continue
                subset = gdf_clean.iloc[candidats]
                subset = subset[subset.geometry.intersects(box(*bounds))]
                if len(subset) == 0:
                    continue

                shape_hw = (tile_size, tile_size)
                height_map = np.zeros(shape_hw, dtype=np.float32)
                valid_mask = np.zeros(shape_hw, dtype=np.uint8)

                # Petits polygones peints en premier (même logique que core CRTS)
                paires = sorted(zip(subset.geometry, subset["HAUTEUR"]), key=lambda p: p[0].area)
                for geom, hauteur in paires:
                    layer = rasterize([(geom, 1)], out_shape=shape_hw, transform=win_transform,
                                      fill=0, dtype="uint8")
                    unpainted = (layer > 0) & (valid_mask == 0)
                    height_map[unpainted] = hauteur
                    valid_mask[unpainted] = 1

                if valid_mask.sum() < min_pixels_batiment:
                    continue  # patch sans assez de signal bâtiment, ignoré

                # boundless=True + fill_value=0 : gère automatiquement les bords de zone
                img_crop = src.read([1, 2, 3], window=window, boundless=True, fill_value=0)
                img_crop = np.transpose(img_crop, (1, 2, 0)).astype(np.uint8)

                patch_id = f"{zone_name}_{compteur:05d}"
                Image.fromarray(img_crop).save(f"{out_dir}/{patch_id}_IMG.png")
                np.save(f"{out_dir}/{patch_id}_height_gt.npy", height_map)
                np.save(f"{out_dir}/{patch_id}_valid_mask.npy", valid_mask)
                manifest.append(patch_id)
                compteur += 1

    return manifest