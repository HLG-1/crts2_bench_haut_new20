"""
Découpage en tuiles chevauchantes, alignées entre image RGB, carte de
hauteur et masque de validité. Padding réfléchi sur les bords.
"""
from __future__ import annotations
import os
import numpy as np
import cv2
from PIL import Image
from rasterio.windows import Window
import rasterio

from core.geo_io import read_window_rgb


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
    height_map: np.ndarray,
    valid_mask: np.ndarray,
    out_dir: str,
    zone_name: str,
    tile_size: int = 512,
    overlap: int = 64,
    min_pixels_batiment: int = 100,
):
    os.makedirs(out_dir, exist_ok=True)
    H, W = valid_mask.shape
    rows, cols = tile_positions(H, W, tile_size, overlap)

    manifest = []
    compteur = 0
    for r in rows:
        for c in cols:
            mask_crop = valid_mask[r:r+tile_size, c:c+tile_size]
            if mask_crop.sum() < min_pixels_batiment:
                continue

            window = Window(c, r, tile_size, tile_size)
            img_crop = read_window_rgb(tif_path, window)
            gt_crop = height_map[r:r+tile_size, c:c+tile_size]

            # Padding réfléchi si la tuile déborde (bords de zone)
            rh, rw = mask_crop.shape
            if rh < tile_size or rw < tile_size:
                pad_h, pad_w = tile_size - rh, tile_size - rw
                img_crop = cv2.copyMakeBorder(img_crop, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
                gt_crop = np.pad(gt_crop, ((0, pad_h), (0, pad_w)), mode="reflect")
                mask_crop = np.pad(mask_crop, ((0, pad_h), (0, pad_w)), mode="reflect")

            patch_id = f"{zone_name}_{compteur:05d}"
            Image.fromarray(img_crop).save(f"{out_dir}/{patch_id}_IMG.png")
            np.save(f"{out_dir}/{patch_id}_height_gt.npy", gt_crop)
            np.save(f"{out_dir}/{patch_id}_valid_mask.npy", mask_crop)
            manifest.append(patch_id)
            compteur += 1

    return manifest