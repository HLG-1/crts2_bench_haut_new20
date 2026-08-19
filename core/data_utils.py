"""
core/data_utils.py

Utilitaires pour charger les splits et les patches de données d'entraînement,
validation et test générés lors de la préparation des données.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image


def charger_split(split_name: str, splits_dir: str = "data/splits") -> list[str]:
    """
    Charge la liste des identifiants de patchs pour un split donné.

    Args:
        split_name: Nom du split ("train", "val", ou "test").
        splits_dir: Répertoire contenant les fichiers de splits.

    Returns:
        Liste des IDs de patchs (e.g. ['zone1_00000', 'zone2_00001', ...]).
    """
    split_path = Path(splits_dir) / f"{split_name}.txt"
    if not split_path.exists():
        raise FileNotFoundError(f"Fichier de split introuvable : {split_path}")

    with open(split_path, "r", encoding="utf-8") as f:
        patch_ids = [line.strip() for line in f if line.strip()]
    return patch_ids


def charger_patch(
    patch_id: str, patches_dir: str = "data/patches"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Charge l'image RGB, la carte de hauteur de référence (GT) et le masque
    bâtiment associé à un patch_id.

    Args:
        patch_id: Identifiant du patch (e.g. 'zone1_00042').
        patches_dir: Répertoire racine des patchs contenant les sous-dossiers par zone.

    Returns:
        img: Image RGB en tableau numpy uint8 (H, W, 3).
        gt: Carte de hauteur réelle en tableau numpy float32 (H, W).
        mask: Masque binaire bâtiment en tableau numpy uint8 (H, W).
    """
    zone = patch_id.split("_")[0]
    zone_dir = Path(patches_dir) / zone

    img_path = zone_dir / f"{patch_id}_IMG.png"
    gt_path = zone_dir / f"{patch_id}_height_gt.npy"
    mask_path = zone_dir / f"{patch_id}_valid_mask.npy"

    if not img_path.exists():
        raise FileNotFoundError(f"Image introuvable : {img_path}")
    if not gt_path.exists():
        raise FileNotFoundError(f"Vérité terrain hauteur introuvable : {gt_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"Masque introuvable : {mask_path}")

    img = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
    gt = np.load(gt_path).astype(np.float32)
    mask = np.load(mask_path).astype(np.uint8)

    return img, gt, mask