"""
dataset/batiments_maroc.py

"""
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

from .augmentations_depth import get_train_transforms, get_val_transforms, apply_transform


class BatimentsMarocDataset(Dataset):
    def __init__(self, list_file, patches_dir, size=(518, 518), is_train=True):
        """
        Args:
            list_file: chemin vers le fichier de split (train.txt, val.txt, test.txt)
            patches_dir: répertoire contenant les patches
            size: taille de sortie (H, W)
            is_train: si True, applique les augmentations d'entraînement; sinon validation/test
        """
        with open(list_file) as f:
            self.ids = [l.strip() for l in f if l.strip()]
        self.patches_dir = patches_dir
        self.size = size
        self.is_train = is_train

        if is_train:
            self.transform = get_train_transforms(image_size=size)
        else:
            self.transform = get_val_transforms(image_size=size)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        patch_id = self.ids[idx]
        zone = patch_id.rsplit("_", 1)[0]  # "zone1_00042" -> "zone1"

        img_path = f"{self.patches_dir}/{zone}/{patch_id}_IMG.png"
        gt_path = f"{self.patches_dir}/{zone}/{patch_id}_height_gt.npy"
        mask_path = f"{self.patches_dir}/{zone}/{patch_id}_valid_mask.npy"

        img = np.array(Image.open(img_path).convert("RGB"))
        depth = np.load(gt_path).astype(np.float32)
        mask = np.load(mask_path).astype(np.uint8)

        # Appliquer les augmentations (géométrie + photométrie cohérentes)
        out = apply_transform(self.transform, img, depth, mask)

        # Convertir en tenseurs.
        # depth_t et mask_t restent en (H, W) -- PAS de unsqueeze(0) -- pour
        # matcher la forme attendue par train.py : pred a la forme (B, H, W)
        # une fois batché, donc depth/mask doivent aussi rester (H, W) avant
        # batching pour donner (B, H, W) après DataLoader.
        img_t = torch.from_numpy(out["image"]).permute(2, 0, 1).float()
        depth_t = torch.from_numpy(out["depth"]).float()          # (H, W)
        mask_t = torch.from_numpy(out["valid_mask"]).bool()       # (H, W)

        return {
            "image": img_t,
            "depth": depth_t,
            "valid_mask": mask_t,
        }