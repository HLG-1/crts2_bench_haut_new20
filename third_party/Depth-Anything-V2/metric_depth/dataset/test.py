"""
Test de non-regression : verifie que
  1. l'import relatif fonctionne (package correctement structure)
  2. depth_t et mask_t ont bien la forme (H, W), PAS (1, H, W)
  3. apres batching par le DataLoader, pred simule (B, H, W) et matche
     bien depth/valid_mask batches, meme avec bs > 1
  4. les pixels mis a depth=0 par CoarseDropout correspondent exactement
     aux pixels ou valid_mask=0 (hypothese du commentaire verifiee)
"""
import sys, os
import tempfile
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from .batiments_maroc import BatimentsMarocDataset

# --- Créer un faux dataset minimal sur disque pour tester __getitem__ ---
patches_dir = tempfile.mkdtemp(prefix="fake_patches_")
zone = "zone1"
os.makedirs(f"{patches_dir}/{zone}", exist_ok=True)

ids = []
for i in range(6):
    patch_id = f"{zone}_{i:05d}"
    ids.append(patch_id)
    img = (np.random.rand(600, 800, 3) * 255).astype(np.uint8)
    depth = (np.random.rand(600, 800) * 30).astype(np.float32)
    mask = np.ones((600, 800), dtype=np.uint8)
    Image.fromarray(img).save(f"{patches_dir}/{zone}/{patch_id}_IMG.png")
    np.save(f"{patches_dir}/{zone}/{patch_id}_height_gt.npy", depth)
    np.save(f"{patches_dir}/{zone}/{patch_id}_valid_mask.npy", mask)

list_file = os.path.join(tempfile.gettempdir(), "fake_train.txt")
with open(list_file, "w") as f:
    f.write("\n".join(ids))

# --- Test 1 : formes individuelles (H, W), pas (1, H, W) ---
ds = BatimentsMarocDataset(list_file, patches_dir=patches_dir, size=(518, 518), is_train=True)
sample = ds[0]
print("Formes individuelles :")
print("  image:", sample["image"].shape)       # attendu (3, 518, 518)
print("  depth:", sample["depth"].shape)        # attendu (518, 518)  <- pas (1,518,518)
print("  valid_mask:", sample["valid_mask"].shape)  # attendu (518, 518)

assert sample["depth"].dim() == 2, f"BUG: depth a {sample['depth'].dim()} dims, attendu 2 (H,W)"
assert sample["valid_mask"].dim() == 2, f"BUG: valid_mask a {sample['valid_mask'].dim()} dims, attendu 2 (H,W)"
print("OK : depth/valid_mask sont bien en (H, W), pas (1, H, W)\n")

# --- Test 2 : formes après batching avec bs=4 (le cas qui plantait) ---
loader = DataLoader(ds, batch_size=4, drop_last=True)
batch = next(iter(loader))
print("Formes après batching (bs=4) :")
print("  image:", batch["image"].shape)        # attendu (4, 3, 518, 518)
print("  depth:", batch["depth"].shape)         # attendu (4, 518, 518)
print("  valid_mask:", batch["valid_mask"].shape)  # attendu (4, 518, 518)

# Simuler pred du modèle : forme (B, H, W), comme DepthAnythingV2 le produit réellement
fake_pred = torch.randn(4, 518, 518)

assert fake_pred.shape == batch["depth"].shape, \
    f"BUG: pred {fake_pred.shape} != depth {batch['depth'].shape}"
assert fake_pred.shape == batch["valid_mask"].shape, \
    f"BUG: pred {fake_pred.shape} != valid_mask {batch['valid_mask'].shape}"
print("OK : pred (B,H,W) matche exactement depth et valid_mask batchés (bs=4)\n")

# --- Test 3 : simuler la ligne de la loss exacte de train.py pour confirmer aucun crash ---
min_depth, max_depth = 0.0, 41.0
valid = (batch["valid_mask"] == 1) & (batch["depth"] >= min_depth) & (batch["depth"] <= max_depth)
selected = fake_pred[valid]  # doit fonctionner sans erreur de shape
print(f"OK : indexation booléenne pred[valid_mask] fonctionne, {selected.numel()} pixels sélectionnés\n")

# --- Test 4 : cohérence CoarseDropout (depth=0 uniquement là où valid_mask=0) ---
from .augmentations_depth import get_train_transforms, apply_transform
tf = get_train_transforms(image_size=(256, 256), rotation=False, flip=False, scale_jitter=False,
                           brightness_contrast=False, blur=False, shadow_simulation=False, cutout=True)
img = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
depth = np.full((256, 256), 15.0, dtype=np.float32)
mask = np.ones((256, 256), dtype=np.uint8)

# Forcer plusieurs essais pour être sûr que CoarseDropout se déclenche au moins une fois (p=0.3)
found_dropout = False
for _ in range(30):
    out = apply_transform(tf, img, depth.copy(), mask.copy())
    zero_depth_px = (out["depth"] == 0)
    zero_mask_px = (out["valid_mask"] == 0)
    if zero_depth_px.sum() > 0:
        found_dropout = True
        # Vérifie que CHAQUE pixel depth=0 correspond bien à valid_mask=0
        assert np.array_equal(zero_depth_px, zero_mask_px), \
            "BUG: des pixels depth=0 ne correspondent PAS à valid_mask=0 -> loss corrompue silencieusement !"
        break

if found_dropout:
    print("OK : tous les pixels depth=0 (cutout) correspondent bien à valid_mask=0 -> filtrés de la loss\n")
else:
    print("ATTENTION : CoarseDropout ne s'est jamais déclenché sur 30 essais (p=0.3, possible par hasard)\n")

print("=== TOUS LES TESTS PASSENT ===")