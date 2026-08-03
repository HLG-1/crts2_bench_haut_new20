import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import random

patches_dir = "data/patches/zone1"
tous_ids = sorted(set(f.split("_IMG")[0] for f in os.listdir(patches_dir) if "_IMG" in f))

echantillon = random.sample(tous_ids, 6)

fig, axes = plt.subplots(len(echantillon), 3, figsize=(12, 4 * len(echantillon)))
for i, patch_id in enumerate(echantillon):
    img = np.array(Image.open(f"{patches_dir}/{patch_id}_IMG.png"))
    gt = np.load(f"{patches_dir}/{patch_id}_height_gt.npy")
    mask = np.load(f"{patches_dir}/{patch_id}_valid_mask.npy")

    axes[i, 0].imshow(img); axes[i, 0].set_title(f"{patch_id} — Orthophoto")
    axes[i, 1].imshow(gt, cmap="viridis"); axes[i, 1].set_title("Hauteur GT (m)")
    axes[i, 2].imshow(mask, cmap="gray"); axes[i, 2].set_title("Masque bâtiment")

plt.tight_layout()
plt.savefig("verif_zone1.png", dpi=100)
print("Sauvegardé : verif_zone1.png")