"""Split train/val/test : zone3 en hold-out, zone1+zone2 en train/val."""
from __future__ import annotations
import os
import random


def generer_splits(manifests_train_val: list[str], manifest_test: list[str],
                    out_dir: str, ratio_train: float = 0.8, seed: int = 42):
    os.makedirs(out_dir, exist_ok=True)
    random.seed(seed)
    pool = manifests_train_val.copy()
    random.shuffle(pool)
    idx = int(ratio_train * len(pool))

    splits = {"train": pool[:idx], "val": pool[idx:], "test": manifest_test}
    for nom, lst in splits.items():
        with open(f"{out_dir}/{nom}.txt", "w") as f:
            f.write("\n".join(lst))
        print(f"{nom}: {len(lst)} patches")
    return splits