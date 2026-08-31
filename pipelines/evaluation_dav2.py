"""
Script isolé pour l'évaluation du modèle DAV2 fine-tuné.
Ne relance PAS l'entraînement : réutilise directement le checkpoint
best.pth déjà sauvegardé.

Réutilise les fonctions de 04_finetuning_dav2.py (préparation splits,
chargement modèle, inférence, éval) via importlib, car le nom du module
commence par un chiffre et ne peut pas être importé avec `import`.

Usage:
    python pipelines/05_evaluation_dav2.py
    python pipelines/05_evaluation_dav2.py --checkpoint exp/batiments_maroc/latest.pth
"""
import sys
import os
import argparse
import importlib.util

import numpy as np

sys.path.append(".")

# --- Import dynamique de 04_finetuning_dav2.py (nom commençant par un chiffre) ---
MODULE_PATH = os.path.join(os.path.dirname(__file__), "04_finetuning_dav2.py")
spec = importlib.util.spec_from_file_location("finetuning_dav2", MODULE_PATH)
finetuning_dav2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(finetuning_dav2)

# Fonctions/constantes réutilisées telles quelles (aucune duplication de logique)
charger_modele_finetune = finetuning_dav2.charger_modele_finetune
evaluer_sur_split = finetuning_dav2.evaluer_sur_split
inferer_finetune = finetuning_dav2.inferer_finetune
DEFAULT_MAX_DEPTH = finetuning_dav2.DEFAULT_MAX_DEPTH

from core.data_utils import charger_split
from core.metrics import tableau_comparatif, mesurer_latence


def main():
    parser = argparse.ArgumentParser(description="Évaluation isolée du modèle DAV2 fine-tuné (sans réentraînement).")
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Chemin explicite vers un checkpoint .pth. Par défaut : best.pth (repli sur latest.pth).",
    )
    parser.add_argument(
        "--max-depth", type=float, default=DEFAULT_MAX_DEPTH,
        help="Profondeur max utilisée pour instancier le modèle (doit correspondre à l'entraînement).",
    )
    parser.add_argument(
        "--splits", nargs="+", default=["val", "test"],
        help="Splits à évaluer (ex: --splits val test, ou juste --splits test).",
    )
    args = parser.parse_args()

    print("=== Évaluation isolée (aucun réentraînement) ===")
    print(f"Checkpoint : {args.checkpoint or '(auto: best.pth puis latest.pth)'}")

    model = charger_modele_finetune(max_depth=args.max_depth, checkpoint_path=args.checkpoint)

    resultats = {}
    latences = {}
    patch_ex_global = None

    for split_name in args.splits:
        ids = charger_split(split_name)
        y_pred, y_true, patch_ex = evaluer_sur_split(model, ids, split_name)
        resultats[split_name] = (y_pred, y_true)
        if patch_ex is not None:
            patch_ex_global = patch_ex

        os.makedirs("results", exist_ok=True)
        np.savez(
            f"results/predictions_dav2_finetune_{split_name}.npz",
            y_pred=y_pred, y_true=y_true,
        )

    if patch_ex_global is not None:
        latences["DAV2_finetune"] = mesurer_latence(
            lambda img: inferer_finetune(model, img), patch_ex_global
        )

    os.makedirs("results", exist_ok=True)
    for split_name, (y_pred, y_true) in resultats.items():
        tableau = tableau_comparatif(
            {"DAV2_finetune": (y_pred, y_true)}, latences_par_modele=latences
        )
        print(f"\n=== RÉSULTATS {split_name.upper()} ===")
        print(tableau)
        tableau.to_csv(f"results/metrics_dav2_finetune_{split_name}.csv")

    # Compatibilité avec les noms de fichiers historiques (basés sur "test")
    if "test" in resultats:
        y_pred_test, y_true_test = resultats["test"]
        np.savez("results/predictions_dav2_finetune.npz", y_pred=y_pred_test, y_true=y_true_test)
        tableau_comparatif(
            {"DAV2_finetune": (y_pred_test, y_true_test)}, latences_par_modele=latences
        ).to_csv("results/metrics_dav2_finetune.csv")

    print("\nSauvegardé : metrics et predictions dans results/")


if __name__ == "__main__":
    main()