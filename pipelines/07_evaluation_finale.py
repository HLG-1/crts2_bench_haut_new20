"""
Rassemble les résultats des 4 (ou 5, si zero-shot inclut 2 variantes) CSV
produits par les scripts 03 à 06, construit le tableau comparatif final,
la matrice de significativité Wilcoxon, et sauvegarde le rapport complet.

Nécessite d'avoir aussi accès aux (y_pred, y_true) bruts de chaque modèle,
pas seulement leurs métriques agrégées, pour le test de Wilcoxon (test
apparié, a besoin des valeurs par bâtiment, pas juste du résumé). On les
recharge donc depuis des fichiers .npz sauvegardés par chaque script
précédent.
"""
import sys, os
sys.path.append(".")
import numpy as np
import pandas as pd

from core.metrics import tableau_comparatif, matrice_significativite


def charger_predictions_brutes(nom_modele: str, chemin_npz: str) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(chemin_npz)
    return data["y_pred"], data["y_true"]


def traiter_split(split_name: str, modeles: list[str]):
    resultats = {}
    for nom in modeles:
        chemins = [
            f"results/predictions_{nom.lower()}_{split_name}.npz",
            f"results/predictions_{nom.lower()}.npz",
        ]
        chemin_trouve = None
        for c in chemins:
            if os.path.exists(c):
                chemin_trouve = c
                break

        if chemin_trouve:
            resultats[nom] = charger_predictions_brutes(nom, chemin_trouve)
        else:
            print(f"  [!] {nom} ({split_name}): aucun fichier de prédictions trouvé parmi {chemins}")

    if not resultats:
        print(f"Aucun résultat trouvé pour le split '{split_name}'.")
        return None

    print(f"\n==========================================")
    print(f"=== Tableau comparatif final — SPLIT {split_name.upper()} ===")
    print(f"==========================================")
    tableau = tableau_comparatif(resultats)
    print(tableau)
    tableau.to_csv(f"results/tableau_final_comparatif_{split_name}.csv")
    if split_name == "test":
        tableau.to_csv("results/tableau_final_comparatif.csv")

    print(f"\n=== Matrice de significativité Wilcoxon ({split_name.upper()}) ===")
    try:
        mat_signif = matrice_significativite(resultats)
        print(mat_signif)
        mat_signif.to_csv(f"results/matrice_significativite_{split_name}.csv")
        if split_name == "test":
            mat_signif.to_csv("results/matrice_significativite.csv")
    except ValueError as e:
        print(f"Impossible de calculer la matrice ({split_name}) : {e}")

    meilleur = tableau["RMSE_B"].idxmin()
    print(f"\nMeilleur modèle ({split_name}) — RMSE-B : {meilleur} ({tableau.loc[meilleur, 'RMSE_B']} m)")
    return tableau


if __name__ == "__main__":
    modeles = ["DepthPro", "DAV2_zeroshot", "DAV2_finetune", "HTC_DC_Net", "TSE_Net"]
    
    print("=== ÉVALUATION FINALE MULTI-SPLITS (VAL & TEST) ===")
    t_val = traiter_split("val", modeles)
    t_test = traiter_split("test", modeles)
    
    print("\nÉvaluation finale terminée. Rapports générés dans results/")