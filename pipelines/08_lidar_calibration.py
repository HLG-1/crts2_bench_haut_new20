"""
pipelines/08_lidar_calibration.py

Étape additive : applique HRF (DepthPro/DAV2) ou BitFit (HTC-DC Net/TSE-Net)
sur les prédictions déjà produites à l'étape 07, à lancer UNIQUEMENT une fois
les vraies données LiDAR reçues.
"""
import sys
sys.path.append(".")
import numpy as np
from core.lidar_calibration.hrf import extraire_features_hrf, entrainer_hrf, corriger_hrf
from core.lidar_calibration.peft_bitfit import entrainer_bitfit
from core.metrics import calculer_metriques, tableau_comparatif

def calibrer_hrf_modele(nom_modele, fonction_inference, train_ids, test_ids, charger_patch):
    features_train, residus_train = [], []
    for patch_id in train_ids:
        img, gt_lidar, mask = charger_patch(patch_id)  # gt_lidar = HAUTEUR_LIDAR une fois dispo
        if mask.sum() == 0:
            continue
        pred_map = fonction_inference(img)
        valeur_brute = float(np.median(pred_map[mask == 1]))
        hauteur_reelle = float(np.median(gt_lidar[mask == 1]))
        features_train.append(extraire_features_hrf(pred_map, img, mask))
        residus_train.append(valeur_brute - hauteur_reelle)

    reg = entrainer_hrf(features_train, residus_train)

    y_pred, y_true = [], []
    for patch_id in test_ids:
        img, gt_lidar, mask = charger_patch(patch_id)
        if mask.sum() == 0:
            continue
        pred_map = fonction_inference(img)
        y_pred.append(corriger_hrf(reg, pred_map, img, mask))
        y_true.append(float(np.median(gt_lidar[mask == 1])))

    return np.array(y_pred), np.array(y_true)


if __name__ == "__main__":
    print("En attente des données LiDAR — squelette prêt, non exécutable pour l'instant.")