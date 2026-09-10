"""
Script pour évaluer HTC-DC Net sur le test set et sauvegarder les prédictions
pour inclusion dans la comparaison finale avec tous les modèles
"""
import sys, os, glob
from pathlib import Path

# Désactiver l'affichage OpenCV pour éviter les problèmes Qt
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

# Ajouter la racine et HTC-DC-Net au PYTHONPATH
sys.path.append(".")
htc_dir_abs = os.path.abspath("third_party/HTC-DC-Net")
if htc_dir_abs not in sys.path:
    sys.path.insert(0, htc_dir_abs)

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from PIL import Image

# Importer les modules HTC-DC Net
from htcdc import UBins

# Configuration
HTC_DIR = "third_party/HTC-DC-Net"
DATA_HTC_DIR = "data/htc_dc_net_format"
RESULTS_DIR = "results/htc_dc_net_evaluation"

def trouver_dernier_checkpoint():
    """Trouve le checkpoint le plus récent dans les résultats d'entraînement."""
    training_dirs = glob.glob("results/htc_dc_net_training/*")
    if not training_dirs:
        raise FileNotFoundError("Aucun répertoire d'entraînement trouvé dans results/htc_dc_net_training/")
    
    # Trier par date de modification
    latest_dir = max(training_dirs, key=os.path.getmtime)
    
    # Chercher le meilleur checkpoint (par RMSE)
    checkpoints = glob.glob(os.path.join(latest_dir, "checkpoint_best_*.pth.tar"))
    if checkpoints:
        return max(checkpoints, key=os.path.getmtime)
    
    # Sinon prendre le dernier checkpoint
    checkpoints = glob.glob(os.path.join(latest_dir, "checkpoint_*.pth.tar"))
    if checkpoints:
        return max(checkpoints, key=os.path.getmtime)
    
    raise FileNotFoundError(f"Aucun checkpoint trouvé dans {latest_dir}")

def charger_modele(checkpoint_path, device='cuda'):
    """Charge le modèle depuis un checkpoint."""
    print(f"Chargement du checkpoint: {checkpoint_path}")
    
    # Charger la configuration
    config_path = os.path.join(os.path.dirname(checkpoint_path), "config.yaml")
    if os.path.exists(config_path):
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Corriger les chemins relatifs dans la configuration
        if 'data_dir' in config and config['data_dir'].startswith('../../'):
            config['data_dir'] = os.path.abspath(DATA_HTC_DIR)
    else:
        # Configuration par défaut
        config = {
            'num_classes': 256,
            'backbone': 'efficientnetb0',
            'fusion_mode': 'last',
            'head_tail_cut': False,
            'earlier': False,
            'prob_loss': False,
            'prob_loss_bg': False,
            'patch_size': 4,
            'dropout_rate': 0.2,
            'chamfer_weight': 0.01,
            'data_dir': os.path.abspath(DATA_HTC_DIR)
        }
    
    # S'assurer que le fichier ndsm_stats existe
    ndsm_stats_file = os.path.join(DATA_HTC_DIR, "ndsm_stats.pickle")
    if not os.path.exists(ndsm_stats_file):
        print(f"Création de ndsm_stats par défaut")
        default_stats = {
            'mean': 0.0,
            'std': 1.0,
            'min': 0.0,
            'max': 50.0,
            'h_max': 50.0
        }
        torch.save(default_stats, ndsm_stats_file)
    
    # Corriger le chemin data_dir
    if 'data_dir' not in config or config['data_dir'].startswith('../../'):
        config['data_dir'] = os.path.abspath(DATA_HTC_DIR)
    
    if 'num_classes' not in config:
        config['num_classes'] = 256
    
    # Créer le modèle
    try:
        model = UBins(config)
    except Exception as e:
        print(f"Erreur création modèle: {e}")
        config_simple = {
            'num_classes': 256,
            'backbone': 'efficientnetb0',
            'fusion_mode': 'last',
            'head_tail_cut': False,
            'earlier': False,
            'prob_loss': False,
            'prob_loss_bg': False,
            'patch_size': 4,
            'dropout_rate': 0.2,
            'chamfer_weight': 0.01,
            'data_dir': os.path.abspath(DATA_HTC_DIR)
        }
        model = UBins(config_simple)
    
    # Charger le checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"Erreur chargement avec weights_only=False: {e}")
        torch.serialization.add_safe_globals([np.ndarray])
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device).eval()
    
    # Charger les statistiques de normalisation
    img_stats_file = os.path.join(DATA_HTC_DIR, "image_stats.pickle")
    if os.path.isfile(img_stats_file):
        try:
            mean, std = torch.load(img_stats_file, weights_only=False)
        except:
            mean, std = [123.675, 116.28, 103.53], [58.395, 57.12, 57.375]
    else:
        mean, std = [123.675, 116.28, 103.53], [58.395, 57.12, 57.375]
    
    model.mean = torch.tensor(mean, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    model.std = torch.tensor(std, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    model.device = device
    
    print(f"Modèle chargé avec succès sur {device}")
    return model, config

def inferer_htc(model, img_rgb: np.ndarray, target_size=256) -> np.ndarray:
    """Inférence HTC-DC Net."""
    device = getattr(model, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    h_orig, w_orig = img_rgb.shape[:2]

    # Redimensionner à la taille attendue par le modèle (256x256)
    from PIL import Image
    img_pil = Image.fromarray(img_rgb)
    img_resized = img_pil.resize((target_size, target_size), Image.BILINEAR)
    img_rgb_resized = np.array(img_resized)

    # Convertir en tenseur
    img_t = torch.from_numpy(img_rgb_resized.astype(np.float32)).permute(2, 0, 1).unsqueeze(0).to(device)

    # Normalisation
    mean = getattr(model, "mean", torch.tensor([123.675, 116.28, 103.53], device=device).view(1, 3, 1, 1))
    std = getattr(model, "std", torch.tensor([58.395, 57.12, 57.375], device=device).view(1, 3, 1, 1))
    img_t = (img_t - mean) / std

    with torch.no_grad():
        res = model.model(img_t)
        pred_list = res[1]
        pred_finest = pred_list[-1]

        # Redimensionnement à la taille originale
        pred_resized = F.interpolate(pred_finest, size=(h_orig, w_orig), mode="bilinear", align_corners=False)
        pred_map = pred_resized.squeeze().cpu().numpy().astype(np.float32)

    return pred_map

def evaluer_sur_test(model, use_stratified=True):
    """Évalue HTC-DC Net sur le test set et sauvegarde les prédictions."""
    sys.path.append(".")
    from core.data_utils import charger_split, charger_patch
    
    # Charger les splits
    if use_stratified:
        test_ids = charger_split("test", splits_dir="results/stratified_split/splits_new")
        suffix = "_stratified"
    else:
        test_ids = charger_split("test")
        suffix = ""
    
    print(f"Évaluation sur {len(test_ids)} patches (test{suffix})")
    
    y_pred = []
    y_true = []
    
    for patch_id in test_ids:
        img, gt, mask = charger_patch(patch_id)
        if mask.sum() == 0:
            continue
        
        # Inférence
        pred_map = inferer_htc(model, img, target_size=256)
        
        # Agréger par médiane sur les pixels bâtiment
        pred_median = float(np.median(pred_map[mask == 1]))
        true_median = float(np.median(gt[mask == 1]))
        
        y_pred.append(pred_median)
        y_true.append(true_median)
        
        if (len(y_pred) % 100) == 0:
            print(f"  {len(y_pred)}/{len(test_ids)} patches évalués")
    
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    
    # Sauvegarder les prédictions
    os.makedirs("results", exist_ok=True)
    np.savez(f"results/predictions_htc_dc_net_test{suffix}.npz", y_pred=y_pred, y_true=y_true)
    print(f"✓ Prédictions sauvegardées: predictions_htc_dc_net_test{suffix}.npz")
    
    # Calculer les métriques
    from core.metrics import calculer_metriques
    metrics = calculer_metriques(y_pred, y_true)
    
    print(f"\n📊 Métriques HTC-DC Net (test{suffix}):")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    return y_pred, y_true, metrics

def main():
    print("="*60)
    print("ÉVALUATION HTC-DC NET POUR COMPARAISON FINALE")
    print("="*60)
    
    # Trouver le checkpoint
    try:
        checkpoint_path = trouver_dernier_checkpoint()
        print(f"✓ Checkpoint trouvé: {checkpoint_path}")
    except FileNotFoundError as e:
        print(f"✗ {e}")
        return
    
    # Charger le modèle
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = charger_modele(checkpoint_path, device)
    
    # Évaluer sur test stratifié
    print("\n📂 Évaluation sur test stratifié...")
    try:
        y_pred_strat, y_true_strat, metrics_strat = evaluer_sur_test(model, use_stratified=True)
    except Exception as e:
        print(f"✗ Erreur évaluation stratifiée: {e}")
        y_pred_strat, y_true_strat, metrics_strat = None, None, None
    
    # Évaluer sur test original
    print("\n📂 Évaluation sur test original...")
    try:
        y_pred_orig, y_true_orig, metrics_orig = evaluer_sur_test(model, use_stratified=False)
    except Exception as e:
        print(f"✗ Erreur évaluation original: {e}")
        y_pred_orig, y_true_orig, metrics_orig = None, None, None
    
    # Résumé
    print("\n" + "="*60)
    print("RÉSUMÉ DES ÉVALUATIONS HTC-DC NET")
    print("="*60)
    
    if metrics_strat:
        print(f"\nTest Stratifié:")
        print(f"  MAE: {metrics_strat['MAE']:.4f} m")
        print(f"  RMSE: {metrics_strat['RMSE_B']:.4f} m")
        print(f"  NMAD: {metrics_strat['NMAD']:.4f}")
        print(f"  Corrélation: {metrics_strat['Correlation']:.4f}")
    
    if metrics_orig:
        print(f"\nTest Original:")
        print(f"  MAE: {metrics_orig['MAE']:.4f} m")
        print(f"  RMSE: {metrics_orig['RMSE_B']:.4f} m")
        print(f"  NMAD: {metrics_orig['NMAD']:.4f}")
        print(f"  Corrélation: {metrics_orig['Correlation']:.4f}")
    
    print("\n✓ Évaluation HTC-DC Net terminée")
    print("  Les prédictions sont sauvegardées et peuvent être utilisées")
    print("  dans la comparaison finale avec tous les modèles")

if __name__ == "__main__":
    main()