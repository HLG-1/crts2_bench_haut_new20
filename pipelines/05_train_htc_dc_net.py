"""
pipelines/05_train_htc_dc_net.py

Conversion des patches au format image/mask/ndsm attendu par HTC-DC Net,
lancement de l'entraînement, évaluation sur zone3.
"""
import sys, os, subprocess, glob
from pathlib import Path

# Ajouter la racine et HTC-DC-Net au PYTHONPATH
sys.path.append(".")
htc_dir_abs = os.path.abspath("third_party/HTC-DC-Net")
if htc_dir_abs not in sys.path:
    sys.path.insert(0, htc_dir_abs)

import numpy as np
import rasterio
from rasterio.transform import from_origin
from PIL import Image
import torch
import torch.nn.functional as F

from core.data_utils import charger_split, charger_patch
from core.metrics import calculer_metriques, tableau_comparatif, mesurer_latence

HTC_DIR = "third_party/HTC-DC-Net"
DATA_HTC_DIR = "data/htc_dc_net_format"
CONFIG_PATH = "third_party/HTC-DC-Net/configs/batiments_maroc.yaml"
EXP_CONFIG_PATH = "third_party/HTC-DC-Net/configs/htcdc.yaml"


def calculer_et_sauvegarder_statistiques(data_dir: str = DATA_HTC_DIR):
    """
    Calcule et enregistre image_stats.pickle et ndsm_stats.pickle nécessaires
    pour initialiser et entraîner HTC-DC Net.
    """
    img_dir = os.path.join(data_dir, "image")
    ndsm_dir = os.path.join(data_dir, "ndsm")

    img_files = glob.glob(os.path.join(img_dir, "*_IMG.tif"))
    if not img_files:
        return

    # Image stats : mean, std par canal RGB (échelle 0..255)
    sum_x = np.zeros(3, dtype=np.float64)
    sum_x2 = np.zeros(3, dtype=np.float64)
    sum_pixels = 0

    for f in img_files:
        with rasterio.open(f) as src:
            arr = src.read().astype(np.float64)  # (3, H, W)
            arr = np.transpose(arr, (1, 2, 0)).reshape(-1, 3)
            sum_x += arr.sum(axis=0)
            sum_x2 += (arr * arr).sum(axis=0)
            sum_pixels += arr.shape[0]

    mean_rgb = list(sum_x / max(1, sum_pixels))
    std_rgb = list(np.sqrt(np.maximum(sum_x2 / max(1, sum_pixels) - (sum_x / max(1, sum_pixels)) ** 2, 1e-6)))
    torch.save([mean_rgb, std_rgb], os.path.join(data_dir, "image_stats.pickle"))

    # nDSM stats : mean, std, minh, maxh, count
    ndsm_files = glob.glob(os.path.join(ndsm_dir, "*_AGL.tif"))
    sum_h = 0.0
    sum_h2 = 0.0
    total_h_pixels = 0
    minh = float("inf")
    maxh = float("-inf")

    for f in ndsm_files:
        with rasterio.open(f) as src:
            arr = src.read(1).flatten().astype(np.float64)
            arr = np.nan_to_num(arr)
            sum_h += float(arr.sum())
            sum_h2 += float((arr ** 2).sum())
            total_h_pixels += arr.size
            cur_min = float(arr.min())
            cur_max = float(arr.max())
            if cur_min < minh:
                minh = cur_min
            if cur_max > maxh:
                maxh = cur_max

    minh = max(0.0, minh if minh != float("inf") else 0.0)
    maxh = max(1.0, maxh if maxh != float("-inf") else 45.0)
    mean_h = sum_h / max(1, total_h_pixels)
    std_h = float(np.sqrt(max(0.0, sum_h2 / max(1, total_h_pixels) - mean_h ** 2)))

    count = np.zeros(int(np.ceil(maxh)) + 1, dtype=np.int64)
    for f in ndsm_files:
        with rasterio.open(f) as src:
            arr = src.read(1).flatten().astype(np.float64)
            arr = np.clip(np.floor(np.nan_to_num(arr)), 0, maxh).astype(np.int64)
            b = np.bincount(arr, minlength=len(count))
            count[:len(b)] += b[:len(count)]

    torch.save([mean_h, std_h, minh, maxh, count], os.path.join(data_dir, "ndsm_stats.pickle"))
    print(f"Statistiques calculées et sauvegardées dans {data_dir} (h_max = {maxh:.2f} m)")


def convertir_format(split_name: str):
    """Convertit un split en fichiers .tif au format image/ndsm/mask attendu."""
    os.makedirs(f"{DATA_HTC_DIR}/image", exist_ok=True)
    os.makedirs(f"{DATA_HTC_DIR}/ndsm", exist_ok=True)
    os.makedirs(f"{DATA_HTC_DIR}/mask", exist_ok=True)
    os.makedirs(f"{DATA_HTC_DIR}/splits", exist_ok=True)

    patch_ids = charger_split(split_name)
    with open(f"{DATA_HTC_DIR}/splits/{split_name}.txt", "w", encoding="utf-8") as sf:
        sf.write("\n".join(patch_ids))

    for patch_id in patch_ids:
        img, gt, mask = charger_patch(patch_id)
        h, w = gt.shape
        transform = from_origin(0, h, 1, 1)  # géoréférencement factice

        with rasterio.open(f"{DATA_HTC_DIR}/image/{patch_id}_IMG.tif", "w", driver="GTiff",
                            height=h, width=w, count=3, dtype="uint8", transform=transform) as dst:
            dst.write(np.transpose(img, (2, 0, 1)))

        with rasterio.open(f"{DATA_HTC_DIR}/ndsm/{patch_id}_AGL.tif", "w", driver="GTiff",
                            height=h, width=w, count=1, dtype="float32", transform=transform) as dst:
            dst.write(gt[np.newaxis, :, :])

        with rasterio.open(f"{DATA_HTC_DIR}/mask/{patch_id}_BLG.tif", "w", driver="GTiff",
                            height=h, width=w, count=1, dtype="uint8", transform=transform) as dst:
            dst.write(mask[np.newaxis, :, :])

    print(f"{split_name} : {len(patch_ids)} patches convertis")


def lancer_entrainement(epochs: int = 200, batch_size: int = 32, use_stratified: bool = True):
    # Utiliser des chemins relatifs depuis le répertoire HTC-DC-Net
    config_rel = os.path.relpath(CONFIG_PATH, HTC_DIR)
    exp_config_rel = os.path.relpath(EXP_CONFIG_PATH, HTC_DIR)
    
    # Configuration des splits
    if use_stratified:
        splits_dir = "results/stratified_split/splits_new"
        print(f"Utilisation des splits stratifiés depuis {splits_dir}")
    else:
        splits_dir = "data/htc_dc_net_format/splits"
        print(f"Utilisation des splits standard depuis {splits_dir}")
    
    # Copier les fichiers de splits vers le répertoire attendu par HTC-DC-Net
    target_splits_dir = os.path.join(DATA_HTC_DIR, "splits")
    os.makedirs(target_splits_dir, exist_ok=True)
    
    for split in ["train", "val", "test"]:
        src_file = os.path.join(splits_dir, f"{split}.txt")
        dst_file = os.path.join(target_splits_dir, f"{split}.txt")
        if os.path.exists(src_file):
            import shutil
            shutil.copy(src_file, dst_file)
            print(f"  - {split}.txt copié ({len(open(src_file).readlines())} patches)")
    
    # Créer le répertoire pour sauvegarder les résultats par epoch
    results_dir = "results/htc_dc_net_training"
    os.makedirs(results_dir, exist_ok=True)
    print(f"Résultats d'entraînement seront sauvegardés dans {results_dir}")
    
    cmd = [
        "python3", "train.py", 
        "--config", config_rel, 
        "--exp_config", exp_config_rel, 
        "--max_epochs", str(epochs),
        "batch_size=" + str(batch_size)
    ]
    print("Commande :", " ".join(cmd))
    subprocess.run(cmd, cwd=HTC_DIR, check=True)
    
    # Copier les checkpoints et logs vers le répertoire de résultats
    print(f"\nCopie des résultats vers {results_dir}...")
    checkpoints_dir = os.path.join(HTC_DIR, "checkpoints", "htc_dc_net")
    if os.path.exists(checkpoints_dir):
        # Trouver le dossier d'expérience le plus récent
        exp_folders = [f for f in os.listdir(checkpoints_dir) if os.path.isdir(os.path.join(checkpoints_dir, f))]
        if exp_folders:
            latest_exp = sorted(exp_folders)[-1]
            src_exp_dir = os.path.join(checkpoints_dir, latest_exp)
            
            # Copier les checkpoints
            import shutil
            dst_exp_dir = os.path.join(results_dir, latest_exp)
            if os.path.exists(dst_exp_dir):
                shutil.rmtree(dst_exp_dir)
            shutil.copytree(src_exp_dir, dst_exp_dir)
            print(f"  - Checkpoints copiés : {dst_exp_dir}")
            
            # Sauvegarder les métriques d'entraînement dans un CSV
            csv_path = os.path.join(results_dir, "training_metrics.csv")
            if os.path.exists(os.path.join(dst_exp_dir, "config.yaml")):
                print(f"  - Configuration sauvegardée")
            print(f"  - Résultats disponibles dans {results_dir}")


def _trouver_checkpoint_htc(checkpoint_path: str | None = None) -> str:
    """Recherche le meilleur ou dernier checkpoint disponible pour HTC-DC Net."""
    if checkpoint_path and os.path.isfile(checkpoint_path):
        return checkpoint_path

    motifs = [
        "checkpoints/htc_dc_net/**/checkpoint_best_rmse.pth.tar",
        "checkpoints/htc_dc_net/**/checkpoint_best_mae.pth.tar",
        "checkpoints/htc_dc_net/**/checkpoint_last.pth.tar",
        "checkpoints/htc_dc_net/**/*.pth.tar",
        "third_party/HTC-DC-Net/checkpoints/**/checkpoint_best_rmse.pth.tar",
        "third_party/HTC-DC-Net/checkpoints/**/checkpoint_last.pth.tar",
        "third_party/HTC-DC-Net/checkpoints/**/*.pth.tar",
    ]
    for motif in motifs:
        matches = sorted(glob.glob(motif, recursive=True))
        if matches:
            return matches[-1]

    raise FileNotFoundError(
        "Aucun checkpoint HTC-DC Net trouvé dans checkpoints/htc_dc_net/. "
        "Lancez l'entraînement d'abord ou spécifiez le chemin du checkpoint."
    )


def charger_modele_entraine(checkpoint_path: str | None = None, device: str | torch.device | None = None):
    """
    Charge le modèle HTC-DC Net (UBins) et les poids du checkpoint entraîné.
    """
    from htcdc import UBins
    from utils import load_yaml

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    chkpt_path = _trouver_checkpoint_htc(checkpoint_path)
    print(f"Chargement du checkpoint HTC-DC Net : {chkpt_path}")

    # Configuration du modèle
    cfgs = {
        "model": "htcdc",
        "backbone": "efficientnetb0",
        "patch_size": 4,
        "num_classes": 256,
        "fusion_mode": "last",
        "head_tail_cut": False,
        "earlier": False,
        "prob_loss": False,
        "data_dir": DATA_HTC_DIR,
        "test": True,
        "device": device.type if isinstance(device, torch.device) else str(device),
    }

    exp_dir = os.path.dirname(chkpt_path)
    cfg_file = os.path.join(exp_dir, "config.yaml")
    if os.path.isfile(cfg_file):
        try:
            saved_cfgs = load_yaml(cfg_file)
            cfgs.update(saved_cfgs)
            cfgs["data_dir"] = DATA_HTC_DIR
        except Exception:
            pass

    # Vérification que ndsm_stats.pickle existe pour initialiser UBins
    stats_file = os.path.join(DATA_HTC_DIR, "ndsm_stats.pickle")
    if not os.path.isfile(stats_file):
        calculer_et_sauvegarder_statistiques(DATA_HTC_DIR)

    model = UBins(cfgs)
    chkpt = torch.load(chkpt_path, map_location="cpu")
    state_dict = chkpt["state_dict"] if "state_dict" in chkpt else chkpt
    model.load_state_dict(state_dict)
    model.to(device).eval()

    # Charger les statistiques de normalisation pour l'inférence
    img_stats_file = os.path.join(DATA_HTC_DIR, "image_stats.pickle")
    if os.path.isfile(img_stats_file):
        mean, std = torch.load(img_stats_file)
    else:
        mean, std = [123.675, 116.28, 103.53], [58.395, 57.12, 57.375]

    model.mean = torch.tensor(mean, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    model.std = torch.tensor(std, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    model.device = device

    return model


def inferer_htc(model, img_rgb: np.ndarray) -> np.ndarray:
    """
    Inférence HTC-DC Net : normalise l'image, passe dans le modèle, et retourne
    la carte de hauteur estimée (H, W) en mètres.
    """
    device = getattr(model, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    h_orig, w_orig = img_rgb.shape[:2]

    # Convertir en tenseur (1, 3, H, W)
    img_t = torch.from_numpy(img_rgb.astype(np.float32)).permute(2, 0, 1).unsqueeze(0).to(device)

    # Normalisation
    mean = getattr(model, "mean", torch.tensor([123.675, 116.28, 103.53], device=device).view(1, 3, 1, 1))
    std = getattr(model, "std", torch.tensor([58.395, 57.12, 57.375], device=device).view(1, 3, 1, 1))
    img_t = (img_t - mean) / std

    with torch.no_grad():
        res = model.model(img_t)
        # res est (bin_edges_list, pred_list, out_list, centers_list)
        pred_list = res[1]
        pred_finest = pred_list[-1]  # (1, 1, H', W')

        # Redimensionnement vers la résolution originale
        pred_resized = F.interpolate(pred_finest, size=(h_orig, w_orig), mode="bilinear", align_corners=False)
        pred_map = pred_resized.squeeze().cpu().numpy().astype(np.float32)

    return pred_map


from tqdm import tqdm

def evaluer_sur_split(model, split_ids: list, split_name: str = "split", save_results: bool = True):
    y_pred, y_true = [], []
    patch_exemple = None
    print(f"\nÉvaluation sur le jeu {split_name.upper()} ({len(split_ids)} patches)...")
    for patch_id in tqdm(split_ids, desc=f"HTC-DC Net Éval ({split_name.upper()})", unit="patch"):
        img, gt, mask = charger_patch(patch_id)
        if mask.sum() == 0:
            continue
        if patch_exemple is None:
            patch_exemple = img
        pred_map = inferer_htc(model, img)
        y_pred.append(float(np.median(pred_map[mask == 1])))
        y_true.append(float(np.median(gt[mask == 1])))
    
    # Sauvegarder les résultats si demandé
    if save_results:
        results_dir = "results/htc_dc_net_training"
        os.makedirs(results_dir, exist_ok=True)
        
        # Sauvegarder les prédictions
        np.savez(os.path.join(results_dir, f"predictions_{split_name}.npz"), 
                 y_pred=np.array(y_pred), y_true=np.array(y_true))
        
        # Calculer et sauvegarder les métriques
        if len(y_pred) > 0:
            metrics = calculer_metriques(np.array(y_pred), np.array(y_true))
            
            # Sauvegarder en CSV
            import pandas as pd
            df_metrics = pd.DataFrame([metrics])
            df_metrics.to_csv(os.path.join(results_dir, f"metrics_{split_name}.csv"), index=False)
            
            print(f"  - Résultats sauvegardés: {results_dir}/predictions_{split_name}.npz")
            print(f"  - Métriques sauvegardées: {results_dir}/metrics_{split_name}.csv")
    
    return np.array(y_pred), np.array(y_true), patch_exemple


if __name__ == "__main__":
    print("=== Étape 1 : conversion des données ===")
    convertir_format("train")
    convertir_format("val")
    convertir_format("test")
    calculer_et_sauvegarder_statistiques()

    print("\n=== Étape 2 : entraînement ===")
    lancer_entrainement(epochs=200, batch_size=32, use_stratified=True)

    print("\n=== Étape 3 : évaluation sur validation et test ===")
    val_ids = charger_split("val")
    test_ids = charger_split("test")
    model = charger_modele_entraine()

    y_pred_val, y_true_val, _ = evaluer_sur_split(model, val_ids, "val", save_results=True)
    y_pred_test, y_true_test, patch_ex = evaluer_sur_split(model, test_ids, "test", save_results=True)

    resultats_val = {"HTC_DC_Net": (y_pred_val, y_true_val)}
    resultats_test = {"HTC_DC_Net": (y_pred_test, y_true_test)}
    latences = {"HTC_DC_Net": mesurer_latence(lambda img: inferer_htc(model, img), patch_ex)}

    tableau_val = tableau_comparatif(resultats_val, latences_par_modele=latences)
    tableau_test = tableau_comparatif(resultats_test, latences_par_modele=latences)

    print("\n=== RÉSULTATS VALIDATION ===")
    print(tableau_val)
    print("\n=== RÉSULTATS TEST ===")
    print(tableau_test)

    os.makedirs("results", exist_ok=True)
    tableau_val.to_csv("results/metrics_htc_dc_net_val.csv")
    tableau_test.to_csv("results/metrics_htc_dc_net_test.csv")
    tableau_test.to_csv("results/metrics_htc_dc_net.csv")

    np.savez("results/predictions_htc_dc_net_val.npz", y_pred=y_pred_val, y_true=y_true_val)
    np.savez("results/predictions_htc_dc_net_test.npz", y_pred=y_pred_test, y_true=y_true_test)
    np.savez("results/predictions_htc_dc_net.npz", y_pred=y_pred_test, y_true=y_true_test)
    print("\nSauvegardé : metrics et predictions pour validation et test dans results/")