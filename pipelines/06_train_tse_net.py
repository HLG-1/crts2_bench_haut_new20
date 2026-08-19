"""
pipelines/06_train_tse_net.py

Même principe que HTC-DC Net (partage le même format image/ndsm), avec
l'étape supplémentaire compute_hbc.py requise par le README de TSE-Net
avant l'entraînement (points de coupure Head-Bin).
"""
import sys, os, subprocess, glob, math, shutil
from pathlib import Path

# Ajouter la racine et TSE-Net au PYTHONPATH
sys.path.append(".")
tse_dir_abs = os.path.abspath("third_party/tse-net")
if tse_dir_abs not in sys.path:
    sys.path.insert(0, tse_dir_abs)

import numpy as np
import rasterio
from rasterio.transform import from_origin
from PIL import Image
from skimage import io
import torch
import torch.nn.functional as F

from core.data_utils import charger_split, charger_patch
from core.metrics import calculer_metriques, tableau_comparatif, mesurer_latence

TSE_DIR = "third_party/tse-net"
DATA_HTC_DIR = "data/htc_dc_net_format"  # réutilise le même format de données
CONFIG_PATH = "configs/batiments_maroc.yaml"


# ── Étape 1 : Calcul des points de coupure Head-Bin (HBC) ──────────

def _discrete_bin(ndsm_list, lower_edge, upper_edge, order):
    count = np.zeros(10, dtype=np.int64)
    edges = np.linspace(lower_edge, upper_edge, 11)
    edges = np.round(edges, decimals=order).astype(np.float64)
    times = 10 ** order
    for ndsm_path in ndsm_list:
        ndsm = io.imread(ndsm_path).flatten().astype(np.float64)
        ndsm = np.nan_to_num(ndsm)
        ndsm = np.clip(ndsm, a_min=0, a_max=None)
        mask = (1000 * ndsm >= 1000 * lower_edge) & (1000 * ndsm < 1000 * upper_edge)
        if mask.sum() == 0:
            continue
        ndsm = np.floor((ndsm[mask] * 1000 - lower_edge * 1000) / 1000 * times).astype(np.int64)
        res = np.bincount(ndsm)
        count[:res.size] += res
    return count, edges


def _collect_values(ndsm_list, lower_edge, upper_edge):
    value_list = []
    for ndsm_path in ndsm_list:
        ndsm = io.imread(ndsm_path).flatten().astype(np.float64)
        ndsm = np.nan_to_num(ndsm)
        ndsm = np.clip(ndsm, a_min=0, a_max=None)
        mask = (ndsm * 1000 >= lower_edge * 1000) & (ndsm * 1000 < upper_edge * 1000)
        if mask.sum() == 0:
            continue
        value_list.append(ndsm[mask])
    return np.concatenate(value_list) if value_list else np.array([], dtype=np.float64)


def _get_quantile_value(ndsm_list, count, edges, cumcount, quantile_point, order):
    if np.all(count[1:] == 0):
        return edges[0]
    bin_ind = int(np.argmax(cumcount >= quantile_point))
    lower_edge = edges[bin_ind]
    upper_edge = edges[bin_ind + 1]
    num = count[bin_ind]
    if lower_edge == upper_edge:
        return lower_edge

    if num > 100000:
        new_count, new_edges = _discrete_bin(ndsm_list, lower_edge, upper_edge, order)
        new_cumcount = np.cumsum(new_count)
        new_quantile_point = quantile_point - (cumcount[bin_ind - 1] if bin_ind > 0 else 0)
        return _get_quantile_value(ndsm_list, new_count, new_edges, new_cumcount, new_quantile_point, order + 1)
    else:
        quantile_ind = quantile_point - (cumcount[bin_ind - 1] if bin_ind > 0 else 0)
        array = _collect_values(ndsm_list, lower_edge, upper_edge)
        if len(array) == 0:
            return lower_edge
        sorted_arr = np.sort(array)
        if (quantile_ind % 1) == 0:
            idx = max(0, min(len(sorted_arr) - 1, int(quantile_ind - 1)))
            return float(sorted_arr[idx])
        else:
            i1 = max(0, min(len(sorted_arr) - 1, math.floor(quantile_ind)))
            i2 = max(0, min(len(sorted_arr) - 1, math.ceil(quantile_ind)))
            return float((sorted_arr[i1] + sorted_arr[i2]) * 0.5)


def calculer_hbc(data_name: str = "batiments_maroc", proportion: str = "100"):
    """
    Étape obligatoire avant l'entraînement de TSE-Net :
    1. Prépare les fichiers de split dans third_party/tse-net/splits/{data_name}/
    2. Calcule les statistiques nDSM et les points de coupure Head-Bin quantiles (QVs)
    3. Sauvegarde bi_ndsm_stats_{proportion}.pickle, qvs_*.npy et samples_*.npy
    """
    print("=== Préparation des splits et calcul des QVs pour TSE-Net ===")
    split_dir = os.path.join(TSE_DIR, "splits", data_name)
    os.makedirs(split_dir, exist_ok=True)

    # Copie des splits
    shutil.copy("data/splits/train.txt", os.path.join(split_dir, f"train{proportion}.txt"))
    shutil.copy("data/splits/val.txt", os.path.join(split_dir, "val.txt"))
    shutil.copy("data/splits/test.txt", os.path.join(split_dir, "test.txt"))

    # Vérification des fichiers nDSM
    ndsm_dir = os.path.join(DATA_HTC_DIR, "ndsm")
    train_split_file = os.path.join(split_dir, f"train{proportion}.txt")
    with open(train_split_file, "r") as f:
        patch_ids = [line.strip() for line in f if line.strip()]

    ndsm_list = [os.path.join(ndsm_dir, f"{p}_AGL.tif") for p in patch_ids if os.path.exists(os.path.join(ndsm_dir, f"{p}_AGL.tif"))]

    if not ndsm_list:
        raise FileNotFoundError(
            f"Aucun fichier nDSM trouvé dans {ndsm_dir}. "
            "Exécutez d'abord 05_train_htc_dc_net.py (convertir_format) ou préparez les données."
        )

    # 1. Calcul des statistiques nDSM
    sum_x, sum_x2, sum_size = 0.0, 0.0, 0
    minh, maxh = float("inf"), float("-inf")
    for p in ndsm_list:
        arr = io.imread(p).flatten().astype(np.float64)
        arr = np.clip(np.nan_to_num(arr), 0, None)
        sum_x += float(arr.sum())
        sum_x2 += float((arr ** 2).sum())
        sum_size += arr.size
        minh = min(minh, float(arr.min()))
        maxh = max(maxh, float(arr.max()))

    minh = max(0.0, minh if minh != float("inf") else 0.0)
    maxh = max(1.0, maxh if maxh != float("-inf") else 45.0)
    mean = sum_x / max(1, sum_size)
    std = float(np.sqrt(max(0.0, sum_x2 / max(1, sum_size) - mean ** 2)))

    count = np.zeros(int(np.ceil(maxh)) + 1, dtype=np.int64)
    for p in ndsm_list:
        arr = io.imread(p).flatten().astype(np.float64)
        arr = np.clip(np.floor(np.nan_to_num(arr)), 0, maxh).astype(np.int64)
        b = np.bincount(arr, minlength=len(count))
        count[:len(b)] += b[:len(count)]

    # Sauvegarde des stats nDSM
    ndsm_stats_data = [mean, std, minh, maxh, count]
    torch.save(ndsm_stats_data, os.path.join(DATA_HTC_DIR, f"bi_ndsm_stats_{proportion}.pickle"))
    torch.save(ndsm_stats_data, os.path.join(DATA_HTC_DIR, "ndsm_stats.pickle"))
    print(f"Stats nDSM sauvegardées (maxh = {maxh:.2f} m)")

    # 2. Calcul des quantiles Head-Bin
    num_classes = 25
    quantile = 1.0 - 0.5 ** np.arange(1, num_classes + 1)
    quantile_points = np.floor(quantile * count.sum())
    sample_nums = np.diff(quantile_points)

    sample_filter = sample_nums > 100
    num_valid = sample_filter.sum()
    quantile_points = quantile_points[:num_valid + 1]
    sample_nums = sample_nums[sample_filter]

    edges = np.linspace(0, math.ceil(maxh), math.ceil(maxh) + 1)
    cumcount = np.cumsum(count)
    qvs = []
    for qp in quantile_points:
        qv = _get_quantile_value(ndsm_list, count, edges, cumcount, qp, order=1)
        qvs.append(qv)

    # Sauvegarde dans TSE_DIR (et répertoire courant)
    for target_dir in [TSE_DIR, "."]:
        np.save(os.path.join(target_dir, f"samples_{data_name}_{proportion}.npy"), np.array(sample_nums))
        np.save(os.path.join(target_dir, f"qvs_{data_name}_{proportion}.npy"), np.array(qvs))

    print(f"Points de coupure QVs générés : {len(qvs)} quantiles valides")


# ── Étape 2 : Lancement de l'entraînement ──────────────────────────

def lancer_entrainement(epochs: int = 100):
    """Lance l'entraînement semi-supervisé TSE-Net (100 époques)."""
    cmd = ["python", "train.py", "--config", CONFIG_PATH, "--max_epochs", str(epochs)]
    print("Commande :", " ".join(cmd))
    subprocess.run(cmd, cwd=TSE_DIR, check=True)


# ── Étape 3 : Chargement du modèle entraîné ('exam' network) ───────

def _trouver_checkpoint_tse(checkpoint_path: str | None = None) -> str:
    """Recherche le meilleur ou dernier checkpoint disponible pour TSE-Net."""
    if checkpoint_path and os.path.isfile(checkpoint_path):
        return checkpoint_path

    motifs = [
        "checkpoints/tse_net/**/checkpoint_best_rmse.pth.tar",
        "checkpoints/tse_net/**/checkpoint_last.pth.tar",
        "checkpoints/tse_net/**/*.pth.tar",
        f"{TSE_DIR}/chkpts/**/checkpoint_best_rmse.pth.tar",
        f"{TSE_DIR}/chkpts/**/checkpoint_last.pth.tar",
        f"{TSE_DIR}/chkpts/**/*.pth.tar",
    ]
    for motif in motifs:
        matches = sorted(glob.glob(motif, recursive=True))
        if matches:
            return matches[-1]

    raise FileNotFoundError(
        "Aucun checkpoint TSE-Net trouvé dans checkpoints/tse_net/. "
        "Lancez l'entraînement d'abord ou spécifiez le chemin du checkpoint."
    )


def charger_modele_entraine(checkpoint_path: str | None = None, device: str | torch.device | None = None):
    """
    Charge le modèle TSE-Net et extrait le réseau 'exam' (EMA du student)
    qui sert de référence pour l'inférence.
    """
    from tsenet import TSENet
    from utils.commons import load_yaml

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    chkpt_path = _trouver_checkpoint_tse(checkpoint_path)
    print(f"Chargement du checkpoint TSE-Net : {chkpt_path}")

    cfg_file = os.path.join(TSE_DIR, CONFIG_PATH)
    if os.path.isfile(cfg_file):
        cfgs = load_yaml(cfg_file)
    else:
        cfgs = {
            "model": "tsenet",
            "data_dir": DATA_HTC_DIR,
            "data_train": "splits/batiments_maroc/train100.txt",
            "data_val": "splits/batiments_maroc/val.txt",
            "data_test": "splits/batiments_maroc/test.txt",
            "device": str(device),
            "test": True,
        }

    cfgs["data_dir"] = DATA_HTC_DIR
    cfgs["device"] = device.type if isinstance(device, torch.device) else str(device)
    cfgs["test"] = True

    # S'assurer que le répertoire courant contient les fichiers QVs
    for f in glob.glob(os.path.join(TSE_DIR, "qvs_*.npy")) + glob.glob(os.path.join(TSE_DIR, "samples_*.npy")):
        dst = os.path.basename(f)
        if not os.path.exists(dst):
            shutil.copy(f, dst)

    model = TSENet(cfgs)
    chkpt = torch.load(chkpt_path, map_location="cpu")
    state_dict = chkpt["state_dict"] if "state_dict" in chkpt else chkpt
    model.load_state_dict(state_dict)
    model.to(device).eval()

    # Charger les statistiques de normalisation
    img_stats_file = os.path.join(DATA_HTC_DIR, "image_stats.pickle")
    if os.path.isfile(img_stats_file):
        mean, std = torch.load(img_stats_file)
    else:
        mean, std = [123.675, 116.28, 103.53], [58.395, 57.12, 57.375]

    model.mean = torch.tensor(mean, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    model.std = torch.tensor(std, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    model.device = device

    return model


# ── Étape 4 : Inférence avec le réseau 'exam' ──────────────────────

def inferer_tse(model, img_rgb: np.ndarray) -> np.ndarray:
    """
    Inférence TSE-Net : passe l'image normalisée dans le sous-réseau 'exam'
    (student EMA) et retourne la carte de hauteur estimée (H, W) en mètres.
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
        # TSE-Net utilise le sous-modèle 'exam' à l'inférence
        if hasattr(model, "exam"):
            pred = model.exam(img_t)
        else:
            pred = model(img_t)

        if pred.shape[-2:] != (h_orig, w_orig):
            pred = F.interpolate(pred, size=(h_orig, w_orig), mode="bilinear", align_corners=False)

        pred_map = pred.squeeze().cpu().numpy().astype(np.float32)

    return pred_map


from tqdm import tqdm

def evaluer_sur_split(model, split_ids: list, split_name: str = "split"):
    y_pred, y_true = [], []
    patch_exemple = None
    print(f"\nÉvaluation sur le jeu {split_name.upper()} ({len(split_ids)} patches)...")
    for patch_id in tqdm(split_ids, desc=f"TSE-Net Éval ({split_name.upper()})", unit="patch"):
        img, gt, mask = charger_patch(patch_id)
        if mask.sum() == 0:
            continue
        if patch_exemple is None:
            patch_exemple = img
        pred_map = inferer_tse(model, img)
        y_pred.append(float(np.median(pred_map[mask == 1])))
        y_true.append(float(np.median(gt[mask == 1])))
    return np.array(y_pred), np.array(y_true), patch_exemple


if __name__ == "__main__":
    print("=== Étape 1 : points de coupure HB (obligatoire avant training) ===")
    calculer_hbc()

    print("\n=== Étape 2 : entraînement (teacher + student, semi-supervisé) ===")
    lancer_entrainement()

    print("\n=== Étape 3 : évaluation sur validation et test (réseau 'exam') ===")
    val_ids = charger_split("val")
    test_ids = charger_split("test")
    model = charger_modele_entraine()

    y_pred_val, y_true_val, _ = evaluer_sur_split(model, val_ids, "val")
    y_pred_test, y_true_test, patch_ex = evaluer_sur_split(model, test_ids, "test")

    resultats_val = {"TSE_Net": (y_pred_val, y_true_val)}
    resultats_test = {"TSE_Net": (y_pred_test, y_true_test)}
    latences = {"TSE_Net": mesurer_latence(lambda img: inferer_tse(model, img), patch_ex)}

    tableau_val = tableau_comparatif(resultats_val, latences_par_modele=latences)
    tableau_test = tableau_comparatif(resultats_test, latences_par_modele=latences)

    print("\n=== RÉSULTATS VALIDATION ===")
    print(tableau_val)
    print("\n=== RÉSULTATS TEST ===")
    print(tableau_test)

    os.makedirs("results", exist_ok=True)
    tableau_val.to_csv("results/metrics_tse_net_val.csv")
    tableau_test.to_csv("results/metrics_tse_net_test.csv")
    tableau_test.to_csv("results/metrics_tse_net.csv")

    np.savez("results/predictions_tse_net_val.npz", y_pred=y_pred_val, y_true=y_true_val)
    np.savez("results/predictions_tse_net_test.npz", y_pred=y_pred_test, y_true=y_true_test)
    np.savez("results/predictions_tse_net.npz", y_pred=y_pred_test, y_true=y_true_test)
    print("\nSauvegardé : metrics et predictions pour validation et test dans results/")