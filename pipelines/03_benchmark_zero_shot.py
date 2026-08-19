"""
pipelines/03_benchmark_zero_shot.py

Benchmark zero-shot : DepthPro + Depth Anything V2, sans entraînement,
suivi d'une calibration linéaire simple (profondeur relative -> hauteur).

Les deux modèles sont chargés depuis leurs checkpoints NATIFS locaux
(checkpoints/depthpro/depth_pro.pt et checkpoints/depth_anything_v2/native/
depth_anything_v2_vitl.pth), jamais via un téléchargement automatique
implicite - pour garantir que le DAV2 zero-shot utilise exactement les
mêmes poids de départ que le DAV2 fine-tuné (pipelines/04_finetuning_dav2.py),
rendant la comparaison "zero-shot vs fine-tuné" valide.
"""
import sys, os
sys.path.append(".")
depth_pro_src = os.path.abspath("third_party/ml-depth-pro/src")
if depth_pro_src not in sys.path:
    sys.path.insert(0, depth_pro_src)

dav2_src = os.path.abspath("third_party/Depth-Anything-V2")
if dav2_src not in sys.path:
    sys.path.insert(0, dav2_src)

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

from core.data_utils import charger_split, charger_patch
from core.metrics import tableau_comparatif, mesurer_latence


# ═══════════════════════════════════════════════════════════════
# DepthPro
# ═══════════════════════════════════════════════════════════════

def _trouver_poids_depthpro(checkpoint_path: str | None = None) -> str:
    """Recherche les poids pré-entraînés locaux de DepthPro. Aucun
    téléchargement automatique : si absent, utilisez explicitement
    `bash get_pretrained_models.sh` dans third_party/ml-depth-pro/."""
    chemins_possibles = [
        checkpoint_path,
        "checkpoints/depthpro/depth_pro.pt",
        "checkpoints/depth_pro.pt",
        "third_party/ml-depth-pro/checkpoints/depth_pro.pt",
    ]
    for p in chemins_possibles:
        if p and os.path.isfile(p):
            print(f"Checkpoint DepthPro trouvé : {p}")
            return p

    raise FileNotFoundError(
        "Checkpoint DepthPro introuvable. Lancez d'abord :\n"
        "  cd third_party/ml-depth-pro && bash get_pretrained_models.sh\n"
        f"Chemins vérifiés : {[p for p in chemins_possibles if p]}"
    )


def charger_modele_depthpro(checkpoint_path: str | None = None, device: torch.device | str | None = None):
    from depth_pro.depth_pro import DepthProConfig, create_model_and_transforms

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str): 
        device = torch.device(device)

    ckpt_path = _trouver_poids_depthpro(checkpoint_path)
    config = DepthProConfig(
        patch_encoder_preset="dinov2l16_384",
        image_encoder_preset="dinov2l16_384",
        checkpoint_uri=ckpt_path,
        decoder_features=256,
        use_fov_head=True,
        fov_encoder_preset="dinov2l16_384",
    )

    precision = torch.half if device.type == "cuda" else torch.float32
    model, transform = create_model_and_transforms(
        config=config, device=device, precision=precision,
    )
    model.eval()
    model.transform = transform
    model.device = device
    return model


def inferer_depthpro(model, img_rgb: np.ndarray) -> np.ndarray:
    transform = getattr(model, "transform", None)
    device = getattr(model, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    if transform is not None:
        img_t = transform(img_rgb)
    else:
        img_t = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        img_t = (img_t - 0.5) / 0.5
        img_t = img_t.to(device)

    with torch.no_grad():
        prediction = model.infer(img_t)

    depth = prediction["depth"]
    if isinstance(depth, torch.Tensor):
        depth = depth.squeeze().detach().cpu().numpy()

    return depth.astype(np.float32)


# ═══════════════════════════════════════════════════════════════
# Depth Anything V2 — chargé depuis le checkpoint NATIF local,
# le même que celui utilisé pour le fine-tuning (04_finetuning_dav2.py)
# ═══════════════════════════════════════════════════════════════

def _trouver_poids_dav2(checkpoint_path: str | None = None) -> str:
    chemins_possibles = [
        checkpoint_path,
        "checkpoints/depth_anything_v2/depth_anything_v2_vitl.pth",
        "checkpoints/depth_anything_v2/native/depth_anything_v2_vitl.pth",
        "checkpoints/depth_anything_v2_vitl.pth",
        "third_party/Depth-Anything-V2/checkpoints/depth_anything_v2_vitl.pth",
    ]
    for p in chemins_possibles:
        if p:
            p_exp = os.path.expanduser(p)
            if os.path.isfile(p_exp):
                print(f"Checkpoint Depth Anything V2 trouvé : {p_exp}")
                return p_exp

    raise FileNotFoundError(
        "Checkpoint Depth Anything V2 (natif, .pth) introuvable.\n"
        "Téléchargez-le depuis :\n"
        "  https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth\n"
        f"Chemins vérifiés : {[p for p in chemins_possibles if p]}"
    )


def charger_modele_dav2(checkpoint_path: str | None = None, encoder: str = "vitl",
                          max_depth: float = 40.0, device: str | None = None):
    """
    Charge Depth Anything V2 depuis le checkpoint natif local — les mêmes
    poids de départ que ceux utilisés pour le fine-tuning, garantissant
    une comparaison zero-shot vs fine-tuné valide.
    """
    from depth_anything_v2.dpt import DepthAnythingV2

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
    }
    ckpt_path = _trouver_poids_dav2(checkpoint_path)

    model = DepthAnythingV2(**model_configs[encoder])
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model = model.to(device).eval()
    model.device = device
    return model


def inferer_dav2(model, img_rgb: np.ndarray) -> np.ndarray:
    """Inférence directe avec le modèle natif (pas de pipeline transformers)."""
    import torchvision.transforms as T

    h_orig, w_orig = img_rgb.shape[:2]
    device = getattr(model, "device", "cuda" if torch.cuda.is_available() else "cpu")

    transform = T.Compose([
        T.Resize((518, 518), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_t = transform(Image.fromarray(img_rgb.astype(np.uint8))).unsqueeze(0).to(device)

    with torch.no_grad():
        depth = model(img_t)
        if depth.shape[-2:] != (h_orig, w_orig):
            depth = torch.nn.functional.interpolate(
                depth[:, None], size=(h_orig, w_orig), mode="bilinear", align_corners=True
            )[:, 0]

    return depth.squeeze().cpu().numpy().astype(np.float32)


# ═══════════════════════════════════════════════════════════════
# Calibration + évaluation (identique pour les deux modèles)
# ═══════════════════════════════════════════════════════════════

def calibrer(valeurs_brutes_train: list, hauteurs_reelles_train: list) -> LinearRegression:
    reg = LinearRegression()
    reg.fit(np.array(valeurs_brutes_train).reshape(-1, 1), hauteurs_reelles_train)
    return reg


def evaluer_modele(fonction_inference, train_ids: list, test_ids: list, modele_name: str = "modele"):
    """Calibre sur train, évalue sur test. Retourne (y_pred, y_true, patch_exemple)."""
    brutes_train, reelles_train = [], []
    print(f"\nCalibration sur {len(train_ids)} patches d'entraînement...")
    for patch_id in tqdm(train_ids, desc=f"{modele_name} Calibration", unit="patch"):
        img, gt, mask = charger_patch(patch_id)
        if mask.sum() == 0:
            continue
        pred_map = fonction_inference(img)
        brutes_train.append(float(np.median(pred_map[mask == 1])))
        reelles_train.append(float(np.median(gt[mask == 1])))

    reg = calibrer(brutes_train, reelles_train)

    y_pred, y_true = [], []
    patch_exemple = None
    print(f"\nÉvaluation sur {len(test_ids)} patches de test...")
    for patch_id in tqdm(test_ids, desc=f"{modele_name} Évaluation", unit="patch"):
        img, gt, mask = charger_patch(patch_id)
        if mask.sum() == 0:
            continue
        if patch_exemple is None:
            patch_exemple = img
        pred_map = fonction_inference(img)
        valeur_brute = float(np.median(pred_map[mask == 1]))
        valeur_calibree = float(reg.predict([[valeur_brute]])[0])
        y_pred.append(valeur_calibree)
        y_true.append(float(np.median(gt[mask == 1])))

    return np.array(y_pred), np.array(y_true), patch_exemple


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark zero-shot DepthPro + Depth Anything V2")
    parser.add_argument("--use-stratified", action="store_true", default=True,
                       help="Utiliser les nouveaux splits stratifiés (défaut: True)")
    parser.add_argument("--use-original", action="store_true", default=False,
                       help="Utiliser les splits originaux au lieu des stratifiés")
    
    args = parser.parse_args()
    
    # Déterminer quel type de splits utiliser
    use_stratified = args.use_stratified and not args.use_original
    
    if use_stratified:
        train_ids = charger_split("train", splits_dir="results/stratified_split/splits_new")
        test_ids = charger_split("test", splits_dir="results/stratified_split/splits_new")
        print("Utilisation des NOUVEAUX splits stratifiés avec contrainte spatiale")
    else:
        train_ids = charger_split("train")
        test_ids = charger_split("test")
        print("Utilisation des splits ORIGINAUX")
    
    print(f"Train : {len(train_ids)} patches | Test : {len(test_ids)} patches")

    # Définir le suffixe pour les fichiers de résultats
    split_suffix = "_stratified" if use_stratified else "_original"

    resultats, latences = {}, {}
    os.makedirs("results", exist_ok=True)

    print("\n=== DepthPro ===")
    model_depthpro = charger_modele_depthpro()
    f_depthpro = lambda img: inferer_depthpro(model_depthpro, img)
    y_pred_dp, y_true_dp, patch_ex = evaluer_modele(f_depthpro, train_ids, test_ids, "DepthPro")
    resultats["DepthPro"] = (y_pred_dp, y_true_dp)
    latences["DepthPro"] = mesurer_latence(f_depthpro, patch_ex)
    np.savez(f"results/predictions_depthpro{split_suffix}.npz", y_pred=y_pred_dp, y_true=y_true_dp)

    print("\n=== Depth Anything V2 (zero-shot, checkpoint natif) ===")
    model_dav2 = charger_modele_dav2()
    f_dav2 = lambda img: inferer_dav2(model_dav2, img)
    y_pred_dav2, y_true_dav2, patch_ex = evaluer_modele(f_dav2, train_ids, test_ids, "DAV2_ZS")
    resultats["DAV2_zeroshot"] = (y_pred_dav2, y_true_dav2)
    latences["DAV2_zeroshot"] = mesurer_latence(f_dav2, patch_ex)
    np.savez(f"results/predictions_dav2_zeroshot{split_suffix}.npz", y_pred=y_pred_dav2, y_true=y_true_dav2)

    tableau = tableau_comparatif(resultats, latences_par_modele=latences)
    print("\n", tableau)

    # Sauvegarder avec un nom de fichier indiquant le type de split utilisé
    tableau.to_csv(f"results/metrics_zero_shot{split_suffix}.csv")
    print(f"\nSauvegardé : results/metrics_zero_shot{split_suffix}.csv, "
          f"results/predictions_depthpro{split_suffix}.npz, results/predictions_dav2_zeroshot{split_suffix}.npz")