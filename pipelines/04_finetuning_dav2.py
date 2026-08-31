"""
Orchestration du fine-tuning Depth Anything V2 : prépare les splits attendus
par le repo officiel, lance l'entraînement (via subprocess/torchrun avec early stopping),
puis évalue le modèle fine-tuné sur zone3.
"""
import sys, os, subprocess, shutil
sys.path.append(".")
import numpy as np
import torch
from tqdm import tqdm

from core.data_utils import charger_split, charger_patch
from core.metrics import tableau_comparatif, mesurer_latence

DAV2_DIR = "third_party/Depth-Anything-V2/metric_depth"
DEFAULT_MAX_DEPTH = 41.0

def _trouver_checkpoint_pretraine() -> str:
    possibles = [
        "checkpoints/depth_anything_v2/depth_anything_v2_vitl.pth",
        "checkpoints/depth_anything_v2/native/depth_anything_v2_vitl.pth",
        "checkpoints/depth_anything_v2_vitl.pth",
    ]
    for p in possibles:
        if os.path.exists(p):
            return p
    return possibles[0]

CHECKPOINT_PRETRAINE = _trouver_checkpoint_pretraine()
SAVE_PATH = "exp/batiments_maroc"


def preparer_splits(use_stratified: bool = True):
    """
    Copie les splits au format attendu par le repo officiel.
    
    Args:
        use_stratified: Si True, utilise les nouveaux splits stratifiés (results/stratified_split/splits_new/)
                        Sinon, utilise les splits originaux (data/splits/)
    """
    if use_stratified:
        source_dir = "results/stratified_split/splits_new"
        print("Utilisation des NOUVEAUX splits stratifiés avec contrainte spatiale")
    else:
        source_dir = "data/splits"
        print("Utilisation des splits ORIGINAUX")
    
    dest = f"{DAV2_DIR}/dataset/splits/batiments_maroc"
    os.makedirs(dest, exist_ok=True)
    
    shutil.copy(f"{source_dir}/train.txt", f"{dest}/train.txt")
    shutil.copy(f"{source_dir}/val.txt", f"{dest}/val.txt")
    shutil.copy(f"{source_dir}/test.txt", f"{dest}/test.txt")
    
    print(f"Splits copiés depuis {source_dir} vers {dest}")
    
    # Afficher les statistiques des splits
    for split in ["train", "val", "test"]:
        with open(f"{source_dir}/{split}.txt") as f:
            n_patches = len([line.strip() for line in f if line.strip()])
        print(f"  - {split}: {n_patches} patches")


def lancer_entrainement(
    epochs: int = 100,
    bs: int = 4,
    max_depth: float = DEFAULT_MAX_DEPTH,
    lr: float = 5e-6,
    patience: int = 8,
    metric_early_stop: str = "abs_rel",
    encoder: str = "vitl",
    img_size: int = 518,
):
    """
    Lance le fine-tuning avec support de l'early stopping (--patience et --metric-early-stop).
    """
    # Définir la variable d'environnement pour éviter la fragmentation de mémoire
    env = os.environ.copy()
    env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    cmd = [
        "torchrun", "--nproc_per_node=1", "--master_port=29500", "train.py",
        "--encoder", encoder,
        "--dataset", "batiments_maroc",
        "--img-size", str(img_size),
        "--min-depth", "0",
        "--max-depth", str(max_depth),
        "--epochs", str(epochs),
        "--bs", str(bs),
        "--lr", str(lr),
        "--patience", str(patience),
        "--metric-early-stop", str(metric_early_stop),
        "--pretrained-from", os.path.relpath(CHECKPOINT_PRETRAINE, DAV2_DIR),
        "--save-path", SAVE_PATH,
        "--patches-dir", os.path.relpath("data/patches", DAV2_DIR),
    ]
    print("Commande :", " ".join(cmd))
    subprocess.run(cmd, cwd=DAV2_DIR, check=True, env=env)


def charger_modele_finetune(max_depth: float = DEFAULT_MAX_DEPTH, checkpoint_path: str | None = None):
    """
    Charge le modèle fine-tuné en priorisant le checkpoint 'best.pth' issu de l'early stopping
    (avec repli sur 'latest.pth').
    """
    sys.path.insert(0, DAV2_DIR)
    from depth_anything_v2.dpt import DepthAnythingV2

    model_configs = {"vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]}}
    model = DepthAnythingV2(**{**model_configs["vitl"], "max_depth": max_depth})

    if checkpoint_path is None:
        best_ckpt = f"{DAV2_DIR}/{SAVE_PATH}/best.pth"
        latest_ckpt = f"{DAV2_DIR}/{SAVE_PATH}/latest.pth"
        if os.path.exists(best_ckpt):
            checkpoint_path = best_ckpt
            print(f"Chargement du meilleur checkpoint (early stopping) : {best_ckpt}")
        elif os.path.exists(latest_ckpt):
            checkpoint_path = latest_ckpt
            print(f"Chargement du dernier checkpoint : {latest_ckpt}")
        else:
            raise FileNotFoundError(f"Aucun checkpoint trouvé dans {DAV2_DIR}/{SAVE_PATH}/")
    else:
        print(f"Chargement du checkpoint : {checkpoint_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model


def inferer_finetune(model, img_rgb: np.ndarray) -> np.ndarray:
    """
    Inférence du modèle fine-tuné avec redimensionnement automatique vers la taille d'origine.
    """
    import torchvision.transforms as T
    from PIL import Image

    h_orig, w_orig = img_rgb.shape[:2]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = T.Compose([
        T.Resize((518, 518), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_t = transform(Image.fromarray(img_rgb)).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(img_t)
        if pred.shape[-2:] != (h_orig, w_orig):
            pred = torch.nn.functional.interpolate(
                pred[:, None], size=(h_orig, w_orig), mode="bilinear", align_corners=True
            )[:, 0]
    return pred[0].cpu().numpy()


def evaluer_sur_split(model, split_ids: list, split_name: str = "split"):
    y_pred, y_true = [], []
    patch_exemple = None
    print(f"\nÉvaluation sur le jeu {split_name.upper()} ({len(split_ids)} patches)...")
    for patch_id in tqdm(split_ids, desc=f"DAV2 Finetune Éval ({split_name.upper()})", unit="patch"):
        img, gt, mask = charger_patch(patch_id)
        if mask.sum() == 0:
            continue
        if patch_exemple is None:
            patch_exemple = img
        pred_map = inferer_finetune(model, img)
        # Pas de calibration nécessaire : le modèle est fine-tuné directement en mètres
        y_pred.append(float(np.median(pred_map[mask == 1])))
        y_true.append(float(np.median(gt[mask == 1])))
    return np.array(y_pred), np.array(y_true), patch_exemple


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tuning Depth Anything V2 avec choix des splits")
    parser.add_argument("--use-stratified", action="store_true", default=True,
                       help="Utiliser les nouveaux splits stratifiés (défaut: True)")
    parser.add_argument("--use-original", action="store_true", default=False,
                       help="Utiliser les splits originaux au lieu des stratifiés")
    parser.add_argument("--epochs", type=int, default=100, help="Nombre d'époques (défaut: 100)")
    parser.add_argument("--patience", type=int, default=8, help="Patience pour early stopping (défaut: 8)")
    parser.add_argument("--batch-size", type=int, default=4, help="Taille de batch (défaut: 4, recommandé 4 avec GPU partagé)")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate (défaut: 5e-6)")
    parser.add_argument("--encoder", type=str, default="vitl", choices=["vits", "vitb", "vitl"], 
                       help="Encodeur ViT (défaut: vitl)")
    parser.add_argument("--img-size", type=int, default=392, help="Taille d'image (défaut: 392 pour GPU partagé, 518 normal, doit être multiple de 14 pour ViT-L)")
    parser.add_argument("--gradient-accumulation", type=int, default=1, help="Gradient accumulation steps (défaut: 1)")
    
    args = parser.parse_args()
    
    # Déterminer quel type de splits utiliser
    use_stratified = args.use_stratified and not args.use_original
    
    print("=== Étape 1 : préparation des splits ===")
    preparer_splits(use_stratified=use_stratified)

    print("\n=== Étape 2 : entraînement avec early stopping ===")
    lancer_entrainement(epochs=args.epochs, patience=args.patience, 
                       bs=args.batch_size, lr=args.lr, metric_early_stop="abs_rel",
                       encoder=args.encoder, img_size=args.img_size)

    print("\n=== Étape 3 : évaluation sur validation et test ===")
    # Charger les splits depuis le bon répertoire
    if use_stratified:
        val_ids = charger_split("val", splits_dir="results/stratified_split/splits_new")
        test_ids = charger_split("test", splits_dir="results/stratified_split/splits_new")
    else:
        val_ids = charger_split("val")
        test_ids = charger_split("test")
    
    model = charger_modele_finetune()

    y_pred_val, y_true_val, _ = evaluer_sur_split(model, val_ids, "val")
    y_pred_test, y_true_test, patch_ex = evaluer_sur_split(model, test_ids, "test")

    resultats_val = {"DAV2_finetune": (y_pred_val, y_true_val)}
    resultats_test = {"DAV2_finetune": (y_pred_test, y_true_test)}
    latences = {"DAV2_finetune": mesurer_latence(lambda img: inferer_finetune(model, img), patch_ex)}

    tableau_val = tableau_comparatif(resultats_val, latences_par_modele=latences)
    tableau_test = tableau_comparatif(resultats_test, latences_par_modele=latences)

    print("\n=== RÉSULTATS VALIDATION ===")
    print(tableau_val)
    print("\n=== RÉSULTATS TEST ===")
    print(tableau_test)

    # Sauvegarder avec un nom de fichier indiquant le type de split utilisé
    split_suffix = "_stratified" if use_stratified else "_original"
    os.makedirs("results", exist_ok=True)
    tableau_val.to_csv(f"results/metrics_dav2_finetune_val{split_suffix}.csv")
    tableau_test.to_csv(f"results/metrics_dav2_finetune_test{split_suffix}.csv")
    tableau_test.to_csv(f"results/metrics_dav2_finetune{split_suffix}.csv")

    np.savez(f"results/predictions_dav2_finetune_val{split_suffix}.npz", y_pred=y_pred_val, y_true=y_true_val)
    np.savez(f"results/predictions_dav2_finetune_test{split_suffix}.npz", y_pred=y_pred_test, y_true=y_true_test)
    np.savez(f"results/predictions_dav2_finetune{split_suffix}.npz", y_pred=y_pred_test, y_true=y_true_test)
    print(f"\nSauvegardé : metrics et predictions pour validation et test dans results/ (suffixe: {split_suffix})")