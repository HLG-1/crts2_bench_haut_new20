"""
pipelines/04_finetuning_dav2.py

Orchestration du fine-tuning Depth Anything V2 : prépare les splits attendus
par le repo officiel, lance l'entraînement (via subprocess/torchrun avec early stopping),
puis évalue le modèle fine-tuné sur zone3.
"""
import sys, os, subprocess, shutil
sys.path.append(".")
import numpy as np
import torch
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Backend non-interactif pour sauvegarder les plots sans affichage
import matplotlib.pyplot as plt
import pandas as pd

from core.data_utils import charger_split, charger_patch
from core.metrics import calculer_metriques, tableau_comparatif, mesurer_latence

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
    epochs: int = 200,
    bs: int = 4,
    max_depth: float = DEFAULT_MAX_DEPTH,
    lr: float = 5e-6,
    patience: int = 20,
    metric_early_stop: str = "abs_rel",
    encoder: str = "vitl",
    img_size: int = 392,
    gradient_accumulation: int = 2,
    use_multi_loss: bool = True,
    label_smoothing: float = 0.1,
    silog_weight: float = 1.0,
    l1_weight: float = 0.5,
    gradient_weight: float = 0.3,
    scale_weight: float = 0.2,
    dropout_rate: float = 0.1,
    building_lighting: bool = True,
    custom_port: int = 29500,
):
    """
    Lance le fine-tuning avec support de l'early stopping (--patience et --metric-early-stop).
    """
    # Définir la variable d'environnement pour éviter la fragmentation de mémoire
    env = os.environ.copy()
    env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    cmd = [
        "torchrun", "--nproc_per_node=1", "--master_port", str(custom_port), "train.py",
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
        "--gradient-accumulation", str(gradient_accumulation),
    ]
    
    # Ajouter les paramètres multi-loss si activé
    if use_multi_loss:
        cmd.append("--use-multi-loss")
        cmd.append(f"--label-smoothing={label_smoothing}")
        cmd.append(f"--silog-weight={silog_weight}")
        cmd.append(f"--l1-weight={l1_weight}")
        cmd.append(f"--gradient-weight={gradient_weight}")
        cmd.append(f"--scale-weight={scale_weight}")
    
    # Ajouter le taux de dropout
    cmd.append(f"--dropout-rate={dropout_rate}")
    
    # Ajouter le flag pour les augmentations spécifiques bâtiments
    if building_lighting:
        cmd.append("--building-lighting")
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


def lancer_multi_lr_test(
    epochs: int = 200,
    bs: int = 4,
    max_depth: float = DEFAULT_MAX_DEPTH,
    patience: int = 20,
    metric_early_stop: str = "abs_rel",
    encoder: str = "vitl",
    img_size: int = 392,
    gradient_accumulation: int = 2,
    lr_list: list = [1e-5, 2e-5, 1e-4],
    use_multi_loss: bool = True,
    label_smoothing: float = 0.1,
    silog_weight: float = 1.0,
    l1_weight: float = 0.5,
    gradient_weight: float = 0.3,
    scale_weight: float = 0.2,
    dropout_rate: float = 0.1,
    building_lighting: bool = True,
    base_port: int = 29500,
):
    """
    Lance plusieurs entraînements avec différents learning rates et compare les résultats.
    """
    global SAVE_PATH
    resultats_complets = {}
    
    for idx, lr in enumerate(lr_list):
        print(f"\n{'='*60}")
        print(f"TESTING LR: {lr}")
        print(f"{'='*60}")
        
        # Modifier le save path pour inclure le LR
        original_save_path = SAVE_PATH
        SAVE_PATH = f"exp/batiments_maroc_lr_{lr}"
        
        # Utiliser un port différent pour chaque LR
        current_port = base_port + idx
        
        try:
            lancer_entrainement(
                epochs=epochs, 
                bs=bs, 
                max_depth=max_depth,
                lr=lr, 
                patience=patience, 
                metric_early_stop=metric_early_stop,
                encoder=encoder, 
                img_size=img_size,
                gradient_accumulation=gradient_accumulation,
                use_multi_loss=use_multi_loss,
                label_smoothing=label_smoothing,
                silog_weight=silog_weight,
                l1_weight=l1_weight,
                gradient_weight=gradient_weight,
                scale_weight=scale_weight,
                dropout_rate=dropout_rate,
                building_lighting=building_lighting,
                custom_port=current_port,
            )
            
            # Évaluer le modèle
            model = charger_modele_finetune(max_depth=max_depth)
            
            # Charger les splits (on suppose stratified)
            val_ids = charger_split("val", splits_dir="results/stratified_split/splits_new")
            test_ids = charger_split("test", splits_dir="results/stratified_split/splits_new")
            
            y_pred_val, y_true_val, _ = evaluer_sur_split(model, val_ids, "val")
            y_pred_test, y_true_test, patch_ex = evaluer_sur_split(model, test_ids, "test")
            
            resultats_complets[f"lr_{lr}"] = {
                "val": (y_pred_val, y_true_val),
                "test": (y_pred_test, y_true_test),
                "patch_exemple": patch_ex
            }
            
            print(f"✓ LR {lr} terminé avec succès")
            
        except Exception as e:
            print(f"✗ LR {lr} échoué: {e}")
            resultats_complets[f"lr_{lr}"] = None
        
        finally:
            SAVE_PATH = original_save_path
    
    return resultats_complets


def comparer_resultats_multi_lr(resultats_complets: dict, split_suffix: str = "_stratified"):
    """
    Compare les résultats de plusieurs LR tests et génère un tableau comparatif et des diagrammes.
    """
    resultats_val = {}
    resultats_test = {}
    latences = {}
    
    successful_runs = 0
    
    for lr_key, data in resultats_complets.items():
        if data is None:
            print(f"⚠ Skipping {lr_key} - no results available")
            continue
            
        model_name = f"DAV2_finetune_{lr_key}"
        resultats_val[model_name] = data["val"]
        resultats_test[model_name] = data["test"]
        
        # Mesurer latence
        latences[model_name] = mesurer_latence(
            lambda img, data=data: inferer_finetune(
                charger_modele_finetune(checkpoint_path=f"{DAV2_DIR}/exp/batiments_maroc_{lr_key}/best.pth"), 
                img
            ), 
            data["patch_exemple"]
        )
        successful_runs += 1
    
    if successful_runs == 0:
        print("❌ Aucun entraînement n'a réussi. Impossible de générer les comparaisons.")
        return None, None
    
    # Générer les tableaux comparatifs
    tableau_val = tableau_comparatif(resultats_val, latences_par_modele=latences)
    tableau_test = tableau_comparatif(resultats_test, latences_par_modele=latences)
    
    print("\n" + "="*60)
    print("COMPARAISON VALIDATION (DIFFÉRENTS LR)")
    print("="*60)
    print(tableau_val)
    
    print("\n" + "="*60)
    print("COMPARAISON TEST (DIFFÉRENTS LR)")
    print("="*60)
    print(tableau_test)
    
    # Sauvegarder
    os.makedirs("results", exist_ok=True)
    tableau_val.to_csv(f"results/metrics_dav2_multi_lr_val{split_suffix}.csv")
    tableau_test.to_csv(f"results/metrics_dav2_multi_lr_test{split_suffix}.csv")
    
    # Sauvegarder les prédictions pour chaque LR
    for lr_key, data in resultats_complets.items():
        if data is None:
            continue
        np.savez(f"results/predictions_dav2_{lr_key}_val{split_suffix}.npz", 
                y_pred=data["val"][0], y_true=data["val"][1])
        np.savez(f"results/predictions_dav2_{lr_key}_test{split_suffix}.npz", 
                y_pred=data["test"][0], y_true=data["test"][1])
    
    # Générer les diagrammes de comparaison seulement si on a des résultats
    if successful_runs > 0:
        generer_diagrammes_comparaison(tableau_val, tableau_test, resultats_complets, split_suffix)
    
    print(f"\n✓ Sauvegardé: résultats multi-LR et diagrammes dans results/")
    return tableau_val, tableau_test


def generer_diagrammes_comparaison(tableau_val: pd.DataFrame, tableau_test: pd.DataFrame, 
                                   resultats_complets: dict, split_suffix: str):
    """
    Génère des diagrammes visuels pour comparer les performances entre différents LR.
    """
    os.makedirs("results/plots", exist_ok=True)
    
    # Extraire les noms de modèles (sans le préfixe DAV2_finetune_)
    lr_names = [name.replace("DAV2_finetune_", "") for name in tableau_val.index if "lr_" in name]
    
    # Diagramme 1: Comparaison des métriques principales (Test)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Comparaison des Learning Rates - Dataset Test', fontsize=16, fontweight='bold')
    
    # Métriques principales
    metrics_to_plot = ['abs_rel', 'rmse', 'delta1', 'latence_ms']
    metric_labels = ['Abs Rel', 'RMSE (m)', 'δ1 (%)', 'Latence (ms)']
    
    for idx, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
        ax = axes[idx // 2, idx % 2]
        
        if metric in tableau_test.columns:
            values = tableau_test.loc[[f"DAV2_finetune_{lr}" for lr in lr_names], metric].values
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
            bars = ax.bar(lr_names, values, color=colors[:len(lr_names)])
            
            # Ajouter les valeurs sur les barres
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.4f}' if metric != 'delta1' else f'{height:.2f}',
                       ha='center', va='bottom', fontsize=10)
            
            ax.set_ylabel(label, fontsize=12)
            ax.set_xlabel('Learning Rate', fontsize=12)
            ax.set_title(f'{label} par Learning Rate', fontsize=11, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'results/plots/comparaison_lr_test{split_suffix}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Diagramme 2: Comparaison Validation vs Test
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Validation vs Test par Learning Rate', fontsize=16, fontweight='bold')
    
    for idx, metric in enumerate(['abs_rel', 'rmse']):
        ax = axes[idx]
        
        val_values = tableau_val.loc[[f"DAV2_finetune_{lr}" for lr in lr_names], metric].values
        test_values = tableau_test.loc[[f"DAV2_finetune_{lr}" for lr in lr_names], metric].values
        
        x = np.arange(len(lr_names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, val_values, width, label='Validation', color='#4ECDC4')
        bars2 = ax.bar(x + width/2, test_values, width, label='Test', color='#FF6B6B')
        
        # Ajouter les valeurs
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.4f}',
                       ha='center', va='bottom', fontsize=9)
        
        ax.set_ylabel(metric.upper(), fontsize=12)
        ax.set_xlabel('Learning Rate', fontsize=12)
        ax.set_title(f'{metric.upper()}: Validation vs Test', fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(lr_names)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'results/plots/comparaison_val_test{split_suffix}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Diagramme 3: Scatter plots Pred vs True pour chaque LR
    fig, axes = plt.subplots(1, len(lr_names), figsize=(6*len(lr_names), 5))
    if len(lr_names) == 1:
        axes = [axes]
    
    fig.suptitle('Prédictions vs Vérité Terrain par Learning Rate', fontsize=16, fontweight='bold')
    
    for idx, lr_key in enumerate(lr_names):
        ax = axes[idx]
        data = resultats_complets[lr_key]
        if data is None:
            continue
            
        y_pred, y_true = data["test"]
        
        # Scatter plot
        ax.scatter(y_true, y_pred, alpha=0.5, s=20, color='#45B7D1')
        
        # Line parfaite
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Parfait')
        
        # Calculer R²
        r2 = np.corrcoef(y_true, y_pred)[0, 1]**2
        
        ax.set_xlabel('Vérité Terrain (m)', fontsize=11)
        ax.set_ylabel('Prédictions (m)', fontsize=11)
        ax.set_title(f'LR: {lr_key}\nR² = {r2:.4f}', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'results/plots/scatter_pred_true{split_suffix}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Diagramme 4: Radar chart des métriques
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Normaliser les métriques pour le radar chart (inverser les métriques d'erreur)
    metrics_for_radar = ['abs_rel', 'rmse', 'delta1', 'silog']
    normalized_data = {}
    
    for lr_key in lr_names:
        row = tableau_test.loc[f"DAV2_finetune_{lr_key}"]
        normalized_data[lr_key] = [
            1 - (row['abs_rel'] / tableau_test['abs_rel'].max()),  # Inverser
            1 - (row['rmse'] / tableau_test['rmse'].max()),        # Inverser
            row['delta1'] / tableau_test['delta1'].max(),           # Normaliser
            1 - (row['silog'] / tableau_test['silog'].max())       # Inverser
        ]
    
    # Configuration du radar chart
    angles = np.linspace(0, 2 * np.pi, len(metrics_for_radar), endpoint=False).tolist()
    angles += angles[:1]  # Fermer le cercle
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for idx, lr_key in enumerate(lr_names):
        values = normalized_data[lr_key]
        values += values[:1]  # Fermer le cercle
        
        ax.plot(angles, values, 'o-', linewidth=2, label=lr_key, color=colors[idx])
        ax.fill(angles, values, alpha=0.15, color=colors[idx])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(['Abs Rel', 'RMSE', 'δ1', 'SiLog'], fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
    ax.set_title('Radar des Performances (Normalisé)', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'results/plots/radar_performance{split_suffix}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Diagrammes générés dans results/plots/")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tuning Depth Anything V2 avec choix des splits")
    parser.add_argument("--use-stratified", action="store_true", default=True,
                       help="Utiliser les nouveaux splits stratifiés (défaut: True)")
    parser.add_argument("--use-original", action="store_true", default=False,
                       help="Utiliser les splits originaux au lieu des stratifiés")
    parser.add_argument("--epochs", type=int, default=200, help="Nombre d'époques (défaut: 200)")
    parser.add_argument("--patience", type=int, default=20, help="Patience pour early stopping (défaut: 20)")
    parser.add_argument("--batch-size", type=int, default=4, help="Taille de batch (défaut: 4 pour économiser la mémoire)")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate (défaut: 5e-6)")
    parser.add_argument("--encoder", type=str, default="vitl", choices=["vits", "vitb", "vitl"], 
                       help="Encodeur ViT (défaut: vitl)")
    parser.add_argument("--img-size", type=int, default=392, help="Taille d'image (défaut: 392 pour économiser la mémoire)")
    parser.add_argument("--gradient-accumulation", type=int, default=2, help="Gradient accumulation steps (défaut: 2)")
    parser.add_argument("--multi-lr-test", action="store_true", default=False,
                       help="Tester plusieurs learning rates (1e-5, 2e-5, 1e-4) et comparer les résultats")
    parser.add_argument("--use-multi-loss", action="store_true", default=True,
                       help="Utiliser multi-loss (SiLog + L1 + Gradient + Scale-invariant)")
    parser.add_argument("--label-smoothing", type=float, default=0.1, help="Label smoothing coefficient (défaut: 0.1)")
    parser.add_argument("--silog-weight", type=float, default=1.0, help="Poids SiLog loss (défaut: 1.0)")
    parser.add_argument("--l1-weight", type=float, default=0.5, help="Poids L1 loss (défaut: 0.5)")
    parser.add_argument("--gradient-weight", type=float, default=0.3, help="Poids Gradient loss (défaut: 0.3)")
    parser.add_argument("--scale-weight", type=float, default=0.2, help="Poids Scale-invariant loss (défaut: 0.2)")
    parser.add_argument("--dropout-rate", type=float, default=0.1, help="Taux de dropout (défaut: 0.1)")
    parser.add_argument("--building-lighting", action="store_true", default=True,
                       help="Activer les augmentations spécifiques bâtiments (défaut: True)")
    
    args = parser.parse_args()
    
    # Déterminer quel type de splits utiliser
    use_stratified = args.use_stratified and not args.use_original
    
    print("=== Étape 1 : préparation des splits ===")
    preparer_splits(use_stratified=use_stratified)

    if args.multi_lr_test:
        print("\n=== MODE MULTI-LR TEST ===")
        resultats_complets = lancer_multi_lr_test(
            epochs=args.epochs,
            bs=args.batch_size,
            patience=args.patience,
            encoder=args.encoder,
            img_size=args.img_size,
            gradient_accumulation=args.gradient_accumulation,
            use_multi_loss=args.use_multi_loss,
            label_smoothing=args.label_smoothing,
            silog_weight=args.silog_weight,
            l1_weight=args.l1_weight,
            gradient_weight=args.gradient_weight,
            scale_weight=args.scale_weight,
            dropout_rate=args.dropout_rate,
            building_lighting=args.building_lighting,
            base_port=29500,
        )
        
        split_suffix = "_stratified" if use_stratified else "_original"
        comparer_resultats_multi_lr(resultats_complets, split_suffix)
    else:
        print("\n=== Étape 2 : entraînement avec early stopping ===")
        lancer_entrainement(epochs=args.epochs, patience=args.patience, 
                           bs=args.batch_size, lr=args.lr, metric_early_stop="abs_rel",
                           encoder=args.encoder, img_size=args.img_size,
                           gradient_accumulation=args.gradient_accumulation,
                           use_multi_loss=args.use_multi_loss,
                           label_smoothing=args.label_smoothing,
                           silog_weight=args.silog_weight,
                           l1_weight=args.l1_weight,
                           gradient_weight=args.gradient_weight,
                           scale_weight=args.scale_weight,
                           dropout_rate=args.dropout_rate,
                           building_lighting=args.building_lighting,
                           custom_port=29500)

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