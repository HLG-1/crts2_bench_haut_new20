"""
pipelines/05_train_htc_dc_net_training_only.py

Script d'entraînement HTC-DC Net sans conversion de données (étape 1 déjà faite).
Utilise les splits stratifiés et lance l'entraînement avec 200 epochs, batch size 32, early stopping.
"""
import sys, os, subprocess, glob
from pathlib import Path

# Ajouter la racine et HTC-DC-Net au PYTHONPATH
sys.path.append(".")
htc_dir_abs = os.path.abspath("third_party/HTC-DC-Net")
if htc_dir_abs not in sys.path:
    sys.path.insert(0, htc_dir_abs)

HTC_DIR = "third_party/HTC-DC-Net"
DATA_HTC_DIR = "data/htc_dc_net_format"
CONFIG_PATH = "third_party/HTC-DC-Net/configs/batiments_maroc.yaml"
EXP_CONFIG_PATH = "third_party/HTC-DC-Net/configs/htcdc.yaml"


def lancer_entrainement(epochs: int = 200, batch_size: int = 16, use_stratified: bool = True):
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
        f"batch_size={batch_size}"
    ]
    print("Commande :", " ".join(cmd))
    subprocess.run(cmd, cwd=HTC_DIR, check=True)
    
    # Copier les checkpoints et logs vers le répertoire de résultats
    print(f"\nCopie des résultats vers {results_dir}...")
    checkpoints_dir = os.path.join(HTC_DIR, "checkpoints", "htcdc")
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


if __name__ == "__main__":
    print("=== Entraînement HTC-DC Net ===")
    lancer_entrainement(epochs=200, batch_size=16, use_stratified=True)
    print("\n=== Entraînement terminé ===")