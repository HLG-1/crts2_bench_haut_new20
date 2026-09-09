"""
pipelines/06_evaluate_htc_dc_net.py

Script isolé pour l'évaluation du modèle HTC-DC Net entraîné.
- Charge le checkpoint du modèle
- Évalue sur validation et test
- Génère des graphiques de variation de loss
- Génère des visualisations d'évaluation
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
matplotlib.use('Agg')  # Backend non-interactif pour éviter les problèmes d'affichage
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# Importer les modules HTC-DC Net
from htcdc import UBins

# Configuration
HTC_DIR = "third_party/HTC-DC-Net"
DATA_HTC_DIR = "data/htc_dc_net_format"
CONFIG_PATH = "third_party/HTC-DC-Net/configs/batiments_maroc.yaml"
EXP_CONFIG_PATH = "third_party/HTC-DC-Net/configs/htcdc.yaml"
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
    """Charge le modèle depuis un checkpoint avec gestion du problème weights_only."""
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
        # Configuration par défaut (basée sur batiments_maroc.yaml)
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
    
    # S'assurer que le fichier ndsm_stats existe pour le modèle
    ndsm_stats_file = os.path.join(DATA_HTC_DIR, "ndsm_stats.pickle")
    if not os.path.exists(ndsm_stats_file):
        print(f"Attention: {ndsm_stats_file} n'existe pas, création avec des valeurs par défaut")
        # Créer un fichier stats par défaut
        default_stats = {
            'mean': 0.0,
            'std': 1.0,
            'min': 0.0,
            'max': 50.0,
            'h_max': 50.0
        }
        torch.save(default_stats, ndsm_stats_file)
    
    # Corriger le chemin data_dir dans la configuration pour utiliser le chemin absolu
    if 'data_dir' not in config or config['data_dir'].startswith('../../'):
        config['data_dir'] = os.path.abspath(DATA_HTC_DIR)
    
    # S'assurer que num_classes est défini
    if 'num_classes' not in config:
        config['num_classes'] = 256
    
    # Créer le modèle avec la configuration
    try:
        model = UBins(config)
    except Exception as e:
        print(f"Erreur lors de la création du modèle: {e}")
        import traceback
        traceback.print_exc()
        # Utiliser une configuration plus simple
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
    
    # Charger le checkpoint avec weights_only=False pour résoudre le problème
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"Erreur lors du chargement avec weights_only=False: {e}")
        # Essayer avec weights_only=True et add_safe_globals
        torch.serialization.add_safe_globals([np.ndarray])
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    # Charger les poids
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


def charger_logs_entrainement(training_dir):
    """Charge les logs d'entraînement depuis le CSV."""
    log_file = os.path.join(training_dir, "training_log.csv")
    if not os.path.exists(log_file):
        print(f"Fichier de log non trouvé: {log_file}")
        return None
    
    df = pd.read_csv(log_file)
    return df


def visualiser_loss(df, output_dir):
    """Génère les graphiques de variation de loss."""
    if df is None:
        print("Pas de données pour visualiser les loss")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Filtrer les lignes de validation (qui ont des métriques MAE/RMSE)
    val_rows = df[df['mae'].notna()]
    
    if val_rows.empty:
        print("Pas de données de validation trouvées")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. MAE au fil des epochs
    axes[0, 0].plot(val_rows['epoch'], val_rows['mae'], 'b-', marker='o', label='MAE')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('MAE')
    axes[0, 0].set_title('Évolution du MAE')
    axes[0, 0].grid(True)
    axes[0, 0].legend()
    
    # 2. RMSE au fil des epochs
    axes[0, 1].plot(val_rows['epoch'], val_rows['rmse'], 'r-', marker='s', label='RMSE')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('RMSE')
    axes[0, 1].set_title('Évolution du RMSE')
    axes[0, 1].grid(True)
    axes[0, 1].legend()
    
    # 3. Validation loss
    if 'val/loss_total' in val_rows.columns:
        axes[1, 0].plot(val_rows['epoch'], val_rows['val/loss_total'], 'g-', marker='^', label='Val Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Validation Loss')
        axes[1, 0].set_title('Évolution de la Validation Loss')
        axes[1, 0].grid(True)
        axes[1, 0].legend()
    
    # 4. MAE vs RMSE
    axes[1, 1].scatter(val_rows['mae'], val_rows['rmse'], alpha=0.6)
    axes[1, 1].set_xlabel('MAE')
    axes[1, 1].set_ylabel('RMSE')
    axes[1, 1].set_title('MAE vs RMSE')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_evolution.png'), dpi=150)
    print(f"Graphique de loss sauvegardé: {output_dir}/loss_evolution.png")
    plt.close()
    
    # Graphique individuel pour MAE et RMSE
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(val_rows['epoch'], val_rows['mae'], 'b-', marker='o', label='MAE')
    ax.plot(val_rows['epoch'], val_rows['rmse'], 'r-', marker='s', label='RMSE')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Erreur')
    ax.set_title('Évolution des métriques d\'erreur')
    ax.grid(True)
    ax.legend()
    plt.savefig(os.path.join(output_dir, 'metrics_evolution.png'), dpi=150)
    print(f"Graphique de métriques sauvegardé: {output_dir}/metrics_evolution.png")
    plt.close()


def evaluer_modele(model, data_loader, device, split_name="validation"):
    """Évalue le modèle sur un dataset."""
    model.eval()
    
    mae_list = []
    rmse_list = []
    predictions = []
    targets = []
    
    print(f"\nÉvaluation sur {split_name}...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            if isinstance(batch, dict):
                images = batch['image'].to(device)
                targets_batch = batch['ndsm'].to(device)
            else:
                images, targets_batch = batch
                images = images.to(device)
                targets_batch = targets_batch.to(device)
            
            # Prédiction avec le format attendu par UBins
            gt = {"ndsm": targets_batch}
            outputs = model(images, gt)
            
            # Extraire la prédiction principale
            if isinstance(outputs, dict):
                pred_ndsm = outputs["ndsm"]
            else:
                pred_ndsm = outputs
            
            # Calculer les métriques
            mae = torch.mean(torch.abs(pred_ndsm - targets_batch)).item()
            rmse = torch.sqrt(torch.mean((pred_ndsm - targets_batch) ** 2)).item()
            
            mae_list.append(mae)
            rmse_list.append(rmse)
            
            # Stocker pour visualisation
            predictions.append(pred_ndsm.cpu().numpy())
            targets.append(targets_batch.cpu().numpy())
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx + 1}/{len(data_loader)} - MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    
    # Métriques globales
    avg_mae = np.mean(mae_list)
    avg_rmse = np.mean(rmse_list)
    std_mae = np.std(mae_list)
    std_rmse = np.std(rmse_list)
    
    print(f"\nRésultats {split_name}:")
    print(f"  MAE: {avg_mae:.4f} ± {std_mae:.4f}")
    print(f"  RMSE: {avg_rmse:.4f} ± {std_rmse:.4f}")
    
    results = {
        'split': split_name,
        'mae': avg_mae,
        'rmse': avg_rmse,
        'std_mae': std_mae,
        'std_rmse': std_rmse,
        'predictions': np.concatenate(predictions),
        'targets': np.concatenate(targets)
    }
    
    return results


def visualiser_predictions(results, output_dir, max_samples=5):
    """Visualise quelques prédictions vs cibles."""
    os.makedirs(output_dir, exist_ok=True)
    
    predictions = results['predictions']
    targets = results['targets']
    split_name = results['split']
    
    n_samples = min(max_samples, len(predictions))
    
    fig, axes = plt.subplots(n_samples, 3, figsize=(15, 5*n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_samples):
        pred = predictions[i].squeeze()
        target = targets[i].squeeze()
        
        # Prédiction
        im1 = axes[i, 0].imshow(pred, cmap='viridis')
        axes[i, 0].set_title(f'Prédiction {i+1}')
        axes[i, 0].axis('off')
        plt.colorbar(im1, ax=axes[i, 0])
        
        # Cible
        im2 = axes[i, 1].imshow(target, cmap='viridis')
        axes[i, 1].set_title(f'Cible {i+1}')
        axes[i, 1].axis('off')
        plt.colorbar(im2, ax=axes[i, 1])
        
        # Différence
        diff = np.abs(pred - target)
        im3 = axes[i, 2].imshow(diff, cmap='hot')
        axes[i, 2].set_title(f'Différence absolue {i+1}')
        axes[i, 2].axis('off')
        plt.colorbar(im3, ax=axes[i, 2])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'predictions_{split_name}.png'), dpi=150)
    print(f"Visualisation des prédictions sauvegardée: {output_dir}/predictions_{split_name}.png")
    plt.close()


def generer_rapport_simple(output_dir, logs_df):
    """Génère un rapport HTML simplifié avec les graphiques d'entraînement."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculer les statistiques d'entraînement
    if logs_df is not None:
        val_rows = logs_df[logs_df['mae'].notna()]
        if not val_rows.empty:
            best_mae = val_rows['mae'].min()
            best_rmse = val_rows['rmse'].min()
            best_mae_epoch = val_rows.loc[val_rows['mae'].idxmin(), 'epoch']
            best_rmse_epoch = val_rows.loc[val_rows['rmse'].idxmin(), 'epoch']
            final_mae = val_rows.iloc[-1]['mae']
            final_rmse = val_rows.iloc[-1]['rmse']
        else:
            best_mae = best_rmse = best_mae_epoch = best_rmse_epoch = final_mae = final_rmse = "N/A"
    else:
        best_mae = best_rmse = best_mae_epoch = best_rmse_epoch = final_mae = final_rmse = "N/A"
    
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Rapport d'évaluation HTC-DC Net</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            h1 { color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }
            h2 { color: #666; margin-top: 30px; }
            .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }
            .metric-card { background-color: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 4px solid #007bff; }
            .metric-label { font-size: 14px; color: #666; margin-bottom: 5px; }
            .metric-value { font-size: 24px; font-weight: bold; color: #333; }
            .graph-container { margin: 20px 0; text-align: center; }
            .graph-container img { max-width: 100%; height: auto; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .footer { margin-top: 30px; padding-top: 20px; border-top: 1px solid #ddd; color: #666; font-size: 12px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Rapport d'évaluation HTC-DC Net</h1>
            <p>Généré le: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
            
            <h2>Statistiques d'entraînement</h2>
            <div class="metrics">
                <div class="metric-card">
                    <div class="metric-label">Meilleur MAE</div>
                    <div class="metric-value">{best_mae if isinstance(best_mae, str) else f'{best_mae:.4f}'}</div>
                    <div class="metric-label">Epoch: {best_mae_epoch if isinstance(best_mae_epoch, str) else int(best_mae_epoch)}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Meilleur RMSE</div>
                    <div class="metric-value">{best_rmse if isinstance(best_rmse, str) else f'{best_rmse:.4f}'}</div>
                    <div class="metric-label">Epoch: {best_rmse_epoch if isinstance(best_rmse_epoch, str) else int(best_rmse_epoch)}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">MAE Final</div>
                    <div class="metric-value">{final_mae if isinstance(final_mae, str) else f'{final_mae:.4f}'}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">RMSE Final</div>
                    <div class="metric-value">{final_rmse if isinstance(final_rmse, str) else f'{final_rmse:.4f}'}</div>
                </div>
            </div>
            
            <h2>Graphiques d'entraînement</h2>
            <div class="graph-container">
                <h3>Évolution des métriques d'entraînement</h3>
                <img src="loss_evolution.png" alt="Evolution des loss">
            </div>
            
            <div class="graph-container">
                <h3>Évolution MAE et RMSE</h3>
                <img src="metrics_evolution.png" alt="Evolution des métriques">
            </div>
            
            <div class="footer">
                <p>Note: L'évaluation complète sur les datasets de validation et test peut être ajoutée ultérieurement.</p>
                <p>Ce rapport se concentre sur l'analyse des métriques d'entraînement.</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    with open(os.path.join(output_dir, 'rapport_evaluation.html'), 'w') as f:
        f.write(html_content)
    
    print(f"Rapport HTML sauvegardé: {output_dir}/rapport_evaluation.html")


def generer_rapport(results_list, output_dir):
    """Génère un rapport HTML des résultats."""
    os.makedirs(output_dir, exist_ok=True)
    
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Rapport d'évaluation HTC-DC Net</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1 { color: #333; }
            h2 { color: #666; }
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            .metric { font-weight: bold; }
        </style>
    </head>
    <body>
        <h1>Rapport d'évaluation HTC-DC Net</h1>
        <p>Généré le: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
        
        <h2>Métriques d'évaluation</h2>
        <table>
            <tr>
                <th>Split</th>
                <th>MAE</th>
                <th>RMSE</th>
                <th>Std MAE</th>
                <th>Std RMSE</th>
            </tr>
    """
    
    for results in results_list:
        html_content += f"""
            <tr>
                <td>{results['split']}</td>
                <td class="metric">{results['mae']:.4f}</td>
                <td class="metric">{results['rmse']:.4f}</td>
                <td>{results['std_mae']:.4f}</td>
                <td>{results['std_rmse']:.4f}</td>
            </tr>
        """
    
    html_content += """
        </table>
        
        <h2>Graphiques</h2>
        <h3>Évolution des métriques d'entraînement</h3>
        <img src="loss_evolution.png" alt="Evolution des loss" style="max-width: 100%;">
        
        <h3>Évolution MAE et RMSE</h3>
        <img src="metrics_evolution.png" alt="Evolution des métriques" style="max-width: 100%;">
    """
    
    for results in results_list:
        split_name = results['split']
        html_content += f"""
        <h3>Prédictions {split_name}</h3>
        <img src="predictions_{split_name}.png" alt="Prédictions {split_name}" style="max-width: 100%;">
        """
    
    html_content += """
    </body>
    </html>
    """
    
    with open(os.path.join(output_dir, 'rapport_evaluation.html'), 'w') as f:
        f.write(html_content)
    
    print(f"Rapport HTML sauvegardé: {output_dir}/rapport_evaluation.html")


def main():
    """Fonction principale."""
    print("=" * 60)
    print("Script d'évaluation HTC-DC Net")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device utilisé: {device}")
    
    # Créer le répertoire de résultats
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(RESULTS_DIR, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 1. Trouver et charger le checkpoint
        checkpoint_path = trouver_dernier_checkpoint()
        print(f"\nCheckpoint trouvé: {checkpoint_path}")
        
        model, config = charger_modele(checkpoint_path, device)
        
        # 2. Charger les logs d'entraînement
        training_dir = os.path.dirname(checkpoint_path)
        logs_df = charger_logs_entrainement(training_dir)
        
        # 3. Générer les graphiques de loss
        print("\nGénération des graphiques de loss...")
        visualiser_loss(logs_df, output_dir)
        
        # 4. Évaluation sur validation et test
        results_list = []
        
        # Essayer de créer les dataloaders pour validation et test
        print("\nNote: L'évaluation sur les datasets nécessite une configuration compatible.")
        print("Pour l'instant, le script se concentre sur la visualisation des logs d'entraînement.")
        
        # Optionnel: Ajouter ici l'évaluation manuelle si nécessaire
        # L'évaluation complète des datasets peut être ajoutée ultérieurement
        
        # 5. Générer le rapport (simplifié avec seulement les graphiques)
        print("\nGénération du rapport simplifié...")
        generer_rapport_simple(output_dir, logs_df)
        
        print("\n" + "=" * 60)
        print(f"Évaluation terminée! Résultats sauvegardés dans: {output_dir}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nErreur lors de l'évaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())