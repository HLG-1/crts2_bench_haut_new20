"""
Script pour régénérer les plots de comparaison multi-LR à partir des données existantes
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Backend non-interactif
import matplotlib.pyplot as plt

def regenerer_plots():
    """Régénère les plots à partir des fichiers existants"""
    
    # Charger les données
    results_dir = "results"
    
    # Charger les métriques
    try:
        tableau_val = pd.read_csv(f"{results_dir}/metrics_dav2_multi_lr_val_stratified.csv", index_col=0)
        tableau_test = pd.read_csv(f"{results_dir}/metrics_dav2_multi_lr_test_stratified.csv", index_col=0)
        print("✓ Métriques chargées")
    except Exception as e:
        print(f"✗ Erreur chargement métriques: {e}")
        return
    
    # Charger les prédictions pour chaque LR
    resultats_complets = {}
    for lr_key in ["lr_2e-05", "lr_0.0001"]:
        try:
            val_data = np.load(f"{results_dir}/predictions_dav2_{lr_key}_val_stratified.npz")
            test_data = np.load(f"{results_dir}/predictions_dav2_{lr_key}_test_stratified.npz")
            
            resultats_complets[lr_key] = {
                "val": (val_data["y_pred"], val_data["y_true"]),
                "test": (test_data["y_pred"], test_data["y_true"]),
                "patch_exemple": None  # Pas nécessaire pour les plots
            }
            print(f"✓ Prédictions chargées pour {lr_key}")
        except Exception as e:
            print(f"✗ Erreur chargement prédictions {lr_key}: {e}")
    
    if not resultats_complets:
        print("❌ Aucune donnée de prédictions disponible")
        return
    
    # Créer le dossier plots
    os.makedirs(f"{results_dir}/plots", exist_ok=True)
    
    # Extraire les noms de LR
    lr_names = [name.replace("DAV2_finetune_", "") for name in tableau_val.index if "lr_" in name]
    print(f"LR à plotter: {lr_names}")
    
    # Diagramme 1: Comparaison des métriques principales (Test)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Comparaison des Learning Rates - Dataset Test', fontsize=16, fontweight='bold')
    
    metrics_to_plot = ['Erreur_relative', 'RMSE_B', 'MAE', 'latence_moy_ms']
    metric_labels = ['Erreur Relative', 'RMSE (m)', 'MAE (m)', 'Latence (ms)']
    
    for idx, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
        ax = axes[idx // 2, idx % 2]
        
        if metric in tableau_test.columns:
            values = tableau_test.loc[[f"DAV2_finetune_{lr}" for lr in lr_names], metric].values
            # Gérer les valeurs NaN
            values = np.array([v if not np.isnan(v) else 0 for v in values])
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
            bars = ax.bar(lr_names, values, color=colors[:len(lr_names)])
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.4f}',
                       ha='center', va='bottom', fontsize=10)
            
            ax.set_ylabel(label, fontsize=12)
            ax.set_xlabel('Learning Rate', fontsize=12)
            ax.set_title(f'{label} par Learning Rate', fontsize=11, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{results_dir}/plots/comparaison_lr_test_stratified.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Plot 1 sauvegardé: comparaison_lr_test_stratified.png")
    
    # Diagramme 2: Comparaison Validation vs Test
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Validation vs Test par Learning Rate', fontsize=16, fontweight='bold')
    
    for idx, metric in enumerate(['Erreur_relative', 'RMSE_B']):
        ax = axes[idx]
        
        val_values = tableau_val.loc[[f"DAV2_finetune_{lr}" for lr in lr_names], metric].values
        test_values = tableau_test.loc[[f"DAV2_finetune_{lr}" for lr in lr_names], metric].values
        
        # Gérer les valeurs NaN
        val_values = np.array([v if not np.isnan(v) else 0 for v in val_values])
        test_values = np.array([v if not np.isnan(v) else 0 for v in test_values])
        
        x = np.arange(len(lr_names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, val_values, width, label='Validation', color='#4ECDC4')
        bars2 = ax.bar(x + width/2, test_values, width, label='Test', color='#FF6B6B')
        
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.4f}',
                       ha='center', va='bottom', fontsize=9)
        
        ax.set_ylabel(metric, fontsize=12)
        ax.set_xlabel('Learning Rate', fontsize=12)
        ax.set_title(f'{metric}: Validation vs Test', fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(lr_names)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{results_dir}/plots/comparaison_val_test_stratified.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Plot 2 sauvegardé: comparaison_val_test_stratified.png")
    
    # Diagramme 3: Scatter plots Pred vs True pour chaque LR
    fig, axes = plt.subplots(1, len(lr_names), figsize=(6*len(lr_names), 5))
    if len(lr_names) == 1:
        axes = [axes]
    
    fig.suptitle('Prédictions vs Vérité Terrain par Learning Rate', fontsize=16, fontweight='bold')
    
    for idx, lr_key in enumerate(lr_names):
        ax = axes[idx]
        data = resultats_complets.get(lr_key)
        if data is None:
            continue
            
        y_pred, y_true = data["test"]
        
        ax.scatter(y_true, y_pred, alpha=0.5, s=20, color='#45B7D1')
        
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Parfait')
        
        r2 = np.corrcoef(y_true, y_pred)[0, 1]**2
        
        ax.set_xlabel('Vérité Terrain (m)', fontsize=11)
        ax.set_ylabel('Prédictions (m)', fontsize=11)
        ax.set_title(f'LR: {lr_key}\nR² = {r2:.4f}', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{results_dir}/plots/scatter_pred_true_stratified.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Plot 3 sauvegardé: scatter_pred_true_stratified.png")
    
    # Diagramme 4: Radar chart des métriques
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    metrics_for_radar = ['Erreur_relative', 'RMSE_B', 'MAE', 'NMAD']
    normalized_data = {}
    
    for lr_key in lr_names:
        row = tableau_test.loc[f"DAV2_finetune_{lr_key}"]
        normalized_data[lr_key] = [
            1 - (row['Erreur_relative'] / tableau_test['Erreur_relative'].max()) if row['Erreur_relative'] and not np.isnan(row['Erreur_relative']) else 0,
            1 - (row['RMSE_B'] / tableau_test['RMSE_B'].max()) if row['RMSE_B'] and not np.isnan(row['RMSE_B']) else 0,
            1 - (row['MAE'] / tableau_test['MAE'].max()) if row['MAE'] and not np.isnan(row['MAE']) else 0,
            1 - (row['NMAD'] / tableau_test['NMAD'].max()) if row['NMAD'] and not np.isnan(row['NMAD']) else 0
        ]
    
    angles = np.linspace(0, 2 * np.pi, len(metrics_for_radar), endpoint=False).tolist()
    angles += angles[:1]
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for idx, lr_key in enumerate(lr_names):
        values = normalized_data[lr_key]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=lr_key, color=colors[idx])
        ax.fill(angles, values, alpha=0.15, color=colors[idx])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(['Erreur Rel', 'RMSE', 'MAE', 'NMAD'], fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
    ax.set_title('Radar des Performances (Normalisé)', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{results_dir}/plots/radar_performance_stratified.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Plot 4 sauvegardé: radar_performance_stratified.png")
    
    print(f"\n✓ Tous les plots ont été régénérés dans {results_dir}/plots/")

if __name__ == "__main__":
    regenerer_plots()