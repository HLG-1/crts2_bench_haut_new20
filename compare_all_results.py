"""
Script pour comparer tous les résultats DAV2 : multi-LR, premier finetune, et zero-shot
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Backend non-interactif
import matplotlib.pyplot as plt

def charger_tous_les_resultats():
    """Charge tous les fichiers de résultats disponibles"""
    results_dir = "results"
    all_results = {}
    
    # 1. Multi-LR results
    try:
        multi_lr_val = pd.read_csv(f"{results_dir}/metrics_dav2_multi_lr_val_stratified.csv", index_col=0)
        multi_lr_test = pd.read_csv(f"{results_dir}/metrics_dav2_multi_lr_test_stratified.csv", index_col=0)
        all_results["multi_lr"] = {"val": multi_lr_val, "test": multi_lr_test}
        print("✓ Multi-LR results chargés")
    except Exception as e:
        print(f"✗ Erreur chargement multi-LR: {e}")
    
    # 2. Premier finetune DAV2
    try:
        finetune = pd.read_csv(f"{results_dir}/metrics_dav2_finetune_stratified.csv", index_col=0)
        # Renommer pour cohérence
        finetune.index = ["DAV2_finetune_v1"]
        all_results["finetune_v1"] = {"test": finetune}
        print("✓ Premier finetune DAV2 chargé")
    except Exception as e:
        print(f"✗ Erreur chargement finetune v1: {e}")
    
    # 3. Zero-shot results
    try:
        zero_shot = pd.read_csv(f"{results_dir}/metrics_zero_shot_stratified.csv", index_col=0)
        all_results["zero_shot"] = {"test": zero_shot}
        print("✓ Zero-shot results chargés")
    except Exception as e:
        print(f"✗ Erreur chargement zero-shot: {e}")
    
    return all_results

def combiner_resultats_test(all_results):
    """Combine tous les résultats test en un seul DataFrame"""
    test_frames = []
    
    # Multi-LR test results
    if "multi_lr" in all_results:
        test_frames.append(all_results["multi_lr"]["test"])
    
    # Premier finetune
    if "finetune_v1" in all_results:
        test_frames.append(all_results["finetune_v1"]["test"])
    
    # Zero-shot
    if "zero_shot" in all_results:
        test_frames.append(all_results["zero_shot"]["test"])
    
    if test_frames:
        combined = pd.concat(test_frames)
        return combined
    return None

def generer_plots_comparaison_complete(combined_df):
    """Génère des plots comparatifs complets"""
    os.makedirs("results/plots", exist_ok=True)
    
    # Renommer les modèles pour plus de clarté
    name_mapping = {
        "DAV2_finetune_lr_2e-05": "DAV2 LR 2e-5",
        "DAV2_finetune_lr_0.0001": "DAV2 LR 1e-4",
        "DAV2_finetune_v1": "DAV2 LR 5e-6",
        "DAV2_zeroshot": "DAV2 Zero-shot",
        "DepthPro": "DepthPro"
    }
    
    combined_df = combined_df.copy()
    combined_df['display_name'] = combined_df.index.map(lambda x: name_mapping.get(x, x))
    
    # Filtrer pour ne garder que les modèles DAV2
    dav2_models = combined_df[combined_df.index.str.contains('DAV2', case=False, na=False)]
    
    # Diagramme 1: Comparaison des métriques principales (tous modèles)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Comparaison Complète - Dataset Test', fontsize=16, fontweight='bold')
    
    metrics_to_plot = ['Erreur_relative', 'RMSE_B', 'MAE', 'latence_moy_ms']
    metric_labels = ['Erreur Relative', 'RMSE (m)', 'MAE (m)', 'Latence (ms)']
    
    for idx, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
        ax = axes[idx // 2, idx % 2]
        
        if metric in dav2_models.columns:
            values = dav2_models[metric].values
            names = dav2_models['display_name'].values
            
            # Trier par performance (pour les métriques d'erreur)
            if metric != 'latence_moy_ms':
                sort_idx = np.argsort(values)
                sorted_names = names[sort_idx]
                sorted_values = values[sort_idx]
                colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(sorted_names)))
            else:
                sorted_names = names
                sorted_values = values
                colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(sorted_names)))
            
            bars = ax.barh(sorted_names, sorted_values, color=colors)
            
            # Ajouter les valeurs
            for bar in bars:
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2.,
                       f'{width:.3f}',
                       ha='left', va='center', fontsize=9)
            
            ax.set_xlabel(label, fontsize=12)
            ax.set_title(f'{label} - Moindre est meilleur', fontsize=11, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/plots/comparaison_complete_test.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Plot 1 sauvegardé: comparaison_complete_test.png")
    
    # Diagramme 2: Focus sur les finetunes DAV2
    finetune_models = dav2_models[dav2_models.index.str.contains('finetune', case=False, na=False)]
    
    if len(finetune_models) > 1:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Comparaison des Finetunes DAV2', fontsize=16, fontweight='bold')
        
        for idx, metric in enumerate(['Erreur_relative', 'RMSE_B', 'MAE']):
            ax = axes[idx]
            
            values = finetune_models[metric].values
            names = finetune_models['display_name'].values
            
            # Trier
            sort_idx = np.argsort(values)
            sorted_names = names[sort_idx]
            sorted_values = values[sort_idx]
            
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
            bars = ax.bar(sorted_names, sorted_values, color=colors[:len(sorted_names)])
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=10)
            
            ax.set_ylabel(metric, fontsize=12)
            ax.set_title(f'{metric}', fontsize=11, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/plots/comparaison_finetunes_dav2.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Plot 2 sauvegardé: comparaison_finetunes_dav2.png")
    
    # Diagramme 3: Radar chart tous modèles DAV2
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    metrics_for_radar = ['Erreur_relative', 'RMSE_B', 'MAE', 'NMAD']
    normalized_data = {}
    
    for idx, (_, row) in enumerate(dav2_models.iterrows()):
        name = row['display_name']
        normalized_data[name] = []
        
        for metric in metrics_for_radar:
            if metric in dav2_models.columns and not pd.isna(row[metric]):
                max_val = dav2_models[metric].max()
                if max_val > 0:
                    # Inverser pour les métriques d'erreur (meilleur = plus petit)
                    norm_val = 1 - (row[metric] / max_val)
                else:
                    norm_val = 0
            else:
                norm_val = 0
            normalized_data[name].append(norm_val)
    
    angles = np.linspace(0, 2 * np.pi, len(metrics_for_radar), endpoint=False).tolist()
    angles += angles[:1]
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(normalized_data)))
    
    for idx, (name, values) in enumerate(normalized_data.items()):
        values_complete = values + values[:1]
        
        ax.plot(angles, values_complete, 'o-', linewidth=2, label=name, color=colors[idx])
        ax.fill(angles, values_complete, alpha=0.15, color=colors[idx])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(['Err Rel', 'RMSE', 'MAE', 'NMAD'], fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
    ax.set_title('Radar des Performances DAV2 (Normalisé)', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('results/plots/radar_complete_dav2.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Plot 3 sauvegardé: radar_complete_dav2.png")
    
    # Diagramme 4: Tableau récapitulatif
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Préparer les données pour le tableau
    table_data = []
    columns = ['Modèle', 'MAE (m)', 'RMSE (m)', 'Err Rel', 'NMAD', 'Corrélation', 'Latence (ms)']
    
    for _, row in dav2_models.iterrows():
        table_data.append([
            row['display_name'],
            f"{row['MAE']:.3f}" if not pd.isna(row['MAE']) else 'N/A',
            f"{row['RMSE_B']:.3f}" if not pd.isna(row['RMSE_B']) else 'N/A',
            f"{row['Erreur_relative']:.3f}" if not pd.isna(row['Erreur_relative']) else 'N/A',
            f"{row['NMAD']:.3f}" if not pd.isna(row['NMAD']) else 'N/A',
            f"{row['Correlation']:.3f}" if not pd.isna(row['Correlation']) else 'N/A',
            f"{row['latence_moy_ms']:.1f}" if not pd.isna(row['latence_moy_ms']) else 'N/A'
        ])
    
    # Trier par MAE
    table_data.sort(key=lambda x: float(x[1].replace('N/A', '999')))
    
    table = ax.table(cellText=table_data, colLabels=columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Colorer les cellules
    for i in range(len(table_data)):
        for j in range(1, 5):  # Colonnes de métriques
            if i == 0:  # Meilleur modèle
                table[(i+1, j)].set_facecolor('#90EE90')  # Vert clair
            elif i == len(table_data) - 1:  # Pire modèle
                table[(i+1, j)].set_facecolor('#FFB6C1')  # Rouge clair
    
    plt.title('Tableau Récapitulatif des Performances DAV2', fontsize=14, fontweight='bold', pad=20)
    plt.savefig('results/plots/tableau_recapitulatif_dav2.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Plot 4 sauvegardé: tableau_recapitulatif_dav2.png")
    
    return dav2_models

def sauvegarder_tableau_combiné(dav2_models):
    """Sauvegarde un tableau combiné en CSV"""
    # Créer un tableau simplifié
    summary_data = []
    for _, row in dav2_models.iterrows():
        summary_data.append({
            'Modèle': row['display_name'],
            'MAE_m': row['MAE'] if not pd.isna(row['MAE']) else None,
            'RMSE_m': row['RMSE_B'] if not pd.isna(row['RMSE_B']) else None,
            'Erreur_Relative': row['Erreur_relative'] if not pd.isna(row['Erreur_relative']) else None,
            'NMAD': row['NMAD'] if not pd.isna(row['NMAD']) else None,
            'Correlation': row['Correlation'] if not pd.isna(row['Correlation']) else None,
            'Latence_ms': row['latence_moy_ms'] if not pd.isna(row['latence_moy_ms']) else None
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('MAE_m')
    summary_df.to_csv('results/comparaison_complete_dav2.csv', index=False)
    print("✓ Tableau combiné sauvegardé: comparaison_complete_dav2.csv")

if __name__ == "__main__":
    print("="*60)
    print("COMPARAISON COMPLÈTE DES RÉSULTATS DAV2")
    print("="*60)
    
    # Charger tous les résultats
    all_results = charger_tous_les_resultats()
    
    # Combiner les résultats test
    combined_df = combiner_resultats_test(all_results)
    
    if combined_df is not None:
        print(f"\n✓ {len(combined_df)} modèles combinés")
        print("\nModèles disponibles:")
        for model in combined_df.index:
            print(f"  - {model}")
        
        # Générer les plots
        dav2_models = generer_plots_comparaison_complete(combined_df)
        
        # Sauvegarder le tableau combiné
        sauvegarder_tableau_combiné(dav2_models)
        
        print(f"\n✓ Tous les plots de comparaison complète ont été générés dans results/plots/")
    else:
        print("❌ Impossible de combiner les résultats")