"""
pipelines/07_compare_models.py

Script de comparaison et évaluation des trois modèles :
- HTC-DC Net (entraîné)
- Depth Anything V2 (zero-shot et fine-tuné)
- DepthPro (zero-shot)

Compare les métriques de performance et génère des graphiques et rapports.
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend non-interactif
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import glob

# Configuration
RESULTS_DIR = "results"
OUTPUT_DIR = "results/model_comparison"
METRICS_FILES = {
    'depthpro_zeroshot': 'metrics_zero_shot_stratified.csv',
    'dav2_zeroshot': 'metrics_zero_shot_stratified.csv', 
    'dav2_finetune': 'metrics_dav2_finetune_stratified.csv',
    'htc_dc_net': None  # À extraire des logs d'entraînement
}

# Configuration pour les graphiques
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12


def charger_resultats_htc_dc_net():
    """Extrait les résultats finaux de HTC-DC Net depuis les logs d'entraînement."""
    training_dirs = glob.glob(os.path.join(RESULTS_DIR, "htc_dc_net_training", "*"))
    if not training_dirs:
        return None
    
    # Prendre le répertoire le plus récent
    latest_dir = max(training_dirs, key=os.path.getmtime)
    log_file = os.path.join(latest_dir, "training_log.csv")
    
    if not os.path.exists(log_file):
        return None
    
    try:
        df = pd.read_csv(log_file)
        # Filtrer les lignes de validation
        val_rows = df[df['mae'].notna()]
        if val_rows.empty:
            return None
        
        # Prendre la dernière epoch
        final_results = val_rows.iloc[-1]
        
        return {
            'modele': 'HTC-DC-Net',
            'MAE': final_results['mae'],
            'RMSE_B': final_results['rmse'],
            'NMAD': np.nan,  # Non disponible dans les logs
            'Biais': np.nan,  # Non disponible dans les logs
            'Erreur_relative': np.nan,  # Non disponible dans les logs
            'Correlation': np.nan,  # Non disponible dans les logs
            'Couverture': 1.0,
            'n_batiments': np.nan,  # Non disponible dans les logs
            'latence_moy_ms': np.nan,  # Non disponible dans les logs
            'epoch': int(final_results['epoch']),
            'val_loss': final_results.get('val/loss_total', np.nan)
        }
    except Exception as e:
        print(f"Erreur lors de la lecture des logs HTC-DC Net: {e}")
        return None


def charger_tous_les_resultats():
    """Charge tous les résultats disponibles."""
    resultats = []
    
    # Charger les résultats depuis les fichiers CSV
    # Charger le fichier zeroshot qui contient DepthPro et DAV2
    zeroshot_file = os.path.join(RESULTS_DIR, 'metrics_zero_shot_stratified.csv')
    if os.path.exists(zeroshot_file):
        try:
            df_zeroshot = pd.read_csv(zeroshot_file)
            # Extraire DepthPro
            depthpro_df = df_zeroshot[df_zeroshot['modele'] == 'DepthPro']
            if not depthpro_df.empty:
                resultats.append(depthpro_df.iloc[0].to_dict())
                resultats[-1]['modele'] = 'DepthPro_ZeroShot'
            
            # Extraire DAV2 zeroshot
            dav2_zeroshot_df = df_zeroshot[df_zeroshot['modele'] == 'DAV2_zeroshot']
            if not dav2_zeroshot_df.empty:
                resultats.append(dav2_zeroshot_df.iloc[0].to_dict())
                resultats[-1]['modele'] = 'DAV2_ZeroShot'
        except Exception as e:
            print(f"Erreur lors de la lecture de {zeroshot_file}: {e}")
    
    # Charger DAV2 finetune
    finetune_file = os.path.join(RESULTS_DIR, 'metrics_dav2_finetune_stratified.csv')
    if os.path.exists(finetune_file):
        try:
            df_finetune = pd.read_csv(finetune_file)
            dav2_finetune_df = df_finetune[df_finetune['modele'] == 'DAV2_finetune']
            if not dav2_finetune_df.empty:
                resultats.append(dav2_finetune_df.iloc[0].to_dict())
                resultats[-1]['modele'] = 'DAV2_FineTune'
        except Exception as e:
            print(f"Erreur lors de la lecture de {finetune_file}: {e}")
    
    # Ajouter les résultats HTC-DC Net
    htc_results = charger_resultats_htc_dc_net()
    if htc_results:
        resultats.append(htc_results)
    
    return pd.DataFrame(resultats)


def generer_graphique_comparaison_barres(df, metric_col, title, ylabel, output_file):
    """Génère un graphique en barres pour une métrique donnée."""
    if metric_col not in df.columns or df[metric_col].isna().all():
        print(f"Métrique {metric_col} non disponible, skip...")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Filtrer les valeurs non-NaN
    plot_data = df[['modele', metric_col]].dropna()
    
    if plot_data.empty:
        print(f"Pas de données valides pour {metric_col}")
        return
    
    # Créer le graphique en barres
    bars = ax.bar(plot_data['modele'], plot_data[metric_col], 
                  color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'][:len(plot_data)])
    
    # Ajouter les valeurs sur les barres
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel('Modèle', fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Graphique sauvegardé: {output_file}")


def generer_graphique_comparaison_multiple(df, metrics, title, output_file):
    """Génère un graphique comparatif pour plusieurs métriques."""
    available_metrics = [m for m in metrics if m in df.columns and not df[m].isna().all()]
    
    if len(available_metrics) < 2:
        print(f"Pas assez de métriques disponibles pour le graphique multiple")
        return
    
    fig, axes = plt.subplots(1, len(available_metrics), figsize=(5*len(available_metrics), 6))
    if len(available_metrics) == 1:
        axes = [axes]
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    for idx, metric in enumerate(available_metrics):
        ax = axes[idx]
        plot_data = df[['modele', metric]].dropna()
        
        if plot_data.empty:
            continue
        
        bars = ax.bar(plot_data['modele'], plot_data[metric], color=colors[:len(plot_data)])
        
        # Ajouter les valeurs sur les barres
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Modèle', fontsize=10, fontweight='bold')
        ax.set_ylabel(metric, fontsize=10, fontweight='bold')
        ax.set_title(metric, fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Graphique multiple sauvegardé: {output_file}")


def generer_graphique_radar(df, metrics, title, output_file):
    """Génère un graphique radar pour comparer les modèles sur plusieurs métriques."""
    available_metrics = [m for m in metrics if m in df.columns and not df[m].isna().all()]
    
    if len(available_metrics) < 3:
        print(f"Pas assez de métriques disponibles pour le graphique radar")
        return
    
    # Normaliser les métriques pour le radar (inverser les métriques d'erreur)
    plot_data = df[['modele'] + available_metrics].dropna()
    
    if plot_data.empty:
        return
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Nombre de métriques
    n_metrics = len(available_metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Fermer le cercle
    
    # Couleurs pour chaque modèle
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    for idx, (_, row) in enumerate(plot_data.iterrows()):
        values = row[available_metrics].values
        
        # Pour les métriques d'erreur, on inverse (1 - normalized)
        # Pour les métriques de performance (corrélation), on garde tel quel
        normalized_values = []
        for metric, value in zip(available_metrics, values):
            if metric in ['MAE', 'RMSE_B', 'NMAD', 'Erreur_relative', 'Biais_abs']:
                # Normaliser et inverser les métriques d'erreur
                max_val = plot_data[metric].max()
                if max_val > 0:
                    norm_val = 1 - (value / max_val)
                else:
                    norm_val = 1
            else:
                # Normaliser les métriques de performance
                max_val = plot_data[metric].max()
                if max_val > 0:
                    norm_val = value / max_val
                else:
                    norm_val = 1
            normalized_values.append(norm_val)
        
        normalized_values += normalized_values[:1]  # Fermer le cercle
        
        ax.plot(angles, normalized_values, 'o-', linewidth=2, 
                label=row['modele'], color=colors[idx % len(colors)])
        ax.fill(angles, normalized_values, alpha=0.15, color=colors[idx % len(colors)])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(available_metrics, size=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=8)
    ax.grid(True)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Graphique radar sauvegardé: {output_file}")


def generer_tableau_comparatif(df, output_file):
    """Génère un tableau comparatif formaté."""
    # Sélectionner et ordonner les colonnes importantes
    important_cols = ['modele', 'MAE', 'RMSE_B', 'NMAD', 'Biais', 'Erreur_relative', 
                     'Correlation', 'latence_moy_ms']
    
    available_cols = [col for col in important_cols if col in df.columns]
    table_df = df[available_cols].copy()
    
    # Renommer les colonnes pour l'affichage
    column_names = {
        'modele': 'Modèle',
        'MAE': 'MAE',
        'RMSE_B': 'RMSE', 
        'NMAD': 'NMAD',
        'Biais': 'Biais',
        'Erreur_relative': 'Erreur Relative',
        'Correlation': 'Corrélation',
        'latence_moy_ms': 'Latence (ms)'
    }
    table_df = table_df.rename(columns=column_names)
    
    # Formater les valeurs numériques
    for col in table_df.columns:
        if col != 'Modèle':
            table_df[col] = table_df[col].apply(lambda x: f'{x:.3f}' if pd.notna(x) else 'N/A')
    
    # Sauvegarder en CSV
    table_df.to_csv(output_file, index=False)
    print(f"Tableau comparatif sauvegardé: {output_file}")
    
    return table_df


def calculer_ameliorations(df):
    """Calcule les améliorations relatives par rapport au meilleur modèle."""
    if 'MAE' not in df.columns or df['MAE'].isna().all():
        return None
    
    # Trouver le meilleur modèle (plus faible MAE)
    best_mae_idx = df['MAE'].idxmin()
    best_model = df.loc[best_mae_idx, 'modele']
    best_mae = df.loc[best_mae_idx, 'MAE']
    
    ameliorations = []
    for idx, row in df.iterrows():
        if pd.notna(row['MAE']) and row['MAE'] > 0:
            improvement = ((row['MAE'] - best_mae) / row['MAE']) * 100
            ameliorations.append({
                'modele': row['modele'],
                'MAE': row['MAE'],
                'vs_meilleur_pct': improvement if idx != best_mae_idx else 0.0,
                'est_meilleur': idx == best_mae_idx
            })
    
    return pd.DataFrame(ameliorations)


def generer_rapport_html(df, ameliorations_df, output_dir):
    """Génère un rapport HTML complet de comparaison."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculer des statistiques
    best_mae_model = df.loc[df['MAE'].idxmin(), 'modele'] if 'MAE' in df.columns else 'N/A'
    best_rmse_model = df.loc[df['RMSE_B'].idxmin(), 'modele'] if 'RMSE_B' in df.columns else 'N/A'
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Comparaison des Modèles de Prédiction de Hauteur</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #f5f7fa; }}
            .container {{ max-width: 1400px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
            h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 15px; margin-bottom: 30px; }}
            h2 {{ color: #34495e; margin-top: 30px; margin-bottom: 15px; }}
            .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 25px 0; }}
            .summary-card {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
            .summary-card h3 {{ margin: 0 0 10px 0; font-size: 16px; opacity: 0.9; }}
            .summary-card .value {{ font-size: 28px; font-weight: bold; }}
            .table-container {{ overflow-x: auto; margin: 20px 0; }}
            table {{ width: 100%; border-collapse: collapse; background-color: white; border-radius: 8px; overflow: hidden; }}
            th {{ background-color: #34495e; color: white; padding: 12px 15px; text-align: left; font-weight: 600; }}
            td {{ padding: 12px 15px; border-bottom: 1px solid #ddd; }}
            tr:hover {{ background-color: #f8f9fa; }}
            .best {{ background-color: #d4edda; font-weight: bold; }}
            .graph-container {{ text-align: center; margin: 30px 0; padding: 20px; background-color: #f8f9fa; border-radius: 8px; }}
            .graph-container img {{ max-width: 100%; height: auto; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .improvement {{ color: #27ae60; font-weight: bold; }}
            .worse {{ color: #e74c3c; font-weight: bold; }}
            .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #7f8c8d; font-size: 14px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📊 Comparaison des Modèles de Prédiction de Hauteur</h1>
            <p><strong>Date:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            
            <h2>🎯 Résumé des Performances</h2>
            <div class="summary">
                <div class="summary-card">
                    <h3>Meilleur MAE</h3>
                    <div class="value">{best_mae_model}</div>
                </div>
                <div class="summary-card">
                    <h3>Meilleur RMSE</h3>
                    <div class="value">{best_rmse_model}</div>
                </div>
                <div class="summary-card">
                    <h3>Nombre de Modèles</h3>
                    <div class="value">{len(df)}</div>
                </div>
            </div>
            
            <h2>📈 Tableau Comparatif</h2>
            <div class="table-container">
    """
    
    # Générer le tableau HTML
    important_cols = ['modele', 'MAE', 'RMSE_B', 'NMAD', 'Biais', 'Erreur_relative', 'Correlation', 'latence_moy_ms']
    available_cols = [col for col in important_cols if col in df.columns]
    
    column_names = {
        'modele': 'Modèle',
        'MAE': 'MAE',
        'RMSE_B': 'RMSE',
        'NMAD': 'NMAD', 
        'Biais': 'Biais',
        'Erreur_relative': 'Erreur Rel.',
        'Correlation': 'Corrélation',
        'latence_moy_ms': 'Latence (ms)'
    }
    
    html_content += "<table><tr>"
    for col in available_cols:
        html_content += f"<th>{column_names.get(col, col)}</th>"
    html_content += "</tr>"
    
    for idx, row in df.iterrows():
        row_class = "best" if (idx == df['MAE'].idxmin() and 'MAE' in df.columns) else ""
        html_content += f"<tr class='{row_class}'>"
        for col in available_cols:
            value = row[col]
            if pd.notna(value):
                try:
                    if col in ['MAE', 'RMSE_B', 'NMAD', 'Erreur_relative']:
                        html_content += f"<td>{float(value):.3f}</td>"
                    elif col == 'latence_moy_ms':
                        html_content += f"<td>{float(value):.1f}</td>"
                    else:
                        html_content += f"<td>{float(value):.3f}</td>"
                except (ValueError, TypeError):
                    html_content += f"<td>{value}</td>"
            else:
                html_content += "<td>N/A</td>"
        html_content += "</tr>"
    
    html_content += """
            </table>
            </div>
            
            <h2>📉 Améliorations Relatives</h2>
    """
    
    if ameliorations_df is not None:
        html_content += "<div class='table-container'><table><tr><th>Modèle</th><th>MAE</th><th>vs Meilleur</th></tr>"
        for _, row in ameliorations_df.iterrows():
            if row['est_meilleur']:
                improvement_html = "<span class='improvement'>★ Meilleur</span>"
            else:
                improvement_class = "improvement" if row['vs_meilleur_pct'] > 0 else "worse"
                improvement_html = f"<span class='{improvement_class}'>{row['vs_meilleur_pct']:.1f}%</span>"
            
            html_content += f"""
                <tr>
                    <td>{row['modele']}</td>
                    <td>{row['MAE']:.3f}</td>
                    <td>{improvement_html}</td>
                </tr>
            """
        html_content += "</table></div>"
    else:
        html_content += "<p>Données non disponibles pour le calcul des améliorations.</p>"
    
    # Ajouter les graphiques
    graph_files = [
        ('mae_comparison.png', 'Comparaison MAE'),
        ('rmse_comparison.png', 'Comparaison RMSE'),
        ('error_metrics_comparison.png', 'Métriques d\'Erreur'),
        ('radar_comparison.png', 'Graphique Radar')
    ]
    
    html_content += "<h2>📊 Graphiques Comparatifs</h2>"
    for graph_file, title in graph_files:
        graph_path = os.path.join(output_dir, graph_file)
        if os.path.exists(graph_path):
            html_content += f"""
            <div class="graph-container">
                <h3>{title}</h3>
                <img src="{graph_file}" alt="{title}">
            </div>
            """
    
    html_content += f"""
            <div class="footer">
                <p>Rapport généré automatiquement par le script de comparaison des modèles.</p>
                <p>Modèles comparés: DepthPro (Zero-Shot), Depth Anything V2 (Zero-Shot & Fine-Tune), HTC-DC Net</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    with open(os.path.join(output_dir, 'rapport_comparaison.html'), 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Rapport HTML sauvegardé: {output_dir}/rapport_comparaison.html")


def main():
    """Fonction principale."""
    print("=" * 70)
    print("🚀 Script de Comparaison des Modèles de Prédiction de Hauteur")
    print("=" * 70)
    
    # Créer le répertoire de sortie
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(OUTPUT_DIR, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 1. Charger tous les résultats
        print("\n📂 Chargement des résultats...")
        df = charger_tous_les_resultats()
        
        if df.empty:
            print("❌ Aucun résultat trouvé!")
            return 1
        
        print(f"✅ {len(df)} modèles chargés:")
        for model in df['modele']:
            print(f"   - {model}")
        
        # 2. Générer les graphiques individuels
        print("\n📊 Génération des graphiques...")
        
        generer_graphique_comparaison_barres(
            df, 'MAE', 
            'Comparaison du MAE par Modèle', 
            'MAE (mètres)',
            os.path.join(output_dir, 'mae_comparison.png')
        )
        
        generer_graphique_comparaison_barres(
            df, 'RMSE_B',
            'Comparaison du RMSE par Modèle',
            'RMSE (mètres)', 
            os.path.join(output_dir, 'rmse_comparison.png')
        )
        
        generer_graphique_comparaison_barres(
            df, 'latence_moy_ms',
            'Comparaison de la Latence par Modèle',
            'Latence (ms)',
            os.path.join(output_dir, 'latency_comparison.png')
        )
        
        # 3. Générer les graphiques multiples
        error_metrics = ['MAE', 'RMSE_B', 'NMAD']
        generer_graphique_comparaison_multiple(
            df, error_metrics,
            'Comparaison des Métriques d\'Erreur',
            os.path.join(output_dir, 'error_metrics_comparison.png')
        )
        
        performance_metrics = ['Correlation', 'Couverture']
        generer_graphique_comparaison_multiple(
            df, performance_metrics,
            'Comparaison des Métriques de Performance',
            os.path.join(output_dir, 'performance_metrics_comparison.png')
        )
        
        # 4. Générer le graphique radar
        radar_metrics = ['MAE', 'RMSE_B', 'NMAD', 'Correlation']
        generer_graphique_radar(
            df, radar_metrics,
            'Comparaison Radar des Modèles',
            os.path.join(output_dir, 'radar_comparison.png')
        )
        
        # 5. Générer le tableau comparatif
        print("\n📋 Génération du tableau comparatif...")
        tableau_df = generer_tableau_comparatif(
            df, 
            os.path.join(output_dir, 'tableau_comparatif.csv')
        )
        
        # 6. Calculer les améliorations
        print("\n📈 Calcul des améliorations relatives...")
        ameliorations_df = calculer_ameliorations(df)
        
        if ameliorations_df is not None:
            print("Améliorations relatives:")
            for _, row in ameliorations_df.iterrows():
                if row['est_meilleur']:
                    print(f"   ★ {row['modele']}: {row['MAE']:.3f} (MEILLEUR)")
                else:
                    print(f"   • {row['modele']}: {row['MAE']:.3f} ({row['vs_meilleur_pct']:.1f}% vs meilleur)")
        
        # 7. Générer le rapport HTML
        print("\n📄 Génération du rapport HTML...")
        generer_rapport_html(df, ameliorations_df, output_dir)
        
        print("\n" + "=" * 70)
        print(f"✅ Comparaison terminée! Résultats sauvegardés dans: {output_dir}")
        print("=" * 70)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Erreur lors de la comparaison: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())