"""
Script d'évaluation complète de tous les modèles (zero-shot à finetuned) sur le test set
avec métriques détaillées et test de Wilcoxon
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import wilcoxon
import seaborn as sns

def charger_predictions_test():
    """Charge toutes les prédictions disponibles sur le test set"""
    results_dir = "results"
    predictions = {}
    
    # Mapping des fichiers de prédictions test (UNIQUEMENT stratified split pour comparaison équitable)
    test_files = {
        # Zero-shot models (stratified split uniquement)
        "DepthPro": "predictions_depthpro_stratified.npz",
        "DAV2_zeroshot": "predictions_dav2_zeroshot_stratified.npz",
        
        # DAV2 Finetuned models (stratified split uniquement)
        "DAV2_finetune": "predictions_dav2_finetune_test_stratified.npz",
        
        # DAV2 Multi-LR models (stratified split uniquement)
        "DAV2_LR_2e-5": "predictions_dav2_lr_2e-05_test_stratified.npz",
        
        # HTC-DC Net (stratified split)
        "HTC_DC_Net": "predictions_htc_dc_net_test_stratified.npz",
    }
    
    for model_name, filename in test_files.items():
        filepath = os.path.join(results_dir, filename)
        if os.path.exists(filepath):
            try:
                data = np.load(filepath)
                predictions[model_name] = {
                    'y_pred': data['y_pred'],
                    'y_true': data['y_true']
                }
                print(f"✓ {model_name}: {len(data['y_pred'])} échantillons")
            except Exception as e:
                print(f"✗ {model_name}: erreur de chargement - {e}")
        else:
            print(f"⚠ {model_name}: fichier non trouvé")
    
    return predictions

def calculer_metriques_avec_ci(y_pred, y_true, confidence=0.95):
    """Calcule les métriques avec intervalles de confiance"""
    from scipy import stats as scipy_stats
    
    # Métriques de base
    mae = np.mean(np.abs(y_pred - y_true))
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    
    # Erreur relative
    mask = y_true > 0.1  # éviter division par zéro
    if mask.sum() > 0:
        abs_rel = np.mean(np.abs(y_pred[mask] - y_true[mask]) / y_true[mask])
    else:
        abs_rel = np.nan
    
    # NMAD (formule correcte : median absolute deviation des erreurs centrées)
    err = y_pred - y_true
    nmad = 1.4826 * np.median(np.abs(err - np.median(err)))
    
    # Corrélation
    correlation = np.corrcoef(y_pred, y_true)[0, 1]
    
    # Biais
    bias = np.mean(y_pred - y_true)
    
    # Intervalles de confiance par bootstrap
    n_bootstrap = 1000
    n_samples = len(y_pred)
    indices = np.arange(n_samples)
    
    mae_bootstrap = []
    rmse_bootstrap = []
    
    for _ in range(n_bootstrap):
        sample_indices = np.random.choice(indices, n_samples, replace=True)
        mae_bootstrap.append(np.mean(np.abs(y_pred[sample_indices] - y_true[sample_indices])))
        rmse_bootstrap.append(np.sqrt(np.mean((y_pred[sample_indices] - y_true[sample_indices]) ** 2)))
    
    mae_bootstrap = np.array(mae_bootstrap)
    rmse_bootstrap = np.array(rmse_bootstrap)
    
    alpha = 1 - confidence
    mae_ci = np.percentile(mae_bootstrap, [alpha/2 * 100, (1 - alpha/2) * 100])
    rmse_ci = np.percentile(rmse_bootstrap, [alpha/2 * 100, (1 - alpha/2) * 100])
    
    return {
        'MAE': mae,
        'MAE_CI': mae_ci,
        'RMSE': rmse, 
        'RMSE_CI': rmse_ci,
        'Abs_Rel': abs_rel,
        'NMAD': nmad,
        'Correlation': correlation,
        'Bias': bias,
        'n_samples': n_samples
    }

def evaluer_tous_les_modeles(predictions):
    """Évalue tous les modèles avec métriques détaillées"""
    results = {}
    
    for model_name, data in predictions.items():
        y_pred = data['y_pred']
        y_true = data['y_true']
        
        metrics = calculer_metriques_avec_ci(y_pred, y_true)
        results[model_name] = metrics
        
        print(f"\n📊 {model_name}:")
        print(f"  MAE: {metrics['MAE']:.4f} [{metrics['MAE_CI'][0]:.4f}, {metrics['MAE_CI'][1]:.4f}]")
        print(f"  RMSE: {metrics['RMSE']:.4f} [{metrics['RMSE_CI'][0]:.4f}, {metrics['RMSE_CI'][1]:.4f}]")
        print(f"  Abs_Rel: {metrics['Abs_Rel']:.4f}")
        print(f"  NMAD: {metrics['NMAD']:.4f}")
        print(f"  Correlation: {metrics['Correlation']:.4f}")
        print(f"  Bias: {metrics['Bias']:.4f}")
    
    return results

def test_wilcoxon_apparie(predictions):
    """Effectue des tests de Wilcoxon appariés entre tous les modèles"""
    models = list(predictions.keys())
    n_models = len(models)
    
    # Créer une matrice de p-values
    p_values = np.zeros((n_models, n_models))
    p_values[:] = np.nan
    
    # Statistiques de test
    test_stats = np.zeros((n_models, n_models))
    test_stats[:] = np.nan
    
    print("\n🔬 Tests de Wilcoxon appariés:")
    
    for i in range(n_models):
        for j in range(i+1, n_models):
            model1 = models[i]
            model2 = models[j]
            
            # Calculer les erreurs absolues pour chaque modèle
            errors1 = np.abs(predictions[model1]['y_pred'] - predictions[model1]['y_true'])
            errors2 = np.abs(predictions[model2]['y_pred'] - predictions[model2]['y_true'])
            
            # S'assurer que les deux vecteurs ont la même longueur
            min_len = min(len(errors1), len(errors2))
            errors1 = errors1[:min_len]
            errors2 = errors2[:min_len]
            
            try:
                # Test de Wilcoxon apparié
                stat, p_value = wilcoxon(errors1, errors2)
                p_values[i, j] = p_value
                p_values[j, i] = p_value
                test_stats[i, j] = stat
                test_stats[j, i] = stat
                
                significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                print(f"  {model1} vs {model2}: p={p_value:.4e} {significance}")
                
            except Exception as e:
                print(f"  {model1} vs {model2}: erreur - {e}")
    
    return {
        'p_values': p_values,
        'test_stats': test_stats,
        'models': models
    }

def creer_tableau_comparatif(results):
    """Crée un tableau comparatif des résultats"""
    data = []
    for model_name, metrics in results.items():
        data.append({
            'Modèle': model_name,
            'MAE (m)': f"{metrics['MAE']:.4f} [{metrics['MAE_CI'][0]:.4f}, {metrics['MAE_CI'][1]:.4f}]",
            'RMSE (m)': f"{metrics['RMSE']:.4f} [{metrics['RMSE_CI'][0]:.4f}, {metrics['RMSE_CI'][1]:.4f}]",
            'Abs_Rel': f"{metrics['Abs_Rel']:.4f}",
            'NMAD': f"{metrics['NMAD']:.4f}",
            'Corrélation': f"{metrics['Correlation']:.4f}",
            'Biais (m)': f"{metrics['Bias']:.4f}",
            'N_échantillons': metrics['n_samples']
        })
    
    df = pd.DataFrame(data)
    df = df.sort_values('Modèle')
    return df

def visualiser_resultats(results, wilcoxon_results):
    """Génère des visualisations des résultats"""
    os.makedirs("results/plots", exist_ok=True)
    
    models = list(results.keys())
    
    # 1. Graphique comparatif MAE avec intervalles de confiance
    fig, ax = plt.subplots(figsize=(12, 6))
    
    mae_values = [results[m]['MAE'] for m in models]
    mae_errors = [
        [results[m]['MAE'] - results[m]['MAE_CI'][0], results[m]['MAE_CI'][1] - results[m]['MAE']]
        for m in models
    ]
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
    bars = ax.barh(models, mae_values, xerr=np.array(mae_errors).T, 
                   color=colors, alpha=0.7, capsize=5)
    
    ax.set_xlabel('MAE (m)')
    ax.set_title('Comparaison des MAE avec intervalles de confiance 95%')
    ax.grid(axis='x', alpha=0.3)
    
    # Ajouter les valeurs
    for i, (bar, value) in enumerate(zip(bars, mae_values)):
        ax.text(value, bar.get_y() + bar.get_height()/2, 
               f'{value:.3f}', va='center', ha='left', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('results/plots/comparaison_mae_all_models.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Graphique MAE sauvegardé")
    
    # 2. Heatmap des p-values Wilcoxon
    fig, ax = plt.subplots(figsize=(10, 8))
    
    p_values = wilcoxon_results['p_values']
    models_short = [m.replace('DAV2_', '').replace('DepthPro', 'DP') for m in models]
    
    # Masquer la diagonale et les valeurs NaN
    mask = np.triu(np.ones_like(p_values, dtype=bool))
    p_values_masked = np.ma.masked_where(mask, p_values)
    
    im = ax.imshow(p_values_masked, cmap='RdYlGn_r', vmin=0, vmax=0.05)
    
    ax.set_xticks(range(len(models)))
    ax.set_yticks(range(len(models)))
    ax.set_xticklabels(models_short, rotation=45, ha='right')
    ax.set_yticklabels(models_short)
    
    # Ajouter les valeurs de p
    for i in range(len(models)):
        for j in range(len(models)):
            if i != j and not np.isnan(p_values[i, j]):
                text = ax.text(j, i, f'{p_values[i, j]:.3f}',
                             ha="center", va="center", color="black", fontsize=8)
    
    ax.set_title('Matrice des p-values (Test de Wilcoxon apparié)')
    plt.colorbar(im, ax=ax, label='p-value')
    plt.tight_layout()
    plt.savefig('results/plots/wilcoxon_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Heatmap Wilcoxon sauvegardée")
    
    # 3. Radar chart des performances normalisées
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    metrics_radar = ['MAE', 'RMSE', 'Abs_Rel', 'NMAD']
    normalized_data = {}
    
    for model in models:
        row = results[model]
        normalized_data[model] = [
            1 - (row['MAE'] / max(results[m]['MAE'] for m in models)),
            1 - (row['RMSE'] / max(results[m]['RMSE'] for m in models)),
            1 - (row['Abs_Rel'] / max(results[m]['Abs_Rel'] for m in models if not np.isnan(results[m]['Abs_Rel']))),
            1 - (row['NMAD'] / max(results[m]['NMAD'] for m in models if not np.isnan(results[m]['NMAD'])))
        ]
    
    angles = np.linspace(0, 2 * np.pi, len(metrics_radar), endpoint=False).tolist()
    angles += angles[:1]
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    
    for idx, model in enumerate(models):
        values = normalized_data[model]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[idx])
        ax.fill(angles, values, alpha=0.15, color=colors[idx])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(['MAE', 'RMSE', 'Abs Rel', 'NMAD'], fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
    ax.set_title('Radar des Performances (Normalisé)', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('results/plots/radar_all_models.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Radar chart sauvegardé")

def sauvegarder_resultats(results, wilcoxon_results):
    """Sauvegarde tous les résultats en CSV"""
    os.makedirs("results", exist_ok=True)
    
    # Tableau comparatif
    tableau = creer_tableau_comparatif(results)
    tableau.to_csv('results/comparaison_complete_test.csv', index=False)
    print("✓ Tableau comparatif sauvegardé")
    
    # Matrice de p-values
    p_values_df = pd.DataFrame(
        wilcoxon_results['p_values'],
        index=wilcoxon_results['models'],
        columns=wilcoxon_results['models']
    )
    p_values_df.to_csv('results/wilcoxon_p_values.csv')
    print("✓ Matrice de p-values sauvegardée")
    
    # Statistiques de test
    test_stats_df = pd.DataFrame(
        wilcoxon_results['test_stats'],
        index=wilcoxon_results['models'],
        columns=wilcoxon_results['models']
    )
    test_stats_df.to_csv('results/wilcoxon_test_stats.csv')
    print("✓ Statistiques de test sauvegardées")

def main():
    print("="*70)
    print("ÉVALUATION COMPLÈTE DE TOUS LES MODÈLES (ZERO-SHOT À FINETUNED)")
    print("="*70)
    
    # 1. Charger les prédictions
    print("\n📂 Chargement des prédictions de test...")
    predictions = charger_predictions_test()
    
    if not predictions:
        print("❌ Aucune prédiction trouvée. Vérifiez les fichiers dans results/")
        return
    
    print(f"\n✓ {len(predictions)} modèles chargés")
    
    # 2. Évaluer tous les modèles
    print("\n📊 Calcul des métriques pour chaque modèle...")
    results = evaluer_tous_les_modeles(predictions)
    
    # 3. Tests de Wilcoxon
    print("\n🔬 Tests de Wilcoxon appariés...")
    wilcoxon_results = test_wilcoxon_apparie(predictions)
    
    # 4. Créer le tableau comparatif
    print("\n📋 Création du tableau comparatif...")
    tableau = creer_tableau_comparatif(results)
    print("\n" + "="*70)
    print("TABLEAU COMPARATIF FINAL (TEST SET)")
    print("="*70)
    print(tableau.to_string(index=False))
    
    # 5. Visualisations
    print("\n📈 Génération des visualisations...")
    visualiser_resultats(results, wilcoxon_results)
    
    # 6. Sauvegarder les résultats
    print("\n💾 Sauvegarde des résultats...")
    sauvegarder_resultats(results, wilcoxon_results)
    
    # 7. Identifier le meilleur modèle
    print("\n🏆 Identification du meilleur modèle...")
    best_model_mae = min(results.keys(), key=lambda m: results[m]['MAE'])
    best_model_rmse = min(results.keys(), key=lambda m: results[m]['RMSE'])
    
    print(f"\nMeilleur modèle (MAE): {best_model_mae} - {results[best_model_mae]['MAE']:.4f} m")
    print(f"Meilleur modèle (RMSE): {best_model_rmse} - {results[best_model_rmse]['RMSE']:.4f} m")
    
    print("\n✓ Évaluation complète terminée!")
    print("  Résultats sauvegardés dans results/")
    print("  Graphiques sauvegardés dans results/plots/")

if __name__ == "__main__":
    main()