"""
Script pour extraire et afficher les statistiques du plot loss_evolution.png de HTC-DC Net
"""
import os
import re
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def extract_stats_from_html(html_path):
    """Extrait les statistiques depuis le rapport HTML"""
    with open(html_path, 'r') as f:
        html_content = f.read()
    
    # Chercher les métriques dans le HTML
    metrics = {}
    
    # Patterns pour extraire les valeurs
    patterns = {
        'best_mae': r'Meilleur MAE.*?metric-value">([0-9.]+|N/A)',
        'best_mae_epoch': r'Epoch: ([0-9]+|N/A)',
        'best_rmse': r'Meilleur RMSE.*?metric-value">([0-9.]+|N/A)',
        'best_rmse_epoch': r'Epoch: ([0-9]+|N/A)',
        'final_mae': r'MAE Final.*?metric-value">([0-9.]+|N/A)',
        'final_rmse': r'RMSE Final.*?metric-value">([0-9.]+|N/A)',
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, html_content, re.DOTALL)
        if match:
            metrics[key] = match.group(1)
    
    return metrics

def find_corresponding_training_log(evaluation_date):
    """Trouve le training_log correspondant à la date d'évaluation"""
    # L'évaluation est du 20260909_110137 (9 septembre 2026)
    # Chercher dans les checkpoints HTC-DC-Net
    htc_checkpoints = "third_party/HTC-DC-Net/checkpoints/htcdc"
    
    if not os.path.exists(htc_checkpoints):
        return None
    
    # Chercher les dossiers autour de cette date
    # Les dates semblent être au format YYMMDD_HHMMSS
    target_date = "0909"  # 9 septembre
    
    matching_dirs = []
    for dir_name in os.listdir(htc_checkpoints):
        if target_date in dir_name:
            matching_dirs.append(os.path.join(htc_checkpoints, dir_name))
    
    if matching_dirs:
        # Prendre le plus récent
        latest_dir = max(matching_dirs, key=os.path.getmtime)
        log_file = os.path.join(latest_dir, "training_log.csv")
        if os.path.exists(log_file):
            return log_file
    
    # Sinon, chercher le training_log le plus récent
    training_logs = []
    for dir_name in os.listdir(htc_checkpoints):
        log_path = os.path.join(htc_checkpoints, dir_name, "training_log.csv")
        if os.path.exists(log_path):
            training_logs.append(log_path)
    
    if training_logs:
        return max(training_logs, key=os.path.getmtime)
    
    return None

def analyze_training_log(log_path):
    """Analyse le training_log pour extraire les statistiques"""
    if not log_path or not os.path.exists(log_path):
        return None
    
    df = pd.read_csv(log_path)
    
    # Filtrer les lignes de validation
    val_rows = df[df['mae'].notna()]
    
    if val_rows.empty:
        return None
    
    stats = {
        'total_epochs': len(df),
        'validation_epochs': len(val_rows),
        'best_mae': val_rows['mae'].min(),
        'best_mae_epoch': val_rows.loc[val_rows['mae'].idxmin(), 'epoch'],
        'best_rmse': val_rows['rmse'].min(),
        'best_rmse_epoch': val_rows.loc[val_rows['rmse'].idxmin(), 'epoch'],
        'final_mae': val_rows.iloc[-1]['mae'],
        'final_rmse': val_rows.iloc[-1]['rmse'],
        'initial_mae': val_rows.iloc[0]['mae'],
        'initial_rmse': val_rows.iloc[0]['rmse'],
    }
    
    return stats, df

def display_stats(stats, df=None):
    """Affiche les statistiques de manière lisible"""
    print("="*60)
    print("STATISTIQUES D'ENTRAÎNEMENT HTC-DC NET")
    print("="*60)
    
    if stats:
        print(f"\n📊 MÉTRIQUES PRINCIPALES:")
        print(f"  Meilleur MAE    : {stats['best_mae']:.4f} m (epoch {int(stats['best_mae_epoch'])})")
        print(f"  Meilleur RMSE   : {stats['best_rmse']:.4f} m (epoch {int(stats['best_rmse_epoch'])})")
        print(f"  MAE final      : {stats['final_mae']:.4f} m")
        print(f"  RMSE final     : {stats['final_rmse']:.4f} m")
        print(f"  MAE initial    : {stats['initial_mae']:.4f} m")
        print(f"  RMSE initial   : {stats['initial_rmse']:.4f} m")
        
        print(f"\n📈 PROGRÈS:")
        mae_improvement = ((stats['initial_mae'] - stats['best_mae']) / stats['initial_mae']) * 100
        rmse_improvement = ((stats['initial_rmse'] - stats['best_rmse']) / stats['initial_rmse']) * 100
        print(f"  Amélioration MAE  : {mae_improvement:.1f}%")
        print(f"  Amélioration RMSE : {rmse_improvement:.1f}%")
        
        print(f"\n⏱️  DURÉE:")
        print(f"  Epochs totaux     : {stats['total_epochs']}")
        print(f"  Epochs validation : {stats['validation_epochs']}")
    
    if df is not None:
        print(f"\n📋 DÉTAILS PAR EPOCH (5 premières et dernières):")
        val_rows = df[df['mae'].notna()]
        print("\nPremières epochs:")
        print(val_rows[['epoch', 'mae', 'rmse']].head().to_string(index=False))
        print("\nDernières epochs:")
        print(val_rows[['epoch', 'mae', 'rmse']].tail().to_string(index=False))

def main():
    # Chemin vers le rapport HTML
    html_path = "results/htc_dc_net_evaluation/20260909_110137/rapport_evaluation.html"
    
    print("🔍 Recherche des statistiques pour HTC-DC Net...")
    print(f"   Évaluation: {html_path}")
    
    # 1. Essayer d'extraire depuis le HTML
    if os.path.exists(html_path):
        print("\n📄 Extraction depuis le rapport HTML...")
        html_stats = extract_stats_from_html(html_path)
        if html_stats:
            print("✓ Statistiques trouvées dans le HTML:")
            for key, value in html_stats.items():
                print(f"  {key}: {value}")
    
    # 2. Chercher le training_log correspondant
    print("\n🔍 Recherche du training_log correspondant...")
    log_path = find_corresponding_training_log("20260909_110137")
    
    if log_path:
        print(f"✓ Training_log trouvé: {log_path}")
        stats, df = analyze_training_log(log_path)
        if stats:
            display_stats(stats, df)
            
            # Sauvegarder les statistiques dans un fichier CSV
            stats_df = pd.DataFrame([stats])
            stats_df.to_csv('results/htc_dc_net_evaluation/20260909_110137/training_stats.csv', index=False)
            print(f"\n✓ Statistiques sauvegardées dans training_stats.csv")
            
            return stats
    else:
        print("✗ Aucun training_log correspondant trouvé")
        print("   Les statistiques détaillées ne sont pas disponibles")
        
        # Afficher quand même les stats du HTML si disponibles
        if html_stats:
            print("\n📊 Statistiques disponibles depuis le HTML:")
            print(f"  Meilleur MAE: {html_stats.get('best_mae', 'N/A')}")
            print(f"  Meilleur RMSE: {html_stats.get('best_rmse', 'N/A')}")
            print(f"  MAE Final: {html_stats.get('final_mae', 'N/A')}")
            print(f"  RMSE Final: {html_stats.get('final_rmse', 'N/A')}")
        
        return html_stats

if __name__ == "__main__":
    main()