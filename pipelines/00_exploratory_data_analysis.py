"""
pipelines/00_exploratory_data_analysis.py

Analyse exploratoire des données d'entraînement avec diagrammes complets :
- Statistiques descriptives des hauteurs
- Distribution des hauteurs par zone
- Diagrammes et visualisations complètes
- Analyse de la qualité des données
"""
import sys, os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import pandas as pd

sys.path.append(".")
from core.data_utils import charger_split, charger_patch

# Configuration des styles
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def analyser_hauteurs_zone(zone_name="train"):
    """Analyse complète des hauteurs pour un split donné"""
    print(f"\n{'='*60}")
    print(f"ANALYSE ZONE: {zone_name.upper()}")
    print(f"{'='*60}")
    
    patch_ids = charger_split(zone_name)
    print(f"Nombre de patches: {len(patch_ids)}")
    
    # Collecter les données
    hauteurs_par_patch = []
    pixels_valides_par_patch = []
    hauteurs_tous_pixels = []
    zone_distribution = {}
    
    for patch_id in tqdm(patch_ids, desc=f"Chargement {zone_name}"):
        img, gt, mask = charger_patch(patch_id)
        
        # Extraire la zone géographique
        zone = patch_id.rsplit("_", 1)[0]  # "zone1_00042" -> "zone1"
        zone_distribution[zone] = zone_distribution.get(zone, 0) + 1
        
        # Statistiques par patch
        valid_pixels = gt[mask == 1]
        if len(valid_pixels) > 0:
            median_hauteur = np.median(valid_pixels)
            hauteurs_par_patch.append(median_hauteur)
            pixels_valides_par_patch.append(len(valid_pixels))
            hauteurs_tous_pixels.extend(valid_pixels)
    
    # Convertir en arrays numpy
    hauteurs_par_patch = np.array(hauteurs_par_patch)
    pixels_valides_par_patch = np.array(pixels_valides_par_patch)
    hauteurs_tous_pixels = np.array(hauteurs_tous_pixels)
    
    # Statistiques descriptives
    print(f"\n📊 STATISTIQUES DESCRIPTIVES (médianes par patch)")
    print(f"{'='*60}")
    print(f"Moyenne: {hauteurs_par_patch.mean():.2f} m")
    print(f"Médiane: {np.median(hauteurs_par_patch):.2f} m")
    print(f"Écart-type: {hauteurs_par_patch.std():.2f} m")
    print(f"Min: {hauteurs_par_patch.min():.2f} m")
    print(f"Max: {hauteurs_par_patch.max():.2f} m")
    print(f"25ème percentile: {np.percentile(hauteurs_par_patch, 25):.2f} m")
    print(f"75ème percentile: {np.percentile(hauteurs_par_patch, 75):.2f} m")
    
    print(f"\n📊 STATISTIQUES (tous pixels valides)")
    print(f"{'='*60}")
    print(f"Nombre total de pixels valides: {len(hauteurs_tous_pixels):,}")
    print(f"Moyenne: {hauteurs_tous_pixels.mean():.2f} m")
    print(f"Médiane: {np.median(hauteurs_tous_pixels):.2f} m")
    print(f"Écart-type: {hauteurs_tous_pixels.std():.2f} m")
    print(f"Min: {hauteurs_tous_pixels.min():.2f} m")
    print(f"Max: {hauteurs_tous_pixels.max():.2f} m")
    
    print(f"\n📊 DISTRIBUTION PAR ZONE GÉOGRAPHIQUE")
    print(f"{'='*60}")
    for zone, count in sorted(zone_distribution.items()):
        print(f"{zone}: {count} patches ({count/len(patch_ids)*100:.1f}%)")
    
    print(f"\n📊 COUVERTURE DES DONNÉES")
    print(f"{'='*60}")
    print(f"Moyenne pixels valides par patch: {pixels_valides_par_patch.mean():.0f}")
    print(f"Min pixels valides: {pixels_valides_par_patch.min()}")
    print(f"Max pixels valides: {pixels_valides_par_patch.max()}")
    
    return {
        'zone_name': zone_name,
        'hauteurs_par_patch': hauteurs_par_patch,
        'hauteurs_tous_pixels': hauteurs_tous_pixels,
        'pixels_valides_par_patch': pixels_valides_par_patch,
        'patch_ids': patch_ids,
        'zone_distribution': zone_distribution
    }

def creer_diagrammes(donnees_train, donnees_val, donnees_test):
    """Créer tous les diagrammes d'analyse"""
    print(f"\n{'='*60}")
    print("CRÉATION DES DIAGRAMMES")
    print(f"{'='*60}")
    
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    fig.suptitle('Analyse Exploratoire des Données de Hauteur', fontsize=18, fontweight='bold')
    
    # 1. Distribution des hauteurs par zone (histogrammes)
    ax = fig.add_subplot(gs[0, 0])
    for data, label, color in [(donnees_train, 'Train', 'blue'), 
                               (donnees_val, 'Val', 'green'), 
                               (donnees_test, 'Test', 'red')]:
        ax.hist(data['hauteurs_par_patch'], bins=30, alpha=0.6, label=label, color=color, density=True)
    ax.set_xlabel('Hauteur (m)', fontsize=11)
    ax.set_ylabel('Densité', fontsize=11)
    ax.set_title('Distribution des Hauteurs (médianes par patch)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Box plot des hauteurs par zone
    ax = fig.add_subplot(gs[0, 1])
    box_data = [donnees_train['hauteurs_par_patch'], 
                donnees_val['hauteurs_par_patch'], 
                donnees_test['hauteurs_par_patch']]
    bp = ax.boxplot(box_data, labels=['Train', 'Val', 'Test'], patch_artist=True)
    colors = ['blue', 'green', 'red']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel('Hauteur (m)', fontsize=11)
    ax.set_title('Box Plot des Hauteurs par Zone', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 3. Distribution cumulative (CDF)
    ax = fig.add_subplot(gs[0, 2])
    for data, label, color in [(donnees_train, 'Train', 'blue'), 
                               (donnees_val, 'Val', 'green'), 
                               (donnees_test, 'Test', 'red')]:
        sorted_heights = np.sort(data['hauteurs_par_patch'])
        cdf = np.arange(1, len(sorted_heights) + 1) / len(sorted_heights)
        ax.plot(sorted_heights, cdf, label=label, color=color, linewidth=2)
    ax.set_xlabel('Hauteur (m)', fontsize=11)
    ax.set_ylabel('Cumulative Probability', fontsize=11)
    ax.set_title('Distribution Cumulative (CDF)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Couverture des données (pixels valides par patch)
    ax = fig.add_subplot(gs[1, 0])
    coverage_data = [donnees_train['pixels_valides_par_patch'], 
                     donnees_val['pixels_valides_par_patch'], 
                     donnees_test['pixels_valides_par_patch']]
    bp = ax.boxplot(coverage_data, labels=['Train', 'Val', 'Test'], patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel('Nombre de pixels valides', fontsize=11)
    ax.set_title('Couverture des Données par Zone', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 5. Distribution des hauteurs par tranches
    ax = fig.add_subplot(gs[1, 1])
    bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 50]
    labels = ['0-5', '5-10', '10-15', '15-20', '20-25', '25-30', '30-35', '35-40', '40+']
    
    x = np.arange(len(labels))
    width = 0.25
    
    for i, (data, label, color) in enumerate([(donnees_train, 'Train', 'blue'), 
                                               (donnees_val, 'Val', 'green'), 
                                               (donnees_test, 'Test', 'red')]):
        hist, _ = np.histogram(data['hauteurs_par_patch'], bins=bins)
        ax.bar(x + i*width, hist, width, alpha=0.6, label=label, color=color)
    
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels, rotation=45)
    ax.set_ylabel('Nombre de patches', fontsize=11)
    ax.set_title('Distribution par Tranches de Hauteur', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Distribution par zone géographique
    ax = fig.add_subplot(gs[1, 2])
    
    all_zones = set()
    for data in [donnees_train, donnees_val, donnees_test]:
        all_zones.update(data['zone_distribution'].keys())
    
    x = np.arange(len(all_zones))
    width = 0.25
    
    for i, (data, label, color) in enumerate([(donnees_train, 'Train', 'blue'), 
                                               (donnees_val, 'Val', 'green'), 
                                               (donnees_test, 'Test', 'red')]):
        counts = [data['zone_distribution'].get(zone, 0) for zone in sorted(all_zones)]
        ax.bar(x + i*width, counts, width, alpha=0.6, label=label, color=color)
    
    ax.set_xticks(x + width)
    ax.set_xticklabels(sorted(all_zones), rotation=45)
    ax.set_ylabel('Nombre de patches', fontsize=11)
    ax.set_title('Distribution par Zone Géographique', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 7. Violin plot
    ax = fig.add_subplot(gs[2, 0])
    parts = ax.violinplot([donnees_train['hauteurs_par_patch'], 
                          donnees_val['hauteurs_par_patch'], 
                          donnees_test['hauteurs_par_patch']], 
                         positions=[1, 2, 3], showmeans=True, showmedians=True)
    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.6)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['Train', 'Val', 'Test'])
    ax.set_ylabel('Hauteur (m)', fontsize=11)
    ax.set_title('Violin Plot des Hauteurs', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 8. Scatter plot (hauteur vs pixels valides)
    ax = fig.add_subplot(gs[2, 1])
    for data, label, color in [(donnees_train, 'Train', 'blue'), 
                               (donnees_val, 'Val', 'green'), 
                               (donnees_test, 'Test', 'red')]:
        ax.scatter(data['hauteurs_par_patch'], data['pixels_valides_par_patch'], 
                  alpha=0.3, label=label, color=color, s=10)
    ax.set_xlabel('Hauteur médiane (m)', fontsize=11)
    ax.set_ylabel('Pixels valides', fontsize=11)
    ax.set_title('Hauteur vs Couverture', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 9. Statistiques comparatives
    ax = fig.add_subplot(gs[2, 2])
    ax.axis('off')
    
    stats_text = "STATISTIQUES COMPARATIVES\n\n"
    stats_data = [
        ('Train', donnees_train['hauteurs_par_patch']),
        ('Val', donnees_val['hauteurs_par_patch']),
        ('Test', donnees_test['hauteurs_par_patch'])
    ]
    
    for name, heights in stats_data:
        stats_text += f"{name.upper()}:\n"
        stats_text += f"  N={len(heights):,}\n"
        stats_text += f"  μ={heights.mean():.2f}m\n"
        stats_text += f"  σ={heights.std():.2f}m\n"
        stats_text += f"  min={heights.min():.2f}m\n"
        stats_text += f"  max={heights.max():.2f}m\n"
        stats_text += f"  median={np.median(heights):.2f}m\n\n"
    
    ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, fontsize=10, 
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Sauvegarder le diagramme
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/exploratory_data_analysis.png", dpi=300, bbox_inches='tight')
    print("✅ Diagramme principal sauvegardé: results/exploratory_data_analysis.png")
    
    # Créer un diagramme séparé pour la distribution par tranches
    fig2, ax = plt.subplots(figsize=(12, 6))
    
    bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 50]
    labels = ['0-5m', '5-10m', '10-15m', '15-20m', '20-25m', '25-30m', '30-35m', '35-40m', '40m+']
    
    x = np.arange(len(labels))
    width = 0.25
    
    for i, (data, label, color) in enumerate([(donnees_train, 'Train', 'blue'), 
                                               (donnees_val, 'Val', 'green'), 
                                               (donnees_test, 'Test', 'red')]):
        hist, _ = np.histogram(data['hauteurs_par_patch'], bins=bins)
        # Normaliser en pourcentage
        hist_pct = hist / len(data['hauteurs_par_patch']) * 100
        bars = ax.bar(x + i*width, hist_pct, width, alpha=0.6, label=label, color=color)
        # Ajouter les valeurs sur les barres
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels, rotation=45)
    ax.set_ylabel('Pourcentage de patches (%)', fontsize=12)
    ax.set_title('Distribution des Hauteurs par Tranches (Normalisée)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("results/height_distribution_by_bins.png", dpi=300, bbox_inches='tight')
    print("✅ Diagramme distribution sauvegardé: results/height_distribution_by_bins.png")
    
    plt.show()

def creer_rapport_complet(donnees_train, donnees_val, donnees_test):
    """Créer un rapport complet en CSV"""
    print(f"\n{'='*60}")
    print("SAUVEGARDE DES STATISTIQUES")
    print(f"{'='*60}")
    
    os.makedirs("results", exist_ok=True)
    
    # 1. Statistiques globales
    stats_global = pd.DataFrame({
        'Train': [
            len(donnees_train['hauteurs_par_patch']),
            donnees_train['hauteurs_par_patch'].mean(),
            donnees_train['hauteurs_par_patch'].std(),
            np.median(donnees_train['hauteurs_par_patch']),
            donnees_train['hauteurs_par_patch'].min(),
            donnees_train['hauteurs_par_patch'].max(),
            np.percentile(donnees_train['hauteurs_par_patch'], 25),
            np.percentile(donnees_train['hauteurs_par_patch'], 75)
        ],
        'Val': [
            len(donnees_val['hauteurs_par_patch']),
            donnees_val['hauteurs_par_patch'].mean(),
            donnees_val['hauteurs_par_patch'].std(),
            np.median(donnees_val['hauteurs_par_patch']),
            donnees_val['hauteurs_par_patch'].min(),
            donnees_val['hauteurs_par_patch'].max(),
            np.percentile(donnees_val['hauteurs_par_patch'], 25),
            np.percentile(donnees_val['hauteurs_par_patch'], 75)
        ],
        'Test': [
            len(donnees_test['hauteurs_par_patch']),
            donnees_test['hauteurs_par_patch'].mean(),
            donnees_test['hauteurs_par_patch'].std(),
            np.median(donnees_test['hauteurs_par_patch']),
            donnees_test['hauteurs_par_patch'].min(),
            donnees_test['hauteurs_par_patch'].max(),
            np.percentile(donnees_test['hauteurs_par_patch'], 25),
            np.percentile(donnees_test['hauteurs_par_patch'], 75)
        ]
    }, index=['N_patches', 'Mean_height', 'Std_height', 'Median_height', 'Min_height', 'Max_height', 'Q25_height', 'Q75_height'])
    
    stats_global.to_csv("results/data_statistics_global.csv")
    print("✅ Statistiques globales: results/data_statistics_global.csv")
    
    # 2. Distribution par tranches
    bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 50]
    labels = ['0-5m', '5-10m', '10-15m', '15-20m', '20-25m', '25-30m', '30-35m', '35-40m', '40m+']
    
    dist_train = dict(zip(labels, np.histogram(donnees_train['hauteurs_par_patch'], bins=bins)[0]))
    dist_val = dict(zip(labels, np.histogram(donnees_val['hauteurs_par_patch'], bins=bins)[0]))
    dist_test = dict(zip(labels, np.histogram(donnees_test['hauteurs_par_patch'], bins=bins)[0]))
    
    dist_df = pd.DataFrame({
        'Train': dist_train.values(),
        'Val': dist_val.values(),
        'Test': dist_test.values()
    }, index=dist_train.keys())
    
    dist_df.to_csv("results/data_distribution_by_bins.csv")
    print("✅ Distribution par tranches: results/data_distribution_by_bins.csv")
    
    # 3. Distribution par zone géographique
    zones_train = donnees_train['zone_distribution']
    zones_val = donnees_val['zone_distribution']
    zones_test = donnees_test['zone_distribution']
    
    all_zones = set(zones_train.keys()) | set(zones_val.keys()) | set(zones_test.keys())
    
    zone_df = pd.DataFrame(index=sorted(all_zones))
    for zone in sorted(all_zones):
        zone_df.loc[zone, 'Train'] = zones_train.get(zone, 0)
        zone_df.loc[zone, 'Val'] = zones_val.get(zone, 0)
        zone_df.loc[zone, 'Test'] = zones_test.get(zone, 0)
    
    zone_df.to_csv("results/data_distribution_by_zone.csv")
    print("✅ Distribution par zone: results/data_distribution_by_zone.csv")
    
    # 4. Données brutes pour visualisation externe
    raw_data = []
    for data, split in [(donnees_train, 'train'), (donnees_val, 'val'), (donnees_test, 'test')]:
        for i, hauteur in enumerate(data['hauteurs_par_patch']):
            raw_data.append({
                'split': split,
                'patch_id': data['patch_ids'][i],
                'median_height': hauteur,
                'valid_pixels': data['pixels_valides_par_patch'][i]
            })
    
    raw_df = pd.DataFrame(raw_data)
    raw_df.to_csv("results/data_raw_heights.csv", index=False)
    print("✅ Données brutes: results/data_raw_heights.csv")

def main():
    print("🔍 ANALYSE EXPLORATOIRE DES DONNÉES AVEC DIAGRAMMES")
    print("="*60)
    
    # Analyser chaque split
    print("\n📂 CHARGEMENT DES DONNÉES...")
    donnees_train = analyser_hauteurs_zone("train")
    donnees_val = analyser_hauteurs_zone("val")
    donnees_test = analyser_hauteurs_zone("test")
    
    # Créer les diagrammes
    creer_diagrammes(donnees_train, donnees_val, donnees_test)
    
    # Créer le rapport complet
    creer_rapport_complet(donnees_train, donnees_val, donnees_test)
    
    print(f"\n{'='*60}")
    print("✅ ANALYSE EXPLORATOIRE TERMINÉE")
    print(f"{'='*60}")
    print("\n📁 Fichiers générés dans results/:")
    print("   - exploratory_data_analysis.png (diagramme principal)")
    print("   - height_distribution_by_bins.png (distribution détaillée)")
    print("   - data_statistics_global.csv (statistiques)")
    print("   - data_distribution_by_bins.csv (distribution par tranches)")
    print("   - data_distribution_by_zone.csv (distribution par zone)")
    print("   - data_raw_heights.csv (données brutes)")

if __name__ == "__main__":
    main()