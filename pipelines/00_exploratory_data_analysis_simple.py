"""
pipelines/00_exploratory_data_analysis_simple.py

Analyse exploratoire des données d'entraînement (version sans matplotlib) :
- Statistiques descriptives des hauteurs
- Distribution des hauteurs par zone
- Analyse de la qualité des données
- Export CSV pour visualisation externe
"""
import sys, os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

sys.path.append(".")
from core.data_utils import charger_split, charger_patch

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

def analyser_distribution_par_tranches(donnees, split_name):
    """Analyse de la distribution par tranches de hauteur"""
    print(f"\n📊 DISTRIBUTION PAR TRANCHES ({split_name.upper()})")
    print(f"{'='*60}")
    
    hauteurs = donnees['hauteurs_par_patch']
    bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 50]
    labels = ['0-5m', '5-10m', '10-15m', '15-20m', '20-25m', '25-30m', '30-35m', '35-40m', '40m+']
    
    hist, _ = np.histogram(hauteurs, bins=bins)
    
    for label, count in zip(labels, hist):
        percentage = count / len(hauteurs) * 100
        print(f"{label}: {count:4d} patches ({percentage:5.1f}%)")
    
    return dict(zip(labels, hist))

def creer_rapport_complet(donnees_train, donnees_val, donnees_test):
    """Créer un rapport complet en CSV"""
    print(f"\n{'='*60}")
    print("CRÉATION DU RAPPORT COMPLET")
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
    dist_train = analyser_distribution_par_tranches(donnees_train, "train")
    dist_val = analyser_distribution_par_tranches(donnees_val, "val")
    dist_test = analyser_distribution_par_tranches(donnees_test, "test")
    
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
    
    print(f"\n📋 RÉSUMÉ DES FICHIERS GÉNÉRÉS")
    print(f"{'='*60}")
    print("results/data_statistics_global.csv - Statistiques descriptives")
    print("results/data_distribution_by_bins.csv - Distribution par tranches")
    print("results/data_distribution_by_zone.csv - Distribution par zone géo")
    print("results/data_raw_heights.csv - Données brutes pour visualisation")

def main():
    print("🔍 ANALYSE EXPLORATOIRE DES DONNÉES (VERSION SIMPLIFIÉE)")
    print("="*60)
    
    # Analyser chaque split
    print("\n📂 CHARGEMENT DES DONNÉES...")
    donnees_train = analyser_hauteurs_zone("train")
    donnees_val = analyser_hauteurs_zone("val")
    donnees_test = analyser_hauteurs_zone("test")
    
    # Créer le rapport complet
    creer_rapport_complet(donnees_train, donnees_val, donnees_test)
    
    print(f"\n{'='*60}")
    print("✅ ANALYSE EXPLORATOIRE TERMINÉE")
    print(f"{'='*60}")
    print("\n💡 Vous pouvez utiliser les fichiers CSV générés pour:")
    print("   - Créer des diagrammes avec Excel/Google Sheets")
    print("   - Importer dans des outils de visualisation (Tableau, PowerBI)")
    print("   - Utiliser avec Python/R si vous installez matplotlib compatible")

if __name__ == "__main__":
    main()