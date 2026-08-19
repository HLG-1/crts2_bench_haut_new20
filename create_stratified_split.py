#!/usr/bin/env python3
"""
Script pour créer un split train/val/test stratifié avec contrainte spatiale.

Objectifs:
- Éviter toute fuite spatiale entre splits (priorité absolue)
- Maintenir des distributions de hauteur et de zone équilibrées
- Reproductible avec seed fixe
"""

import os
import sys
import random
import pickle
import numpy as np
import pandas as pd
import rasterio
from pathlib import Path
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Configuration
RANDOM_SEED = 42
DATA_DIR = "data/htc_dc_net_format"
RESULTS_DIR = "results"
SPATIAL_BUFFER_METERS = 100  # Distance minimale entre patches de splits différents
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Bins de hauteur (en mètres)
HEIGHT_BINS = [0, 5, 10, 15, 20, 25, 30, 35, 40, float('inf')]
HEIGHT_BIN_LABELS = ['0-5m', '5-10m', '10-15m', '15-20m', '20-25m', '25-30m', '30-35m', '35-40m', '40m+']

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


def load_patch_metadata():
    """Charge les métadonnées de tous les patches disponibles."""
    print("Chargement des métadonnées des patches...")
    
    # Charger les hauteurs depuis le CSV existant
    heights_csv = os.path.join(RESULTS_DIR, "data_raw_heights.csv")
    if os.path.exists(heights_csv):
        df_heights = pd.read_csv(heights_csv)
        # Extraire la zone depuis le patch_id (format: zoneX_YYYYY)
        df_heights['zone'] = df_heights['patch_id'].apply(lambda x: x.split('_')[0])
        print(f"  - {len(df_heights)} patches trouvés dans le CSV")
    else:
        print("  - Fichier de hauteurs non trouvé, extraction depuis les fichiers...")
        df_heights = extract_heights_from_files()
    
    # Extraire les coordonnées géographiques depuis les fichiers GeoTIFF
    metadata = extract_geographic_metadata(df_heights)
    
    return metadata


def extract_heights_from_files():
    """Extrait les hauteurs médianes depuis les fichiers height_gt.npy."""
    data_dir = Path(DATA_DIR)
    patch_data = []
    
    # Chercher dans les différents formats possibles
    for npy_file in data_dir.glob("**/*_height_gt.npy"):
        patch_id = npy_file.stem.replace("_height_gt", "")
        height_map = np.load(npy_file)
        valid_pixels = height_map[height_map > 0]
        
        if len(valid_pixels) > 0:
            median_height = np.median(valid_pixels)
            zone = patch_id.split('_')[0]
            patch_data.append({
                'patch_id': patch_id,
                'median_height': median_height,
                'valid_pixels': len(valid_pixels),
                'zone': zone
            })
    
    df = pd.DataFrame(patch_data)
    return df


def extract_geographic_metadata(df_heights):
    """Extrait les coordonnées géographiques depuis les fichiers GeoTIFF."""
    print("Extraction des coordonnées géographiques...")
    
    metadata = []
    img_dir = os.path.join(DATA_DIR, "image")
    
    for _, row in df_heights.iterrows():
        patch_id = row['patch_id']
        median_height = row['median_height']
        zone = row.get('zone', patch_id.split('_')[0])  # Extraire zone depuis patch_id si non disponible
        
        # Chercher le fichier image correspondant
        tif_path = os.path.join(img_dir, f"{patch_id}_IMG.tif")
        
        if not os.path.exists(tif_path):
            # Essayer avec .png
            tif_path = os.path.join(img_dir, f"{patch_id}_IMG.png")
        
        if os.path.exists(tif_path):
            try:
                with rasterio.open(tif_path) as src:
                    # Obtenir les bounds et le centroïde
                    bounds = src.bounds
                    centroid_x = (bounds.left + bounds.right) / 2
                    centroid_y = (bounds.bottom + bounds.top) / 2
                    
                    metadata.append({
                        'patch_id': patch_id,
                        'zone': zone,
                        'median_height': median_height,
                        'centroid_x': centroid_x,
                        'centroid_y': centroid_y,
                        'bounds': bounds,
                        'crs': str(src.crs) if src.crs else None
                    })
            except Exception as e:
                print(f"  - Erreur lecture {patch_id}: {e}")
                # Fallback: coordonnées basées sur la zone et l'index
                zone_idx = hash(zone) % 1000
                patch_idx = int(patch_id.split('_')[1]) if '_' in patch_id else 0
                metadata.append({
                    'patch_id': patch_id,
                    'zone': zone,
                    'median_height': median_height,
                    'centroid_x': zone_idx + patch_idx * 0.001,
                    'centroid_y': zone_idx + patch_idx * 0.001,
                    'bounds': None,
                    'crs': None
                })
        else:
            print(f"  - Fichier non trouvé: {tif_path}")
            # Coordonnées basées sur la zone et l'index
            zone_idx = hash(zone) % 1000
            patch_idx = int(patch_id.split('_')[1]) if '_' in patch_id else 0
            metadata.append({
                'patch_id': patch_id,
                'zone': zone,
                'median_height': median_height,
                'centroid_x': zone_idx + patch_idx * 0.001,
                'centroid_y': zone_idx + patch_idx * 0.001,
                'bounds': None,
                'crs': None
            })
    
    df = pd.DataFrame(metadata)
    print(f"  - {len(df)} patches avec métadonnées complètes")
    
    # Afficher des statistiques sur les coordonnées
    print(f"  - Étendue X: {df['centroid_x'].min():.2f} à {df['centroid_x'].max():.2f}")
    print(f"  - Étendue Y: {df['centroid_y'].min():.2f} à {df['centroid_y'].max():.2f}")
    
    return df


def assign_height_bins(df):
    """Assigne chaque patch à une tranche de hauteur."""
    df['height_bin'] = pd.cut(df['median_height'], bins=HEIGHT_BINS, labels=HEIGHT_BIN_LABELS, right=False)
    return df


def spatial_clustering(df, eps_meters=SPATIAL_BUFFER_METERS):
    """Effectue un clustering spatial pour grouper les patches proches."""
    print(f"Clustering spatial (DBSCAN, eps={eps_meters}m)...")
    
    # Préparer les coordonnées pour le clustering
    coords = df[['centroid_x', 'centroid_y']].values
    
    # Vérifier si les coordonnées sont utilisables
    coord_std = np.std(coords)
    coord_range = np.ptp(coords)
    
    print(f"  - Écart-type coordonnées: {coord_std:.2f}, Étendue: {coord_range:.2f}")
    
    # Si les coordonnées sont identiques ou très peu variées, elles sont factices
    if coord_std < 1.0 or coord_range < 10.0:
        print("  - Coordonnées géographiques non disponibles ou factices")
        print("  - Utilisation d'une contrainte de zone pour éviter la fuite spatiale")
        
        # Créer des clusters basés sur la zone uniquement
        # Chaque zone devient un cluster spatial indépendant
        df['spatial_cluster'] = df['zone']
        
        # Puis subdiviser chaque zone en sous-clusters pour permettre le split
        for zone in df['zone'].unique():
            zone_mask = df['zone'] == zone
            zone_patches = df[zone_mask]
            n_zone_patches = len(zone_patches)
            
            # Diviser en sous-clusters de taille approximative
            n_subclusters = max(1, n_zone_patches // 20)  # ~20 patches par sous-cluster
            if n_subclusters > 1:
                subcluster_ids = np.repeat(range(n_subclusters), 
                                         np.ceil(n_zone_patches / n_subclusters).astype(int))[:n_zone_patches]
                df.loc[zone_mask, 'spatial_cluster'] = df.loc[zone_mask, 'zone'] + '_' + subcluster_ids.astype(str)
        
        n_clusters = df['spatial_cluster'].nunique()
        print(f"  - {n_clusters} clusters spatiaux créés (basés sur les zones)")
    else:
        # DBSCAN pour grouper les patches proches
        # eps est en unités de coordonnées, on assume des mètres
        clustering = DBSCAN(eps=eps_meters, min_samples=1, metric='euclidean')
        df['spatial_cluster'] = clustering.fit_predict(coords)
        
        n_clusters = df['spatial_cluster'].nunique()
        print(f"  - {n_clusters} clusters spatiaux identifiés (DBSCAN)")
    
    # Afficher la distribution des tailles de clusters
    cluster_sizes = df['spatial_cluster'].value_counts().sort_values(ascending=False)
    print(f"  - Plus grand cluster: {cluster_sizes.iloc[0]} patches")
    print(f"  - Plus petit cluster: {cluster_sizes.iloc[-1]} patches")
    
    return df


def stratified_split_on_clusters(df, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, test_ratio=TEST_RATIO):
    """Effectue un split stratifié sur les clusters (pas sur les patches individuels)."""
    print("Split stratifié sur les clusters...")
    
    # Grouper par cluster spatial
    clusters = df.groupby('spatial_cluster')
    
    # Pour chaque cluster, déterminer sa zone dominante
    cluster_info = []
    for cluster_id, cluster_df in clusters:
        zone_counts = cluster_df['zone'].value_counts()
        dominant_zone = zone_counts.index[0]
        n_patches = len(cluster_df)
        
        cluster_info.append({
            'cluster_id': cluster_id,
            'n_patches': n_patches,
            'dominant_zone': dominant_zone
        })
    
    clusters_df = pd.DataFrame(cluster_info)
    
    # Split par zone pour garantir une bonne distribution
    train_clusters = []
    val_clusters = []
    test_clusters = []
    
    for zone in df['zone'].unique():
        zone_clusters = clusters_df[clusters_df['dominant_zone'] == zone]['cluster_id'].values
        if len(zone_clusters) < 3:
            # Si trop peu de clusters pour une zone, tous les mettre dans train
            train_clusters.extend(zone_clusters)
            continue
        
        # Split stratifié par zone
        zone_train, zone_temp = train_test_split(
            zone_clusters,
            test_size=(val_ratio + test_ratio),
            random_state=RANDOM_SEED
        )
        
        zone_val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
        zone_val, zone_test = train_test_split(
            zone_temp,
            test_size=(1 - zone_val_ratio_adjusted),
            random_state=RANDOM_SEED
        )
        
        train_clusters.extend(zone_train)
        val_clusters.extend(zone_val)
        test_clusters.extend(zone_test)
    
    # Assigner les splits aux patches
    df['split'] = 'unknown'
    df.loc[df['spatial_cluster'].isin(train_clusters), 'split'] = 'train'
    df.loc[df['spatial_cluster'].isin(val_clusters), 'split'] = 'val'
    df.loc[df['spatial_cluster'].isin(test_clusters), 'split'] = 'test'
    
    print(f"  - Train: {len(df[df['split']=='train'])} patches ({len(train_clusters)} clusters)")
    print(f"  - Val: {len(df[df['split']=='val'])} patches ({len(val_clusters)} clusters)")
    print(f"  - Test: {len(df[df['split']=='test'])} patches ({len(test_clusters)} clusters)")
    
    return df


def validate_spatial_separation(df, min_distance=SPATIAL_BUFFER_METERS):
    """Vérifie qu'aucun patch de deux splits différents n'est trop proche."""
    print("Validation de la séparation spatiale...")
    
    # Vérifier si les coordonnées sont réelles ou factices
    coords = df[['centroid_x', 'centroid_y']].values
    coord_std = np.std(coords)
    coord_range = np.ptp(coords)
    
    if coord_std < 1.0 or coord_range < 10.0:
        print("  - Coordonnées géographiques non disponibles, validation basée sur les clusters")
        # Vérifier que les clusters sont bien séparés entre splits
        cluster_splits = df.groupby('spatial_cluster')['split'].nunique()
        mixed_clusters = cluster_splits[cluster_splits > 1]
        
        if len(mixed_clusters) > 0:
            print(f"  - ⚠️ {len(mixed_clusters)} clusters contiennent des patches de plusieurs splits!")
            return False
        else:
            print(f"  - ✓ Tous les clusters sont homogènes par split")
            
            # Vérifier que chaque zone est bien répartie entre les splits
            print("\n  - Distribution des zones par split:")
            zone_split_dist = pd.crosstab(df['zone'], df['split'], normalize='index') * 100
            print(zone_split_dist.round(1))
            
            return True
    else:
        # Validation par distance réelle
        splits = ['train', 'val', 'test']
        violations = []
        
        for i, split1 in enumerate(splits):
            for split2 in splits[i+1:]:
                patches1 = df[df['split'] == split1][['patch_id', 'centroid_x', 'centroid_y']].values
                patches2 = df[df['split'] == split2][['patch_id', 'centroid_x', 'centroid_y']].values
                
                for p1_id, x1, y1 in patches1:
                    for p2_id, x2, y2 in patches2:
                        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                        if distance < min_distance:
                            violations.append({
                                'patch1': p1_id,
                                'split1': split1,
                                'patch2': p2_id,
                                'split2': split2,
                                'distance': distance
                            })
        
        if violations:
            print(f"  - ⚠️ {len(violations)} violations trouvées!")
            for v in violations[:5]:  # Afficher les 5 premières
                print(f"    {v['patch1']} ({v['split1']}) <-> {v['patch2']} ({v['split2']}): {v['distance']:.1f}m")
            return False
        else:
            print(f"  - ✓ Aucune violation: tous les patches de splits différents sont séparés de >{min_distance}m")
            return True


def generate_validation_report(df, output_dir):
    """Génère un rapport complet de validation du split."""
    print("Génération du rapport de validation...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Distribution de hauteurs par split
    print("\n=== Distribution de hauteurs par split ===")
    height_dist = pd.crosstab(df['height_bin'], df['split'], normalize='columns') * 100
    print(height_dist.round(1))
    
    # 2. Distribution par zone par split
    print("\n=== Distribution par zone par split ===")
    zone_dist = pd.crosstab(df['zone'], df['split'], normalize='columns') * 100
    print(zone_dist.round(1))
    
    # 3. Statistiques de hauteur par split
    print("\n=== Statistiques de hauteur par split ===")
    height_stats = df.groupby('split')['median_height'].agg(['mean', 'std', 'min', 'max', 'median'])
    print(height_stats.round(2))
    
    # 4. Nombre de patches par split
    print("\n=== Nombre de patches par split ===")
    split_counts = df['split'].value_counts()
    for split in ['train', 'val', 'test']:
        count = split_counts.get(split, 0)
        pct = count / len(df) * 100
        print(f"  {split}: {count} patches ({pct:.1f}%)")
    
    # 5. Visualisations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Distribution de hauteurs
    for split in ['train', 'val', 'test']:
        split_data = df[df['split'] == split]['median_height']
        axes[0, 0].hist(split_data, alpha=0.7, label=split, bins=20)
    axes[0, 0].set_xlabel('Hauteur médiane (m)')
    axes[0, 0].set_ylabel('Nombre de patches')
    axes[0, 0].set_title('Distribution des hauteurs par split')
    axes[0, 0].legend()
    
    # Distribution par zone
    zone_split_counts = pd.crosstab(df['zone'], df['split'])
    zone_split_counts.plot(kind='bar', ax=axes[0, 1])
    axes[0, 1].set_xlabel('Zone')
    axes[0, 1].set_ylabel('Nombre de patches')
    axes[0, 1].set_title('Distribution des zones par split')
    axes[0, 1].legend()
    
    # Distribution par bins de hauteur
    height_bin_counts = pd.crosstab(df['height_bin'], df['split'], normalize='index') * 100
    height_bin_counts.plot(kind='bar', ax=axes[1, 0])
    axes[1, 0].set_xlabel('Tranche de hauteur')
    axes[1, 0].set_ylabel('Pourcentage')
    axes[1, 0].set_title('Distribution des tranches de hauteur par split')
    axes[1, 0].legend()
    
    # Scatter plot spatial
    colors = {'train': 'blue', 'val': 'green', 'test': 'red'}
    for split in ['train', 'val', 'test']:
        split_data = df[df['split'] == split]
        axes[1, 1].scatter(split_data['centroid_x'], split_data['centroid_y'], 
                          c=colors[split], label=split, alpha=0.6, s=10)
    axes[1, 1].set_xlabel('Coordonnée X')
    axes[1, 1].set_ylabel('Coordonnée Y')
    axes[1, 1].set_title('Distribution spatiale des patches par split')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'new_split_validation.png'), dpi=150)
    print(f"  - Visualisation sauvegardée: new_split_validation.png")
    
    # Sauvegarder les statistiques en CSV
    height_dist.to_csv(os.path.join(output_dir, 'new_split_height_distribution.csv'))
    zone_dist.to_csv(os.path.join(output_dir, 'new_split_zone_distribution.csv'))
    height_stats.to_csv(os.path.join(output_dir, 'new_split_height_stats.csv'))
    
    return height_dist, zone_dist, height_stats


def save_splits(df, output_dir):
    """Sauvegarde les listes de fichiers pour chaque split."""
    print("Sauvegarde des splits...")
    
    splits_dir = os.path.join(output_dir, "splits_new")
    os.makedirs(splits_dir, exist_ok=True)
    
    # Fichiers texte simples
    for split in ['train', 'val', 'test']:
        patch_ids = df[df['split'] == split]['patch_id'].values
        with open(os.path.join(splits_dir, f"{split}.txt"), 'w') as f:
            f.write('\n'.join(patch_ids))
        print(f"  - {split}.txt: {len(patch_ids)} patches")
    
    # CSV complet avec métadonnées
    df.to_csv(os.path.join(output_dir, 'new_split_metadata.csv'), index=False)
    print(f"  - Métadonnées complètes sauvegardées: new_split_metadata.csv")
    
    return splits_dir


def compare_with_current_split(df_new, output_dir):
    """Compare le nouveau split avec le split actuel."""
    print("\n=== Comparaison avec le split actuel ===")
    
    # Charger le split actuel
    current_train = set()
    current_val = set()
    current_test = set()
    
    splits_dir = os.path.join(DATA_DIR, "splits")
    
    if os.path.exists(os.path.join(splits_dir, "train.txt")):
        with open(os.path.join(splits_dir, "train.txt")) as f:
            current_train = set(line.strip() for line in f)
    
    if os.path.exists(os.path.join(splits_dir, "val.txt")):
        with open(os.path.join(splits_dir, "val.txt")) as f:
            current_val = set(line.strip() for line in f)
    
    if os.path.exists(os.path.join(splits_dir, "test.txt")):
        with open(os.path.join(splits_dir, "test.txt")) as f:
            current_test = set(line.strip() for line in f)
    
    # Créer un DataFrame pour le split actuel
    current_df = pd.DataFrame({
        'patch_id': list(current_train) + list(current_val) + list(current_test),
        'split': ['train'] * len(current_train) + ['val'] * len(current_val) + ['test'] * len(current_test)
    })
    
    # Fusionner avec les métadonnées
    current_df = current_df.merge(df_new[['patch_id', 'median_height', 'zone', 'height_bin']], 
                                 on='patch_id', how='left')
    
    # Comparaison des distributions
    print("\n--- Distribution actuelle vs nouvelle ---")
    
    # Distribution de hauteurs
    current_height_dist = pd.crosstab(current_df['height_bin'], current_df['split'], normalize='columns') * 100
    new_height_dist = pd.crosstab(df_new['height_bin'], df_new['split'], normalize='columns') * 100
    
    print("\nDistribution de hauteurs - Actuel:")
    print(current_height_dist.round(1))
    print("\nDistribution de hauteurs - Nouveau:")
    print(new_height_dist.round(1))
    
    # Distribution par zone
    current_zone_dist = pd.crosstab(current_df['zone'], current_df['split'], normalize='columns') * 100
    new_zone_dist = pd.crosstab(df_new['zone'], df_new['split'], normalize='columns') * 100
    
    print("\nDistribution par zone - Actuel:")
    print(current_zone_dist.round(1))
    print("\nDistribution par zone - Nouveau:")
    print(new_zone_dist.round(1))
    
    # Sauvegarder la comparaison
    comparison = {
        'current_height': current_height_dist,
        'new_height': new_height_dist,
        'current_zone': current_zone_dist,
        'new_zone': new_zone_dist
    }
    
    with open(os.path.join(output_dir, 'split_comparison.txt'), 'w') as f:
        f.write("=== Comparaison Split Actuel vs Nouveau ===\n\n")
        f.write("Distribution de hauteurs - Actuel:\n")
        f.write(current_height_dist.round(1).to_string())
        f.write("\n\nDistribution de hauteurs - Nouveau:\n")
        f.write(new_height_dist.round(1).to_string())
        f.write("\n\nDistribution par zone - Actuel:\n")
        f.write(current_zone_dist.round(1).to_string())
        f.write("\n\nDistribution par zone - Nouveau:\n")
        f.write(new_zone_dist.round(1).to_string())
    
    print(f"\n  - Comparaison sauvegardée: split_comparison.txt")
    
    return comparison


def main():
    """Fonction principale."""
    print("=" * 60)
    print("Création d'un split stratifié avec contrainte spatiale")
    print("=" * 60)
    
    # 1. Charger les métadonnées
    df = load_patch_metadata()
    
    # 2. Assigner les bins de hauteur
    df = assign_height_bins(df)
    print(f"\nDistribution par bins de hauteur:")
    print(df['height_bin'].value_counts().sort_index())
    
    # 3. Clustering spatial
    df = spatial_clustering(df)
    
    # 4. Split stratifié sur les clusters
    df = stratified_split_on_clusters(df)
    
    # 5. Validation de la séparation spatiale
    spatial_valid = validate_spatial_separation(df)
    if not spatial_valid:
        print("\n⚠️ ATTENTION: Des violations spatiales ont été détectées!")
        print("   Considérer d'augmenter SPATIAL_BUFFER_METERS")
    
    # 6. Générer le rapport de validation
    output_dir = os.path.join(RESULTS_DIR, "stratified_split")
    height_dist, zone_dist, height_stats = generate_validation_report(df, output_dir)
    
    # 7. Sauvegarder les splits
    splits_dir = save_splits(df, output_dir)
    
    # 8. Comparer avec le split actuel
    comparison = compare_with_current_split(df, output_dir)
    
    print("\n" + "=" * 60)
    print("Terminé!")
    print(f"Splits sauvegardés dans: {splits_dir}")
    print(f"Rapports dans: {output_dir}")
    print("=" * 60)
    
    return df


if __name__ == "__main__":
    df = main()
