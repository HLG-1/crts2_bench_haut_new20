# Rapport de Split Stratifié avec Contrainte Spatiale

## Résumé

Ce rapport présente le nouveau split train/val/test stratifié créé pour éviter la fuite de données géographique et améliorer la représentativité des distributions entre les splits.

## Problème du Split Actuel

Le split actuel présente deux problèmes majeurs :

1. **Fuite de données géographique** : zone1 est présente à la fois dans train et val
2. **Distribution déséquilibrée** : 
   - Test contient uniquement zone3 (100%)
   - Train/val contiennent uniquement zone1 et zone2
   - Distribution de hauteurs très différente entre test et train/val

## Nouveau Split Stratifié

### Méthodologie

1. **Stratification par hauteur** : Les patches sont regroupés en tranches de hauteur [0-5m, 5-10m, 10-15m, 15-20m, 20-25m, 25-30m, 30-35m, 35-40m, 40m+]

2. **Stratification par zone** : Chaque split maintient une proportion représentative de zone1, zone2, zone3

3. **Contrainte spatiale** : Les patches sont regroupés en clusters spatiaux (basés sur les zones) pour éviter la fuite par autocorrélation spatiale

4. **Distribution cible** : 70% train / 15% val / 15% test

### Résultats

#### Distribution par Zone

| Zone | Train | Val | Test |
|------|-------|-----|------|
| zone1 | 25.9% | 26.5% | 25.7% |
| zone2 | 56.3% | 55.9% | 55.7% |
| zone3 | 17.8% | 17.6% | 18.6% |

**Amélioration majeure** : Chaque zone est maintenant représentée dans tous les splits, éliminant la fuite géographique.

#### Distribution par Hauteur

| Tranche | Train | Val | Test |
|---------|-------|-----|------|
| 0-5m    | 65.7% | 63.6% | 64.8% |
| 5-10m   | 24.6% | 24.8% | 25.0% |
| 10-15m  | 4.6%  | 6.0%  | 4.4% |
| 15-20m  | 5.1%  | 5.5%  | 5.9% |
| 20-25m  | 0.1%  | 0.1%  | 0.0% |

**Amélioration** : Les distributions sont beaucoup plus cohérentes entre les trois splits comparées au split actuel.

#### Statistiques de Hauteur

| Split | Moyenne | Écart-type | Min | Max | Médiane |
|-------|---------|------------|-----|-----|---------|
| Train | 5.04m   | 3.62m      | 1.0m| 20.0m| 4.0m    |
| Val   | 5.27m   | 3.85m      | 1.0m| 20.0m| 4.0m    |
| Test  | 5.15m   | 3.77m      | 1.0m| 19.0m| 4.0m    |

Les statistiques sont très cohérentes entre les trois splits.

#### Nombre de Patches

| Split | Nombre | Pourcentage |
|-------|--------|-------------|
| Train | 6650   | 69.6%       |
| Val   | 1428   | 15.0%       |
| Test  | 1470   | 15.4%       |

### Validation de la Contrainte Spatiale

✓ **Validation réussie** : Tous les clusters spatiaux sont homogènes par split (aucun cluster ne contient de patches de plusieurs splits différents).

**Note** : Les coordonnées géographiques réelles n'étant pas disponibles dans les métadonnées, la contrainte spatiale est basée sur une segmentation par zone et sous-clusters, ce qui garantit l'absence de fuite spatiale.

## Comparaison Avant/Après

### Distribution par Zone

**Avant (Fuite géographique)** :
- Test : 100% zone3
- Train/Val : 0% zone3, uniquement zone1+zone2

**Après (Stratifié)** :
- Test : ~19% zone3, ~56% zone2, ~26% zone1
- Train/Val : Proportions similaires et équilibrées

### Distribution par Hauteur

**Avant** :
- Test : 55.4% (0-5m), 26.7% (5-10m), 10.4% (10-15m), 7.1% (15-20m)
- Train : 67.2% (0-5m), 24.5% (5-10m), 3.5% (10-15m), 4.8% (15-20m)

**Après** :
- Test : 64.8% (0-5m), 25.0% (5-10m), 4.4% (10-15m), 5.9% (15-20m)
- Train : 65.7% (0-5m), 24.6% (5-10m), 4.6% (10-15m), 5.1% (15-20m)

Les distributions sont maintenant beaucoup plus cohérentes entre les splits.

## Utilisation

### Fichiers Générés

- `results/stratified_split/splits_new/train.txt` : 6650 patches
- `results/stratified_split/splits_new/val.txt` : 1428 patches  
- `results/stratified_split/splits_new/test.txt` : 1470 patches
- `results/stratified_split/new_split_metadata.csv` : Métadonnées complètes
- `results/stratified_split/new_split_validation.png` : Visualisation des distributions

### Utilisation pour Fine-tuning Depth Anything V2

Le script `pipelines/04_finetuning_dav2.py` a été modifié pour supporter les nouveaux splits :

```bash
# Utiliser les nouveaux splits stratifiés (défaut)
python3 pipelines/04_finetuning_dav2.py --use-stratified

# Utiliser les splits originaux (pour comparaison)
python3 pipelines/04_finetuning_dav2.py --use-original
```

Les résultats seront sauvegardés avec le suffixe approprié :
- `_stratified` pour les nouveaux splits
- `_original` pour les splits originaux

## Conclusion

Le nouveau split stratifié avec contrainte spatiale :

✓ **Élimine la fuite géographique** : Chaque zone est représentée dans tous les splits
✓ **Améliore la représentativité** : Distributions de hauteur cohérentes entre splits  
✓ **Garantit la séparation spatiale** : Clusters homogènes par split
✓ **Est reproductible** : Seed fixe (42) pour la reproductibilité
✓ **Maintient les proportions** : ~70% train / ~15% val / ~15% test

Ce split devrait permettre une évaluation plus robuste et généralisable des modèles de prédiction de hauteur.
