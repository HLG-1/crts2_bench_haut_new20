"""
core/metrics.py

Métriques d'évaluation pour le benchmark des modèles d'estimation de hauteur
de bâtiments (DepthPro, Depth Anything V2, HTC-DC Net, TSE-Net).

Convention alignée sur la littérature :
  - HTC-DC Net (arXiv:2309.16486) : RMSE, RMSE-M, RMSE-NM, RMSE-B
  - TSE-Net    (arXiv:2511.13552) : RMSE-B (médiane par bâtiment) + erreur relative

Deux niveaux d'évaluation :
  - niveau PIXEL   : diagnostic technique (rmse_pixel)
  - niveau BATIMENT: métrique de décision finale (calculer_metriques, agreger_par_batiment)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import rasterio.features
from scipy import stats


# ═══════════════════════════════════════════════════════════════
# 1. NIVEAU PIXEL — diagnostic technique
# ═══════════════════════════════════════════════════════════════

def rmse_pixel(pred: np.ndarray, true: np.ndarray, mask: np.ndarray | None = None) -> float:
    """
    RMSE au niveau pixel.

    Sans mask : RMSE globale (tous pixels confondus, convention HTC-DC Net "RMSE").
    Avec mask (valid_mask == 1) : RMSE-M (pixels bâtiment uniquement).
    Avec mask (valid_mask == 0) : RMSE-NM (pixels non-bâtiment uniquement).
    """
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        pred, true = pred[mask], true[mask]
    if pred.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def rmse_triplet(pred_map: np.ndarray, true_map: np.ndarray, valid_mask: np.ndarray) -> dict:
    """Calcule RMSE, RMSE-M, RMSE-NM en un seul appel, sur un patch complet."""
    m = np.asarray(valid_mask, dtype=bool)
    return {
        "RMSE": rmse_pixel(pred_map, true_map),
        "RMSE_M": rmse_pixel(pred_map, true_map, mask=m),
        "RMSE_NM": rmse_pixel(pred_map, true_map, mask=~m),
    }


# ═══════════════════════════════════════════════════════════════
# 2. AGRÉGATION PIXEL → BÂTIMENT (médiane, convention HTC-DC Net / TSE-Net)
# ═══════════════════════════════════════════════════════════════

def agreger_par_batiment(
    pred_map: np.ndarray,
    gdf_zone,
    window_transform,
    shape_hw: tuple[int, int],
    champ_hauteur: str = "HAUTEUR",
    min_pixels: int = 20,
) -> list[dict]:
    """
    Pour chaque bâtiment (polygone) présent dans la fenêtre :
      - rasterise son emprise dans le repère du patch,
      - agrège la carte de hauteur prédite PAR MÉDIANE à l'intérieur du masque,
      - associe la valeur HAUTEUR réelle du polygone.

    Retourne une liste de dicts {batiment_id, HAUTEUR_reelle, HAUTEUR_predite}.
    Les bâtiments dont l'emprise dans le patch est trop petite (bord de tuile,
    quasi hors-cadre) sont ignorés via `min_pixels`.
    """
    resultats = []
    for idx, row in gdf_zone.iterrows():
        mask = rasterio.features.rasterize(
            [(row.geometry, 1)],
            out_shape=shape_hw,
            transform=window_transform,
            fill=0,
            dtype="uint8",
        )
        if mask.sum() < min_pixels:
            continue
        valeur_pred = float(np.median(pred_map[mask == 1]))
        resultats.append({
            "batiment_id": idx,
            "HAUTEUR_reelle": float(row[champ_hauteur]),
            "HAUTEUR_predite": valeur_pred,
        })
    return resultats


# ═══════════════════════════════════════════════════════════════
# 3. MÉTRIQUES PAR BÂTIMENT — décision finale
# ═══════════════════════════════════════════════════════════════

def nmad(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Normalized Median Absolute Deviation.
    Version robuste de l'écart-type des erreurs, peu sensible aux outliers
    (contrairement au RMSE, qui peut être dominé par quelques erreurs énormes).
    """
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    err = y_pred - y_true
    return float(1.4826 * np.median(np.abs(err - np.median(err))))


def calculer_metriques(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    n_total_attendu: int | None = None,
) -> dict:
    """
    Métriques de décision, calculées sur des valeurs DÉJÀ agrégées par bâtiment
    (une valeur prédite + une valeur réelle par bâtiment, pas des pixels bruts).

    n_total_attendu : nombre de bâtiments théoriquement évaluables dans le test
                       (avant filtrage min_pixels / NaN) -> sert à calculer la
                       couverture, pour détecter les échecs silencieux d'un modèle.
    """
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)

    # Filtrage des NaN/Inf éventuels (sorties invalides d'un modèle)
    valid = np.isfinite(y_pred) & np.isfinite(y_true)
    n_avant_filtre = len(y_pred)
    y_pred, y_true = y_pred[valid], y_true[valid]

    if len(y_pred) == 0:
        return {
            "MAE": None, "RMSE_B": None, "NMAD": None, "Biais": None,
            "Erreur_relative": None, "Correlation": None,
            "Couverture": 0.0, "n_batiments": 0,
        }

    err = y_pred - y_true
    mae = float(np.abs(err).mean())
    rmse_b = float(np.sqrt((err ** 2).mean()))
    biais = float(err.mean())

    mask_rel = y_true > 0
    rel = float((np.abs(err[mask_rel]) / y_true[mask_rel]).mean()) if mask_rel.any() else None

    corr = float(np.corrcoef(y_pred, y_true)[0, 1]) if len(y_pred) > 1 else None

    if n_total_attendu:
        couverture = len(y_pred) / n_total_attendu
    elif n_avant_filtre:
        couverture = len(y_pred) / n_avant_filtre
    else:
        couverture = None

    return {
        "MAE": round(mae, 3),
        "RMSE_B": round(rmse_b, 3),
        "NMAD": round(nmad(y_pred, y_true), 3),
        "Biais": round(biais, 3),
        "Erreur_relative": round(rel, 3) if rel is not None else None,
        "Correlation": round(corr, 3) if corr is not None else None,
        "Couverture": round(couverture, 3) if couverture is not None else None,
        "n_batiments": len(y_pred),
    }


def metriques_par_tranche(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    bins: tuple = (0, 10, 20, np.inf),
) -> pd.DataFrame:
    """Mêmes métriques, ventilées par tranche de hauteur réelle (petit/moyen/grand bâti)."""
    df = pd.DataFrame({"pred": np.asarray(y_pred), "true": np.asarray(y_true)})
    df["tranche"] = pd.cut(df["true"], bins=bins)

    rows = []
    for tranche, group in df.groupby("tranche", observed=True):
        if len(group) == 0:
            continue
        m = calculer_metriques(group["pred"].values, group["true"].values)
        m["tranche"] = str(tranche)
        rows.append(m)
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════
# 4. COMPARAISON ENTRE MODÈLES — significativité statistique
# ═══════════════════════════════════════════════════════════════

def comparer_modeles_wilcoxon(
    y_pred_a: np.ndarray,
    y_true_a: np.ndarray,
    y_pred_b: np.ndarray,
    y_true_b: np.ndarray,
) -> dict:
    """
    Test de Wilcoxon apparié sur les erreurs absolues par bâtiment.
    À utiliser uniquement quand les deux modèles sont évalués sur EXACTEMENT
    les mêmes bâtiments (même ordre), ce qui est le cas ici puisque le test
    (zone3) est fixe pour les 4 modèles.

    p_value < 0.05 -> la différence de performance entre A et B est
    statistiquement significative (pas due au hasard de l'échantillon test).
    """
    err_a = np.abs(np.asarray(y_pred_a) - np.asarray(y_true_a))
    err_b = np.abs(np.asarray(y_pred_b) - np.asarray(y_true_b))

    if len(err_a) != len(err_b):
        raise ValueError(
            f"Les deux modèles doivent être évalués sur le même nombre de "
            f"bâtiments pour un test apparié ({len(err_a)} vs {len(err_b)})."
        )

    stat, p_value = stats.wilcoxon(err_a, err_b)
    return {
        "statistique": float(stat),
        "p_value": float(p_value),
        "significatif_5pct": bool(p_value < 0.05),
        "meilleur_modele": "A" if err_a.mean() < err_b.mean() else "B",
    }


def matrice_significativite(resultats_par_modele: dict[str, tuple[np.ndarray, np.ndarray]]) -> pd.DataFrame:
    """
    Matrice p-value de Wilcoxon pour chaque paire de modèles.
    resultats_par_modele = {"DepthPro": (y_pred, y_true), "DAV2": (...), ...}
    Suppose que les bâtiments évalués sont identiques (mêmes indices/ordre)
    pour tous les modèles.
    """
    noms = list(resultats_par_modele.keys())
    mat = pd.DataFrame(index=noms, columns=noms, dtype=float)

    for i, nom_a in enumerate(noms):
        for nom_b in noms[i + 1:]:
            y_pred_a, y_true_a = resultats_par_modele[nom_a]
            y_pred_b, y_true_b = resultats_par_modele[nom_b]
            try:
                res = comparer_modeles_wilcoxon(y_pred_a, y_true_a, y_pred_b, y_true_b)
                mat.loc[nom_a, nom_b] = res["p_value"]
                mat.loc[nom_b, nom_a] = res["p_value"]
            except ValueError:
                mat.loc[nom_a, nom_b] = np.nan
                mat.loc[nom_b, nom_a] = np.nan

    np.fill_diagonal(mat.values, np.nan)
    return mat


# ═══════════════════════════════════════════════════════════════
# 5. LATENCE D'INFÉRENCE — critère opérationnel
# ═══════════════════════════════════════════════════════════════

def mesurer_latence(fonction_inference, patch_exemple, n_repeats: int = 20, n_warmup: int = 3) -> dict:
    """
    Latence moyenne d'inférence sur un patch représentatif.

    fonction_inference : callable(patch) -> carte de hauteur prédite.
    n_warmup : appels initiaux non comptabilisés (chargement CUDA, cache, etc.)
    """
    import time

    for _ in range(n_warmup):
        fonction_inference(patch_exemple)

    temps = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        fonction_inference(patch_exemple)
        temps.append((time.perf_counter() - t0) * 1000)  # ms

    return {
        "latence_moy_ms": round(float(np.mean(temps)), 1),
        "latence_std_ms": round(float(np.std(temps)), 1),
        "latence_min_ms": round(float(np.min(temps)), 1),
        "latence_max_ms": round(float(np.max(temps)), 1),
    }


# ═══════════════════════════════════════════════════════════════
# 6. TABLEAU COMPARATIF FINAL
# ═══════════════════════════════════════════════════════════════

def tableau_comparatif(
    resultats_par_modele: dict[str, tuple[np.ndarray, np.ndarray]],
    latences_par_modele: dict[str, dict] | None = None,
    n_total_attendu: int | None = None,
) -> pd.DataFrame:
    """
    Construit le tableau comparatif final des 4 modèles.

    resultats_par_modele = {"DepthPro": (y_pred, y_true), "DAV2_calibre": (...),
                            "DAV2_finetune": (...), "HTC_DC_Net": (...), "TSE_Net": (...)}
    latences_par_modele  = {"DepthPro": {"latence_moy_ms": ...}, ...}  (optionnel)
    """
    rows = []
    for nom_modele, (y_pred, y_true) in resultats_par_modele.items():
        m = calculer_metriques(y_pred, y_true, n_total_attendu=n_total_attendu)
        m["modele"] = nom_modele
        if latences_par_modele and nom_modele in latences_par_modele:
            m["latence_moy_ms"] = latences_par_modele[nom_modele].get("latence_moy_ms")
        rows.append(m)

    df = pd.DataFrame(rows).set_index("modele")
    colonnes = ["MAE", "RMSE_B", "NMAD", "Biais", "Erreur_relative",
                "Correlation", "Couverture", "n_batiments"]
    if latences_par_modele:
        colonnes.append("latence_moy_ms")
    return df[colonnes]


# ═══════════════════════════════════════════════════════════════
# 7. AUTO-TEST (à lancer directement : python core/metrics.py)
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    np.random.seed(42)

    y_true = np.array([8, 12, 20, 5, 15, 30, 7, 18, 22, 9], dtype=float)
    y_pred_bon = y_true + np.random.normal(0, 0.5, len(y_true))
    y_pred_biaise = y_true + 3 + np.random.normal(0, 0.3, len(y_true))
    y_pred_bruite = y_true + np.random.normal(0, 4, len(y_true))

    print("=== Modèle précis ===")
    print(calculer_metriques(y_pred_bon, y_true))

    print("\n=== Modèle biaisé (+3m systématique) ===")
    print(calculer_metriques(y_pred_biaise, y_true))

    print("\n=== Modèle bruité ===")
    print(calculer_metriques(y_pred_bruite, y_true))

    print("\n=== Tableau comparatif ===")
    comparatif = tableau_comparatif({
        "modele_precis": (y_pred_bon, y_true),
        "modele_biaise": (y_pred_biaise, y_true),
        "modele_bruite": (y_pred_bruite, y_true),
    })
    print(comparatif)

    print("\n=== Test Wilcoxon précis vs bruité ===")
    print(comparer_modeles_wilcoxon(y_pred_bon, y_true, y_pred_bruite, y_true))

    print("\n=== Matrice de significativité ===")
    mat = matrice_significativite({
        "modele_precis": (y_pred_bon, y_true),
        "modele_biaise": (y_pred_biaise, y_true),
        "modele_bruite": (y_pred_bruite, y_true),
    })
    print(mat)

    print("\n=== Métriques par tranche (modèle bruité) ===")
    print(metriques_par_tranche(y_pred_bruite, y_true))