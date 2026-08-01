"""Métriques d'évaluation hauteur : MAE, RMSE, biais, par tranche."""
from __future__ import annotations
import numpy as np
import pandas as pd


def calculer_metriques(y_pred: np.ndarray, y_true: np.ndarray) -> dict:
    err = y_pred - y_true
    mae = np.abs(err).mean()
    rmse = np.sqrt((err ** 2).mean())
    biais = err.mean()
    corr = np.corrcoef(y_pred, y_true)[0, 1] if len(y_pred) > 1 else np.nan
    return {"MAE": round(float(mae), 3), "RMSE": round(float(rmse), 3),
            "Biais": round(float(biais), 3), "Corr": round(float(corr), 3)}


def metriques_par_tranche(y_pred: np.ndarray, y_true: np.ndarray,
                           bins=(0, 10, 20, np.inf)) -> pd.DataFrame:
    df = pd.DataFrame({"pred": y_pred, "true": y_true})
    df["tranche"] = pd.cut(df["true"], bins=bins)
    rows = []
    for tranche, group in df.groupby("tranche", observed=True):
        if len(group) == 0:
            continue
        m = calculer_metriques(group["pred"].values, group["true"].values)
        m["tranche"] = str(tranche)
        m["n"] = len(group)
        rows.append(m)
    return pd.DataFrame(rows)