"""
P2-ETF-COPULA-ENGINE  ·  regime.py
KMeans clustering on macro features to assign a regime label to each day.
The current regime conditions which set of copula parameters is used.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from config import N_REGIMES, REGIME_FEATURES, RANDOM_SEED


def fit_regime_model(macro_df: pd.DataFrame) -> dict:
    """
    Fit KMeans on the macro regime features.

    Returns
    -------
    dict with:
        kmeans    – fitted KMeans object
        scaler    – fitted StandardScaler
        labels    – pd.Series of regime labels (same index as macro_df)
        centers   – pd.DataFrame of cluster centres (in original scale)
        regime_names – dict  label → human-readable name
    """
    feat   = macro_df[REGIME_FEATURES].ffill().dropna()
    scaler = StandardScaler()
    X      = scaler.fit_transform(feat.values)

    km     = KMeans(n_clusters=N_REGIMES, random_state=RANDOM_SEED, n_init=10)
    km.fit(X)

    labels = pd.Series(km.labels_, index=feat.index, name="regime")

    # Label regimes by average VIX level (ascending = risk-on ... risk-off)
    centers_raw = scaler.inverse_transform(km.cluster_centers_)
    centers_df  = pd.DataFrame(centers_raw, columns=REGIME_FEATURES)
    vix_order   = centers_df["VIX"].argsort().values  # low → high VIX

    # Remap labels so 0 = low-vol (risk-on), N-1 = high-vol (risk-off)
    remap   = {old: new for new, old in enumerate(vix_order)}
    labels  = labels.map(remap)

    regime_names = {
        0: "Risk-On (Low Vol)",
        1: "Transitional",
        2: "Risk-Off (High Vol / Stress)",
    }
    if N_REGIMES == 2:
        regime_names = {0: "Risk-On", 1: "Risk-Off"}

    return {
        "kmeans":       km,
        "scaler":       scaler,
        "labels":       labels,
        "centers":      centers_df,
        "regime_names": regime_names,
    }


def predict_regime(regime_model: dict, macro_row: pd.Series) -> int:
    """
    Predict the current regime from a single macro observation.

    Parameters
    ----------
    regime_model : output of fit_regime_model
    macro_row    : pd.Series with at least the REGIME_FEATURES columns

    Returns
    -------
    int  regime label (0 = risk-on, N_REGIMES-1 = risk-off)
    """
    feat  = macro_row[REGIME_FEATURES].values.reshape(1, -1)
    X     = regime_model["scaler"].transform(feat)
    raw_label = int(regime_model["kmeans"].predict(X)[0])

    # Apply the same VIX-based remap used during training
    km     = regime_model["kmeans"]
    scaler = regime_model["scaler"]
    centers_raw = scaler.inverse_transform(km.cluster_centers_)
    centers_df  = pd.DataFrame(centers_raw, columns=REGIME_FEATURES)
    vix_order   = centers_df["VIX"].argsort().values
    remap       = {old: new for new, old in enumerate(vix_order)}
    return remap.get(raw_label, raw_label)
