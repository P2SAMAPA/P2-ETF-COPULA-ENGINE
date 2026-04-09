"""
P2-ETF-COPULA-ENGINE  ·  marginals.py
Fit a GARCH(1,1) model to each ETF's log-return series to capture
heteroskedasticity, then apply the Probability Integral Transform (PIT)
to obtain uniform pseudo-observations u ∈ (0,1) for copula fitting.
"""

from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from arch import arch_model
from scipy.stats import norm

from config import GARCH_P, GARCH_Q, GARCH_DIST

warnings.filterwarnings("ignore")


# ── Single-ETF GARCH fit ──────────────────────────────────────────────────────

def fit_garch(returns: pd.Series) -> dict:
    """
    Fit GARCH(p,q) to a return series.

    Returns
    -------
    dict with:
        model_result  – arch ModelResult
        std_resid     – pd.Series of standardised residuals
        u             – pd.Series of uniform PIT pseudo-observations
    """
    # arch_model expects returns in percentage points for numerical stability
    r_scaled = returns * 100.0

    am     = arch_model(r_scaled, p=GARCH_P, q=GARCH_Q, dist=GARCH_DIST,
                        rescale=False)
    result = am.fit(disp="off", show_warning=False)

    std_resid = result.std_resid.dropna()
    std_resid = pd.Series(std_resid, index=returns.index[-len(std_resid):])

    # PIT: map standardised residuals to U(0,1) via standard normal CDF
    u = pd.Series(norm.cdf(std_resid.values), index=std_resid.index,
                  name=returns.name)

    # Clamp strictly inside (0,1) to avoid copula boundary issues
    u = u.clip(1e-6, 1 - 1e-6)

    return {"model_result": result, "std_resid": std_resid, "u": u}


# ── Fit all ETFs in a module ──────────────────────────────────────────────────

def fit_all_marginals(returns_df: pd.DataFrame) -> dict[str, dict]:
    """
    Fit GARCH marginals for every ETF in returns_df.

    Parameters
    ----------
    returns_df : pd.DataFrame  columns = ETF tickers

    Returns
    -------
    dict  ticker -> {model_result, std_resid, u}
    """
    fitted = {}
    for ticker in returns_df.columns:
        fitted[ticker] = fit_garch(returns_df[ticker].dropna())
    return fitted


# ── Build uniform pseudo-observation matrix ───────────────────────────────────

def build_uniform_matrix(fitted: dict[str, dict],
                         returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Stack PIT pseudo-observations into a (T × N) matrix aligned to
    the common date index.
    """
    u_dict  = {t: fitted[t]["u"] for t in fitted}
    u_df    = pd.DataFrame(u_dict)
    common  = u_df.dropna().index
    return u_df.loc[common]


# ── Transform a new window using a fitted GARCH model ────────────────────────

def transform_window(fitted_marginals: dict[str, dict],
                     window_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Given already-fitted GARCH models, transform a new rolling window of
    returns into uniform pseudo-observations using forecast volatility.

    Falls back to empirical CDF ranking if forecast fails.
    """
    u_cols = {}
    for ticker, info in fitted_marginals.items():
        if ticker not in window_returns.columns:
            continue
        r  = window_returns[ticker].dropna()
        try:
            res      = info["model_result"]
            forecast = res.forecast(horizon=1, reindex=False)
            sigma    = float(np.sqrt(forecast.variance.iloc[-1, 0]))
            if sigma <= 0 or np.isnan(sigma):
                raise ValueError("invalid sigma")
            std_r = (r.values * 100.0) / sigma
            u     = norm.cdf(std_r)
        except Exception:
            # Fallback: empirical rank-based CDF
            n   = len(r)
            u   = (np.argsort(np.argsort(r.values)) + 1) / (n + 1)
        u_cols[ticker] = pd.Series(u.clip(1e-6, 1 - 1e-6),
                                   index=r.index, name=ticker)
    return pd.DataFrame(u_cols).dropna()
