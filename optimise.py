"""
P2-ETF-COPULA-ENGINE  ·  optimise.py
Grid-search over LOOKBACK_CANDIDATES on the validation set.
For each candidate window length, run a walk-forward backtest and
record cumulative return.  Best lookback is the one with highest
cumulative return on the validation set — locked in for test + live.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from config import LOOKBACK_CANDIDATES, N_MC_SAMPLES
from marginals   import fit_all_marginals, build_uniform_matrix, transform_window
from copula_model import fit_copula, tail_dependence, mc_simulate
from scorer       import score_etfs


def _walk_forward_returns(returns_df: pd.DataFrame,
                           lookback: int,
                           n_mc: int = 1_000) -> float:
    """
    Simplified walk-forward on returns_df:
    At each step t, use the [t-lookback : t] window to fit copula,
    predict best ETF, record actual next-day return.
    Returns cumulative log-return over the period.
    """
    tickers   = returns_df.columns.tolist()
    cum_lr    = 0.0
    prev_pick = None
    tc        = 12 / 10_000.0

    for t in range(lookback, len(returns_df) - 1):
        window = returns_df.iloc[t - lookback: t]

        try:
            fitted_m = fit_all_marginals(window)
            u_df     = build_uniform_matrix(fitted_m, window)
            if len(u_df) < 10:
                continue

            cop      = fit_copula(u_df)
            td       = tail_dependence(u_df)
            sim      = mc_simulate(cop, fitted_m, window, n_samples=n_mc)
            scores   = score_etfs(sim, td, prev_pick)

            pick = scores.iloc[0]["ticker"]
            actual_r = float(returns_df.iloc[t + 1][pick])

            cost = tc if (prev_pick is not None and pick != prev_pick) else 0.0
            cum_lr += actual_r - cost
            prev_pick = pick
        except Exception:
            continue

    return cum_lr


def optimise_lookback(val_returns: pd.DataFrame,
                      verbose: bool = True) -> dict:
    """
    Try each LOOKBACK_CANDIDATES on val_returns and return the best.

    Returns
    -------
    dict with:
        best_lookback   int
        scores          dict  lookback → cumulative return on val
    """
    results = {}

    for lb in LOOKBACK_CANDIDATES:
        if lb >= len(val_returns) - 2:
            results[lb] = -999.0
            continue
        cum_r = _walk_forward_returns(val_returns, lb)
        results[lb] = cum_r
        if verbose:
            print(f"  lookback={lb:3d}d  val_cum_return={cum_r:.4f}")

    best = max(results, key=results.get)
    if verbose:
        print(f"  → Best lookback: {best} days  (val return {results[best]:.4f})")

    return {"best_lookback": best, "scores": results}
