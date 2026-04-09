"""
P2-ETF-COPULA-ENGINE  ·  backtest.py
Walk-forward OOS backtest on the held-out test set using the
optimised lookback window.  Records daily picks, returns, and
computes summary metrics vs the benchmark.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from config import TRANSACTION_COST_BPS
from marginals    import fit_all_marginals, build_uniform_matrix
from copula_model import fit_copula, tail_dependence, mc_simulate
from scorer       import score_etfs


def run_backtest(returns_df: pd.DataFrame,
                 benchmark_r: pd.Series,
                 lookback: int,
                 n_mc: int = 2_000,
                 verbose: bool = True) -> dict:
    """
    Walk-forward backtest over returns_df using the given lookback.
    At each day t, fit on [t-lookback:t], predict pick, record realised return.

    Parameters
    ----------
    returns_df  : pd.DataFrame  daily log returns (test set)
    benchmark_r : pd.Series     benchmark log returns (same index)
    lookback    : int           optimised window length
    n_mc        : int           MC samples per step (lower for speed)
    verbose     : bool

    Returns
    -------
    dict with:
        equity_curve    pd.Series  cumulative log return of strategy
        bm_curve        pd.Series  cumulative log return of benchmark
        signal_log      pd.DataFrame daily picks and returns
        metrics         dict  summary statistics
    """
    tc        = TRANSACTION_COST_BPS / 10_000.0
    prev_pick = None
    rows      = []

    dates   = returns_df.index
    n       = len(dates)

    for t in range(lookback, n - 1):
        window = returns_df.iloc[t - lookback: t]
        date   = dates[t + 1]

        try:
            fitted_m = fit_all_marginals(window)
            u_df     = build_uniform_matrix(fitted_m, window)
            if len(u_df) < 8:
                continue

            cop    = fit_copula(u_df)
            td     = tail_dependence(u_df)
            sim    = mc_simulate(cop, fitted_m, window, n_samples=n_mc)
            scores = score_etfs(sim, td, prev_pick)

            pick       = scores.iloc[0]["ticker"]
            conviction = float(scores.iloc[0]["conviction_pct"])
            exp_ret    = float(scores.iloc[0]["expected_return"])
            family     = cop["best_family"]

            actual_r   = float(returns_df.iloc[t + 1][pick])
            switch     = (prev_pick is not None and pick != prev_pick)
            net_r      = actual_r - (tc if switch else 0.0)
            hit        = actual_r > 0

            rows.append({
                "date":           date,
                "pick":           pick,
                "conviction_pct": conviction,
                "expected_return":exp_ret,
                "actual_return":  actual_r,
                "net_return":     net_r,
                "switched":       switch,
                "copula_family":  family,
                "hit":            hit,
            })
            prev_pick = pick

        except Exception as e:
            if verbose:
                print(f"  [backtest] skip {date}: {e}")
            continue

    if not rows:
        return {"equity_curve": pd.Series(dtype=float),
                "bm_curve":     pd.Series(dtype=float),
                "signal_log":   pd.DataFrame(),
                "metrics":      {}}

    log      = pd.DataFrame(rows).set_index("date")
    eq_curve = log["net_return"].cumsum()

    # Align benchmark
    bm_common = benchmark_r.reindex(log.index).fillna(0.0)
    bm_curve  = bm_common.cumsum()

    metrics = _compute_metrics(log, eq_curve, bm_curve, benchmark_r)

    if verbose:
        print(f"\n  OOS backtest ({len(log)} days):")
        print(f"    Ann. return : {metrics['ann_return_pct']:.2f}%")
        print(f"    Sharpe      : {metrics['sharpe']:.3f}")
        print(f"    Max DD      : {metrics['max_drawdown_pct']:.2f}%")
        print(f"    Hit rate    : {metrics['hit_rate_pct']:.1f}%")
        print(f"    vs Benchmark: {metrics['ann_alpha_pct']:.2f}% alpha")

    return {
        "equity_curve": eq_curve,
        "bm_curve":     bm_curve,
        "signal_log":   log,
        "metrics":      metrics,
    }


def _compute_metrics(log: pd.DataFrame,
                     eq: pd.Series,
                     bm: pd.Series,
                     full_bm: pd.Series) -> dict:
    n_days   = len(log)
    ann_r    = float(log["net_return"].mean() * 252)
    ann_vol  = float(log["net_return"].std() * np.sqrt(252))
    sharpe   = ann_r / ann_vol if ann_vol > 1e-9 else 0.0

    # Max drawdown
    cum      = np.exp(eq.values)
    peak     = np.maximum.accumulate(cum)
    dd       = (cum - peak) / peak
    max_dd   = float(dd.min())

    # Hit rate
    hit_rate = float(log["hit"].mean())

    # Alpha vs benchmark
    bm_aligned = full_bm.reindex(log.index).fillna(0.0)
    ann_bm     = float(bm_aligned.mean() * 252)
    alpha      = ann_r - ann_bm

    # Positive years
    log_copy       = log.copy()
    log_copy["yr"] = log_copy.index.year
    yr_ret         = log_copy.groupby("yr")["net_return"].sum()
    pos_years      = int((yr_ret > 0).sum())

    return {
        "n_days":           n_days,
        "ann_return_pct":   round(ann_r * 100, 2),
        "ann_vol_pct":      round(ann_vol * 100, 2),
        "sharpe":           round(sharpe, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "hit_rate_pct":     round(hit_rate * 100, 1),
        "ann_alpha_pct":    round(alpha * 100, 2),
        "positive_years":   pos_years,
    }
