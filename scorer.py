"""
P2-ETF-COPULA-ENGINE  ·  scorer.py
Score each ETF from Monte Carlo simulated returns and produce a ranked
pick with conviction, macro pills, and tail-risk diagnostics.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from config import W_MEAN, W_TAIL, W_SHARPE, TRANSACTION_COST_BPS


def score_etfs(sim_returns: pd.DataFrame,
               tail_dep: pd.DataFrame,
               prev_pick: str | None = None) -> pd.DataFrame:
    """
    Score each ETF from the simulated return matrix.

    Parameters
    ----------
    sim_returns : pd.DataFrame  shape (N_MC, n_etfs)  MC simulated returns
    tail_dep    : pd.DataFrame  lower tail dependence matrix (n_etfs × n_etfs)
    prev_pick   : ticker of yesterday's pick (for transaction cost)

    Returns
    -------
    pd.DataFrame  sorted by final_score descending, with columns:
        ticker, expected_return, sharpe_proxy, avg_lower_tail,
        gross_score, net_score, rank, conviction_pct
    """
    rows = []
    tc   = TRANSACTION_COST_BPS / 10_000.0

    for ticker in sim_returns.columns:
        r     = sim_returns[ticker].values
        mu    = float(np.mean(r))
        sigma = float(np.std(r)) + 1e-9

        sharpe_proxy = mu / sigma

        # Average lower tail dependence with all OTHER ETFs
        others = [c for c in tail_dep.columns if c != ticker]
        if others and ticker in tail_dep.index:
            avg_td = float(tail_dep.loc[ticker, others].mean())
        else:
            avg_td = 0.0

        gross = W_MEAN * mu - W_TAIL * avg_td + W_SHARPE * sharpe_proxy

        # Deduct transaction cost if this pick differs from prev
        switch_cost = tc if (prev_pick is not None and ticker != prev_pick) else 0.0
        net         = gross - switch_cost

        rows.append({
            "ticker":          ticker,
            "expected_return": mu,
            "sharpe_proxy":    sharpe_proxy,
            "avg_lower_tail":  avg_td,
            "gross_score":     gross,
            "net_score":       net,
        })

    df = pd.DataFrame(rows).sort_values("net_score", ascending=False).reset_index(drop=True)
    df["rank"]           = df.index + 1
    df["conviction_pct"] = _compute_conviction(df["net_score"].values)
    return df


def _compute_conviction(scores: np.ndarray) -> np.ndarray:
    """
    Convert raw scores to conviction percentages using softmax-like
    normalisation so all convictions sum to 100 and top pick is meaningful.
    """
    shifted = scores - scores.min()
    if shifted.sum() < 1e-12:
        return np.full(len(scores), 100 / len(scores))
    pct = 100.0 * shifted / shifted.sum()
    return np.round(pct, 1)


def build_signal(scores_df: pd.DataFrame,
                 module: str,
                 regime_label: int,
                 regime_name: str,
                 best_family: str,
                 best_lookback: int,
                 next_trading_day: str,
                 macro_latest: pd.Series) -> dict:
    """
    Package the top pick and supporting information into a signal dict
    ready to be serialised to JSON.
    """
    top     = scores_df.iloc[0]
    second  = scores_df.iloc[1] if len(scores_df) > 1 else None
    third   = scores_df.iloc[2] if len(scores_df) > 2 else None

    signal = {
        "module":             module,
        "next_trading_day":   next_trading_day,
        "pick":               top["ticker"],
        "conviction_pct":     float(top["conviction_pct"]),
        "expected_return":    round(float(top["expected_return"]), 6),
        "sharpe_proxy":       round(float(top["sharpe_proxy"]), 4),
        "avg_lower_tail_dep": round(float(top["avg_lower_tail"]), 4),
        "second_pick":        second["ticker"] if second is not None else None,
        "second_conviction":  float(second["conviction_pct"]) if second is not None else None,
        "third_pick":         third["ticker"] if third is not None else None,
        "third_conviction":   float(third["conviction_pct"]) if third is not None else None,
        "copula_family":      best_family,
        "lookback_days":      best_lookback,
        "regime_id":          int(regime_label),
        "regime_name":        regime_name,
        "macro_pills": {
            "VIX":       round(float(macro_latest.get("VIX", 0)), 2),
            "T10Y2Y":    round(float(macro_latest.get("T10Y2Y", 0)), 3),
            "HY_SPREAD": round(float(macro_latest.get("HY_SPREAD", 0)), 2),
            "IG_SPREAD": round(float(macro_latest.get("IG_SPREAD", 0)), 2),
            "DXY":       round(float(macro_latest.get("DXY", 0)), 2),
        },
        "all_scores": scores_df[["ticker", "expected_return", "net_score",
                                  "conviction_pct", "avg_lower_tail"]].to_dict(orient="records"),
    }
    return signal
