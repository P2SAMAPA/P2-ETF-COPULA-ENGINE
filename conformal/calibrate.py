# conformal/calibrate.py — Conformal calibration for P2-ETF-COPULA-ENGINE
#
# WHY COPULA CONFORMAL IS DIFFERENT FROM NCDE CONFORMAL
# ═══════════════════════════════════════════════════════
# NCDE outputs a single (μ, σ) per ETF. The σ is trained via Gaussian NLL
# and ends up ≫ actual daily returns, making q̂ ≈ 0 and intervals trivial.
#
# COPULA already produces 5,000 Monte Carlo return samples per ETF per day
# from mc_simulate(). These samples are already in return space. The empirical
# quantiles of these 5k samples ARE the natural prediction intervals.
#
# The conformal question here is:
#   "At what empirical quantile level α_adj must we set our MC interval
#    to guarantee ≥ 1−α marginal coverage?"
#
# CALIBRATION APPROACH
# ════════════════════
# For each day in the val set (10% holdout, never used in training):
#   1. Re-run the full copula pipeline on the window ending that day
#      (fit GARCH on window, build uniforms, fit copula, run MC 5k)
#   2. For each ETF, find the empirical quantile of the 5k MC samples
#      that the actual return corresponds to → this is the "rank"
#   3. The nonconformity score  s_i  = max(lo_rank_i, 1 - hi_rank_i)
#      where lo_rank_i = F̂(y_i) from the MC distribution
#      Intuitively: how far into the tail was the actual return?
#      s_i close to 0 = actual return was near the centre of MC mass
#      s_i close to 0.5 = actual return was at the extreme edge
#
# At coverage level 1−α, the conformal quantile q̂ is the
#   ⌈(n+1)(1−α)⌉/n -th quantile of {s_1, …, s_n}
#
# PREDICTION INTERVAL
# ═══════════════════
# Given today's MC samples for ETF i, the conformal interval at level α is:
#   [ MC_quantile(q̂_adj_lo),  MC_quantile(q̂_adj_hi) ]
#   where q̂_adj_lo = q̂  and  q̂_adj_hi = 1 − q̂
#
# Because q̂ is derived from empirical MC quantiles, the interval width
# naturally varies per ETF and regime — wider for volatile ETFs, narrower
# for stable ones. This is qualitatively different from NCDE's constant-width
# absolute-mode intervals.
#
# Usage:
#   python -m conformal.calibrate --module both
#
# Output:
#   models/conformal_fi.json
#   models/conformal_eq.json
#   Uploaded to HF: conformal/conformal_fi.json  and  conformal/conformal_eq.json

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    HF_DATASET_OUT,
    FI_ETFS, EQUITY_ETFS,
    TRAIN_RATIO, VAL_RATIO,
    N_MC_SAMPLES,
)
from loader import get_module_data
from marginals import fit_all_marginals, build_uniform_matrix
from copula_model import fit_copula, tail_dependence, mc_simulate

ALPHA_LEVELS   = [0.90, 0.80, 0.70]
MODELS_DIR     = "models"
MAX_CAL_DAYS   = 200   # cap val-set re-simulation to keep runtime < 5 min


# ── Core calibration ──────────────────────────────────────────────────────────

def _empirical_rank(mc_samples: np.ndarray, actual: float) -> float:
    """
    Return the empirical CDF rank of `actual` in `mc_samples`.
    rank = fraction of MC samples ≤ actual  ∈ [0, 1]
    """
    return float(np.mean(mc_samples <= actual))


def _nonconformity_score(mc_samples: np.ndarray, actual: float) -> float:
    """
    Nonconformity score for a single (MC distribution, actual return) pair.

    s = max(rank, 1 - rank)  where rank = F̂(actual | MC)

    Interpretation:
      s ≈ 0.5  →  actual return near the median of MC (well-covered)
      s ≈ 1.0  →  actual return in the far tail of MC (badly covered)

    We use max(rank, 1−rank) so that both tails count equally.
    """
    rank = _empirical_rank(mc_samples, actual)
    return float(max(rank, 1.0 - rank))


def collect_calibration_scores(module: str, best_lookback: int) -> dict:
    """
    Run the copula pipeline on every val-set day and collect nonconformity
    scores per ETF.

    Parameters
    ----------
    module        : "FI" or "EQ"
    best_lookback : the optimised lookback window (from optimise.py on val set)

    Returns
    -------
    dict with scores array (n_cal, n_etfs), tickers, metadata
    """
    print(f"\n[calibrate] Collecting calibration scores — {module} "
          f"(lookback={best_lookback}d)...")

    data    = get_module_data(module)
    rets    = data["returns"]
    tickers = list(rets.columns)

    n_total = len(rets)
    n_train = int(n_total * TRAIN_RATIO)
    n_val   = int(n_total * VAL_RATIO)

    val_returns = rets.iloc[n_train : n_train + n_val]
    n_cal       = min(len(val_returns), MAX_CAL_DAYS)

    # Use the tail of the val set if it's very large (speed)
    if len(val_returns) > MAX_CAL_DAYS:
        val_returns = val_returns.iloc[-MAX_CAL_DAYS:]
        print(f"[calibrate]   Val set capped at {MAX_CAL_DAYS} days for speed.")

    print(f"[calibrate]   Calibration days: {n_cal}  "
          f"({val_returns.index[0].date()} → {val_returns.index[-1].date()})")

    if n_cal < 30:
        raise ValueError(
            f"Only {n_cal} calibration days — too few. "
            "Check your data range / TRAIN_RATIO / VAL_RATIO."
        )

    scores_list = []   # each entry = list of scores per ETF for one day

    for day_idx in range(n_cal):
        # Window ends at val_returns.index[day_idx] - 1 (we don't know today's return yet)
        # The window of historical data used to fit the copula
        global_idx = n_train + (len(data["returns"].iloc[n_train : n_train + n_val]) - n_cal) + day_idx
        window_end = global_idx          # exclusive — index of today
        window_start = max(0, window_end - best_lookback)

        window_rets = rets.iloc[window_start:window_end]

        if len(window_rets) < 10:
            # Not enough history — skip
            continue

        # Actual returns for this day (what we're trying to cover)
        actual_rets = val_returns.iloc[day_idx]

        try:
            fitted_m = fit_all_marginals(window_rets)
            u_df     = build_uniform_matrix(fitted_m, window_rets)
            cop_fit  = fit_copula(u_df)
            sim      = mc_simulate(cop_fit, fitted_m, window_rets,
                                   n_samples=N_MC_SAMPLES)

            day_scores = []
            for ticker in tickers:
                if ticker not in sim.columns or ticker not in actual_rets.index:
                    day_scores.append(np.nan)
                    continue
                mc_arr = sim[ticker].values
                actual = float(actual_rets[ticker])
                s      = _nonconformity_score(mc_arr, actual)
                day_scores.append(s)

            scores_list.append(day_scores)

        except Exception as e:
            print(f"[calibrate]   Day {day_idx} failed ({e}) — skipping.")
            continue

        if (day_idx + 1) % 20 == 0:
            print(f"[calibrate]   Progress: {day_idx+1}/{n_cal} days done...")

    if not scores_list:
        raise RuntimeError("No valid calibration days produced scores.")

    scores_arr = np.array(scores_list)   # (n_cal_valid, n_etfs)
    # Drop NaN columns (ETFs with missing data)
    valid_cols = ~np.isnan(scores_arr).all(axis=0)
    scores_arr = scores_arr[:, valid_cols]
    tickers    = [t for t, v in zip(tickers, valid_cols) if v]

    print(f"[calibrate]   Valid calibration days: {scores_arr.shape[0]}")
    print(f"[calibrate]   Score stats: "
          f"mean={scores_arr.mean():.4f}  "
          f"p50={np.nanmedian(scores_arr):.4f}  "
          f"p90={np.nanpercentile(scores_arr, 90):.4f}")
    print(f"[calibrate]   (Scores near 0.5 = MC well-centred; near 1.0 = MC missed the tail)")

    return {
        "module":     module,
        "tickers":    tickers,
        "scores":     scores_arr.tolist(),
        "n_cal":      scores_arr.shape[0],
        "lookback":   best_lookback,
        "val_start":  str(val_returns.index[0].date()),
        "val_end":    str(val_returns.index[-1].date()),
        "score_mean": round(float(np.nanmean(scores_arr)), 5),
        "score_p50":  round(float(np.nanmedian(scores_arr)), 5),
        "score_p90":  round(float(np.nanpercentile(scores_arr, 90)), 5),
    }


# ── Quantile thresholds ───────────────────────────────────────────────────────

def compute_quantiles(scores_dict: dict) -> dict:
    """
    Compute conformal quantile q̂ per alpha level per ETF.

    q̂ is in [0.5, 1.0] — it represents the ADJUSTED quantile level
    to use when extracting the prediction interval from the MC distribution:
      interval_lo = MC_quantile(1 - q̂)   (lower tail)
      interval_hi = MC_quantile(q̂)        (upper tail)

    A q̂ of 0.90 means "take the 10th–90th percentile of MC samples".
    A q̂ of 0.95 means "take the 5th–95th percentile" (wider).

    Unlike NCDE, q̂ here is naturally bounded and interpretable:
      q̂ close to 0.5 = model is well-calibrated, narrow intervals suffice
      q̂ close to 1.0 = model is mis-calibrated, must use nearly all MC mass
    """
    scores  = np.array(scores_dict["scores"])   # (n_cal, n_etfs)
    tickers = scores_dict["tickers"]
    n_cal   = scores.shape[0]

    quantiles = {}
    for alpha in ALPHA_LEVELS:
        # Finite-sample correction
        level = np.ceil((n_cal + 1) * (1 - alpha)) / n_cal
        level = min(level, 1.0)

        # q̂ is the level-th empirical quantile of nonconformity scores
        # Since s ∈ [0.5, 1.0], q̂ ∈ [0.5, 1.0]
        per_etf = {}
        for j, ticker in enumerate(tickers):
            col_scores = scores[:, j]
            col_scores = col_scores[~np.isnan(col_scores)]
            if len(col_scores) == 0:
                per_etf[ticker] = 0.9
                continue
            q = float(np.quantile(col_scores, level))
            # q̂ is clipped to [0.5, 0.999] — must be ≥ 0.5 (symmetric interval)
            q = float(np.clip(q, 0.5, 0.999))
            per_etf[ticker] = round(q, 5)

        pooled_scores = scores[~np.isnan(scores)].ravel()
        pooled_q = float(np.clip(np.quantile(pooled_scores, level), 0.5, 0.999))

        quantiles[str(alpha)] = {
            "per_etf":    per_etf,
            "pooled":     round(pooled_q, 5),
            "level_used": round(float(level), 5),
            "meaning":    (
                f"Use MC quantiles [{round((1-pooled_q)*100,1)}%, "
                f"{round(pooled_q*100,1)}%] for {int(alpha*100)}% coverage"
            ),
        }

    return quantiles


# ── Empirical coverage diagnostics ───────────────────────────────────────────

def empirical_coverage(scores_dict: dict, quantiles: dict) -> dict:
    """
    Verify that the calibration-set empirical coverage ≥ target.
    Coverage here = fraction of cal days where s_i ≤ q̂  (i.e. actual
    return fell within the interval defined by q̂).
    """
    scores  = np.array(scores_dict["scores"])
    tickers = scores_dict["tickers"]

    coverage = {}
    for alpha_str, q_info in quantiles.items():
        target = 1 - float(alpha_str)
        per_etf_cov = {}
        for j, ticker in enumerate(tickers):
            q       = q_info["per_etf"][ticker]
            col     = scores[:, j]
            col     = col[~np.isnan(col)]
            covered = float((col <= q).mean()) if len(col) > 0 else 0.0
            per_etf_cov[ticker] = round(covered, 4)

        pooled_q = q_info["pooled"]
        flat     = scores[~np.isnan(scores)].ravel()
        pooled_cov = float((flat <= pooled_q).mean()) if len(flat) > 0 else 0.0

        coverage[alpha_str] = {
            "per_etf": per_etf_cov,
            "pooled":  round(pooled_cov, 4),
            "target":  round(target, 4),
        }

    return coverage


# ── Save + upload ─────────────────────────────────────────────────────────────

def save_conformal(module: str, scores_dict: dict,
                   quantiles: dict, coverage: dict) -> str:
    os.makedirs(MODELS_DIR, exist_ok=True)
    out = {
        "module":        module,
        "calibrated_at": datetime.utcnow().isoformat(),
        "n_cal":         scores_dict["n_cal"],
        "val_start":     scores_dict["val_start"],
        "val_end":       scores_dict["val_end"],
        "lookback":      scores_dict["lookback"],
        "tickers":       scores_dict["tickers"],
        "alpha_levels":  ALPHA_LEVELS,
        "score_stats": {
            "mean":   scores_dict["score_mean"],
            "p50":    scores_dict["score_p50"],
            "p90":    scores_dict["score_p90"],
            "note":   "Scores ∈ [0.5,1.0]. Near 0.5 = MC well-centred. Near 1.0 = MC missed the tail.",
        },
        "quantiles": quantiles,
        "coverage":  coverage,
    }
    fname = f"conformal_{module.lower()}.json"
    path  = os.path.join(MODELS_DIR, fname)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[calibrate] Saved → {path}")
    return fname


def upload_conformal(module: str):
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        print("[calibrate] No HF_TOKEN — skipping upload.")
        return
    try:
        from huggingface_hub import HfApi
        api   = HfApi(token=token)
        fname = f"conformal_{module.lower()}.json"
        path  = os.path.join(MODELS_DIR, fname)
        with open(path, "rb") as f:
            api.upload_file(
                path_or_fileobj=f,
                path_in_repo=f"conformal/{fname}",
                repo_id=HF_DATASET_OUT,
                repo_type="dataset",
                commit_message=f"Update conformal calibration {module}",
            )
        print(f"[calibrate] Uploaded → {HF_DATASET_OUT}/conformal/{fname}")
    except Exception as e:
        print(f"[calibrate] WARNING: Upload failed: {e}")


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(module: str, scores_dict: dict,
                  quantiles: dict, coverage: dict):
    print(f"\n{'─'*60}")
    print(f"Conformal calibration summary — {module}")
    print(f"{'─'*60}")
    print(f"  Score stats: "
          f"mean={scores_dict['score_mean']:.4f}  "
          f"p50={scores_dict['score_p50']:.4f}  "
          f"p90={scores_dict['score_p90']:.4f}")
    print()
    for alpha_str in sorted(quantiles.keys(), reverse=True):
        q_info   = quantiles[alpha_str]
        cov      = coverage[alpha_str]
        target   = cov["target"]
        achieved = cov["pooled"]
        pooled_q = q_info["pooled"]
        status   = "✓" if achieved >= target - 0.01 else "✗"
        lo_pct   = round((1 - pooled_q) * 100, 1)
        hi_pct   = round(pooled_q * 100, 1)
        print(f"  α={float(alpha_str):.0%}  "
              f"target≥{target:.0%}  achieved={achieved:.1%}  {status}  "
              f"(MC interval: [{lo_pct}%, {hi_pct}%] of samples)")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

def calibrate_module(module: str):
    """
    Full calibration pipeline for one module.
    Reads the best lookback from the existing copula_signal.json on HF
    (written by train_fi.py / train_equity.py), or falls back to 45d default.
    """
    best_lookback = _get_best_lookback(module)
    print(f"[calibrate] Module={module}  best_lookback={best_lookback}d")

    scores_dict = collect_calibration_scores(module, best_lookback)
    quantiles   = compute_quantiles(scores_dict)
    coverage    = empirical_coverage(scores_dict, quantiles)
    save_conformal(module, scores_dict, quantiles, coverage)
    upload_conformal(module)
    print_summary(module, scores_dict, quantiles, coverage)


def _get_best_lookback(module: str) -> int:
    """
    Try to read the best lookback from the already-uploaded signal JSON.
    Falls back to 45 if not available.
    """
    token = os.environ.get("HF_TOKEN", "")
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id=HF_DATASET_OUT,
            filename="results/copula_signal.json",
            repo_type="dataset",
            token=token or None,
            force_download=True,
        )
        with open(path) as f:
            sig = json.load(f)
        lb = sig.get(module, {}).get("lookback_days")
        if lb:
            return int(lb)
    except Exception:
        pass
    print(f"[calibrate] Could not read lookback from HF — defaulting to 45d.")
    return 45


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute conformal quantiles from COPULA val-set MC distributions."
    )
    parser.add_argument("--module", choices=["FI", "EQ", "both"], default="both")
    args = parser.parse_args()

    modules = ["FI", "EQ"] if args.module == "both" else [args.module]
    for mod in modules:
        calibrate_module(mod)

    print("[calibrate] All done.")
