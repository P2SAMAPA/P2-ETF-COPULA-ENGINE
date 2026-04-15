# conformal/predict_conformal.py — Wrap COPULA signals with conformal intervals
#
# HOW IT WORKS
# ════════════
# Each daily run of train_fi.py / train_equity.py already generates 5,000 MC
# return samples per ETF (from mc_simulate). This script re-uses that same
# pipeline on today's window to generate fresh MC samples, then applies the
# conformal quantile q̂ from calibrate.py to extract guaranteed intervals:
#
#   interval_lo(ETF_i) = MC_quantile(1 − q̂_i)
#   interval_hi(ETF_i) = MC_quantile(q̂_i)
#
# q̂_i ∈ [0.5, 1.0] — e.g. q̂=0.92 means use the 8th–92nd percentile of MC.
# Coverage guarantee: P(actual return ∈ interval) ≥ 1−α.
#
# SELF-HEALING: auto-calibrates if conformal params are missing.
#
# Output:
#   results/copula_signal_conformal.json  (uploaded to HF alongside the main signal)
#   conformal/signal_history_conformal_{fi|eq}.json
#
# Usage:
#   python -m conformal.predict_conformal --module both

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
    N_MC_SAMPLES,
)
from loader import get_module_data
from marginals import fit_all_marginals, build_uniform_matrix
from copula_model import fit_copula, tail_dependence, mc_simulate

MODELS_DIR = "models"
ALPHA_LEVELS = [0.90, 0.80, 0.70]


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_copula_signal() -> dict:
    """Load the latest copula signal JSON — local first, then HF."""
    local = "results/copula_signal.json"
    if os.path.exists(local):
        with open(local) as f:
            return json.load(f)
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id=HF_DATASET_OUT,
            filename="results/copula_signal.json",
            repo_type="dataset",
            token=os.environ.get("HF_TOKEN") or None,
            force_download=True,
        )
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        raise FileNotFoundError(
            f"Could not find copula_signal.json: {e}. "
            "Run train_fi.py / train_equity.py first."
        )


def load_conformal_params(module: str) -> dict | None:
    """Return conformal params or None if not yet calibrated."""
    fname = f"conformal_{module.lower()}.json"
    local = os.path.join(MODELS_DIR, fname)
    if os.path.exists(local):
        with open(local) as f:
            return json.load(f)
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id=HF_DATASET_OUT,
            filename=f"conformal/{fname}",
            repo_type="dataset",
            token=os.environ.get("HF_TOKEN") or None,
            local_dir=MODELS_DIR,
        )
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def ensure_calibrated(module: str) -> dict:
    """Return conformal params, auto-calibrating if missing."""
    params = load_conformal_params(module)
    if params is not None:
        print(f"[predict_conformal] Conformal params loaded for {module}.")
        return params

    print(f"[predict_conformal] No conformal params for {module} — "
          f"auto-calibrating now (first run)...")

    from conformal.calibrate import calibrate_module
    calibrate_module(module)

    params = load_conformal_params(module)
    if params is None:
        raise RuntimeError(
            f"Calibration ran but conformal_{module.lower()}.json still not found."
        )
    return params


# ── Live MC simulation ────────────────────────────────────────────────────────

def run_live_mc(module: str, lookback: int) -> tuple[pd.DataFrame, dict]:
    """
    Re-run the copula pipeline on the live window to generate fresh MC samples.
    Returns (sim_returns DataFrame (N_MC, n_etfs), copula_fit dict).
    """
    data       = get_module_data(module)
    rets       = data["returns"]
    live_window = rets.iloc[-lookback:]

    fitted_m   = fit_all_marginals(live_window)
    u_df       = build_uniform_matrix(fitted_m, live_window)
    cop_fit    = fit_copula(u_df)
    sim        = mc_simulate(cop_fit, fitted_m, live_window, n_samples=N_MC_SAMPLES)

    return sim, cop_fit


# ── Interval computation ──────────────────────────────────────────────────────

def compute_conformal_intervals(sim_returns: pd.DataFrame,
                                conformal_params: dict,
                                module_signal: dict) -> dict:
    """
    For each ETF, extract the conformal prediction interval from MC samples.

    interval_lo = MC_quantile(1 − q̂)
    interval_hi = MC_quantile(q̂)

    Also computes:
    - interval_contains_zero: bool (signal classification)
    - interval_fully_positive: bool (confident long)
    - interval_fully_negative: bool (confirmed avoid)
    - mc_percentile_of_expected_return: where the copula's expected_return
      falls in the MC distribution (sanity check)
    """
    quantiles = conformal_params["quantiles"]
    tickers   = conformal_params["tickers"]

    conformal_per_etf = {}
    for ticker in tickers:
        if ticker not in sim_returns.columns:
            continue

        mc_arr = sim_returns[ticker].values
        mc_arr_sorted = np.sort(mc_arr)

        # Expected return from copula signal (point estimate)
        expected_r = None
        for row in module_signal.get("all_scores", []):
            if row.get("ticker") == ticker:
                expected_r = row.get("expected_return")
                break

        intervals = {}
        for alpha_str, q_info in quantiles.items():
            q    = q_info["per_etf"].get(ticker, q_info["pooled"])
            q    = float(np.clip(q, 0.5, 0.999))
            lo_q = 1.0 - q
            hi_q = q
            lo   = float(np.quantile(mc_arr, lo_q))
            hi   = float(np.quantile(mc_arr, hi_q))
            intervals[alpha_str] = {
                "lo":                  round(lo, 6),
                "hi":                  round(hi, 6),
                "width":               round(hi - lo, 6),
                "q_hat":               round(q, 5),
                "mc_lo_pct":           round(lo_q * 100, 1),
                "mc_hi_pct":           round(hi_q * 100, 1),
                "fully_positive":      bool(lo > 0),
                "fully_negative":      bool(hi < 0),
                "contains_zero":       bool(lo <= 0 <= hi),
            }

        # Where does the expected_return fall in MC distribution?
        mc_pct_of_er = None
        if expected_r is not None:
            mc_pct_of_er = round(float(np.mean(mc_arr <= expected_r)) * 100, 1)

        conformal_per_etf[ticker] = {
            "expected_return":         round(float(np.mean(mc_arr)), 6),
            "mc_std":                  round(float(np.std(mc_arr)), 6),
            "mc_p10":                  round(float(np.percentile(mc_arr, 10)), 6),
            "mc_p25":                  round(float(np.percentile(mc_arr, 25)), 6),
            "mc_median":               round(float(np.median(mc_arr)), 6),
            "mc_p75":                  round(float(np.percentile(mc_arr, 75)), 6),
            "mc_p90":                  round(float(np.percentile(mc_arr, 90)), 6),
            "mc_pct_positive":         round(float(np.mean(mc_arr > 0)) * 100, 1),
            "mc_percentile_of_expected_return": mc_pct_of_er,
            "intervals":               intervals,
        }

    return conformal_per_etf


# ── Build conformal signal ────────────────────────────────────────────────────

def build_conformal_signal(module: str,
                           module_signal: dict,
                           conformal_per_etf: dict,
                           conformal_params: dict,
                           alpha: str = "0.9") -> dict:
    """
    Package conformal results into a signal dict parallel to the copula signal.
    Top pick is unchanged (same as copula signal — conformal only adds intervals).
    """
    pick        = module_signal.get("pick", "")
    conviction  = module_signal.get("conviction_pct", 0)
    ntd         = module_signal.get("next_trading_day", "")

    top_interval = conformal_per_etf.get(pick, {}).get("intervals", {}).get(alpha, {})
    lo90 = top_interval.get("lo")
    hi90 = top_interval.get("hi")

    if lo90 is not None and hi90 is not None:
        if lo90 > 0:
            signal_class = "STRONG — entire 90% CI positive"
        elif hi90 < 0:
            signal_class = "AVOID — entire 90% CI negative"
        else:
            signal_class = "UNCERTAIN — CI crosses zero"
    else:
        signal_class = "UNKNOWN"

    return {
        "module":               module,
        "next_trading_day":     ntd,
        "generated_at":         datetime.utcnow().isoformat(),
        "copula_generated_at":  module_signal.get("generated_at",
                                                   module_signal.get("last_updated")),

        # Top pick — identical to copula signal
        "pick":                 pick,
        "conviction_pct":       conviction,
        "signal_class":         signal_class,
        "top_interval_90":      top_interval,

        # Copula metadata carried through
        "copula_family":        module_signal.get("copula_family"),
        "lookback_days":        module_signal.get("lookback_days"),
        "regime_id":            module_signal.get("regime_id"),
        "regime_name":          module_signal.get("regime_name"),
        "macro_pills":          module_signal.get("macro_pills", {}),

        # Conformal metadata
        "n_cal":                conformal_params["n_cal"],
        "cal_period":           (f"{conformal_params['val_start']} → "
                                 f"{conformal_params['val_end']}"),
        "calibrated_at":        conformal_params["calibrated_at"],
        "coverage_diagnostics": conformal_params["coverage"],
        "alpha_levels":         ALPHA_LEVELS,

        # Per-ETF conformal results
        "conformal_per_etf":    conformal_per_etf,

        # Carry through all_scores for comparison
        "all_scores":           module_signal.get("all_scores", []),
    }


# ── Save + upload ─────────────────────────────────────────────────────────────

def save_and_upload(sig_fi=None, sig_eq=None):
    os.makedirs("results", exist_ok=True)

    combined = {
        "generated_at": datetime.utcnow().isoformat(),
        "FI":           sig_fi or None,
        "EQ":           sig_eq or None,
    }

    local_path = "results/copula_signal_conformal.json"
    with open(local_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"[predict_conformal] Saved locally → {local_path}")

    token = os.environ.get("HF_TOKEN", "")
    if not token:
        print("[predict_conformal] No HF_TOKEN — skipping upload.")
        return

    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        with open(local_path, "rb") as f:
            api.upload_file(
                path_or_fileobj=f,
                path_in_repo="conformal/copula_signal_conformal.json",
                repo_id=HF_DATASET_OUT,
                repo_type="dataset",
                commit_message=f"Update conformal signals ({combined['generated_at']})",
            )
        print(f"[predict_conformal] Uploaded → {HF_DATASET_OUT}")
    except Exception as e:
        print(f"[predict_conformal] WARNING: upload failed: {e}")

    for sig, mod in [(sig_fi, "fi"), (sig_eq, "eq")]:
        if sig:
            _update_history(sig, mod)


def _update_history(sig: dict, mod: str):
    os.makedirs(MODELS_DIR, exist_ok=True)
    history_path = os.path.join(MODELS_DIR, f"signal_history_conformal_{mod}.json")
    history = []
    if os.path.exists(history_path):
        with open(history_path) as f:
            history = json.load(f)
    else:
        try:
            from huggingface_hub import hf_hub_download
            dl = hf_hub_download(
                repo_id=HF_DATASET_OUT,
                filename=f"conformal/signal_history_conformal_{mod}.json",
                repo_type="dataset",
                token=os.environ.get("HF_TOKEN") or None,
                local_dir=MODELS_DIR,
            )
            with open(dl) as f:
                history = json.load(f)
        except Exception:
            history = []

    iv90 = sig.get("top_interval_90", {})
    record = {
        "signal_date":       sig.get("next_trading_day"),
        "pick":              sig.get("pick"),
        "conviction_pct":    sig.get("conviction_pct"),
        "signal_class":      sig.get("signal_class"),
        "generated_at":      sig.get("generated_at"),
        "interval_90_lo":    iv90.get("lo"),
        "interval_90_hi":    iv90.get("hi"),
        "interval_90_width": iv90.get("width"),
        "q_hat_90":          iv90.get("q_hat"),
        "copula_family":     sig.get("copula_family"),
        "regime_name":       sig.get("regime_name"),
        "actual_return":     None,   # backfilled next run
        "interval_covered":  None,
    }

    existing = {r.get("signal_date") for r in history}
    if record["signal_date"] not in existing:
        history.append(record)
    else:
        # Backfill if actual return now known
        for r in history:
            if r.get("signal_date") == record["signal_date"]:
                if record.get("actual_return") is not None:
                    r["actual_return"] = record["actual_return"]
                    lo = r.get("interval_90_lo")
                    hi = r.get("interval_90_hi")
                    if lo is not None and hi is not None:
                        r["interval_covered"] = bool(
                            lo <= record["actual_return"] <= hi
                        )
                break

    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    token = os.environ.get("HF_TOKEN", "")
    if not token:
        return
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        with open(history_path, "rb") as f:
            api.upload_file(
                path_or_fileobj=f,
                path_in_repo=f"conformal/signal_history_conformal_{mod}.json",
                repo_id=HF_DATASET_OUT,
                repo_type="dataset",
                commit_message=f"Update conformal history {mod.upper()} "
                               f"({record['signal_date']})",
            )
        print(f"[predict_conformal] Uploaded history {mod.upper()}")
    except Exception as e:
        print(f"[predict_conformal] WARNING: history upload failed: {e}")


# ── Per-module runner ─────────────────────────────────────────────────────────

def run_module(module: str, copula_signal: dict) -> dict | None:
    """
    Returns a conformal-wrapped signal dict, or None if the copula signal
    is missing / incomplete.
    """
    mod_sig = copula_signal.get(module)
    if not mod_sig or not isinstance(mod_sig, dict) or "pick" not in mod_sig:
        print(f"[predict_conformal] Copula signal for {module} is absent — skipping.")
        return None

    lookback = mod_sig.get("lookback_days", 45)

    try:
        params = ensure_calibrated(module)
    except Exception as e:
        print(f"[predict_conformal] {module} calibration failed: {e}")
        return None

    print(f"[predict_conformal] Running live MC for {module} "
          f"(lookback={lookback}d, {N_MC_SAMPLES} samples)...")

    try:
        sim, cop_fit = run_live_mc(module, lookback)
    except Exception as e:
        print(f"[predict_conformal] MC simulation failed for {module}: {e}")
        return None

    conformal_per_etf = compute_conformal_intervals(sim, params, mod_sig)
    signal            = build_conformal_signal(module, mod_sig, conformal_per_etf, params)

    iv90 = signal.get("top_interval_90", {})
    print(f"[predict_conformal]   {module}: {signal['pick']}  "
          f"90% CI=[{iv90.get('lo','?')}, {iv90.get('hi','?')}]  "
          f"class='{signal['signal_class']}'  ntd={signal['next_trading_day']}")
    return signal


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Wrap COPULA signals with conformal prediction intervals. "
            "Auto-calibrates on first run if conformal params are missing."
        )
    )
    parser.add_argument("--module", choices=["FI", "EQ", "both"], default="both")
    args = parser.parse_args()

    print("[predict_conformal] Loading copula signal...")
    copula_signal = load_copula_signal()

    modules = ["FI", "EQ"] if args.module == "both" else [args.module]
    sig_fi = sig_eq = None

    if "FI" in modules:
        sig_fi = run_module("FI", copula_signal)
    if "EQ" in modules:
        sig_eq = run_module("EQ", copula_signal)

    save_and_upload(sig_fi, sig_eq)
    print("[predict_conformal] Done.")
