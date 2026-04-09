"""
P2-ETF-COPULA-ENGINE  ·  train_fi.py
Full training pipeline for the Fixed Income / Commodities module.

Steps
-----
1. Load data  (train 80% / val 10% / test 10%)
2. Fit GARCH marginals on training set
3. Optimise lookback window on validation set (30 / 45 / 60 days)
4. Fit best copula on training set with optimal lookback
5. Walk-forward backtest on test set
6. Generate live prediction for next NYSE trading day
7. Download existing signal JSON from HF, merge FI key, re-upload
"""

import os, json, datetime
import pandas as pd
from huggingface_hub import hf_hub_download

from config import (
    HF_DATASET_OUT, OUTPUT_JSON,
    SIGNAL_HISTORY_FI, METRICS_FI,
    N_MC_SAMPLES,
)
from loader        import get_module_data
from marginals     import fit_all_marginals, build_uniform_matrix
from copula_model  import fit_copula, tail_dependence, mc_simulate
from regime        import fit_regime_model, predict_regime
from scorer        import score_etfs, build_signal
from optimise      import optimise_lookback
from backtest      import run_backtest
from calendar_utils import next_trading_day
from upload        import upload_results


def run_fi():
    print("=" * 60)
    print("P2-ETF-COPULA-ENGINE  |  Module: FI / Commodities")
    print("=" * 60)

    # ── 1. Load data ──────────────────────────────────────────────
    print("\n[1/7] Loading data...")
    data   = get_module_data("FI")
    rets   = data["returns"]
    macro  = data["macro"]
    bm_r   = data["benchmark"]

    train_r, train_m = data["splits"]["train"]
    val_r,   val_m   = data["splits"]["val"]
    test_r,  test_m  = data["splits"]["test"]

    print(f"      Total days : {len(rets)}")
    print(f"      Train      : {len(train_r)}  ({train_r.index[0].date()} -> {train_r.index[-1].date()})")
    print(f"      Val        : {len(val_r)}   ({val_r.index[0].date()} -> {val_r.index[-1].date()})")
    print(f"      Test       : {len(test_r)}   ({test_r.index[0].date()} -> {test_r.index[-1].date()})")

    # ── 2. Fit GARCH marginals on training set ────────────────────
    print("\n[2/7] Fitting GARCH marginals (train set)...")
    fitted_marginals = fit_all_marginals(train_r)
    print(f"      Fitted {len(fitted_marginals)} GARCH models.")

    # ── 3. Optimise lookback on validation set ────────────────────
    print("\n[3/7] Optimising lookback window on validation set...")
    opt     = optimise_lookback(val_r, verbose=True)
    best_lb = opt["best_lookback"]

    # ── 4. Fit copula on training set with optimal lookback ────────
    print(f"\n[4/7] Fitting copula (lookback={best_lb}d, train set)...")
    train_window = train_r.iloc[-best_lb:]
    fitted_m_win = fit_all_marginals(train_window)
    u_df         = build_uniform_matrix(fitted_m_win, train_window)
    cop_fit      = fit_copula(u_df)

    print(f"      Best copula family : {cop_fit['best_family']}")
    print(f"      AIC scores         : {cop_fit['aic_scores']}")

    # ── 5. Backtest on test set ───────────────────────────────────
    print(f"\n[5/7] Running walk-forward backtest on test set...")
    bt = run_backtest(test_r, bm_r.reindex(test_r.index), best_lb, verbose=True)

    # ── 6. Live prediction ────────────────────────────────────────
    print(f"\n[6/7] Generating live prediction...")
    live_window   = rets.iloc[-best_lb:]
    live_fitted_m = fit_all_marginals(live_window)
    live_u        = build_uniform_matrix(live_fitted_m, live_window)
    live_cop      = fit_copula(live_u)
    live_td       = tail_dependence(live_u)
    live_sim      = mc_simulate(live_cop, live_fitted_m, live_window,
                                n_samples=N_MC_SAMPLES)

    regime_model  = fit_regime_model(macro)
    latest_macro  = macro.iloc[-1]
    regime_id     = predict_regime(regime_model, latest_macro)
    regime_name   = regime_model["regime_names"].get(regime_id, str(regime_id))

    prev_pick = _load_prev_pick_from_hf(SIGNAL_HISTORY_FI)
    scores    = score_etfs(live_sim, live_td, prev_pick)
    ntd       = next_trading_day(rets.index[-1].date())
    signal    = build_signal(
        scores, "FI", regime_id, regime_name,
        live_cop["best_family"], best_lb, ntd, latest_macro
    )

    print(f"      Pick             : {signal['pick']}  ({signal['conviction_pct']:.1f}%)")
    print(f"      Next trading day : {ntd}")
    print(f"      Regime           : {regime_name}")
    print(f"      Copula family    : {signal['copula_family']}")
    print(f"      Lookback         : {signal['lookback_days']} days")

    # ── 7. Save and upload ────────────────────────────────────────
    print("\n[7/7] Saving results and uploading to Hugging Face...")

    # CRITICAL: download the existing JSON from HF first so we preserve
    # any EQ key that a parallel/prior run may have written.
    existing = _fetch_signal_json_from_hf()
    existing["FI"]           = signal
    existing["generated_at"] = datetime.datetime.utcnow().isoformat()
    _save_json(existing, OUTPUT_JSON)

    # Metrics
    _save_json(bt["metrics"], METRICS_FI)

    # Signal history CSV
    if not bt["signal_log"].empty:
        hist           = bt["signal_log"].reset_index()
        hist["module"] = "FI"
        _append_csv(hist, SIGNAL_HISTORY_FI)

    upload_results([OUTPUT_JSON, METRICS_FI, SIGNAL_HISTORY_FI])
    print("\nDone — FI module complete.")
    print(f"Signal JSON keys written: {list(existing.keys())}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fetch_signal_json_from_hf() -> dict:
    """
    Download the current copula_signal.json from HF so we can merge into it
    rather than overwriting the other module's key.
    Returns empty dict if file doesn't exist yet (first run).
    """
    token = os.environ.get("HF_TOKEN")
    try:
        path = hf_hub_download(
            repo_id=HF_DATASET_OUT,
            filename="results/copula_signal.json",
            repo_type="dataset",
            token=token,
            force_download=True,
        )
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f"  [info] No existing signal JSON on HF (first run or error: {e}). Starting fresh.")
        return {}


def _load_prev_pick_from_hf(csv_filename: str) -> str | None:
    """Download signal history CSV from HF to get yesterday's pick."""
    token = os.environ.get("HF_TOKEN")
    try:
        path = hf_hub_download(
            repo_id=HF_DATASET_OUT,
            filename=f"results/{csv_filename}",
            repo_type="dataset",
            token=token,
            force_download=True,
        )
        df = pd.read_csv(path)
        if "pick" in df.columns and len(df) > 0:
            return str(df["pick"].iloc[-1])
    except Exception:
        pass
    return None


def _save_json(obj: dict, path: str):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def _append_csv(df: pd.DataFrame, path: str):
    header = not os.path.exists(path)
    df.to_csv(path, mode="a", header=header, index=False)


if __name__ == "__main__":
    run_fi()
