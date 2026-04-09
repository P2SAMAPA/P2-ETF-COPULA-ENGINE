"""
P2-ETF-COPULA-ENGINE  ·  loader.py
Load fi-etf-macro-signal-master-data from Hugging Face, compute log-returns,
and split into train / val / test sets.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from datasets import load_dataset

from config import (
    HF_DATASET_IN,
    FI_ETFS, EQUITY_ETFS,
    BENCHMARK_FI, BENCHMARK_EQUITY,
    MACRO_COLS,
    TRAIN_RATIO, VAL_RATIO,
)


# ── Raw data ──────────────────────────────────────────────────────────────────

def load_raw() -> pd.DataFrame:
    """
    Download the master dataset from Hugging Face and return a clean
    DataFrame indexed by date (NYSE trading days only, 2008-present).
    """
    ds  = load_dataset(HF_DATASET_IN, split="train")
    df  = ds.to_pandas()

    # Date column is stored as __index_level_0__
    if "__index_level_0__" in df.columns:
        df = df.rename(columns={"__index_level_0__": "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    return df


# ── Returns ───────────────────────────────────────────────────────────────────

def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns from a price DataFrame.  First row dropped (NaN)."""
    return np.log(prices / prices.shift(1)).iloc[1:]


# ── Module-level data prep ────────────────────────────────────────────────────

def get_module_data(module: str, start_date: str | None = None) -> dict:
    """
    Prepare returns and macro for one module ('FI' or 'EQ').

    Parameters
    ----------
    module     : 'FI' or 'EQ'
    start_date : optional ISO date string to trim the start of the data

    Returns
    -------
    dict with keys:
        returns    – pd.DataFrame  log returns for the ETFs in this module
        benchmark  – pd.Series     log returns for the benchmark
        macro      – pd.DataFrame  macro features (forward-filled, same index)
        splits     – dict {train, val, test} each a tuple (returns, macro)
        etfs       – list[str]
        bm_name    – str
    """
    raw = load_raw()

    if start_date:
        raw = raw[raw.index >= start_date]

    if module == "FI":
        etfs    = FI_ETFS
        bm_name = BENCHMARK_FI
    else:
        etfs    = EQUITY_ETFS
        bm_name = BENCHMARK_EQUITY

    all_tickers = etfs + [bm_name]

    # Safety: only keep tickers that actually exist in the dataset
    available   = [t for t in all_tickers if t in raw.columns]
    missing     = [t for t in all_tickers if t not in raw.columns]
    if missing:
        print(f"  [loader] WARNING: tickers not found in dataset, dropping: {missing}")

    # Update etfs list to only those that are available (exclude benchmark from etf list)
    etfs    = [t for t in etfs    if t in available]
    bm_name = bm_name if bm_name in available else None
    if bm_name is None:
        raise ValueError(f"Benchmark {bm_name} not found in dataset columns.")

    all_tickers = etfs + [bm_name]
    prices      = raw[all_tickers].ffill().dropna()

    rets = log_returns(prices[etfs])
    bm_r = log_returns(prices[[bm_name]])[bm_name]

    # Macro: forward-fill (FRED series update weekly/monthly)
    macro = raw[MACRO_COLS].ffill()
    macro = macro.reindex(rets.index, method="ffill").dropna()

    # Align returns to macro index
    common  = rets.index.intersection(macro.index)
    rets    = rets.loc[common]
    bm_r    = bm_r.loc[common]
    macro   = macro.loc[common]

    # Train / val / test split — strictly chronological, no shuffling
    n       = len(rets)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)

    splits = {
        "train": (rets.iloc[:n_train],        macro.iloc[:n_train]),
        "val":   (rets.iloc[n_train:n_train + n_val],
                  macro.iloc[n_train:n_train + n_val]),
        "test":  (rets.iloc[n_train + n_val:], macro.iloc[n_train + n_val:]),
    }

    return {
        "returns":   rets,
        "benchmark": bm_r,
        "macro":     macro,
        "splits":    splits,
        "etfs":      etfs,
        "bm_name":   bm_name,
    }
