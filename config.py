"""
P2-ETF-COPULA-ENGINE  ·  config.py
All constants, universe definitions, and hyperparameters.
"""

# ── Hugging Face ──────────────────────────────────────────────────────────────
HF_DATASET_IN  = "P2SAMAPA/fi-etf-macro-signal-master-data"
HF_DATASET_OUT = "P2SAMAPA/p2-etf-copula-results"

# ── ETF universes ─────────────────────────────────────────────────────────────
FI_ETFS     = ["TLT", "LQD", "HYG", "VNQ", "GLD", "SLV", "VCIT"]
EQUITY_ETFS = ["QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "IWF", "XSD", "XBI" 
               "XLP", "XLU", "GDX", "XME", "IWM"]

BENCHMARK_FI     = "AGG"
BENCHMARK_EQUITY = "SPY"

# ── Macro features ────────────────────────────────────────────────────────────
MACRO_COLS = ["VIX", "DXY", "T10Y2Y", "TBILL_3M", "IG_SPREAD", "HY_SPREAD"]

# ── Lookback windows evaluated during validation (model picks best) ───────────
LOOKBACK_CANDIDATES = [30, 45, 60]   # trading days

# ── Train / val / test proportions ───────────────────────────────────────────
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
# test = remaining 0.10 — never seen during training or hyperparam selection

# ── GARCH marginal settings ───────────────────────────────────────────────────
GARCH_P    = 1
GARCH_Q    = 1
GARCH_DIST = "normal"          # "normal" | "t"

# ── Copula families evaluated (best by AIC wins per module) ──────────────────
COPULA_FAMILIES = ["gaussian", "student", "clayton", "gumbel", "frank"]

# ── Monte Carlo ───────────────────────────────────────────────────────────────
N_MC_SAMPLES = 5_000
RANDOM_SEED  = 42

# ── Regime detection (KMeans on macro — conditions copula params) ─────────────
N_REGIMES       = 3
REGIME_FEATURES = ["VIX", "T10Y2Y", "HY_SPREAD"]

# ── Scoring weights ───────────────────────────────────────────────────────────
# score = W_MEAN * E[R] - W_TAIL * lambda_lower + W_SHARPE * sharpe_proxy
W_MEAN   = 0.60
W_TAIL   = 0.25
W_SHARPE = 0.15

# ── Transaction cost ──────────────────────────────────────────────────────────
TRANSACTION_COST_BPS = 12      # deducted only when switching from previous pick

# ── Output filenames ──────────────────────────────────────────────────────────
OUTPUT_JSON       = "copula_signal.json"
SIGNAL_HISTORY_FI = "signal_history_fi.csv"
SIGNAL_HISTORY_EQ = "signal_history_eq.csv"
METRICS_FI        = "metrics_fi.json"
METRICS_EQ        = "metrics_eq.json"
