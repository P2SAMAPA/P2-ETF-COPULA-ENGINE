# P2-ETF-COPULA-ENGINE

**Copula-Based Joint Tail Dependency Engine for ETF Selection**
Absolute return maximisation · Tail-risk aware · CPU-only · 12bps transaction cost

---

## Overview

P2-ETF-COPULA-ENGINE models the **joint distribution of ETF returns** rather than forecasting each ETF in isolation. Instead of asking "what return will TLT produce tomorrow?", it asks "given the full co-movement structure between all ETFs today, which one has the best risk-adjusted expected return tomorrow?"

Markets are non-linear and tail events are correlated — especially in the fixed income universe where all rates-sensitive ETFs crash together. Traditional correlation (Pearson) misses this. Copulas capture it exactly.

**Core principle:**
```
Fit GARCH marginals → Transform to uniforms (PIT)
→ Fit copula to joint structure → Sample correlated returns (MC)
→ Score ETFs: E[R] − λ_lower penalty + Sharpe proxy
→ Pick ETF with highest net score → Predict for next NYSE trading day
```

---

## Two Modules

| Module | ETF Universe | Benchmark |
|--------|-------------|-----------|
| **FI / Commodities** | TLT · LQD · HYG · VNQ · GLD · SLV · MBB | AGG |
| **Equity Sectors**   | QQQ · XLK · XLF · XLE · XLV · XLI · XLY · XLP · XLU · GDX · XME · IWM | SPY |

No CASH output — the engine always picks the best ETF from the universe.

---

## Data

**Source:** [`P2SAMAPA/fi-etf-macro-signal-master-data`](https://huggingface.co/datasets/P2SAMAPA/fi-etf-macro-signal-master-data)

**Period:** 2008-01-01 → present (daily)

**Features used:**
- ETF prices → daily log-returns
- Macro: VIX · DXY · T10Y2Y · TBILL_3M · IG_SPREAD · HY_SPREAD (regime detection)

---

## Train / Val / Test Split

Strictly chronological — no shuffling, no data leakage:

| Set | Proportion | Approx. dates | Purpose |
|-----|-----------|--------------|---------|
| Train | 80% | 2008 → ~2021 | GARCH + copula fitting |
| Val   | 10% | ~2021 → ~2023 | Lookback window optimisation |
| Test  | 10% | ~2023 → present | OOS evaluation (never touched during training) |

---

## Lookback Window Optimisation

The engine tests **three rolling window lengths** (30, 45, 60 trading days) on the **validation set only**, selecting the window that maximises cumulative log-return with 12bps transaction cost. This window is then locked in for test-set evaluation and live prediction.

---

## Architecture

```
config.py            All constants, ETF universes, hyperparameters
loader.py            Load HF dataset, compute log-returns, split data
marginals.py         GARCH(1,1) per ETF → PIT → uniform pseudo-observations
copula_model.py      Fit Gaussian/Student/Clayton/Gumbel/Frank, select by AIC
                     Tail dependence matrix · Monte Carlo sampling
regime.py            KMeans (K=3) on VIX + T10Y2Y + HY_SPREAD → regime label
scorer.py            Score = W_mean·E[R] - W_tail·λ_L + W_sharpe·Sharpe
optimise.py          Grid search over lookback [30,45,60] on val set
backtest.py          Walk-forward OOS backtest on test set
calendar_utils.py    NYSE next trading day (pandas_market_calendars)
train_fi.py          Full pipeline for FI module
train_equity.py      Full pipeline for Equity module
upload.py            Push results to HF dataset
app.py               Streamlit UI (two tabs: FI · Equity)
```

---

## Copula Families

Five families are evaluated per run; the one with the lowest AIC wins:

| Family | Tail dependence | Best for |
|--------|----------------|---------|
| Gaussian | None | Normal market regimes |
| Student-t | Symmetric (upper + lower) | Stress regimes |
| Clayton | **Lower tail** (joint drawdowns) | Risk-off / crash scenarios |
| Gumbel | **Upper tail** (joint rallies) | Risk-on / momentum regimes |
| Frank | None (Archimedean) | Moderate dependence |

---

## Scoring

```
score(ETF_i) = 0.60 × E[R_i]
             − 0.25 × avg_λ_lower(i, all others)
             + 0.15 × Sharpe_proxy(i)

net_score    = score − 12bps  (only if switching from previous pick)
```

The **lower tail dependence penalty** prevents the engine from picking an ETF that co-crashes with everything else in a stress scenario — even if its individual expected return looks attractive.

---

## Output

Results are uploaded to [`P2SAMAPA/p2-etf-copula-results`](https://huggingface.co/datasets/P2SAMAPA/p2-etf-copula-results):

| File | Contents |
|------|---------|
| `results/copula_signal.json` | Daily picks for FI + Equity with conviction, macro pills, regime |
| `results/signal_history_fi.csv` | Running log of all FI picks and realised returns |
| `results/signal_history_eq.csv` | Running log of all Equity picks and realised returns |
| `results/metrics_fi.json` | OOS backtest metrics for FI module |
| `results/metrics_eq.json` | OOS backtest metrics for Equity module |

### Signal JSON schema

```json
{
  "FI": {
    "module": "FI",
    "next_trading_day": "2026-04-10",
    "pick": "GLD",
    "conviction_pct": 34.2,
    "expected_return": 0.000421,
    "avg_lower_tail_dep": 0.087,
    "second_pick": "TLT",
    "copula_family": "clayton",
    "lookback_days": 45,
    "regime_id": 2,
    "regime_name": "Risk-Off (High Vol / Stress)",
    "macro_pills": { "VIX": 18.4, "T10Y2Y": -0.12, ... }
  },
  "EQ": { ... },
  "generated_at": "2026-04-09T22:55:00"
}
```

---

## Daily Cron Schedule

Runs at **22:45 UTC Mon–Fri** (after the data update at 22:00 UTC):
- `train_fi.py`  → FI module (runs first)
- `train_equity.py` → Equity module (runs after FI, sequential to avoid HF conflicts)

---

## GitHub Actions Setup

1. Fork this repository
2. Add `HF_TOKEN` as a repository secret (Settings → Secrets → Actions)
3. The workflow runs automatically at 22:45 UTC Mon–Fri

For manual runs: Actions → Daily Copula Training → Run workflow → choose `fi`, `eq`, or `both`

---

## CPU Requirements

All computation runs on the **GitHub Actions free tier (ubuntu-latest, CPU-only)**:

| Step | Estimated time |
|------|---------------|
| Load HF dataset | ~30s |
| GARCH fits (7 or 12 ETFs) | ~60s |
| Val lookback optimisation | ~3 min |
| Copula fit (AIC selection) | ~30s |
| MC simulation (5k samples) | ~20s |
| Backtest (test set) | ~2 min |
| **Total per module** | **~6–8 min** |

Both modules together run in approximately 15 minutes, well within the 6-hour free-tier limit.

---

## Streamlit UI

The app (`app.py`) shows two tabs (FI and Equity), each with:
- Hero pick box: ticker, conviction %, next NYSE trading day
- 2nd and 3rd best picks with conviction
- Macro environment pills (VIX, T10Y2Y, HY Spread, IG Spread, DXY)
- Copula family, lookback window, and regime label
- Full ETF score table with tail dependence
- OOS backtest metrics (annualised return, Sharpe, max drawdown, hit rate, alpha)
- Cumulative return chart (test set walk-forward)
- Signal history table (last 60 days)

---

## Key Papers

- Sklar, A. (1959). *Fonctions de répartition à n dimensions et leurs marges.*
- Joe, H. (1997). *Multivariate Models and Dependence Concepts.* Chapman & Hall.
- Patton, A.J. (2006). Modelling asymmetric exchange rate dependence. *International Economic Review.*
- Bollerslev, T. (1986). Generalised autoregressive conditional heteroskedasticity. *Journal of Econometrics.*

---

## Disclaimer

Research and educational purposes only. Not financial advice. Past performance does not guarantee future results. Use at your own risk.
