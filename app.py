"""
P2-ETF-COPULA-ENGINE  ·  app.py
Streamlit dashboard — two tabs:
  Tab 1  Fixed Income / Commodities
  Tab 2  Equity Sectors

Reads copula_signal.json, signal_history_fi/eq.csv, and metrics_fi/eq.json
from the Hugging Face results dataset, then renders:
  • Hero pick box (ticker, conviction, next trading day)
  • 2nd / 3rd best picks
  • Macro pills (VIX, T10Y2Y, HY, IG, DXY)
  • Copula family + lookback + regime label
  • Tail dependence note
  • OOS backtest metrics table
  • Equity curve vs benchmark chart
  • Signal history table
"""

import json, os
import streamlit as st
import pandas as pd
import numpy as np
from datasets import load_dataset
from huggingface_hub import hf_hub_download

from config import HF_DATASET_OUT

st.set_page_config(
    page_title="P2 ETF Copula Engine",
    page_icon="📐",
    layout="wide",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.hero-box {
    background: #0e1117;
    border: 2px solid #00c2ff;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 0.6rem;
}
.hero-ticker {
    font-size: 3rem;
    font-weight: 700;
    color: #00c2ff;
    letter-spacing: 2px;
}
.hero-sub {
    font-size: 1rem;
    color: #7a8ba8;
    margin-top: 0.2rem;
}
.pill {
    display: inline-block;
    background: #1a2233;
    border: 1px solid #2a3a55;
    border-radius: 20px;
    padding: 4px 14px;
    margin: 3px;
    font-size: 0.82rem;
    color: #c8d8f0;
}
.pill-key  { color: #7a8ba8; }
.pill-val  { color: #f5a623; font-weight: 600; }
.metric-label { font-size: 0.78rem; color: #7a8ba8; }
.metric-value { font-size: 1.3rem; font-weight: 600; color: #e8edf5; }
</style>
""", unsafe_allow_html=True)


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=600)
def load_signal():
    try:
        token = os.environ.get("HF_TOKEN")
        path  = hf_hub_download(
            repo_id=HF_DATASET_OUT,
            filename="results/copula_signal.json",
            repo_type="dataset",
            token=token,
        )
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        st.warning(f"Could not load signal JSON: {e}")
        return {}


@st.cache_data(ttl=600)
def load_history(module: str) -> pd.DataFrame:
    fname = "signal_history_fi.csv" if module == "FI" else "signal_history_eq.csv"
    try:
        token = os.environ.get("HF_TOKEN")
        path  = hf_hub_download(
            repo_id=HF_DATASET_OUT,
            filename=f"results/{fname}",
            repo_type="dataset",
            token=token,
        )
        return pd.read_csv(path, parse_dates=["date"])
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=600)
def load_metrics(module: str) -> dict:
    fname = "metrics_fi.json" if module == "FI" else "metrics_eq.json"
    try:
        token = os.environ.get("HF_TOKEN")
        path  = hf_hub_download(
            repo_id=HF_DATASET_OUT,
            filename=f"results/{fname}",
            repo_type="dataset",
            token=token,
        )
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


# ── Rendering helpers ─────────────────────────────────────────────────────────

def render_hero(sig: dict, benchmark: str):
    col1, col2 = st.columns([2, 3])
    with col1:
        st.markdown(f"""
        <div class="hero-box">
          <div class="hero-ticker">{sig.get('pick','—')}</div>
          <div class="hero-sub">
            Conviction: <b>{sig.get('conviction_pct', 0):.1f}%</b> &nbsp;|&nbsp;
            Next trading day: <b>{sig.get('next_trading_day','—')}</b>
          </div>
          <div class="hero-sub" style="margin-top:0.5rem;">
            Copula: <b>{sig.get('copula_family','—').capitalize()}</b> &nbsp;|&nbsp;
            Lookback: <b>{sig.get('lookback_days','—')}d</b> &nbsp;|&nbsp;
            Regime: <b>{sig.get('regime_name','—')}</b>
          </div>
          <div class="hero-sub" style="margin-top:0.3rem;">
            Expected return: <b>{sig.get('expected_return', 0)*100:.3f}%</b> &nbsp;|&nbsp;
            Avg tail dep λ<sub>L</sub>: <b>{sig.get('avg_lower_tail_dep', 0):.3f}</b>
          </div>
          <div class="hero-sub" style="margin-top:0.4rem; color:#888;">
            Benchmark: {benchmark} (not traded)
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # 2nd and 3rd picks
        s2 = sig.get("second_pick")
        s3 = sig.get("third_pick")
        if s2:
            st.markdown(f"**2nd choice:** `{s2}` &nbsp; ({sig.get('second_conviction', 0):.1f}%)")
        if s3:
            st.markdown(f"**3rd choice:** `{s3}` &nbsp; ({sig.get('third_conviction', 0):.1f}%)")

        # Macro pills
        pills = sig.get("macro_pills", {})
        pill_html = "".join([
            f'<span class="pill"><span class="pill-key">{k}</span> '
            f'<span class="pill-val">{v}</span></span>'
            for k, v in pills.items()
        ])
        st.markdown(pill_html, unsafe_allow_html=True)


def render_scores(sig: dict):
    scores = sig.get("all_scores", [])
    if not scores:
        return
    df = pd.DataFrame(scores)
    df["expected_return"] = (df["expected_return"] * 100).round(4)
    df.columns = [c.replace("_", " ").title() for c in df.columns]
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_metrics(metrics: dict):
    if not metrics:
        st.info("Backtest metrics not yet available.")
        return

    cols = st.columns(6)
    fields = [
        ("Ann. Return",    f"{metrics.get('ann_return_pct', 0):.2f}%"),
        ("Ann. Vol",       f"{metrics.get('ann_vol_pct', 0):.2f}%"),
        ("Sharpe",         f"{metrics.get('sharpe', 0):.3f}"),
        ("Max Drawdown",   f"{metrics.get('max_drawdown_pct', 0):.2f}%"),
        ("Hit Rate",       f"{metrics.get('hit_rate_pct', 0):.1f}%"),
        ("Alpha vs BM",    f"{metrics.get('ann_alpha_pct', 0):.2f}%"),
    ]
    for col, (label, value) in zip(cols, fields):
        with col:
            st.markdown(f'<div class="metric-label">{label}</div>'
                        f'<div class="metric-value">{value}</div>',
                        unsafe_allow_html=True)


def render_equity_curve(history: pd.DataFrame):
    if history.empty or "net_return" not in history.columns:
        return
    hist = history.sort_values("date").copy()
    hist["cumulative_return"] = hist["net_return"].cumsum()
    st.line_chart(hist.set_index("date")["cumulative_return"],
                  use_container_width=True)


def render_history_table(history: pd.DataFrame):
    if history.empty:
        st.info("Signal history will populate after the first daily run.")
        return
    display_cols = ["date", "pick", "conviction_pct", "expected_return",
                    "actual_return", "net_return", "copula_family", "hit"]
    avail = [c for c in display_cols if c in history.columns]
    h = history.sort_values("date", ascending=False)[avail].head(60)
    st.dataframe(h, use_container_width=True, hide_index=True)


# ── Main layout ───────────────────────────────────────────────────────────────

def main():
    st.title("📐 P2-ETF-COPULA-ENGINE")
    st.markdown("*Copula-based joint tail dependency · Next-day ETF selection*")

    signal_data = load_signal()
    gen_at      = signal_data.get("generated_at", "—")
    st.caption(f"Last generated: {gen_at} UTC")

    tab_fi, tab_eq = st.tabs(["🏦 Fixed Income / Commodities", "📈 Equity Sectors"])

    # ── FI TAB ────────────────────────────────────────────────────
    with tab_fi:
        sig_fi   = signal_data.get("FI", {})
        hist_fi  = load_history("FI")
        met_fi   = load_metrics("FI")

        if not sig_fi:
            st.info("FI signal not yet generated. Run `train_fi.py` first.")
        else:
            render_hero(sig_fi, "AGG")

            st.markdown("---")
            st.subheader("All ETF Scores — FI")
            render_scores(sig_fi)

            st.markdown("---")
            st.subheader("OOS Backtest Metrics — FI (test set)")
            render_metrics(met_fi)

            st.markdown("---")
            st.subheader("Cumulative Return — FI (test set walk-forward)")
            render_equity_curve(hist_fi)

            st.markdown("---")
            st.subheader("Signal History — FI")
            render_history_table(hist_fi)

    # ── EQUITY TAB ────────────────────────────────────────────────
    with tab_eq:
        sig_eq  = signal_data.get("EQ", {})
        hist_eq = load_history("EQ")
        met_eq  = load_metrics("EQ")

        if not sig_eq:
            st.info("Equity signal not yet generated. Run `train_equity.py` first.")
        else:
            render_hero(sig_eq, "SPY")

            st.markdown("---")
            st.subheader("All ETF Scores — Equity")
            render_scores(sig_eq)

            st.markdown("---")
            st.subheader("OOS Backtest Metrics — Equity (test set)")
            render_metrics(met_eq)

            st.markdown("---")
            st.subheader("Cumulative Return — Equity (test set walk-forward)")
            render_equity_curve(hist_eq)

            st.markdown("---")
            st.subheader("Signal History — Equity")
            render_history_table(hist_eq)

    st.markdown("---")
    st.caption("P2 Engine Suite · Copula Engine · Research only · Not financial advice")


if __name__ == "__main__":
    main()
