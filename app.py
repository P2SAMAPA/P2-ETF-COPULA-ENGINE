"""
P2-ETF-COPULA-ENGINE  ·  app.py
Streamlit dashboard — two tabs: FI / Commodities and Equity Sectors.
"""

import json, os, traceback
import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files
from config import HF_DATASET_OUT

st.set_page_config(
    page_title="P2 ETF Copula Engine",
    page_icon="📐",
    layout="wide",
)

# ── CSS — light hero cards ────────────────────────────────────────────────────
st.markdown("""
<style>
/* Hero card — white/light with coloured left border */
.hero-box {
    background: #ffffff;
    border: 1px solid #e0e6ef;
    border-left: 6px solid #1a6ef5;
    border-radius: 10px;
    padding: 1.4rem 1.8rem;
    margin-bottom: 0.8rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.07);
}
.hero-ticker {
    font-size: 3rem;
    font-weight: 800;
    color: #1a6ef5;
    letter-spacing: 3px;
    line-height: 1.1;
}
.hero-conviction {
    font-size: 1.5rem;
    font-weight: 700;
    color: #0a3d91;
    margin-top: 0.1rem;
}
.hero-row {
    font-size: 0.95rem;
    color: #4a5568;
    margin-top: 0.45rem;
}
.hero-row b { color: #1a202c; }
.hero-bm {
    font-size: 0.82rem;
    color: #9aa5b4;
    margin-top: 0.5rem;
}

/* 2nd / 3rd pick cards */
.pick-card {
    background: #f7f9fc;
    border: 1px solid #dde3ed;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
}
.pick-rank   { font-size: 0.72rem; color: #9aa5b4; text-transform: uppercase; letter-spacing: 1px; }
.pick-ticker { font-size: 1.4rem; font-weight: 700; color: #2d3748; }
.pick-pct    { font-size: 0.9rem; color: #4a5568; }

/* Macro pills */
.pill {
    display: inline-block;
    background: #eef2fa;
    border: 1px solid #c9d6f0;
    border-radius: 20px;
    padding: 5px 14px;
    margin: 3px 3px 3px 0;
    font-size: 0.83rem;
}
.pill-key { color: #6b7a99; }
.pill-val { color: #1a6ef5; font-weight: 700; margin-left: 4px; }

/* Metric tiles */
.metric-tile {
    background: #f7f9fc;
    border: 1px solid #e0e6ef;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    text-align: center;
}
.metric-label { font-size: 0.75rem; color: #9aa5b4; text-transform: uppercase; letter-spacing: 0.5px; }
.metric-value { font-size: 1.35rem; font-weight: 700; color: #1a202c; margin-top: 0.2rem; }
.metric-pos   { color: #16a34a; }
.metric-neg   { color: #dc2626; }
</style>
""", unsafe_allow_html=True)


# ── Data loading ──────────────────────────────────────────────────────────────

def _hf_token():
    return os.environ.get("HF_TOKEN") or st.secrets.get("HF_TOKEN", None)


@st.cache_data(ttl=300, show_spinner="Loading signals from Hugging Face...")
def load_signal() -> dict:
    token = _hf_token()
    errors = []

    # Attempt 1: hf_hub_download
    try:
        path = hf_hub_download(
            repo_id=HF_DATASET_OUT,
            filename="results/copula_signal.json",
            repo_type="dataset",
            token=token,
            force_download=True,   # always get the freshest version
        )
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        errors.append(f"hf_hub_download failed: {e}")

    # Attempt 2: datasets library
    try:
        from huggingface_hub import hf_hub_download as dl
        path = dl(
            repo_id=HF_DATASET_OUT,
            filename="copula_signal.json",   # try without subfolder
            repo_type="dataset",
            token=token,
            force_download=True,
        )
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        errors.append(f"root-level download failed: {e}")

    st.error("Could not load signal JSON from HF. Errors:\n" + "\n".join(errors))
    return {}


@st.cache_data(ttl=300, show_spinner=False)
def load_history(module: str) -> pd.DataFrame:
    fname = "signal_history_fi.csv" if module == "FI" else "signal_history_eq.csv"
    token = _hf_token()
    for subfolder in ["results/", ""]:
        try:
            path = hf_hub_download(
                repo_id=HF_DATASET_OUT,
                filename=f"{subfolder}{fname}",
                repo_type="dataset",
                token=token,
                force_download=True,
            )
            return pd.read_csv(path, parse_dates=["date"])
        except Exception:
            continue
    return pd.DataFrame()


@st.cache_data(ttl=300, show_spinner=False)
def load_metrics(module: str) -> dict:
    fname = "metrics_fi.json" if module == "FI" else "metrics_eq.json"
    token = _hf_token()
    for subfolder in ["results/", ""]:
        try:
            path = hf_hub_download(
                repo_id=HF_DATASET_OUT,
                filename=f"{subfolder}{fname}",
                repo_type="dataset",
                token=token,
                force_download=True,
            )
            with open(path) as f:
                return json.load(f)
        except Exception:
            continue
    return {}


# ── Render helpers ────────────────────────────────────────────────────────────

def render_hero(sig: dict, benchmark: str):
    col_hero, col_side = st.columns([5, 4], gap="large")

    with col_hero:
        exp_pct = sig.get("expected_return", 0) * 100
        st.markdown(f"""
        <div class="hero-box">
          <div class="hero-ticker">{sig.get('pick', '—')}</div>
          <div class="hero-conviction">{sig.get('conviction_pct', 0):.1f}% conviction</div>
          <div class="hero-row">
            <b>Next trading day:</b> {sig.get('next_trading_day', '—')}
          </div>
          <div class="hero-row">
            <b>Copula:</b> {str(sig.get('copula_family', '—')).capitalize()} &nbsp;·&nbsp;
            <b>Lookback:</b> {sig.get('lookback_days', '—')}d &nbsp;·&nbsp;
            <b>Regime:</b> {sig.get('regime_name', '—')}
          </div>
          <div class="hero-row">
            <b>Expected return:</b> {exp_pct:.4f}% &nbsp;·&nbsp;
            <b>Avg λ<sub>L</sub>:</b> {sig.get('avg_lower_tail_dep', 0):.3f}
          </div>
          <div class="hero-bm">Benchmark: {benchmark} (not traded · no CASH output)</div>
        </div>
        """, unsafe_allow_html=True)

    with col_side:
        # 2nd and 3rd picks
        for rank, pick_key, conv_key in [
            ("2nd pick", "second_pick", "second_conviction"),
            ("3rd pick", "third_pick",  "third_conviction"),
        ]:
            ticker = sig.get(pick_key)
            conv   = sig.get(conv_key, 0) or 0
            if ticker:
                st.markdown(f"""
                <div class="pick-card">
                  <div class="pick-rank">{rank}</div>
                  <div class="pick-ticker">{ticker}</div>
                  <div class="pick-pct">{conv:.1f}% conviction</div>
                </div>
                """, unsafe_allow_html=True)

        # Macro pills
        pills = sig.get("macro_pills", {})
        if pills:
            pill_html = "".join([
                f'<span class="pill">'
                f'<span class="pill-key">{k}</span>'
                f'<span class="pill-val">{v}</span>'
                f'</span>'
                for k, v in pills.items()
            ])
            st.markdown(pill_html, unsafe_allow_html=True)


def render_scores(sig: dict):
    scores = sig.get("all_scores", [])
    if not scores:
        st.info("No score data available.")
        return
    df = pd.DataFrame(scores)
    if "expected_return" in df.columns:
        df["expected_return"] = (df["expected_return"] * 100).round(5)
    df.columns = [c.replace("_", " ").title() for c in df.columns]
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_metrics(metrics: dict):
    if not metrics:
        st.info("Backtest metrics not yet available.")
        return

    fields = [
        ("Ann. Return",  f"{metrics.get('ann_return_pct', 0):.2f}%",  metrics.get('ann_return_pct', 0) >= 0),
        ("Ann. Vol",     f"{metrics.get('ann_vol_pct', 0):.2f}%",     True),
        ("Sharpe",       f"{metrics.get('sharpe', 0):.3f}",           metrics.get('sharpe', 0) >= 0),
        ("Max Drawdown", f"{metrics.get('max_drawdown_pct', 0):.2f}%",False),
        ("Hit Rate",     f"{metrics.get('hit_rate_pct', 0):.1f}%",    metrics.get('hit_rate_pct', 0) >= 50),
        ("Alpha vs BM",  f"{metrics.get('ann_alpha_pct', 0):.2f}%",   metrics.get('ann_alpha_pct', 0) >= 0),
    ]
    cols = st.columns(6)
    for col, (label, value, positive) in zip(cols, fields):
        colour = "metric-pos" if positive else "metric-neg"
        with col:
            st.markdown(
                f'<div class="metric-tile">'
                f'<div class="metric-label">{label}</div>'
                f'<div class="metric-value {colour}">{value}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


def render_equity_curve(history: pd.DataFrame):
    if history.empty or "net_return" not in history.columns:
        return
    hist = history.sort_values("date").copy()
    hist["Strategy (cumulative log-return)"] = hist["net_return"].cumsum()
    st.line_chart(
        hist.set_index("date")[["Strategy (cumulative log-return)"]],
        use_container_width=True,
    )


def render_history_table(history: pd.DataFrame):
    if history.empty:
        st.info("Signal history will populate after the first daily run.")
        return
    want = ["date", "pick", "conviction_pct", "expected_return",
            "actual_return", "net_return", "copula_family", "hit"]
    cols = [c for c in want if c in history.columns]
    h = history.sort_values("date", ascending=False)[cols].head(60).copy()
    if "expected_return" in h.columns:
        h["expected_return"] = (h["expected_return"] * 100).round(5)
    st.dataframe(h, use_container_width=True, hide_index=True)


def render_debug(signal_data: dict):
    """Sidebar debug panel — shows raw keys and any loading issues."""
    with st.sidebar:
        st.markdown("### Debug")
        st.write("**Signal keys:**", list(signal_data.keys()))
        st.write("**FI present:**", "FI" in signal_data)
        st.write("**EQ present:**", "EQ" in signal_data)
        if "FI" in signal_data:
            st.write("**FI pick:**", signal_data["FI"].get("pick"))
        st.write("**Generated at:**", signal_data.get("generated_at", "—"))
        token = _hf_token()
        st.write("**HF_TOKEN set:**", token is not None)

        # List files in HF repo
        if st.button("List HF repo files"):
            try:
                files = list(list_repo_files(HF_DATASET_OUT, repo_type="dataset", token=token))
                st.write(files)
            except Exception as e:
                st.error(str(e))


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    st.title("📐 P2-ETF-COPULA-ENGINE")
    st.markdown("*Copula-based joint tail dependency · Next-day ETF selection*")

    signal_data = load_signal()
    render_debug(signal_data)

    gen_at = signal_data.get("generated_at", "—")
    st.caption(f"Last generated: {gen_at} UTC  ·  Source: {HF_DATASET_OUT}")

    tab_fi, tab_eq = st.tabs(["🏦 Fixed Income / Commodities", "📈 Equity Sectors"])

    # ── FI TAB ────────────────────────────────────────────────────
    with tab_fi:
        sig_fi  = signal_data.get("FI", {})
        hist_fi = load_history("FI")
        met_fi  = load_metrics("FI")

        if not sig_fi:
            st.warning(
                "No FI signal found in the results JSON. "
                "Check the sidebar debug panel for details. "
                "The file may be uploading to a different path — "
                "click **List HF repo files** in the sidebar to verify."
            )
        else:
            render_hero(sig_fi, "AGG")
            st.markdown("---")
            st.subheader("All ETF Scores")
            render_scores(sig_fi)
            st.markdown("---")
            st.subheader("OOS Backtest Metrics (test set)")
            render_metrics(met_fi)
            st.markdown("---")
            st.subheader("Cumulative Return (test set walk-forward)")
            render_equity_curve(hist_fi)
            st.markdown("---")
            st.subheader("Signal History")
            render_history_table(hist_fi)

    # ── EQUITY TAB ────────────────────────────────────────────────
    with tab_eq:
        sig_eq  = signal_data.get("EQ", {})
        hist_eq = load_history("EQ")
        met_eq  = load_metrics("EQ")

        if not sig_eq:
            st.warning(
                "No Equity signal found. Run `train_equity.py` or check the sidebar debug panel."
            )
        else:
            render_hero(sig_eq, "SPY")
            st.markdown("---")
            st.subheader("All ETF Scores")
            render_scores(sig_eq)
            st.markdown("---")
            st.subheader("OOS Backtest Metrics (test set)")
            render_metrics(met_eq)
            st.markdown("---")
            st.subheader("Cumulative Return (test set walk-forward)")
            render_equity_curve(hist_eq)
            st.markdown("---")
            st.subheader("Signal History")
            render_history_table(hist_eq)

    st.markdown("---")
    st.caption("P2 Engine Suite · Copula Engine · Research only · Not financial advice")


if __name__ == "__main__":
    main()
