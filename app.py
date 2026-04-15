"""
P2-ETF-COPULA-ENGINE · app.py

Tabs:
  🏦 Fixed Income / Commodities   (original copula — unchanged)
  📈 Equity Sectors               (original copula — unchanged)
  🎯 Conformal — FI               (MC-based guaranteed intervals)
  🎯 Conformal — Equities         (MC-based guaranteed intervals)
"""

import json
import os
import traceback

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from huggingface_hub import hf_hub_download, list_repo_files

from config import HF_DATASET_OUT, FI_ETFS, EQUITY_ETFS

st.set_page_config(
    page_title="P2 ETF Copula Engine",
    page_icon="📐",
    layout="wide",
)

st.markdown("""
<style>
.hero-box {
  background: #ffffff; border: 1px solid #e0e6ef;
  border-left: 6px solid #1a6ef5; border-radius: 10px;
  padding: 1.4rem 1.8rem; margin-bottom: 0.8rem;
  box-shadow: 0 2px 8px rgba(0,0,0,0.07);
}
.hero-ticker   { font-size:3rem; font-weight:800; color:#1a6ef5; letter-spacing:3px; line-height:1.1; }
.hero-conviction { font-size:1.5rem; font-weight:700; color:#0a3d91; margin-top:0.1rem; }
.hero-row      { font-size:0.95rem; color:#4a5568; margin-top:0.45rem; }
.hero-row b    { color:#1a202c; }
.hero-bm       { font-size:0.82rem; color:#9aa5b4; margin-top:0.5rem; }
.pick-card     { background:#f7f9fc; border:1px solid #dde3ed; border-radius:8px;
                 padding:0.75rem 1rem; margin-bottom:0.5rem; }
.pick-rank     { font-size:0.72rem; color:#9aa5b4; text-transform:uppercase; letter-spacing:1px; }
.pick-ticker   { font-size:1.4rem; font-weight:700; color:#2d3748; }
.pick-pct      { font-size:0.9rem; color:#4a5568; }
.pill          { display:inline-block; background:#eef2fa; border:1px solid #c9d6f0;
                 border-radius:20px; padding:5px 14px; margin:3px 3px 3px 0; font-size:0.83rem; }
.pill-key      { color:#6b7a99; }
.pill-val      { color:#1a6ef5; font-weight:700; margin-left:4px; }
.metric-tile   { background:#f7f9fc; border:1px solid #e0e6ef; border-radius:8px;
                 padding:0.8rem 1rem; text-align:center; }
.metric-label  { font-size:0.75rem; color:#9aa5b4; text-transform:uppercase; letter-spacing:0.5px; }
.metric-value  { font-size:1.35rem; font-weight:700; color:#1a202c; margin-top:0.2rem; }
.metric-pos    { color:#16a34a; }
.metric-neg    { color:#dc2626; }
.badge-g  { background:#d1fae5; color:#065f46; border-radius:4px; padding:2px 8px;
            font-size:0.82rem; font-weight:600; }
.badge-r  { background:#fee2e2; color:#991b1b; border-radius:4px; padding:2px 8px;
            font-size:0.82rem; font-weight:600; }
.badge-y  { background:#fef3c7; color:#92400e; border-radius:4px; padding:2px 8px;
            font-size:0.82rem; font-weight:600; }
.badge-a  { background:#f3f4f6; color:#374151; border-radius:4px; padding:2px 8px;
            font-size:0.82rem; }
.section-hdr { font-weight:600; font-size:1rem; margin:1rem 0 0.3rem; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _hf_token():
    return os.environ.get("HF_TOKEN") or st.secrets.get("HF_TOKEN", None)


def _fmt_dt(s):
    try:
        from datetime import datetime
        return datetime.fromisoformat(s).strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return s or "—"


def _hex_to_rgba(hex_color, alpha=0.5):
    """Convert #RRGGBB to rgba(R,G,B,alpha)."""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 6:
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"
    return hex_color  # fallback


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner="Loading signals...")
def load_signal() -> dict:
    token  = _hf_token()
    errors = []
    for fname in ["results/copula_signal.json", "copula_signal.json"]:
        try:
            path = hf_hub_download(
                repo_id=HF_DATASET_OUT, filename=fname,
                repo_type="dataset", token=token, force_download=True,
            )
            with open(path) as f:
                return json.load(f)
        except Exception as e:
            errors.append(str(e))
    st.error("Could not load signal JSON:\n" + "\n".join(errors))
    return {}


@st.cache_data(ttl=300, show_spinner=False)
def load_conformal_signal() -> dict:
    token = _hf_token()
    for fname in ["conformal/copula_signal_conformal.json",
                  "results/copula_signal_conformal.json"]:
        try:
            path = hf_hub_download(
                repo_id=HF_DATASET_OUT, filename=fname,
                repo_type="dataset", token=token, force_download=True,
            )
            with open(path) as f:
                return json.load(f)
        except Exception:
            continue
    return {}


@st.cache_data(ttl=300, show_spinner=False)
def load_history(module: str) -> pd.DataFrame:
    fname = "signal_history_fi.csv" if module == "FI" else "signal_history_eq.csv"
    token = _hf_token()
    for subfolder in ["results/", ""]:
        try:
            path = hf_hub_download(
                repo_id=HF_DATASET_OUT, filename=f"{subfolder}{fname}",
                repo_type="dataset", token=token, force_download=True,
            )
            return pd.read_csv(path, parse_dates=["date"])
        except Exception:
            continue
    return pd.DataFrame()


@st.cache_data(ttl=300, show_spinner=False)
def load_conformal_history(module: str) -> pd.DataFrame:
    mod_lower = "fi" if module == "FI" else "eq"
    token = _hf_token()
    try:
        path = hf_hub_download(
            repo_id=HF_DATASET_OUT,
            filename=f"conformal/signal_history_conformal_{mod_lower}.json",
            repo_type="dataset", token=token, force_download=True,
        )
        with open(path) as f:
            return pd.DataFrame(json.load(f))
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300, show_spinner=False)
def load_metrics(module: str) -> dict:
    fname = "metrics_fi.json" if module == "FI" else "metrics_eq.json"
    token = _hf_token()
    for subfolder in ["results/", ""]:
        try:
            path = hf_hub_download(
                repo_id=HF_DATASET_OUT, filename=f"{subfolder}{fname}",
                repo_type="dataset", token=token, force_download=True,
            )
            with open(path) as f:
                return json.load(f)
        except Exception:
            continue
    return {}


# ═══════════════════════════════════════════════════════════════════════════════
# ORIGINAL COPULA TABS (FI and EQ) — completely unchanged
# ═══════════════════════════════════════════════════════════════════════════════

def render_hero(sig: dict, benchmark: str):
    col_hero, col_side = st.columns([5, 4], gap="large")
    with col_hero:
        exp_pct = sig.get("expected_return", 0) * 100
        st.markdown(f"""
<div class="hero-box">
  <div class="hero-ticker">{sig.get('pick', '—')}</div>
  <div class="hero-conviction">{sig.get('conviction_pct', 0):.1f}% conviction</div>
  <div class="hero-row"><b>Next trading day:</b> {sig.get('next_trading_day', '—')}</div>
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
        for rank, pk, ck in [("2nd pick", "second_pick", "second_conviction"),
                              ("3rd pick", "third_pick",  "third_conviction")]:
            ticker = sig.get(pk)
            conv   = sig.get(ck, 0) or 0
            if ticker:
                st.markdown(f"""
<div class="pick-card">
  <div class="pick-rank">{rank}</div>
  <div class="pick-ticker">{ticker}</div>
  <div class="pick-pct">{conv:.1f}% conviction</div>
</div>""", unsafe_allow_html=True)

        pills = sig.get("macro_pills", {})
        if pills:
            html = "".join([
                f'<span class="pill"><span class="pill-key">{k}</span>'
                f'<span class="pill-val">{v}</span></span>'
                for k, v in pills.items()
            ])
            st.markdown(html, unsafe_allow_html=True)


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
        ("Ann. Return",  f"{metrics.get('ann_return_pct', 0):.2f}%",
         metrics.get('ann_return_pct', 0) >= 0),
        ("Ann. Vol",     f"{metrics.get('ann_vol_pct', 0):.2f}%",      True),
        ("Sharpe",       f"{metrics.get('sharpe', 0):.3f}",
         metrics.get('sharpe', 0) >= 0),
        ("Max Drawdown", f"{metrics.get('max_drawdown_pct', 0):.2f}%",  False),
        ("Hit Rate",     f"{metrics.get('hit_rate_pct', 0):.1f}%",
         metrics.get('hit_rate_pct', 0) >= 50),
        ("Alpha vs BM",  f"{metrics.get('ann_alpha_pct', 0):.2f}%",
         metrics.get('ann_alpha_pct', 0) >= 0),
    ]
    cols = st.columns(6)
    for col, (label, value, positive) in zip(cols, fields):
        colour = "metric-pos" if positive else "metric-neg"
        with col:
            st.markdown(
                f'<div class="metric-tile"><div class="metric-label">{label}</div>'
                f'<div class="metric-value {colour}">{value}</div></div>',
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
        if st.button("List HF repo files"):
            try:
                files = list(list_repo_files(
                    HF_DATASET_OUT, repo_type="dataset", token=token
                ))
                st.write(files)
            except Exception as e:
                st.error(str(e))


def render_copula_tab(module: str, signal_data: dict, benchmark: str):
    sig  = signal_data.get(module, {})
    hist = load_history(module)
    met  = load_metrics(module)
    if not sig:
        st.warning(
            f"No {module} signal found. "
            "Check the sidebar debug panel for details."
        )
        return
    render_hero(sig, benchmark)
    st.markdown("---")
    st.subheader("All ETF Scores")
    render_scores(sig)
    st.markdown("---")
    st.subheader("OOS Backtest Metrics (test set)")
    render_metrics(met)
    st.markdown("---")
    st.subheader("Cumulative Return (test set walk-forward)")
    render_equity_curve(hist)
    st.markdown("---")
    st.subheader("Signal History")
    render_history_table(hist)


# ═══════════════════════════════════════════════════════════════════════════════
# CONFORMAL TABS (new) — genuinely different from copula tabs
# ═══════════════════════════════════════════════════════════════════════════════

def _etfs_for(module: str) -> list:
    return FI_ETFS if module == "FI" else EQUITY_ETFS


def render_conformal_hero(conf_sig: dict, module: str):
    """
    Hero card for conformal tab. Shows pick + signal class badge + q̂ info.
    """
    if not conf_sig or "pick" not in conf_sig:
        st.info(
            "Conformal signal not available yet. "
            "The workflow auto-calibrates on first run after training. "
            "Check that the conformal workflow ran in Actions."
        )
        return

    pick     = conf_sig["pick"]
    conv     = conf_sig.get("conviction_pct", 0)
    ntd      = conf_sig.get("next_trading_day", "—")
    gen      = _fmt_dt(conf_sig.get("generated_at", ""))
    n_cal    = conf_sig.get("n_cal", "?")
    cal_p    = conf_sig.get("cal_period", "?")
    cls      = conf_sig.get("signal_class", "")
    fam      = str(conf_sig.get("copula_family", "—")).capitalize()
    regime   = conf_sig.get("regime_name", "—")
    lb       = conf_sig.get("lookback_days", "—")

    iv90 = conf_sig.get("top_interval_90", {})
    lo90 = iv90.get("lo")
    hi90 = iv90.get("hi")
    q90  = iv90.get("q_hat")

    iv_str = (f"[{lo90:.4f}, {hi90:.4f}]"
              if lo90 is not None and hi90 is not None else "—")

    if "STRONG" in cls:
        badge = f'<span class="badge-g">{cls}</span>'
    elif "AVOID" in cls:
        badge = f'<span class="badge-r">{cls}</span>'
    elif "UNCERTAIN" in cls:
        badge = f'<span class="badge-y">{cls}</span>'
    else:
        badge = f'<span class="badge-a">{cls}</span>'

    q_str = f"q̂={q90:.4f}" if isinstance(q90, float) else ""

    # Macro pills
    pills_html = ""
    for k, v in conf_sig.get("macro_pills", {}).items():
        pills_html += (
            f'<span class="pill">'
            f'<span class="pill-key">{k}</span>'
            f'<span class="pill-val">{v}</span>'
            f'</span>'
        )

    # Ranked picks from conformal_per_etf
    cpe = conf_sig.get("conformal_per_etf", {})
    ranked = sorted(
        [(t, d.get("expected_return", 0)) for t, d in cpe.items()],
        key=lambda x: x[1], reverse=True,
    )
    runner = ""
    if len(ranked) > 1:
        runner += f"2nd: **{ranked[1][0]}** E[R]={ranked[1][1]:.4f}"
    if len(ranked) > 2:
        runner += f"&nbsp;&nbsp;3rd: **{ranked[2][0]}** E[R]={ranked[2][1]:.4f}"

    st.markdown(f"""
<div class="hero-box">
  <div class="hero-ticker">{pick}</div>
  <div class="hero-conviction">{conv:.1f}% conviction</div>
  <div class="hero-row">
    <b>Next trading day:</b> {ntd} &nbsp;·&nbsp; Generated {gen}
  </div>
  <div class="hero-row">
    <b>Copula:</b> {fam} &nbsp;·&nbsp;
    <b>Lookback:</b> {lb}d &nbsp;·&nbsp;
    <b>Regime:</b> {regime}
  </div>
  <div style="margin-top:0.5rem">
    {badge}&nbsp;
    <span class="badge-a">90% CI {iv_str}</span>&nbsp;
    <span class="badge-a">{q_str}</span>
  </div>
  <div style="margin-top:0.4rem">
    <span class="badge-a">cal n={n_cal} &nbsp;·&nbsp; {cal_p}</span>
  </div>
  <div style="margin-top:0.4rem">{runner}</div>
  <div style="margin-top:0.4rem">{pills_html}</div>
</div>
""", unsafe_allow_html=True)


def render_conformal_dot_chart(conf_sig: dict, module: str, alpha: str):
    """
    Dot-plot: E[R] as dot, conformal CI as horizontal bar.
    Colour encodes signal quality. Sorted by E[R] descending.
    Completely different from the copula bar chart.
    """
    cpe     = conf_sig.get("conformal_per_etf", {})
    tickers = _etfs_for(module)
    valid   = [t for t in tickers
               if t in cpe and alpha in cpe[t].get("intervals", {})]
    if not valid:
        st.info("No conformal interval data yet.")
        return

    valid   = sorted(valid, key=lambda t: cpe[t]["expected_return"], reverse=True)
    mus     = [cpe[t]["expected_return"] for t in valid]
    iv_lo   = [cpe[t]["intervals"][alpha]["lo"] for t in valid]
    iv_hi   = [cpe[t]["intervals"][alpha]["hi"] for t in valid]
    mc_pos  = [cpe[t].get("mc_pct_positive", 50) for t in valid]
    top     = conf_sig.get("pick", "")

    dot_colors  = []
    line_colors = []
    for t, lo, hi in zip(valid, iv_lo, iv_hi):
        if t == top:
            dot_colors.append("#1a6ef5")
            line_colors.append("rgba(26,110,245,0.4)")
        elif lo > 0:
            dot_colors.append("#16a34a")
            line_colors.append("rgba(22,163,74,0.35)")
        elif hi < 0:
            dot_colors.append("#dc2626")
            line_colors.append("rgba(220,38,38,0.35)")
        else:
            dot_colors.append("#9ca3af")
            line_colors.append("rgba(156,163,175,0.25)")

    fig = go.Figure()

    for i, t in enumerate(valid):
        fig.add_trace(go.Scatter(
            x=[iv_lo[i], iv_hi[i]], y=[t, t],
            mode="lines",
            line=dict(color=line_colors[i], width=7),
            showlegend=False, hoverinfo="skip",
        ))

    fig.add_trace(go.Scatter(
        x=mus, y=valid, mode="markers",
        marker=dict(color=dot_colors, size=11,
                    line=dict(width=1.5, color="white")),
        customdata=list(zip(
            iv_lo, iv_hi,
            [cpe[t]["intervals"][alpha]["q_hat"] for t in valid],
            mc_pos,
            [cpe[t].get("mc_pct_positive", 0) for t in valid],
        )),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "E[R] = %{x:.4f}<br>"
            f"CI lo = %{{customdata[0]:.4f}}<br>"
            f"CI hi = %{{customdata[1]:.4f}}<br>"
            "q̂ = %{customdata[2]:.4f}<br>"
            "MC P(R>0) = %{customdata[3]:.1f}%"
        ),
        showlegend=False,
    ))

    fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="#9ca3af")
    fig.update_layout(
        height=max(300, len(valid) * 36),
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="white", plot_bgcolor="white",
        xaxis=dict(
            title="Return  (dot = E[R],  bar = conformal CI)",
            showgrid=True, gridcolor="#f3f4f6",
        ),
        yaxis=dict(showgrid=False),
    )
    st.plotly_chart(fig, use_container_width=True,
                    config={"displayModeBar": False},
                    key=f"conf_dot_{module}_{alpha}")


def render_mc_distribution_chart(conf_sig: dict, module: str, alpha: str):
    """
    For each ETF, show the full MC distribution as a box-and-whisker
    alongside the conformal interval. This has NO equivalent in the copula tab.
    """
    cpe     = conf_sig.get("conformal_per_etf", {})
    tickers = _etfs_for(module)
    valid   = [t for t in tickers
               if t in cpe and alpha in cpe[t].get("intervals", {})]
    if not valid:
        return

    valid   = sorted(valid, key=lambda t: cpe[t]["expected_return"], reverse=True)
    top     = conf_sig.get("pick", "")

    # ---- FIX: use rgba() instead of hex+alpha for compatibility ----
    colors = ["#1a6ef5" if t == top else "#6b7280" for t in valid]
    # Convert hex to rgba with alpha 0.35 for the conformal CI bar
    rgba_colors = [_hex_to_rgba(c, alpha=0.35) for c in colors]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            "MC distribution (p10–p90) vs conformal CI",
            "P(return > 0) from MC simulation",
        ],
        horizontal_spacing=0.14,
    )

    for i, t in enumerate(valid):
        d   = cpe[t]
        ivs = d["intervals"][alpha]
        # MC box: p10–p90 as a grey line
        fig.add_trace(go.Scatter(
            x=[d["mc_p10"], d["mc_p90"]], y=[t, t],
            mode="lines",
            line=dict(color="#e5e7eb", width=5),
            showlegend=False, hoverinfo="skip",
        ), row=1, col=1)
        # Conformal CI overlay (now using rgba string)
        fig.add_trace(go.Scatter(
            x=[ivs["lo"], ivs["hi"]], y=[t, t],
            mode="lines",
            line=dict(color=rgba_colors[i], width=9),
            showlegend=False, hoverinfo="skip",
        ), row=1, col=1)
        # Median dot
        fig.add_trace(go.Scatter(
            x=[d["mc_median"]], y=[t],
            mode="markers",
            marker=dict(color=colors[i], size=8,
                        line=dict(width=1.5, color="white")),
            showlegend=False,
            hovertemplate=(
                f"<b>{t}</b><br>"
                f"Median = {d['mc_median']:.4f}<br>"
                f"p10={d['mc_p10']:.4f}  p90={d['mc_p90']:.4f}<br>"
                f"CI=[{ivs['lo']:.4f}, {ivs['hi']:.4f}]"
            ),
        ), row=1, col=1)

    # Right panel: P(R>0) bar chart
    mc_pos = [cpe[t].get("mc_pct_positive", 50) for t in valid]
    bar_colors = [
        "#16a34a" if p >= 60 else ("#dc2626" if p < 40 else "#9ca3af")
        for p in mc_pos
    ]
    bar_colors = ["#1a6ef5" if t == top else c
                  for t, c in zip(valid, bar_colors)]

    fig.add_trace(go.Bar(
        x=mc_pos, y=valid, orientation="h",
        marker_color=bar_colors,
        hovertemplate="<b>%{y}</b><br>P(R>0) = %{x:.1f}%",
        showlegend=False,
    ), row=1, col=2)
    fig.add_vline(x=50, line_width=1.5, line_dash="dash",
                  line_color="#9ca3af", row=1, col=2)

    fig.update_layout(
        height=max(320, len(valid) * 36),
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(size=11),
        showlegend=False,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#f3f4f6", row=1, col=1)
    fig.update_xaxes(showgrid=True, gridcolor="#f3f4f6",
                     title_text="P(return > 0)  [%]", row=1, col=2)
    fig.update_yaxes(showgrid=False)

    st.plotly_chart(fig, use_container_width=True,
                    config={"displayModeBar": False},
                    key=f"mc_box_{module}_{alpha}")


def render_full_conformal_table(conf_sig: dict, module: str, alpha: str):
    """Full per-ETF table with MC distribution stats + conformal intervals."""
    cpe     = conf_sig.get("conformal_per_etf", {})
    tickers = _etfs_for(module)
    valid   = [t for t in tickers if t in cpe]
    if not valid:
        return

    rows = []
    for t in valid:
        d   = cpe[t]
        ivs = d.get("intervals", {})
        rows.append({
            "ETF":             t,
            "E[R] MC":         round(d.get("expected_return", 0), 5),
            "MC std":          round(d.get("mc_std", 0), 5),
            "MC p10":          round(d.get("mc_p10", 0), 5),
            "MC median":       round(d.get("mc_median", 0), 5),
            "MC p90":          round(d.get("mc_p90", 0), 5),
            "P(R>0) %":        round(d.get("mc_pct_positive", 0), 1),
            f"CI lo ({int(float(alpha)*100)}%)":
                               round(ivs.get(alpha, {}).get("lo", 0), 5),
            f"CI hi ({int(float(alpha)*100)}%)":
                               round(ivs.get(alpha, {}).get("hi", 0), 5),
            f"CI width ({int(float(alpha)*100)}%)":
                               round(ivs.get(alpha, {}).get("width", 0), 5),
            "q̂":              round(ivs.get(alpha, {}).get("q_hat", 0), 4),
            "MC%ofCI":         round(ivs.get(alpha, {}).get("mc_hi_pct", 90)
                                     - ivs.get(alpha, {}).get("mc_lo_pct", 10), 1),
        })

    df = pd.DataFrame(rows).sort_values("E[R] MC", ascending=False)
    st.caption(
        "q̂ ∈ [0.5, 1.0] — the adjusted MC quantile level for guaranteed coverage. "
        "MC% of CI = fraction of MC mass used (e.g. 84% = 8th–92nd percentile). "
        "P(R>0) = raw copula-simulation probability of a positive return."
    )
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_coverage_diagnostics(conf_sig: dict):
    diag = (conf_sig or {}).get("coverage_diagnostics") or {}
    if not diag:
        return

    st.markdown('<div class="section-hdr">Calibration coverage diagnostics</div>',
                unsafe_allow_html=True)
    st.caption(
        "Empirical coverage on the val set. "
        "Achieved must be ≥ target by the conformal theorem."
    )
    rows = []
    for alpha_str in sorted(diag.keys(), reverse=True):
        info     = diag[alpha_str]
        target   = info.get("target", 1 - float(alpha_str))
        achieved = info.get("pooled", 0)
        ok       = achieved >= target - 0.005
        rows.append({
            "Level":   f"{int(float(alpha_str)*100)}%",
            "Target":  f"≥ {target:.0%}",
            "Achieved (pooled)": f"{achieved:.1%}",
            "Status":  "✓ pass" if ok else "✗ fail",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=False, hide_index=True)


def render_conformal_history(hist_df: pd.DataFrame):
    if hist_df.empty:
        st.info("Conformal signal history will appear after the first run.")
        return

    disp = hist_df.sort_values("signal_date", ascending=False).copy()

    if "interval_covered" in disp.columns:
        covered = disp["interval_covered"].dropna()
        if len(covered) > 0:
            cov_rate = covered.mean()
            badge    = "badge-g" if cov_rate >= 0.88 else "badge-r"
            st.markdown(
                f"<div style='margin-bottom:0.5rem'>"
                f"90% interval coverage: "
                f"<span class='{badge}'>{cov_rate:.1%}</span> "
                f"({int(covered.sum())}/{len(covered)} signals)</div>",
                unsafe_allow_html=True,
            )

    col_map = {
        "signal_date":       "Date",
        "pick":              "Pick",
        "conviction_pct":    "Conv.",
        "signal_class":      "Class",
        "interval_90_lo":    "CI lo",
        "interval_90_hi":    "CI hi",
        "interval_90_width": "Width",
        "q_hat_90":          "q̂",
        "copula_family":     "Copula",
        "regime_name":       "Regime",
        "actual_return":     "Actual R",
        "interval_covered":  "CI covered",
    }
    cols = [c for c in col_map if c in disp.columns]
    disp = disp[cols].rename(columns=col_map)

    for col in ["CI covered"]:
        if col in disp.columns:
            disp[col] = disp[col].apply(
                lambda x: "✓" if x is True else ("✗" if x is False else "—")
            )
    if "Actual R" in disp.columns:
        disp["Actual R"] = disp["Actual R"].apply(
            lambda x: f"{x*100:.2f}%" if pd.notna(x) else "—"
        )
    st.dataframe(disp, use_container_width=True, hide_index=True)


def render_conformal_tab(module: str, signal_data: dict,
                         conformal_data: dict, benchmark: str):
    conf_sig = conformal_data.get(module, {})

    render_conformal_hero(conf_sig, module)

    if not conf_sig or "conformal_per_etf" not in conf_sig:
        st.markdown("---")
        return

    st.markdown("<hr style='margin:1.5rem 0 0.5rem'>", unsafe_allow_html=True)

    alpha_choice = st.radio(
        "Coverage level",
        options=["0.9", "0.8", "0.7"],
        index=0, horizontal=True,
        key=f"alpha_{module}",
        format_func=lambda x: f"{int(float(x)*100)}%",
    )

    # ── 1. Signal classification dot-plot ─────────────────────────────────
    st.markdown(
        f'<div class="section-hdr">'
        f'Signal classification — conformal {int(float(alpha_choice)*100)}% intervals'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Dot = E[R] (MC mean). Bar = conformal CI at selected level. "
        "🔵 top pick &nbsp;·&nbsp; "
        "🟢 CI entirely positive &nbsp;·&nbsp; "
        "🔴 CI entirely negative (avoid) &nbsp;·&nbsp; "
        "⚫ CI crosses zero."
    )
    render_conformal_dot_chart(conf_sig, module, alpha_choice)

    # ── 2. MC distribution chart ───────────────────────────────────────────
    st.markdown(
        '<div class="section-hdr">'
        'MC distribution (p10–p90) with conformal CI overlay, and P(R>0)'
        '</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Grey bar = MC p10–p90 range. Coloured overlay = conformal CI. "
        "Right panel: MC-simulated probability of positive return. "
        "These come directly from the 5,000 copula samples — no Gaussian assumptions."
    )
    render_mc_distribution_chart(conf_sig, module, alpha_choice)

    # ── 3. Full table ──────────────────────────────────────────────────────
    st.markdown(
        '<div class="section-hdr">Full conformal table — MC stats + intervals</div>',
        unsafe_allow_html=True,
    )
    render_full_conformal_table(conf_sig, module, alpha_choice)

    # ── 4. Coverage diagnostics ────────────────────────────────────────────
    st.markdown("<hr style='margin:1.5rem 0 0.5rem'>", unsafe_allow_html=True)
    render_coverage_diagnostics(conf_sig)

    # ── 5. History ─────────────────────────────────────────────────────────
    st.markdown("<hr style='margin:1.5rem 0 0.5rem'>", unsafe_allow_html=True)
    st.markdown('<div class="section-hdr">Conformal signal history</div>',
                unsafe_allow_html=True)
    render_conformal_history(load_conformal_history(module))

    # ── Footnote ───────────────────────────────────────────────────────────
    st.markdown(
        f"<div style='font-size:0.8rem;color:#9aa5b4;margin-top:1rem'>"
        f"Calibration: {conf_sig.get('n_cal','?')} val-set days &nbsp;·&nbsp; "
        f"{conf_sig.get('cal_period','?')} &nbsp;·&nbsp; "
        f"Calibrated {_fmt_dt(conf_sig.get('calibrated_at',''))}"
        f"</div>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    st.title("📐 P2-ETF-COPULA-ENGINE")
    st.markdown(
        "*Copula-based joint tail dependency · Next-day ETF selection "
        "· Conformal prediction intervals*"
    )

    signal_data    = load_signal()
    conformal_data = load_conformal_signal()

    render_debug(signal_data)

    gen_at = signal_data.get("generated_at", "—")
    st.caption(f"Last generated: {gen_at} UTC · Source: {HF_DATASET_OUT}")

    col1, col2 = st.columns([8, 1])
    with col2:
        if st.button("🔄 Refresh"):
            st.cache_data.clear()
            st.rerun()

    tab_fi, tab_eq, tab_cfi, tab_ceq = st.tabs([
        "🏦 Fixed Income / Commodities",
        "📈 Equity Sectors",
        "🎯 Conformal — FI",
        "🎯 Conformal — Equities",
    ])

    with tab_fi:
        render_copula_tab("FI", signal_data, "AGG")

    with tab_eq:
        render_copula_tab("EQ", signal_data, "SPY")

    with tab_cfi:
        with st.expander("ℹ️ Why COPULA conformal is better than NCDE conformal",
                         expanded=False):
            st.markdown("""
**NCDE issue:** The NCDE outputs σ ≈ 1.0 (much larger than daily returns), making
the normalised nonconformity score `|y−μ|/σ` ≈ 0 and the conformal intervals trivially narrow.

**COPULA advantage:** The copula already generates **5,000 Monte Carlo return samples**
per ETF per day. These samples are directly in return space. The nonconformity score is:
s_i = max(F̂(y_i), 1 − F̂(y_i))
where `F̂(y_i)` is the empirical rank of the actual return `y_i` in the 5k MC samples.

At coverage level 1−α, the conformal quantile `q̂ ∈ [0.5, 1.0]` gives:
interval = [MC_quantile(1−q̂), MC_quantile(q̂)]


**What makes this meaningful:**
- Interval width varies per ETF and per regime (unlike NCDE absolute mode)
- Width is driven by the copula's joint simulation — tail-heavy copulas (Clayton/Student)
  naturally produce wider intervals for correlated ETFs in stress regimes
- `q̂` close to 0.5 = copula MC is well-centred on reality
- `q̂` close to 1.0 = copula is systematically missing the tails
""")
        render_conformal_tab("FI", signal_data, conformal_data, "AGG")

    with tab_ceq:
        render_conformal_tab("EQ", signal_data, conformal_data, "SPY")

    st.markdown("---")
    st.caption(
        "P2 Engine Suite · Copula Engine · Research only · Not financial advice"
    )


if __name__ == "__main__":
    main()
