"""
app.py
======
PrakritiNetra — Interactive forecast blending dashboard.

Launch with:
    streamlit run app.py

Architecture
------------
* Backend: full Python pipeline (load → regime → score → weight → blend)
  via Streamlit session-state caching.
* Maps / heatmaps: Plotly (best geospatial support).
* RMSE bar charts & skill comparisons: Recharts (React) embedded via
  streamlit.components.v1.html, styled with Tailwind CSS.
* All non-map UI elements use Tailwind utility classes injected once via
  a custom CSS component for consistent light-theme styling.

UI structure
------------
  Header + tagline
  ─── Sidebar ────────────────────────────────────────────────
    Variable selector        (labelled in plain English)
    Lead-time selector       (plain English labels)
    Date/time selector
    Regime method + n_regimes
  ─── Main area ──────────────────────────────────────────────
    Row 1: Detected regime badge + season + info
    Row 2: Truth map | Blended forecast map | Extreme-flag map
    Row 3: Per-source forecast maps (3 maps)
    Row 4: Weight map (spatial heatmap) + caption
    Row 5: RMSE bar chart (Recharts) + skill improvement table
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import traceback

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from logo_loader import get_logo_html

# ─────────────────────────────────────────────────────────────
# Streamlit page config (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PrakritiNetra — Smarter Forecasts | Healthier Tomorrow",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# Dark-mode colour tokens (permanent — light mode removed)
# ─────────────────────────────────────────────────────────────
_BG          = "#0f1117"
_SURFACE     = "#1a1f2e"
_SURFACE2    = "rgba(255,255,255,0.04)"
_BORDER      = "rgba(255,255,255,0.10)"
_TEXT        = "#e2e8f0"
_TEXT_MUTED  = "#94a3b8"
_TEXT_FAINT  = "#475569"
_LABEL       = "#64748b"
_HL          = "#60a5fa"
_DIVIDER     = "rgba(255,255,255,0.08)"
_DIVIDER2    = "rgba(255,255,255,0.10)"
_BADGE_CONV  = "rgba(251,191,36,0.2)"
_BADGE_STRA  = "rgba(96,165,250,0.2)"
_BADGE_CLEAR = "rgba(52,211,153,0.2)"
_BADGE_MONS  = "rgba(167,139,250,0.2)"
_BADGE_WINT  = "rgba(56,189,248,0.2)"
_MAP_PAPER   = "#1a1f2e"
_MAP_PLOT    = "#1a1f2e"
_MAP_TEXT_C  = "#e2e8f0"
_MAP_SUBTEXT = "#94a3b8"
_MAP_GRID    = "#2d3748"
_CONF_WRAP   = "rgba(255,255,255,0.06)"
_FB_CARD_BG  = "rgba(99,102,241,0.08)"
_FB_CARD_BR  = "rgba(99,102,241,0.25)"
_SECT_ACT_BG = "rgba(59,130,212,0.08)"
_SECT_ACT_BR = "#3b82d4"

# ─────────────────────────────────────────────────────────────
# Global CSS injection — theme-aware
# ─────────────────────────────────────────────────────────────
TAILWIND_CSS = f"""
<style>
  /* ── Layout ── */
  .block-container {{ padding-top: 1rem !important; max-width: 100% !important; }}
  footer {{ visibility: hidden; }}
  #MainMenu {{ visibility: hidden; }}

  /* ── Theme surface ── */
  .stApp {{ background-color: {_BG} !important; }}
  section[data-testid="stSidebar"] {{ background-color: {_SURFACE} !important; }}
  .stApp, .stApp * {{ color: {_TEXT}; }}

  /* ── Header ── */
  .fb-header-title {{
    font-size: 1.75rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    margin: 0 0 4px 0;
    line-height: 1.2;
  }}
  .fb-header-sub {{
    font-size: 0.9rem;
    color: {_TEXT_MUTED};
    margin: 0;
    font-weight: 400;
  }}
  .fb-header-sub span.hl {{ color: {_HL}; font-weight: 500; }}
  .fb-header-bar {{
    border-bottom: 2px solid {_DIVIDER2};
    padding-bottom: 0.9rem;
    margin-bottom: 1rem;
  }}

  /* ── Cards ── */
  .fb-card {{
    background: {_SURFACE2};
    border: 1px solid {_BORDER};
    border-radius: 0.75rem;
    padding: 1.25rem 1.5rem;
    margin-bottom: 0.5rem;
  }}
  .fb-card-label {{
    font-size: 0.68rem;
    font-weight: 700;
    color: {_LABEL};
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-bottom: 4px;
  }}
  .fb-card-value {{
    font-size: 1.05rem;
    font-weight: 600;
    color: {_TEXT};
  }}

  /* ── Section headings ── */
  .fb-section-head {{
    font-size: 0.78rem;
    font-weight: 700;
    color: {_LABEL};
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 0.75rem 0 0.3rem;
  }}

  /* ── Badges ── */
  .fb-badge {{
    display: inline-block;
    padding: 0.2rem 0.65rem;
    border-radius: 9999px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.03em;
  }}
  .badge-convective  {{ background:{_BADGE_CONV};  color:#d97706; }}
  .badge-stratiform  {{ background:{_BADGE_STRA};  color:#2563eb; }}
  .badge-clear       {{ background:{_BADGE_CLEAR}; color:#059669; }}
  .badge-monsoon     {{ background:{_BADGE_MONS};  color:#7c3aed; }}
  .badge-winter      {{ background:{_BADGE_WINT};  color:#0ea5e9; }}

  /* ── Footer ── */
  .fb-footer {{
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid {_DIVIDER};
    text-align: center;
    font-size: 0.72rem;
    color: {_TEXT_FAINT};
  }}

  /* ── Sector cards ── */
  .fb-sector-card {{
    background: {_SURFACE2};
    border: 1px solid {_BORDER};
    border-radius: 0.65rem;
    padding: 0.85rem 1rem;
    margin-bottom: 0.4rem;
  }}
  .fb-sector-active {{
    border-color: {_SECT_ACT_BR};
    background: {_SECT_ACT_BG};
  }}
  .fb-sector-icon {{ font-size: 1.4rem; }}
  .fb-sector-name {{ font-weight: 700; color: {_TEXT}; font-size: 0.92rem; }}
  .fb-sector-focus {{ font-size: 0.72rem; color: {_TEXT_MUTED}; margin-top: 2px; }}

  /* ── Confidence bars ── */
  .fb-conf-bar-wrap {{
    background: {_CONF_WRAP};
    border-radius: 9999px;
    height: 8px;
    width: 100%;
    overflow: hidden;
    margin: 4px 0 2px;
  }}
  .fb-conf-bar-fill {{
    height: 100%;
    border-radius: 9999px;
  }}
  .conf-high  {{ background: #22c55e; }}
  .conf-mid   {{ background: #f59e0b; }}
  .conf-low   {{ background: #ef4444; }}

  /* ── Alert severity badges ── */
  .badge-watch   {{ background: rgba(250,204,21,0.15);  color: #b45309; }}
  .badge-warning {{ background: rgba(249,115,22,0.20);  color: #c2410c; }}
  .badge-severe  {{ background: rgba(239,68,68,0.25);   color: #b91c1c; }}

  /* ── Feedback card ── */
  .fb-feedback-card {{
    background: {_FB_CARD_BG};
    border: 1px solid {_FB_CARD_BR};
    border-radius: 0.65rem;
    padding: 0.9rem 1.1rem;
    font-family: monospace;
    font-size: 0.8rem;
    color: {_TEXT};
    white-space: pre-wrap;
    line-height: 1.55;
  }}

  /* ── Section dividers ── */
  .fb-new-section {{
    margin-top: 1.5rem;
    padding-top: 1rem;
    border-top: 1px solid {_DIVIDER};
  }}
</style>
"""

st.markdown(TAILWIND_CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Pipeline imports & caching
# ─────────────────────────────────────────────────────────────
from data.real_loader import load_dataset, LEAD_TIMES, VARIABLES, SOURCES
from regime import fit_and_classify, extract_features
from scoring import compute_skill_scores, build_region_map, region_id_to_label, REGION_TILES
from weighting import build_weight_table, get_weight_map, get_weights
from blending import blend_forecast, flag_extremes, compute_blend_skill_all_leads, EXTREME_THRESHOLDS


@st.cache_resource(show_spinner="Loading and processing data…")
def load_pipeline(method: str = "kmeans", n_regimes: int = 3):
    """Cache the full pipeline; re-runs only when params change."""
    ds = load_dataset(
        gfs_dir=str(_HERE / "datasets" / "gfs"),
        gefs_dir=str(_HERE / "datasets" / "gefs"),
        era5_dir=str(_HERE / "datasets" / "era5"),
        pangu_dir=str(_HERE / "datasets" / "pangu"),
    )
    clf, regimes = fit_and_classify(ds, n_regimes=n_regimes, method=method)
    skill_df = compute_skill_scores(ds, regimes)
    wt = build_weight_table(skill_df)
    return ds, clf, regimes, skill_df, wt


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
LEAD_LABELS = {6: "6 h (nowcast)", 24: "24 h (day-1)",
               72: "72 h (day-3)", 168: "168 h (day-7)"}
VAR_LABELS  = {"rainfall": "Rainfall (mm)", "temperature": "Temperature (°C)",
               "wind": "Wind Speed (m/s)"}
VAR_UNITS   = {"rainfall": "mm", "temperature": "°C", "wind": "m/s"}
SOURCE_COLORS = {"nwp": "#3b82f6", "ensemble": "#10b981", "ai": "#f59e0b",
                 "blended": "#6366f1", "naive": "#6b7280"}
REGIME_INFO = {
    "convective": "Active convection — heavy rainfall, gusty winds.",
    "stratiform": "Widespread stratiform precipitation — moderate, steady.",
    "clear":      "Dry/clear conditions — little to no rainfall.",
}

# Map tokens — driven by theme selection
_MAP_PAPER   = _MAP_PAPER   # set in theme block above
_MAP_PLOT    = _MAP_PLOT
_MAP_TEXT    = _MAP_TEXT_C
_MAP_SUBTEXT = _MAP_SUBTEXT
_MAP_GRID    = _MAP_GRID

def _map_axis(title_text: str) -> dict:
    """Reusable dark-theme axis config."""
    return dict(
        title=dict(text=title_text, font=dict(color=_MAP_TEXT, size=11)),
        tickfont=dict(color=_MAP_TEXT, size=10),
        showgrid=False,
        linecolor=_MAP_GRID,
        zerolinecolor=_MAP_GRID,
    )

def _map_colorbar(title_text: str, thickness: int = 12, **kwargs) -> dict:
    """Reusable dark-theme colorbar config."""
    return dict(
        title=dict(text=title_text, font=dict(color=_MAP_TEXT, size=11)),
        tickfont=dict(color=_MAP_TEXT, size=10),
        thickness=thickness,
        len=0.8,
        **kwargs,
    )


def _plotly_heatmap(
    data: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    title: str,
    colorscale: str = "RdYlBu_r",
    zmin=None, zmax=None,
    unit: str = "",
) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=data,
        x=lons,
        y=lats,
        colorscale=colorscale,
        zmin=zmin,
        zmax=zmax,
        colorbar=_map_colorbar(unit),
        hovertemplate="Lat: %{y:.1f}°<br>Lon: %{x:.1f}°<br>Value: %{z:.2f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color=_MAP_TEXT, family="system-ui")),
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor=_MAP_PAPER,
        plot_bgcolor=_MAP_PLOT,
        font=dict(color=_MAP_TEXT),
        xaxis=_map_axis("Longitude"),
        yaxis=_map_axis("Latitude"),
        height=280,
    )
    return fig


def _plotly_bool_map(
    data: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    title: str,
    event_name: str,
) -> go.Figure:
    z = data.astype(float)
    fig = go.Figure(go.Heatmap(
        z=z,
        x=lons,
        y=lats,
        colorscale=[[0, "#1e3a2f"], [1, "#ef4444"]],
        zmin=0, zmax=1,
        showscale=True,
        colorbar=_map_colorbar("Flagged", tickvals=[0, 1], ticktext=["No", "Yes"]),
        hovertemplate="Lat: %{y:.1f}°<br>Lon: %{x:.1f}°<br>Flagged: %{z}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(
            text=f"{title} — {event_name}",
            font=dict(size=13, color=_MAP_TEXT, family="system-ui"),
        ),
        margin=dict(l=10, r=10, t=45, b=10),
        paper_bgcolor=_MAP_PAPER,
        plot_bgcolor=_MAP_PLOT,
        font=dict(color=_MAP_TEXT),
        xaxis=_map_axis("Longitude"),
        yaxis=_map_axis("Latitude"),
        height=280,
    )
    return fig


def _plotly_weight_map(
    weight_map: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    source: str,
    variable: str,
    lead_label: str,
) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=weight_map,
        x=lons,
        y=lats,
        colorscale="Viridis",
        zmin=0, zmax=1,
        colorbar=_map_colorbar("Weight"),
        hovertemplate="Lat: %{y:.1f}°<br>Lon: %{x:.1f}°<br>Weight: %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(
            text=f"{source.upper()} weight — {VAR_LABELS[variable]}, {lead_label}",
            font=dict(size=13, color=_MAP_TEXT, family="system-ui"),
        ),
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor=_MAP_PAPER,
        plot_bgcolor=_MAP_PLOT,
        font=dict(color=_MAP_TEXT),
        xaxis=_map_axis("Longitude"),
        yaxis=_map_axis("Latitude"),
        height=260,
    )
    return fig


def _plotly_rmse_bar(skill_data: list[dict], variable: str) -> go.Figure:
    """Grouped bar chart of RMSE by source for each lead time — native Plotly."""
    if not skill_data:
        return go.Figure()
    sources = [k for k in skill_data[0].keys() if k != "lead_time"]
    lead_labels = [f"{row['lead_time']}h" for row in skill_data]
    fig = go.Figure()
    for src in sources:
        vals = [row.get(src, None) for row in skill_data]
        fig.add_trace(go.Bar(
            name=src,
            x=lead_labels,
            y=vals,
            marker_color=SOURCE_COLORS.get(src, "#94a3b8"),
            text=[f"{v:.3f}" if v is not None else "" for v in vals],
            textposition="outside",
        ))
    fig.update_layout(
        title=dict(
            text=f"RMSE by Lead Time — {variable.title()}",
            font=dict(size=14, color="#e2e8f0"),
            x=0, xanchor="left",
        ),
        barmode="group",
        xaxis=dict(title="Lead time", tickfont=dict(size=11)),
        yaxis=dict(title="RMSE", tickfont=dict(size=11)),
        # Legend placed BELOW the chart — no overlap with title
        legend=dict(
            orientation="h",
            yanchor="top", y=-0.18,
            xanchor="center", x=0.5,
            font=dict(size=11),
        ),
        margin=dict(l=55, r=20, t=45, b=90),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=400,
    )
    return fig


def _plotly_region_skill(region_data: list[dict], variable: str, lead_time: int) -> go.Figure:
    """Line chart of RMSE across regions per source — native Plotly."""
    if not region_data:
        return go.Figure()
    sources = [k for k in region_data[0].keys() if k != "region"]
    regions = [row["region"] for row in region_data]
    fig = go.Figure()
    for src in sources:
        vals = [row.get(src, None) for row in region_data]
        fig.add_trace(go.Scatter(
            name=src,
            x=regions,
            y=vals,
            mode="lines+markers",
            line=dict(
                color=SOURCE_COLORS.get(src, "#94a3b8"),
                width=3 if src == "blended" else 2 if src == "naive" else 1.5,
            ),
            marker=dict(size=5),
        ))
    fig.update_layout(
        title=dict(
            text=f"RMSE by Region — {variable.title()} @ {lead_time}h",
            font=dict(size=14, color="#e2e8f0"),
            x=0, xanchor="left",
        ),
        xaxis=dict(title="Region", tickfont=dict(size=10)),
        yaxis=dict(title="RMSE", tickfont=dict(size=11)),
        # Legend placed BELOW the chart
        legend=dict(
            orientation="h",
            yanchor="top", y=-0.22,
            xanchor="center", x=0.5,
            font=dict(size=11),
        ),
        margin=dict(l=55, r=20, t=45, b=100),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=380,
    )
    return fig


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    _sidebar_logo = get_logo_html(height_px=48)
    st.markdown(
        f'<div class="fb-card" style="margin-bottom:1rem;">'
        f'<div style="display:flex; align-items:flex-start; gap:10px;">'
        f'{_sidebar_logo}'
        f'<div>'
        f'<div style="font-size:1.25rem; font-weight:800; color:{_TEXT}; letter-spacing:-0.02em; line-height:1.25;">'
        f'<span style="color:#22c55e;">Prakriti</span><span style="color:#3b82f6;">Netra</span>'
        f'</div>'
        f'<div style="font-size:0.72rem; color:{_TEXT_MUTED}; margin-top:2px;">Smarter Forecasts | Healthier Tomorrow</div>'
        f'</div></div></div>',
        unsafe_allow_html=True
    )

    st.markdown("**Controls**")
    sel_var = st.selectbox(
        "Variable",
        options=VARIABLES,
        format_func=lambda v: VAR_LABELS[v],
        help="The weather field to display and score.",
    )
    sel_lead = st.selectbox(
        "Forecast lead time",
        options=LEAD_TIMES,
        format_func=lambda l: LEAD_LABELS[l],
        index=1,
        help="How far ahead the forecast is valid. Shorter = higher accuracy.",
    )

    st.markdown("---")
    st.markdown("**Model settings**")
    method  = st.radio("Regime clustering method",
                       ["kmeans", "gmm"], index=0,
                       help="k-Means is faster; GMM can model softer boundaries.")
    n_regimes = st.slider("Number of regimes", 2, 5, 3,
                          help="Unsupervised clusters inferred from the data.")

    st.markdown("---")
    st.markdown(
        f"<div style='font-size:0.75rem; color:{_TEXT_FAINT};'>"
        "Data source: <b>Real GFS (NOAA NOMADS)</b> — India domain (8–37°N, 68–97°E). "
        "Files loaded from <code>datasets/gfs/</code>."
        "</div>", unsafe_allow_html=True
    )

# ─────────────────────────────────────────────────────────────
# Load pipeline
# ─────────────────────────────────────────────────────────────
with st.spinner("Running pipeline…"):
    ds, clf, regimes, skill_df, wt = load_pipeline(
        method=method, n_regimes=n_regimes
    )

lats  = ds.coords["lat"].values
lons  = ds.coords["lon"].values
times = pd.DatetimeIndex(ds.coords["time"].values)

# Time selector (after pipeline loads)
with st.sidebar:
    sel_t = st.select_slider(
        "Date / time step",
        options=list(range(len(times))),
        value=len(times) // 2,
        format_func=lambda i: times[i].strftime("%Y-%m-%d %H:%M"),
        help="Select a specific forecast issue time to visualise.",
    )

sel_time  = times[sel_t]
sel_season = str(ds.coords["season"].values[sel_t])
sel_regime = str(regimes.iloc[sel_t])

# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
_header_logo = get_logo_html(height_px=48)
st.markdown(
    f'<div class="fb-header-bar" style="overflow:visible;">'
    f'<div style="display:flex; align-items:center; gap:16px; flex-wrap:nowrap; overflow:visible;">'
    f'{_header_logo}'
    f'<div style="min-width:0; flex:1;">'
    f'<h1 class="fb-header-title" style="-webkit-text-fill-color:unset; background:none; margin:0; font-size:2rem; line-height:1.15; white-space:nowrap;">'
    f'<span style="color:#22c55e; font-weight:900;">Prakriti</span>'
    f'<span style="color:#3b82f6; font-weight:900;">Netra</span>'
    f'</h1>'
    f'<p class="fb-header-sub" style="margin:3px 0 0;">'
    f'Smarter Forecasts &nbsp;|&nbsp; Healthier Tomorrow &nbsp;•&nbsp; '
    f'<span class="hl">Rainfall, Temperature &amp; Wind</span>'
    f'&nbsp;•&nbsp; India domain &nbsp;•&nbsp; Hybrid AI‑NWP Adaptive Blending'
    f'</p>'
    f'</div></div></div>',
    unsafe_allow_html=True
)

# ─────────────────────────────────────────────────────────────
# ROW 1 — Context badges
# ─────────────────────────────────────────────────────────────
regime_class = f"badge-{sel_regime.lower()}" if sel_regime.lower() in ("convective","stratiform","clear") else "badge-clear"
season_class = f"badge-{sel_season.lower()}" if sel_season.lower() in ("monsoon","winter") else ""
regime_desc  = REGIME_INFO.get(sel_regime, "")

st.markdown(f"""
<div class="fb-card" style="display:flex; align-items:flex-start; gap:2rem; flex-wrap:wrap;">
  <div>
    <div class="fb-card-label">Selected date</div>
    <div class="fb-card-value">{sel_time.strftime("%d %b %Y, %H:%M UTC")}</div>
  </div>
  <div>
    <div class="fb-card-label">
      Detected weather regime
      <span title="Inferred by unsupervised k-Means/GMM — not hand-labelled." style="cursor:help;"> ℹ️</span>
    </div>
    <span class="fb-badge {regime_class}">{sel_regime.title()}</span>
    <span style="font-size:0.78rem; color:{_TEXT_MUTED}; margin-left:8px;">{regime_desc}</span>
  </div>
  <div>
    <div class="fb-card-label">Season</div>
    <span class="fb-badge {season_class}">{sel_season.title()}</span>
  </div>
  <div>
    <div class="fb-card-label">Variable / Lead time</div>
    <div class="fb-card-value">{VAR_LABELS[sel_var]} &nbsp;·&nbsp; {LEAD_LABELS[sel_lead]}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Compute blend for selected time/variable/lead
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _get_blend(_ds_id, variable, lead_time, method, n_regimes):
    """Cache blending computation per (variable, lead_time, pipeline params)."""
    return blend_forecast(ds, wt, regimes, variable, lead_time)


blend_result = _get_blend(id(ds), sel_var, sel_lead, method, n_regimes)
truth_t    = blend_result["truth"][sel_t]
blended_t  = blend_result["blended"][sel_t]
naive_t    = blend_result["naive"][sel_t]
unit       = VAR_UNITS[sel_var]

# Extreme flags
extreme_thresholds = EXTREME_THRESHOLDS.get(sel_var, {})
flags = flag_extremes(blended_t[None], sel_var)   # add batch dim
flag_t = {k: v[0] for k, v in flags.items()}       # remove batch dim

# ─────────────────────────────────────────────────────────────
# ROW 2 — Truth | Blended | Extreme flags
# ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="font-size:0.78rem; font-weight:700; color:{_LABEL};
            text-transform:uppercase; letter-spacing:0.08em; margin:0.5rem 0 0.25rem;">
  Forecast Maps
</div>
""", unsafe_allow_html=True)

col_t, col_b, col_e = st.columns(3)
zmin = float(min(truth_t.min(), blended_t.min()))
zmax = float(max(truth_t.max(), blended_t.max()))

with col_t:
    st.plotly_chart(
        _plotly_heatmap(truth_t, lats, lons,
                        "📍 Observed (Truth)",
                        colorscale="Blues" if sel_var == "rainfall" else "RdYlBu_r",
                        zmin=zmin, zmax=zmax, unit=unit),
        use_container_width=True, config={"displayModeBar": False}
    )
    st.caption("The 'ground truth' field (stand-in for ERA5/observations).")

with col_b:
    st.plotly_chart(
        _plotly_heatmap(blended_t, lats, lons,
                        "🔀 Adaptive Blended Forecast",
                        colorscale="Blues" if sel_var == "rainfall" else "RdYlBu_r",
                        zmin=zmin, zmax=zmax, unit=unit),
        use_container_width=True, config={"displayModeBar": False}
    )
    st.caption("Spatially adaptive blend — each grid cell weighted by regional skill.")

with col_e:
    if flag_t:
        evt_name, evt_arr = next(iter(flag_t.items()))
        st.plotly_chart(
            _plotly_bool_map(evt_arr, lats, lons,
                             "⚠️ Extreme Weather Flag", evt_name.replace("_"," ").title()),
            use_container_width=True, config={"displayModeBar": False}
        )
        frac = evt_arr.mean()
        st.caption(f"**{evt_name.replace('_',' ').title()}** threshold "
                   f"≥ {extreme_thresholds.get(evt_name, '?')} {unit}. "
                   f"{frac*100:.1f}% of cells flagged.")
    else:
        st.info("No extreme threshold defined for this variable.")

# ─────────────────────────────────────────────────────────────
# ROW 3 — Per-source forecast maps
# ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="font-size:0.78rem; font-weight:700; color:{_LABEL};
            text-transform:uppercase; letter-spacing:0.08em; margin:0.75rem 0 0.25rem;">
  Individual Model Forecasts
</div>
""", unsafe_allow_html=True)

src_cols = st.columns(len(SOURCES))
src_arrays = {
    src: blend_result[src][sel_t]
    for src in SOURCES if src in blend_result
}
src_desc = {
    "nwp":      "Physical NWP — accurate short-term, degrades fast.",
    "ensemble": "Ensemble — smooth, underestimates extremes.",
    "ai":       "AI/ML model — stable at long lead times.",
}
for i, (src, col) in enumerate(zip(SOURCES, src_cols)):
    icon = '🏗️' if src == 'nwp' else '🧩' if src == 'ensemble' else '🤖'
    arr = src_arrays.get(src, None)
    with col:
        if arr is None or np.all(np.isnan(arr)):
            st.markdown(
                f"<div style='border:1px solid #e5e7eb; border-radius:0.5rem; "
                f"padding:2rem 1rem; text-align:center; color:#57606a; "
                f"background:#f7f8fa; height:280px; display:flex; "
                f"flex-direction:column; justify-content:center;'>"
                f"<div style='font-size:2rem;'>{icon}</div>"
                f"<div style='font-weight:600; margin-top:0.5rem;'>{src.upper()}</div>"
                f"<div style='font-size:0.78rem; margin-top:0.4rem;'>No data available<br>"
                f"<span style='color:#9ca3af;'>Add {src.upper()} files to enable</span></div>"
                f"</div>",
                unsafe_allow_html=True
            )
        else:
            st.plotly_chart(
                _plotly_heatmap(arr, lats, lons,
                                f"{icon} {src.upper()}",
                                colorscale="Blues" if sel_var == "rainfall" else "RdYlBu_r",
                                zmin=zmin, zmax=zmax, unit=unit),
                use_container_width=True, config={"displayModeBar": False}
            )
        st.caption(src_desc.get(src, src))

# ─────────────────────────────────────────────────────────────
# ROW 4 — Weight maps
# ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="font-size:0.78rem; font-weight:700; color:{_LABEL};
            text-transform:uppercase; letter-spacing:0.08em; margin:0.75rem 0 0.25rem;">
  Model Weight Maps
  <span title="Shows how much each model is trusted in each region for this variable, lead time, season and weather regime. Spatial variation means different regions rely on different models." style="cursor:help;font-size:0.75rem;"> ℹ️</span>
</div>
<div style="font-size:0.78rem; color:#57606a; margin-bottom:0.5rem;">
  This map shows which forecast source is trusted most in each region right now.
  Weights vary spatially — brighter cells trust that model more.
</div>
""", unsafe_allow_html=True)

wmap_cols = st.columns(len(SOURCES))
for src, col in zip(SOURCES, wmap_cols):
    wmap = get_weight_map(wt, ds, sel_var, sel_lead, sel_season, sel_regime, source=src)
    with col:
        st.plotly_chart(
            _plotly_weight_map(wmap, lats, lons, src, sel_var, LEAD_LABELS[sel_lead]),
            use_container_width=True, config={"displayModeBar": False}
        )

# Dominant source map
dom_map_numeric = get_weight_map(wt, ds, sel_var, sel_lead, sel_season, sel_regime)
source_sorted   = sorted(SOURCES)
colorscale_dom  = [
    [0.0, SOURCE_COLORS.get(source_sorted[0], "#3b82f6")],
    [0.5, SOURCE_COLORS.get(source_sorted[1], "#10b981")],
    [1.0, SOURCE_COLORS.get(source_sorted[2], "#f59e0b")],
]
fig_dom = go.Figure(go.Heatmap(
    z=dom_map_numeric,
    x=lons, y=lats,
    colorscale=colorscale_dom,
    zmin=0, zmax=len(SOURCES) - 1,
    colorbar=_map_colorbar(
        "Source",
        tickvals=list(range(len(source_sorted))),
        ticktext=[s.upper() for s in source_sorted],
        thickness=14,
    ),
    hovertemplate="Lat: %{y:.1f}°<br>Lon: %{x:.1f}°<br>Source: %{z}<extra></extra>",
))
fig_dom.update_layout(
    title=dict(
        text="Dominant Source Map — most-trusted model per region",
        font=dict(size=13, color=_MAP_TEXT, family="system-ui"),
    ),
    margin=dict(l=10, r=10, t=40, b=10),
    paper_bgcolor=_MAP_PAPER,
    plot_bgcolor=_MAP_PLOT,
    font=dict(color=_MAP_TEXT),
    xaxis=_map_axis("Longitude"),
    yaxis=_map_axis("Latitude"),
    height=300,
)
st.plotly_chart(fig_dom, use_container_width=True,
                config={"displayModeBar": False})
st.caption("Each region is coloured by the model with the highest weight for the "
           "current variable/lead/season/regime. Spatial variation confirms "
           "region is a real weighting dimension.")

# ─────────────────────────────────────────────────────────────
# ROW 5 — RMSE bar chart + regional skill lines
# ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="font-size:0.78rem; font-weight:700; color:{_LABEL};
            text-transform:uppercase; letter-spacing:0.08em; margin:0.75rem 0 0.25rem;">
  Forecast Skill Comparison
  <span title="RMSE = Root Mean Square Error. Lower is better. The adaptive blend should outperform both individual models and the naive average (equal weights)." style="cursor:help;font-size:0.75rem;"> ℹ️</span>
</div>
""", unsafe_allow_html=True)

# Compute skill across all lead times for bar chart
@st.cache_data(show_spinner=False)
def _compute_all_skill(_ds_hash, variable, method, n_regimes):
    return compute_blend_skill_all_leads(ds, wt, regimes, variables=[variable])

all_skill = _compute_all_skill(id(ds), sel_var, method, n_regimes)

# Prepare bar-chart data: mean RMSE per (lead_time, source)
bar_data = []
for lt in LEAD_TIMES:
    row = {"lead_time": lt}
    grp = all_skill[(all_skill["variable"] == sel_var) & (all_skill["lead_time"] == lt)]
    for src in grp["source"].unique():
        row[src] = round(float(grp[grp["source"] == src]["rmse"].mean()), 4)
    bar_data.append(row)

col_chart, col_region = st.columns([3, 2])

with col_chart:
    st.plotly_chart(
        _plotly_rmse_bar(bar_data, sel_var),
        use_container_width=True, config={"displayModeBar": False}
    )

with col_region:
    region_grp = all_skill[
        (all_skill["variable"] == sel_var) & (all_skill["lead_time"] == sel_lead)
    ]
    region_pivot = region_grp.groupby(["region", "source"])["rmse"] \
                              .mean().unstack("source").reset_index()
    region_data = region_pivot.to_dict(orient="records")
    st.plotly_chart(
        _plotly_region_skill(region_data, sel_var, sel_lead),
        use_container_width=True, config={"displayModeBar": False}
    )

# ─────────────────────────────────────────────────────────────
# Skill improvement table
# ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="font-size:0.78rem; font-weight:700; color:{_LABEL};
            text-transform:uppercase; letter-spacing:0.08em; margin:0.75rem 0 0.25rem;">
  Skill Improvement Table
  <span title="Shows whether the adaptive blend beats the naive equal-weight average. ⚠️ flags honest under-performance — not hidden." style="cursor:help;font-size:0.75rem;"> ℹ️</span>
</div>
""", unsafe_allow_html=True)

# Pivot: blended RMSE vs naive RMSE per (variable, lead_time)
summary = []
for (var, lt), grp in all_skill[all_skill["variable"] == sel_var].groupby(["variable", "lead_time"]):
    b_rmse = grp[grp["source"] == "blended"]["rmse"].mean()
    n_rmse = grp[grp["source"] == "naive"]["rmse"].mean()
    src_rmses = {src: grp[grp["source"] == src]["rmse"].mean()
                 for src in SOURCES if src in grp["source"].values}
    best_single = min(src_rmses.values()) if src_rmses else float("nan")
    beats_naive  = bool(b_rmse <= n_rmse)
    beats_best   = bool(b_rmse <= best_single)
    summary.append({
        "Variable":          VAR_LABELS[var],
        "Lead time":         LEAD_LABELS[int(lt)],
        "Blended RMSE":      round(b_rmse, 4),
        "Naive RMSE":        round(n_rmse, 4),
        "Best single RMSE":  round(best_single, 4),
        "Beats naive?":      "✅ Yes" if beats_naive else "⚠️ No",
        "Beats best model?": "✅ Yes" if beats_best  else "⚠️ No",
    })

summary_df = pd.DataFrame(summary)

# Highlight rows where blend loses to naive
def _highlight_row(row):
    if "⚠️" in row["Beats naive?"]:
        # Dark-mode compatible: deep red tint with light text
        return [
            "background-color: #3b1010; color: #f87171"
        ] * len(row)
    return ["background-color: transparent; color: #e2e8f0"] * len(row)

st.dataframe(
    summary_df.style.apply(_highlight_row, axis=1),
    use_container_width=True,
    hide_index=True,
)
st.caption(
    "🟥 Red rows = buckets where the adaptive blend underperforms naive equal-weighting. "
    "This is an honest signal, not hidden. Likely diagnosis: regime/season imbalance "
    "causes noisy weight estimates for rare combinations."
)

# All new-feature CSS is now consolidated in TAILWIND_CSS above (theme-aware).

# ─────────────────────────────────────────────────────────────
# ── EXTREME WEATHER INDICATORS (dedicated section, all 3 hazards)
# Shows heavy rainfall + heatwave + gale flags simultaneously,
# independent of the sidebar variable selector.
# ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="fb-new-section">
  <div style="font-size:0.78rem; font-weight:700; color:{_LABEL};
              text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.3rem;">
    ⚡ Extreme Weather Indicators
    <span title="All three hazards — Heavy Rainfall, Heatwave and Gale — shown simultaneously for the selected date and lead time, independent of the sidebar variable selector." style="cursor:help;font-size:0.75rem;"> ℹ️</span>
  </div>
  <div style="font-size:0.75rem; color:#57606a; margin-bottom:0.6rem;">
    Binary flag maps (red = threshold exceeded) for all three extreme-weather types at the
    selected date &amp; lead time. Thresholds: Heavy Rainfall ≥50 mm · Heatwave ≥40 °C · Gales ≥17.2 m/s.
  </div>
</div>
""", unsafe_allow_html=True)

# Pre-compute blended fields for all three variables (cached, so effectively free
# after the first call — the blending cache keyed by variable is shared with Row 2)
@st.cache_data(show_spinner=False)
def _get_all_blends(_ds_id, lead_time, _method, _n_regimes):
    out = {}
    for _v in ["rainfall", "temperature", "wind"]:
        out[_v] = blend_forecast(ds, wt, regimes, _v, lead_time)
    return out

_all_blends = _get_all_blends(id(ds), sel_lead, method, n_regimes)

# Summary tiles: one per hazard
_EW_META = {
    "rainfall":    {"hazard": "heavy_rainfall",  "label": "Heavy Rainfall",
                    "icon": "🌧️", "unit": "mm",  "color": "#3b82f6",
                    "thresh_color": "rgba(59,130,246,0.15)", "thresh_border": "#3b82f6"},
    "temperature": {"hazard": "heat_wave",        "label": "Heatwave",
                    "icon": "🌡️", "unit": "°C",  "color": "#ef4444",
                    "thresh_color": "rgba(239,68,68,0.15)",  "thresh_border": "#ef4444"},
    "wind":        {"hazard": "high_wind",        "label": "Gale-force Wind",
                    "icon": "💨", "unit": "m/s", "color": "#10b981",
                    "thresh_color": "rgba(16,185,129,0.15)", "thresh_border": "#10b981"},
}

_ew_tile_cols = st.columns(3)
_ew_map_cols  = st.columns(3)

for _col_idx, (_vname, _meta) in enumerate(_EW_META.items()):
    _br   = _all_blends[_vname]
    _bt   = _br["blended"][sel_t]      # (n_lat, n_lon) for selected time step
    _thr  = EXTREME_THRESHOLDS.get(_vname, {}).get(_meta["hazard"], None)
    _flgs = flag_extremes(_bt[None], _vname)
    _farr = _flgs.get(_meta["hazard"], np.zeros_like(_bt, dtype=bool))[0]
    _frac = float(_farr.mean()) * 100.0
    _vmax = float(np.nanmax(_bt)) if not np.all(np.isnan(_bt)) else 0.0
    _vmean= float(np.nanmean(_bt)) if not np.all(np.isnan(_bt)) else 0.0

    # Determine status
    if _thr is None:
        _status_txt = "No threshold"
        _status_col = _TEXT_MUTED
        _status_icon = "⬜"
    elif _frac > 0:
        _status_txt = f"{_frac:.1f}% cells flagged"
        _status_col = "#ef4444"
        _status_icon = "🔴"
    elif _vmax >= 0.7 * (_thr or 1):
        _status_txt = "Approaching threshold"
        _status_col = "#f59e0b"
        _status_icon = "🟡"
    else:
        _status_txt = "No event"
        _status_col = "#22c55e"
        _status_icon = "🟢"

    # Summary tile
    with _ew_tile_cols[_col_idx]:
        st.markdown(
            f'<div style="background:{_meta["thresh_color"]}; border:1px solid {_meta["thresh_border"]}; '
            f'border-radius:0.65rem; padding:0.8rem 1rem; margin-bottom:0.4rem;">'
            f'<div style="font-size:1.4rem; line-height:1;">{_meta["icon"]}</div>'
            f'<div style="font-size:0.88rem; font-weight:700; color:{_TEXT}; margin-top:4px;">'
            f'{_meta["label"]}</div>'
            f'<div style="font-size:0.72rem; color:{_TEXT_MUTED}; margin-top:2px;">'
            f'Threshold: ≥{_thr} {_meta["unit"]}</div>'
            f'<div style="font-size:0.82rem; font-weight:600; color:{_status_col}; margin-top:6px;">'
            f'{_status_icon} {_status_txt}</div>'
            f'<div style="font-size:0.72rem; color:{_TEXT_MUTED}; margin-top:3px;">'
            f'Domain max: <b style="color:{_TEXT}">{_vmax:.1f} {_meta["unit"]}</b> &nbsp;·&nbsp; '
            f'Mean: <b style="color:{_TEXT}">{_vmean:.1f} {_meta["unit"]}</b></div>'
            f'</div>',
            unsafe_allow_html=True
        )

    # Spatial flag map
    with _ew_map_cols[_col_idx]:
        if np.all(np.isnan(_bt)):
            st.info(f"{_meta['icon']} {_meta['label']}: no blended data available for this time step.")
        else:
            # Overlay: show actual values with threshold contour highlighted
            _zmin_ew = float(np.nanmin(_bt))
            _zmax_ew = float(np.nanmax(_bt))
            _cs_ew = (
                "Blues"   if _vname == "rainfall"    else
                "RdYlBu_r" if _vname == "temperature" else
                "YlOrRd"
            )
            _fig_ew = go.Figure()
            # Base: continuous value heatmap (shows magnitude)
            _fig_ew.add_trace(go.Heatmap(
                z=_bt,
                x=lons, y=lats,
                colorscale=_cs_ew,
                zmin=_zmin_ew, zmax=_zmax_ew,
                colorbar=_map_colorbar(_meta["unit"], thickness=10),
                hovertemplate=(
                    f"Lat: %{{y:.1f}}°<br>Lon: %{{x:.1f}}°<br>"
                    f"{_meta['label']}: %{{z:.2f}} {_meta['unit']}<extra></extra>"
                ),
                name="Value",
            ))
            # Overlay: flag cells in vivid red with opacity
            if _frac > 0:
                _flag_overlay = np.where(_farr, _vmax, np.nan)
                _fig_ew.add_trace(go.Heatmap(
                    z=_flag_overlay,
                    x=lons, y=lats,
                    colorscale=[[0, "rgba(239,68,68,0)"], [1, "rgba(239,68,68,0.65)"]],
                    zmin=_zmin_ew, zmax=_zmax_ew,
                    showscale=False,
                    hovertemplate=f"⚠️ THRESHOLD EXCEEDED<extra></extra>",
                    name="Flagged",
                ))
            _thr_display = f"≥{_thr} {_meta['unit']}" if _thr else ""
            _fig_ew.update_layout(
                title=dict(
                    text=f"{_meta['icon']} {_meta['label']} {_thr_display}",
                    font=dict(size=12, color=_MAP_TEXT, family="system-ui"),
                ),
                margin=dict(l=10, r=10, t=40, b=10),
                paper_bgcolor=_MAP_PAPER,
                plot_bgcolor=_MAP_PLOT,
                font=dict(color=_MAP_TEXT),
                xaxis=_map_axis("Lon"),
                yaxis=_map_axis("Lat"),
                height=270,
                showlegend=False,
            )
            st.plotly_chart(_fig_ew, use_container_width=True,
                            config={"displayModeBar": False})

# Cross-variable extreme summary table
_ew_summary_rows = []
for _vname, _meta in _EW_META.items():
    _br   = _all_blends[_vname]
    _bt   = _br["blended"][sel_t]
    _thr  = EXTREME_THRESHOLDS.get(_vname, {}).get(_meta["hazard"], None)
    _flgs = flag_extremes(_bt[None], _vname)
    _farr = _flgs.get(_meta["hazard"], np.zeros_like(_bt, dtype=bool))[0]
    _frac = float(_farr.mean()) * 100.0
    _vmax = float(np.nanmax(_bt)) if not np.all(np.isnan(_bt)) else float("nan")
    _vmin = float(np.nanmin(_bt)) if not np.all(np.isnan(_bt)) else float("nan")
    _vmean= float(np.nanmean(_bt)) if not np.all(np.isnan(_bt)) else float("nan")
    _flagged_n = int(_farr.sum())
    _status = (
        "🔴 EXCEEDED" if _frac > 0 else
        "🟡 APPROACHING" if (not np.isnan(_vmax) and _thr and _vmax >= 0.7 * _thr) else
        "🟢 NORMAL"
    )
    _ew_summary_rows.append({
        "Hazard":         f"{_meta['icon']} {_meta['label']}",
        "Threshold":      f"≥{_thr} {_meta['unit']}" if _thr else "N/A",
        "Domain Max":     f"{_vmax:.1f} {_meta['unit']}",
        "Domain Mean":    f"{_vmean:.1f} {_meta['unit']}",
        "Cells Flagged":  f"{_flagged_n} / 256 ({_frac:.1f}%)",
        "Status":         _status,
    })

_ew_df = pd.DataFrame(_ew_summary_rows)
st.markdown(f"<div style='font-size:0.78rem; font-weight:700; color:{_LABEL}; "
            f"text-transform:uppercase; letter-spacing:0.08em; margin:0.5rem 0 0.2rem;'>"
            f"Summary — all three hazards at {sel_time.strftime('%d %b %Y')} · {LEAD_LABELS[sel_lead]}"
            f"</div>", unsafe_allow_html=True)
st.dataframe(_ew_df, use_container_width=True, hide_index=True)
st.caption(
    "Blended forecast values from all three sources (NWP + Ensemble + AI where available). "
    "Red overlay on maps marks grid cells where the threshold is exceeded. "
    f"Lead time: {LEAD_LABELS[sel_lead]}. "
    "⚠️ Note: domain-mean values are low on this Sep 2026 dataset — "
    "thresholds will be triggered by real extreme events in production data."
)

# ─────────────────────────────────────────────────────────────
# Import new feature modules (additive — no existing modules changed)
# ─────────────────────────────────────────────────────────────
try:
    from sector_profiles import (
        SECTORS, SECTOR_METADATA, SECTOR_VAR_WEIGHTS,
        sector_forecast_view, compute_sector_weights,
    )
    _SECTOR_OK = True
except Exception as _e:
    _SECTOR_OK = False
    _SECTOR_ERR = traceback.format_exc()

try:
    from weight_map_enhanced import (
        build_enhanced_weight_map, build_region_type_map,
        compute_uncertainty_map,
    )
    _WMAP_OK = True
except Exception as _e:
    _WMAP_OK = False
    _WMAP_ERR = traceback.format_exc()

try:
    from feedback_loop import (
        run_feedback_update, build_demo_record, compute_weight_shift,
        find_best_demo_time_idx,
    )
    _FEEDBACK_OK = True
except Exception as _e:
    _FEEDBACK_OK = False
    _FEEDBACK_ERR = traceback.format_exc()

try:
    from confidence_score import (
        compute_spatial_confidence, compute_confidence_readout,
    )
    _CONF_OK = True
except Exception as _e:
    _CONF_OK = False
    _CONF_ERR = traceback.format_exc()

try:
    from alerts import (
        generate_alerts, alerts_to_dataframe, save_alert_history,
        load_alert_history, alert_summary_html, SEVERITY_ICONS,
        build_district_map,
    )
    _ALERTS_OK = True
except Exception as _e:
    _ALERTS_OK = False
    _ALERTS_ERR = traceback.format_exc()

from weighting import get_all_source_weight_maps


# ─────────────────────────────────────────────────────────────
# SOURCE ARRAYS helper
# ─────────────────────────────────────────────────────────────
def _get_source_arrays(blend_result: dict) -> dict:
    """Return only per-source forecast arrays (exclude truth/blended/naive)."""
    return {
        s: v for s, v in blend_result.items()
        if s not in ("truth", "blended", "naive")
        and not np.all(np.isnan(v))
    }


# ─────────────────────────────────────────────────────────────
# ── FEATURE 1 — SECTOR-ADAPTIVE FORECAST BLENDING ────────────
# ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="fb-new-section">
  <div style="font-size:0.78rem; font-weight:700; color:{_LABEL};
              text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.3rem;">
    🏭 Sector-Adaptive Forecast Blending
    <span title="Each sector emphasises different variables and source preferences on top of the existing inverse-error weights." style="cursor:help;font-size:0.75rem;"> ℹ️</span>
  </div>
  <div style="font-size:0.75rem; color:#57606a; margin-bottom:0.6rem;">
    Select a sector to view a forecast framing and re-weighting optimised for that domain.
    The underlying blended values are unchanged — only the emphasis and explanation shift.
  </div>
</div>
""", unsafe_allow_html=True)

if _SECTOR_OK:
    sel_sector = st.selectbox(
        "Sector",
        options=SECTORS,
        format_func=lambda s: f"{SECTOR_METADATA[s]['icon']} {s}",
        key="sel_sector",
        help="Switch sector to see domain-specific weighting and explanations.",
    )

    # Sector cards row
    s_cols = st.columns(len(SECTORS))
    for sc, col in zip(SECTORS, s_cols):
        meta = SECTOR_METADATA[sc]
        active_cls = "fb-sector-active" if sc == sel_sector else ""
        with col:
            st.markdown(
                f'<div class="fb-sector-card {active_cls}">'
                f'<div class="fb-sector-icon">{meta["icon"]}</div>'
                f'<div class="fb-sector-name">{sc}</div>'
                f'<div class="fb-sector-focus">{meta["focus"]}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

    # Compute sector forecast
    source_arrays = _get_source_arrays(blend_result)
    existing_weights_spatial = get_all_source_weight_maps(
        wt, ds, sel_var, sel_lead, sel_season, sel_regime
    )

    sector_result = sector_forecast_view(
        blend_result=blend_result,
        existing_weights_spatial=existing_weights_spatial,
        sector=sel_sector,
        variable=sel_var,
        skill_df=skill_df,
        lead_time=sel_lead,
        season=sel_season,
        regime=sel_regime,
    )

    sec_col_map, sec_col_explain = st.columns([3, 2])
    with sec_col_map:
        sector_t = sector_result["sector_blended"][sel_t]
        fig_sec = _plotly_heatmap(
            sector_t, lats, lons,
            f"{SECTOR_METADATA[sel_sector]['icon']} {sel_sector} Sector Forecast",
            colorscale="Blues" if sel_var == "rainfall" else "RdYlBu_r",
            zmin=float(sector_t.min()), zmax=float(sector_t.max()),
            unit=unit,
        )
        st.plotly_chart(fig_sec, use_container_width=True,
                        config={"displayModeBar": False})
        st.caption(
            f"Sector-adjusted forecast for {sel_sector}. "
            f"Variable importance for this sector: "
            + ", ".join(
                f"{v}: {w*100:.0f}%"
                for v, w in SECTOR_VAR_WEIGHTS[sel_sector].items()
            )
        )

    with sec_col_explain:
        st.markdown("**Sector Weight Explanation**")
        pct = sector_result["sector_pct"]

        # Determine which sources are applicable vs not applicable for this variable
        _all_sources = ["nwp", "ensemble", "ai"]
        _SOURCE_FRIENDLY = {"nwp": "NWP", "ensemble": "Ensemble", "ai": "AI"}
        _na_sources = [s for s in _all_sources if s not in pct]

        # Applicable sources — weight bars sorted by weight descending
        for src, p in sorted(pct.items(), key=lambda x: -x[1]):
            conf_cls = "conf-high" if p >= 50 else "conf-mid" if p >= 25 else "conf-low"
            bar_w    = max(int(p), 3)
            st.markdown(
                f'<div style="margin:3px 0;">'
                f'<div style="font-size:0.78rem; color:{_TEXT}; '
                f'display:flex; justify-content:space-between;">'
                f'<span style="font-weight:600;">{_SOURCE_FRIENDLY.get(src, src.upper())}</span>'
                f'<span>{p:.0f}%</span></div>'
                f'<div class="fb-conf-bar-wrap">'
                f'<div class="fb-conf-bar-fill {conf_cls}" style="width:{bar_w}%;"></div>'
                f'</div></div>',
                unsafe_allow_html=True
            )

        # Not-applicable sources — shown explicitly with reason, never silently omitted
        for src in _na_sources:
            _fname = _SOURCE_FRIENDLY.get(src, src.upper())
            st.markdown(
                f'<div style="margin:3px 0; opacity:0.55;">'
                f'<div style="font-size:0.78rem; color:{_TEXT_MUTED}; '
                f'display:flex; justify-content:space-between;">'
                f'<span style="font-weight:600;">{_fname}</span>'
                f'<span style="font-style:italic;">N/A</span></div>'
                f'<div class="fb-conf-bar-wrap">'
                f'<div style="height:100%; width:100%; background:repeating-linear-gradient('
                f'90deg, rgba(148,163,184,0.15) 0px, rgba(148,163,184,0.15) 4px, '
                f'transparent 4px, transparent 8px); border-radius:9999px;"></div>'
                f'</div>'
                f'<div style="font-size:0.68rem; color:{_TEXT_MUTED}; margin-top:1px;">'
                f'No {sel_var} data available from this source</div>'
                f'</div>',
                unsafe_allow_html=True
            )

        st.markdown(sector_result["explanation"])

        proxy_note = SECTOR_METADATA[sel_sector].get("proxy_note")
        if proxy_note:
            st.info(f"⚠️ **Proxy note:** {proxy_note}")
else:
    st.warning("Sector module unavailable.")


# ─────────────────────────────────────────────────────────────
# ── FEATURE 2 — ENHANCED DYNAMIC WEIGHT MAP ──────────────────
# ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="fb-new-section">
  <div style="font-size:0.78rem; font-weight:700; color:{_LABEL};
              text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.3rem;">
    🗺️ Enhanced Weight Map + Forecast Uncertainty
    <span title="Hover over cells to see: dominant source, numeric weights, region type, lead/season/regime/sector context, and uncertainty (std across sources)." style="cursor:help;font-size:0.75rem;"> ℹ️</span>
  </div>
  <div style="font-size:0.75rem; color:#57606a; margin-bottom:0.6rem;">
    Interactive map: hover any cell for full contextual breakdown.
    Uncertainty overlay shows std-dev across raw source forecasts per cell.
    <b>Region-type classification is a lat/lon proxy</b> — replace with real
    administrative boundaries when available.
  </div>
</div>
""", unsafe_allow_html=True)

if _WMAP_OK:
    show_uncertainty = st.checkbox(
        "Show uncertainty overlay instead of dominant-weight map",
        value=False, key="show_unc"
    )

    src_list = list(_get_source_arrays(blend_result).keys())

    # Use sector weights if sector module loaded
    sector_wt_spatial = sector_result["sector_weights"] if _SECTOR_OK else None

    enhanced = build_enhanced_weight_map(
        weight_table=wt,
        ds=ds,
        blend_result=blend_result,
        variable=sel_var,
        lead_time=sel_lead,
        season=sel_season,
        regime=sel_regime,
        sector=sel_sector if _SECTOR_OK else "Agriculture",
        time_idx=sel_t,
        sources=src_list,
        sector_weights_spatial=sector_wt_spatial,
    )

    ew_col1, ew_col2 = st.columns(2)

    # Dominant-weight map or uncertainty overlay
    with ew_col1:
        if show_uncertainty:
            z_data = enhanced["uncertainty_map"]
            map_title = "Forecast Uncertainty (σ across sources)"
            cs = "Oranges"
        else:
            z_data = enhanced["dominant_weight_map"]
            map_title = "Dominant Source Weight"
            cs = "Viridis"

        fig_ew = go.Figure(go.Heatmap(
            z=z_data,
            x=lons, y=lats,
            colorscale=cs,
            colorbar=_map_colorbar("σ" if show_uncertainty else "Weight"),
            text=enhanced["hover_text"],
            hovertemplate="%{text}<extra></extra>",
        ))
        fig_ew.update_layout(
            title=dict(text=map_title, font=dict(size=13, color=_MAP_TEXT)),
            margin=dict(l=10, r=10, t=40, b=10),
            paper_bgcolor=_MAP_PAPER,
            plot_bgcolor=_MAP_PLOT,
            font=dict(color=_MAP_TEXT),
            xaxis=_map_axis("Longitude"),
            yaxis=_map_axis("Latitude"),
            height=300,
        )
        st.plotly_chart(fig_ew, use_container_width=True,
                        config={"displayModeBar": False})

    with ew_col2:
        # Region-type proxy map
        rtype_map = build_region_type_map(lats, lons)
        rtype_code = np.zeros_like(lats[:, None] * np.ones((1, len(lons))))
        rtype_labels = ["Agricultural Interior", "Coastal",
                        "Northern Plains / Mountains", "Western Semi-arid"]
        rtype_to_int = {r: i for i, r in enumerate(rtype_labels)}
        for ii in range(len(lats)):
            for jj in range(len(lons)):
                rtype_code[ii, jj] = rtype_to_int.get(str(rtype_map[ii, jj]), 0)

        cs_rtype = [
            [0.0,  "#1e4976"],
            [0.33, "#0e6655"],
            [0.66, "#6b4226"],
            [1.0,  "#7d6608"],
        ]
        fig_rtype = go.Figure(go.Heatmap(
            z=rtype_code,
            x=lons, y=lats,
            colorscale=cs_rtype,
            zmin=0, zmax=3,
            colorbar=_map_colorbar(
                "Region type (proxy)",
                tickvals=[0, 1, 2, 3],
                ticktext=["Agri.", "Coastal", "N.Plains", "W.Semi-arid"],
            ),
            hovertemplate="Lat: %{y:.1f}°<br>Lon: %{x:.1f}°<extra></extra>",
        ))
        fig_rtype.update_layout(
            title=dict(
                text="Region Type Map (lat/lon proxy — not real admin boundaries)",
                font=dict(size=12, color=_MAP_TEXT)
            ),
            margin=dict(l=10, r=10, t=40, b=10),
            paper_bgcolor=_MAP_PAPER,
            plot_bgcolor=_MAP_PLOT,
            font=dict(color=_MAP_TEXT),
            xaxis=_map_axis("Longitude"),
            yaxis=_map_axis("Latitude"),
            height=300,
        )
        st.plotly_chart(fig_rtype, use_container_width=True,
                        config={"displayModeBar": False})
        st.caption(
            "⚠️ Region types derived from lat/lon heuristics only. "
            "Replace with real GIS boundary data for production use."
        )
else:
    st.warning("Weight map enhancement module unavailable.")


# ─────────────────────────────────────────────────────────────
# ── FEATURE 4 — FORECAST CONFIDENCE SCORE ────────────────────
# (shown before Feature 3 so it feeds into the alerts section)
# ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="fb-new-section">
  <div style="font-size:0.78rem; font-weight:700; color:{_LABEL};
              text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.3rem;">
    📊 Forecast Confidence Score
    <span title="Confidence combines model agreement (how closely sources agree) with historical skill for this bucket. They are not the same — low agreement always reduces confidence even if historical skill is high." style="cursor:help;font-size:0.75rem;"> ℹ️</span>
  </div>
</div>
""", unsafe_allow_html=True)

# Build spatial confidence
if _CONF_OK:
    source_arrs = _get_source_arrays(blend_result)
    conf_maps = compute_spatial_confidence(
        source_arrays=source_arrs,
        blended_array=blend_result["blended"],
        time_idx=sel_t,
        skill_df=skill_df,
        variable=sel_var,
        lead_time=sel_lead,
        season=sel_season,
        regime=sel_regime,
    )
    agreement_map_t  = conf_maps["agreement"]
    confidence_map_t = conf_maps["confidence"]
    range_low_t      = conf_maps["range_low"]
    range_high_t     = conf_maps["range_high"]

    conf_c1, conf_c2, conf_c3 = st.columns(3)
    with conf_c1:
        fig_conf = _plotly_heatmap(
            confidence_map_t, lats, lons,
            "Confidence %", colorscale="RdYlGn",
            zmin=0, zmax=100, unit="%"
        )
        st.plotly_chart(fig_conf, use_container_width=True,
                        config={"displayModeBar": False})
    with conf_c2:
        fig_agree = _plotly_heatmap(
            agreement_map_t, lats, lons,
            "Model Agreement %", colorscale="RdYlGn",
            zmin=0, zmax=100, unit="%"
        )
        st.plotly_chart(fig_agree, use_container_width=True,
                        config={"displayModeBar": False})
    with conf_c3:
        range_spread = range_high_t - range_low_t
        fig_range = _plotly_heatmap(
            range_spread, lats, lons,
            f"Expected Range Width ({unit})",
            colorscale="Oranges",
            zmin=0, zmax=float(np.nanpercentile(range_spread, 95)),
            unit=unit,
        )
        st.plotly_chart(fig_range, use_container_width=True,
                        config={"displayModeBar": False})

    # Point-level readout for centre cell
    ci = len(lats) // 2
    cj = len(lons) // 2
    readout = compute_confidence_readout(
        source_arrays=source_arrs,
        blended_array=blend_result["blended"],
        time_idx=sel_t,
        lat_idx=ci, lon_idx=cj,
        skill_df=skill_df,
        variable=sel_var,
        lead_time=sel_lead,
        season=sel_season,
        regime=sel_regime,
    )
    conf_pct  = readout["confidence_pct"]
    agree_pct = readout["agreement_pct"]
    conf_cls  = "conf-high" if conf_pct >= 70 else "conf-mid" if conf_pct >= 45 else "conf-low"
    ag_cls    = "conf-high" if agree_pct >= 70 else "conf-mid" if agree_pct >= 45 else "conf-low"

    st.markdown("**Domain centre-cell confidence readout**")
    cr1, cr2 = st.columns(2)
    with cr1:
        st.markdown(
            f'<div class="fb-sector-card">'
            f'<div style="font-size:0.82rem; color:{_TEXT}; font-weight:600;">Confidence</div>'
            f'<div class="fb-conf-bar-wrap">'
            f'<div class="fb-conf-bar-fill {conf_cls}" style="width:{conf_pct:.0f}%;"></div>'
            f'</div>'
            f'<div style="font-size:1.1rem; font-weight:700; color:{_TEXT};">{conf_pct:.0f}%</div>'
            f'<div style="font-size:0.72rem; color:{_LABEL}; margin-top:4px;">'
            f'Expected range: {readout["range_low"]:.1f}–{readout["range_high"]:.1f} {unit}</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    with cr2:
        st.markdown(
            f'<div class="fb-sector-card">'
            f'<div style="font-size:0.82rem; color:{_TEXT}; font-weight:600;">Model Agreement</div>'
            f'<div class="fb-conf-bar-wrap">'
            f'<div class="fb-conf-bar-fill {ag_cls}" style="width:{agree_pct:.0f}%;"></div>'
            f'</div>'
            f'<div style="font-size:1.1rem; font-weight:700; color:{_TEXT};">{agree_pct:.0f}%</div>'
            f'<div style="font-size:0.72rem; color:{_LABEL}; margin-top:4px;">'
            f'Blended: {readout["blended_value"]:.2f} {unit}</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    st.markdown(readout["explanation"])
else:
    # Fallback: provide neutral confidence map for alerts
    confidence_map_t = np.full((len(lats), len(lons)), 50.0)
    agreement_map_t  = np.full((len(lats), len(lons)), 50.0)
    st.warning("Confidence scoring module unavailable — using 50% neutral confidence for alerts.")


# ─────────────────────────────────────────────────────────────
# ── FEATURE 5 — AUTOMATED EXTREME WEATHER ALERTS ─────────────
# ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="fb-new-section">
  <div style="font-size:0.78rem; font-weight:700; color:{_LABEL};
              text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.3rem;">
    🚨 Extreme Weather Alert Trigger
    <span title="Three-tier severity: WATCH (approaching), WARNING (threshold expected to cross), SEVERE (significantly exceeds + high confidence). Confidence-aware: low-confidence crossings downgraded to WATCH." style="cursor:help;font-size:0.75rem;"> ℹ️</span>
  </div>
  <div style="font-size:0.75rem; color:#57606a; margin-bottom:0.6rem;">
    Alerts use the existing blended forecast — no re-computation. District boundaries are
    <b>coarse proxy tiles</b> labelled with placeholder names. Replace with real
    administrative boundary data for production use.
  </div>
</div>
""", unsafe_allow_html=True)

if _ALERTS_OK:
    # Configurable thresholds
    with st.expander("⚙️ Configure alert thresholds", expanded=False):
        default_thr = EXTREME_THRESHOLDS.get(sel_var, {})
        custom_thr = {}
        for hazard, default_val in default_thr.items():
            custom_thr[hazard] = st.slider(
                f"{hazard.replace('_', ' ').title()} threshold ({unit})",
                min_value=0.0,
                max_value=float(default_val) * 3,
                value=float(default_val),
                step=float(default_val) * 0.05,
                key=f"thr_{hazard}",
            )

    alert_sector = sel_sector if _SECTOR_OK else "Agriculture"
    forecast_date_str = sel_time.strftime("%Y-%m-%d")

    current_alerts = generate_alerts(
        blended_t=blended_t,
        confidence_map=confidence_map_t,
        lats=lats,
        lons=lons,
        variable=sel_var,
        lead_time=sel_lead,
        forecast_date=forecast_date_str,
        sector=alert_sector,
        thresholds=custom_thr if custom_thr else None,
        truth_t=truth_t,
    )

    if current_alerts:
        save_alert_history(current_alerts)

    # Summary counts
    from collections import Counter
    sev_counts = Counter(a["severity"] for a in current_alerts)
    al_s1, al_s2, al_s3, al_s4 = st.columns(4)
    metric_style = "font-size:1.5rem; font-weight:700;"
    with al_s1:
        st.markdown(
            f'<div class="fb-sector-card">'
            f'<div class="fb-card-label">Total Alerts</div>'
            f'<div style="{metric_style} color:{_TEXT};">{len(current_alerts)}</div>'
            f'</div>', unsafe_allow_html=True
        )
    with al_s2:
        c = sev_counts.get("WATCH", 0)
        st.markdown(
            f'<div class="fb-sector-card">'
            f'<div class="fb-card-label">🟡 WATCH</div>'
            f'<div style="{metric_style} color:#fbbf24;">{c}</div>'
            f'</div>', unsafe_allow_html=True
        )
    with al_s3:
        c = sev_counts.get("WARNING", 0)
        st.markdown(
            f'<div class="fb-sector-card">'
            f'<div class="fb-card-label">🟠 WARNING</div>'
            f'<div style="{metric_style} color:#fb923c;">{c}</div>'
            f'</div>', unsafe_allow_html=True
        )
    with al_s4:
        c = sev_counts.get("SEVERE", 0)
        st.markdown(
            f'<div class="fb-sector-card">'
            f'<div class="fb-card-label">🔴 SEVERE</div>'
            f'<div style="{metric_style} color:#f87171;">{c}</div>'
            f'</div>', unsafe_allow_html=True
        )

    if current_alerts:
        # Alert cards (sorted by severity: SEVERE first)
        sev_order = {"SEVERE": 0, "WARNING": 1, "WATCH": 2}
        sorted_alerts = sorted(current_alerts, key=lambda a: sev_order.get(a["severity"], 9))
        st.markdown("**Active alerts for selected date/variable/sector:**")
        for alert in sorted_alerts[:12]:  # cap display at 12
            st.markdown(alert_summary_html(alert), unsafe_allow_html=True)
        if len(sorted_alerts) > 12:
            st.caption(f"… and {len(sorted_alerts)-12} more alerts not shown.")
    else:
        st.success("✅ No alerts triggered for current selection.")

    # District map
    st.markdown("**Alert map — district severity**")
    district_map_arr = build_district_map(lats, lons)
    sev_int_map = np.zeros((len(lats), len(lons)))
    sev_to_int = {"WATCH": 1, "WARNING": 2, "SEVERE": 3}
    if current_alerts:
        dist_sev = {a["district"]: sev_to_int.get(a["severity"], 0) for a in current_alerts}
        for i in range(len(lats)):
            for j in range(len(lons)):
                d = str(district_map_arr[i, j])
                sev_int_map[i, j] = dist_sev.get(d, 0)

    fig_alert_map = go.Figure(go.Heatmap(
        z=sev_int_map,
        x=lons, y=lats,
        colorscale=[
            [0.0,  "#1a1f2e"],
            [0.34, "#92400e"],
            [0.67, "#c2410c"],
            [1.0,  "#991b1b"],
        ],
        zmin=0, zmax=3,
        colorbar=_map_colorbar(
            "Severity",
            tickvals=[0, 1, 2, 3],
            ticktext=["None", "WATCH", "WARNING", "SEVERE"],
        ),
        hovertemplate="Lat: %{y:.1f}°<br>Lon: %{x:.1f}°<br>Severity: %{z}<extra></extra>",
    ))
    fig_alert_map.update_layout(
        title=dict(
            text=f"District Alert Map — {sel_var.title()} @ {sel_lead}h (proxy districts)",
            font=dict(size=13, color=_MAP_TEXT)
        ),
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor=_MAP_PAPER,
        plot_bgcolor=_MAP_PLOT,
        font=dict(color=_MAP_TEXT),
        xaxis=_map_axis("Longitude"),
        yaxis=_map_axis("Latitude"),
        height=300,
    )
    st.plotly_chart(fig_alert_map, use_container_width=True,
                    config={"displayModeBar": False})

    # Alert history
    st.markdown("**Alert history**")
    hist_df = load_alert_history()
    if not hist_df.empty:
        display_cols = [
            "forecast_date", "district", "hazard", "forecast_value",
            "threshold", "confidence_pct", "severity", "sector",
            "accuracy_status",
        ]
        hist_show = hist_df[[c for c in display_cols if c in hist_df.columns]]
        st.dataframe(hist_show.tail(30), use_container_width=True, hide_index=True)
    else:
        st.info("No alert history yet — run the dashboard with various dates to populate.")
else:
    st.warning("Alerts module unavailable.")


# ─────────────────────────────────────────────────────────────
# ── FEATURE 3 — SELF-LEARNING FEEDBACK LOOP ──────────────────
# ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="fb-new-section">
  <div style="font-size:0.78rem; font-weight:700; color:{_LABEL};
              text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.3rem;">
    🔄 Self-Learning Feedback Loop
    <span title="When new ERA5 truth data becomes available, errors are recomputed and fed back into the skill table, updating source weights for similar future situations." style="cursor:help;font-size:0.75rem;"> ℹ️</span>
  </div>
  <div style="font-size:0.75rem; color:#57606a; margin-bottom:0.6rem;">
    The feedback module reads the existing skill table and incrementally
    merges new error observations (weighted by sample count) to improve future weights.
  </div>
</div>
""", unsafe_allow_html=True)

if _FEEDBACK_OK:
    # ── Bug 2 fix: find best-contrast demo day automatically ──────────────────
    _best_demo_idx, _best_spread, _best_msg = find_best_demo_time_idx(
        ds, blend_result, sel_var
    )

    # Manual date selector — separate from the main date slider
    _demo_date_options = list(range(len(times)))
    _demo_default = min(
        _demo_date_options,
        key=lambda i: abs(i - _best_demo_idx)
    )
    fb_top_l, fb_top_r = st.columns([3, 2])
    with fb_top_l:
        sel_demo_t = st.select_slider(
            "Demo date (auto-selected = highest error contrast)",
            options=_demo_date_options,
            value=_demo_default,
            format_func=lambda i: times[i].strftime("%Y-%m-%d"),
            key="feedback_demo_date",
            help="Auto-selection picks the day where models disagreed most. "
                 "Change this to walk through any specific date.",
        )
    with fb_top_r:
        st.markdown(
            f'<div class="fb-sector-card" style="margin-top:1.6rem;">'
            f'<div style="font-size:0.72rem; color:{_TEXT_MUTED};">Auto-selection reason</div>'
            f'<div style="font-size:0.78rem; color:{_TEXT}; margin-top:3px;">{_best_msg}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    run_feedback = st.button(
        "▶ Run feedback update (simulated — uses last 20% of data as 'new truth')",
        key="run_feedback_btn"
    )

    # Re-run whenever the demo date slider changes OR button is pressed
    _demo_key = (sel_var, sel_lead, sel_demo_t)
    if run_feedback or st.session_state.get("_fb_demo_key") != _demo_key:
        st.session_state["_fb_demo_key"] = _demo_key
        with st.spinner("Running feedback cycle…"):
            try:
                updated_skill_df, update_records = run_feedback_update(
                    ds=ds,
                    original_skill_df=skill_df,
                    regime_series=regimes,
                    variables=[sel_var],
                    lead_times=[sel_lead],
                )
                demo = build_demo_record(
                    ds=ds,
                    blend_result=blend_result,
                    regime_series=regimes,
                    variable=sel_var,
                    lead_time=sel_lead,
                    demo_time_idx=sel_demo_t,   # Bug 2: use selected day, not sel_t
                    original_skill_df=skill_df,
                )
                st.session_state["feedback_demo"] = demo
                st.session_state["update_records"] = update_records
                st.session_state["updated_skill_df"] = updated_skill_df
            except Exception as _fe:
                st.error(f"Feedback update error: {_fe}")

    if "feedback_demo" in st.session_state:
        demo = st.session_state["feedback_demo"]

        # Warn if the selected day has low contrast
        if demo.get("no_contrast_msg"):
            st.warning(demo["no_contrast_msg"])

        fb_c1, fb_c2 = st.columns([3, 2])
        with fb_c1:
            st.markdown("**Before/After Learning Demonstration**")
            st.markdown(
                f'<div class="fb-feedback-card">{demo["narrative"]}</div>',
                unsafe_allow_html=True
            )
        with fb_c2:
            shifts = demo.get("weight_shifts", {})
            if shifts:
                st.markdown("**Weight shift chart**")

                # Bug 1 fix: separate applicable sources from N/A ones
                applicable_srcs = [s for s in shifts if not shifts[s].get("na")]
                na_srcs         = [s for s in shifts if shifts[s].get("na")]

                SOURCE_FRIENDLY = {"nwp": "NWP", "ensemble": "Ensemble", "ai": "AI"}

                if applicable_srcs:
                    shift_fig = go.Figure()
                    before_vals = [shifts[s]["before"] for s in applicable_srcs]
                    after_vals  = [shifts[s]["after"]  for s in applicable_srcs]
                    x_labels    = [SOURCE_FRIENDLY.get(s, s.upper()) for s in applicable_srcs]

                    shift_fig.add_trace(go.Bar(
                        name="Before", x=x_labels, y=before_vals,
                        marker_color="#60a5fa",
                        text=[f"{v:.0f}%" for v in before_vals],
                        textposition="outside",
                    ))
                    shift_fig.add_trace(go.Bar(
                        name="After", x=x_labels, y=after_vals,
                        marker_color="#34d399",
                        text=[f"{v:.0f}%" for v in after_vals],
                        textposition="outside",
                    ))
                    shift_fig.update_layout(
                        barmode="group",
                        title=dict(
                            text="Source weight: before vs after",
                            font=dict(size=12, color=_TEXT)
                        ),
                        xaxis=dict(tickfont=dict(color=_TEXT)),
                        yaxis=dict(title="Weight %", tickfont=dict(color=_TEXT),
                                   range=[0, 110]),
                        legend=dict(
                            orientation="h", y=-0.25, x=0.5, xanchor="center",
                            font=dict(color=_TEXT)
                        ),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        margin=dict(l=40, r=20, t=40, b=80),
                        height=320,
                    )
                    st.plotly_chart(shift_fig, use_container_width=True,
                                    config={"displayModeBar": False})

                # Bug 1 fix: show excluded sources explicitly below the chart
                if na_srcs:
                    na_labels = ", ".join(
                        f"{SOURCE_FRIENDLY.get(s, s.upper())}" for s in na_srcs
                    )
                    st.caption(
                        f"ℹ️ {na_labels}: not shown — no {sel_var} data from "
                        f"{'this source' if len(na_srcs)==1 else 'these sources'}."
                    )
    else:
        st.info("Select a demo date above — the chart updates automatically.")
else:
    st.warning("Feedback loop module unavailable.")


# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
_footer_logo = get_logo_html(height_px=32)
st.markdown(
    f'<div class="fb-footer" style="display:flex; flex-direction:column; align-items:center; gap:4px;">'
    f'<div style="display:flex; align-items:center; gap:8px;">'
    f'{_footer_logo}'
    f'<span style="font-weight:700; font-size:0.82rem;">'
    f'<span style="color:#22c55e;">Prakriti</span><span style="color:#3b82f6;">Netra</span>'
    f'</span>'
    f'</div>'
    f'<div>'
    f'Smarter Forecasts &nbsp;|&nbsp; Healthier Tomorrow'
    f'&nbsp;&nbsp;&middot;&nbsp;&nbsp; GFS + GEFS + Pangu + ERA5'
    f'&nbsp;&nbsp;&middot;&nbsp;&nbsp; India domain (8&ndash;37&deg;N, 68&ndash;97&deg;E)'
    f'&nbsp;&nbsp;&middot;&nbsp;&nbsp; Sector Blend &nbsp;&middot;&nbsp; Confidence &nbsp;&middot;&nbsp; Alerts &nbsp;&middot;&nbsp; Feedback Loop'
    f'</div>'
    f'</div>',
    unsafe_allow_html=True
)
