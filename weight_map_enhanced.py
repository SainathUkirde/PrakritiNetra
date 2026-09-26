"""
weight_map_enhanced.py
======================
Enhanced Dynamic Model Weight Map (Feature 2).

Adds on top of the existing weight map (which already exists in app.py):
  1. Per-cell dominant source + numeric weight (not just colour)
  2. Contextual breakdown: region type proxy, lead time, season, regime, sector
  3. Forecast uncertainty indicator: std-deviation across raw source forecasts
     per grid cell — shown as an additional overlay/toggle
  4. Hover-ready data structures for interactive Plotly maps

Design constraints
------------------
* Does NOT modify weighting.py / scoring.py / blending.py.
* Reads from existing weight_table and blend_result outputs.

Region-type proxy
-----------------
Real administrative boundary data is NOT available.  A simple lat/lon proxy
is used to classify each cell:
  - lat < 14°N and lon near coast (lon < 74° or lon > 86°) → "Coastal"
  - lat > 28°N → "Northern Plains / Mountains"
  - lon between 72-76°E and lat 18-28°N → "Western semi-arid"
  - otherwise → "Agricultural Interior"
This is explicitly a proxy.  Replace classify_region_type() with real
administrative boundary data when available.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from weighting import get_weight_map, get_weights, build_region_map
from scoring import region_id_to_label


# ── Region-type proxy ──────────────────────────────────────────────────────────

def classify_region_type(lat: float, lon: float) -> str:
    """
    Assign a coarse region-type label from lat/lon position.

    NOTE: This is a PROXY classification based on geographic heuristics.
    It is NOT derived from real administrative boundary or land-use data.
    Replace this function with real boundaries when available.
    """
    if lat < 14.5 and (lon < 75.0 or lon > 85.0):
        return "Coastal"
    if lat > 28.0:
        return "Northern Plains / Mountains"
    if 72.0 <= lon <= 77.0 and 18.0 <= lat <= 28.0:
        return "Western Semi-arid"
    return "Agricultural Interior"


def build_region_type_map(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """
    Return (n_lat, n_lon) string array of region-type labels.
    """
    out = np.empty((len(lats), len(lons)), dtype=object)
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            out[i, j] = classify_region_type(float(lat), float(lon))
    return out


# ── Uncertainty map ────────────────────────────────────────────────────────────

def compute_uncertainty_map(
    blend_result: dict[str, np.ndarray],
    time_idx: int,
    sources: list[str],
) -> np.ndarray:
    """
    Compute per-cell forecast uncertainty as the standard deviation across
    the raw source forecasts at the given time step.

    A high value = large source disagreement = higher uncertainty.
    A low value  = sources agree closely     = lower uncertainty.

    Parameters
    ----------
    blend_result : output of blend_forecast() containing per-source arrays
    time_idx     : which time step to evaluate
    sources      : list of source keys to include

    Returns
    -------
    (n_lat, n_lon) float array of std-dev across sources
    """
    avail = [s for s in sources
             if s in blend_result
             and not np.all(np.isnan(blend_result[s][time_idx]))]

    if len(avail) < 2:
        ref = blend_result[avail[0]][time_idx] if avail else np.zeros((1, 1))
        return np.zeros_like(ref)

    stack = np.stack([blend_result[s][time_idx] for s in avail], axis=0)  # (n_src, H, W)
    return np.nanstd(stack, axis=0)


# ── Per-cell contextual breakdown ─────────────────────────────────────────────

def build_cell_context(
    lat: float,
    lon: float,
    weight_table: pd.DataFrame,
    variable: str,
    lead_time: int,
    season: str,
    regime: str,
    sector: str,
    region_label: str,
    sector_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """
    Build the full contextual breakdown for a single grid cell.

    Parameters
    ----------
    sector_weights : if provided, use sector-adjusted weights instead of base

    Returns
    -------
    dict with all fields needed for the interactive popup / hover card
    """
    base_weights = get_weights(weight_table, variable, lead_time,
                               region_label, season, regime)

    weights_to_show = sector_weights if sector_weights is not None else base_weights
    total = sum(weights_to_show.values()) or 1.0
    norm_weights = {s: w / total for s, w in weights_to_show.items()}

    dominant_src = max(norm_weights, key=norm_weights.get) if norm_weights else "unknown"
    dominant_pct = norm_weights.get(dominant_src, 0.0) * 100.0

    region_type = classify_region_type(lat, lon)

    return {
        "lat":          round(lat, 2),
        "lon":          round(lon, 2),
        "region":       region_label,
        "region_type":  region_type,
        "lead_time":    lead_time,
        "season":       season,
        "regime":       regime,
        "sector":       sector,
        "weights":      {s: round(w * 100, 1) for s, w in norm_weights.items()},
        "dominant_src": dominant_src,
        "dominant_pct": round(dominant_pct, 1),
    }


def build_hover_text_map(
    lats: np.ndarray,
    lons: np.ndarray,
    weight_table: pd.DataFrame,
    ds: xr.Dataset,
    variable: str,
    lead_time: int,
    season: str,
    regime: str,
    sector: str,
    uncertainty_map: np.ndarray,
    sector_weights_spatial: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """
    Return (n_lat, n_lon) array of hover-text strings for the enhanced weight map.
    Each cell's text shows: dominant source, weights%, region type, uncertainty.
    """
    region_map = build_region_map(ds)
    hover = np.empty((len(lats), len(lons)), dtype=object)

    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            region_id = int(region_map[i, j])
            rlabel = region_id_to_label(region_id)

            # Per-cell sector weights if available
            sw = None
            if sector_weights_spatial is not None:
                sw = {
                    src: float(sector_weights_spatial[src][i, j])
                    for src in sector_weights_spatial
                }

            ctx = build_cell_context(
                lat, lon, weight_table, variable, lead_time,
                season, regime, sector, rlabel, sector_weights=sw
            )
            unc = float(uncertainty_map[i, j])

            weight_str = " | ".join(
                f"{s.upper()}:{pct:.0f}%" if not np.isnan(pct) else f"{s.upper()}:N/A"
                for s, pct in sorted(ctx["weights"].items(), key=lambda x: -x[1])
            )

            dom_pct_str = f"{ctx['dominant_pct']:.0f}%" if not np.isnan(ctx['dominant_pct']) else "N/A"
            hover[i, j] = (
                f"<b>{ctx['dominant_src'].upper()}</b> {dom_pct_str}<br>"
                f"{weight_str}<br>"
                f"Region type: {ctx['region_type']}<br>"
                f"Lead: {lead_time}h | {season.title()} | {regime.title()}<br>"
                f"Sector: {sector}<br>"
                f"Uncertainty (σ): {unc:.3f}"
            )

    return hover


def build_enhanced_weight_map(
    weight_table: pd.DataFrame,
    ds: xr.Dataset,
    blend_result: dict[str, np.ndarray],
    variable: str,
    lead_time: int,
    season: str,
    regime: str,
    sector: str,
    time_idx: int,
    sources: list[str],
    sector_weights_spatial: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """
    Build all data needed for the enhanced weight map panel in the dashboard.

    Returns
    -------
    dict:
      'dominant_weight_map'  : (n_lat, n_lon) float — weight of dominant source per cell
      'dominant_source_map'  : (n_lat, n_lon) str   — name of dominant source per cell
      'uncertainty_map'      : (n_lat, n_lon) float — std across sources
      'hover_text'           : (n_lat, n_lon) str   — HTML hover text
      'per_source_weights'   : {src: (n_lat, n_lon)} — adjusted weight maps
    """
    lats = ds.coords["lat"].values
    lons = ds.coords["lon"].values
    region_map = build_region_map(ds)

    # Per-source spatial weights (sector-adjusted or base)
    if sector_weights_spatial is not None:
        per_src = sector_weights_spatial
    else:
        per_src = {}
        for src in sources:
            per_src[src] = get_weight_map(
                weight_table, ds, variable, lead_time, season, regime, source=src
            )

    # Dominant source + its weight per cell
    n_lat, n_lon = len(lats), len(lons)
    dom_weight_map  = np.zeros((n_lat, n_lon))
    dom_source_map  = np.empty((n_lat, n_lon), dtype=object)

    weight_stack = np.stack(
        [per_src.get(s, np.zeros((n_lat, n_lon))) for s in sources], axis=0
    )  # (n_src, H, W)
    # Replace NaN weights (e.g. from absent AI source) with 0 before normalising
    weight_stack = np.nan_to_num(weight_stack, nan=0.0)
    total = weight_stack.sum(axis=0, keepdims=True)
    total = np.where(total == 0, 1.0, total)
    norm_stack = weight_stack / total

    dom_idx = np.argmax(norm_stack, axis=0)  # (H, W)
    for i, src in enumerate(sources):
        mask = dom_idx == i
        dom_weight_map[mask]  = norm_stack[i][mask]
        dom_source_map[mask]  = src

    # Uncertainty map
    unc_map = compute_uncertainty_map(blend_result, time_idx, sources)

    # Hover text
    hover_text = build_hover_text_map(
        lats, lons, weight_table, ds, variable, lead_time,
        season, regime, sector, unc_map, sector_weights_spatial
    )

    return {
        "dominant_weight_map":  dom_weight_map,
        "dominant_source_map":  dom_source_map,
        "uncertainty_map":      unc_map,
        "hover_text":           hover_text,
        "per_source_weights":   {s: norm_stack[i] for i, s in enumerate(sources)},
    }
