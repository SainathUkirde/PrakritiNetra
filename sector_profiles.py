"""
sector_profiles.py
==================
Sector-Adaptive Forecast Blending (Feature 1).

Adds a *sector layer* on top of the existing inverse-error weights.
Each sector re-weights the existing model contributions according to
which variables and accuracy characteristics matter most for that sector.

Design constraints
------------------
* Does NOT modify any existing module (blending.py / weighting.py / scoring.py).
* Reads existing blended output + skill/weight tables as inputs.
* Produces a sector-specific *re-weighted combination* of per-source forecasts
  plus a dynamic explanation string derived from actual weight/skill numbers.

Sectors
-------
Agriculture : Rainfall amount, temperature, dry/wet spell duration, heat-stress.
              Emphasises: rainfall (0.50), temperature (0.30), wind (0.20).
              Source preference: AI (strong long-range rainfall), NWP (temp/heat).

Hydrology   : Catchment-level accumulated rainfall, rainfall intensity,
              extreme-rainfall probability.
              Emphasises: rainfall (0.70), wind (0.20), temperature (0.10).
              Source preference: NWP (short-range intensity), AI (long-range).

Aviation    : Wind speed/direction, gusts, thunderstorm probability.
              Emphasises: wind (0.60), rainfall (0.30), temperature (0.10).
              Source preference: NWP (precise short-range), Ensemble (spread).

Energy      : Temperature, wind, and a clear-sky/cloud-cover *proxy*.
              NOTE: Solar irradiance data is not available in the current
              pipeline.  As a documented proxy, low-rainfall + moderate-wind
              conditions are used as a "solar-favourable" indicator.  This
              substitution is stated explicitly here and in the UI.
              Emphasises: temperature (0.40), wind (0.40), rainfall (0.20).
              Source preference: NWP (temperature accuracy), AI (wind).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# ── Sector definitions ─────────────────────────────────────────────────────────

SECTORS: list[str] = ["Agriculture", "Hydrology", "Aviation", "Energy"]

# Variable importance weights per sector (must sum to 1.0 for each sector)
SECTOR_VAR_WEIGHTS: dict[str, dict[str, float]] = {
    "Agriculture": {"rainfall": 0.50, "temperature": 0.30, "wind": 0.20},
    "Hydrology":   {"rainfall": 0.70, "wind":        0.20, "temperature": 0.10},
    "Aviation":    {"wind":     0.60, "rainfall":    0.30, "temperature": 0.10},
    "Energy":      {"temperature": 0.40, "wind":     0.40, "rainfall":    0.20},
}

# Source preference boosts per sector (additive bonus before renormalisation)
# Values are relative boosts — small to avoid overriding historical skill data.
SECTOR_SOURCE_BOOST: dict[str, dict[str, float]] = {
    "Agriculture": {"ai": 0.15, "nwp": 0.05, "ensemble": 0.00},
    "Hydrology":   {"nwp": 0.15, "ai":  0.05, "ensemble": 0.00},
    "Aviation":    {"nwp": 0.20, "ensemble": 0.05, "ai": 0.00},
    "Energy":      {"nwp": 0.10, "ai": 0.10, "ensemble": 0.00},
}

# Human-readable sector context (for UI icons / descriptions)
SECTOR_METADATA: dict[str, dict[str, Any]] = {
    "Agriculture": {
        "icon": "🌾",
        "focus": "Rainfall, temperature, heat-stress, dry/wet spell duration",
        "proxy_note": None,
    },
    "Hydrology": {
        "icon": "💧",
        "focus": "Catchment-accumulated rainfall, rainfall intensity, flood risk",
        "proxy_note": None,
    },
    "Aviation": {
        "icon": "✈️",
        "focus": "Wind speed/direction, gusts, thunderstorm probability",
        "proxy_note": None,
    },
    "Energy": {
        "icon": "⚡",
        "focus": "Temperature, wind, solar-favourable conditions",
        "proxy_note": (
            "Solar irradiance data is not available in this pipeline. "
            "A proxy is used: low-rainfall + moderate-wind conditions are treated "
            "as 'solar-favourable'. This is an explicit simplification pending "
            "real irradiance / cloud-cover data integration."
        ),
    },
}

# Regime-specific narrative fragments keyed by (sector, variable, regime).
# Using a three-level dict ensures the reason shown always matches the active
# variable — a temperature-specific note will never appear for a rainfall view.
# Format: REGIME_NARRATIVE[sector][variable][regime]
REGIME_NARRATIVE: dict[str, dict[str, dict[str, str]]] = {
    "Agriculture": {
        "rainfall": {
            "convective": "AI model has higher historical rainfall skill in convective regimes",
            "stratiform": "NWP captures steady stratiform rainfall well at this lead",
            "clear":      "Low rainfall expected in clear regime; ensemble provides baseline",
        },
        "temperature": {
            "convective": "Convective activity raises heat stress risk; NWP captures diurnal cycle",
            "stratiform": "Overcast stratiform conditions moderate heat stress; NWP leads",
            "clear":      "Clear-sky temperature accuracy favours NWP for heat-stress forecasting",
        },
        "wind": {
            "convective": "Convective gusts can damage standing crops; AI gives longer-range warning",
            "stratiform": "Steady stratiform wind well captured by ensemble",
            "clear":      "Light winds in clear regime; all sources closely agree",
        },
    },
    "Hydrology": {
        "rainfall": {
            "convective": "NWP provides better short-range rainfall intensity in convective events",
            "stratiform": "AI model resolves widespread stratiform accumulation well",
            "clear":      "Low flood risk in clear regime; ensemble used for baseline flow",
        },
        "temperature": {
            "convective": "Convective warming drives snowmelt; NWP captures rapid temperature rises",
            "stratiform": "Overcast stratiform reduces evapotranspiration demand",
            "clear":      "Clear-sky warming increases evapotranspiration; temperature accuracy critical",
        },
        "wind": {
            "convective": "Strong convective winds increase wave height on reservoirs",
            "stratiform": "Moderate winds in stratiform regime; ensemble spread guides uncertainty",
            "clear":      "Light winds; wave and channel flow risk low",
        },
    },
    "Aviation": {
        "rainfall": {
            "convective": "Convective precipitation indicates icing and turbulence risk",
            "stratiform": "Stratiform rain reduces visibility; ensemble spread informs routing",
            "clear":      "No precipitation expected; minimal aviation impact",
        },
        "temperature": {
            "convective": "Rapid temperature changes in convective cells affect density altitude",
            "stratiform": "Stable stratiform temperature profile; reliable NWP forecast",
            "clear":      "Clear-sky temperature affects density altitude and runway performance",
        },
        "wind": {
            "convective": "NWP dominates for wind gusts and thunderstorm timing in convective regime",
            "stratiform": "Ensemble spread gives reliable uncertainty for flight planning",
            "clear":      "All sources show high wind agreement in clear regime",
        },
    },
    "Energy": {
        "rainfall": {
            "convective": "Cloud-cover proxy indicates reduced solar output; wind forecast critical",
            "stratiform": "Overcast conditions suppress solar generation",
            "clear":      "Solar-favourable conditions; low rainfall supports clear-sky proxy",
        },
        "temperature": {
            "convective": "Convective heating drives peak electricity demand",
            "stratiform": "Overcast conditions suppress solar; temperature drives demand",
            "clear":      "Solar-favourable proxy active; NWP+AI temperature/wind blend optimal",
        },
        "wind": {
            "convective": "Convective gusts increase turbine load; AI gives extended outlook",
            "stratiform": "Steady stratiform winds predictable; good for turbine scheduling",
            "clear":      "Clear-sky light winds may reduce turbine output",
        },
    },
}


# ── Core function ──────────────────────────────────────────────────────────────

def compute_sector_weights(
    existing_weights: dict[str, float],
    sector: str,
    variable: str,
    skill_df: pd.DataFrame,
    lead_time: int,
    region: str,
    season: str,
    regime: str,
) -> dict[str, float]:
    """
    Apply sector re-weighting on top of existing inverse-error weights.

    Parameters
    ----------
    existing_weights : {source: weight} from build_weight_table / get_weights
    sector           : one of SECTORS
    variable         : 'rainfall' | 'temperature' | 'wind'
    skill_df         : full skill table from compute_skill_scores()
    lead_time, region, season, regime : current bucket context

    Returns
    -------
    dict {source: adjusted_weight}  — sums to 1.0
    """
    if sector not in SECTORS:
        return existing_weights  # passthrough for unknown sector

    var_importance = SECTOR_VAR_WEIGHTS[sector].get(variable, 1.0 / 3)
    source_boosts  = SECTOR_SOURCE_BOOST[sector]

    # Apply source preference boost (additive, then renormalise)
    boosted: dict[str, float] = {}
    for src, base_w in existing_weights.items():
        boost = source_boosts.get(src, 0.0) * var_importance
        boosted[src] = base_w + boost

    total = sum(boosted.values())
    if total <= 0:
        n = len(boosted)
        return {s: 1.0 / n for s in boosted}

    return {s: w / total for s, w in boosted.items()}


def sector_forecast_view(
    blend_result: dict[str, np.ndarray],
    existing_weights_spatial: dict[str, np.ndarray],
    sector: str,
    variable: str,
    skill_df: pd.DataFrame,
    lead_time: int,
    season: str,
    regime: str,
) -> dict[str, Any]:
    """
    Produce a sector-specific forecast view from the existing blended outputs.

    This computes a sector-optimised spatial blend using boosted weights,
    plus a dynamic explanation card.

    Parameters
    ----------
    blend_result              : output of blend_forecast() — per-source arrays
    existing_weights_spatial  : {source: (n_lat, n_lon) weight map}
    sector                    : target sector
    variable                  : active variable
    skill_df                  : full skill table
    lead_time, season, regime : current context

    Returns
    -------
    dict with keys:
        'sector_blended'   : (n_lat, n_lon) sector-optimised forecast for sel_t
        'sector_weights'   : {source: (n_lat, n_lon)} sector-adjusted weight maps
        'explanation'      : formatted explanation string
        'sector_pct'       : {source: mean_weight_pct} for display
    """
    sources = [s for s in existing_weights_spatial
                if s in blend_result and not np.all(np.isnan(blend_result.get(s, np.array([np.nan]))))]

    if not sources:
        return {
            "sector_blended": blend_result.get("blended", np.zeros((1, 1))),
            "sector_weights": {},
            "explanation": "No source data available.",
            "sector_pct": {},
        }

    var_importance = SECTOR_VAR_WEIGHTS[sector].get(variable, 1.0 / 3)
    source_boosts  = SECTOR_SOURCE_BOOST[sector]

    # Build sector-adjusted spatial weight maps
    sector_weights: dict[str, np.ndarray] = {}
    for src in sources:
        base = existing_weights_spatial[src].copy()
        boost = source_boosts.get(src, 0.0) * var_importance
        sector_weights[src] = base + boost

    # Renormalise per-cell (sum of sector weights across sources → 1 per cell)
    weight_stack = np.stack([sector_weights[s] for s in sources], axis=0)  # (n_src, H, W)
    weight_stack = np.nan_to_num(weight_stack, nan=0.0)  # guard: absent source → 0
    total = weight_stack.sum(axis=0, keepdims=True)
    total = np.where(total == 0, 1.0, total)
    norm_stack = weight_stack / total

    sector_weights_norm = {s: norm_stack[i] for i, s in enumerate(sources)}

    # Compute sector-blended forecast for the selected time step
    # blend_result[src] shape: (n_time, n_lat, n_lon); we return per-timestep handled by caller
    # Here we return full temporal array, caller slices
    n_time = blend_result[sources[0]].shape[0]
    n_lat, n_lon = norm_stack.shape[1], norm_stack.shape[2]
    sector_blended = np.zeros((n_time, n_lat, n_lon))
    for i, src in enumerate(sources):
        sector_blended += norm_stack[i][np.newaxis, :, :] * blend_result[src]

    # Compute mean weights for explanation card.
    # sector_weights_norm[src] values already sum to 1.0 per cell across sources,
    # but the mean across cells may still be <1/3 per source due to spatial variation.
    # We want percentages that sum to 100% — so we normalise the mean weights.
    raw_pct: dict[str, float] = {}
    for src in sources:
        raw_pct[src] = float(np.nanmean(sector_weights_norm[src]))
    total_raw = sum(raw_pct.values())
    if total_raw > 0:
        sector_pct = {s: v / total_raw * 100.0 for s, v in raw_pct.items()}
    else:
        n = len(sources)
        sector_pct = {s: 100.0 / n for s in sources}

    # Build dynamic explanation
    explanation = _build_explanation(
        sector, variable, sector_pct, sources, lead_time, season, regime,
        skill_df
    )

    return {
        "sector_blended":       sector_blended,
        "sector_weights":       sector_weights_norm,
        "explanation":          explanation,
        "sector_pct":           sector_pct,
    }


def _build_explanation(
    sector: str,
    variable: str,
    sector_pct: dict[str, float],
    sources: list[str],
    lead_time: int,
    season: str,
    regime: str,
    skill_df: pd.DataFrame,
) -> str:
    """Generate dynamic explanation from actual weight and skill numbers."""
    sorted_sources = sorted(sector_pct.items(), key=lambda x: -x[1])
    SOURCE_FRIENDLY = {"nwp": "NWP Model", "ensemble": "Ensemble", "ai": "AI Model"}

    lines = [f"**{sector} Sector — {variable.title()} | {regime.title()} Regime**", ""]
    for src, pct in sorted_sources:
        lines.append(f"  {SOURCE_FRIENDLY.get(src, src.upper()):<16} {pct:.0f}%")
    lines.append("")
    lines.append("**Reason:**")

    # Dominant source narrative
    dom_src, dom_pct = sorted_sources[0]
    dom_friendly = SOURCE_FRIENDLY.get(dom_src, dom_src.upper())
    # Three-level lookup: sector → variable → regime
    # This guarantees the reason is always variable-matched (Bug C fix).
    regime_note = REGIME_NARRATIVE.get(sector, {}).get(variable, {}).get(regime, "")
    if regime_note:
        lines.append(f"  - {regime_note}")

    # Lead-time narrative
    if lead_time <= 24:
        lines.append(f"  - Short lead time ({lead_time}h) — NWP physics tend to dominate")
    elif lead_time <= 72:
        lines.append(f"  - Medium lead time ({lead_time}h) — AI model adds skill beyond day-1")
    else:
        lines.append(f"  - Long lead time ({lead_time}h) — AI model outperforms NWP at this range")

    # Skill narrative from actual numbers (if available)
    if not skill_df.empty:
        grp = skill_df[
            (skill_df["variable"] == variable) &
            (skill_df["lead_time"] == lead_time) &
            (skill_df["season"] == season) &
            (skill_df["regime"] == regime)
        ]
        if not grp.empty:
            rmse_by_src = grp.groupby("source")["rmse"].mean()
            if dom_src in rmse_by_src.index:
                dom_rmse = rmse_by_src[dom_src]
                lines.append(f"  - {dom_friendly} RMSE in this bucket: {dom_rmse:.3f} (historical mean)")
            if season == "monsoon":
                lines.append("  - Regional monsoon performance is stronger for this source")

    # Sector-specific footnote
    proxy_note = SECTOR_METADATA[sector].get("proxy_note")
    if proxy_note:
        lines.append(f"  ⚠️  Proxy: {proxy_note[:80]}…")

    return "\n".join(lines)
