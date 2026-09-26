"""
alerts.py
=========
Automated Extreme Weather Alert Trigger (Feature 5).

Evaluates the existing blended forecast against configurable per-variable
thresholds and generates structured alert records with three severity tiers.

Design constraints
------------------
* Does NOT recompute blending — reads from blend_forecast() output directly.
* Reuses existing EXTREME_THRESHOLDS from blending.py as defaults.
* Confidence-aware severity: a threshold-crossing with low confidence is
  downgraded to WATCH.

District-level grid
-------------------
Real district boundary data is NOT available.  A coarse tiling of the
16×16 India domain (8–37°N, 68–97°E) into named placeholder districts is
used instead.  The 4×4 coarse region tiles from scoring.py are subdivided
to produce ~16 districts.  District names are illustrative placeholders
labelled by quadrant position — replace with real administrative boundaries
when GIS boundary data is available.  This is explicitly stated as a
simplification.

Severity tiers
--------------
WATCH  : Value approaching threshold (≥ 70% of threshold) or crossing with
         low confidence (< 50%)
WARNING: Value expected to cross threshold, moderate confidence (≥ 50%)
SEVERE : Significantly exceeds threshold (≥ 130% of threshold) with high
         confidence (≥ 75%) OR exceeds threshold with very high confidence (≥ 90%)

Confidence-aware logic (as specified)
--------------------------------------
  threshold crossing + HIGH confidence (≥ 75%) → WARNING or SEVERE
  threshold crossing + LOW  confidence (< 50%)  → WATCH only
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from blending import EXTREME_THRESHOLDS

# Persistence path for alert history
_ALERT_HISTORY_CSV = Path(__file__).parent / "outputs" / "alert_history.csv"

# ── Severity configuration ─────────────────────────────────────────────────────

SEVERITY_LEVELS  = ["WATCH", "WARNING", "SEVERE"]
WATCH_FRACTION   = 0.70   # ≥ 70% of threshold → approaching
SEVERE_FRACTION  = 1.30   # ≥ 130% of threshold → significantly exceeds
HIGH_CONF_THRESH = 75.0   # % — required for WARNING/SEVERE
LOW_CONF_THRESH  = 50.0   # % — below this → downgrade to WATCH

SEVERITY_ICONS = {
    "WATCH":   "🟡",
    "WARNING": "🟠",
    "SEVERE":  "🔴",
}

# ── District proxy grid ────────────────────────────────────────────────────────
# India domain: lat 8–37°N, lon 68–97°E → 29°×29° split into 4×4 named tiles
# Each tile is further split 2×2 → 16×16 = 16 coarse "districts"
# PROXY NOTE: These are placeholder names based on quadrant position.
# Replace with real district boundary data when available.

_LAT_BOUNDS = (8.0, 37.0)
_LON_BOUNDS = (68.0, 97.0)
_N_DIST_LAT = 4
_N_DIST_LON = 4

_DISTRICT_NAMES: list[list[str]] = [
    # lat-row 0 (southernmost: ~8–15°N)
    ["South-West Coast", "South Peninsular West", "South Peninsular East", "South-East Coast"],
    # lat-row 1 (~15–22°N)
    ["Central West Deccan", "Central Deccan", "Central East Deccan", "East Coast Central"],
    # lat-row 2 (~22–29°N)
    ["North-West Semi-arid", "Central Plains West", "Central Plains East", "North-East Bay"],
    # lat-row 3 (northernmost: ~29–37°N)
    ["North-West Arid", "Northern Plains", "Upper Gangetic Plains", "North-East Hills"],
]


def _assign_district(lat: float, lon: float) -> str:
    """
    Assign a placeholder district name from lat/lon position.
    PROXY: Replace with real boundary data.
    """
    lat_frac = (lat - _LAT_BOUNDS[0]) / (_LAT_BOUNDS[1] - _LAT_BOUNDS[0])
    lon_frac = (lon - _LON_BOUNDS[0]) / (_LON_BOUNDS[1] - _LON_BOUNDS[0])
    row = int(np.clip(lat_frac * _N_DIST_LAT, 0, _N_DIST_LAT - 1))
    col = int(np.clip(lon_frac * _N_DIST_LON, 0, _N_DIST_LON - 1))
    return _DISTRICT_NAMES[row][col]


def build_district_map(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Return (n_lat, n_lon) string array of district names (proxy)."""
    out = np.empty((len(lats), len(lons)), dtype=object)
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            out[i, j] = _assign_district(float(lat), float(lon))
    return out


# ── Severity determination ─────────────────────────────────────────────────────

def _determine_severity(
    value: float,
    threshold: float,
    confidence_pct: float,
) -> str | None:
    """
    Return severity tier string or None if no alert should be raised.

    Implements the confidence-aware logic as specified.
    """
    # Guard: NaN or Inf forecast value → no alert (do not produce false alert)
    if value != value or not np.isfinite(value):  # NaN or Inf check
        return None

    ratio = value / threshold if threshold > 0 else 0.0

    if ratio < WATCH_FRACTION:
        return None  # well below threshold — no alert

    # Threshold-crossing or approaching
    if confidence_pct < LOW_CONF_THRESH:
        # Low confidence → maximum WATCH, even if value exceeds threshold
        return "WATCH"

    if ratio >= SEVERE_FRACTION and confidence_pct >= HIGH_CONF_THRESH:
        return "SEVERE"
    if ratio >= 1.0 and confidence_pct >= HIGH_CONF_THRESH:
        return "WARNING"
    if ratio >= 1.0 and confidence_pct >= LOW_CONF_THRESH:
        # Crosses threshold but moderate confidence
        return "WARNING"
    # Approaching threshold (ratio in [0.7, 1.0))
    return "WATCH"


# ── Sector-specific alert phrasing ────────────────────────────────────────────

_SECTOR_PHRASES: dict[str, dict[str, str]] = {
    "Agriculture": {
        "heavy_rainfall": "Monitor waterlogging, field drainage and crop damage risk.",
        "heat_wave":      "Monitor heat-stress on livestock and high-value crops.",
        "high_wind":      "Secure farm structures; monitor wind damage to standing crops.",
    },
    "Hydrology": {
        "heavy_rainfall": "Monitor downstream flow, reservoir inflow and catchment saturation.",
        "heat_wave":      "Monitor evapotranspiration rates and irrigation demand.",
        "high_wind":      "Monitor wind-driven wave heights on reservoirs and open channels.",
    },
    "Aviation": {
        "heavy_rainfall": "Monitor convective activity and low-visibility conditions.",
        "heat_wave":      "Monitor density altitude effects and runway surface temperatures.",
        "high_wind":      "Monitor wind gusts, crosswind limits and thunderstorm advisories.",
    },
    "Energy": {
        "heavy_rainfall": "Monitor solar generation suppression due to cloud cover (proxy).",
        "heat_wave":      "Monitor peak electricity demand and grid stress.",
        "high_wind":      "Monitor wind turbine output and structural load limits.",
    },
}

_DEFAULT_PHRASE: dict[str, str] = {
    "heavy_rainfall": "Monitor extreme precipitation conditions.",
    "heat_wave":      "Monitor extreme heat conditions.",
    "high_wind":      "Monitor high wind and structural risk.",
}


def _recommended_action(hazard: str, severity: str, sector: str) -> str:
    base = _SECTOR_PHRASES.get(sector, {}).get(hazard) or _DEFAULT_PHRASE.get(hazard, "Take precautions.")
    prefix = {
        "WATCH":   "Stay informed.",
        "WARNING": "Prepare for impact.",
        "SEVERE":  "Immediate action required.",
    }.get(severity, "")
    return f"{prefix} {base}".strip()


# ── Core alert generation ──────────────────────────────────────────────────────

def generate_alerts(
    blended_t: np.ndarray,
    confidence_map: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    variable: str,
    lead_time: int,
    forecast_date: str,
    sector: str = "Agriculture",
    thresholds: dict | None = None,
    truth_t: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """
    Evaluate the blended forecast for a single time step and return a list
    of structured alert records.

    Parameters
    ----------
    blended_t      : (n_lat, n_lon) blended forecast
    confidence_map : (n_lat, n_lon) confidence % [0,100]
    lats, lons     : coordinate arrays
    variable       : 'rainfall' | 'temperature' | 'wind'
    lead_time      : forecast lead time in hours
    forecast_date  : ISO date string for the forecast period
    sector         : current sector (for phrasing)
    thresholds     : override default thresholds (dict {event: value})
    truth_t        : (n_lat, n_lon) truth array if available, for accuracy check

    Returns
    -------
    list of alert record dicts
    """
    if thresholds is None:
        thresholds = EXTREME_THRESHOLDS.get(variable, {})

    district_map = build_district_map(lats, lons)
    alerts = []

    for hazard, threshold in thresholds.items():
        # Aggregate per district: max blended value and mean confidence
        districts: dict[str, dict] = {}
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                d = str(district_map[i, j])
                val = float(blended_t[i, j])
                conf = float(confidence_map[i, j])
                if d not in districts:
                    districts[d] = {"vals": [], "confs": [], "lat": float(lat), "lon": float(lon)}
                districts[d]["vals"].append(val)
                districts[d]["confs"].append(conf)

        for district, data in districts.items():
            valid_vals = [v for v in data["vals"] if not (v != v)]  # exclude NaN
            if not valid_vals:
                continue  # entire district is NaN — skip, do not emit false alert
            max_val    = float(max(valid_vals))
            mean_conf  = float(np.nanmean(data["confs"]))
            severity   = _determine_severity(max_val, threshold, mean_conf)
            if severity is None:
                continue

            # Accuracy check if truth available
            accuracy_status = "pending"
            if truth_t is not None:
                # Find district cells in truth
                truth_vals = []
                for i, lat in enumerate(lats):
                    for j, lon in enumerate(lons):
                        if str(district_map[i, j]) == district:
                            truth_vals.append(float(truth_t[i, j]))
                if truth_vals:
                    truth_max = float(np.nanmax(truth_vals))
                    was_crossed = truth_max >= threshold
                    if was_crossed:
                        accuracy_status = "correct"
                    else:
                        accuracy_status = "false_alarm"

            alerts.append({
                "district":       district,
                "hazard":         hazard,
                "variable":       variable,
                "forecast_date":  forecast_date,
                "lead_time_h":    lead_time,
                "forecast_value": round(max_val, 2),
                "threshold":      threshold,
                "confidence_pct": round(mean_conf, 1),
                "severity":       severity,
                "sector":         sector,
                "recommended_action": _recommended_action(hazard, severity, sector),
                "accuracy_status": accuracy_status,
                "generated_at":   datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            })

    return alerts


def alerts_to_dataframe(alerts: list[dict]) -> pd.DataFrame:
    """Convert list of alert dicts to a tidy DataFrame."""
    if not alerts:
        return pd.DataFrame(columns=[
            "district", "hazard", "variable", "forecast_date", "lead_time_h",
            "forecast_value", "threshold", "confidence_pct", "severity",
            "sector", "recommended_action", "accuracy_status", "generated_at",
        ])
    return pd.DataFrame(alerts)


def save_alert_history(new_alerts: list[dict]) -> None:
    """Append new alerts to persistent alert history CSV."""
    if not new_alerts:
        return
    _ALERT_HISTORY_CSV.parent.mkdir(parents=True, exist_ok=True)
    df = alerts_to_dataframe(new_alerts)
    if _ALERT_HISTORY_CSV.exists():
        existing = pd.read_csv(_ALERT_HISTORY_CSV)
        combined = pd.concat([existing, df], ignore_index=True)
    else:
        combined = df
    combined.to_csv(_ALERT_HISTORY_CSV, index=False)


def load_alert_history() -> pd.DataFrame:
    """Load persisted alert history CSV, or return empty DataFrame."""
    if _ALERT_HISTORY_CSV.exists():
        try:
            return pd.read_csv(_ALERT_HISTORY_CSV)
        except Exception:
            pass
    return pd.DataFrame()


def alert_summary_html(alert: dict, show_sector_phrasing: bool = True) -> str:
    """
    Render a single alert as a compact HTML card (for use in Streamlit markdown).
    """
    sev   = alert["severity"]
    icon  = SEVERITY_ICONS.get(sev, "⚪")
    conf  = alert["confidence_pct"]
    val   = alert["forecast_value"]
    thr   = alert["threshold"]
    dist  = alert["district"]
    haz   = alert["hazard"].replace("_", " ").title()
    date_ = alert["forecast_date"]
    rec   = alert["recommended_action"] if show_sector_phrasing else _DEFAULT_PHRASE.get(alert["hazard"], "")
    acc   = alert.get("accuracy_status", "pending")
    acc_str = {"correct": "✅ Confirmed", "false_alarm": "❌ False alarm",
               "pending": "⏳ Pending", "missed_event": "⚠️ Missed"}.get(acc, acc)

    sev_bg = {
        "WATCH":   "rgba(250,204,21,0.15)",
        "WARNING": "rgba(249,115,22,0.2)",
        "SEVERE":  "rgba(239,68,68,0.25)",
    }.get(sev, "rgba(100,116,139,0.15)")
    sev_border = {
        "WATCH":   "#ca8a04",
        "WARNING": "#ea580c",
        "SEVERE":  "#dc2626",
    }.get(sev, "#64748b")
    sev_text = {
        "WATCH":   "#fbbf24",
        "WARNING": "#fb923c",
        "SEVERE":  "#f87171",
    }.get(sev, "#e2e8f0")

    return f"""
<div style="background:{sev_bg}; border-left:3px solid {sev_border};
            border-radius:0.4rem; padding:0.6rem 0.8rem; margin:0.3rem 0;">
  <div style="display:flex; justify-content:space-between; align-items:center;">
    <span style="color:{sev_text}; font-weight:700; font-size:0.85rem;">
      {icon} {sev} — {haz}
    </span>
    <span style="font-size:0.72rem; color:#94a3b8;">{date_}</span>
  </div>
  <div style="font-size:0.82rem; color:#e2e8f0; margin-top:2px;">
    <b>{dist}</b> — {val:.1f} (threshold {thr}) | Confidence: {conf:.0f}%
  </div>
  <div style="font-size:0.75rem; color:#94a3b8; margin-top:2px;">
    {rec}
  </div>
  <div style="font-size:0.72rem; color:#64748b; margin-top:2px;">
    Accuracy: {acc_str}
  </div>
</div>
"""
