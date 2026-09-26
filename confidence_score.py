"""
confidence_score.py
===================
Forecast Confidence Score (Feature 4).

For every forecast point (grid cell, variable, lead time), computes:

  1. Model agreement %
     Derived from how close the three raw source forecasts are to each other.
     Computed as inverse of normalised spread:
       agreement = 1 - (std_across_sources / (mean_abs_value + epsilon))
     Clipped to [0, 1].

  2. Confidence %
     Combined score using model agreement AND the historical skill of the
     involved sources for that bucket.
     confidence = 0.6 * agreement + 0.4 * skill_score
     where skill_score = 1 - normalised_mean_rmse
     (agreement alone ≠ confidence — this is intentional)

  3. Expected range [min, max]
     Derived from the spread across the three sources (±1.5σ).
     Falls back to historical error distribution for the bucket if spread = 0.

Design constraints
------------------
* Does NOT modify blending.py, scoring.py, or weighting.py.
* Reads from existing blend_result and skill_df as inputs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

_EPS = 1e-6  # prevents divide-by-zero in normalisation


# ── Agreement score ────────────────────────────────────────────────────────────

def compute_model_agreement(
    source_arrays: dict[str, np.ndarray],
    time_idx: int,
) -> np.ndarray:
    """
    Compute per-cell model agreement percentage at a given time step.

    agreement = (1 - normalised_std) * 100
    where normalised_std = std(sources) / (mean_abs(sources) + eps)

    High agreement (close to 100%) = sources closely aligned.
    Low agreement (close to 0%)    = sources strongly disagree.

    Parameters
    ----------
    source_arrays : {src: (n_time, n_lat, n_lon)}
    time_idx      : which time step

    Returns
    -------
    (n_lat, n_lon) array in [0, 100]
    """
    avail = {s: arr[time_idx]
             for s, arr in source_arrays.items()
             if not np.all(np.isnan(arr[time_idx]))}

    if len(avail) < 2:
        # Only one source — perfect agreement by default
        ref = next(iter(avail.values())) if avail else np.zeros((1, 1))
        return np.full_like(ref, 100.0)

    stack = np.stack(list(avail.values()), axis=0)   # (n_src, H, W)
    std   = np.nanstd(stack, axis=0)
    mean_abs = np.nanmean(np.abs(stack), axis=0) + _EPS
    norm_std = std / mean_abs
    agreement = np.clip(1.0 - norm_std, 0.0, 1.0) * 100.0
    return agreement


# ── Skill score (normalised) ───────────────────────────────────────────────────

def _skill_score_for_bucket(
    skill_df: pd.DataFrame,
    variable: str,
    lead_time: int,
    season: str,
    regime: str,
    sources: list[str],
) -> float:
    """
    Compute a normalised skill score for a bucket from the existing skill table.

    Returns a value in [0, 1]: 1 = perfect (zero RMSE), 0 = very poor skill.
    Uses the mean inverse-RMSE across available sources (higher skill = higher score).
    """
    if skill_df.empty:
        return 0.5  # neutral default if no skill data

    mask = (
        (skill_df["variable"]  == variable)  &
        (skill_df["lead_time"] == lead_time) &
        (skill_df["season"]    == season)    &
        (skill_df["regime"]    == regime)    &
        (skill_df["source"].isin(sources))
    )
    grp = skill_df[mask]
    if grp.empty:
        return 0.5

    rmses = grp.groupby("source")["rmse"].mean()
    if rmses.empty or rmses.isna().all():
        return 0.5

    # Normalise: 0 RMSE → score 1.0; large RMSE → score near 0
    # Use a soft normalisation: score_i = 1 / (1 + rmse_i)
    scores = 1.0 / (1.0 + rmses.values)
    return float(np.nanmean(scores))


# ── Combined confidence ────────────────────────────────────────────────────────

def compute_confidence_map(
    agreement_map: np.ndarray,
    skill_score: float,
    agreement_weight: float = 0.6,
    skill_weight: float = 0.4,
) -> np.ndarray:
    """
    Combine per-cell agreement % with historical skill score into a
    combined confidence %.

    confidence = agreement_weight * agreement + skill_weight * (skill_score * 100)

    Parameters
    ----------
    agreement_map : (n_lat, n_lon) in [0, 100]
    skill_score   : scalar in [0, 1]
    """
    skill_pct = skill_score * 100.0
    confidence = agreement_weight * agreement_map + skill_weight * skill_pct
    return np.clip(confidence, 0.0, 100.0)


# ── Expected range ─────────────────────────────────────────────────────────────

def compute_expected_range(
    source_arrays: dict[str, np.ndarray],
    time_idx: int,
    skill_df: pd.DataFrame,
    variable: str,
    lead_time: int,
    season: str,
    regime: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-cell plausible [min, max] range.

    Primary: blended_value ± 1.5 * std(sources)
    Fallback (if std == 0): use historical mean error from skill_df

    Returns
    -------
    (range_low, range_high) each (n_lat, n_lon)
    """
    avail = {s: arr[time_idx]
             for s, arr in source_arrays.items()
             if not np.all(np.isnan(arr[time_idx]))}

    if not avail:
        z = np.zeros((1, 1))
        return z, z

    stack = np.stack(list(avail.values()), axis=0)  # (n_src, H, W)
    mean_fc = np.nanmean(stack, axis=0)
    spread  = np.nanstd(stack, axis=0)

    # Historical error fallback — mean MAE for this bucket
    fallback_error = 0.0
    if not skill_df.empty:
        mask = (
            (skill_df["variable"]  == variable) &
            (skill_df["lead_time"] == lead_time) &
            (skill_df["season"]    == season)    &
            (skill_df["regime"]    == regime)
        )
        grp = skill_df[mask]
        if not grp.empty:
            fallback_error = float(grp["mae"].mean())

    effective_spread = np.where(spread < _EPS, fallback_error, spread * 1.5)
    range_low  = mean_fc - effective_spread
    range_high = mean_fc + effective_spread
    return range_low, range_high


# ── Point-level readout (for dashboard card) ───────────────────────────────────

def compute_confidence_readout(
    source_arrays: dict[str, np.ndarray],
    blended_array: np.ndarray,
    time_idx: int,
    lat_idx: int,
    lon_idx: int,
    skill_df: pd.DataFrame,
    variable: str,
    lead_time: int,
    season: str,
    regime: str,
) -> dict[str, Any]:
    """
    Full confidence readout for a single grid cell — used in the dashboard card.

    Returns
    -------
    dict:
      'blended_value'  : float
      'agreement_pct'  : float [0,100]
      'confidence_pct' : float [0,100]
      'range_low'      : float
      'range_high'     : float
      'source_values'  : {src: float}
      'explanation'    : markdown string
    """
    avail = {s: arr[time_idx, lat_idx, lon_idx]
             for s, arr in source_arrays.items()
             if not np.all(np.isnan(arr[time_idx]))}
    avail = {s: v for s, v in avail.items() if not np.isnan(v)}

    blended_val = float(blended_array[time_idx, lat_idx, lon_idx])
    sources_list = list(avail.keys())

    # Scalar agreement
    if len(avail) >= 2:
        vals = np.array(list(avail.values()))
        std_ = float(np.std(vals))
        mean_abs_ = float(np.mean(np.abs(vals))) + _EPS
        agreement_pct = float(np.clip((1.0 - std_ / mean_abs_) * 100, 0, 100))
    else:
        agreement_pct = 100.0

    skill_score = _skill_score_for_bucket(
        skill_df, variable, lead_time, season, regime, sources_list
    )
    confidence_pct = float(
        np.clip(0.6 * agreement_pct + 0.4 * skill_score * 100, 0, 100)
    )

    # Range
    fallback_error = 0.0
    if not skill_df.empty:
        m = (
            (skill_df["variable"]  == variable) &
            (skill_df["lead_time"] == lead_time) &
            (skill_df["season"]    == season)    &
            (skill_df["regime"]    == regime)
        )
        grp = skill_df[m]
        fallback_error = float(grp["mae"].mean()) if not grp.empty else 0.0

    if len(avail) >= 2:
        vals = np.array(list(avail.values()))
        spread = float(np.std(vals))
        effective = spread * 1.5 if spread > _EPS else fallback_error
    else:
        effective = fallback_error

    range_low  = round(blended_val - effective, 2)
    range_high = round(blended_val + effective, 2)

    # Build explanation
    explanation = _build_confidence_explanation(
        avail, blended_val, agreement_pct, confidence_pct,
        range_low, range_high, variable
    )

    return {
        "blended_value":  round(blended_val, 2),
        "agreement_pct":  round(agreement_pct, 1),
        "confidence_pct": round(confidence_pct, 1),
        "range_low":      range_low,
        "range_high":     range_high,
        "source_values":  {s: round(v, 2) for s, v in avail.items()},
        "explanation":    explanation,
    }


def _build_confidence_explanation(
    source_values: dict[str, float],
    blended: float,
    agreement_pct: float,
    confidence_pct: float,
    range_low: float,
    range_high: float,
    variable: str,
) -> str:
    """Build a dynamic explanation string matching the required format."""
    unit_map = {"rainfall": "mm", "temperature": "°C", "wind": "m/s"}
    unit = unit_map.get(variable, "")
    var_label_map = {"rainfall": "Rain forecast", "temperature": "Temp forecast",
                     "wind": "Wind forecast"}
    var_label = var_label_map.get(variable, "Forecast")

    SOURCE_FRIENDLY = {"nwp": "NWP", "ensemble": "Ensemble", "ai": "AI"}
    src_str = ", ".join(
        f"{SOURCE_FRIENDLY.get(s, s.upper())}={v:.1f}{unit}"
        for s, v in sorted(source_values.items())
    )

    if agreement_pct >= 75:
        why = f"models are closely aligned, so confidence is high"
    elif agreement_pct >= 50:
        why = f"moderate source spread introduces uncertainty"
    else:
        why = (
            f"large disagreement between sources — "
            f"treat this forecast with caution"
        )

    lines = [
        f"**{var_label}:** {blended:.1f}{unit}",
        f"**Confidence:** {confidence_pct:.0f}%",
        f"**Model agreement:** {agreement_pct:.0f}%",
        f"**Expected range:** {range_low:.1f}–{range_high:.1f}{unit}",
        f"**Why:** {src_str} — {why}.",
    ]
    return "\n".join(lines)


# ── Spatial maps for dashboard ─────────────────────────────────────────────────

def compute_spatial_confidence(
    source_arrays: dict[str, np.ndarray],
    blended_array: np.ndarray,
    time_idx: int,
    skill_df: pd.DataFrame,
    variable: str,
    lead_time: int,
    season: str,
    regime: str,
) -> dict[str, np.ndarray]:
    """
    Compute full spatial confidence maps for a given time step.

    Returns
    -------
    dict:
      'agreement'   : (H, W) float [0,100]
      'confidence'  : (H, W) float [0,100]
      'range_low'   : (H, W) float
      'range_high'  : (H, W) float
    """
    avail = {s: arr for s, arr in source_arrays.items()
             if not np.all(np.isnan(arr[time_idx]))}

    agreement = compute_model_agreement(avail, time_idx)

    skill_score = _skill_score_for_bucket(
        skill_df, variable, lead_time, season, regime, list(avail.keys())
    )
    confidence = compute_confidence_map(agreement, skill_score)

    range_low, range_high = compute_expected_range(
        avail, time_idx, skill_df, variable, lead_time, season, regime
    )

    return {
        "agreement":  agreement,
        "confidence": confidence,
        "range_low":  range_low,
        "range_high": range_high,
    }
