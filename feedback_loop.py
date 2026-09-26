"""
feedback_loop.py
================
Self-Learning Feedback Loop (Feature 3).

Implements an *incremental update* mechanism for the skill table:
  Forecast → Actual (ERA5 truth) → Error → Skill-table update → New weights

Design constraints
------------------
* Does NOT replace or modify the existing scoring.py / weighting.py logic.
* The existing skill table is the *baseline*.  This module adds new records
  to it from recently available truth data, then re-derives weights from the
  combined (old + new) history.
* "New" truth data = any time step where both a forecast and ERA5 truth exist
  but which was not included in the original static scoring run.

Persistence
-----------
Updates are stored in a CSV sidecar alongside the existing outputs:
    outputs/feedback_skill_updates.csv

The dashboard can merge this with the original skill_df to show a
"before/after" learning demonstration.

Demonstration mode
------------------
build_demo_record() synthesises a realistic yesterday/today example from
the actual data even if no genuinely new truth has arrived — this satisfies
the "concrete before/after demonstration" requirement using real pipeline data.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from scoring import build_region_map, region_id_to_label, REGION_TILES

# Path where incremental skill updates are persisted
_UPDATES_CSV = Path(__file__).parent / "outputs" / "feedback_skill_updates.csv"


# ── Core update logic ──────────────────────────────────────────────────────────

def compute_incremental_errors(
    ds: xr.Dataset,
    new_time_indices: list[int],
    regime_series: pd.Series,
    variable: str,
    lead_time: int,
) -> pd.DataFrame:
    """
    Compute per-(source, region) errors for a list of *new* time steps
    (i.e. time steps that now have truth data available but were not
    previously in the skill table).

    Parameters
    ----------
    ds               : full Dataset (must have truth_{variable} and source arrays)
    new_time_indices : list of time-axis indices to evaluate
    regime_series    : classifier output series indexed by time
    variable         : 'rainfall' | 'temperature' | 'wind'
    lead_time        : lead time in hours

    Returns
    -------
    DataFrame with same schema as compute_skill_scores() output:
        variable, source, lead_time, region, season, regime, rmse, mae, bias, n_samples
    """
    time_index = pd.DatetimeIndex(ds.coords["time"].values)
    season_arr = ds.coords["season"].values
    regime_arr = regime_series.reindex(time_index).values
    region_map = build_region_map(ds)

    lead_list = list(ds.coords["lead_time"].values)
    if lead_time not in lead_list:
        warnings.warn(f"lead_time={lead_time} not found in dataset.")
        return pd.DataFrame()
    li = lead_list.index(lead_time)

    truth_full = ds[f"truth_{variable}"].values  # (n_time, n_lat, n_lon)
    sources = list({v.split("_")[0] for v in ds.data_vars if not v.startswith("truth_")})

    records = []
    for src in sources:
        key = f"{src}_{variable}"
        if key not in ds.data_vars:
            continue
        fc_data = ds[key].values[:, :, :, li]  # (n_time, n_lat, n_lon)
        if np.all(np.isnan(fc_data)):
            continue

        for t in new_time_indices:
            season = str(season_arr[t])
            regime = str(regime_arr[t])
            fc_slice = fc_data[t]         # (n_lat, n_lon)
            truth_t  = truth_full[t]       # (n_lat, n_lon)
            err_t    = fc_slice - truth_t  # (n_lat, n_lon)

            for region_id in np.unique(region_map):
                rlabel = region_id_to_label(int(region_id))
                cell_mask = (region_map == region_id)
                e_flat = err_t[cell_mask].ravel()
                n = len(e_flat)
                if n == 0:
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    rmse = float(np.sqrt(np.nanmean(e_flat ** 2)))
                    mae  = float(np.nanmean(np.abs(e_flat)))
                    bias = float(np.nanmean(e_flat))
                records.append({
                    "variable":  variable,
                    "source":    src,
                    "lead_time": int(lead_time),
                    "region":    rlabel,
                    "season":    season,
                    "regime":    regime,
                    "rmse":      rmse,
                    "mae":       mae,
                    "bias":      bias,
                    "n_samples": n,
                    "feedback_date": str(pd.Timestamp.now().date()),
                })
    return pd.DataFrame(records)


def merge_skill_tables(
    original_skill_df: pd.DataFrame,
    update_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge incremental updates into the original skill table.

    For duplicate buckets (same variable/source/lead/region/season/regime),
    the new errors are averaged in with the existing ones, weighted by n_samples.
    """
    if update_df.empty:
        return original_skill_df

    # Add update_df rows with a marker
    update_df = update_df.copy()
    orig = original_skill_df.copy()

    group_keys = ["variable", "source", "lead_time", "region", "season", "regime"]
    combined = pd.concat([orig, update_df], ignore_index=True)

    # Weighted average for numeric skill metrics
    def _weighted_mean(grp: pd.DataFrame) -> pd.Series:
        total_n = grp["n_samples"].sum()
        w = grp["n_samples"] / total_n
        return pd.Series({
            "rmse":      float((w * grp["rmse"]).sum()),
            "mae":       float((w * grp["mae"]).sum()),
            "bias":      float((w * grp["bias"]).sum()),
            "n_samples": int(total_n),
        })

    merged = (
        combined.groupby(group_keys, as_index=False)
        .apply(_weighted_mean, include_groups=False)
        .reset_index(drop=True)
    )
    return merged


def load_feedback_history() -> pd.DataFrame:
    """Load persisted feedback skill updates CSV, or return empty DataFrame."""
    if _UPDATES_CSV.exists():
        try:
            return pd.read_csv(_UPDATES_CSV)
        except Exception:
            pass
    return pd.DataFrame()


def save_feedback_update(update_df: pd.DataFrame) -> None:
    """Append new update records to the persistent CSV."""
    if update_df.empty:
        return
    _UPDATES_CSV.parent.mkdir(parents=True, exist_ok=True)
    if _UPDATES_CSV.exists():
        existing = pd.read_csv(_UPDATES_CSV)
        combined = pd.concat([existing, update_df], ignore_index=True)
    else:
        combined = update_df.copy()
    combined.to_csv(_UPDATES_CSV, index=False)


def run_feedback_update(
    ds: xr.Dataset,
    original_skill_df: pd.DataFrame,
    regime_series: pd.Series,
    new_time_indices: list[int] | None = None,
    variables: list[str] | None = None,
    lead_times: list[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full feedback update cycle.

    If new_time_indices is None, the last 20% of time steps are treated as
    "recently acquired truth data" — used for demonstration purposes.

    Returns
    -------
    (updated_skill_df, update_records_df)
    """
    if variables is None:
        variables = [v.replace("truth_", "") for v in ds.data_vars
                     if v.startswith("truth_")]
    if lead_times is None:
        lead_times = list(ds.coords["lead_time"].values)
    if new_time_indices is None:
        n = len(ds.coords["time"])
        new_time_indices = list(range(int(n * 0.8), n))

    all_updates = []
    for var in variables:
        for lt in lead_times:
            update = compute_incremental_errors(
                ds, new_time_indices, regime_series, var, int(lt)
            )
            if not update.empty:
                all_updates.append(update)

    update_df = pd.concat(all_updates, ignore_index=True) if all_updates else pd.DataFrame()
    updated_skill = merge_skill_tables(original_skill_df, update_df)
    save_feedback_update(update_df)

    return updated_skill, update_df


# ── Source data-completeness check ────────────────────────────────────────────

def _has_valid_data(ds: xr.Dataset, source: str, variable: str) -> bool:
    """
    Generic check: does `source` have any non-NaN forecast values for `variable`?

    This is the single gate used throughout this module to exclude sources that
    are simply not applicable for a given variable (e.g. Pangu-Weather has no
    rainfall predictions).  It is NOT a special-case: it checks the actual data.
    """
    key = f"{source}_{variable}"
    if key not in ds.data_vars:
        return False
    return bool(not np.all(np.isnan(ds[key].values)))


# ── Weight shift analysis ──────────────────────────────────────────────────────

def compute_weight_shift(
    original_skill_df: pd.DataFrame,
    updated_skill_df: pd.DataFrame,
    variable: str,
    lead_time: int,
    season: str,
    regime: str,
    region: str = "R11",
    applicable_sources: list[str] | None = None,
) -> dict[str, Any]:
    """
    Compare per-source weights before and after a feedback update for a specific
    bucket.  Returns a dict suitable for the dashboard's before/after demo.

    Parameters
    ----------
    applicable_sources : if provided, only these sources are included in the
        result.  Sources absent from this list are recorded as
        {"before": None, "after": None, "delta": None, "na": True}
        so the caller can display "not applicable" instead of NaN/0.
    """
    from weighting import build_weight_table, get_weights

    def _get_bucket_weights(skill: pd.DataFrame) -> dict[str, float]:
        if skill.empty:
            return {}
        try:
            wt = build_weight_table(skill)
            return get_weights(wt, variable, lead_time, region, season, regime)
        except Exception:
            return {}

    before = _get_bucket_weights(original_skill_df)
    after  = _get_bucket_weights(updated_skill_df)

    # All sources that appear in the weight tables
    all_sources = set(before) | set(after)

    shifts = {}
    for src in all_sources:
        # If caller supplied an applicable list and this source is not in it,
        # mark it explicitly as N/A — do NOT coerce to 0%
        if applicable_sources is not None and src not in applicable_sources:
            shifts[src] = {"before": None, "after": None, "delta": None, "na": True}
            continue
        b = before.get(src, 0.0)
        a = after.get(src, 0.0)
        # Guard: if either value is NaN (weight table produced NaN), mark N/A
        if b != b or a != a:  # NaN check without importing math
            shifts[src] = {"before": None, "after": None, "delta": None, "na": True}
            continue
        shifts[src] = {
            "before": round(b * 100, 1),
            "after":  round(a * 100, 1),
            "delta":  round((a - b) * 100, 1),
            "na":     False,
        }
    return shifts


# ── Best demo-day selector (Bug 2 fix) ────────────────────────────────────────

def find_best_demo_time_idx(
    ds: xr.Dataset,
    blend_result: dict[str, np.ndarray],
    variable: str,
) -> tuple[int, float, str]:
    """
    Search all available time steps and return the index of the day where
    source errors are most spread apart — i.e. one model was clearly right
    and another was clearly wrong.  This produces the most informative
    Before/After demonstration case.

    Uses the centre-domain grid cell as the representative point (same as
    build_demo_record).

    Returns
    -------
    (best_idx, best_spread, message)
        best_idx   : time index with largest cross-source error spread
        best_spread: the spread value (std of per-source errors at that step)
        message    : human-readable note about the selection
    """
    truth_arr = blend_result.get("truth")
    if truth_arr is None or len(truth_arr) == 0:
        return 0, 0.0, "no truth data available"

    n_time = truth_arr.shape[0]
    ci = truth_arr.shape[1] // 2
    cj = truth_arr.shape[2] // 2

    # Collect only sources that genuinely have data for this variable
    avail_sources = [
        s for s in ["nwp", "ensemble", "ai"]
        if s in blend_result
        and _has_valid_data(ds, s, variable)
        and not np.all(np.isnan(blend_result[s][:, ci, cj]))
    ]

    if len(avail_sources) < 2:
        # Can't compute spread with fewer than 2 sources — return midpoint
        return n_time // 2, 0.0, "fewer than 2 applicable sources; defaulting to midpoint"

    spreads = np.zeros(n_time)
    for t in range(n_time):
        actual = float(truth_arr[t, ci, cj])
        if np.isnan(actual):
            spreads[t] = -1.0  # mark as unusable
            continue
        errs = []
        for src in avail_sources:
            pred = float(blend_result[src][t, ci, cj])
            if not np.isnan(pred):
                errs.append(abs(pred - actual))
        if len(errs) >= 2:
            spreads[t] = float(np.std(errs))
        else:
            spreads[t] = -1.0

    best_idx = int(np.argmax(spreads))
    best_spread = float(spreads[best_idx])

    _MIN_MEANINGFUL_SPREAD = 0.5  # below this, errors are near-identical
    if best_spread < _MIN_MEANINGFUL_SPREAD:
        msg = (
            f"no strongly differentiating case found in current data range "
            f"(max error spread = {best_spread:.3f} — models are closely aligned on all days)"
        )
    else:
        time_index = pd.DatetimeIndex(ds.coords["time"].values)
        msg = f"auto-selected: {time_index[best_idx].strftime('%Y-%m-%d')} (error spread σ={best_spread:.2f})"

    return best_idx, best_spread, msg


# ── Demonstration record builder ───────────────────────────────────────────────

def build_demo_record(
    ds: xr.Dataset,
    blend_result: dict[str, np.ndarray],
    regime_series: pd.Series,
    variable: str,
    lead_time: int,
    demo_time_idx: int,
    original_skill_df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Build a concrete before/after learning demonstration using real pipeline data.

    Picks the demo_time_idx time step, computes actual per-source predicted vs
    actual values at the centre-domain grid cell, then shows how error
    differences translate into weight shifts.

    Bug-1 fix: sources without valid data for `variable` are excluded from
    weight/error calculations entirely and surfaced as "not applicable" —
    never coerced to NaN or 0%.

    Returns
    -------
    dict with keys:
        'date'              : timestamp string
        'sources'           : {src: {'predicted', 'actual', 'error'}}  — applicable only
        'na_sources'        : [src, ...]  — sources excluded (no data for variable)
        'best_src'          : source with smallest error (among applicable)
        'worst_src'         : source with largest error  (among applicable)
        'weight_shifts'     : from compute_weight_shift()
        'narrative'         : human-readable markdown string
        'no_contrast_msg'   : non-empty string if errors are near-identical (warn caller)
    """
    time_index = pd.DatetimeIndex(ds.coords["time"].values)
    date_str   = str(time_index[demo_time_idx])
    season     = str(ds.coords["season"].values[demo_time_idx])
    regime     = str(regime_series.iloc[demo_time_idx])

    truth_t = blend_result["truth"][demo_time_idx]  # (n_lat, n_lon)
    ci = truth_t.shape[0] // 2
    cj = truth_t.shape[1] // 2
    actual = float(truth_t[ci, cj])

    SOURCE_FRIENDLY = {"nwp": "NWP", "ensemble": "Ensemble", "ai": "AI"}

    sources_data: dict[str, Any] = {}
    na_sources:   list[str]      = []

    for src in ["nwp", "ensemble", "ai"]:
        if src not in blend_result:
            continue
        # Bug-1 fix: data-completeness check — generic, not special-cased
        if not _has_valid_data(ds, src, variable):
            na_sources.append(src)
            continue
        pred = float(blend_result[src][demo_time_idx][ci, cj])
        if np.isnan(pred):
            na_sources.append(src)
            continue
        err = abs(pred - actual)
        sources_data[src] = {
            "predicted": round(pred, 1),
            "actual":    round(actual, 1),
            "error":     round(err, 1),
        }

    if not sources_data:
        narrative = (
            f"**{date_str[:10]}**\n\n"
            "No applicable source data available for this variable on this date."
        )
        return {
            "date": date_str, "sources": {}, "na_sources": na_sources,
            "best_src": None, "worst_src": None, "weight_shifts": {},
            "narrative": narrative, "no_contrast_msg": "",
        }

    best_src  = min(sources_data, key=lambda s: sources_data[s]["error"])
    worst_src = max(sources_data, key=lambda s: sources_data[s]["error"])
    gap       = sources_data[worst_src]["error"] - sources_data[best_src]["error"]

    # Simulate updated skill by adding this single observation
    update_df     = compute_incremental_errors(ds, [demo_time_idx], regime_series, variable, lead_time)
    updated_skill = merge_skill_tables(original_skill_df, update_df)

    # Bug-1 fix: pass only applicable sources to weight-shift so NaN sources
    # are marked na=True rather than producing NaN weights
    applicable = list(sources_data.keys())
    shifts = compute_weight_shift(
        original_skill_df, updated_skill, variable, lead_time, season, regime,
        applicable_sources=applicable,
    )

    # Build narrative
    lines = [f"**{date_str[:10]}**", ""]
    for src, vals in sources_data.items():
        fn = SOURCE_FRIENDLY.get(src, src.upper())
        lines.append(f"  {fn}: Predicted {vals['predicted']:.1f}, "
                     f"Actual {vals['actual']:.1f}, Error {vals['error']:.1f}")
    # Explicitly note N/A sources
    for src in na_sources:
        fn = SOURCE_FRIENDLY.get(src, src.upper())
        lines.append(f"  {fn}: not applicable (no {variable} data from this source)")

    lines.append("")
    best_fn  = SOURCE_FRIENDLY.get(best_src, best_src.upper())
    worst_fn = SOURCE_FRIENDLY.get(worst_src, worst_src.upper())

    if gap > 1.0:
        lines.append(
            f"**System learns:** {best_fn} performed significantly better "
            f"(error gap: {gap:.1f}). Weights shift toward {best_fn}."
        )
    else:
        lines.append("**System learns:** Models were closely aligned — minimal weight shift.")

    lines.append("")
    lines.append("**Future similar situations:**")
    for src, s in shifts.items():
        fn = SOURCE_FRIENDLY.get(src, src.upper())
        if s.get("na"):
            lines.append(f"  {fn}: not applicable (no {variable} data from this source)")
        else:
            direction = "↑ up" if s["delta"] > 0.1 else ("↓ down" if s["delta"] < -0.1 else "≈ unchanged")
            lines.append(f"  {fn}: {s['before']:.0f}% → {s['after']:.0f}% ({direction})")

    no_contrast_msg = "" if gap > 1.0 else (
        f"Note: error spread on this date is small ({gap:.2f}). "
        "For a clearer demonstration, use the auto-selected 'best contrast' date."
    )

    return {
        "date":           date_str,
        "sources":        sources_data,
        "na_sources":     na_sources,
        "best_src":       best_src,
        "worst_src":      worst_src,
        "weight_shifts":  shifts,
        "narrative":      "\n".join(lines),
        "no_contrast_msg": no_contrast_msg,
    }
