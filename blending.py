"""
blending.py
===========
Forecast blending and extreme-weather detection layer.

Blend per grid cell using that cell's region-specific weights → spatially
adaptive blended forecast.  Also computes a naive equal-weight average as
baseline.

Extreme weather
---------------
Threshold-based flags (configurable defaults, override via EXTREME_THRESHOLDS).

Skill comparison
----------------
Returns RMSE/MAE/bias for:
  • each individual source
  • naive average
  • adaptive blend
Broken down per (variable, lead_time, region) so regional gains/regressions
are visible.  Under-performance vs naive average is surfaced, not hidden.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

from scoring import build_region_map, region_id_to_label, REGION_TILES
from weighting import get_weights, get_all_source_weight_maps

# ── configurable extreme thresholds ───────────────────────────────────────────
EXTREME_THRESHOLDS = {
    "rainfall":    {"heavy_rainfall": 50.0},   # mm — 24h equiv
    "temperature": {"heat_wave":      40.0},   # °C
    "wind":        {"high_wind":      17.2},   # m/s (Beaufort 8 gale)
}


# ── blending ──────────────────────────────────────────────────────────────────

def blend_forecast(
    ds: xr.Dataset,
    weight_table: pd.DataFrame,
    regime_series: pd.Series,
    variable: str,
    lead_time: int,
) -> dict[str, np.ndarray]:
    """
    Produce the adaptive-blend and naive-average forecast fields for a single
    (variable, lead_time) combination.

    Parameters
    ----------
    ds           : full Dataset from load_dataset()
    weight_table : output of build_weight_table()
    regime_series: pd.Series indexed by time (classifier output, never true_regime)
    variable     : variable name string
    lead_time    : lead time in hours

    Returns
    -------
    dict with keys:
      'blended'          (n_time, n_lat, n_lon) float
      'naive'            (n_time, n_lat, n_lon) float — equal-weight average
      'truth'            (n_time, n_lat, n_lon) float
      '{source}'         (n_time, n_lat, n_lon) float — per-source forecast
    """
    time_index   = pd.DatetimeIndex(ds.coords["time"].values)
    season_arr   = ds.coords["season"].values
    regime_arr   = regime_series.reindex(time_index).values

    lead_list    = list(ds.coords["lead_time"].values)
    li           = lead_list.index(lead_time)

    n_time = len(time_index)
    n_lat  = len(ds.coords["lat"])
    n_lon  = len(ds.coords["lon"])

    source_cols  = sorted(c for c in weight_table.columns
                          if c not in ["variable", "lead_time", "region",
                                       "season", "regime", "dominant_source"])

    # Load forecast arrays: (n_time, n_lat, n_lon) per source
    # Skip sources that are entirely NaN (e.g. ensemble/AI placeholders in GFS-only mode)
    fc: dict[str, np.ndarray] = {}
    for src in source_cols:
        key = f"{src}_{variable}"
        if key in ds.data_vars:
            arr = ds[key].values[:, :, :, li]
            if not np.all(np.isnan(arr)):   # only keep sources with real data
                fc[src] = arr

    truth = ds[f"truth_{variable}"].values   # (n_time, n_lat, n_lon)
    region_map = build_region_map(ds)         # (n_lat, n_lon)

    blended = np.zeros((n_time, n_lat, n_lon))
    naive   = np.zeros((n_time, n_lat, n_lon))

    # Naive average (equal weights, skip NaN sources)
    avail_sources = list(fc.keys())
    if avail_sources:
        for src in avail_sources:
            naive += fc[src] / len(avail_sources)

    # Adaptive blend: per time step × per grid cell
    # Re-normalise weights to only available (non-NaN) sources
    for t in range(n_time):
        season = season_arr[t]
        regime = regime_arr[t]

        for region_id in np.unique(region_map):
            rlabel    = region_id_to_label(int(region_id))
            cell_mask = (region_map == region_id)   # (n_lat, n_lon) bool
            weights   = get_weights(weight_table, variable, lead_time,
                                    rlabel, season, regime)
            # Keep only sources that have real data and re-normalise
            avail_w = {s: w for s, w in weights.items() if s in fc}
            total_w = sum(avail_w.values())
            if total_w > 0:
                avail_w = {s: w / total_w for s, w in avail_w.items()}
            blend_cell = np.zeros((n_lat, n_lon))
            for src, w in avail_w.items():
                blend_cell += w * fc[src][t]
            blended[t][cell_mask] = blend_cell[cell_mask]

    result = {"blended": blended, "naive": naive, "truth": truth}
    result.update(fc)
    return result


# ── extreme weather flagging ───────────────────────────────────────────────────

def flag_extremes(
    blended: np.ndarray,
    variable: str,
    thresholds: Optional[dict] = None,
) -> dict[str, np.ndarray]:
    """
    Apply threshold-based extreme flags to the blended forecast field.

    Parameters
    ----------
    blended    : (n_time, n_lat, n_lon) blended forecast
    variable   : variable name
    thresholds : dict {event_name: threshold_value}; defaults to EXTREME_THRESHOLDS

    Returns
    -------
    dict {event_name: (n_time, n_lat, n_lon) bool array}
    """
    if thresholds is None:
        thresholds = EXTREME_THRESHOLDS.get(variable, {})
    flags: dict[str, np.ndarray] = {}
    for event, thresh in thresholds.items():
        flags[event] = blended >= thresh
    return flags


# ── skill comparison ───────────────────────────────────────────────────────────

def compute_blend_skill(
    blend_result: dict[str, np.ndarray],
    ds: xr.Dataset,
    variable: str,
    lead_time: int,
) -> pd.DataFrame:
    """
    Compare RMSE/MAE/bias for blended, naive, and each source against truth.

    Returns a DataFrame with columns:
        source | lead_time | region | rmse | mae | bias
    where source includes 'blended', 'naive', and each model source.
    Under-performing buckets (blended RMSE > naive RMSE) are flagged with
    a 'beats_naive' boolean column so they are never hidden.
    """
    truth = blend_result["truth"]
    region_map = build_region_map(ds)
    records = []

    comparison_keys = ["blended", "naive"] + [
        k for k in blend_result if k not in ("blended", "naive", "truth")
    ]

    for region_id in np.unique(region_map):
        rlabel    = region_id_to_label(int(region_id))
        cell_mask = (region_map == region_id)

        for key in comparison_keys:
            if key not in blend_result:
                continue
            pred  = blend_result[key]
            err   = (pred - truth)[:, cell_mask].ravel()
            # Use nanmean so that NaN source values (e.g. ai_rainfall) don't
            # propagate NaN into the skill scores.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                rmse  = float(np.sqrt(np.nanmean(err ** 2)))
                mae   = float(np.nanmean(np.abs(err)))
                bias  = float(np.nanmean(err))
            records.append({
                "source":    key,
                "lead_time": lead_time,
                "region":    rlabel,
                "rmse":      rmse,
                "mae":       mae,
                "bias":      bias,
            })

    df = pd.DataFrame(records)

    # Flag under-performance
    naive_rmse = df[df["source"] == "naive"].set_index("region")["rmse"]
    blend_rmse = df[df["source"] == "blended"].set_index("region")["rmse"]
    beats = {}
    for reg in naive_rmse.index:
        if reg in blend_rmse.index:
            b = blend_rmse[reg]
            n = naive_rmse[reg]
            # If either is NaN, we can't fairly judge; mark as None
            beats[reg] = bool(b <= n) if (not np.isnan(b) and not np.isnan(n)) else None
        else:
            beats[reg] = None

    df["beats_naive"] = df.apply(
        lambda r: beats.get(r["region"]) if r["source"] == "blended" else None,
        axis=1,
    )
    return df.sort_values(["region", "source"]).reset_index(drop=True)


def compute_blend_skill_all_leads(
    ds: xr.Dataset,
    weight_table: pd.DataFrame,
    regime_series: pd.Series,
    variables: Optional[list[str]] = None,
    lead_times: Optional[list[int]] = None,
) -> pd.DataFrame:
    """
    Run blend + skill comparison for all (variable, lead_time) combinations.
    Returns a combined DataFrame with a 'variable' column prepended.
    Prints a diagnostic for any bucket where blended < naive.
    """
    if variables is None:
        variables = [v.replace("truth_", "") for v in ds.data_vars
                     if v.startswith("truth_")]
    if lead_times is None:
        lead_times = list(ds.coords["lead_time"].values)

    all_records = []
    for var in variables:
        for lt in lead_times:
            result = blend_forecast(ds, weight_table, regime_series, var, int(lt))
            skill  = compute_blend_skill(result, ds, var, int(lt))
            skill.insert(0, "variable", var)
            all_records.append(skill)

            # Surface underperformance (only flag when beats_naive is explicitly False,
            # not when it is None/NaN — which means data was unavailable)
            blend_rows = skill[skill["source"] == "blended"]
            bad = blend_rows[blend_rows["beats_naive"].eq(False)]
            if not bad.empty:
                print(
                    f"[WARN]  DIAGNOSTIC: Blended underperforms naive average for "
                    f"({var}, lead={lt}h) in regions: "
                    f"{bad['region'].tolist()}"
                )

    return pd.concat(all_records, ignore_index=True)


# ── Standalone smoke test ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))

    from data.synthetic import load_dataset
    from regime import fit_and_classify
    from scoring import compute_skill_scores
    from weighting import build_weight_table

    print("Loading dataset...")
    ds = load_dataset()
    print("Classifying regimes...")
    _, regimes = fit_and_classify(ds)
    print("Scoring sources...")
    skill_df = compute_skill_scores(ds, regimes)
    print("Building weights...")
    wt = build_weight_table(skill_df)

    print("\nBlending forecasts for all variables and lead times...")
    combined_skill = compute_blend_skill_all_leads(ds, wt, regimes)
    print("\nSkill comparison summary (mean RMSE per source/variable/lead_time):")
    pivot = combined_skill.groupby(["variable", "lead_time", "source"])["rmse"].mean().unstack("source")
    print(pivot.round(3).to_string())

    # Verify blended beats at least some individual sources
    for (var, lt), grp in combined_skill.groupby(["variable", "lead_time"]):
        blended_rmse = grp[grp["source"] == "blended"]["rmse"].mean()
        naive_rmse   = grp[grp["source"] == "naive"]["rmse"].mean()
        best_src     = grp[~grp["source"].isin(["blended", "naive"])]["rmse"].mean()
        print(f"\n{var} @ {lt}h | blended={blended_rmse:.3f}  naive={naive_rmse:.3f}  "
              f"mean_source={best_src:.3f}  beats_naive={blended_rmse <= naive_rmse}")

    # Check extreme flags
    result = blend_forecast(ds, wt, regimes, "rainfall", 24)
    flags  = flag_extremes(result["blended"], "rainfall")
    heavy  = flags.get("heavy_rainfall", np.array([]))
    print(f"\nHeavy rainfall flags — total: {heavy.sum()}, "
          f"fraction: {heavy.mean():.4f}")

    result_t = blend_forecast(ds, wt, regimes, "temperature", 24)
    flags_t  = flag_extremes(result_t["blended"], "temperature")
    hw = flags_t.get("heat_wave", np.array([]))
    print(f"Heat wave flags      — total: {hw.sum()}, "
          f"fraction: {hw.mean():.4f}")
    print("\n[OK] Blending layer completed successfully.")
