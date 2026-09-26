"""
scoring.py
==========
Skill-scoring layer.

For every (variable, source, lead_time, region, season, regime) combination,
compute RMSE, MAE, and bias against the truth field, averaged over all
matching historical time steps.

Region definition
-----------------
The domain is divided into a REGION_TILES × REGION_TILES grid of coarse
sub-regions (tiles).  Each grid cell belongs to exactly one region.
Weights must vary by region — this is enforced by the scoring granularity.

Contract
--------
* `regime` column always comes from the classifier's output (a pd.Series
  passed in), NEVER from the synthetic data's hidden `true_regime` field.
* Returns a tidy pandas DataFrame with columns:
      variable | source | lead_time | region | season | regime | rmse | mae | bias | n_samples
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

# ── region tiling ──────────────────────────────────────────────────────────────
REGION_TILES = 4  # splits each axis → 4×4 = 16 coarse regions

def build_region_map(ds: xr.Dataset) -> np.ndarray:
    """
    Return an (n_lat, n_lon) integer array of region IDs 0 … (REGION_TILES²-1).

    Region IDs increase left-to-right, bottom-to-top (matches NetCDF convention).
    """
    n_lat = len(ds.coords["lat"])
    n_lon = len(ds.coords["lon"])
    lat_bins = np.floor(np.linspace(0, REGION_TILES, n_lat, endpoint=False)).astype(int)
    lon_bins = np.floor(np.linspace(0, REGION_TILES, n_lon, endpoint=False)).astype(int)
    lon_bins = np.clip(lon_bins, 0, REGION_TILES - 1)
    lat_bins = np.clip(lat_bins, 0, REGION_TILES - 1)
    # 2-D region map
    region_map = (lat_bins[:, None] * REGION_TILES + lon_bins[None, :])
    return region_map   # shape (n_lat, n_lon), values 0..15


def region_id_to_label(region_id: int) -> str:
    row = region_id // REGION_TILES
    col = region_id % REGION_TILES
    return f"R{row}{col}"   # e.g. R00, R01, … R33


# ── core scoring ───────────────────────────────────────────────────────────────

def compute_skill_scores(
    ds: xr.Dataset,
    regime_series: pd.Series,
    variables: Optional[list[str]] = None,
    sources: Optional[list[str]] = None,
    lead_times: Optional[list[int]] = None,
) -> pd.DataFrame:
    """
    Compute per-(variable, source, lead_time, region, season, regime) skill metrics.

    Parameters
    ----------
    ds            : xarray.Dataset returned by load_dataset()
    regime_series : pd.Series of regime labels indexed by time (from classifier)
                    Must NOT be the hidden true_regime from ds.
    variables     : list of variable names to score (default: all)
    sources       : list of source names (default: all present in ds)
    lead_times    : list of lead_time values (default: all)

    Returns
    -------
    pd.DataFrame with columns:
        variable, source, lead_time, region, season, regime,
        rmse, mae, bias, n_samples
    """
    # ── resolve defaults ──────────────────────────────────────────────────────
    if variables is None:
        variables = [v.replace("truth_", "") for v in ds.data_vars
                     if v.startswith("truth_")]
    if sources is None:
        sources = list({v.split("_")[0] for v in ds.data_vars
                        if not v.startswith("truth_")})
    if lead_times is None:
        lead_times = list(ds.coords["lead_time"].values)

    # ── build regime series aligned to ds.time ───────────────────────────────
    time_index = pd.DatetimeIndex(ds.coords["time"].values)
    regime_arr = regime_series.reindex(time_index).values   # (n_time,)
    season_arr = ds.coords["season"].values                 # (n_time,)

    region_map = build_region_map(ds)  # (n_lat, n_lon)
    n_lat = len(ds.coords["lat"])
    n_lon = len(ds.coords["lon"])
    n_time = len(time_index)

    records = []
    unique_regions = np.unique(region_map)

    for var in variables:
        truth_key = f"truth_{var}"
        if truth_key not in ds.data_vars:
            warnings.warn(f"No truth field for variable '{var}' — skipping.")
            continue
        truth_full = ds[truth_key].values  # (n_time, n_lat, n_lon)

        for src in sources:
            fc_key = f"{src}_{var}"
            if fc_key not in ds.data_vars:
                continue
            fc_data = ds[fc_key].values   # (n_time, n_lat, n_lon, n_lead)
            if np.all(np.isnan(fc_data)):  # skip placeholder NaN sources
                continue
            lead_idx_map = {int(lt): i
                            for i, lt in enumerate(ds.coords["lead_time"].values)}

            for lead in lead_times:
                if lead not in lead_idx_map:
                    continue
                li = lead_idx_map[lead]
                fc_slice = fc_data[:, :, :, li]  # (n_time, n_lat, n_lon)
                err = fc_slice - truth_full       # (n_time, n_lat, n_lon)

                for season in ("monsoon", "winter"):
                    for regime in ("convective", "stratiform", "clear"):
                        # Time mask: season AND regime match
                        mask = (season_arr == season) & (regime_arr == regime)
                        if mask.sum() == 0:
                            continue
                        err_sub = err[mask]       # (n_sel, n_lat, n_lon)
                        truth_sub = truth_full[mask]

                        for region_id in unique_regions:
                            # Spatial mask: cells belonging to this region
                            cell_mask = (region_map == region_id)  # (lat, lon) bool
                            if not cell_mask.any():
                                continue

                            # Flatten over matched time steps and region cells
                            e_flat = err_sub[:, cell_mask].ravel()
                            n_samples = len(e_flat)
                            if n_samples == 0:
                                continue

                            rmse = float(np.sqrt(np.mean(e_flat ** 2)))
                            mae  = float(np.mean(np.abs(e_flat)))
                            bias = float(np.mean(e_flat))

                            records.append({
                                "variable":  var,
                                "source":    src,
                                "lead_time": int(lead),
                                "region":    region_id_to_label(int(region_id)),
                                "season":    season,
                                "regime":    regime,
                                "rmse":      rmse,
                                "mae":       mae,
                                "bias":      bias,
                                "n_samples": n_samples,
                            })

    df = pd.DataFrame(records)
    if df.empty:
        return df
    df = df.sort_values(
        ["variable", "source", "lead_time", "region", "season", "regime"]
    ).reset_index(drop=True)
    return df


def best_source_per_bucket(skill_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (variable, lead_time, region, season, regime) bucket,
    return the source with the lowest RMSE.
    """
    idx = skill_df.groupby(
        ["variable", "lead_time", "region", "season", "regime"]
    )["rmse"].idxmin()
    best = skill_df.loc[idx].copy()
    best = best.rename(columns={"source": "best_source"})
    return best[["variable", "lead_time", "region", "season", "regime",
                 "best_source", "rmse", "mae", "bias"]]


# ── Standalone smoke test ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))

    from data.synthetic import load_dataset
    from regime import fit_and_classify

    print("Loading dataset...")
    ds = load_dataset()
    print("Classifying regimes (no true_regime used)...")
    clf, regimes = fit_and_classify(ds)
    print(f"Regime distribution:\n{regimes.value_counts()}\n")

    print("Computing skill scores...")
    skill_df = compute_skill_scores(ds, regimes)
    print(f"Skill table shape: {skill_df.shape}")
    print(skill_df.head(10).to_string())

    print("\nRegion IDs present:", sorted(skill_df["region"].unique()))
    print("RMSE by region (mean over all buckets):")
    print(skill_df.groupby(["source", "region"])["rmse"].mean().unstack("region").round(3).to_string())

    print("\nBest source per bucket (sample):")
    best = best_source_per_bucket(skill_df)
    print(best.head(12).to_string())

    # Check that region dimension actually varies (spec requirement)
    pivot = skill_df[skill_df["variable"] == "rainfall"].pivot_table(
        index=["source", "lead_time"], columns="region", values="rmse"
    )
    print("\nRainfall RMSE per region (must vary spatially):")
    print(pivot.round(3).to_string())
    assert pivot.std(axis=1).mean() > 0.0, "Region dimension is flat — spec violation!"
    print("\n[OK] Region dimension shows spatial variation.")
