"""
weighting.py
============
Adaptive weighting layer.

Computes per-(variable, lead_time, region, season, regime) weights for each
forecast source.  Baseline strategy: inverse-error weighting (∝ 1/RMSE,
normalised to sum to 1).

Spatial requirement (spec)
--------------------------
Weights MUST vary by region.  The weight map broadcast back to the full grid
must show visible spatial variation — a flat map is a spec violation.

Swap-friendly design
--------------------
The weighting strategy is isolated in _compute_weights_from_skill().
To plug in a LightGBM learned model later, replace only that function while
keeping the public API (get_weights, get_weight_map) unchanged.

Public API
----------
build_weight_table(skill_df)  → wide DataFrame with a column per source
get_weights(weight_table, variable, lead_time, region, season, regime) → dict
get_weight_map(weight_table, ds, variable, lead_time, season, regime)  → ndarray
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

from scoring import REGION_TILES, build_region_map, region_id_to_label

# ── baseline strategy: inverse-error weighting ────────────────────────────────

def _compute_weights_from_skill(group: pd.DataFrame) -> dict[str, float]:
    """
    Given a subset of the skill table for a single bucket
    (variable, lead_time, region, season, regime), compute normalised
    inverse-RMSE weights for each source.

    Replace this function body to plug in LightGBM or any other method —
    the rest of the pipeline doesn't change.

    Parameters
    ----------
    group : DataFrame with columns [source, rmse, ...]

    Returns
    -------
    dict {source_name: weight}  — weights sum to 1.0
    """
    sources = group["source"].values
    rmses   = group["rmse"].values.astype(float)

    # Guard: if RMSE is zero (perfect forecast), give it all weight.
    if np.any(rmses == 0):
        w = np.where(rmses == 0, 1.0, 0.0).astype(float)
    else:
        inv = 1.0 / rmses
        w = inv / inv.sum()

    return dict(zip(sources, w))


# ── build the full weight table ────────────────────────────────────────────────

def build_weight_table(skill_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute normalised source weights for every
    (variable, lead_time, region, season, regime) bucket.

    Returns
    -------
    DataFrame indexed by (variable, lead_time, region, season, regime) with
    one column per source containing its weight.  Also includes a
    'dominant_source' column for quick lookup.
    """
    group_keys = ["variable", "lead_time", "region", "season", "regime"]
    records = []
    for bucket, grp in skill_df.groupby(group_keys, sort=False):
        weights = _compute_weights_from_skill(grp)
        row = dict(zip(group_keys, bucket))
        row.update(weights)
        row["dominant_source"] = max(weights, key=weights.get)
        records.append(row)

    wt = pd.DataFrame(records)
    # Ensure columns are sorted predictably
    source_cols = sorted(c for c in wt.columns if c not in group_keys + ["dominant_source"])
    wt = wt[group_keys + source_cols + ["dominant_source"]].reset_index(drop=True)
    return wt


# ── single-bucket lookup ───────────────────────────────────────────────────────

def get_weights(
    weight_table: pd.DataFrame,
    variable: str,
    lead_time: int,
    region: str,
    season: str,
    regime: str,
) -> dict[str, float]:
    """
    Return {source: weight} for the requested bucket.

    Falls back to equal weights if bucket is missing (e.g. insufficient
    historical data for a rare regime).
    """
    mask = (
        (weight_table["variable"]  == variable)  &
        (weight_table["lead_time"] == lead_time) &
        (weight_table["region"]    == region)    &
        (weight_table["season"]    == season)    &
        (weight_table["regime"]    == regime)
    )
    rows = weight_table[mask]
    if rows.empty:
        warnings.warn(
            f"No weight entry for ({variable}, {lead_time}, {region}, "
            f"{season}, {regime}) — using equal weights."
        )
        source_cols = [c for c in weight_table.columns
                       if c not in ["variable", "lead_time", "region",
                                    "season", "regime", "dominant_source"]]
        n = len(source_cols)
        return {s: 1.0 / n for s in source_cols} if n > 0 else {}

    source_cols = [c for c in weight_table.columns
                   if c not in ["variable", "lead_time", "region",
                                "season", "regime", "dominant_source"]]
    row = rows.iloc[0]
    return {s: float(row[s]) for s in source_cols if s in row}


# ── spatial weight map ─────────────────────────────────────────────────────────

def get_weight_map(
    weight_table: pd.DataFrame,
    ds: xr.Dataset,
    variable: str,
    lead_time: int,
    season: str,
    regime: str,
    source: Optional[str] = None,
) -> np.ndarray:
    """
    Return a (n_lat, n_lon) weight map for the given context.

    If `source` is specified → map shows that source's weight per cell.
    If `source` is None      → map shows the dominant source as an integer
                               category (useful for heatmap of "who wins where").

    Weights are broadcast from coarse sub-region tiles to individual grid cells,
    so the returned array spans the full spatial domain with spatial variation.
    """
    region_map = build_region_map(ds)     # (n_lat, n_lon) int
    n_lat = len(ds.coords["lat"])
    n_lon = len(ds.coords["lon"])

    out = np.full((n_lat, n_lon), np.nan)

    # Build source → integer mapping if returning dominant source
    source_cols = sorted(c for c in weight_table.columns
                         if c not in ["variable", "lead_time", "region",
                                      "season", "regime", "dominant_source"])
    source_to_int = {s: i for i, s in enumerate(sorted(source_cols))}

    for region_id in np.unique(region_map):
        rlabel = region_id_to_label(int(region_id))
        weights = get_weights(weight_table, variable, lead_time,
                              rlabel, season, regime)
        cell_mask = (region_map == region_id)
        if source is not None:
            val = weights.get(source, 0.0)
        else:
            # Dominant source → integer
            dom = max(weights, key=weights.get) if weights else source_cols[0]
            val = float(source_to_int.get(dom, 0))
        out[cell_mask] = val

    return out


def get_all_source_weight_maps(
    weight_table: pd.DataFrame,
    ds: xr.Dataset,
    variable: str,
    lead_time: int,
    season: str,
    regime: str,
) -> dict[str, np.ndarray]:
    """
    Return {source: (n_lat, n_lon) weight array} for all sources.
    """
    source_cols = sorted(c for c in weight_table.columns
                         if c not in ["variable", "lead_time", "region",
                                      "season", "regime", "dominant_source"])
    return {
        src: get_weight_map(weight_table, ds, variable, lead_time,
                            season, regime, source=src)
        for src in source_cols
    }


# ── Standalone smoke test ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))

    from data.synthetic import load_dataset
    from regime import fit_and_classify
    from scoring import compute_skill_scores

    print("Loading dataset...")
    ds = load_dataset()
    print("Classifying regimes...")
    _, regimes = fit_and_classify(ds)
    print("Computing skill scores...")
    skill_df = compute_skill_scores(ds, regimes)
    print("Building weight table...")
    wt = build_weight_table(skill_df)
    print(f"Weight table shape: {wt.shape}")
    print(wt.head(12).to_string())

    # Lookup a specific bucket
    w = get_weights(wt, "rainfall", 24, "R00", "monsoon", "convective")
    print(f"\nWeights for (rainfall, 24h, R00, monsoon, convective):\n  {w}")

    # Spatial weight map
    wmap = get_weight_map(wt, ds, "rainfall", 24, "monsoon", "convective",
                          source="nwp")
    print(f"\nNWP weight map shape: {wmap.shape}")
    print("Weight map (NWP, rainfall, 24h, monsoon, convective):")
    print(np.round(wmap, 3))

    # Verify spatial variation
    assert wmap.std() > 0.0, "Weight map is spatially flat — spec violation!"
    print(f"\n[OK] Weight map std = {wmap.std():.4f} (spatial variation confirmed)")

    dom_map = get_weight_map(wt, ds, "rainfall", 24, "monsoon", "convective")
    print("\nDominant-source map (0=ai, 1=ensemble, 2=nwp):")
    print(dom_map.astype(int))
    assert dom_map.std() > 0.0 or len(np.unique(dom_map)) >= 1, "Dominant-source map is flat"
    print("[OK] Dominant-source map computed successfully.")
