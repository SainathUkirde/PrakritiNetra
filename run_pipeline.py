#!/usr/bin/env python
"""
run_pipeline.py
===============
Operational CLI script — executes the full blending pipeline end-to-end:

  load → classify regime → score → weight → blend → flag extremes → save outputs

Uses REAL data from datasets/gfs/, datasets/gefs/, datasets/era5/, datasets/pangu/.

Usage
-----
    python run_pipeline.py [options]

Options
-------
    --method METHOD    Regime clustering method: kmeans | gmm (default: kmeans)
    --n-regimes N      Number of weather regimes (default: 3)
    --out-dir DIR      Output directory (default: ./outputs)
    --save-model PATH  Save the fitted classifier to PATH (.pkl)

Outputs saved
-------------
    outputs/skill_scores.csv
    outputs/weight_table.csv
    outputs/blend_skill.csv
    outputs/blended_{variable}_{lead_time}h.csv    (blended forecast per var/lead)
    outputs/extremes_{variable}_{lead_time}h.csv   (extreme flags per var/lead)
    outputs/classifier.pkl                         (fitted regime classifier)

Exit code 0 on success, 1 on error.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

# ── ensure local imports resolve regardless of working directory ───────────────
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from data.real_loader import load_dataset as _load_real_dataset
from regime import fit_and_classify, extract_features, RegimeClassifier
from scoring import compute_skill_scores
from weighting import build_weight_table
from blending import (
    blend_forecast, flag_extremes,
    compute_blend_skill_all_leads, EXTREME_THRESHOLDS,
)
from config import GFS_DIR, GEFS_DIR, ERA5_DIR, PANGU_DIR


# ── helpers ────────────────────────────────────────────────────────────────────

def _log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# ── pipeline steps ─────────────────────────────────────────────────────────────

def step_load():
    _log("STEP 1 — Loading real dataset (GFS + GEFS + Pangu + ERA5)")
    _log(f"  Datasets root: {Path(GFS_DIR).parent}")
    ds = _load_real_dataset(
        gfs_dir=GFS_DIR,
        gefs_dir=GEFS_DIR,
        era5_dir=ERA5_DIR,
        pangu_dir=PANGU_DIR,
    )
    _log(f"  Dataset loaded: {dict(ds.sizes)} | variables: {list(ds.data_vars)}")
    return ds


def step_regime(ds, method: str, n_regimes: int):
    _log(f"STEP 2 — Classifying regimes (method={method}, n_regimes={n_regimes})")
    clf, regimes = fit_and_classify(ds, n_regimes=n_regimes, method=method)
    dist = regimes.value_counts().to_dict()
    _log(f"  Regime distribution: {dist}")
    return clf, regimes


def step_score(ds, regimes):
    _log("STEP 3 — Computing skill scores per (var, src, lead, region, season, regime)")
    skill_df = compute_skill_scores(ds, regimes)
    _log(f"  Skill table rows: {len(skill_df)}")
    return skill_df


def step_weight(skill_df):
    _log("STEP 4 — Building adaptive weight table")
    wt = build_weight_table(skill_df)
    _log(f"  Weight table rows: {len(wt)}")
    return wt


def step_blend_and_extreme(ds, wt, regimes):
    _log("STEP 5 — Blending forecasts + flagging extremes")
    variables = [v.replace("truth_", "") for v in ds.data_vars if v.startswith("truth_")]
    lead_times = list(ds.coords["lead_time"].values)
    blend_results = {}
    extreme_flags = {}

    for var in variables:
        for lt in lead_times:
            key = (var, int(lt))
            result = blend_forecast(ds, wt, regimes, var, int(lt))
            blend_results[key] = result
            extreme_flags[key] = flag_extremes(result["blended"], var)
            n_extremes = {evt: int(arr.sum()) for evt, arr in extreme_flags[key].items()}
            _log(f"  ({var}, {lt}h) blended — extremes: {n_extremes}")

    _log("STEP 6 — Computing skill comparison (blended vs naive vs individual sources)")
    combined_skill = compute_blend_skill_all_leads(ds, wt, regimes)
    return blend_results, extreme_flags, combined_skill


def step_save(
    out_dir: Path,
    ds,
    clf: RegimeClassifier,
    skill_df: pd.DataFrame,
    wt: pd.DataFrame,
    blend_results: dict,
    extreme_flags: dict,
    combined_skill: pd.DataFrame,
    save_model: str | None,
):
    _log(f"STEP 7 — Saving outputs to {out_dir}")
    _ensure_dir(out_dir)

    # Skill scores
    skill_path = out_dir / "skill_scores.csv"
    skill_df.to_csv(skill_path, index=False)
    _log(f"  Saved: {skill_path}")

    # Weight table
    wt_path = out_dir / "weight_table.csv"
    wt.to_csv(wt_path, index=False)
    _log(f"  Saved: {wt_path}")

    # Blend skill summary
    bs_path = out_dir / "blend_skill.csv"
    combined_skill.to_csv(bs_path, index=False)
    _log(f"  Saved: {bs_path}")

    # Blended forecast arrays → CSV (mean over lat/lon per time step for compactness)
    for (var, lt), result in blend_results.items():
        times = pd.DatetimeIndex(ds.coords["time"].values)
        df_out = pd.DataFrame({
            "time":    times,
            "blended": result["blended"].mean(axis=(1, 2)),
            "naive":   result["naive"].mean(axis=(1, 2)),
            "truth":   result["truth"].mean(axis=(1, 2)),
        })
        for src_key in [k for k in result if k not in ("blended", "naive", "truth")]:
            df_out[src_key] = result[src_key].mean(axis=(1, 2))
        fc_path = out_dir / f"blended_{var}_{lt}h.csv"
        df_out.to_csv(fc_path, index=False)
        _log(f"  Saved: {fc_path}")

    # Extreme flags → CSV (count of flagged cells per time step)
    for (var, lt), flags in extreme_flags.items():
        times = pd.DatetimeIndex(ds.coords["time"].values)
        df_ext = pd.DataFrame({"time": times})
        for evt, arr in flags.items():
            df_ext[evt + "_count"] = arr.sum(axis=(1, 2))
            df_ext[evt + "_frac"]  = arr.mean(axis=(1, 2))
        ext_path = out_dir / f"extremes_{var}_{lt}h.csv"
        df_ext.to_csv(ext_path, index=False)
        _log(f"  Saved: {ext_path}")

    # Classifier
    model_path = Path(save_model) if save_model else out_dir / "classifier.pkl"
    clf.save(model_path)
    _log(f"  Saved classifier: {model_path}")

    # Summary JSON
    summary = {
        "n_time":           int(ds.sizes["time"]),
        "n_lat":            int(ds.sizes["lat"]),
        "n_lon":            int(ds.sizes["lon"]),
        "lead_times":       [int(lt) for lt in ds.coords["lead_time"].values],
        "variables":        [v.replace("truth_", "") for v in ds.data_vars if v.startswith("truth_")],
        "sources":          [s for s in set(c.split("_")[0] for c in ds.data_vars if not c.startswith("truth_"))],
        "skill_rows":       int(len(skill_df)),
        "weight_rows":      int(len(wt)),
        "blend_skill_rows": int(len(combined_skill)),
    }
    with open(out_dir / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    _log(f"  Saved: {out_dir / 'run_summary.json'}")


# ── CLI entry point ────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hybrid AI-NWP Forecast Blending Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--method",    type=str, default="kmeans",
                        choices=["kmeans", "gmm"],                help="Regime clustering method")
    parser.add_argument("--n-regimes", type=int, default=3,       help="Number of weather regimes")
    parser.add_argument("--out-dir",   type=str, default="outputs",
                        help="Output directory")
    parser.add_argument("--save-model", type=str, default=None,
                        help="Path to save the fitted classifier (.pkl)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = _HERE / args.out_dir

    _log("=" * 60)
    _log("Hybrid AI-NWP Forecast Blending Pipeline — STARTING")
    _log("=" * 60)

    t0 = time.time()
    try:
        ds            = step_load()
        clf, regimes  = step_regime(ds, args.method, args.n_regimes)
        skill_df      = step_score(ds, regimes)
        wt            = step_weight(skill_df)
        blend_results, extreme_flags, combined_skill = step_blend_and_extreme(ds, wt, regimes)
        step_save(out_dir, ds, clf, skill_df, wt,
                  blend_results, extreme_flags, combined_skill,
                  args.save_model)
    except Exception:
        _log("PIPELINE FAILED:")
        traceback.print_exc()
        return 1

    elapsed = time.time() - t0
    _log(f"Pipeline completed successfully in {elapsed:.1f}s.")
    _log("=" * 60)

    # Print a brief skill summary to stdout
    print("\n-- Skill summary (mean RMSE, blended vs naive vs sources) --")
    pivot = combined_skill.groupby(["variable", "lead_time", "source"])["rmse"] \
                          .mean().unstack("source")
    print(pivot.round(3).to_string())

    # Highlight underperforming buckets
    bad = combined_skill[combined_skill["beats_naive"].eq(False)]
    if not bad.empty:
        print("\n[WARN]  Buckets where ADAPTIVE BLEND does NOT beat naive average:")
        print(bad[["variable", "lead_time", "region", "source",
                   "rmse", "beats_naive"]].to_string(index=False))
        print("  Diagnosis: likely bias mismatch across sources in these buckets.")
    else:
        print("\n[OK] Adaptive blend beats naive average in all regions.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
