"""
api.py
======
FastAPI backend for PrakritiNetra — exposes the forecast blending pipeline
as a REST JSON API for consumption by the Vercel React frontend.

Endpoints
---------
GET  /                          health check
GET  /api/config                metadata (variables, lead_times, sources)
GET  /api/pipeline              run full pipeline; returns skill summary
GET  /api/forecast              blended + source forecasts for a time step
GET  /api/skill                 RMSE bar-chart data across all lead times
GET  /api/extreme               extreme-weather indicator tiles + flag maps

Run locally:
    uvicorn api:app --reload --port 8000
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from data.synthetic import load_dataset as _load_synthetic
from data.real_loader import load_dataset as _load_real, LEAD_TIMES, VARIABLES, SOURCES
from regime import fit_and_classify
from scoring import compute_skill_scores
from weighting import build_weight_table, get_weight_map, get_weights
from blending import blend_forecast, flag_extremes, compute_blend_skill_all_leads, EXTREME_THRESHOLDS

# ──────────────────────────────────────────────────────────────────────────────
# App setup
# ──────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="PrakritiNetra API",
    description="Hybrid AI-NWP Adaptive Forecast Blending — India domain",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten to Vercel domain in production
    allow_methods=["GET"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────────────────────────────────────
# Pipeline singleton (loaded once at startup)
# ──────────────────────────────────────────────────────────────────────────────
_DATASETS_DIR = _HERE / "datasets"
_REAL_DATA_AVAIL = (
    (_DATASETS_DIR / "gfs").exists() and
    any((_DATASETS_DIR / "gfs").iterdir()) if (_DATASETS_DIR / "gfs").exists() else False
)

_pipeline_cache: dict = {}


def _get_pipeline(method: str = "kmeans", n_regimes: int = 3) -> dict:
    key = (method, n_regimes)
    if key not in _pipeline_cache:
        if _REAL_DATA_AVAIL:
            ds = _load_real(
                gfs_dir=str(_DATASETS_DIR / "gfs"),
                gefs_dir=str(_DATASETS_DIR / "gefs"),
                era5_dir=str(_DATASETS_DIR / "era5"),
                pangu_dir=str(_DATASETS_DIR / "pangu"),
            )
        else:
            ds = _load_synthetic()
        clf, regimes = fit_and_classify(ds, n_regimes=n_regimes, method=method)
        skill_df = compute_skill_scores(ds, regimes)
        wt = build_weight_table(skill_df)
        _pipeline_cache[key] = dict(ds=ds, clf=clf, regimes=regimes, skill_df=skill_df, wt=wt)
    return _pipeline_cache[key]


def _safe_float(v):
    """Convert numpy scalar/nan to JSON-safe Python float."""
    if v is None:
        return None
    f = float(v)
    return None if (f != f) else round(f, 4)  # nan check


# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/")
def health():
    return {"status": "ok", "service": "PrakritiNetra API"}


@app.get("/api/config")
def get_config():
    """Return static metadata about variables, lead times, sources."""
    return {
        "variables": VARIABLES,
        "lead_times": LEAD_TIMES,
        "sources": SOURCES,
        "domain": {"lat_min": 8.0, "lat_max": 37.0, "lon_min": 68.0, "lon_max": 97.0},
        "real_data": _REAL_DATA_AVAIL,
    }


@app.get("/api/pipeline")
def run_pipeline(
    method: str = Query("kmeans", enum=["kmeans", "gmm"]),
    n_regimes: int = Query(3, ge=2, le=5),
):
    """Run (or return cached) pipeline; returns time steps and skill summary."""
    p = _get_pipeline(method, n_regimes)
    ds = p["ds"]
    times = pd.DatetimeIndex(ds.coords["time"].values)
    regimes = p["regimes"]
    seasons = ds.coords["season"].values
    return {
        "n_times": len(times),
        "time_steps": [t.strftime("%Y-%m-%d %H:%M") for t in times],
        "regimes": regimes.tolist(),
        "seasons": seasons.tolist(),
        "method": method,
        "n_regimes": n_regimes,
    }


@app.get("/api/forecast")
def get_forecast(
    variable: str = Query("rainfall", enum=VARIABLES),
    lead_time: int = Query(24, enum=LEAD_TIMES),
    time_index: int = Query(0),
    method: str = Query("kmeans", enum=["kmeans", "gmm"]),
    n_regimes: int = Query(3, ge=2, le=5),
):
    """Return blended + per-source forecast grids for a single time step."""
    p = _get_pipeline(method, n_regimes)
    ds, wt, regimes = p["ds"], p["wt"], p["regimes"]
    times = pd.DatetimeIndex(ds.coords["time"].values)

    if time_index < 0 or time_index >= len(times):
        raise HTTPException(400, f"time_index must be 0–{len(times)-1}")

    blend_result = blend_forecast(ds, wt, regimes, variable, lead_time)
    lats = ds.coords["lat"].values.tolist()
    lons = ds.coords["lon"].values.tolist()

    def _grid(arr2d):
        return np.where(np.isnan(arr2d), None, arr2d).tolist()

    result = {
        "time": times[time_index].strftime("%Y-%m-%d %H:%M"),
        "season": str(ds.coords["season"].values[time_index]),
        "regime": str(regimes.iloc[time_index]),
        "lats": lats,
        "lons": lons,
        "truth": _grid(blend_result["truth"][time_index]),
        "blended": _grid(blend_result["blended"][time_index]),
        "naive": _grid(blend_result["naive"][time_index]),
    }
    for src in SOURCES:
        if src in blend_result:
            result[src] = _grid(blend_result[src][time_index])
    return result


@app.get("/api/skill")
def get_skill(
    variable: str = Query("rainfall", enum=VARIABLES),
    method: str = Query("kmeans", enum=["kmeans", "gmm"]),
    n_regimes: int = Query(3, ge=2, le=5),
):
    """Return RMSE data for bar chart (all lead times × all sources)."""
    p = _get_pipeline(method, n_regimes)
    ds, wt, regimes = p["ds"], p["wt"], p["regimes"]
    all_skill = compute_blend_skill_all_leads(ds, wt, regimes, variables=[variable])

    bar_data = []
    for lt in LEAD_TIMES:
        row: dict = {"lead_time": lt}
        grp = all_skill[(all_skill["variable"] == variable) & (all_skill["lead_time"] == lt)]
        for src in grp["source"].unique():
            row[src] = _safe_float(grp[grp["source"] == src]["rmse"].mean())
        bar_data.append(row)

    # Region breakdown for the first lead time as default
    region_rows = []
    lt0 = LEAD_TIMES[1]
    grp2 = all_skill[(all_skill["variable"] == variable) & (all_skill["lead_time"] == lt0)]
    pivot = grp2.groupby(["region", "source"])["rmse"].mean().unstack("source").reset_index()
    for _, row in pivot.iterrows():
        rdict = {"region": row["region"]}
        for src in SOURCES + ["blended", "naive"]:
            if src in pivot.columns:
                rdict[src] = _safe_float(row.get(src))
        region_rows.append(rdict)

    return {"bar_data": bar_data, "region_data": region_rows}


@app.get("/api/extreme")
def get_extreme(
    lead_time: int = Query(24, enum=LEAD_TIMES),
    time_index: int = Query(0),
    method: str = Query("kmeans", enum=["kmeans", "gmm"]),
    n_regimes: int = Query(3, ge=2, le=5),
):
    """Return extreme-weather flag tiles for all three hazards."""
    p = _get_pipeline(method, n_regimes)
    ds, wt, regimes = p["ds"], p["wt"], p["regimes"]
    times = pd.DatetimeIndex(ds.coords["time"].values)

    if time_index < 0 or time_index >= len(times):
        raise HTTPException(400, f"time_index must be 0–{len(times)-1}")

    _ALL_VARS = {"rainfall": "heavy_rainfall", "temperature": "heat_wave", "wind": "high_wind"}
    result = {}
    for vname, hazard in _ALL_VARS.items():
        br = blend_forecast(ds, wt, regimes, vname, lead_time)
        bt = br["blended"][time_index]
        flgs = flag_extremes(bt[None], vname)
        farr = flgs.get(hazard, np.zeros_like(bt, dtype=bool))[0]
        thr = EXTREME_THRESHOLDS.get(vname, {}).get(hazard)
        vmax = float(np.nanmax(bt)) if not np.all(np.isnan(bt)) else 0.0
        vmean = float(np.nanmean(bt)) if not np.all(np.isnan(bt)) else 0.0
        frac = float(farr.mean()) * 100.0
        result[vname] = {
            "hazard": hazard,
            "threshold": thr,
            "frac_flagged": round(frac, 2),
            "domain_max": round(vmax, 2),
            "domain_mean": round(vmean, 2),
            "flag_grid": farr.astype(int).tolist(),
        }
    return {"time": times[time_index].strftime("%Y-%m-%d %H:%M"), "hazards": result}
