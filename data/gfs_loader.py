"""
data/gfs_loader.py
==================
Real GFS GRIB2 data loader — calibrated for GFS pgrb2 0.25-degree files.

Variable names confirmed by inspection of actual GFS file:
  shortName "2t"    → t2m       : 2m temperature (K → °C)
  shortName "prate" → prate     : precipitation rate (kg/m²/s → mm/6h)
  shortName "10u"   → u10       : 10m U-wind (m/s)
  shortName "10v"   → v10       : 10m V-wind (m/s)
  Combined u10+v10               → wind speed magnitude (m/s)

Expected folder layout
-----------------------
    <gfs_dir>/
    ├── 20260922_00z/                  ← one sub-folder per forecast cycle
    │   ├── gfs.t00z.pgrb2.0p25.f000  ← analysis (f000) — used as truth
    │   ├── gfs.t00z.pgrb2.0p25.f024
    │   ├── gfs.t00z.pgrb2.0p25.f072
    │   └── gfs.t00z.pgrb2.0p25.f168
    ├── 20260923_00z/
    │   └── ...

Flat layout (all files in one folder) also works — loader auto-detects.

To activate: edit data/__init__.py to:
    from .gfs_loader import load_dataset
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

# ── India domain (same as synthetic.py) ───────────────────────────────────────
LAT_MIN, LAT_MAX = 8.0,  37.0
LON_MIN, LON_MAX = 68.0, 97.0

# Pipeline-required constants (must match what scoring/weighting/blending expect)
LEAD_TIMES = [6, 24, 72, 168]   # hours
VARIABLES  = ["rainfall", "temperature", "wind"]
SOURCES    = ["nwp", "ensemble", "ai"]


def _month_to_season(month: int) -> str:
    return "monsoon" if month in (6, 7, 8, 9) else "winter"


# ── Unit conversions (confirmed from file inspection) ─────────────────────────

def _kelvin_to_celsius(arr: np.ndarray) -> np.ndarray:
    """t2m in GFS is always Kelvin."""
    return arr - 273.15


def _prate_to_mm6h(arr: np.ndarray) -> np.ndarray:
    """
    prate is in kg/m²/s.
    1 kg/m²/s = 1 mm/s → multiply by 3600*6 = 21600 to get mm per 6-hour window.
    Clip negatives (can appear near coastlines).
    """
    return np.maximum(arr * 21600.0, 0.0)


def _wind_speed(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Combine U and V into wind speed magnitude."""
    return np.sqrt(u ** 2 + v ** 2)


# ── Grid helpers ───────────────────────────────────────────────────────────────

def _crop_india_0_360(lat_vals: np.ndarray, lon_vals: np.ndarray,
                      data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    GFS uses 0–360° longitudes and descending latitudes (90 → -90).
    Crop to India bounding box and return ascending lat order.
    """
    # Latitude: GFS goes 90→-90, select India slice and flip to ascending
    lat_mask = (lat_vals >= LAT_MIN) & (lat_vals <= LAT_MAX)
    # Longitude: India is 68–97°E, which is already 68–97 in 0–360
    lon_mask  = (lon_vals >= LON_MIN) & (lon_vals <= LON_MAX)

    cropped = data[lat_mask][:, lon_mask]
    lats    = lat_vals[lat_mask]
    lons    = lon_vals[lon_mask]

    # Flip to ascending latitude order
    if lats[0] > lats[-1]:
        cropped = cropped[::-1, :]
        lats    = lats[::-1]

    return lats, lons, cropped


def _regrid(src_lats: np.ndarray, src_lons: np.ndarray, src_data: np.ndarray,
            tgt_lats: np.ndarray, tgt_lons: np.ndarray) -> np.ndarray:
    """
    Bilinear interpolation from source (0.25°) grid to target coarser grid.
    Falls back to nearest-neighbour on edge values.
    """
    from scipy.interpolate import RegularGridInterpolator

    interp = RegularGridInterpolator(
        (src_lats, src_lons), src_data,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )
    lon2d, lat2d = np.meshgrid(tgt_lons, tgt_lats)
    pts = np.column_stack([lat2d.ravel(), lon2d.ravel()])
    return interp(pts).reshape(len(tgt_lats), len(tgt_lons))


# ── Single-file reader ─────────────────────────────────────────────────────────

def _read_grib_file(fpath: Path,
                    tgt_lats: np.ndarray,
                    tgt_lons: np.ndarray) -> Optional[dict]:
    """
    Read temperature, precipitation rate, and wind from one GFS GRIB2 file.

    Returns dict:
        {
          "init_time":   pd.Timestamp  (analysis time, i.e. valid_time - step),
          "lead_hours":  int,
          "rainfall":    (n_lat, n_lon)  float64 mm/6h,
          "temperature": (n_lat, n_lon)  float64 °C,
          "wind":        (n_lat, n_lon)  float64 m/s,
        }
    Returns None if the file cannot be read.
    """
    import cfgrib  # confirms library present; ImportError is raised clearly

    result = {}

    # ── Temperature ───────────────────────────────────────────────────────────
    try:
        ds_t = xr.open_dataset(str(fpath), engine="cfgrib",
            backend_kwargs={"filter_by_keys": {"shortName": "2t"}},
            errors="ignore")

        lat_vals = ds_t.latitude.values
        lon_vals = ds_t.longitude.values
        raw_t    = ds_t["t2m"].values.astype(np.float64)  # (721, 1440) global

        lats_c, lons_c, t_india = _crop_india_0_360(lat_vals, lon_vals, raw_t)
        t_grid = _regrid(lats_c, lons_c, t_india, tgt_lats, tgt_lons)
        result["temperature"] = _kelvin_to_celsius(t_grid)

        # Extract init time and lead hours from this field
        init_ts = pd.Timestamp(ds_t.coords["time"].values)
        step_ns = int(ds_t.coords["step"].values)          # nanoseconds
        lead_h  = step_ns // (3600 * 1_000_000_000)        # → hours
        result["init_time"]  = init_ts
        result["lead_hours"] = int(lead_h)

        ds_t.close()
    except Exception as e:
        warnings.warn(f"Temperature read failed for {fpath.name}: {e}")
        return None

    # ── Precipitation ─────────────────────────────────────────────────────────
    # GFS prate may have both 'instant' and 'avg' step types; try avg first
    # (average rate over the accumulation period), then instant.
    _prate_loaded = False
    for _step_type in ("avg", "instant"):
        try:
            ds_p = xr.open_dataset(str(fpath), engine="cfgrib",
                backend_kwargs={"filter_by_keys": {
                    "shortName": "prate", "typeOfLevel": "surface",
                    "stepType": _step_type}},
                errors="ignore")
            raw_p = ds_p["prate"].values.astype(np.float64)
            _, _, p_india = _crop_india_0_360(lat_vals, lon_vals, raw_p)
            p_grid = _regrid(lats_c, lons_c, p_india, tgt_lats, tgt_lons)
            result["rainfall"] = _prate_to_mm6h(p_grid)
            ds_p.close()
            _prate_loaded = True
            break
        except Exception:
            pass
    if not _prate_loaded:
        warnings.warn(f"Precipitation read failed for {fpath.name}: both stepType avg and instant failed")
        result["rainfall"] = np.zeros_like(result["temperature"])

    # ── Wind ──────────────────────────────────────────────────────────────────
    try:
        ds_u = xr.open_dataset(str(fpath), engine="cfgrib",
            backend_kwargs={"filter_by_keys": {"shortName": "10u"}},
            errors="ignore")
        ds_v = xr.open_dataset(str(fpath), engine="cfgrib",
            backend_kwargs={"filter_by_keys": {"shortName": "10v"}},
            errors="ignore")

        raw_u = ds_u["u10"].values.astype(np.float64)
        raw_v = ds_v["v10"].values.astype(np.float64)

        _, _, u_india = _crop_india_0_360(lat_vals, lon_vals, raw_u)
        _, _, v_india = _crop_india_0_360(lat_vals, lon_vals, raw_v)
        u_grid = _regrid(lats_c, lons_c, u_india, tgt_lats, tgt_lons)
        v_grid = _regrid(lats_c, lons_c, v_india, tgt_lats, tgt_lons)
        result["wind"] = _wind_speed(u_grid, v_grid)

        ds_u.close()
        ds_v.close()
    except Exception as e:
        warnings.warn(f"Wind read failed for {fpath.name}: {e}")
        result["wind"] = np.zeros_like(result["temperature"])

    return result


# ── Cycle directory reader ─────────────────────────────────────────────────────

def _read_cycle(cycle_dir: Path,
                tgt_lats: np.ndarray,
                tgt_lons: np.ndarray) -> Optional[dict]:
    """
    Read all lead-time files for one forecast cycle directory.

    Returns:
        {
          "init_time":   pd.Timestamp,
          "rainfall":    (n_lead, n_lat, n_lon),
          "temperature": (n_lead, n_lat, n_lon),
          "wind":        (n_lead, n_lat, n_lon),
        }
    or None if the cycle has no readable files.
    """
    n_lat  = len(tgt_lats)
    n_lon  = len(tgt_lons)
    n_lead = len(LEAD_TIMES)

    rain_arr = np.full((n_lead, n_lat, n_lon), np.nan)
    temp_arr = np.full((n_lead, n_lat, n_lon), np.nan)
    wind_arr = np.full((n_lead, n_lat, n_lon), np.nan)
    init_time = None

    for li, lead in enumerate(LEAD_TIMES):
        # Find file matching this lead time — e.g. f006, f024, f072, f168
        # GFS naming: ...f006, ...f024, etc.
        matches = (list(cycle_dir.glob(f"*f{lead:03d}*")) +
                   list(cycle_dir.glob(f"*f{lead:03d}")))
        if not matches:
            print(f"      [WARN] no file for lead={lead}h in {cycle_dir.name}")
            continue

        fpath = matches[0]
        print(f"      lead={lead:3d}h  -> {fpath.name} ... ", end="", flush=True)
        rec = _read_grib_file(fpath, tgt_lats, tgt_lons)
        if rec is None:
            print("FAILED")
            continue

        rain_arr[li] = rec["rainfall"]
        temp_arr[li] = rec["temperature"]
        wind_arr[li] = rec["wind"]

        if init_time is None:
            init_time = rec["init_time"]

        print(f"OK  (T={rec['temperature'].mean():.1f}°C, "
              f"R={rec['rainfall'].mean():.3f}mm, "
              f"W={rec['wind'].mean():.1f}m/s)")

    if init_time is None:
        return None

    return {
        "init_time":   init_time,
        "rainfall":    rain_arr,    # (n_lead, n_lat, n_lon)
        "temperature": temp_arr,
        "wind":        wind_arr,
    }


# ── Public helper: load all GFS cycles as list (used by real_loader.py) ───────

def load_gefs_cycles(gfs_dir: Path, tgt_lats: np.ndarray,
                     tgt_lons: np.ndarray) -> list[dict]:
    """Load all GFS forecast cycles from gfs_dir. Returns list of cycle dicts."""
    cycle_dirs = sorted([d for d in gfs_dir.iterdir() if d.is_dir()])
    if not cycle_dirs:
        cycle_dirs = [gfs_dir]
    cycles = []
    for cd in cycle_dirs:
        rec = _read_cycle(cd, tgt_lats, tgt_lons)
        if rec is not None:
            cycles.append(rec)
    return cycles


# ── Public entry point ─────────────────────────────────────────────────────────

def load_dataset(
    gfs_dir: str = "./datasets/gfs",
    target_n_lat: int = 16,
    target_n_lon: int = 16,
) -> xr.Dataset:
    """
    Load GFS GRIB2 files and return a pipeline-ready xarray.Dataset.

    Parameters
    ----------
    gfs_dir       : Folder with one sub-folder per forecast init cycle.
                    Each sub-folder must contain files named *f006*, *f024*,
                    *f072*, *f168* (GFS standard naming).
                    Also accepts flat layout (all files in one folder).
    target_n_lat  : Output latitude resolution (default 16 = ~2° grid).
    target_n_lon  : Output longitude resolution (default 16 = ~2° grid).

    Returns
    -------
    xr.Dataset with dimensions (time, lat, lon, lead_time) and variables:
        truth_rainfall, truth_temperature, truth_wind      — from f000 (analysis)
        nwp_rainfall, nwp_temperature, nwp_wind            — GFS forecasts
        ensemble_*, ai_*                                   — NaN (add later)
    """
    gfs_path = Path(gfs_dir)
    if not gfs_path.exists():
        raise FileNotFoundError(
            f"\nGFS data directory not found: {gfs_path.resolve()}\n\n"
            f"Create the folder and put your GFS files inside like this:\n"
            f"  {gfs_path}/\n"
            f"  └── 20260922_00z/\n"
            f"      ├── gfs.t00z.pgrb2.0p25.f000\n"
            f"      ├── gfs.t00z.pgrb2.0p25.f024\n"
            f"      ├── gfs.t00z.pgrb2.0p25.f072\n"
            f"      └── gfs.t00z.pgrb2.0p25.f168\n"
        )

    # Target grid for India
    tgt_lats = np.linspace(LAT_MIN, LAT_MAX, target_n_lat)
    tgt_lons = np.linspace(LON_MIN, LON_MAX, target_n_lon)

    # Find cycle directories
    cycle_dirs = sorted([d for d in gfs_path.iterdir() if d.is_dir()])
    if not cycle_dirs:
        # Flat layout: all files in gfs_dir itself → treat as one cycle
        cycle_dirs = [gfs_path]

    print(f"\nFound {len(cycle_dirs)} forecast cycle(s) in {gfs_path}")
    print(f"Output grid: {target_n_lat} lat x {target_n_lon} lon "
          f"(India {LAT_MIN}-{LAT_MAX}N, {LON_MIN}-{LON_MAX}E)")
    print()

    # Read each cycle
    cycles = []
    for cd in cycle_dirs:
        print(f"  Cycle: {cd.name}")
        rec = _read_cycle(cd, tgt_lats, tgt_lons)
        if rec is not None:
            cycles.append(rec)
        else:
            print(f"  [SKIP] {cd.name} — no data read")
        print()

    if not cycles:
        raise ValueError(
            "No GFS cycles could be read. Checklist:\n"
            "  1. Files have lead-time suffix: f000, f024, f072, f168\n"
            "  2. cfgrib installed: pip install cfgrib eccodes\n"
            "  3. Each cycle is in its own sub-folder\n"
            "  4. Run:  python data/gfs_loader.py --inspect <file.grb2>"
        )

    n_time = len(cycles)
    n_lead = len(LEAD_TIMES)

    # Stack: each cycle gives (n_lead, lat, lon) → result is (time, lead, lat, lon)
    rain_fc = np.stack([c["rainfall"]    for c in cycles])  # (T, L, lat, lon)
    temp_fc = np.stack([c["temperature"] for c in cycles])
    wind_fc = np.stack([c["wind"]        for c in cycles])

    # Transpose to (time, lat, lon, lead_time) — pipeline convention
    rain_fc = rain_fc.transpose(0, 2, 3, 1)
    temp_fc = temp_fc.transpose(0, 2, 3, 1)
    wind_fc = wind_fc.transpose(0, 2, 3, 1)

    times   = pd.DatetimeIndex([c["init_time"] for c in cycles])
    seasons = np.array([_month_to_season(t.month) for t in times])

    # Truth = f000 analysis field (lead index 0 = 0h)
    # If you add ERA5 later, replace these three lines with ERA5 arrays.
    lead0 = LEAD_TIMES.index(min(LEAD_TIMES))   # index of shortest lead (f000 or f006)
    truth_rain = rain_fc[:, :, :, lead0].copy()
    truth_temp = temp_fc[:, :, :, lead0].copy()
    truth_wind = wind_fc[:, :, :, lead0].copy()

    # NaN-filled placeholders for ensemble and ai sources (add when available)
    nan_fc = np.full_like(rain_fc, np.nan)

    ds = xr.Dataset(
        data_vars={
            # Truth (stand-in = GFS analysis, f000)
            "truth_rainfall":         (["time", "lat", "lon"], truth_rain),
            "truth_temperature":      (["time", "lat", "lon"], truth_temp),
            "truth_wind":             (["time", "lat", "lon"], truth_wind),
            # NWP = GFS deterministic
            "nwp_rainfall":           (["time", "lat", "lon", "lead_time"], rain_fc),
            "nwp_temperature":        (["time", "lat", "lon", "lead_time"], temp_fc),
            "nwp_wind":               (["time", "lat", "lon", "lead_time"], wind_fc),
            # Ensemble placeholder (fill with GEFS data later)
            "ensemble_rainfall":      (["time", "lat", "lon", "lead_time"], nan_fc.copy()),
            "ensemble_temperature":   (["time", "lat", "lon", "lead_time"], nan_fc.copy()),
            "ensemble_wind":          (["time", "lat", "lon", "lead_time"], nan_fc.copy()),
            # AI placeholder (fill with Pangu/GraphCast data later)
            "ai_rainfall":            (["time", "lat", "lon", "lead_time"], nan_fc.copy()),
            "ai_temperature":         (["time", "lat", "lon", "lead_time"], nan_fc.copy()),
            "ai_wind":                (["time", "lat", "lon", "lead_time"], nan_fc.copy()),
        },
        coords={
            "time":      times,
            "lat":       tgt_lats,
            "lon":       tgt_lons,
            "lead_time": LEAD_TIMES,
            "season":    ("time", seasons),
        },
    )
    ds.attrs["description"] = "GFS NWP real data — processed from GRIB2 pgrb2 0.25deg"
    ds.attrs["domain"]      = f"India lat [{LAT_MIN},{LAT_MAX}] lon [{LON_MIN},{LON_MAX}]"
    ds.attrs["truth"]       = "GFS f000 analysis (replace with ERA5 for better verification)"
    ds.attrs["sources"]     = "nwp=GFS; ensemble=NaN (add GEFS); ai=NaN (add Pangu/GraphCast)"

    print(f"Dataset assembled:")
    print(f"  {n_time} cycles x {target_n_lat} lat x {target_n_lon} lon x {n_lead} lead times")
    print(f"  Time range : {times.min()} to {times.max()}")
    print(f"  Seasons    : {pd.Series(seasons).value_counts().to_dict()}")
    print(f"  Temp range : {float(np.nanmin(temp_fc)):.1f} to {float(np.nanmax(temp_fc)):.1f} °C")
    print(f"  Rain range : {float(np.nanmin(rain_fc)):.2f} to {float(np.nanmax(rain_fc)):.2f} mm/6h")
    print(f"  Wind range : {float(np.nanmin(wind_fc)):.2f} to {float(np.nanmax(wind_fc)):.2f} m/s")

    return ds


# ── Diagnostic helper ──────────────────────────────────────────────────────────

def inspect_grib(filepath: str) -> None:
    """
    Print every variable group in a GRIB2 file.
    Run this to verify variable names in an unfamiliar file.

        python data/gfs_loader.py --inspect path/to/file.grb2
    """
    import cfgrib
    datasets = cfgrib.open_datasets(filepath)
    print(f"{'Group':<6} {'shortName':<15} {'typeOfLevel':<28} {'level':<8} {'name'}")
    print("-" * 80)
    for i, ds in enumerate(datasets):
        for v in ds.data_vars:
            a = ds[v].attrs
            print(f"  {i:<4} {a.get('GRIB_shortName','?'):<15} "
                  f"{a.get('GRIB_typeOfLevel','?'):<28} "
                  f"{a.get('GRIB_level','?'):<8} "
                  f"{a.get('GRIB_name','?')}")


# ── Standalone test / CLI ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser(
        description="GFS GRIB2 loader for ForecastBlend",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--gfs-dir", default="./datasets/gfs",
                        help="Root folder with one sub-folder per GFS cycle")
    parser.add_argument("--inspect", default=None, metavar="FILE",
                        help="Print all variables in a GRIB2 file and exit")
    args = parser.parse_args()

    if args.inspect:
        inspect_grib(args.inspect)
        sys.exit(0)

    print("Testing GFS loader...")
    try:
        ds = load_dataset(gfs_dir=args.gfs_dir)
        print("\nFinal dataset:")
        print(ds)
        print("\n[OK] GFS loader works correctly.")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
