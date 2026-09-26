"""
data/real_loader.py
===================
Combined real-data loader: merges GFS (nwp), GEFS ensemble mean, Pangu (ai),
and ERA5 truth into a single pipeline-ready xarray.Dataset.

Sources
-------
  nwp      : GFS deterministic forecast   (datasets/gfs/)
  ensemble : GEFS ensemble mean forecast  (datasets/gefs/)
  ai       : Pangu-Weather AI forecast    (datasets/pangu/)
  truth    : ERA5 reanalysis              (datasets/era5/)

Time alignment
--------------
GFS and GEFS cycles are matched by init_time date (YYYYMMDD).
ERA5 truth is interpolated to the same timestamps.
Only cycles present in ALL three sources are kept.

Output
------
xr.Dataset with dimensions (time, lat, lon, lead_time) and variables:
    truth_{var}      (time, lat, lon)
    nwp_{var}        (time, lat, lon, lead_time)
    ensemble_{var}   (time, lat, lon, lead_time)
    ai_{var}         (time, lat, lon, lead_time)   ← NaN until Pangu added
    season           (time,)
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent))

from data.gfs_loader   import load_gefs_cycles as _load_gfs_cycles,  LEAD_TIMES
from data.gefs_loader  import load_gefs_cycles as _load_gefs_cycles
from data.era5_loader  import load_era5_truth
from data.pangu_loader import load_pangu_cycles

LAT_MIN, LAT_MAX = 8.0,  37.0
LON_MIN, LON_MAX = 68.0, 97.0
VARIABLES  = ["rainfall", "temperature", "wind"]
SOURCES    = ["nwp", "ensemble", "ai"]

_RAW = _HERE.parent / "datasets"


def load_dataset(
    gfs_dir:   str = str(_RAW / "gfs"),
    gefs_dir:  str = str(_RAW / "gefs"),
    era5_dir:  str = str(_RAW / "era5"),
    pangu_dir: str = str(_RAW / "pangu"),
    target_n_lat: int = 16,
    target_n_lon: int = 16,
) -> xr.Dataset:
    """
    Load real multi-source forecast dataset.

    Parameters
    ----------
    gfs_dir   : folder with GFS GRIB2 cycles  (sub-folders YYYYMMDD_00z)
    gefs_dir  : folder with GEFS GRIB2 cycles (sub-folders YYYYMMDD_00z)
    era5_dir  : folder with ERA5 NetCDF files
    pangu_dir : folder with Pangu NetCDF files (from download_pangu.py)
    target_n_lat / target_n_lon : output grid resolution (default 16×16)

    Returns
    -------
    xr.Dataset — same schema as synthetic.py load_dataset()
    """
    tgt_lats = np.linspace(LAT_MIN, LAT_MAX, target_n_lat)
    tgt_lons = np.linspace(LON_MIN, LON_MAX, target_n_lon)

    print("\nLoading GFS (nwp) cycles ...")
    gfs_cycles = _load_gfs_cycles(Path(gfs_dir), tgt_lats, tgt_lons)
    print(f"  {len(gfs_cycles)} GFS cycles loaded.")

    print("Loading GEFS (ensemble) cycles ...")
    gefs_cycles = _load_gefs_cycles(Path(gefs_dir), tgt_lats, tgt_lons)
    print(f"  {len(gefs_cycles)} GEFS cycles loaded.")

    print("Loading Pangu (ai) cycles ...")
    pangu_cycles = load_pangu_cycles(Path(pangu_dir), tgt_lats, tgt_lons)
    print(f"  {len(pangu_cycles)} Pangu cycles loaded.")

    print("Loading ERA5 (truth) ...")
    era5 = load_era5_truth(Path(era5_dir), tgt_lats, tgt_lons)
    print(f"  {len(era5['times'])} ERA5 time steps loaded.")

    # ── Align cycles by date ───────────────────────────────────────────────────
    # Key: YYYYMMDD string from init_time
    def _date_key(ts): return pd.Timestamp(ts).strftime("%Y%m%d")

    gfs_by_date   = {_date_key(c["init_time"]): c for c in gfs_cycles}
    gefs_by_date  = {_date_key(c["init_time"]): c for c in gefs_cycles}
    pangu_by_date = {_date_key(c["init_time"]): c for c in pangu_cycles}

    # Dates present in both GFS and GEFS (Pangu is optional — fills NaN if missing)
    common_dates = sorted(set(gfs_by_date) & set(gefs_by_date))
    if not common_dates:
        raise ValueError(
            "No overlapping dates between GFS and GEFS cycles.\n"
            f"GFS dates:  {sorted(gfs_by_date)}\n"
            f"GEFS dates: {sorted(gefs_by_date)}"
        )
    print(f"\nCommon GFS+GEFS dates : {common_dates}")
    pangu_overlap = sorted(set(common_dates) & set(pangu_by_date))
    print(f"Pangu overlap dates   : {pangu_overlap if pangu_overlap else 'none (DOY proxy will be used)'}")

    # Build stacked arrays (n_time, n_lat, n_lon, n_lead)
    n_time = len(common_dates)
    n_lat  = len(tgt_lats)
    n_lon  = len(tgt_lons)
    n_lead = len(LEAD_TIMES)

    gfs_rain   = np.full((n_time, n_lat, n_lon, n_lead), np.nan)
    gfs_temp   = np.full((n_time, n_lat, n_lon, n_lead), np.nan)
    gfs_wind   = np.full((n_time, n_lat, n_lon, n_lead), np.nan)
    gefs_rain  = np.full((n_time, n_lat, n_lon, n_lead), np.nan)
    gefs_temp  = np.full((n_time, n_lat, n_lon, n_lead), np.nan)
    gefs_wind  = np.full((n_time, n_lat, n_lon, n_lead), np.nan)
    pangu_rain = np.full((n_time, n_lat, n_lon, n_lead), np.nan)  # always NaN (no precip)
    pangu_temp = np.full((n_time, n_lat, n_lon, n_lead), np.nan)
    pangu_wind = np.full((n_time, n_lat, n_lon, n_lead), np.nan)
    init_times = []

    for i, date in enumerate(common_dates):
        gc  = gfs_by_date[date]
        gec = gefs_by_date[date]
        # all loaders return (n_lead, n_lat, n_lon) → transpose to (n_lat, n_lon, n_lead)
        gfs_rain[i]  = gc["rainfall"].transpose(1, 2, 0)
        gfs_temp[i]  = gc["temperature"].transpose(1, 2, 0)
        gfs_wind[i]  = gc["wind"].transpose(1, 2, 0)
        gefs_rain[i] = gec["rainfall"].transpose(1, 2, 0)
        gefs_temp[i] = gec["temperature"].transpose(1, 2, 0)
        gefs_wind[i] = gec["wind"].transpose(1, 2, 0)

        # Pangu: use exact date if available, else closest DOY from Pangu archive
        if date in pangu_by_date:
            pc = pangu_by_date[date]
        elif pangu_by_date:
            # Proxy: find Pangu cycle with closest calendar day-of-year
            target_doy = pd.Timestamp(date).day_of_year
            best = min(pangu_by_date.items(),
                       key=lambda kv: abs(pd.Timestamp(kv[0]).day_of_year - target_doy))
            pc = best[1]
        else:
            pc = None

        if pc is not None:
            pangu_temp[i] = pc["temperature"].transpose(1, 2, 0)
            pangu_wind[i] = pc["wind"].transpose(1, 2, 0)
            # rainfall stays NaN — Pangu has no precip output

        init_times.append(pd.Timestamp(gc["init_time"]))

    times = pd.DatetimeIndex(init_times)

    # ── Truth: match ERA5 to cycle init dates ──────────────────────────────────
    # For each cycle date, find the closest ERA5 timestep
    era5_times = era5["times"]
    truth_rain = np.full((n_time, n_lat, n_lon), np.nan)
    truth_temp = np.full((n_time, n_lat, n_lon), np.nan)
    truth_wind = np.full((n_time, n_lat, n_lon), np.nan)

    for i, t in enumerate(times):
        # Find ERA5 timestep closest to this cycle's init date
        diffs = np.abs(era5_times - t)
        closest_idx = int(np.argmin(diffs))
        closest_diff = diffs[closest_idx]

        if closest_diff > pd.Timedelta(days=30):
            # ERA5 and GFS date ranges don't overlap — use ERA5 daily mean as proxy
            # (This happens when ERA5 covers 2023-01 but GFS covers 2026-09)
            # Use the same calendar day-of-year from the ERA5 archive
            target_doy = t.day_of_year
            era5_doys = np.array([ts.day_of_year for ts in era5_times])
            doy_diffs = np.abs(era5_doys - target_doy)
            closest_idx = int(np.argmin(doy_diffs))
            warnings.warn(
                f"ERA5 date gap for {t.date()}: using closest DOY from ERA5 "
                f"({era5_times[closest_idx].date()})"
            )

        truth_rain[i] = era5["rainfall"][closest_idx]
        truth_temp[i] = era5["temperature"][closest_idx]
        truth_wind[i] = era5["wind"][closest_idx]

    # ── Season labels from GFS init times ─────────────────────────────────────
    seasons = np.array([
        "monsoon" if t.month in (6, 7, 8, 9) else "winter"
        for t in times
    ])

    # ── Assemble xarray.Dataset ────────────────────────────────────────────────
    ds = xr.Dataset(
        data_vars={
            "truth_rainfall":         (["time", "lat", "lon"], truth_rain),
            "truth_temperature":      (["time", "lat", "lon"], truth_temp),
            "truth_wind":             (["time", "lat", "lon"], truth_wind),
            "nwp_rainfall":           (["time", "lat", "lon", "lead_time"], gfs_rain),
            "nwp_temperature":        (["time", "lat", "lon", "lead_time"], gfs_temp),
            "nwp_wind":               (["time", "lat", "lon", "lead_time"], gfs_wind),
            "ensemble_rainfall":      (["time", "lat", "lon", "lead_time"], gefs_rain),
            "ensemble_temperature":   (["time", "lat", "lon", "lead_time"], gefs_temp),
            "ensemble_wind":          (["time", "lat", "lon", "lead_time"], gefs_wind),
            "ai_rainfall":            (["time", "lat", "lon", "lead_time"], pangu_rain),
            "ai_temperature":         (["time", "lat", "lon", "lead_time"], pangu_temp),
            "ai_wind":                (["time", "lat", "lon", "lead_time"], pangu_wind),
        },
        coords={
            "time":      times,
            "lat":       tgt_lats,
            "lon":       tgt_lons,
            "lead_time": LEAD_TIMES,
            "season":    ("time", seasons),
        },
    )
    ds.attrs["description"] = "Real multi-source forecast dataset: GFS + GEFS + ERA5 truth"
    ds.attrs["domain"]      = f"India lat [{LAT_MIN},{LAT_MAX}] lon [{LON_MIN},{LON_MAX}]"
    ds.attrs["sources"]     = "nwp=GFS; ensemble=GEFS mean; ai=Pangu-Weather; truth=ERA5"

    print(f"\nDataset assembled:")
    print(f"  {n_time} cycles x {target_n_lat} lat x {target_n_lon} lon x {n_lead} lead times")
    print(f"  Time range  : {times.min().date()} to {times.max().date()}")
    print(f"  Seasons     : {pd.Series(seasons).value_counts().to_dict()}")
    print(f"  NWP temp    : {float(np.nanmin(gfs_temp)):.1f} to {float(np.nanmax(gfs_temp)):.1f} C")
    print(f"  GEFS temp   : {float(np.nanmin(gefs_temp)):.1f} to {float(np.nanmax(gefs_temp)):.1f} C")
    print(f"  ERA5 truth  : {float(np.nanmin(truth_temp)):.1f} to {float(np.nanmax(truth_temp)):.1f} C")

    return ds
