"""
data/pangu_loader.py
====================
Reads Pangu-Weather AI forecast NetCDF files from datasets/pangu/.

File structure (from download_pangu.py):
  Dims    : (time, prediction_timedelta, longitude, latitude)
  Coords  :
    time                  : daily init times (2022-01-01 ...)
    prediction_timedelta  : lead hours [6, 24, 72, 168]
    longitude             : 69.0 ... 96.0  (0-360 convention, India subset)
    latitude              : 9.0  ... 36.0  (ascending)
  Variables:
    2m_temperature           (K  → °C)
    10m_u_component_of_wind  (m/s)
    10m_v_component_of_wind  (m/s)
    10m_wind_speed           (m/s, pre-computed magnitude)
    NOTE: no precipitation — ai_rainfall stays NaN.

Output per cycle dict:
    {
      "init_time"  : pd.Timestamp,
      "rainfall"   : (n_lead, n_lat, n_lon)  — all NaN (no precip in Pangu)
      "temperature": (n_lead, n_lat, n_lon)  — °C
      "wind"       : (n_lead, n_lat, n_lon)  — m/s
    }
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

LAT_MIN, LAT_MAX = 8.0, 37.0
LON_MIN, LON_MAX = 68.0, 97.0
LEAD_TIMES = [6, 24, 72, 168]


def _kelvin_to_celsius(arr: np.ndarray) -> np.ndarray:
    return arr - 273.15


def _regrid(src_lats, src_lons, src_data, tgt_lats, tgt_lons):
    """Bilinear interpolation to target grid."""
    from scipy.interpolate import RegularGridInterpolator
    if len(src_lats) > 1 and src_lats[0] > src_lats[-1]:
        src_data = src_data[::-1, :]
        src_lats = src_lats[::-1]
    interp = RegularGridInterpolator(
        (src_lats, src_lons), src_data,
        method="linear", bounds_error=False, fill_value=np.nan,
    )
    lon2d, lat2d = np.meshgrid(tgt_lons, tgt_lats)
    pts = np.column_stack([lat2d.ravel(), lon2d.ravel()])
    return interp(pts).reshape(len(tgt_lats), len(tgt_lons))


def load_pangu_cycles(pangu_dir: Path, tgt_lats: np.ndarray,
                      tgt_lons: np.ndarray) -> list[dict]:
    """
    Load all Pangu NetCDF files from pangu_dir.
    Returns list of cycle dicts with keys:
        init_time, rainfall, temperature, wind
        each forecast array shaped (n_lead, n_lat, n_lon).
    """
    nc_files = sorted(pangu_dir.glob("pangu_*.nc"))
    if not nc_files:
        warnings.warn(f"No Pangu NetCDF files found in {pangu_dir}. "
                      "Run download_pangu.py first.")
        return []

    # Load and concatenate all files
    datasets = []
    for f in nc_files:
        try:
            datasets.append(xr.open_dataset(f))
        except Exception as e:
            warnings.warn(f"Could not open {f.name}: {e}")

    if not datasets:
        return []

    ds_all = xr.concat(datasets, dim="time").sortby("time")
    # Drop duplicate times if files overlap
    _, idx = np.unique(ds_all.coords["time"].values, return_index=True)
    ds_all = ds_all.isel(time=idx)

    file_lats = ds_all.coords["latitude"].values   # ascending
    file_lons = ds_all.coords["longitude"].values  # 0-360
    all_leads = list(ds_all.coords["prediction_timedelta"].values)

    # Select only our pipeline lead times
    sel_leads = [lt for lt in LEAD_TIMES if lt in all_leads]
    if not sel_leads:
        warnings.warn(f"No matching lead times. File has: {all_leads}")
        return []

    n_lat  = len(tgt_lats)
    n_lon  = len(tgt_lons)
    n_lead = len(LEAD_TIMES)
    cycles = []

    for ti in range(len(ds_all.coords["time"])):
        init_time = pd.Timestamp(ds_all.coords["time"].values[ti])
        rain_arr  = np.full((n_lead, n_lat, n_lon), np.nan)  # Pangu has no precip
        temp_arr  = np.full((n_lead, n_lat, n_lon), np.nan)
        wind_arr  = np.full((n_lead, n_lat, n_lon), np.nan)

        for li, lead in enumerate(LEAD_TIMES):
            if lead not in all_leads:
                continue
            lead_idx = all_leads.index(lead)

            # Raw slice: dims are (time, prediction_timedelta, longitude, latitude)
            # → shape (lon, lat) → transpose to (lat, lon)
            t2m_raw = ds_all["2m_temperature"].values[ti, lead_idx]   # (lon, lat)

            # Detect dimension order: if shape[0] == len(lons) → (lon,lat)
            if t2m_raw.shape[0] == len(file_lons):
                t2m_raw = t2m_raw.T   # → (lat, lon)

            t2m_c = _kelvin_to_celsius(t2m_raw.astype(np.float64))
            temp_arr[li] = _regrid(file_lats, file_lons, t2m_c, tgt_lats, tgt_lons)

            # Wind: prefer pre-computed 10m_wind_speed if available
            if "10m_wind_speed" in ds_all.data_vars:
                spd_raw = ds_all["10m_wind_speed"].values[ti, lead_idx]
                if spd_raw.shape[0] == len(file_lons):
                    spd_raw = spd_raw.T
                wind_arr[li] = _regrid(file_lats, file_lons,
                                       spd_raw.astype(np.float64),
                                       tgt_lats, tgt_lons)
            elif ("10m_u_component_of_wind" in ds_all.data_vars and
                  "10m_v_component_of_wind" in ds_all.data_vars):
                u_raw = ds_all["10m_u_component_of_wind"].values[ti, lead_idx]
                v_raw = ds_all["10m_v_component_of_wind"].values[ti, lead_idx]
                if u_raw.shape[0] == len(file_lons):
                    u_raw = u_raw.T
                    v_raw = v_raw.T
                spd = np.sqrt(u_raw.astype(np.float64)**2 +
                              v_raw.astype(np.float64)**2)
                wind_arr[li] = _regrid(file_lats, file_lons, spd,
                                       tgt_lats, tgt_lons)

        cycles.append({
            "init_time":   init_time,
            "rainfall":    rain_arr,
            "temperature": temp_arr,
            "wind":        wind_arr,
        })

    print(f"  Pangu: {len(cycles)} cycles loaded from {pangu_dir.name}/")
    return cycles
