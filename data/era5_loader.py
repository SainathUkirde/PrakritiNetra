"""
data/era5_loader.py
===================
Reads ERA5 reanalysis NetCDF files from datasets/era5/ as ground truth.

Variable mapping (from download_era5.py / WeatherBench2 zarr):
  2m_temperature           → truth_temperature  (K → °C)
  total_precipitation_6hr  → truth_rainfall      (m → mm, already 6h accumulation)
  10m_u_component_of_wind  → combined →
  10m_v_component_of_wind  → truth_wind          (m/s magnitude)

Coordinate convention in the NetCDF:
  time      : datetime64  (6-hourly)
  latitude  : float64     (ascending, e.g. 9.0 … 36.0)
  longitude : float64     (ascending, e.g. 69.0 … 96.0)
  NOTE: dims order in file is (time, longitude, latitude) — transposed vs
        standard convention. This loader normalises to (time, lat, lon).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

LAT_MIN, LAT_MAX = 8.0,  37.0
LON_MIN, LON_MAX = 68.0, 97.0


def _kelvin_to_celsius(arr: np.ndarray) -> np.ndarray:
    return arr - 273.15


def _precip_m_to_mm(arr: np.ndarray) -> np.ndarray:
    """ERA5 total_precipitation_6hr is in metres → mm. Clip negatives."""
    return np.maximum(arr * 1000.0, 0.0)


def _wind_speed(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.sqrt(u ** 2 + v ** 2)


def _regrid_to_target(src_lats, src_lons, src_data, tgt_lats, tgt_lons):
    """Bilinear interpolation to target grid."""
    from scipy.interpolate import RegularGridInterpolator
    # Ensure ascending lat order
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


def load_era5_truth(era5_dir: Path, tgt_lats: np.ndarray,
                    tgt_lons: np.ndarray) -> dict[str, object]:
    """
    Load all ERA5 NetCDF files in era5_dir and return aligned truth arrays.

    Returns
    -------
    dict with keys:
        "times"       : pd.DatetimeIndex
        "rainfall"    : (n_time, n_lat, n_lon)  float64  mm/6h
        "temperature" : (n_time, n_lat, n_lon)  float64  °C
        "wind"        : (n_time, n_lat, n_lon)  float64  m/s
        "seasons"     : (n_time,)               str
    """
    nc_files = sorted(era5_dir.glob("era5_*.nc"))
    if not nc_files:
        raise FileNotFoundError(
            f"No ERA5 NetCDF files found in {era5_dir}. "
            "Run download_era5.py first."
        )

    # Load and concatenate all files
    datasets = []
    for f in nc_files:
        try:
            ds = xr.open_dataset(f)
            datasets.append(ds)
        except Exception as e:
            warnings.warn(f"Could not open {f.name}: {e}")

    if not datasets:
        raise ValueError("No ERA5 files could be opened.")

    ds_all = xr.concat(datasets, dim="time").sortby("time")

    times = pd.DatetimeIndex(ds_all.coords["time"].values)
    n_time = len(times)
    n_lat  = len(tgt_lats)
    n_lon  = len(tgt_lons)

    # ERA5 file dims are (time, longitude, latitude) — non-standard order
    # Determine actual dimension order from the file
    t2m_raw = ds_all["2m_temperature"].values       # may be (T, lon, lat) or (T, lat, lon)
    prcp_raw = ds_all["total_precipitation_6hr"].values
    u_raw    = ds_all["10m_u_component_of_wind"].values
    v_raw    = ds_all["10m_v_component_of_wind"].values

    file_lats = ds_all.coords["latitude"].values
    file_lons = ds_all.coords["longitude"].values

    # Detect if dims are transposed: if shape[-1] matches lats and shape[-2] matches lons
    if t2m_raw.ndim == 3:
        if t2m_raw.shape[1] == len(file_lons) and t2m_raw.shape[2] == len(file_lats):
            # (time, lon, lat) — transpose to (time, lat, lon)
            t2m_raw  = t2m_raw.transpose(0, 2, 1)
            prcp_raw = prcp_raw.transpose(0, 2, 1)
            u_raw    = u_raw.transpose(0, 2, 1)
            v_raw    = v_raw.transpose(0, 2, 1)

    truth_temp = np.full((n_time, n_lat, n_lon), np.nan)
    truth_rain = np.full((n_time, n_lat, n_lon), np.nan)
    truth_wind = np.full((n_time, n_lat, n_lon), np.nan)

    for t in range(n_time):
        truth_temp[t] = _regrid_to_target(
            file_lats, file_lons, _kelvin_to_celsius(t2m_raw[t].astype(np.float64)),
            tgt_lats, tgt_lons)
        truth_rain[t] = _regrid_to_target(
            file_lats, file_lons, _precip_m_to_mm(prcp_raw[t].astype(np.float64)),
            tgt_lats, tgt_lons)
        truth_wind[t] = _regrid_to_target(
            file_lats, file_lons,
            _wind_speed(u_raw[t].astype(np.float64), v_raw[t].astype(np.float64)),
            tgt_lats, tgt_lons)

    seasons = np.array([
        "monsoon" if t.month in (6, 7, 8, 9) else "winter"
        for t in times
    ])

    return {
        "times":       times,
        "rainfall":    truth_rain,
        "temperature": truth_temp,
        "wind":        truth_wind,
        "seasons":     seasons,
    }
