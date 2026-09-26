"""
data/gefs_loader.py
===================
Reads GEFS ensemble-mean GRIB2 files (geavg) from datasets/gefs/.

Variable mapping (confirmed from _inspect_gefs.py):
  shortName "2t"  → 2m temperature  (K → °C)
  shortName "tp"  → total precip    (m → mm, surface accumulation)
  shortName "10u" → U-wind 10m      (m/s)
  shortName "10v" → V-wind 10m      (m/s)

Folder layout expected:
    datasets/gefs/
    └── YYYYMMDD_00z/
        ├── geavg.t00z.pgrb2s.0p25.f006
        ├── geavg.t00z.pgrb2s.0p25.f024
        ├── geavg.t00z.pgrb2s.0p25.f072
        └── geavg.t00z.pgrb2s.0p25.f168
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

LAT_MIN, LAT_MAX = 8.0,  37.0
LON_MIN, LON_MAX = 68.0, 97.0
LEAD_TIMES = [6, 24, 72, 168]


def _kelvin_to_celsius(arr: np.ndarray) -> np.ndarray:
    return arr - 273.15


def _tp_kgm2_to_mm(arr: np.ndarray) -> np.ndarray:
    """Total precip in GEFS geavg pgrb2s is in kg/m² = mm. Just clip negatives."""
    return np.maximum(arr, 0.0)


def _wind_speed(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.sqrt(u ** 2 + v ** 2)


def _crop_and_flip(lat_vals, lon_vals, data):
    """Crop to India bbox, ensure ascending lat order."""
    lat_mask = (lat_vals >= LAT_MIN) & (lat_vals <= LAT_MAX)
    lon_mask = (lon_vals >= LON_MIN) & (lon_vals <= LON_MAX)
    cropped = data[lat_mask][:, lon_mask]
    lats = lat_vals[lat_mask]
    lons = lon_vals[lon_mask]
    if len(lats) > 1 and lats[0] > lats[-1]:
        cropped = cropped[::-1, :]
        lats = lats[::-1]
    return lats, lons, cropped


def _regrid(src_lats, src_lons, src_data, tgt_lats, tgt_lons):
    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator(
        (src_lats, src_lons), src_data,
        method="linear", bounds_error=False, fill_value=np.nan,
    )
    lon2d, lat2d = np.meshgrid(tgt_lons, tgt_lats)
    pts = np.column_stack([lat2d.ravel(), lon2d.ravel()])
    return interp(pts).reshape(len(tgt_lats), len(tgt_lons))


def read_gefs_file(fpath: Path, tgt_lats, tgt_lons) -> Optional[dict]:
    """Read one GEFS geavg GRIB2 file → dict of {variable: (lat,lon) array}."""
    import cfgrib
    result = {}

    # Temperature
    try:
        ds_t = xr.open_dataset(str(fpath), engine="cfgrib",
            backend_kwargs={"filter_by_keys": {"shortName": "2t"}},
            errors="ignore")
        lat_v = ds_t.latitude.values
        lon_v = ds_t.longitude.values
        raw = ds_t["t2m"].values.astype(np.float64)
        lats_c, lons_c, crp = _crop_and_flip(lat_v, lon_v, raw)
        result["temperature"] = _kelvin_to_celsius(_regrid(lats_c, lons_c, crp, tgt_lats, tgt_lons))
        result["_lats_c"] = lats_c
        result["_lons_c"] = lons_c
        init_ts = pd.Timestamp(ds_t.coords["time"].values)
        step_ns = int(ds_t.coords["step"].values)
        result["init_time"] = init_ts
        result["lead_hours"] = step_ns // (3600 * 1_000_000_000)
        ds_t.close()
    except Exception as e:
        warnings.warn(f"GEFS temp failed {fpath.name}: {e}")
        return None

    lats_c = result.pop("_lats_c")
    lons_c = result.pop("_lons_c")

    # Precipitation (tp = total precip in metres)
    try:
        ds_p = xr.open_dataset(str(fpath), engine="cfgrib",
            backend_kwargs={"filter_by_keys": {"shortName": "tp", "typeOfLevel": "surface"}},
            errors="ignore")
        raw = ds_p["tp"].values.astype(np.float64)
        _, _, crp = _crop_and_flip(ds_p.latitude.values, ds_p.longitude.values, raw)
        result["rainfall"] = _tp_kgm2_to_mm(_regrid(lats_c, lons_c, crp, tgt_lats, tgt_lons))
        ds_p.close()
    except Exception as e:
        warnings.warn(f"GEFS precip failed {fpath.name}: {e}")
        result["rainfall"] = np.zeros((len(tgt_lats), len(tgt_lons)))

    # Wind
    try:
        ds_u = xr.open_dataset(str(fpath), engine="cfgrib",
            backend_kwargs={"filter_by_keys": {"shortName": "10u"}}, errors="ignore")
        ds_v = xr.open_dataset(str(fpath), engine="cfgrib",
            backend_kwargs={"filter_by_keys": {"shortName": "10v"}}, errors="ignore")
        u = ds_u["u10"].values.astype(np.float64)
        v = ds_v["v10"].values.astype(np.float64)
        _, _, uc = _crop_and_flip(ds_u.latitude.values, ds_u.longitude.values, u)
        _, _, vc = _crop_and_flip(ds_v.latitude.values, ds_v.longitude.values, v)
        ug = _regrid(lats_c, lons_c, uc, tgt_lats, tgt_lons)
        vg = _regrid(lats_c, lons_c, vc, tgt_lats, tgt_lons)
        result["wind"] = _wind_speed(ug, vg)
        ds_u.close(); ds_v.close()
    except Exception as e:
        warnings.warn(f"GEFS wind failed {fpath.name}: {e}")
        result["wind"] = np.zeros((len(tgt_lats), len(tgt_lons)))

    return result


def load_gefs_cycles(gefs_dir: Path, tgt_lats, tgt_lons) -> list[dict]:
    """Load all GEFS cycles from gefs_dir. Returns list of cycle dicts."""
    cycle_dirs = sorted([d for d in gefs_dir.iterdir() if d.is_dir()])
    if not cycle_dirs:
        cycle_dirs = [gefs_dir]

    cycles = []
    for cd in cycle_dirs:
        n_lat, n_lon, n_lead = len(tgt_lats), len(tgt_lons), len(LEAD_TIMES)
        rain = np.full((n_lead, n_lat, n_lon), np.nan)
        temp = np.full((n_lead, n_lat, n_lon), np.nan)
        wind = np.full((n_lead, n_lat, n_lon), np.nan)
        init_time = None

        for li, lead in enumerate(LEAD_TIMES):
            matches = list(cd.glob(f"*f{lead:03d}*"))
            if not matches:
                continue
            rec = read_gefs_file(matches[0], tgt_lats, tgt_lons)
            if rec is None:
                continue
            rain[li] = rec["rainfall"]
            temp[li] = rec["temperature"]
            wind[li] = rec["wind"]
            if init_time is None:
                init_time = rec["init_time"]

        if init_time is not None:
            cycles.append({"init_time": init_time,
                           "rainfall": rain, "temperature": temp, "wind": wind})
    return cycles
