"""
download_pangu.py
=================
Downloads Pangu-Weather AI forecast data from WeatherBench2 Google Cloud zarr.

Source : gs://weatherbench2/datasets/pangu/2018-2022_0012_240x121...zarr
Time   : 2018-2022 daily (3652 init times)
Leads  : 6h, 24h, 72h, 168h  (from prediction_timedelta coord)
Region : India (lat 8-37N, lon 68-97E)

Variables available:
  2m_temperature           (K -> C)
  10m_u_component_of_wind  (m/s)
  10m_v_component_of_wind  (m/s)
  NOTE: Pangu does NOT forecast precipitation directly.
        Rainfall is derived in real_loader.py as a GFS-based proxy with
        AI-characteristic bias/noise (slight -0.3 mm bias, stable noise).

Output : datasets/pangu/pangu_YYYYMMDD_YYYYMMDD.nc

Usage
-----
    python download_pangu.py            # downloads 3 days
    python download_pangu.py --days 7
    python download_pangu.py --start 20220101 --end 20220107
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

LAT_MIN, LAT_MAX = 8.0,  37.0
LON_MIN, LON_MAX = 68.0, 97.0

# Lead times our pipeline uses (hours)
LEAD_TIMES = [6, 24, 72, 168]

PANGU_ZARR = (
    "gcs://weatherbench2/datasets/pangu/"
    "2018-2022_0012_240x121_equiangular_with_poles_conservative.zarr"
)

OUT_DIR = Path(__file__).parent / "datasets" / "pangu"


def download_pangu(start: str = "20220101", end: str = "20220103"):
    """
    Download Pangu AI forecasts for a date range and save as NetCDF.

    Parameters
    ----------
    start : YYYYMMDD start date (must be within 2018-2022)
    end   : YYYYMMDD end date   (inclusive)
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    t_start = pd.Timestamp(start)
    t_end   = pd.Timestamp(end)

    print("=" * 60)
    print("Pangu-Weather AI Forecast Downloader")
    print(f"Period     : {t_start.date()} to {t_end.date()}")
    print(f"Lead times : {LEAD_TIMES} h")
    print(f"Region     : lat {LAT_MIN}-{LAT_MAX}N, lon {LON_MIN}-{LON_MAX}E")
    print(f"Output dir : {OUT_DIR.resolve()}")
    print("=" * 60)

    print("\nOpening Pangu zarr store (may take 15-30s) ...")
    try:
        ds = xr.open_zarr(
            PANGU_ZARR,
            storage_options={"token": "anon"},
            consolidated=True,
        )
    except Exception as e:
        print(f"ERROR opening zarr: {e}")
        return

    print(f"Zarr opened. Time range: {str(ds.coords['time'].values[0])[:10]} "
          f"to {str(ds.coords['time'].values[-1])[:10]}")

    # Select time range
    ds_time = ds.sel(time=slice(t_start, t_end))
    n_times = len(ds_time.coords["time"])
    if n_times == 0:
        print(f"ERROR: No data for {t_start.date()} to {t_end.date()}.")
        print("Pangu zarr covers 2018-01-01 to 2022-12-31 only.")
        return
    print(f"Selected {n_times} init times.")

    # Select only needed lead times (prediction_timedelta coord is in hours)
    avail_leads = list(ds_time.coords["prediction_timedelta"].values)
    sel_leads   = [lt for lt in LEAD_TIMES if lt in avail_leads]
    if not sel_leads:
        print(f"ERROR: None of lead times {LEAD_TIMES} found. Available: {avail_leads[:10]}")
        return
    print(f"Lead times available and selected: {sel_leads} h")

    ds_leads = ds_time.sel(prediction_timedelta=sel_leads)

    # Crop to India domain
    # Pangu longitude is 0-360, India is 68-97E (same in 0-360 convention)
    # Pangu latitude is -90 to +90 (ascending)
    ds_india = ds_leads.sel(
        latitude=slice(LAT_MIN, LAT_MAX),
        longitude=slice(LON_MIN, LON_MAX),
    )

    # Select only surface variables we need (no pressure levels)
    keep = ["2m_temperature", "10m_u_component_of_wind",
            "10m_v_component_of_wind", "10m_wind_speed"]
    keep = [v for v in keep if v in ds_india.data_vars]
    ds_out = ds_india[keep]

    out_path = OUT_DIR / f"pangu_{t_start.strftime('%Y%m%d')}_{t_end.strftime('%Y%m%d')}.nc"
    print(f"\nDownloading and saving to {out_path} ...")
    print("(This streams real Pangu data from Google Cloud — may take 2-10 min)")

    ds_out.load()   # triggers actual download
    ds_out.to_netcdf(out_path)

    size_mb = out_path.stat().st_size // 1024 // 1024
    print(f"\nSaved: {out_path.name}  ({size_mb} MB)")
    print(f"Shape  : {n_times} times x {len(sel_leads)} lead times x "
          f"{len(ds_out.coords['latitude'])} lat x "
          f"{len(ds_out.coords['longitude'])} lon")
    print(f"Temp   : {float(ds_out['2m_temperature'].min()):.1f} to "
          f"{float(ds_out['2m_temperature'].max()):.1f} K")
    print("\nPangu download complete.")
    print("\nNext: run python run_pipeline.py to use Pangu as 'ai' source.")


def main():
    parser = argparse.ArgumentParser(
        description="Download Pangu-Weather AI forecasts from WeatherBench2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--days",  type=int, default=3,
                        help="Number of days (ending at --end or 2022-01-03)")
    parser.add_argument("--start", type=str, default=None,
                        help="Start date YYYYMMDD (overrides --days)")
    parser.add_argument("--end",   type=str, default="20220103",
                        help="End date YYYYMMDD (must be within 2018-2022)")
    args = parser.parse_args()

    if args.start:
        start = args.start
        end   = args.end
    else:
        t_end   = pd.Timestamp(args.end)
        t_start = t_end - pd.Timedelta(days=args.days - 1)
        start   = t_start.strftime("%Y%m%d")
        end     = t_end.strftime("%Y%m%d")

    download_pangu(start=start, end=end)


if __name__ == "__main__":
    main()
