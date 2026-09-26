"""
download_era5.py
================
Downloads ERA5 reanalysis data via WeatherBench2 Google Cloud zarr store.
Used as ground-truth (observations) to replace GFS f000 analysis as truth.

Variables downloaded:
  - 2m_temperature         (K -> converted to C in loader)
  - total_precipitation_6hr (m -> mm)
  - 10m_u_component_of_wind
  - 10m_v_component_of_wind

Region: India (lat 8-37N, lon 68-97E)
Time: last N days at 6-hourly resolution

Output: datasets/era5/era5_YYYYMMDD.nc  (one file per day)

Usage
-----
    python download_era5.py            # last 3 days
    python download_era5.py --days 7
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

LAT_MIN, LAT_MAX = 8.0,  37.0
LON_MIN, LON_MAX = 68.0, 97.0

OUT_DIR = Path(__file__).parent / "datasets" / "era5"

# WeatherBench2 ERA5 zarr — publicly accessible, no auth
WB2_ERA5_ZARR = (
    "https://storage.googleapis.com/weatherbench2/datasets/era5/"
    "1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
)

VARIABLES = [
    "2m_temperature",
    "total_precipitation_6hr",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
]


def download_era5(days: int = 3):
    try:
        import xarray as xr
        import pandas as pd
    except ImportError:
        print("ERROR: xarray not installed. Run: pip install xarray zarr gcsfs")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    today = datetime.utcnow()
    # WeatherBench2 ERA5 only goes up to end of 2023 — use recent available period
    # Use 2022-2023 data which is guaranteed in the zarr
    end_date   = datetime(2023, 9, 23)
    start_date = end_date - timedelta(days=days - 1)

    print("=" * 60)
    print("ERA5 Downloader via WeatherBench2 zarr")
    print(f"Period     : {start_date.date()} to {end_date.date()}")
    print(f"Variables  : {VARIABLES}")
    print(f"Region     : lat {LAT_MIN}-{LAT_MAX}N, lon {LON_MIN}-{LON_MAX}E")
    print(f"Output dir : {OUT_DIR.resolve()}")
    print("=" * 60)

    print("\nOpening ERA5 zarr store (this may take 10-30s) ...")
    try:
        ds = xr.open_zarr(
            WB2_ERA5_ZARR,
            storage_options={"token": "anon"},
            consolidated=True,
        )
    except Exception as e:
        print(f"ERROR opening zarr: {e}")
        print("Trying without gcsfs token...")
        try:
            import fsspec
            ds = xr.open_zarr(
                "gcs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr",
                storage_options={"token": "anon"},
                consolidated=True,
            )
        except Exception as e2:
            print(f"ERROR: {e2}")
            return

    print(f"Zarr opened. Variables: {list(ds.data_vars)}")

    # Inspect actual coordinate names
    coord_names = list(ds.coords)
    print(f"Coordinates: {coord_names}")
    time_coord = "time" if "time" in coord_names else coord_names[0]
    lat_coord  = "latitude"  if "latitude"  in coord_names else "lat"
    lon_coord  = "longitude" if "longitude" in coord_names else "lon"

    # Print available time range in the zarr
    times = ds.coords[time_coord].values
    print(f"Available time range: {str(times[0])[:10]} to {str(times[-1])[:10]}")

    # Use a fixed window guaranteed within the zarr (ends 2023-01-10)
    t_end   = pd.Timestamp("2023-01-10")
    t_start = t_end - pd.Timedelta(days=days - 1)
    print(f"Selecting: {t_start.date()} to {t_end.date()}")

    # Crop to India domain
    # ERA5 latitudes are descending (90→-90), so slice(MAX,MIN) is correct for .sel
    # But after cropping we flip to ascending order for consistency
    ds_india = ds.sel(
        {lat_coord: slice(LAT_MAX, LAT_MIN), lon_coord: slice(LON_MIN, LON_MAX)},
    )
    # If latitude is still empty, try ascending slice (some zarrs are ascending)
    if ds_india.dims.get(lat_coord, 0) == 0:
        ds_india = ds.sel(
            {lat_coord: slice(LAT_MIN, LAT_MAX), lon_coord: slice(LON_MIN, LON_MAX)},
        )

    # Select time range
    ds_india = ds_india.sel({time_coord: slice(t_start, t_end)})

    # Select only needed variables (skip if not present)
    keep_vars = [v for v in VARIABLES if v in ds_india.data_vars]
    if not keep_vars:
        print(f"ERROR: None of {VARIABLES} found. Available: {list(ds.data_vars)[:10]}")
        return

    ds_out = ds_india[keep_vars]
    print(f"Selected {len(ds_out.coords[time_coord])} time steps, variables: {keep_vars}")

    out_path = OUT_DIR / f"era5_{t_start.strftime('%Y%m%d')}_{t_end.strftime('%Y%m%d')}.nc"
    print(f"\nDownloading and saving to {out_path} ...")
    print("(Downloading real ERA5 data - may take 1-5 minutes)")

    ds_out.load()   # trigger actual download
    # rename coords to standard names for loader compatibility
    rename = {}
    if lat_coord != "latitude": rename[lat_coord] = "latitude"
    if lon_coord != "longitude": rename[lon_coord] = "longitude"
    if rename:
        ds_out = ds_out.rename(rename)
    ds_out.to_netcdf(out_path)

    size_mb = out_path.stat().st_size // 1024 // 1024
    print(f"\nSaved: {out_path} ({size_mb} MB)")
    print(f"Time steps : {len(ds_out.coords[time_coord])}")
    print("\nERA5 download complete.")


def main():
    parser = argparse.ArgumentParser(description="Download ERA5 from WeatherBench2 zarr")
    parser.add_argument("--days", type=int, default=3, help="Number of days to download")
    args = parser.parse_args()
    download_era5(days=args.days)


if __name__ == "__main__":
    main()
