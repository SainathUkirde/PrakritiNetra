"""
download_gfs.py
===============
Auto-downloads GFS 0.25-degree GRIB2 files for India domain from NOAA NOMADS.

Downloads 4 lead times per cycle: f006, f024, f072, f168
Variables included: temperature (2m), precipitation rate, U/V wind (10m)
Region: India bounding box (lat 8-37N, lon 68-97E)

Usage
-----
    python download_gfs.py               # downloads last 7 days
    python download_gfs.py --days 14     # downloads last 14 days
    python download_gfs.py --date 20260920  # specific date

Output
------
    datasets/gfs/YYYYMMDD_00z/gfs.t00z.pgrb2.0p25.fXXX
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

# ── India domain (must match gfs_loader.py) ───────────────────────────────────
LAT_MIN, LAT_MAX = 8,  37
LON_MIN, LON_MAX = 68, 97

# Lead times to download
LEAD_TIMES = [6, 24, 72, 168]

# Variables to download (NOAA filter names)
# lev_2_m_above_ground  → TMP (temperature)
# lev_surface           → PRATE (precip rate)
# lev_10_m_above_ground → UGRD, VGRD (wind)
VARIABLE_FILTERS = (
    "var_TMP=on"
    "&var_PRATE=on"
    "&var_UGRD=on"
    "&var_VGRD=on"
    "&lev_2_m_above_ground=on"
    "&lev_surface=on"
    "&lev_10_m_above_ground=on"
)

NOMADS_BASE = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"

OUT_DIR = Path(__file__).parent / "datasets" / "gfs"


def _build_url(date_str: str, lead: int) -> str:
    """
    Build a NOMADS filter URL for one GFS file.
    date_str: YYYYMMDD (e.g. '20260923')
    lead:     lead time in hours (e.g. 24)
    """
    fhour = f"f{lead:03d}"
    filename = f"gfs.t00z.pgrb2.0p25.{fhour}"

    url = (
        f"{NOMADS_BASE}"
        f"?dir=%2Fgfs.{date_str}%2F00%2Fatmos"
        f"&file={filename}"
        f"&{VARIABLE_FILTERS}"
        f"&subregion="
        f"&leftlon={LON_MIN}&rightlon={LON_MAX}"
        f"&toplat={LAT_MAX}&bottomlat={LAT_MIN}"
    )
    return url, filename


def _download_file(url: str, dest: Path, retries: int = 3) -> bool:
    """Download url to dest. Returns True on success."""
    for attempt in range(1, retries + 1):
        try:
            print(f"    Downloading (attempt {attempt}): {dest.name} ... ", end="", flush=True)
            urllib.request.urlretrieve(url, dest)
            size_kb = dest.stat().st_size // 1024
            print(f"OK ({size_kb} KB)")
            return True
        except Exception as e:
            print(f"FAILED — {e}")
            if dest.exists():
                dest.unlink()
            if attempt < retries:
                time.sleep(3)
    return False


def download_cycle(date_str: str) -> int:
    """
    Download all lead-time files for one GFS cycle (00z run).
    Returns number of successfully downloaded files.
    """
    cycle_dir = OUT_DIR / f"{date_str}_00z"
    cycle_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Cycle: {date_str}_00z  ->  {cycle_dir}")
    ok_count = 0

    for lead in LEAD_TIMES:
        url, filename = _build_url(date_str, lead)
        dest = cycle_dir / filename

        if dest.exists() and dest.stat().st_size > 10_000:
            print(f"    {filename} — already exists, skipping.")
            ok_count += 1
            continue

        success = _download_file(url, dest)
        if success:
            ok_count += 1
        time.sleep(1)   # be polite to NOMADS

    return ok_count


def main():
    parser = argparse.ArgumentParser(
        description="Download GFS GRIB2 files for India domain from NOAA NOMADS",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--days",  type=int, default=7,
                        help="Number of past days to download (max ~7 for 0.25deg on NOMADS)")
    parser.add_argument("--date",  type=str, default=None,
                        help="Download a specific date only (YYYYMMDD)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.date:
        dates = [args.date]
    else:
        today = datetime.utcnow()
        # NOMADS keeps ~7 days of 0.25deg data
        dates = [
            (today - timedelta(days=i)).strftime("%Y%m%d")
            for i in range(1, args.days + 1)
        ]

    print("=" * 60)
    print("GFS Downloader — India domain (8-37N, 68-97E)")
    print(f"Lead times : {LEAD_TIMES}")
    print(f"Variables  : TMP, PRATE, UGRD, VGRD")
    print(f"Output dir : {OUT_DIR.resolve()}")
    print(f"Cycles     : {len(dates)}")
    print("=" * 60)

    total_ok = 0
    for d in dates:
        total_ok += download_cycle(d)

    print(f"\n{'='*60}")
    print(f"Done. {total_ok} files downloaded into {OUT_DIR.resolve()}")
    print()
    print("Next steps:")
    print("  1. In data/__init__.py — comment out synthetic, uncomment gfs_loader")
    print("  2. Run: python run_pipeline.py")
    print("  3. Run: streamlit run app.py")


if __name__ == "__main__":
    main()
