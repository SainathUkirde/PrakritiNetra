"""
download_gefs.py
================
Downloads GEFS ensemble mean (geavg) GRIB2 files from AWS S3 for India domain.

Same lead times as GFS: f006, f024, f072, f168
Variables: TMP (2m temp), APCP (precip), UGRD+VGRD (10m wind)

Usage
-----
    python download_gefs.py            # last 3 days
    python download_gefs.py --days 7
    python download_gefs.py --date 20260922
"""

from __future__ import annotations

import argparse
import time
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

LAT_MIN, LAT_MAX = 8,  37
LON_MIN, LON_MAX = 68, 97

LEAD_TIMES = [6, 24, 72, 168]

OUT_DIR = Path(__file__).parent / "datasets" / "gefs"

# GEFS ensemble mean files on AWS S3 (no auth needed, requester-pays disabled for pgrb2s)
# geavg = ensemble mean (best for blending as a single "ensemble" source)
S3_BASE = "https://noaa-gefs-pds.s3.amazonaws.com"


def _build_url(date_str: str, lead: int) -> tuple[str, str]:
    fhour = f"f{lead:03d}"
    filename = f"geavg.t00z.pgrb2s.0p25.{fhour}"
    url = f"{S3_BASE}/gefs.{date_str}/00/atmos/pgrb2sp25/{filename}"
    return url, filename


def _download(url: str, dest: Path, retries: int = 3) -> bool:
    for attempt in range(1, retries + 1):
        try:
            print(f"    [{attempt}] {dest.name} ... ", end="", flush=True)
            urllib.request.urlretrieve(url, dest)
            size_kb = dest.stat().st_size // 1024
            print(f"OK ({size_kb} KB)")
            return True
        except Exception as e:
            print(f"FAILED - {e}")
            if dest.exists():
                dest.unlink()
            if attempt < retries:
                time.sleep(3)
    return False


def download_cycle(date_str: str) -> int:
    cycle_dir = OUT_DIR / f"{date_str}_00z"
    cycle_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  Cycle: {date_str}_00z -> {cycle_dir}")
    ok = 0
    for lead in LEAD_TIMES:
        url, filename = _build_url(date_str, lead)
        dest = cycle_dir / filename
        if dest.exists() and dest.stat().st_size > 10_000:
            print(f"    {filename} - already exists, skipping.")
            ok += 1
            continue
        if _download(url, dest):
            ok += 1
        time.sleep(1)
    return ok


def main():
    parser = argparse.ArgumentParser(description="Download GEFS ensemble mean GRIB2 files")
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--date", type=str, default=None)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.date:
        dates = [args.date]
    else:
        today = datetime.utcnow()
        dates = [(today - timedelta(days=i)).strftime("%Y%m%d") for i in range(1, args.days + 1)]

    print("=" * 60)
    print("GEFS Ensemble Mean Downloader - India domain")
    print(f"Lead times : {LEAD_TIMES}h")
    print(f"Output dir : {OUT_DIR.resolve()}")
    print(f"Cycles     : {len(dates)}")
    print("=" * 60)

    total = 0
    for d in dates:
        total += download_cycle(d)

    print(f"\n{'='*60}")
    print(f"Done. {total} GEFS files downloaded.")


if __name__ == "__main__":
    main()
