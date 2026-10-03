"""
config.py
=========
Centralised path resolution for PrakritiNetra.

Datasets folder resolution order (first path that exists wins):
  1. Environment variable  PRAKRITI_DATASETS_DIR  (absolute or relative)
  2. datasets/  sibling of this file  (normal cloned/installed layout)
  3. ../forecast_blend/datasets/  (running from a parent workspace folder)
  4. ../../forecast_blend/datasets/  (one more level up)

Set the env var to point at your data wherever it lives:
    Windows:  set PRAKRITI_DATASETS_DIR=C:\\path\\to\\forecast_blend\\datasets
    Linux:    export PRAKRITI_DATASETS_DIR=/path/to/forecast_blend/datasets

USE_SYNTHETIC environment variable
-----------------------------------
Setting USE_SYNTHETIC=1 forces synthetic demo data even when the datasets/
folder is present. This is the recommended setting for Render free-tier
deployments where attaching a 600 MB disk is not practical.
To enable real data on Render, upgrade to a paid plan, attach a persistent
disk (≥ 1 GB), upload your datasets/ folder there, and set USE_SYNTHETIC=0.
"""
from __future__ import annotations

import os
from pathlib import Path

_HERE = Path(__file__).parent


def _find_datasets_root() -> Path:
    """Return the first existing datasets/ root, searching several locations."""
    # 1. Explicit environment variable override
    env = os.environ.get("PRAKRITI_DATASETS_DIR", "").strip()
    if env:
        p = Path(env)
        if p.is_dir():
            return p.resolve()
        # Env var set but path doesn't exist — warn but don't crash yet
        import warnings
        warnings.warn(
            f"PRAKRITI_DATASETS_DIR='{env}' is set but the folder does not exist. "
            "Falling back to auto-detection.",
            stacklevel=2,
        )

    # 2–4. Auto-detect relative to this file
    candidates = [
        _HERE / "datasets",                          # sibling (normal layout)
        _HERE.parent / "forecast_blend" / "datasets",  # one level up
        _HERE.parent.parent / "forecast_blend" / "datasets",  # two levels up
    ]
    for c in candidates:
        if c.is_dir():
            return c.resolve()

    # Nothing found — return the default sibling path and let the loader
    # raise a descriptive FileNotFoundError later (better than a silent wrong path).
    return (_HERE / "datasets").resolve()


DATASETS_ROOT = _find_datasets_root()

GFS_DIR   = str(DATASETS_ROOT / "gfs")
GEFS_DIR  = str(DATASETS_ROOT / "gefs")
ERA5_DIR  = str(DATASETS_ROOT / "era5")
PANGU_DIR = str(DATASETS_ROOT / "pangu")


if __name__ == "__main__":
    print(f"DATASETS_ROOT : {DATASETS_ROOT}")
    print(f"GFS_DIR       : {GFS_DIR}  (exists={Path(GFS_DIR).is_dir()})")
    print(f"GEFS_DIR      : {GEFS_DIR} (exists={Path(GEFS_DIR).is_dir()})")
    print(f"ERA5_DIR      : {ERA5_DIR} (exists={Path(ERA5_DIR).is_dir()})")
    print(f"PANGU_DIR     : {PANGU_DIR}(exists={Path(PANGU_DIR).is_dir()})")
