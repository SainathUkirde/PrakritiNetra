# Datasets — ForecastBlend Project

All real-world weather datasets used in this project are stored here.

---

## Folder Structure

```
datasets/
├── gfs/          ← GFS NWP forecast data          (nwp source)
├── gefs/         ← GEFS Ensemble mean forecast     (ensemble source)
├── pangu/        ← Pangu-Weather AI forecast       (ai source)
└── era5/         ← ERA5 Reanalysis observations    (ground truth / truth)
```

---

## Dataset Details

| Folder | Dataset | Source | Format | Role |
|--------|---------|--------|--------|------|
| `gfs/` | GFS 0.25deg deterministic | NOAA NOMADS | GRIB2 (.grb2) | NWP forecast |
| `gefs/` | GEFS ensemble mean (geavg) | AWS S3 noaa-gefs-pds | GRIB2 (.grb2) | Ensemble forecast |
| `pangu/` | Pangu-Weather AI forecast | WeatherBench2 / Google Cloud | NetCDF (.nc) | AI model forecast |
| `era5/` | ERA5 reanalysis | WeatherBench2 / Copernicus CDS | NetCDF (.nc) | Ground truth |

---

## Domain
- **Region**: India (lat 8–37°N, lon 68–97°E)
- **Grid**: 16×16 (~2° resolution)
- **Lead times**: 6h, 24h, 72h, 168h
- **Variables**: Rainfall (mm), Temperature (°C), Wind speed (m/s)

---

## How to Re-download

Run from the `forecast_blend/` folder:

```bash
# GFS (NWP)
python download_gfs.py --days 7

# GEFS (Ensemble)
python download_gefs.py --days 7

# Pangu (AI)
python download_pangu.py --days 3

# ERA5 (Truth)
python download_era5.py --days 7
```

---

## File Naming

| Source | File pattern | Example |
|--------|-------------|---------|
| GFS | `YYYYMMDD_00z/gfs.t00z.pgrb2.0p25.fXXX` | `20260922_00z/gfs.t00z.pgrb2.0p25.f024` |
| GEFS | `YYYYMMDD_00z/geavg.t00z.pgrb2s.0p25.fXXX` | `20260922_00z/geavg.t00z.pgrb2s.0p25.f024` |
| Pangu | `pangu_YYYYMMDD_YYYYMMDD.nc` | `pangu_20220101_20220103.nc` |
| ERA5 | `era5_YYYYMMDD_YYYYMMDD.nc` | `era5_20230104_20230110.nc` |
