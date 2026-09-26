# GFS Real Data Folder

Put your GFS GRIB2 files here in the following structure:

```
raw_data/gfs/
└── YYYYMMDD_00z/           ← one folder per forecast cycle (date)
    ├── gfs.t00z.pgrb2.0p25.f000   ← analysis / truth
    ├── gfs.t00z.pgrb2.0p25.f024   ← 24h forecast
    ├── gfs.t00z.pgrb2.0p25.f072   ← 72h forecast
    └── gfs.t00z.pgrb2.0p25.f168   ← 168h forecast
```

For best results: add 10–30 cycles (days).
Run `python download_gfs.py` from this folder to auto-download.
