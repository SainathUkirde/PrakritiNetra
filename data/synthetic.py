"""
data/synthetic.py
=================
Generates synthetic gridded weather forecast data for India-sized domain.

Design contract
---------------
* load_dataset() is the ONLY public entry point.
* It returns an xarray.Dataset whose shape/coords mirror what a real loader
  (GFS/GEFS/ERA5/WeatherBench2) would return.
* Swapping this file for a real loader later requires NO changes to
  regime.py, scoring.py, weighting.py, blending.py, or app.py.

Synthetic properties
--------------------
Three forecast sources (nwp, ensemble, ai) each have deliberately distinct
bias/noise profiles as a function of lead time:

  NWP       : Low bias short-term → grows quickly with lead time.
               Good at heavy rainfall, poor by day 7.
  Ensemble  : Moderate smoothing bias (underestimates extremes);
               moderately stable across lead times.
  AI        : Slight cold bias short-term, very stable at long lead time.

Each variable (rainfall, temperature, wind) also has per-source offsets.

The dataset contains a `true_regime` coordinate (convective/stratiform/clear)
that is ONLY for validating the unsupervised classifier — never passed to
scoring/weighting/blending.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

# ── domain constants ──────────────────────────────────────────────────────────
LAT_MIN, LAT_MAX = 8.0, 37.0   # India bounding box
LON_MIN, LON_MAX = 68.0, 97.0
N_LAT = 16
N_LON = 16

LEAD_TIMES = [6, 24, 72, 168]   # hours
VARIABLES = ["rainfall", "temperature", "wind"]
SOURCES = ["nwp", "ensemble", "ai"]

# Season thresholds (month → season)
def _month_to_season(month: int) -> str:
    if month in (6, 7, 8, 9):
        return "monsoon"
    return "winter"


# ── bias/noise profiles ───────────────────────────────────────────────────────
# Per (source, variable) → callable(lead_time_hours) → bias scalar
def _bias(source: str, variable: str, lead: int) -> float:
    lead_day = lead / 24.0
    if source == "nwp":
        base = {"rainfall": 0.5, "temperature": 0.3, "wind": 0.2}
        return base[variable] * (1 + 0.4 * lead_day)
    elif source == "ensemble":
        base = {"rainfall": -0.8, "temperature": -0.2, "wind": -0.3}
        return base[variable] * (1 + 0.05 * lead_day)   # smoother, stable
    else:  # ai
        base = {"rainfall": -0.3, "temperature": -0.5, "wind": 0.1}
        return base[variable] * (1 - 0.1 * lead_day)    # improves at long lead


def _noise_std(source: str, variable: str, lead: int) -> float:
    lead_day = lead / 24.0
    if source == "nwp":
        base = {"rainfall": 1.5, "temperature": 1.0, "wind": 0.8}
        return base[variable] * (1 + 0.35 * lead_day)
    elif source == "ensemble":
        base = {"rainfall": 1.0, "temperature": 0.7, "wind": 0.6}
        return base[variable] * (1 + 0.10 * lead_day)
    else:  # ai
        base = {"rainfall": 1.8, "temperature": 1.2, "wind": 1.0}
        return base[variable] * (1 + 0.05 * lead_day)


# ── "truth" field construction ────────────────────────────────────────────────
def _make_truth(rng: np.random.Generator, n_time: int) -> dict[str, np.ndarray]:
    """Truth shape: (time, lat, lon)"""
    lats = np.linspace(LAT_MIN, LAT_MAX, N_LAT)
    lons = np.linspace(LON_MIN, LON_MAX, N_LON)
    lon2d, lat2d = np.meshgrid(lons, lats)

    # Rainfall: heavier in south/west, skewed positive
    # Scale factor will be modulated by season outside this function
    rain_spatial = 2.0 + 3.0 * np.exp(-0.05 * (lat2d - LAT_MIN))
    rain = rng.gamma(shape=2.0, scale=1.0, size=(n_time, N_LAT, N_LON)) * rain_spatial

    # Temperature: warmer in north-west plains, seasonal modulation applied later
    temp_spatial = 25.0 + 5.0 * (lon2d - LON_MIN) / (LON_MAX - LON_MIN)
    temp = temp_spatial + rng.normal(0, 2, size=(n_time, N_LAT, N_LON))

    # Wind: stronger along coasts
    coast_effect = 2.0 * np.exp(-0.1 * np.abs(lon2d - LON_MIN))
    wind = np.abs(rng.normal(5, 2, size=(n_time, N_LAT, N_LON))) + coast_effect

    return {"rainfall": rain, "temperature": temp, "wind": wind}


# ── regime assignment (hidden ground truth for validation only) ───────────────
def _assign_true_regime(rain_mean: np.ndarray, temp_mean: np.ndarray,
                        wind_mean: np.ndarray) -> list[str]:
    """
    Deterministic rule-based true regime — exists ONLY for validating the
    unsupervised classifier.  Never exposed to scoring/weighting pipeline.
    """
    regimes = []
    for r, t, w in zip(rain_mean, temp_mean, wind_mean):
        if r > 6.0 and w > 6.5:
            regimes.append("convective")
        elif r > 2.0:
            regimes.append("stratiform")
        else:
            regimes.append("clear")
    return regimes


# ── main generator ────────────────────────────────────────────────────────────
def load_dataset(
    n_time: int = 240,
    seed: int = 42,
) -> xr.Dataset:
    """
    Public entry point.  Returns an xarray.Dataset with dimensions
    (time, lat, lon, lead_time) and variables:

        truth_{var}          : (time, lat, lon)      — observed field
        {source}_{var}       : (time, lat, lon, lead_time) — forecast fields
        season               : (time,)               — 'monsoon' | 'winter'
        true_regime          : (time,)               — ONLY for classifier QA

    Coordinates
    -----------
    time      : pandas DatetimeIndex
    lat       : float array
    lon       : float array
    lead_time : int array (hours)
    source    : str — encoded in variable names, and also as a coord for stacking

    Swap contract
    -------------
    Replace this function with a real GFS/ERA5 loader that returns a Dataset
    with the same variable names and dimension structure.  All downstream code
    imports only load_dataset from data/__init__.py.
    """
    rng = np.random.default_rng(seed)

    lats = np.linspace(LAT_MIN, LAT_MAX, N_LAT)
    lons = np.linspace(LON_MIN, LON_MAX, N_LON)

    # Time axis: 2 full years at 6-hourly resolution → covers both monsoon and
    # winter seasons.  n_time steps are sampled uniformly without replacement
    # across the full 2-year window so season coverage is guaranteed.
    total_slots = 365 * 2 * 4   # 2 years × 4 six-hour periods/day = 2920 slots
    chosen = np.sort(rng.choice(total_slots, size=min(n_time, total_slots),
                                replace=False))
    base  = pd.Timestamp("2022-01-01")
    times = pd.DatetimeIndex(
        [base + pd.Timedelta(hours=int(h) * 6) for h in chosen]
    )

    # Season per time step
    seasons = np.array([_month_to_season(t.month) for t in times])

    # Temperature seasonal modulation (monsoon warmer, winter cooler)
    season_temp_offset = np.where(seasons == "monsoon", 5.0, -5.0)  # (time,)

    # Build truth fields
    truth = _make_truth(rng, n_time)
    # Apply seasonal temp offset
    truth["temperature"] += season_temp_offset[:, None, None]
    # Monsoon: heavier rainfall (multiplicative scale 2-3x), winter: lighter
    rain_season_scale = np.where(seasons == "monsoon", 2.5, 0.5)
    truth["rainfall"] *= rain_season_scale[:, None, None]
    truth["rainfall"]  = np.maximum(truth["rainfall"], 0.0)
    # Monsoon: stronger winds
    wind_season_scale  = np.where(seasons == "monsoon", 1.4, 0.8)
    truth["wind"]     *= wind_season_scale[:, None, None]

    # Build hidden regime label
    rain_mean = truth["rainfall"].mean(axis=(1, 2))
    temp_mean = truth["temperature"].mean(axis=(1, 2))
    wind_mean = truth["wind"].mean(axis=(1, 2))
    true_regime = _assign_true_regime(rain_mean, temp_mean, wind_mean)

    # Build forecast fields per source × lead_time
    data_vars: dict[str, tuple] = {}

    for var in VARIABLES:
        truth_field = truth[var]
        data_vars[f"truth_{var}"] = (["time", "lat", "lon"], truth_field)

        for src in SOURCES:
            forecast = np.zeros((n_time, N_LAT, N_LON, len(LEAD_TIMES)))
            for li, lead in enumerate(LEAD_TIMES):
                bias = _bias(src, var, lead)
                noise = _noise_std(src, var, lead)
                spatial_noise_factor = rng.uniform(0.8, 1.2, (N_LAT, N_LON))
                err = rng.normal(bias, noise, (n_time, N_LAT, N_LON)) * spatial_noise_factor
                fc = truth_field + err
                if var == "rainfall":
                    fc = np.maximum(fc, 0.0)   # non-negative rainfall
                forecast[:, :, :, li] = fc
            data_vars[f"{src}_{var}"] = (
                ["time", "lat", "lon", "lead_time"], forecast
            )

    # ── Compose xarray.Dataset ────────────────────────────────────────────────
    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "time": times,
            "lat": lats,
            "lon": lons,
            "lead_time": LEAD_TIMES,
            "season": ("time", seasons),
            "true_regime": ("time", true_regime),
        },
    )
    ds.attrs["description"] = "Synthetic blending benchmark dataset"
    ds.attrs["domain"] = f"lat [{LAT_MIN},{LAT_MAX}] lon [{LON_MIN},{LON_MAX}]"
    ds.attrs["sources"] = ",".join(SOURCES)
    ds.attrs["variables"] = ",".join(VARIABLES)
    ds.attrs["lead_times_hours"] = str(LEAD_TIMES)
    ds.attrs["note_true_regime"] = (
        "true_regime is present ONLY for classifier validation — "
        "it must NOT be consumed by scoring/weighting/blending."
    )

    return ds


# ── Standalone smoke test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    ds = load_dataset()
    print(ds)
    print("\nSeason distribution:\n", pd.Series(ds["season"].values).value_counts())
    print("\nTrue-regime distribution:\n",
          pd.Series(ds["true_regime"].values).value_counts())
    print("\nSample truth_rainfall stats:")
    print(ds["truth_rainfall"].to_series().describe())
    print("\nNWP rainfall (lead=6h) bias vs truth:")
    err = (ds["nwp_rainfall"].sel(lead_time=6) - ds["truth_rainfall"]).mean().item()
    print(f"  mean error = {err:.4f}")
    print("\nAI rainfall (lead=168h) bias vs truth:")
    err2 = (ds["ai_rainfall"].sel(lead_time=168) - ds["truth_rainfall"]).mean().item()
    print(f"  mean error = {err2:.4f}")
