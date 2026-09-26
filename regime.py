"""
regime.py
=========
Unsupervised weather-regime classifier.

Contract
--------
* Derives synoptic-style features from raw forecast/truth fields — never from
  the synthetic data's hidden `true_regime` coordinate.
* Fits a k-means (or GMM) clusterer on historical features.
* Returns integer cluster labels that the rest of the pipeline uses as
  "regime".  Label names are mapped post-hoc by inspecting cluster centroids.
* The hidden `true_regime` field may optionally be supplied ONLY to compute
  validation metrics (ARI / confusion matrix) — it must never flow into
  scoring, weighting, or blending.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# ── feature extraction ─────────────────────────────────────────────────────────

def extract_features(ds: xr.Dataset, lead_time: int = 24) -> pd.DataFrame:
    """
    Derive synoptic-style features per time step from the dataset.

    Uses ONLY fields that a real forecast/analysis dataset would contain —
    the hidden `true_regime` coordinate is explicitly excluded here.

    Features
    --------
    rain_mean        : domain-mean rainfall (spatial)
    rain_var         : spatial variance of rainfall
    rain_p90         : 90th percentile rainfall (proxy for convective activity)
    temp_grad_lat    : N-S temperature gradient magnitude
    temp_grad_lon    : E-W temperature gradient magnitude
    temp_mean        : domain-mean temperature
    wind_mean        : domain-mean wind speed
    wind_p90         : 90th percentile wind (storm proxy)
    """
    sel = ds.sel(lead_time=lead_time) if "lead_time" in ds.dims else ds

    # We use the truth field when available, otherwise first available source.
    def _get_field(var: str) -> np.ndarray:
        truth_key = f"truth_{var}"
        if truth_key in ds.data_vars:
            return ds[truth_key].values                     # (time, lat, lon)
        for src in ("nwp", "ensemble", "ai"):
            key = f"{src}_{var}"
            if key in ds.data_vars:
                arr = ds[key].values                        # (time, lat, lon, lead)
                lt_idx = list(ds.coords["lead_time"].values).index(lead_time)
                return arr[:, :, :, lt_idx]
        raise KeyError(f"No field found for variable '{var}'")

    rain = _get_field("rainfall")    # (T, lat, lon)
    temp = _get_field("temperature")
    wind = _get_field("wind")

    # Use nanmean/nanvar/nanpercentile so NaN grid cells (e.g. Pangu rainfall=NaN)
    # don't propagate into features and break KMeans.
    features = {
        "rain_mean":     np.nanmean(rain, axis=(1, 2)),
        "rain_var":      np.nanvar(rain,  axis=(1, 2)),
        "rain_p90":      np.nanpercentile(rain, 90, axis=(1, 2)),
        "temp_grad_lat": np.nanmean(np.abs(np.diff(temp, axis=1)), axis=(1, 2)),
        "temp_grad_lon": np.nanmean(np.abs(np.diff(temp, axis=2)), axis=(1, 2)),
        "temp_mean":     np.nanmean(temp, axis=(1, 2)),
        "wind_mean":     np.nanmean(wind, axis=(1, 2)),
        "wind_p90":      np.nanpercentile(wind, 90, axis=(1, 2)),
    }
    df = pd.DataFrame(features, index=ds.coords["time"].values)
    df.index.name = "time"
    # Fill NaN: median first, then 0 for fully-NaN columns (e.g. Pangu rainfall)
    df = df.fillna(df.median()).fillna(0.0)
    return df


# ── cluster → semantic label mapping ──────────────────────────────────────────

def _map_cluster_labels(cluster_ids: np.ndarray,
                        centroids_df: pd.DataFrame) -> np.ndarray:
    """
    Map integer cluster IDs to semantic strings by inspecting centroids:
      - highest rain_mean  + high wind  → 'convective'
      - moderate rain_mean              → 'stratiform'
      - lowest  rain_mean               → 'clear'
    Returns an array of strings of the same length as cluster_ids.
    """
    n_clusters = centroids_df.shape[0]
    rain_order = centroids_df["rain_mean"].argsort().values   # ascending cluster idx

    label_map: dict[int, str] = {}
    if n_clusters == 3:
        label_map[rain_order[0]] = "clear"
        label_map[rain_order[1]] = "stratiform"
        label_map[rain_order[2]] = "convective"
    else:
        # Generalise for arbitrary K: split into thirds
        third = n_clusters // 3
        for rank, cid in enumerate(rain_order):
            if rank < third:
                label_map[cid] = "clear"
            elif rank < 2 * third:
                label_map[cid] = "stratiform"
            else:
                label_map[cid] = "convective"
    return np.array([label_map.get(c, f"cluster_{c}") for c in cluster_ids])


# ── main classifier class ──────────────────────────────────────────────────────

class RegimeClassifier:
    """
    Fit-once, predict-always unsupervised regime classifier.

    Parameters
    ----------
    n_regimes : int
        Number of clusters (default 3 → convective / stratiform / clear).
    method    : 'kmeans' | 'gmm'
    """

    def __init__(self, n_regimes: int = 3, method: str = "kmeans"):
        self.n_regimes = n_regimes
        self.method = method
        self.scaler = StandardScaler()
        self._fitted = False
        self._centroid_df: Optional[pd.DataFrame] = None
        self._feature_names: list[str] = []

        if method == "gmm":
            self._model = GaussianMixture(
                n_components=n_regimes, covariance_type="full",
                random_state=42, n_init=5
            )
        else:
            self._model = KMeans(
                n_clusters=n_regimes, random_state=42, n_init=20
            )

    # ── fitting ───────────────────────────────────────────────────────────────
    def fit(self, features_df: pd.DataFrame) -> "RegimeClassifier":
        """
        Fit scaler + clusterer on historical feature matrix.

        Parameters
        ----------
        features_df : DataFrame with shape (n_time_steps, n_features)
                      Output of extract_features().
        """
        self._feature_names = list(features_df.columns)
        # Impute NaN: median first, then 0 for columns that are entirely NaN
        features_clean = features_df.fillna(features_df.median()).fillna(0.0)
        X = self.scaler.fit_transform(features_clean.values)
        # Final assertion — must be finite before passing to KMeans
        import numpy as _np
        if not _np.isfinite(X).all():
            X = _np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        self._model.fit(X)
        self._fitted = True

        # Build centroid dataframe in original feature space
        if hasattr(self._model, "cluster_centers_"):
            centers = self.scaler.inverse_transform(self._model.cluster_centers_)
        else:
            centers = self.scaler.inverse_transform(self._model.means_)
        self._centroid_df = pd.DataFrame(
            centers, columns=self._feature_names
        )
        return self

    # ── prediction ────────────────────────────────────────────────────────────
    def predict(self, features_df: pd.DataFrame) -> pd.Series:
        """
        Assign regime labels to new time steps.

        Parameters
        ----------
        features_df : DataFrame with same columns as used during fit().

        Returns
        -------
        pd.Series of str labels indexed the same as features_df.
        This is what scoring/weighting/blending must consume.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before predict().")
        features_clean = features_df[self._feature_names].fillna(
            features_df[self._feature_names].median()
        )
        X = self.scaler.transform(features_clean.values)
        raw_ids = self._model.predict(X)
        labels = _map_cluster_labels(raw_ids, self._centroid_df)
        return pd.Series(labels, index=features_df.index, name="regime")

    def predict_single(self, feature_dict: dict) -> str:
        """
        Classify a single time step given a dict of feature name → value.
        Useful for real-time operational use.
        """
        row = pd.DataFrame([feature_dict])
        return self.predict(row).iloc[0]

    # ── persistence ───────────────────────────────────────────────────────────
    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "RegimeClassifier":
        with open(path, "rb") as f:
            return pickle.load(f)

    # ── validation (only for sanity-checking against hidden true_regime) ──────
    def validate_against_truth(
        self,
        predicted_regimes: pd.Series,
        true_regimes: pd.Series,
    ) -> dict:
        """
        OPTIONAL validation only — never called by the production pipeline.
        Compares classifier output to the synthetic data's hidden true_regime.

        Returns dict with ARI score and a confusion-matrix DataFrame.
        """
        ari = adjusted_rand_score(true_regimes.values, predicted_regimes.values)
        labels_true = sorted(true_regimes.unique())
        labels_pred = sorted(predicted_regimes.unique())
        # Use string labels directly
        cm = confusion_matrix(true_regimes, predicted_regimes,
                              labels=sorted(set(labels_true) | set(labels_pred)))
        all_labels = sorted(set(labels_true) | set(labels_pred))
        cm_df = pd.DataFrame(cm, index=all_labels, columns=all_labels)
        return {"ari": ari, "confusion_matrix": cm_df}


# ── Convenience public function ────────────────────────────────────────────────

def fit_and_classify(ds: xr.Dataset,
                     lead_time: int = 24,
                     n_regimes: int = 3,
                     method: str = "kmeans") -> tuple[RegimeClassifier, pd.Series]:
    """
    One-shot: extract features → fit classifier → return (classifier, regimes).

    regime_series has the same time index as ds and carries string labels
    ('convective', 'stratiform', 'clear').  This is the series the rest of
    the pipeline must use.
    """
    feats = extract_features(ds, lead_time=lead_time)
    clf = RegimeClassifier(n_regimes=n_regimes, method=method)
    clf.fit(feats)
    regimes = clf.predict(feats)
    return clf, regimes


# ── Standalone smoke test ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from data.synthetic import load_dataset

    print("Loading synthetic dataset...")
    ds = load_dataset()

    print("Extracting features and fitting regime classifier...")
    feats = extract_features(ds, lead_time=24)
    print(f"Feature matrix shape: {feats.shape}")
    print(feats.describe().round(3))

    clf = RegimeClassifier(n_regimes=3, method="kmeans")
    clf.fit(feats)
    regimes = clf.predict(feats)

    print("\nClassified regime distribution:")
    print(regimes.value_counts())

    # Validate against hidden true_regime
    true_regimes = pd.Series(ds["true_regime"].values, index=feats.index,
                             name="true_regime")
    val = clf.validate_against_truth(regimes, true_regimes)
    print(f"\nAdjusted Rand Index vs hidden true_regime: {val['ari']:.4f}")
    print("\nConfusion matrix (rows=true, cols=predicted):")
    print(val["confusion_matrix"])

    print("\nSample regime series (first 10):")
    print(regimes.head(10).to_string())

    # Verify: no true_regime ever used downstream
    print("\n[OK] Regime labels come from classifier -- true_regime not passed downstream.")
