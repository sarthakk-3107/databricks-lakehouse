"""Feature lists, time-based splitting, and the shared sklearn preprocessor.

Both the training notebook (06) and the evaluation notebook (07) import from
here so the split and preprocessing can never silently drift apart.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

LABEL = "is_fraud"
ID_COLS = ["transaction_id", "account_id", "merchant_id", "txn_ts"]

# Non-negative, right-skewed features -> log1p before scaling.
HEAVY_TAILED_FEATURES = [
    "amount",
    "txn_cnt_1h",
    "txn_cnt_24h",
    "txn_cnt_7d",
    "amt_sum_24h",
    "small_txn_cnt_1h",
    "distinct_merchants_24h",
    "amt_mean_30d",
    "amt_ratio_30d",
    "secs_since_prev",
    "dist_prev_km",
    "speed_kmh",
    "dist_home_km",
    "account_age_days",
    "amt_to_limit",
    "merchant_prior_txn_cnt",
]

# Roughly symmetric / bounded / binary features -> impute + scale only.
LINEAR_FEATURES = [
    "amt_z_30d",
    "hour",
    "day_of_week",
    "is_night",
    "is_cross_border",
    "is_new_merchant",
    "is_new_category",
    "is_new_device",
    "merchant_prior_fraud_rate",
]

CATEGORICAL_FEATURES = ["merchant_category", "channel"]

NUMERIC_FEATURES = HEAVY_TAILED_FEATURES + LINEAR_FEATURES
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Columns kept for analysis only. NEVER add these to FEATURES:
#   fraud_scenario is synthetic ground truth and would be pure leakage.
ANALYSIS_ONLY_COLS = ["fraud_scenario"]


def log1p_clip(x):
    """log1p on values clipped at 0. Module-level so the pipeline pickles cleanly."""
    return np.log1p(np.clip(np.asarray(x, dtype=float), 0, None))


def build_preprocessor() -> ColumnTransformer:
    heavy = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("log", FunctionTransformer(log1p_clip, feature_names_out="one-to-one")),
            ("scale", StandardScaler()),
        ]
    )
    linear = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]
    )
    cat = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("heavy", heavy, HEAVY_TAILED_FEATURES),
            ("linear", linear, LINEAR_FEATURES),
            ("cat", cat, CATEGORICAL_FEATURES),
        ]
    )


def prepare_frame(pdf: pd.DataFrame) -> pd.DataFrame:
    """Normalise dtypes after Spark -> pandas (ints with nulls, tz-aware timestamps)."""
    pdf = pdf.copy()
    pdf[NUMERIC_FEATURES] = pdf[NUMERIC_FEATURES].astype("float64")
    for c in CATEGORICAL_FEATURES:
        pdf[c] = pdf[c].astype("object")
    ts = pd.to_datetime(pdf["txn_ts"])
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_convert(None)
    pdf["txn_ts"] = ts
    pdf[LABEL] = pdf[LABEL].astype(int)
    return pdf.sort_values("txn_ts").reset_index(drop=True)


def time_split(
    pdf: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    cutoffs: tuple | None = None,
):
    """Out-of-time split: train on the past, validate on the next window, test on the latest.

    Random splits leak future behaviour (velocity windows, merchant fraud rates)
    into training and overstate performance, so we never use them here.
    Returns (train, val, test, (train_end, val_end)).
    """
    ts = pd.to_datetime(pdf["txn_ts"])
    if cutoffs is None:
        c1 = ts.quantile(train_frac)
        c2 = ts.quantile(train_frac + val_frac)
    else:
        c1, c2 = (pd.Timestamp(c) for c in cutoffs)
    train = pdf[ts < c1]
    val = pdf[(ts >= c1) & (ts < c2)]
    test = pdf[ts >= c2]
    return train, val, test, (c1, c2)
