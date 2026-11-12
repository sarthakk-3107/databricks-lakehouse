import numpy as np
import pandas as pd
import pytest

from lakehouse.geo import haversine_km
from lakehouse.modeling import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, build_preprocessor, time_split


def test_haversine_known_distance():
    # New York -> Los Angeles is ~3936 km
    assert haversine_km(40.7128, -74.0060, 34.0522, -118.2437) == pytest.approx(3936, rel=0.01)
    assert haversine_km(10, 10, 10, 10) == pytest.approx(0)


def test_time_split_is_ordered_and_disjoint():
    df = pd.DataFrame({"txn_ts": pd.date_range("2026-01-01", periods=1000, freq="h")})
    tr, va, te, (c1, c2) = time_split(df)
    assert tr.txn_ts.max() < va.txn_ts.min() <= va.txn_ts.max() < te.txn_ts.min()
    assert len(tr) + len(va) + len(te) == len(df)


def test_preprocessor_handles_nulls_and_unknown_categories():
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({c: rng.random(n) for c in NUMERIC_FEATURES})
    df.loc[:10, "amt_mean_30d"] = np.nan
    df["account_age_days"] = np.nan  # fully empty column (e.g. public data with no open_date)
    df["merchant_category"] = rng.choice(["grocery", "travel"], n)
    df["channel"] = rng.choice(["pos", "online"], n)
    prep = build_preprocessor().fit(df[FEATURES])
    new = df[FEATURES].head(3).copy()
    new["merchant_category"] = "never_seen"
    out = prep.transform(new)
    assert not np.isnan(out).any()
    assert out.shape[1] == len(prep.get_feature_names_out())
    assert set(CATEGORICAL_FEATURES) <= set(FEATURES)
