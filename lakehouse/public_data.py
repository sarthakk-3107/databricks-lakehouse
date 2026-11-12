"""Adapter: Sparkov synthetic credit-card dataset -> this project's landing schema.

Dataset: "Credit Card Transactions Fraud Detection Dataset" on Kaggle
(fraudTrain.csv + fraudTest.csv, generated with the Sparkov tool). It is
independent of this repo's generator, so model results on it are a fairer
check that the features generalise beyond rules we wrote ourselves.

Expected columns: trans_date_trans_time, cc_num, merchant, category, amt,
city, lat, long, merch_lat, merch_long, trans_num, is_fraud (others ignored).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED = [
    "trans_date_trans_time", "cc_num", "merchant", "category", "amt",
    "city", "lat", "long", "merch_lat", "merch_long", "trans_num", "is_fraud",
]


def from_sparkov(raw: pd.DataFrame) -> dict[str, pd.DataFrame]:
    missing = [c for c in REQUIRED if c not in raw.columns]
    if missing:
        raise ValueError(f"Sparkov file is missing columns: {missing}")

    df = raw.copy()
    ts = pd.to_datetime(df["trans_date_trans_time"])
    online = df["category"].str.endswith("_net").to_numpy()
    category = df["category"].str.replace(r"_(net|pos)$", "", regex=True)
    codes, names = pd.factorize(df["merchant"])
    merchant_ids = np.array([f"M{i:05d}" for i in range(len(names))])
    account_ids = "A" + df["cc_num"].astype(str)

    tx = pd.DataFrame(
        {
            "transaction_id": df["trans_num"].astype(str),
            "account_id": account_ids,
            "merchant_id": merchant_ids[codes],
            "merchant_category": category,
            "amount": df["amt"].astype(float),
            "currency": "USD",
            "txn_ts": ts.dt.strftime("%Y-%m-%d %H:%M:%S"),
            "channel": np.where(online, "online", "pos"),
            "device_id": None,
            "txn_city": np.where(online, df["city"], None),
            "txn_country": "US",
            # card-present -> merchant location; card-not-present -> cardholder home (IP proxy)
            "txn_lat": np.where(online, df["lat"], df["merch_lat"]).astype(float),
            "txn_lon": np.where(online, df["long"], df["merch_long"]).astype(float),
            "is_fraud": df["is_fraud"].astype(int),
            "fraud_scenario": None,
            "_file_date": ts.dt.strftime("%Y-%m-%d"),
        }
    )

    accounts = (
        df.assign(account_id=account_ids)
        .groupby("account_id", as_index=False)
        .agg(home_city=("city", "first"), home_lat=("lat", "first"), home_lon=("long", "first"))
    )
    accounts["segment"] = "unknown"
    accounts["home_country"] = "US"
    accounts["credit_limit"] = np.nan
    accounts["open_date"] = None
    accounts = accounts[
        ["account_id", "segment", "home_city", "home_country", "home_lat", "home_lon", "credit_limit", "open_date"]
    ]

    mdf = pd.DataFrame({"code": codes, "category": category, "online": online})
    agg = mdf.groupby("code").agg(category=("category", "first"), online_share=("online", "mean"))
    merchants = pd.DataFrame(
        {
            "merchant_id": merchant_ids[agg.index.to_numpy()],
            "merchant_name": [str(names[i]).removeprefix("fraud_") for i in agg.index],
            "category": agg["category"].to_numpy(),
            "city": None,
            "country": "US",
            "lat": np.nan,
            "lon": np.nan,
            "is_online": agg["online_share"].to_numpy() > 0.5,
        }
    )
    return {"transactions": tx, "accounts": accounts, "merchants": merchants}
