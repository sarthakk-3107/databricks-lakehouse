"""Synthetic card-transaction generator with behaviour-driven fraud.

Design goals (so the ML half is not a toy):
  * Legit customers have *habits*: home city, favourite merchants, category
    mix, time-of-day pattern, spend level by segment, 1-2 devices.
  * Legit behaviour also produces *false-positive bait*: travel trips, new
    phones, occasional large purchases.
  * Fraud comes from four scenarios with different signatures:
      - card_testing      : bursts of tiny online charges, often followed by a big one
      - account_takeover  : new device, high-value online purchases (electronics/travel/gift cards)
      - cloned_card       : card-present spend in a distant city -> impossible travel
      - mimic             : fraud that looks like the customer's normal behaviour (hard / ~undetectable)
  * A minority of high-risk online merchants attract more fraud, so a
    merchant-level prior is useful *if computed without leakage*.
  * Label noise: some fraud is never reported; a few legit txns are disputed
    ("friendly fraud").
  * Dirty records (nulls, negatives, unknown merchants, bad timestamps, bad
    currency, replayed duplicates) exercise the silver data-quality gates.

The label `is_fraud` is learnable but not trivially so. `fraud_scenario` is
ground truth for *analysis only* and must never be used as a feature.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

# (name, country, lat, lon)
US_CITIES = [
    ("New York", "US", 40.7128, -74.0060),
    ("Los Angeles", "US", 34.0522, -118.2437),
    ("Chicago", "US", 41.8781, -87.6298),
    ("Houston", "US", 29.7604, -95.3698),
    ("Phoenix", "US", 33.4484, -112.0740),
    ("Philadelphia", "US", 39.9526, -75.1652),
    ("San Antonio", "US", 29.4241, -98.4936),
    ("San Diego", "US", 32.7157, -117.1611),
    ("Dallas", "US", 32.7767, -96.7970),
    ("Austin", "US", 30.2672, -97.7431),
    ("Seattle", "US", 47.6062, -122.3321),
    ("Denver", "US", 39.7392, -104.9903),
    ("Boston", "US", 42.3601, -71.0589),
    ("Atlanta", "US", 33.7490, -84.3880),
    ("Miami", "US", 25.7617, -80.1918),
    ("Richmond", "US", 37.5407, -77.4360),
    ("Minneapolis", "US", 44.9778, -93.2650),
    ("Portland", "US", 45.5152, -122.6784),
]
FOREIGN_CITIES = [
    ("London", "GB", 51.5074, -0.1278),
    ("Toronto", "CA", 43.6532, -79.3832),
    ("Mexico City", "MX", 19.4326, -99.1332),
    ("Lagos", "NG", 6.5244, 3.3792),
    ("Bucharest", "RO", 44.4268, 26.1025),
    ("Manila", "PH", 14.5995, 120.9842),
]
CITY_INFO = {c[0]: c for c in US_CITIES + FOREIGN_CITIES}
US_CITY_NAMES = [c[0] for c in US_CITIES]
FOREIGN_CITY_NAMES = [c[0] for c in FOREIGN_CITIES]

# category -> (log-mean amount, log-sd, share of merchants that are online)
CATEGORIES = {
    "grocery": (3.6, 0.6, 0.10),
    "restaurants": (3.2, 0.6, 0.05),
    "gas": (3.7, 0.3, 0.0),
    "retail": (4.0, 0.8, 0.40),
    "electronics": (5.0, 0.9, 0.60),
    "travel": (5.8, 0.8, 0.80),
    "entertainment": (3.4, 0.7, 0.50),
    "health": (3.8, 0.7, 0.20),
    "utilities": (4.6, 0.4, 0.90),
    "gift_cards": (4.2, 0.6, 0.70),
    "digital_goods": (2.5, 0.8, 1.00),
}
CATEGORY_NAMES = list(CATEGORIES)
CAT_MU = np.array([CATEGORIES[c][0] for c in CATEGORY_NAMES])
CAT_SD = np.array([CATEGORIES[c][1] for c in CATEGORY_NAMES])
CATEGORY_BASE_PREF = np.array([3.0, 3.0, 2.0, 2.0, 0.5, 0.3, 1.0, 0.7, 1.0, 0.2, 1.0])

SEGMENTS = {
    "student": dict(rate=0.8, amt_mult=0.6, limit=(500, 3000), share=0.20),
    "mass": dict(rate=1.0, amt_mult=1.0, limit=(2000, 10000), share=0.50),
    "affluent": dict(rate=1.4, amt_mult=1.8, limit=(10000, 40000), share=0.20),
    "small_business": dict(rate=2.0, amt_mult=1.5, limit=(5000, 25000), share=0.10),
}

HOUR_WEIGHTS = np.array(
    [0.3, 0.2, 0.15, 0.1, 0.1, 0.2, 0.5, 1.0, 1.5, 1.8, 2.0, 2.2,
     2.6, 2.4, 2.2, 2.2, 2.4, 2.8, 3.0, 2.8, 2.4, 1.8, 1.2, 0.6]
)
HOUR_P = HOUR_WEIGHTS / HOUR_WEIGHTS.sum()

FRAUD_SCENARIOS = {
    "card_testing": 0.30,
    "account_takeover": 0.30,
    "cloned_card": 0.25,
    "mimic": 0.15,
}

TX_COLUMNS = [
    "transaction_id", "account_id", "merchant_id", "merchant_category", "amount",
    "currency", "txn_ts", "channel", "device_id", "txn_city", "txn_country",
    "txn_lat", "txn_lon", "is_fraud", "fraud_scenario",
]


@dataclass
class GeneratorConfig:
    n_accounts: int = 5000
    days: int = 180
    start_date: str = "2026-01-01"
    seed: int = 42
    base_daily_rate: float = 0.5        # legit txns/day for a 'mass' customer
    compromise_rate: float = 0.08       # share of accounts with >= 1 fraud episode
    second_episode_rate: float = 0.10   # of compromised accounts, share hit twice
    first_fraud_day: int = 20           # no fraud before this day (history warm-up)
    travel_rate: float = 0.10           # share of accounts that take one trip
    new_device_rate: float = 0.03       # legit online txns from a new device
    label_miss_rate: float = 0.05       # fraud never reported -> labelled 0
    friendly_fraud_rate: float = 0.0003 # legit txns disputed -> labelled 1
    dirty_rate: float = 0.004           # share of rows corrupted for DQ demos


def _make_merchants(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    i = 0
    for city, country, lat, lon in US_CITIES + FOREIGN_CITIES:
        foreign = country != "US"
        for cat, (_, _, p_online) in CATEGORIES.items():
            if p_online >= 1.0:
                continue
            k = 1 if foreign else int(rng.integers(2, 5))
            for _ in range(k):
                i += 1
                rows.append(
                    dict(
                        merchant_id=f"M{i:05d}",
                        merchant_name=f"{cat.replace('_', ' ').title()} {city} #{i}",
                        category=cat,
                        city=city,
                        country=country,
                        lat=lat + rng.normal(0, 0.05),
                        lon=lon + rng.normal(0, 0.05),
                        is_online=False,
                    )
                )
    for cat, (_, _, p_online) in CATEGORIES.items():
        if p_online <= 0:
            continue
        for _ in range(int(rng.integers(5, 15))):
            i += 1
            rows.append(
                dict(
                    merchant_id=f"M{i:05d}",
                    merchant_name=f"{cat.replace('_', ' ').title()} Online #{i}",
                    category=cat,
                    city=None,
                    country="US",
                    lat=np.nan,
                    lon=np.nan,
                    is_online=True,
                )
            )
    m = pd.DataFrame(rows)
    risky = m["is_online"] & m["category"].isin(["gift_cards", "digital_goods", "electronics"])
    risky_idx = m.index[risky].to_numpy()
    n_high = max(1, int(0.25 * len(risky_idx)))
    m["_high_risk"] = False
    m.loc[rng.choice(risky_idx, size=n_high, replace=False), "_high_risk"] = True
    return m


def _pick_other_city(rng: np.random.Generator, home: str, foreign_p: float) -> str:
    if rng.random() < foreign_p:
        return str(rng.choice(FOREIGN_CITY_NAMES))
    choices = [c for c in US_CITY_NAMES if c != home]
    return str(rng.choice(choices))


def generate(cfg: GeneratorConfig | None = None) -> dict[str, pd.DataFrame]:
    """Return {'transactions', 'accounts', 'merchants'} DataFrames."""
    cfg = cfg or GeneratorConfig()
    rng = np.random.default_rng(cfg.seed)
    start = pd.Timestamp(cfg.start_date)
    horizon = cfg.days * 86400

    # ---------------- merchants ----------------
    m = _make_merchants(rng)
    m_online = m["is_online"].to_numpy()
    m_lat = m["lat"].to_numpy(dtype=float)
    m_lon = m["lon"].to_numpy(dtype=float)
    m_city = m["city"].to_numpy(dtype=object)
    m_country = m["country"].to_numpy(dtype=object)

    physical: dict[tuple[str, str], np.ndarray] = {}
    for (city, cat), g in m[~m["is_online"]].groupby(["city", "category"]):
        physical[(city, cat)] = g.index.to_numpy()
    online_by_cat = {cat: g.index.to_numpy() for cat, g in m[m["is_online"]].groupby("category")}
    high_risk_by_cat = {
        cat: g.index.to_numpy() for cat, g in m[m["is_online"] & m["_high_risk"]].groupby("category")
    }

    pool_cache: dict[tuple[str, str], np.ndarray] = {}

    def pool_for(city: str, cat: str) -> np.ndarray:
        key = (city, cat)
        if key not in pool_cache:
            parts = [physical.get(key, np.array([], dtype=int)), online_by_cat.get(cat, np.array([], dtype=int))]
            pool_cache[key] = np.concatenate(parts).astype(int)
        return pool_cache[key]

    def pick_online(cat: str, high_risk_p: float) -> int:
        hr = high_risk_by_cat.get(cat)
        if hr is not None and len(hr) and rng.random() < high_risk_p:
            return int(rng.choice(hr))
        return int(rng.choice(online_by_cat[cat]))

    # ---------------- accounts ----------------
    n = cfg.n_accounts
    seg_names = list(SEGMENTS)
    seg_p = np.array([SEGMENTS[s]["share"] for s in seg_names])
    acc_seg = rng.choice(seg_names, size=n, p=seg_p / seg_p.sum())
    acc_city = rng.choice(US_CITY_NAMES, size=n)
    acc_lat = np.array([CITY_INFO[c][2] for c in acc_city]) + rng.normal(0, 0.05, n)
    acc_lon = np.array([CITY_INFO[c][3] for c in acc_city]) + rng.normal(0, 0.05, n)
    acc_limit = np.array(
        [round(rng.uniform(*SEGMENTS[s]["limit"]) / 100) * 100 for s in acc_seg], dtype=float
    )
    acc_open = [start - pd.Timedelta(days=int(d)) for d in rng.integers(30, 3000, size=n)]
    acc_ids = np.array([f"A{a:06d}" for a in range(n)])
    acc_devices = [[f"D{a:06d}{j}" for j in range(int(rng.integers(1, 3)))] for a in range(n)]
    prefs = rng.dirichlet(CATEGORY_BASE_PREF * 5, size=n)

    favorites: list[dict[str, np.ndarray]] = []
    for a in range(n):
        fav = {}
        for cat in CATEGORY_NAMES:
            pool = pool_for(acc_city[a], cat)
            size = min(len(pool), int(rng.integers(1, 4)))
            fav[cat] = rng.choice(pool, size=size, replace=False)
        favorites.append(fav)

    def new_device() -> str:
        return f"D{int(rng.integers(0, 10**10)):010d}"

    # ---------------- legit transactions ----------------
    parts = []
    for a in range(n):
        seg = SEGMENTS[acc_seg[a]]
        counts = rng.poisson(cfg.base_daily_rate * seg["rate"], size=cfg.days)
        N = int(counts.sum())
        if N == 0:
            continue
        day = np.repeat(np.arange(cfg.days), counts)
        secs = day * 86400 + rng.choice(24, size=N, p=HOUR_P) * 3600 + rng.integers(0, 3600, size=N)
        cat_idx = rng.choice(len(CATEGORY_NAMES), size=N, p=prefs[a])
        home = acc_city[a]

        trip_city = None
        in_trip = np.zeros(N, dtype=bool)
        if rng.random() < cfg.travel_rate and cfg.days > 25:
            t0 = int(rng.integers(10, cfg.days - 10))
            in_trip = (day >= t0) & (day < t0 + int(rng.integers(3, 10)))
            trip_city = _pick_other_city(rng, home, foreign_p=0.25)

        merch = np.empty(N, dtype=int)
        for ci in np.unique(cat_idx):
            mask = cat_idx == ci
            cat = CATEGORY_NAMES[ci]
            n_c = int(mask.sum())
            fav = favorites[a][cat]
            pool = pool_for(home, cat)
            merch[mask] = np.where(rng.random(n_c) < 0.8, rng.choice(fav, n_c), rng.choice(pool, n_c))
            if trip_city is not None:
                sub = np.where(mask & in_trip)[0]
                tpool = physical.get((trip_city, cat))
                if len(sub) and tpool is not None and len(tpool):
                    phys = ~m_online[merch[sub]]
                    if phys.any():
                        merch[sub[phys]] = rng.choice(tpool, int(phys.sum()))

        online = m_online[merch]
        lat = np.where(online, acc_lat[a], m_lat[merch])
        lon = np.where(online, acc_lon[a], m_lon[merch])
        city = np.where(online, home, m_city[merch]).astype(object)
        country = np.where(online, "US", m_country[merch]).astype(object)
        if trip_city is not None:
            tm = online & in_trip
            _, tc, tlat, tlon = CITY_INFO[trip_city]
            lat[tm], lon[tm], city[tm], country[tm] = tlat, tlon, trip_city, tc

        dev = np.where(online, rng.choice(acc_devices[a], N), None).astype(object)
        nd = np.where(online & (rng.random(N) < cfg.new_device_rate))[0]
        for j in nd:
            dev[j] = new_device()

        mu = CAT_MU[cat_idx] + np.log(seg["amt_mult"])
        amt = np.round(np.clip(np.exp(rng.normal(mu, CAT_SD[cat_idx])), 0.5, acc_limit[a]), 2)

        parts.append(
            pd.DataFrame(
                dict(
                    account_idx=a, merchant_idx=merch, ts_sec=secs, amount=amt,
                    channel=np.where(online, "online", "pos"), device_id=dev,
                    txn_city=city, txn_country=country, txn_lat=lat, txn_lon=lon,
                    is_fraud=0, fraud_scenario=None,
                )
            )
        )

    # ---------------- fraud episodes ----------------
    fraud_rows: list[dict] = []

    def add(a, mi, t, amt, channel, device, city, country, lat, lon, scenario):
        if t >= horizon:
            return
        fraud_rows.append(
            dict(
                account_idx=a, merchant_idx=int(mi), ts_sec=int(t),
                amount=round(float(min(max(amt, 0.5), acc_limit[a])), 2),
                channel=channel, device_id=device, txn_city=city, txn_country=country,
                txn_lat=float(lat), txn_lon=float(lon), is_fraud=1, fraud_scenario=scenario,
            )
        )

    def ip_location(home: str, foreign_p: float):
        c = _pick_other_city(rng, home, foreign_p)
        name, country, lat, lon = CITY_INFO[c]
        return name, country, lat + rng.normal(0, 0.05), lon + rng.normal(0, 0.05)

    scen_names = list(FRAUD_SCENARIOS)
    scen_p = np.array(list(FRAUD_SCENARIOS.values()))
    compromised = np.where(rng.random(n) < cfg.compromise_rate)[0]
    for a in compromised:
        episodes = 2 if rng.random() < cfg.second_episode_rate else 1
        for _ in range(episodes):
            t = int(rng.integers(cfg.first_fraud_day, cfg.days)) * 86400 + int(rng.integers(0, 86400))
            scenario = str(rng.choice(scen_names, p=scen_p))
            home = acc_city[a]

            if scenario == "card_testing":
                dev = new_device()
                loc = ip_location(home, foreign_p=0.5)
                for _ in range(int(rng.integers(3, 9))):
                    t += int(rng.integers(30, 300))
                    cat = str(rng.choice(["digital_goods", "gift_cards"]))
                    add(a, pick_online(cat, 0.7), t, rng.uniform(0.5, 4.99), "online", dev, *loc, scenario)
                if rng.random() < 0.6:
                    for _ in range(int(rng.integers(1, 4))):
                        t += int(rng.integers(300, 7200))
                        cat = str(rng.choice(["electronics", "gift_cards"]))
                        add(a, pick_online(cat, 0.6), t, rng.uniform(300, 2000), "online", dev, *loc, scenario)

            elif scenario == "account_takeover":
                dev = new_device()
                loc = ip_location(home, foreign_p=0.4)
                for _ in range(int(rng.integers(2, 7))):
                    t += int(rng.integers(600, 5400))
                    cat = str(rng.choice(["electronics", "travel", "gift_cards"]))
                    amt = min(np.exp(rng.normal(6.0, 0.6)), 0.9 * acc_limit[a])
                    add(a, pick_online(cat, 0.5), t, amt, "online", dev, *loc, scenario)

            elif scenario == "cloned_card":
                far = _pick_other_city(rng, home, foreign_p=0.5)
                for _ in range(int(rng.integers(2, 6))):
                    t += int(rng.integers(1200, 10800))
                    cat = str(rng.choice(["retail", "electronics", "gas", "grocery"]))
                    pool = physical.get((far, cat))
                    if pool is None or not len(pool):
                        continue
                    mi = int(rng.choice(pool))
                    add(a, mi, t, np.exp(rng.normal(4.8, 0.8)), "pos", None,
                        m_city[mi], m_country[mi], m_lat[mi], m_lon[mi], scenario)

            else:  # mimic: looks like the customer's normal behaviour
                seg = SEGMENTS[acc_seg[a]]
                for _ in range(int(rng.integers(2, 5))):
                    t += int(rng.integers(3600, 86400))
                    ci = int(rng.choice(len(CATEGORY_NAMES), p=prefs[a]))
                    cat = CATEGORY_NAMES[ci]
                    mi = int(rng.choice(favorites[a][cat]))
                    amt = np.exp(rng.normal(CAT_MU[ci] + np.log(seg["amt_mult"]), CAT_SD[ci]))
                    if m_online[mi]:
                        add(a, mi, t, amt, "online", str(rng.choice(acc_devices[a])),
                            home, "US", acc_lat[a], acc_lon[a], scenario)
                    else:
                        add(a, mi, t, amt, "pos", None, m_city[mi], m_country[mi], m_lat[mi], m_lon[mi], scenario)

    if fraud_rows:
        parts.append(pd.DataFrame(fraud_rows))
    df = pd.concat(parts, ignore_index=True)
    df = df[df["ts_sec"] < horizon].sort_values("ts_sec", kind="stable").reset_index(drop=True)

    # ---------------- label noise ----------------
    fraud_idx = np.where(df["is_fraud"].to_numpy() == 1)[0]
    missed = fraud_idx[rng.random(len(fraud_idx)) < cfg.label_miss_rate]
    df.loc[missed, "is_fraud"] = 0
    legit_idx = np.where(df["fraud_scenario"].isna().to_numpy())[0]
    friendly = legit_idx[rng.random(len(legit_idx)) < cfg.friendly_fraud_rate]
    df.loc[friendly, "is_fraud"] = 1
    df.loc[friendly, "fraud_scenario"] = "friendly_fraud"

    # ---------------- final columns ----------------
    ts = start + pd.to_timedelta(df["ts_sec"], unit="s")
    out = pd.DataFrame(
        {
            "transaction_id": [f"T{x:012x}" for x in rng.permutation(len(df)) + 0x10000000],
            "account_id": acc_ids[df["account_idx"].to_numpy()],
            "merchant_id": m["merchant_id"].to_numpy()[df["merchant_idx"].to_numpy()],
            "merchant_category": m["category"].to_numpy()[df["merchant_idx"].to_numpy()],
            "amount": df["amount"].astype(float),
            "currency": "USD",
            "txn_ts": ts.dt.strftime("%Y-%m-%d %H:%M:%S"),
            "channel": df["channel"],
            "device_id": df["device_id"],
            "txn_city": df["txn_city"],
            "txn_country": df["txn_country"],
            "txn_lat": df["txn_lat"].astype(float).round(5),
            "txn_lon": df["txn_lon"].astype(float).round(5),
            "is_fraud": df["is_fraud"].astype(int),
            "fraud_scenario": df["fraud_scenario"],
        }
    )
    out["_file_date"] = ts.dt.strftime("%Y-%m-%d")
    out = _inject_dirty_records(out, rng, cfg.dirty_rate)

    accounts = pd.DataFrame(
        {
            "account_id": acc_ids,
            "segment": acc_seg,
            "home_city": acc_city,
            "home_country": "US",
            "home_lat": acc_lat.round(5),
            "home_lon": acc_lon.round(5),
            "credit_limit": acc_limit,
            "open_date": [d.strftime("%Y-%m-%d") for d in acc_open],
        }
    )
    merchants = m.drop(columns=["_high_risk"]).copy()
    merchants["lat"] = merchants["lat"].round(5)
    merchants["lon"] = merchants["lon"].round(5)
    return {"transactions": out, "accounts": accounts, "merchants": merchants}


def _inject_dirty_records(df: pd.DataFrame, rng: np.random.Generator, rate: float) -> pd.DataFrame:
    """Corrupt a small share of rows so the silver quality gates have work to do."""
    if rate <= 0 or len(df) < 100:
        return df
    df = df.copy()
    df["currency"] = df["currency"].astype(object)
    df["txn_ts"] = df["txn_ts"].astype(object)
    df["merchant_id"] = df["merchant_id"].astype(object)
    k = max(1, int(len(df) * rate / 6))
    idx = rng.choice(len(df), size=6 * k, replace=False)
    null_amt, neg_amt, bad_merch, bad_ts, bad_ccy, dup = np.split(idx, 6)

    df.loc[null_amt, "amount"] = np.nan
    df.loc[neg_amt, "amount"] = -df.loc[neg_amt, "amount"].abs()
    df.loc[bad_merch, "merchant_id"] = [f"M_UNKNOWN_{i}" for i in range(len(bad_merch))]
    df.loc[bad_ts, "txn_ts"] = rng.choice(["not-a-timestamp", "2026-13-45 99:00:00", ""], size=len(bad_ts))
    df.loc[bad_ccy, "currency"] = rng.choice(["EUR", "XXX", None], size=len(bad_ccy))
    replay = df.loc[dup].copy()  # exact duplicates, like an upstream retry
    return pd.concat([df, replay], ignore_index=True)


def write_landing(
    data: dict[str, pd.DataFrame],
    landing_dir: str,
    hold_back_days: int = 0,
    pending_dir: str | None = None,
) -> dict[str, int]:
    """Write JSON-lines transactions (one file per day) and CSV dimensions.

    With hold_back_days > 0 the last N daily files go to `pending_dir` instead,
    so you can later move them into landing and watch Auto Loader pick up only
    the new files.
    """
    tx = data["transactions"]
    tx_dir = os.path.join(landing_dir, "transactions")
    os.makedirs(tx_dir, exist_ok=True)
    os.makedirs(os.path.join(landing_dir, "accounts"), exist_ok=True)
    os.makedirs(os.path.join(landing_dir, "merchants"), exist_ok=True)

    dates = sorted(tx["_file_date"].unique())
    held = set(dates[-hold_back_days:]) if hold_back_days > 0 else set()
    if held:
        if not pending_dir:
            raise ValueError("pending_dir is required when hold_back_days > 0")
        os.makedirs(os.path.join(pending_dir, "transactions"), exist_ok=True)

    files = 0
    for d, g in tx.groupby("_file_date"):
        base = os.path.join(pending_dir, "transactions") if d in held else tx_dir
        g.drop(columns=["_file_date"]).to_json(
            os.path.join(base, f"transactions_{d}.json"), orient="records", lines=True
        )
        files += 1

    data["accounts"].to_csv(os.path.join(landing_dir, "accounts", "accounts.csv"), index=False)
    data["merchants"].to_csv(os.path.join(landing_dir, "merchants", "merchants.csv"), index=False)
    return {
        "transactions": int(len(tx)),
        "fraud_labelled": int(tx["is_fraud"].sum()),
        "accounts": int(len(data["accounts"])),
        "merchants": int(len(data["merchants"])),
        "daily_files": files,
        "held_back_files": len(held),
    }
