"""PySpark feature engineering for transaction-level fraud scoring.

Every feature is *point-in-time correct*: it only uses information available
strictly before the transaction being scored.

  * Account behaviour windows use rangeBetween(-N, -1) on epoch seconds, so the
    current row (and anything later) is excluded.
  * The merchant fraud-rate prior only counts labels older than
    `label_delay_days`, because chargebacks/fraud reports arrive weeks after the
    transaction. Using same-day labels would leak the answer.
"""
from __future__ import annotations

from pyspark.sql import Column, DataFrame, Window
from pyspark.sql import functions as F

DAY = 86_400
EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1: Column, lon1: Column, lat2: Column, lon2: Column) -> Column:
    dlat = F.radians(lat2 - lat1)
    dlon = F.radians(lon2 - lon1)
    a = F.pow(F.sin(dlat / 2), 2) + F.cos(F.radians(lat1)) * F.cos(F.radians(lat2)) * F.pow(F.sin(dlon / 2), 2)
    return F.lit(2 * EARTH_RADIUS_KM) * F.asin(F.sqrt(F.least(F.lit(1.0), a)))


def build_fraud_features(
    tx: DataFrame,
    accounts: DataFrame,
    label_delay_days: int = 30,
    warmup_days: int = 30,
    merchant_prior: float = 0.003,
    merchant_alpha: float = 50.0,
) -> DataFrame:
    acc = accounts.select("account_id", "home_lat", "home_lon", "home_country", "credit_limit", "open_date")
    df = tx.join(acc, "account_id", "left").withColumn("ts_sec", F.col("txn_ts").cast("long"))

    # ---- account behaviour (strictly prior rows) -------------------------
    w_acct = Window.partitionBy("account_id").orderBy("ts_sec")

    def prior(seconds: int):
        return w_acct.rangeBetween(-seconds, -1)

    w_seq = Window.partitionBy("account_id").orderBy("ts_sec", "transaction_id")

    df = df.select(
        "*",
        F.count(F.lit(1)).over(prior(3600)).alias("txn_cnt_1h"),
        F.count(F.lit(1)).over(prior(DAY)).alias("txn_cnt_24h"),
        F.count(F.lit(1)).over(prior(7 * DAY)).alias("txn_cnt_7d"),
        F.coalesce(F.sum("amount").over(prior(DAY)), F.lit(0.0)).alias("amt_sum_24h"),
        F.coalesce(
            F.sum(F.when(F.col("amount") < 5, 1).otherwise(0)).over(prior(3600)), F.lit(0)
        ).alias("small_txn_cnt_1h"),
        F.size(F.collect_set("merchant_id").over(prior(DAY))).alias("distinct_merchants_24h"),
        F.avg("amount").over(prior(30 * DAY)).alias("amt_mean_30d"),
        F.stddev("amount").over(prior(30 * DAY)).alias("_amt_std_30d"),
        F.lag("ts_sec").over(w_seq).alias("_prev_ts"),
        F.lag("txn_lat").over(w_seq).alias("_prev_lat"),
        F.lag("txn_lon").over(w_seq).alias("_prev_lon"),
        (F.row_number().over(Window.partitionBy("account_id", "merchant_id").orderBy("ts_sec", "transaction_id")) == 1)
        .cast("int").alias("is_new_merchant"),
        (F.row_number().over(Window.partitionBy("account_id", "merchant_category").orderBy("ts_sec", "transaction_id")) == 1)
        .cast("int").alias("is_new_category"),
        F.row_number().over(Window.partitionBy("account_id", "device_id").orderBy("ts_sec", "transaction_id")).alias("_dev_rank"),
    )

    # ---- merchant prior with label delay ---------------------------------
    w_m = Window.partitionBy("merchant_id").orderBy("ts_sec").rangeBetween(
        Window.unboundedPreceding, -label_delay_days * DAY
    )
    df = df.select(
        "*",
        F.coalesce(F.sum(F.coalesce(F.col("is_fraud"), F.lit(0))).over(w_m), F.lit(0)).alias("_m_fraud"),
        F.count(F.lit(1)).over(w_m).alias("merchant_prior_txn_cnt"),
    )

    dist_prev = haversine_km(F.col("_prev_lat"), F.col("_prev_lon"), F.col("txn_lat"), F.col("txn_lon"))
    secs_since_prev = (F.col("ts_sec") - F.col("_prev_ts")).cast("double")
    hour = F.hour("txn_ts")

    df = (
        df.withColumn("secs_since_prev", secs_since_prev)
        .withColumn("dist_prev_km", dist_prev)
        .withColumn("speed_kmh", F.col("dist_prev_km") / (F.greatest(F.col("secs_since_prev"), F.lit(60.0)) / 3600.0))
        .withColumn("dist_home_km", haversine_km(F.col("home_lat"), F.col("home_lon"), F.col("txn_lat"), F.col("txn_lon")))
        .withColumn("amt_z_30d", (F.col("amount") - F.col("amt_mean_30d")) / (F.coalesce(F.col("_amt_std_30d"), F.lit(0.0)) + 1.0))
        .withColumn("amt_ratio_30d", F.col("amount") / (F.col("amt_mean_30d") + 1.0))
        .withColumn("is_new_device", F.when(F.col("device_id").isNull(), 0).otherwise((F.col("_dev_rank") == 1).cast("int")))
        .withColumn("hour", hour)
        .withColumn("day_of_week", F.dayofweek("txn_ts"))
        .withColumn("is_night", (hour < 6).cast("int"))
        .withColumn("is_cross_border", F.coalesce((F.col("txn_country") != F.col("home_country")).cast("int"), F.lit(0)))
        .withColumn("account_age_days", F.datediff(F.to_date("txn_ts"), F.col("open_date")).cast("double"))
        .withColumn("amt_to_limit", F.when(F.col("credit_limit") > 0, F.col("amount") / F.col("credit_limit")))
        .withColumn(
            "merchant_prior_fraud_rate",
            (F.col("_m_fraud") + F.lit(merchant_alpha * merchant_prior)) / (F.col("merchant_prior_txn_cnt") + F.lit(merchant_alpha)),
        )
    )

    min_ts = df.agg(F.min("ts_sec")).first()[0]
    df = df.withColumn("is_warmup", F.col("ts_sec") < F.lit(min_ts + warmup_days * DAY))

    return df.select(
        "transaction_id", "account_id", "merchant_id", "txn_ts", "txn_date",
        "is_fraud", "fraud_scenario", "is_warmup",
        "amount", "merchant_category", "channel",
        "txn_cnt_1h", "txn_cnt_24h", "txn_cnt_7d", "amt_sum_24h", "small_txn_cnt_1h",
        "distinct_merchants_24h", "amt_mean_30d", "amt_z_30d", "amt_ratio_30d",
        "secs_since_prev", "dist_prev_km", "speed_kmh", "dist_home_km",
        "is_new_merchant", "is_new_category", "is_new_device",
        "hour", "day_of_week", "is_night", "is_cross_border",
        "account_age_days", "amt_to_limit",
        "merchant_prior_fraud_rate", "merchant_prior_txn_cnt",
    )
