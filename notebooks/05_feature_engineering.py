# Databricks notebook source
# MAGIC %md
# MAGIC # 05 · Fraud feature engineering (PySpark)
# MAGIC
# MAGIC Builds `fin_ml.fraud_features`, one row per silver transaction, with **point-in-time correct**
# MAGIC features (each one uses only data available *before* the transaction):
# MAGIC
# MAGIC | Family | Features | Targets |
# MAGIC |---|---|---|
# MAGIC | Velocity | `txn_cnt_1h/24h/7d`, `amt_sum_24h`, `distinct_merchants_24h`, `small_txn_cnt_1h` | card testing, takeover bursts |
# MAGIC | Spend deviation | `amt_mean_30d`, `amt_z_30d`, `amt_ratio_30d`, `amt_to_limit` | unusually large purchases |
# MAGIC | Geo | `dist_home_km`, `dist_prev_km`, `speed_kmh`, `is_cross_border` | cloned cards / impossible travel |
# MAGIC | Novelty | `is_new_merchant`, `is_new_category`, `is_new_device` | account takeover |
# MAGIC | Time | `hour`, `day_of_week`, `is_night` | off-hours activity |
# MAGIC | Merchant prior | `merchant_prior_fraud_rate` (smoothed, **30-day label delay**) | compromised / high-risk merchants |
# MAGIC
# MAGIC The first 30 days are flagged `is_warmup` (behaviour windows are immature) and excluded from training.
# MAGIC Logic lives in `lakehouse/spark_features.py` so batch scoring can reuse it.

# COMMAND ----------

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..")))
from lakehouse.config import Paths
from lakehouse.spark_features import build_fraud_features
from pyspark.sql import functions as F

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("label_delay_days", "30")
dbutils.widgets.text("warmup_days", "30")
P = Paths(dbutils.widgets.get("catalog"))

# COMMAND ----------

tx = spark.table(P.t("silver", "transactions"))
accounts = spark.table(P.t("silver", "accounts"))

features = build_fraud_features(
    tx,
    accounts,
    label_delay_days=int(dbutils.widgets.get("label_delay_days")),
    warmup_days=int(dbutils.widgets.get("warmup_days")),
)

target = P.t("ml", "fraud_features")
features.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(target)
spark.sql(f"COMMENT ON TABLE {target} IS 'Point-in-time fraud features, one row per silver transaction'")

# COMMAND ----------

ft = spark.table(target)
display(
    ft.groupBy("is_warmup")
    .agg(F.count("*").alias("rows"), F.sum("is_fraud").alias("fraud"), F.round(F.avg("is_fraud") * 100, 4).alias("fraud_rate_pct"))
)

# COMMAND ----------

# MAGIC %md ### Sanity check: do features separate fraud from legit?

# COMMAND ----------

check_cols = [
    "amount", "txn_cnt_1h", "small_txn_cnt_1h", "amt_z_30d", "speed_kmh",
    "dist_home_km", "is_new_device", "is_new_merchant", "merchant_prior_fraud_rate",
]
display(
    ft.where("NOT is_warmup")
    .groupBy("is_fraud")
    .agg(*[F.round(F.avg(c), 3).alias(f"avg_{c}") for c in check_cols])
    .orderBy("is_fraud")
)
