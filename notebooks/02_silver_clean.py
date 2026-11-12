# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Silver: type, validate, deduplicate, quarantine
# MAGIC
# MAGIC | Rule type | Behaviour | Example |
# MAGIC |---|---|---|
# MAGIC | **drop → quarantine** | row excluded from silver, kept in `silver.transactions_quarantine` with the failed rule names | null/negative amount, bad timestamp, unknown merchant |
# MAGIC | **warn** | row kept, violation counted in `dq_run_log` | amount > $50k |
# MAGIC | **duplicate** | first-seen copy kept, replays quarantined | upstream retry wrote the same `transaction_id` twice |
# MAGIC
# MAGIC Silver is written with `MERGE` on `transaction_id`, so re-running never creates duplicates
# MAGIC and late-arriving corrections (newer `_ingest_ts`) overwrite older versions.
# MAGIC Uses `try_cast` / `try_to_timestamp` so ANSI mode never crashes on bad input.

# COMMAND ----------

import datetime as dt
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..")))
from lakehouse.config import Paths
from pyspark.sql import Window
from pyspark.sql import functions as F

dbutils.widgets.text("catalog", "workspace")
P = Paths(dbutils.widgets.get("catalog"))

# COMMAND ----------

# MAGIC %md ## Dimensions

# COMMAND ----------


def latest_per_key(df, key):
    w = Window.partitionBy(key).orderBy(F.col("_ingest_ts").desc())
    return df.where(F.col(key).isNotNull()).withColumn("_rn", F.row_number().over(w)).where("_rn = 1").drop("_rn")


accounts = latest_per_key(spark.table(P.t("bronze", "accounts")), "account_id").select(
    "account_id",
    "segment",
    "home_city",
    "home_country",
    F.expr("try_cast(home_lat AS DOUBLE)").alias("home_lat"),
    F.expr("try_cast(home_lon AS DOUBLE)").alias("home_lon"),
    F.expr("try_cast(credit_limit AS DOUBLE)").alias("credit_limit"),
    F.expr("try_cast(open_date AS DATE)").alias("open_date"),
)
accounts.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(P.t("silver", "accounts"))

merchants = latest_per_key(spark.table(P.t("bronze", "merchants")), "merchant_id").select(
    "merchant_id",
    "merchant_name",
    "category",
    "city",
    "country",
    F.expr("try_cast(lat AS DOUBLE)").alias("lat"),
    F.expr("try_cast(lon AS DOUBLE)").alias("lon"),
    F.expr("try_cast(is_online AS BOOLEAN)").alias("is_online"),
)
merchants.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(P.t("silver", "merchants"))

print("accounts:", spark.table(P.t("silver", "accounts")).count(), "merchants:", spark.table(P.t("silver", "merchants")).count())

# COMMAND ----------

# MAGIC %md ## Transactions

# COMMAND ----------

DROP_RULES = {
    "transaction_id_not_null": "transaction_id IS NOT NULL",
    "valid_timestamp": "txn_ts IS NOT NULL",
    "amount_not_null": "amount IS NOT NULL",
    "amount_positive": "amount > 0",
    "valid_currency": "currency = 'USD'",
    "valid_channel": "channel IN ('pos', 'online')",
    "known_account": "_acct_ok IS NOT NULL",
    "known_merchant": "_merch_ok IS NOT NULL",
}
WARN_RULES = {
    "amount_below_50k": "amount <= 50000",
}

typed = spark.table(P.t("bronze", "transactions")).select(
    "transaction_id",
    "account_id",
    "merchant_id",
    "merchant_category",
    F.expr("try_cast(amount AS DOUBLE)").alias("amount"),
    F.upper(F.trim("currency")).alias("currency"),
    F.expr("try_to_timestamp(txn_ts, 'yyyy-MM-dd HH:mm:ss')").alias("txn_ts"),
    F.lower(F.trim("channel")).alias("channel"),
    "device_id",
    "txn_city",
    "txn_country",
    F.expr("try_cast(txn_lat AS DOUBLE)").alias("txn_lat"),
    F.expr("try_cast(txn_lon AS DOUBLE)").alias("txn_lon"),
    F.expr("try_cast(is_fraud AS INT)").alias("is_fraud"),
    "fraud_scenario",
    "_source_file",
    "_ingest_ts",
)

dup_w = Window.partitionBy("transaction_id").orderBy(F.col("_ingest_ts").asc(), F.col("_source_file").asc())
acct_keys = spark.table(P.t("silver", "accounts")).select("account_id", F.lit(True).alias("_acct_ok"))
merch_keys = spark.table(P.t("silver", "merchants")).select("merchant_id", F.lit(True).alias("_merch_ok"))

checked = (
    typed.withColumn("_dup_rank", F.row_number().over(dup_w))
    .join(acct_keys, "account_id", "left")
    .join(merch_keys, "merchant_id", "left")
)

failed = F.filter(
    F.array(*[F.when(~F.coalesce(F.expr(cond), F.lit(False)), F.lit(name)) for name, cond in DROP_RULES.items()]),
    lambda x: x.isNotNull(),
)
checked = checked.withColumn("failed_rules", failed).withColumn(
    "failed_rules",
    F.when(
        (F.col("_dup_rank") > 1) & F.col("transaction_id").isNotNull(),
        F.array_union("failed_rules", F.array(F.lit("duplicate_transaction_id"))),
    ).otherwise(F.col("failed_rules")),
)

SILVER_COLS = [
    "transaction_id", "account_id", "merchant_id", "merchant_category", "amount", "currency",
    "txn_ts", "txn_date", "channel", "device_id", "txn_city", "txn_country", "txn_lat", "txn_lon",
    "is_fraud", "fraud_scenario", "_source_file", "_ingest_ts",
]
valid = checked.where(F.size("failed_rules") == 0).withColumn("txn_date", F.to_date("txn_ts")).select(*SILVER_COLS)
quarantine = checked.where(F.size("failed_rules") > 0).drop("_acct_ok", "_merch_ok", "_dup_rank").withColumn(
    "_quarantined_at", F.current_timestamp()
)

# COMMAND ----------

target = P.t("silver", "transactions")
if not spark.catalog.tableExists(target):
    valid.limit(0).write.saveAsTable(target)
    spark.sql(f"ALTER TABLE {target} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

valid.createOrReplaceTempView("silver_tx_updates")
spark.sql(
    f"""
    MERGE INTO {target} t
    USING silver_tx_updates s
    ON t.transaction_id = s.transaction_id
    WHEN MATCHED AND s._ingest_ts > t._ingest_ts THEN UPDATE SET *
    WHEN NOT MATCHED THEN INSERT *
    """
)

# Quarantine reflects the current state of rejected bronze rows (full rebuild each run).
quarantine.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(P.t("silver", "transactions_quarantine"))

# COMMAND ----------

# MAGIC %md ## Data-quality run log

# COMMAND ----------

bronze_rows = checked.count()
q_rows = spark.table(P.t("silver", "transactions_quarantine")).count()
rule_counts = {
    r["rule"]: int(r["count"])
    for r in spark.table(P.t("silver", "transactions_quarantine"))
    .select(F.explode("failed_rules").alias("rule"))
    .groupBy("rule")
    .count()
    .collect()
}
warn_counts = {
    name: int(checked.where(~F.coalesce(F.expr(cond), F.lit(True))).count()) for name, cond in WARN_RULES.items()
}

log = spark.createDataFrame(
    [(dt.datetime.utcnow(), bronze_rows, bronze_rows - q_rows, q_rows, rule_counts, warn_counts)],
    "run_ts timestamp, bronze_rows long, valid_rows long, quarantined_rows long, "
    "failed_rule_counts map<string,long>, warning_counts map<string,long>",
)
log.write.mode("append").saveAsTable(P.t("silver", "dq_run_log"))

print(f"bronze rows      : {bronze_rows:,}")
print(f"silver rows      : {spark.table(target).count():,}")
print(f"quarantined rows : {q_rows:,}  ({q_rows / max(bronze_rows, 1):.3%})")
for k, v in sorted(rule_counts.items(), key=lambda kv: -kv[1]):
    print(f"  {k:<28}{v:,}")
print("warnings:", warn_counts)
