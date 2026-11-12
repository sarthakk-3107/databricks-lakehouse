# Databricks notebook source
# MAGIC %md
# MAGIC # Declarative pipeline (Delta Live Tables / Lakeflow Declarative Pipelines)
# MAGIC
# MAGIC Same bronze → silver → gold logic as notebooks 01–03, expressed declaratively.
# MAGIC Databricks manages checkpoints, schema locations, retries, lineage and the data-quality UI.
# MAGIC
# MAGIC **Expectation modes demonstrated**
# MAGIC
# MAGIC | Decorator | Effect | Used for |
# MAGIC |---|---|---|
# MAGIC | `@dlt.expect` | keep row, record violation | `amount_below_50k` (suspicious but legal) |
# MAGIC | `@dlt.expect_all_or_drop` | drop row, record violation | nulls, negative amounts, bad currency, unknown keys |
# MAGIC | `@dlt.expect_or_fail` | fail the update | structural problems (missing lineage column) |
# MAGIC | quarantine table | rows that failed any drop rule, for triage | `quarantine_transactions` |
# MAGIC
# MAGIC Pipeline configuration (see `dlt/pipeline_settings.json`):
# MAGIC `lakehouse.landing = /Volumes/<catalog>/fin_raw/files/landing`
# MAGIC
# MAGIC Tables publish to the pipeline's target schema (default `fin_dlt`). To train on DLT output,
# MAGIC point notebook 05 at `<catalog>.fin_dlt.silver_transactions` / `silver_accounts`.
# MAGIC `import dlt` continues to work after the rename to Lakeflow Declarative Pipelines.

# COMMAND ----------

import dlt
from pyspark.sql import Window
from pyspark.sql import functions as F

LANDING = spark.conf.get("lakehouse.landing")


def autoload(subdir: str, fmt: str, **options):
    reader = (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", fmt)
        .option("cloudFiles.inferColumnTypes", "false")
        .option("cloudFiles.schemaEvolutionMode", "rescue")
    )
    for k, v in options.items():
        reader = reader.option(k, v)
    return reader.load(f"{LANDING}/{subdir}").select(
        "*",
        F.col("_metadata.file_path").alias("_source_file"),
        F.current_timestamp().alias("_ingest_ts"),
    )

# COMMAND ----------

# ----------------------------- BRONZE -----------------------------------


@dlt.table(name="bronze_transactions", comment="Raw transactions via Auto Loader; all strings")
def bronze_transactions():
    return autoload("transactions", "json")


@dlt.table(name="bronze_accounts", comment="Raw account dimension")
def bronze_accounts():
    return autoload("accounts", "csv", header="true")


@dlt.table(name="bronze_merchants", comment="Raw merchant dimension")
def bronze_merchants():
    return autoload("merchants", "csv", header="true")

# COMMAND ----------

# ----------------------------- SILVER -----------------------------------


def _latest(df, key):
    w = Window.partitionBy(key).orderBy(F.col("_ingest_ts").desc())
    return df.where(F.col(key).isNotNull()).withColumn("_rn", F.row_number().over(w)).where("_rn = 1").drop("_rn")


@dlt.table(name="silver_accounts", comment="Latest version of each account")
@dlt.expect_or_drop("account_id_present", "account_id IS NOT NULL")
def silver_accounts():
    return _latest(dlt.read("bronze_accounts"), "account_id").select(
        "account_id", "segment", "home_city", "home_country",
        F.expr("try_cast(home_lat AS DOUBLE)").alias("home_lat"),
        F.expr("try_cast(home_lon AS DOUBLE)").alias("home_lon"),
        F.expr("try_cast(credit_limit AS DOUBLE)").alias("credit_limit"),
        F.expr("try_cast(open_date AS DATE)").alias("open_date"),
    )


@dlt.table(name="silver_merchants", comment="Latest version of each merchant")
def silver_merchants():
    return _latest(dlt.read("bronze_merchants"), "merchant_id").select(
        "merchant_id", "merchant_name", "category", "city", "country",
        F.expr("try_cast(lat AS DOUBLE)").alias("lat"),
        F.expr("try_cast(lon AS DOUBLE)").alias("lon"),
        F.expr("try_cast(is_online AS BOOLEAN)").alias("is_online"),
    )


DROP_RULES = {
    "transaction_id_not_null": "transaction_id IS NOT NULL",
    "valid_timestamp": "txn_ts IS NOT NULL",
    "amount_positive": "amount IS NOT NULL AND amount > 0",
    "valid_currency": "currency IS NOT NULL AND currency = 'USD'",
    "valid_channel": "channel IS NOT NULL AND channel IN ('pos', 'online')",
    "known_account": "_acct_ok IS NOT NULL",
    "known_merchant": "_merch_ok IS NOT NULL",
}
ALL_VALID = " AND ".join(f"coalesce(({c}), false)" for c in DROP_RULES.values())


@dlt.view(name="transactions_typed", comment="Typed bronze transactions with referential flags")
def transactions_typed():
    acct = dlt.read("silver_accounts").select("account_id", F.lit(True).alias("_acct_ok"))
    merch = dlt.read("silver_merchants").select("merchant_id", F.lit(True).alias("_merch_ok"))
    return (
        dlt.read("bronze_transactions")
        .select(
            "transaction_id", "account_id", "merchant_id", "merchant_category",
            F.expr("try_cast(amount AS DOUBLE)").alias("amount"),
            F.upper(F.trim("currency")).alias("currency"),
            F.expr("try_to_timestamp(txn_ts, 'yyyy-MM-dd HH:mm:ss')").alias("txn_ts"),
            F.lower(F.trim("channel")).alias("channel"),
            "device_id", "txn_city", "txn_country",
            F.expr("try_cast(txn_lat AS DOUBLE)").alias("txn_lat"),
            F.expr("try_cast(txn_lon AS DOUBLE)").alias("txn_lon"),
            F.expr("try_cast(is_fraud AS INT)").alias("is_fraud"),
            "fraud_scenario", "_source_file", "_ingest_ts",
        )
        .join(acct, "account_id", "left")
        .join(merch, "merchant_id", "left")
    )


@dlt.table(name="silver_transactions", comment="Validated, deduplicated transactions")
@dlt.expect_all_or_drop(DROP_RULES)
@dlt.expect("amount_below_50k", "amount <= 50000")
@dlt.expect_or_fail("has_source_file", "_source_file IS NOT NULL")
def silver_transactions():
    w = Window.partitionBy("transaction_id").orderBy(F.col("_ingest_ts").asc(), F.col("_source_file").asc())
    return (
        dlt.read("transactions_typed")
        .withColumn("_dup_rank", F.row_number().over(w))
        .where("_dup_rank = 1")
        .drop("_dup_rank")
        .withColumn("txn_date", F.to_date("txn_ts"))
    )


@dlt.table(name="quarantine_transactions", comment="Rows that failed at least one drop rule")
def quarantine_transactions():
    return dlt.read("transactions_typed").where(f"NOT ({ALL_VALID})").withColumn("_quarantined_at", F.current_timestamp())

# COMMAND ----------

# ------------------------------ GOLD ------------------------------------


@dlt.table(name="gold_daily_transaction_summary")
def gold_daily_transaction_summary():
    return (
        dlt.read("silver_transactions")
        .groupBy("txn_date")
        .agg(
            F.count("*").alias("n_txn"),
            F.round(F.sum("amount"), 2).alias("total_amount"),
            F.sum("is_fraud").alias("n_fraud"),
            F.round(F.avg("is_fraud") * 100, 4).alias("fraud_rate_pct"),
        )
    )


@dlt.table(name="gold_fraud_by_category_channel")
def gold_fraud_by_category_channel():
    return (
        dlt.read("silver_transactions")
        .groupBy("merchant_category", "channel")
        .agg(
            F.count("*").alias("n_txn"),
            F.sum("is_fraud").alias("n_fraud"),
            F.round(F.avg("is_fraud") * 100, 4).alias("fraud_rate_pct"),
        )
    )


@dlt.table(name="gold_dq_summary", comment="Quarantined rows per failed rule")
def gold_dq_summary():
    checks = [F.when(~F.coalesce(F.expr(c), F.lit(False)), F.lit(n)) for n, c in DROP_RULES.items()]
    return (
        dlt.read("quarantine_transactions")
        .select(F.explode(F.filter(F.array(*checks), lambda x: x.isNotNull())).alias("rule"))
        .groupBy("rule")
        .count()
    )
