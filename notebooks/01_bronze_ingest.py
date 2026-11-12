# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Bronze ingest with Auto Loader
# MAGIC
# MAGIC * `cloudFiles` incrementally discovers **only new files** in the landing zone; the
# MAGIC   checkpoint remembers what has already been loaded, so re-running is safe.
# MAGIC * `inferColumnTypes=false` keeps every column a string in bronze. Typing happens in
# MAGIC   silver, so a malformed value (e.g. `"not-a-timestamp"`) is preserved for debugging
# MAGIC   instead of failing the load.
# MAGIC * `schemaEvolutionMode=rescue` captures unexpected fields in `_rescued_data`.
# MAGIC * `trigger(availableNow=True)` processes the backlog and stops (serverless-friendly batch-style streaming).
# MAGIC
# MAGIC **Incremental demo:** generate with `hold_back_days=7`, run this notebook, then move the
# MAGIC files from `pending/transactions/` into `landing/transactions/` and run it again;
# MAGIC only the 7 new files are processed.

# COMMAND ----------

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..")))
from lakehouse.config import Paths
from pyspark.sql import functions as F

dbutils.widgets.text("catalog", "workspace")
P = Paths(dbutils.widgets.get("catalog"))

# COMMAND ----------


def autoload(source: str, fmt: str, table: str, extra_options: dict | None = None):
    target = P.t("bronze", table)
    ckpt = f"{P.checkpoints}/bronze_{table}"
    reader = (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", fmt)
        .option("cloudFiles.schemaLocation", f"{ckpt}/_schema")
        .option("cloudFiles.inferColumnTypes", "false")
        .option("cloudFiles.schemaEvolutionMode", "rescue")
    )
    for k, v in (extra_options or {}).items():
        reader = reader.option(k, v)

    q = (
        reader.load(f"{P.landing}/{source}")
        .select(
            "*",
            F.col("_metadata.file_path").alias("_source_file"),
            F.current_timestamp().alias("_ingest_ts"),
        )
        .writeStream.option("checkpointLocation", ckpt)
        .option("mergeSchema", "true")
        .trigger(availableNow=True)
        .toTable(target)
    )
    q.awaitTermination()
    n = spark.table(target).count()
    print(f"{target:<45} rows={n:,}")


autoload("accounts", "csv", "accounts", {"header": "true"})
autoload("merchants", "csv", "merchants", {"header": "true"})
autoload("transactions", "json", "transactions")

# COMMAND ----------

display(
    spark.table(P.t("bronze", "transactions"))
    .groupBy("_source_file")
    .count()
    .orderBy(F.col("_source_file").desc())
    .limit(10)
)
