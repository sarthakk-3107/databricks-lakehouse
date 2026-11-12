# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · Unity Catalog setup
# MAGIC
# MAGIC Creates one schema per layer plus a managed volume for raw files and checkpoints:
# MAGIC
# MAGIC ```
# MAGIC <catalog>
# MAGIC ├── fin_raw      (volume: files/ → landing/, pending/, checkpoints/, public/)
# MAGIC ├── fin_bronze   raw, append-only, all strings
# MAGIC ├── fin_silver   typed, validated, deduplicated (+ quarantine, dq_run_log)
# MAGIC ├── fin_gold     business aggregates, scores, segments
# MAGIC └── fin_ml       feature table, model candidates, registered model
# MAGIC ```
# MAGIC
# MAGIC Default catalog is `workspace` (exists in Free Edition). Set `reset=true` to drop everything and start over.

# COMMAND ----------

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..")))
from lakehouse.config import LAYERS, Paths

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.dropdown("reset", "false", ["false", "true"])
P = Paths(dbutils.widgets.get("catalog"))
RESET = dbutils.widgets.get("reset") == "true"

# COMMAND ----------

if P.catalog != "workspace":
    try:
        spark.sql(f"CREATE CATALOG IF NOT EXISTS {P.catalog}")
    except Exception as e:  # Free Edition / limited permissions
        print(f"Could not create catalog {P.catalog!r} ({e}). Use an existing catalog instead.")
        raise

if RESET:
    for layer in reversed(LAYERS):
        print("dropping", P.schema(layer))
        spark.sql(f"DROP SCHEMA IF EXISTS {P.schema(layer)} CASCADE")

COMMENTS = {
    "raw": "Raw landing files (volume) and streaming checkpoints",
    "bronze": "Raw ingested records, append-only, schema-on-read",
    "silver": "Typed, validated, deduplicated records + quarantine",
    "gold": "Business-level aggregates, fraud scores, customer segments",
    "ml": "Feature tables, model candidates, registered models",
}
for layer in LAYERS:
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {P.schema(layer)} COMMENT '{COMMENTS[layer]}'")

spark.sql(f"CREATE VOLUME IF NOT EXISTS {P.schema('raw')}.{P.volume} COMMENT 'Landing zone + checkpoints'")

for d in (P.landing, P.pending, P.checkpoints, P.public_data):
    os.makedirs(d, exist_ok=True)

display(spark.sql(f"SHOW SCHEMAS IN {P.catalog} LIKE '{P.prefix}_*'"))
print("Landing zone:", P.landing)
