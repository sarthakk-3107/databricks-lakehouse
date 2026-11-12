# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · Pipeline & model monitoring
# MAGIC
# MAGIC Runs last in the Workflow. Sections:
# MAGIC 1. **Layer row counts:** bronze → silver → quarantine reconciliation
# MAGIC 2. **Data quality:** trend from `dq_run_log`, rule breakdown from `dq_daily`
# MAGIC 3. **Freshness check:** minutes since last ingest vs `freshness_sla_hours`
# MAGIC 4. **Delta history:** what wrote to silver, when, how many rows
# MAGIC 5. **Job health:** `system.lakeflow` tables (skipped gracefully if system tables aren't enabled)
# MAGIC 6. **Model monitoring:** daily alert rate and fraud capture from `fraud_scores`, score PSI
# MAGIC
# MAGIC Pin the `display()` outputs to a notebook dashboard (View → New dashboard) or rebuild them
# MAGIC in AI/BI dashboards from `sql/monitoring_queries.sql`.

# COMMAND ----------

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..")))
from lakehouse.config import Paths
from pyspark.sql import functions as F

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("freshness_sla_hours", "24")
P = Paths(dbutils.widgets.get("catalog"))
SLA_H = float(dbutils.widgets.get("freshness_sla_hours"))


def exists(name: str) -> bool:
    return spark.catalog.tableExists(name)

# COMMAND ----------

# MAGIC %md ## 1 · Reconciliation

# COMMAND ----------

tables = [
    ("bronze", "transactions"), ("silver", "transactions"), ("silver", "transactions_quarantine"),
    ("ml", "fraud_features"), ("gold", "fraud_scores"),
]
counts = [(f"{l}.{t}", spark.table(P.t(l, t)).count()) for l, t in tables if exists(P.t(l, t))]
display(spark.createDataFrame(counts, "table string, rows long"))

b = dict(counts).get("bronze.transactions", 0)
s = dict(counts).get("silver.transactions", 0)
q = dict(counts).get("silver.transactions_quarantine", 0)
status = "OK" if b == s + q else "MISMATCH"
print(f"bronze ({b:,}) = silver ({s:,}) + quarantine ({q:,}) → {status}")

# COMMAND ----------

# MAGIC %md ## 2 · Data quality

# COMMAND ----------

display(
    spark.table(P.t("silver", "dq_run_log"))
    .select("run_ts", "bronze_rows", "valid_rows", "quarantined_rows",
            F.round(F.col("quarantined_rows") / F.col("bronze_rows") * 100, 3).alias("quarantine_pct"),
            "failed_rule_counts", "warning_counts")
    .orderBy(F.col("run_ts").desc())
)
display(spark.table(P.t("gold", "dq_daily")).groupBy("rule").agg(F.sum("n_rows").alias("rows")).orderBy(F.desc("rows")))

# COMMAND ----------

# MAGIC %md ## 3 · Freshness check

# COMMAND ----------

fresh = spark.table(P.t("gold", "freshness")).withColumn(
    "sla_status", F.when(F.col("minutes_since_ingest") <= SLA_H * 60, "OK").otherwise("BREACH")
)
display(fresh)

# COMMAND ----------

# MAGIC %md ## 4 · Delta history (silver.transactions)

# COMMAND ----------

display(
    spark.sql(f"DESCRIBE HISTORY {P.t('silver', 'transactions')}")
    .select("version", "timestamp", "operation", "operationMetrics")
    .limit(20)
)

# COMMAND ----------

# MAGIC %md ## 5 · Job health (system tables)

# COMMAND ----------

try:
    display(spark.sql("""
        SELECT j.name AS job_name, r.run_id, r.period_start_time, r.period_end_time,
               timestampdiff(SECOND, r.period_start_time, r.period_end_time) AS duration_s,
               r.result_state
        FROM system.lakeflow.job_run_timeline r
        LEFT JOIN (SELECT job_id, name, row_number() OVER (PARTITION BY job_id ORDER BY change_time DESC) rn
                   FROM system.lakeflow.jobs) j
          ON r.job_id = j.job_id AND j.rn = 1
        WHERE r.period_start_time >= current_date() - INTERVAL 7 DAYS
          AND j.name LIKE 'fin-lakehouse%'
        ORDER BY r.period_start_time DESC
    """))
except Exception as e:
    print(f"System tables unavailable in this workspace ({type(e).__name__}); skipping job-health section.")

# COMMAND ----------

# MAGIC %md ## 6 · Model monitoring

# COMMAND ----------

if exists(P.t("gold", "fraud_scores")):
    daily = (
        spark.table(P.t("gold", "fraud_scores"))
        .groupBy(F.to_date("txn_ts").alias("day"), "split")
        .agg(
            F.count("*").alias("txns"),
            F.round(F.avg(F.col("flagged").cast("int")) * 100, 3).alias("alert_rate_pct"),
            F.sum("is_fraud").alias("fraud"),
            F.sum(F.when(F.col("flagged") & (F.col("is_fraud") == 1), 1).otherwise(0)).alias("fraud_caught"),
        )
        .withColumn("daily_recall", F.round(F.col("fraud_caught") / F.greatest(F.col("fraud"), F.lit(1)), 3))
        .orderBy("day")
    )
    display(daily)
if exists(P.t("gold", "model_evaluation")):
    display(spark.table(P.t("gold", "model_evaluation")).orderBy(F.desc("evaluated_at")).limit(5))
