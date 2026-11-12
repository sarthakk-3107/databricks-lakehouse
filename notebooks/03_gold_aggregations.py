# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Gold: business aggregates
# MAGIC
# MAGIC | Table | Grain | Used for |
# MAGIC |---|---|---|
# MAGIC | `daily_transaction_summary` | day | volume, fraud rate trend |
# MAGIC | `fraud_by_category_channel` | category × channel × segment | where fraud concentrates |
# MAGIC | `merchant_risk` | merchant (≥ 50 txns) | high-risk merchant watchlist |
# MAGIC | `dq_daily` | source file date × rule | data-quality trend |
# MAGIC | `freshness` | dataset | freshness SLA check in 04 |
# MAGIC
# MAGIC All tables are rebuilt with `CREATE OR REPLACE TABLE ... AS SELECT`, so the notebook is idempotent.

# COMMAND ----------

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..")))
from lakehouse.config import Paths

dbutils.widgets.text("catalog", "workspace")
P = Paths(dbutils.widgets.get("catalog"))
S, G = P.schema("silver"), P.schema("gold")

# COMMAND ----------

statements = {
    "daily_transaction_summary": f"""
        SELECT txn_date,
               count(*)                                              AS n_txn,
               round(sum(amount), 2)                                 AS total_amount,
               count(DISTINCT account_id)                            AS active_accounts,
               sum(is_fraud)                                         AS n_fraud,
               round(sum(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END), 2) AS fraud_amount,
               round(avg(is_fraud) * 100, 4)                         AS fraud_rate_pct
        FROM {S}.transactions
        GROUP BY txn_date
    """,
    "fraud_by_category_channel": f"""
        SELECT t.merchant_category, t.channel, a.segment,
               count(*)                         AS n_txn,
               sum(t.is_fraud)                  AS n_fraud,
               round(avg(t.is_fraud) * 100, 4)  AS fraud_rate_pct,
               round(sum(CASE WHEN t.is_fraud = 1 THEN t.amount ELSE 0 END), 2) AS fraud_amount
        FROM {S}.transactions t
        LEFT JOIN {S}.accounts a USING (account_id)
        GROUP BY ALL
    """,
    "merchant_risk": f"""
        SELECT m.merchant_id, m.merchant_name, m.category, m.is_online,
               count(*)                         AS n_txn,
               sum(t.is_fraud)                  AS n_fraud,
               round(avg(t.is_fraud) * 100, 4)  AS fraud_rate_pct,
               round(sum(t.amount), 2)          AS volume
        FROM {S}.transactions t
        JOIN {S}.merchants m USING (merchant_id)
        GROUP BY ALL
        HAVING count(*) >= 50
    """,
    "dq_daily": f"""
        SELECT regexp_extract(_source_file, '([0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}})', 1) AS file_date,
               rule,
               count(*) AS n_rows
        FROM {S}.transactions_quarantine
        LATERAL VIEW explode(failed_rules) r AS rule
        GROUP BY ALL
    """,
    "freshness": f"""
        SELECT 'transactions'                                                  AS dataset,
               max(txn_ts)                                                     AS latest_event_ts,
               max(_ingest_ts)                                                 AS latest_ingest_ts,
               timestampdiff(MINUTE, max(_ingest_ts), current_timestamp())     AS minutes_since_ingest,
               count(*)                                                        AS row_count
        FROM {S}.transactions
    """,
}

for name, sql in statements.items():
    spark.sql(f"CREATE OR REPLACE TABLE {G}.{name} AS {sql}")
    print(f"{G}.{name:<28} rows={spark.table(f'{G}.{name}').count():,}")

# COMMAND ----------

display(spark.sql(f"SELECT * FROM {G}.fraud_by_category_channel ORDER BY fraud_rate_pct DESC LIMIT 15"))

# COMMAND ----------

display(spark.sql(f"SELECT * FROM {G}.daily_transaction_summary ORDER BY txn_date"))
