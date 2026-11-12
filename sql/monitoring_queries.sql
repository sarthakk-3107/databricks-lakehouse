-- Monitoring queries for an AI/BI (Lakeview) dashboard or SQL alerts.
-- Replace `workspace` with your catalog if you changed it.
-- System tables (system.*) must be enabled by a metastore admin; Free Edition may not expose them.

-- ===================== Pipeline health =====================

-- Job runs in the last 7 days
SELECT j.name AS job_name, r.run_id, r.period_start_time, r.period_end_time,
       timestampdiff(SECOND, r.period_start_time, r.period_end_time) AS duration_s,
       r.result_state, r.trigger_type
FROM system.lakeflow.job_run_timeline r
LEFT JOIN (
  SELECT job_id, name, row_number() OVER (PARTITION BY job_id ORDER BY change_time DESC) AS rn
  FROM system.lakeflow.jobs
) j ON r.job_id = j.job_id AND j.rn = 1
WHERE r.period_start_time >= current_date() - INTERVAL 7 DAYS
  AND j.name LIKE 'fin-lakehouse%'
ORDER BY r.period_start_time DESC;

-- Task-level durations (find the slow step)
SELECT t.task_key, avg(timestampdiff(SECOND, t.period_start_time, t.period_end_time)) AS avg_duration_s,
       sum(CASE WHEN t.result_state <> 'SUCCEEDED' THEN 1 ELSE 0 END) AS failures
FROM system.lakeflow.job_task_run_timeline t
WHERE t.period_start_time >= current_date() - INTERVAL 7 DAYS
GROUP BY t.task_key
ORDER BY avg_duration_s DESC;

-- Serverless DBU usage by day
SELECT usage_date, sku_name, round(sum(usage_quantity), 2) AS dbus
FROM system.billing.usage
WHERE usage_date >= current_date() - INTERVAL 30 DAYS
GROUP BY ALL
ORDER BY usage_date DESC;

-- ===================== Data freshness & quality =====================

-- Freshness check (alert when status = 'BREACH')
SELECT dataset, latest_event_ts, latest_ingest_ts, minutes_since_ingest,
       CASE WHEN minutes_since_ingest <= 24 * 60 THEN 'OK' ELSE 'BREACH' END AS status
FROM workspace.fin_gold.freshness;

-- Quarantine rate trend per DQ run
SELECT run_ts, bronze_rows, quarantined_rows,
       round(quarantined_rows / bronze_rows * 100, 3) AS quarantine_pct
FROM workspace.fin_silver.dq_run_log
ORDER BY run_ts DESC;

-- Write history of the silver table
DESCRIBE HISTORY workspace.fin_silver.transactions;

-- ===================== Model monitoring =====================

-- Daily alert rate and fraud capture at the chosen threshold
SELECT to_date(txn_ts) AS day, split,
       count(*) AS txns,
       round(avg(CAST(flagged AS INT)) * 100, 3) AS alert_rate_pct,
       sum(is_fraud) AS fraud,
       sum(CASE WHEN flagged AND is_fraud = 1 THEN 1 ELSE 0 END) AS fraud_caught,
       round(sum(CASE WHEN flagged AND is_fraud = 1 THEN amount ELSE 0 END), 2) AS fraud_dollars_caught
FROM workspace.fin_gold.fraud_scores
GROUP BY ALL
ORDER BY day;

-- Latest evaluation summary
SELECT evaluated_at, model, test_pr_auc, test_recall_at_fpr_1pct, op_precision, op_recall,
       op_dollar_recall, op_alert_rate, op_savings_vs_no_model, score_psi
FROM workspace.fin_gold.model_evaluation
ORDER BY evaluated_at DESC
LIMIT 10;

-- Segment-level fraud exposure
SELECT cluster, accounts, fraud_exposure, txn_per_day, avg_amount, online_share, travel_share
FROM workspace.fin_gold.segment_profiles
ORDER BY fraud_exposure DESC;
