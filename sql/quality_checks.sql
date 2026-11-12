-- Standalone data-quality checks. Run in the SQL editor.
-- Replace `workspace` with your catalog if you changed it.

-- 1. Reconciliation: every bronze row is either in silver or quarantine
SELECT
  (SELECT count(*) FROM workspace.fin_bronze.transactions)            AS bronze_rows,
  (SELECT count(*) FROM workspace.fin_silver.transactions)            AS silver_rows,
  (SELECT count(*) FROM workspace.fin_silver.transactions_quarantine) AS quarantined_rows,
  bronze_rows = silver_rows + quarantined_rows                        AS reconciled;

-- 2. Primary key uniqueness in silver (expect 0 rows)
SELECT transaction_id, count(*) AS n
FROM workspace.fin_silver.transactions
GROUP BY transaction_id
HAVING count(*) > 1;

-- 3. Referential integrity (expect 0)
SELECT count(*) AS orphan_merchants
FROM workspace.fin_silver.transactions t
LEFT ANTI JOIN workspace.fin_silver.merchants m USING (merchant_id);

SELECT count(*) AS orphan_accounts
FROM workspace.fin_silver.transactions t
LEFT ANTI JOIN workspace.fin_silver.accounts a USING (account_id);

-- 4. Null rate per column (silver)
SELECT
  round(avg(CASE WHEN amount      IS NULL THEN 1 ELSE 0 END) * 100, 3) AS amount_null_pct,
  round(avg(CASE WHEN txn_ts      IS NULL THEN 1 ELSE 0 END) * 100, 3) AS txn_ts_null_pct,
  round(avg(CASE WHEN device_id   IS NULL THEN 1 ELSE 0 END) * 100, 3) AS device_null_pct,   -- expected for POS
  round(avg(CASE WHEN txn_lat     IS NULL THEN 1 ELSE 0 END) * 100, 3) AS lat_null_pct,
  round(avg(CASE WHEN is_fraud    IS NULL THEN 1 ELSE 0 END) * 100, 3) AS label_null_pct
FROM workspace.fin_silver.transactions;

-- 5. Value ranges
SELECT min(amount) AS min_amt, max(amount) AS max_amt,
       percentile_approx(amount, 0.5) AS median_amt,
       min(txn_ts) AS first_ts, max(txn_ts) AS last_ts
FROM workspace.fin_silver.transactions;

-- 6. Quarantine breakdown by rule
SELECT rule, count(*) AS n
FROM workspace.fin_silver.transactions_quarantine
LATERAL VIEW explode(failed_rules) r AS rule
GROUP BY rule
ORDER BY n DESC;

-- 7. Volume anomaly: days whose count deviates > 3 sigma from the trailing 28-day mean
WITH d AS (
  SELECT txn_date, n_txn,
         avg(n_txn)    OVER (ORDER BY txn_date ROWS BETWEEN 28 PRECEDING AND 1 PRECEDING) AS mu,
         stddev(n_txn) OVER (ORDER BY txn_date ROWS BETWEEN 28 PRECEDING AND 1 PRECEDING) AS sd
  FROM workspace.fin_gold.daily_transaction_summary
)
SELECT * FROM d WHERE sd > 0 AND abs(n_txn - mu) > 3 * sd ORDER BY txn_date;

-- 8. Feature table leakage guard: label delay respected (expect 0)
--    merchant_prior_txn_cnt must be 0 for every row in a merchant's first 30 days
SELECT count(*) AS violations
FROM (
  SELECT merchant_id, txn_ts, merchant_prior_txn_cnt,
         min(txn_ts) OVER (PARTITION BY merchant_id) AS first_seen
  FROM workspace.fin_ml.fraud_features
)
WHERE txn_ts < first_seen + INTERVAL 30 DAYS AND merchant_prior_txn_cnt > 0;
