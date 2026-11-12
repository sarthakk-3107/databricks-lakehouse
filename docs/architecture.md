# Architecture

## Layers and tables

| Layer | Table | Written by | Pattern |
|---|---|---|---|
| raw | volume `fin_raw.files` → `landing/`, `pending/`, `checkpoints/`, `public/` | generator / adapter | files |
| bronze | `transactions`, `accounts`, `merchants` | 01 | Auto Loader, append, all strings, `_source_file`, `_ingest_ts`, `_rescued_data` |
| silver | `accounts`, `merchants` | 02 | latest record per key, overwrite |
| silver | `transactions` | 02 | typed + validated, `MERGE` on `transaction_id`, change data feed on |
| silver | `transactions_quarantine` | 02 | failed rows + `failed_rules` array |
| silver | `dq_run_log` | 02 | one row per run: counts per rule, warnings |
| gold | `daily_transaction_summary`, `fraud_by_category_channel`, `merchant_risk`, `dq_daily`, `freshness` | 03 | `CREATE OR REPLACE TABLE AS` |
| ml | `fraud_features` | 05 | point-in-time features, overwrite |
| ml | `model_candidates` | 06 | one row per trained model, append |
| ml | model `fraud_classifier@champion` | 06 | UC model registry |
| gold | `fraud_scores`, `model_evaluation` | 07 | scores at chosen threshold; evaluation history |
| gold | `account_segments`, `segment_profiles` | 08 | K-Means output |

## Workflow DAG

```
setup_catalog → generate_data → bronze_ingest → silver_clean ─┬─→ gold_aggregations ─────────────────┐
                                                              └─→ feature_engineering ─┬─→ train → evaluate ─┤
                                                                                       └─→ segmentation ─────┤
                                                                                                             ▼
                                                                                                        monitoring
```

## Feature timeline (why there's no leakage)

```
 time ───────────────────────────────────────────────────────────────▶
          |<──────── 30 days ────────>|<─ label delay (30d) ─>|  txn t
 account windows:     [t-30d, t-1s]  (strictly before t)
 merchant prior:  (-∞ ............ t-30d]  (only labels old enough to be known)
 split:   [ train 70% ][ val 15% ][ test 15% ]   (by time, never random)
```
