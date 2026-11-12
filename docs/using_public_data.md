# Running the pipeline on a public dataset

The synthetic generator is good for demonstrating the pipeline, but a model trained on it partly learns
rules this repo wrote. To test the features on independent data, use the Kaggle
**"Credit Card Transactions Fraud Detection Dataset"** (Sparkov simulator; `fraudTrain.csv` +
`fraudTest.csv`, about 1.85M transactions, roughly 0.5% fraud).

## Steps

1. Download both CSVs from Kaggle.
2. Run `notebooks/00_setup_catalog` with `reset=true`.
3. Upload the two files to `/Volumes/workspace/fin_raw/files/public/`
   (Catalog Explorer → `fin_raw` → `files` → *Upload to this volume*).
4. Run `data/adapt_sparkov` instead of `data/generate_transactions`.
5. Run notebooks `01 → 02 → 03 → 05 → 06 → 07 → 08 → 04` as usual.

## What changes

| Field | Sparkov mapping | Effect on features |
|---|---|---|
| `account_id` | `cc_num` | — |
| `merchant_id` | factorised `merchant` name | — |
| `channel` | `*_net` categories → `online`, `*_pos` → `pos` | category suffix removed |
| `txn_lat/lon` | merchant location for POS, cardholder home for online | `dist_home_km`, `speed_kmh` still work |
| `device_id` | not available | `is_new_device` = 0 everywhere |
| `credit_limit`, `open_date` | not available | `amt_to_limit`, `account_age_days` imputed (constant) |
| `fraud_scenario` | not available | per-scenario recall table is skipped |
| `segment` | not available | ARI check in 08 is skipped |

The dataset is about 3.5× larger than the default synthetic data. On Free Edition, set
`legit_sample_frac=0.2` in notebook 06 if `toPandas()` runs out of memory.

## Reporting

Report both runs honestly: "PR-AUC X on synthetic data, Y on the independent Sparkov dataset". If the two
numbers are close, that's evidence the features generalise. If Sparkov is much lower, that's an honest
finding, and it's worth explaining why.
