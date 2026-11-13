# Financial Transaction Lakehouse + Fraud Detection on Databricks

An end-to-end project on Databricks. It takes raw card transactions through a **medallion lakehouse**
(bronze → silver → gold) with Delta Lake and declarative data-quality gates, then trains a
**fraud-detection model** on point-in-time PySpark features. Models are tracked in MLflow, evaluated on an
out-of-time test window, and given a **cost-based decision threshold**. Customers are also segmented
with clustering.

## Architecture

```
   Raw JSON / CSV  (Unity Catalog volume)
            │
            ▼
   ┌──────────────────┐
   │  BRONZE (raw)    │  Auto Loader, all-string schema, rescued data, append-only
   └────────┬─────────┘
            ▼
   ┌──────────────────┐   ┌────────────────────┐
   │  SILVER (clean)  │──▶│ QUARANTINE + DQ log │  typed, validated, deduped (MERGE)
   └────────┬─────────┘   └────────────────────┘
            │
     ┌──────┴───────────────────────────┐
     ▼                                  ▼
┌──────────────────┐        ┌──────────────────────────┐
│ GOLD (analytics) │        │ ML FEATURES (PySpark)    │  velocity, spend deviation, geo /
│ daily volume,    │        │ point-in-time correct,   │  impossible travel, novelty,
│ fraud by segment,│        │ label-delay aware        │  merchant fraud prior
│ merchant risk,   │        └───────────┬──────────────┘
│ DQ, freshness    │                    ▼
└────────┬─────────┘        ┌──────────────────────────┐
         │                  │ TRAIN (MLflow)           │  logistic regression baseline vs XGBoost,
         │                  │ out-of-time split        │  early stopping on val PR-AUC,
         │                  └───────────┬──────────────┘  champion registered in Unity Catalog
         │                              ▼
         │                  ┌──────────────────────────┐
         │                  │ EVALUATE + THRESHOLD     │  cost curve on val → fixed threshold on test,
         │                  │ → gold.fraud_scores      │  calibration, feature importance, PSI drift
         │                  └───────────┬──────────────┘
         │                              │      ┌──────────────────────────┐
         │                              │      │ SEGMENTATION (K-Means)   │ → gold.account_segments
         │                              │      └───────────┬──────────────┘
         ▼                              ▼                  ▼
   ┌───────────────────────────────────────────────────────────┐
   │ MONITORING: reconciliation, DQ trend, freshness, Delta     │
   │ history, job health (system tables), model alert rate      │
   └───────────────────────────────────────────────────────────┘
```

## Tech stack

* **Databricks** Free Edition (serverless), or any Unity Catalog workspace on AWS
* **Delta Lake**: ACID, MERGE upserts, schema enforcement, time travel, change data feed
* **Auto Loader** (`cloudFiles`): incremental file ingestion
* **Delta Live Tables / Lakeflow Declarative Pipelines**: `expect`, `expect_or_drop`, `expect_or_fail`
* **Databricks Workflows**: a DAG of 10 tasks
* **Unity Catalog**: governance, lineage, model registry
* **PySpark + SQL**: ETL and window-function feature engineering
* **MLflow**: experiment tracking, model logging, UC model registry
* **scikit-learn, XGBoost**: classification, K-Means clustering

## Repo layout

```
databricks-lakehouse/
├── README.md
├── SETUP.md                            ← step-by-step setup from zero
├── lakehouse/                          ← shared, unit-tested Python package
│   ├── config.py                       ← catalog / schema / volume naming
│   ├── generator.py                    ← behaviour-driven synthetic data + fraud scenarios
│   ├── public_data.py                  ← adapter for the Kaggle Sparkov fraud dataset
│   ├── spark_features.py               ← point-in-time PySpark feature engineering
│   ├── modeling.py                     ← feature lists, time split, sklearn preprocessor
│   ├── metrics.py                      ← PR-AUC, recall@FPR, cost curve, PSI
│   └── geo.py                          ← haversine
├── data/
│   ├── generate_transactions.py        ← synthetic data → landing zone (notebook or CLI)
│   └── adapt_sparkov.py                ← optional: public dataset → landing zone
├── notebooks/
│   ├── 00_setup_catalog.py             ← schemas + volume (reset option)
│   ├── 01_bronze_ingest.py             ← Auto Loader → bronze
│   ├── 02_silver_clean.py              ← typing, DQ rules, quarantine, MERGE → silver
│   ├── 03_gold_aggregations.py         ← business aggregates → gold
│   ├── 04_monitoring_dashboard.py      ← pipeline + model monitoring (runs last)
│   ├── 05_feature_engineering.py       ← fraud features → fin_ml.fraud_features
│   ├── 06_train_fraud_model.py         ← LR vs XGBoost, MLflow, UC registry
│   ├── 07_evaluate_and_threshold.py    ← test-window eval, cost-based threshold, scores
│   └── 08_customer_segmentation.py     ← K-Means behaviour segments
├── dlt/
│   ├── transaction_pipeline.py         ← declarative version of 01–03
│   └── pipeline_settings.json
├── sql/
│   ├── quality_checks.sql              ← reconciliation, PK, referential, nulls, leakage guard
│   └── monitoring_queries.sql          ← system tables, freshness, model alert rate
├── workflows/
│   └── workflow.json                   ← Databricks Workflow (Jobs API) definition
├── tests/                              ← pytest for generator, metrics, preprocessing
└── docs/
    ├── architecture.md
    ├── design_decisions.md             ← why each choice was made (interview prep)
    ├── using_public_data.md
    └── screenshots/
```

## Quick start

See [SETUP.md](SETUP.md) for the full walkthrough. Short version:

1. Create a free Databricks workspace.
2. Add this repo as a **Git folder** (Workspace → Create → Git folder).
3. Run `notebooks/00_setup_catalog`, then `data/generate_transactions`.
4. Run notebooks `01 → 02 → 03 → 05 → 06 → 07 → 08 → 04`, **or** create the Workflow from `workflows/workflow.json`.
5. Open the MLflow experiment `/Users/<you>/fin-lakehouse-fraud` and the `04_monitoring_dashboard` outputs.

Local tests (no Databricks needed): `pip install -r requirements-dev.txt && pytest -q`

## What this demonstrates

**Data engineering**
* Medallion architecture on Delta Lake with idempotent `MERGE` upserts
* Incremental ingestion with Auto Loader (try it with `hold_back_days`, then drop in the held-back files)
* Data-quality gates in three modes (warn / drop to quarantine / fail), reconciliation, DQ run log
* Orchestration with Workflows; monitoring with Delta history and system tables

**Data science / ML**
* Fraud features built with PySpark window functions that **only use past data** (`rangeBetween(-N, -1)`)
* A merchant fraud-rate prior that respects a **30-day label delay**, since chargebacks arrive late
* **Out-of-time** train / validation / test split instead of a random one
* Class imbalance handled with class weights and `scale_pos_weight`, judged on **PR-AUC** and **recall at 1% FPR**
* Logistic-regression baseline vs XGBoost, all runs tracked in **MLflow**, champion registered in **Unity Catalog**
* **Cost-based threshold**: trades missed-fraud dollars against false-decline friction; chosen on validation and reported on test
* Diagnostics: calibration, feature importance, recall per fraud scenario, score drift (PSI)
* **K-Means** customer segmentation, with k chosen by silhouette score and checked against the hidden segment (ARI)

## Results

> Fill these in after running notebook 07. The numbers come from `fin_gold.model_evaluation` and the MLflow run.

| Metric (test window) | Logistic regression | XGBoost |
|---|---|---|
| PR-AUC | | |
| ROC-AUC | | |
| Recall @ 1% FPR | | |
| Precision in top 0.5% | | |

| Operating point (XGBoost, FP cost = $10) | Value |
|---|---|
| Threshold (chosen on validation) | |
| Alert rate | |
| Precision / recall | |
| Dollar recall | |
| Savings vs. no model | |

## About the data

The default data is **synthetic**, so results should be read with that in mind. The generator is
built so the problem is not trivial. Customers have habits (home city, favourite merchants, spend by
segment) and do legitimate things that look suspicious (travel, new phones, large purchases). Fraud
comes from four scenarios, one of which (`mimic`) is designed to look like normal spending. There is
label noise (unreported fraud, friendly-fraud disputes) and about 0.4% dirty records. `fraud_scenario`
is ground truth for analysis only and is never used as a feature.

To check that the features work on data this repo did not generate, run the same pipeline on the
public Kaggle **Sparkov** dataset. See [docs/using_public_data.md](docs/using_public_data.md).



## Results (synthetic data, out-of-time test window: 63,551 transactions, 262 fraud)

| Metric | Logistic regression | XGBoost |
|---|---|---|
| PR-AUC | 0.686 | **0.757** |
| ROC-AUC | 0.924 | **0.929** |
| Recall @ 1% FPR | 0.840 | 0.840 |
| Recall @ 0.1% FPR | 0.611 | **0.832** |
| Precision in top 0.5% | 0.607 | **0.689** |

| Operating point (XGBoost, FP cost = $10, threshold chosen on validation) | Value |
|---|---|
| Alert rate | 0.33% (208 alerts) |
| Precision / recall | 0.90 / 0.71 |
| Dollar recall | 95.3% |
| Cost vs. no model | $3.6K vs. $73.0K (simulated) |
| Score PSI (val → test) | 0.0004 |

| Fraud scenario | Flag rate |
|---|---|
| account_takeover | 100% |
| cloned_card | 89% |
| card_testing | 67% |
| mimic (designed to look normal) | 0% |

Customer segmentation: K-Means, k = 3 (silhouette 0.26), ARI 0.43 vs. hidden segments.
