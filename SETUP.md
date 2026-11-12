# Setup, from zero

Expect about 30–45 minutes the first time, most of it waiting for serverless compute.

## 1. Workspace

1. Sign up for **Databricks Free Edition** (or use any workspace with Unity Catalog and serverless).
2. Confirm you have the `workspace` catalog: **Catalog** → `workspace`. If you use a different catalog,
   pass its name in the `catalog` widget/parameter everywhere.

## 2. Get the code into Databricks

1. Push this folder to a GitHub repo, e.g. `https://github.com/<you>/databricks-lakehouse`.
2. In Databricks: **Workspace → Create → Git folder**, paste the URL and create it.
   The notebooks expect this layout: each one adds the repo root (`..`) to `sys.path` so
   `import lakehouse` works.

## 3. Run step by step (recommended the first time)

Open each notebook, attach **Serverless** compute, and click **Run all**.

| Step | Notebook | Takes | Creates |
|---|---|---|---|
| 1 | `notebooks/00_setup_catalog` | < 1 min | `fin_raw` (+ volume), `fin_bronze`, `fin_silver`, `fin_gold`, `fin_ml` |
| 2 | `data/generate_transactions` | 1–3 min | ~0.5M transactions as 180 daily JSON files + 2 CSVs |
| 3 | `notebooks/01_bronze_ingest` | 1–2 min | `fin_bronze.transactions / accounts / merchants` |
| 4 | `notebooks/02_silver_clean` | 1–2 min | `fin_silver.*`, quarantine, `dq_run_log` |
| 5 | `notebooks/03_gold_aggregations` | < 1 min | 5 gold tables |
| 6 | `notebooks/05_feature_engineering` | 2–4 min | `fin_ml.fraud_features` |
| 7 | `notebooks/06_train_fraud_model` | 3–8 min | MLflow runs, `fin_ml.model_candidates`, registered model |
| 8 | `notebooks/07_evaluate_and_threshold` | 1–3 min | `fin_gold.fraud_scores`, `fin_gold.model_evaluation` |
| 9 | `notebooks/08_customer_segmentation` | 1–2 min | `fin_gold.account_segments`, `segment_profiles` |
| 10 | `notebooks/04_monitoring_dashboard` | < 1 min | monitoring views |

The generator's `n_accounts` and `days` widgets control data size. If the serverless notebook runs out
of memory during `toPandas()` in 06/07, set `legit_sample_frac` in 06 to `0.3` (train-only down-sampling)
or reduce `n_accounts`.

## 4. Run it as a Workflow

1. Edit `workflows/workflow.json`: set `git_url` to your repo.
2. **Jobs & Pipelines → Create → Job →** ⋮ **→ Edit as JSON** (or *View JSON → Edit*), paste the file, save.
   With the Databricks CLI: `databricks jobs create --json @workflows/workflow.json`.
3. Click **Run now**. With no cluster specified, tasks run on serverless compute.
   The schedule is included but **paused**.

## 5. (Optional) Declarative pipeline (DLT)

1. **Jobs & Pipelines → Create → ETL pipeline**.
2. Source code: `dlt/transaction_pipeline` from your Git folder.
3. Destination: catalog `workspace`, schema `fin_dlt`.
4. Configuration: add key `lakehouse.landing` with value `/Volumes/workspace/fin_raw/files/landing`.
   (`dlt/pipeline_settings.json` contains the same settings as JSON.)
5. Start it, then open the pipeline graph and the **Data quality** tab to see expectation pass/fail counts.

## 6. Show incremental ingestion

1. Run `00_setup_catalog` with `reset=true`.
2. Run `generate_transactions` with `hold_back_days=7` and `overwrite=true`.
3. Run `01_bronze_ingest` and note the row count.
4. Move the held-back files:
   ```python
   P = "/Volumes/workspace/fin_raw/files"
   for f in dbutils.fs.ls(f"{P}/pending/transactions"):
       dbutils.fs.mv(f.path, f"{P}/landing/transactions/{f.name}")
   ```
5. Run `01_bronze_ingest` again. Only the 7 new files are read, and `02` merges them without duplicates.

## 7. Screenshots for the README

Save these to `docs/screenshots/`: the Workflow DAG run, the DLT graph with the Data quality tab,
the MLflow run comparison, the PR and cost curves from 07, and the monitoring dashboard.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: lakehouse` | Run notebooks from the Git folder; don't copy them out of the repo. |
| `CREATE CATALOG` permission error | Keep `catalog=workspace`. |
| Rewritten files are not re-ingested | Expected: Auto Loader tracks files it has seen. Run `00` with `reset=true`. |
| `system.lakeflow...` not found | System tables aren't enabled in the workspace; the monitoring notebook skips that section. |
| Model registration skipped | UC model registry permissions vary; the model is still logged in the MLflow run and 07 still works. |
| `xgboost` import error after `%pip` | Make sure the `restartPython()` cell ran; re-run the notebook from the top. |
