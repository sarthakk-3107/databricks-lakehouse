# Databricks notebook source
# MAGIC %md
# MAGIC # 06 · Train fraud classifiers (tracked in MLflow)
# MAGIC
# MAGIC * **Out-of-time split:** train = first 70% of time, validation = next 15%, test = last 15%.
# MAGIC   Test is not touched here; it is only scored in `07`.
# MAGIC * **Candidates:** class-weighted logistic regression (interpretable baseline) and XGBoost
# MAGIC   (`scale_pos_weight` for imbalance, early stopping on validation PR-AUC).
# MAGIC * **Selection metric:** validation PR-AUC. With a fraud rate around 0.3%, ROC-AUC and accuracy look good
# MAGIC   even for weak models, so PR-AUC and recall at a fixed false-positive rate are the headline numbers.
# MAGIC * Every run logs params, metrics, signature and the full sklearn pipeline (preprocessing + model).
# MAGIC   The best run is registered in Unity Catalog as `<catalog>.fin_ml.fraud_classifier@champion`.

# COMMAND ----------

# MAGIC %pip install -q "xgboost>=2.0,<3.1"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import datetime as dt
import os
import sys
import uuid

ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.insert(0, ROOT)

import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from lakehouse.config import Paths
from lakehouse.metrics import score_report
from lakehouse.modeling import (
    ANALYSIS_ONLY_COLS, FEATURES, ID_COLS, LABEL, build_preprocessor, prepare_frame, time_split,
)

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("legit_sample_frac", "1.0")
P = Paths(dbutils.widgets.get("catalog"))
LEGIT_FRAC = float(dbutils.widgets.get("legit_sample_frac"))

# COMMAND ----------

# MAGIC %md ## Load features and split by time

# COMMAND ----------

feat = spark.table(P.t("ml", "fraud_features")).where("NOT is_warmup AND is_fraud IS NOT NULL")
pdf = prepare_frame(feat.select(*ID_COLS, *FEATURES, LABEL, *ANALYSIS_ONLY_COLS).toPandas())
train, val, test, (c1, c2) = time_split(pdf)

if LEGIT_FRAC < 1.0:  # optional negative down-sampling, TRAIN ONLY (val/test keep the real fraud rate)
    train = pd.concat(
        [train[train[LABEL] == 1], train[train[LABEL] == 0].sample(frac=LEGIT_FRAC, random_state=7)]
    ).sort_values("txn_ts")

for name, part in [("train", train), ("val", val), ("test", test)]:
    print(f"{name:<5} rows={len(part):>9,}  fraud={int(part[LABEL].sum()):>6,}  rate={part[LABEL].mean():.4%}  "
          f"{part.txn_ts.min()} → {part.txn_ts.max()}")

Xtr, ytr = train[FEATURES], train[LABEL].to_numpy()
Xva, yva = val[FEATURES], val[LABEL].to_numpy()

# COMMAND ----------

user = spark.sql("SELECT current_user()").first()[0]
EXPERIMENT = f"/Users/{user}/fin-lakehouse-fraud"
mlflow.set_experiment(EXPERIMENT)
mlflow.set_registry_uri("databricks-uc")
BATCH_ID = uuid.uuid4().hex[:12]

COMMON_PARAMS = {
    "train_rows": len(train),
    "train_frauds": int(ytr.sum()),
    "val_rows": len(val),
    "cutoff_train_end": str(c1),
    "cutoff_val_end": str(c2),
    "legit_sample_frac": LEGIT_FRAC,
    "n_features": len(FEATURES),
}


def log_candidate(run_name: str, family: str, pipe: Pipeline, params: dict) -> dict:
    s_val = pipe.predict_proba(Xva)[:, 1]
    metrics = score_report(yva, s_val, prefix="val_")
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags({"model_family": family, "training_batch": BATCH_ID, "project": "fin-lakehouse"})
        mlflow.log_params({**COMMON_PARAMS, **params})
        mlflow.log_metrics(metrics)
        sig = infer_signature(Xva.head(200), s_val[:200])
        info = mlflow.sklearn.log_model(
            pipe,
            artifact_path="model",
            signature=sig,
            input_example=Xva.head(5),
            code_paths=[os.path.join(ROOT, "lakehouse")],
        )
        mlflow.set_tag("model_uri", info.model_uri)
    print(f"{run_name:<22} " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items() if "rows" not in k))
    return {"run_id": run.info.run_id, "run_name": run_name, "model_family": family, "model_uri": info.model_uri, **metrics}


results = []

# COMMAND ----------

# MAGIC %md ## Baseline: class-weighted logistic regression

# COMMAND ----------

lr_params = {"C": 0.5, "class_weight": "balanced", "max_iter": 3000}
lr = Pipeline([("prep", build_preprocessor()), ("clf", LogisticRegression(**lr_params))])
lr.fit(Xtr, ytr)
results.append(log_candidate("logreg_baseline", "logreg", lr, lr_params))

# COMMAND ----------

# MAGIC %md ## XGBoost with early stopping on validation PR-AUC

# COMMAND ----------

spw = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
xgb_params = {
    "n_estimators": 1500,
    "learning_rate": 0.05,
    "max_depth": 5,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "scale_pos_weight": spw,
    "tree_method": "hist",
    "eval_metric": "aucpr",
    "early_stopping_rounds": 75,
    "random_state": 42,
    "n_jobs": -1,
}
prep = build_preprocessor()
Xtr_t = prep.fit_transform(Xtr)
Xva_t = prep.transform(Xva)
xgb = XGBClassifier(**xgb_params)
xgb.fit(Xtr_t, ytr, eval_set=[(Xva_t, yva)], verbose=False)
xgb_pipe = Pipeline([("prep", prep), ("clf", xgb)])  # already-fitted steps
results.append(log_candidate("xgboost", "xgboost", xgb_pipe, {**xgb_params, "best_iteration": int(xgb.best_iteration)}))

# COMMAND ----------

# MAGIC %md ## Record candidates and register the champion

# COMMAND ----------

cand = pd.DataFrame(results)
cand["training_batch"] = BATCH_ID
cand["trained_at"] = dt.datetime.utcnow()
cand["cutoff_train_end"] = str(c1)
cand["cutoff_val_end"] = str(c2)
cand["experiment"] = EXPERIMENT
spark.createDataFrame(cand).write.mode("append").option("mergeSchema", "true").saveAsTable(P.t("ml", "model_candidates"))

best = max(results, key=lambda r: r["val_pr_auc"])
print(f"Champion: {best['run_name']}  val_pr_auc={best['val_pr_auc']:.4f}")

model_name = f"{P.schema('ml')}.fraud_classifier"
try:
    mv = mlflow.register_model(best["model_uri"], model_name)
    mlflow.MlflowClient().set_registered_model_alias(model_name, "champion", mv.version)
    print(f"Registered {model_name} v{mv.version} as @champion")
except Exception as e:  # registry permissions vary by workspace
    print(f"Model registration skipped: {e}")

display(cand[["run_name", "val_pr_auc", "val_roc_auc", "val_recall_at_fpr_1pct", "val_precision_at_top_0p5pct"]])
