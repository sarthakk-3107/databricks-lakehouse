# Databricks notebook source
# MAGIC %md
# MAGIC # 07 · Evaluate on held-out time window and choose an operating threshold
# MAGIC
# MAGIC 1. Score **validation** and **test** with every candidate from the latest training batch.
# MAGIC 2. Pick the decision threshold on **validation** by minimising expected dollar cost:
# MAGIC    `cost = missed fraud $ + false_positive_cost × #false alarms`.
# MAGIC 3. Report that *fixed* threshold on **test** (no peeking), including dollar recall,
# MAGIC    alert rate, and savings versus no model.
# MAGIC 4. Diagnostics: PR curves, cost curve, calibration, feature importance, recall by fraud scenario,
# MAGIC    and score drift (PSI) between validation and test.
# MAGIC 5. Write scores to `fin_gold.fraud_scores` and a summary row to `fin_gold.model_evaluation`.

# COMMAND ----------

# MAGIC %pip install -q "xgboost>=2.0,<3.1"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import datetime as dt
import os
import sys

ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.insert(0, ROOT)

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve

from lakehouse.config import Paths
from lakehouse.metrics import apply_threshold, best_threshold, cost_curve, psi, score_report
from lakehouse.modeling import ANALYSIS_ONLY_COLS, FEATURES, ID_COLS, LABEL, prepare_frame, time_split

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("false_positive_cost", "10")
P = Paths(dbutils.widgets.get("catalog"))
FP_COST = float(dbutils.widgets.get("false_positive_cost"))
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

cands = spark.table(P.t("ml", "model_candidates")).toPandas()
latest_batch = cands.sort_values("trained_at").iloc[-1]["training_batch"]
cands = cands[cands["training_batch"] == latest_batch].sort_values("val_pr_auc", ascending=False)
cutoffs = (cands["cutoff_train_end"].iloc[0], cands["cutoff_val_end"].iloc[0])
mlflow.set_experiment(cands["experiment"].iloc[0])

feat = spark.table(P.t("ml", "fraud_features")).where("NOT is_warmup AND is_fraud IS NOT NULL")
pdf = prepare_frame(feat.select(*ID_COLS, *FEATURES, LABEL, *ANALYSIS_ONLY_COLS).toPandas())
_, val, test, _ = time_split(pdf, cutoffs=cutoffs)
print(f"val rows={len(val):,} fraud={int(val[LABEL].sum())} | test rows={len(test):,} fraud={int(test[LABEL].sum())}")

models = {r.run_name: mlflow.sklearn.load_model(r.model_uri) for r in cands.itertuples()}
scores = {name: {"val": m.predict_proba(val[FEATURES])[:, 1], "test": m.predict_proba(test[FEATURES])[:, 1]} for name, m in models.items()}

# COMMAND ----------

# MAGIC %md ## Candidate comparison on the test window

# COMMAND ----------

comparison = pd.DataFrame(
    [{"model": n, **score_report(test[LABEL], s["test"], prefix="test_")} for n, s in scores.items()]
)
display(comparison)

fig_pr, ax = plt.subplots(figsize=(6, 4.5))
for name, s in scores.items():
    p, r, _ = precision_recall_curve(test[LABEL], s["test"])
    ap = comparison.loc[comparison.model == name, "test_pr_auc"].iloc[0]
    ax.plot(r, p, label=f"{name} (PR-AUC {ap:.3f})")
ax.axhline(test[LABEL].mean(), ls="--", c="grey", lw=1, label="no-skill")
ax.set(xlabel="Recall", ylabel="Precision", title="Precision–recall, test window")
ax.legend()
fig_pr.tight_layout()
display(fig_pr)

# COMMAND ----------

# MAGIC %md ## Cost-based threshold (chosen on validation, applied to test)

# COMMAND ----------

champ = cands.iloc[0]
champ_name = champ["run_name"]
s_val, s_test = scores[champ_name]["val"], scores[champ_name]["test"]

curve = cost_curve(val[LABEL], s_val, val["amount"], fp_cost=FP_COST)
chosen = best_threshold(curve)
thr = chosen["threshold"]
test_ops = apply_threshold(test[LABEL], s_test, test["amount"], thr, fp_cost=FP_COST)

print(f"Champion: {champ_name}   false-positive cost assumption: ${FP_COST:.0f}")
print(f"Threshold (from validation): {thr:.4f}")
for k in ["alerts", "alert_rate", "precision", "recall", "dollar_recall", "fraud_amount_caught",
          "fraud_amount_missed", "fp", "total_cost", "baseline_cost_no_model", "savings_vs_no_model"]:
    v = test_ops[k]
    print(f"  test {k:<24} {v:,.4f}" if isinstance(v, float) and v < 10 else f"  test {k:<24} {v:,.0f}")

fig_cost, ax = plt.subplots(figsize=(6, 4))
ax.plot(curve["alert_rate"] * 100, curve["total_cost"], lw=1.5)
ax.axvline(chosen["alert_rate"] * 100, ls="--", c="grey", lw=1)
ax.set(xscale="log", xlabel="Alert rate (% of transactions, log)", ylabel="Expected cost ($)",
       title=f"Validation cost curve (FP cost ${FP_COST:.0f})")
fig_cost.tight_layout()
display(fig_cost)

# COMMAND ----------

# MAGIC %md ## Diagnostics

# COMMAND ----------

flag_test = s_test >= thr
scen = test.assign(flagged=flag_test)
scenario_table = (
    scen[scen["fraud_scenario"].notna()]
    .groupby("fraud_scenario")
    .agg(n=("flagged", "size"), labelled_fraud=(LABEL, "sum"), flagged=("flagged", "sum"))
    .assign(flag_rate=lambda d: (d["flagged"] / d["n"]).round(3))
    .sort_values("flag_rate", ascending=False)
    .reset_index()
)
if len(scenario_table):
    print("Recall by fraud scenario (synthetic ground truth; 'mimic' is designed to be hard):")
    display(scenario_table)

pipe = models[champ_name]
names = pipe.named_steps["prep"].get_feature_names_out()
clf = pipe.named_steps["clf"]
imp = clf.feature_importances_ if hasattr(clf, "feature_importances_") else np.abs(clf.coef_[0])
importance = pd.DataFrame({"feature": names, "importance": imp}).sort_values("importance", ascending=False).head(20)
fig_imp, ax = plt.subplots(figsize=(6, 5))
ax.barh(importance["feature"][::-1], importance["importance"][::-1])
ax.set(title=f"Top features ({champ_name})", xlabel="importance")
fig_imp.tight_layout()
display(fig_imp)

prob_true, prob_pred = calibration_curve(test[LABEL], s_test, n_bins=10, strategy="quantile")
fig_cal, ax = plt.subplots(figsize=(4.5, 4.5))
ax.plot(prob_pred, prob_true, marker="o")
ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=1)
ax.set(xscale="log", yscale="log", xlabel="Mean predicted", ylabel="Observed fraud rate",
       title="Calibration (test)")
fig_cal.tight_layout()
display(fig_cal)

score_psi = psi(s_val, s_test)
print(f"Score PSI validation→test: {score_psi:.4f}  (<0.1 stable, 0.1–0.25 watch, >0.25 investigate)")

# COMMAND ----------

# MAGIC %md ## Log to MLflow and write gold outputs

# COMMAND ----------

with mlflow.start_run(run_id=champ["run_id"]):
    mlflow.log_metrics({k: v for k, v in score_report(test[LABEL], s_test, prefix="test_").items()})
    mlflow.log_metrics({f"test_op_{k}": float(v) for k, v in test_ops.items()})
    mlflow.log_metric("score_psi_val_to_test", score_psi)
    mlflow.log_dict({"threshold": thr, "false_positive_cost": FP_COST, "chosen_on": "validation"}, "threshold.json")
    for fname, fig in [("pr_curve.png", fig_pr), ("cost_curve.png", fig_cost),
                       ("feature_importance.png", fig_imp), ("calibration.png", fig_cal)]:
        mlflow.log_figure(fig, f"plots/{fname}")
    if len(scenario_table):
        mlflow.log_table(scenario_table, "scenario_recall.json")

out = pd.concat(
    [
        val[["transaction_id", "account_id", "txn_ts", "amount", LABEL]].assign(split="val", fraud_score=s_val),
        test[["transaction_id", "account_id", "txn_ts", "amount", LABEL]].assign(split="test", fraud_score=s_test),
    ]
)
out["flagged"] = out["fraud_score"] >= thr
out["threshold"] = thr
out["model_run_id"] = champ["run_id"]
out["scored_at"] = dt.datetime.utcnow()
spark.createDataFrame(out).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(P.t("gold", "fraud_scores"))

summary = pd.DataFrame([{
    "evaluated_at": dt.datetime.utcnow(), "training_batch": latest_batch, "model": champ_name,
    "run_id": champ["run_id"], "false_positive_cost": FP_COST, "score_psi": score_psi,
    **{k: float(v) for k, v in score_report(test[LABEL], s_test, prefix="test_").items()},
    **{f"op_{k}": float(v) for k, v in test_ops.items()},
}])
spark.createDataFrame(summary).write.mode("append").option("mergeSchema", "true").saveAsTable(P.t("gold", "model_evaluation"))
display(summary.T)
