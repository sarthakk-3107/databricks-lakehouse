# Databricks notebook source
# MAGIC %md
# MAGIC # 08 · Customer behaviour segmentation (clustering)
# MAGIC
# MAGIC Unsupervised view of the portfolio: aggregate each account's **legit** behaviour in PySpark,
# MAGIC then cluster with K-Means (scikit-learn; ~5k rows doesn't need distributed ML).
# MAGIC
# MAGIC * `k` chosen by silhouette score over k = 3…8.
# MAGIC * Clusters are profiled (spend level, frequency, online share, travel, etc.) and compared with
# MAGIC   fraud exposure; they're useful for segment-specific thresholds or marketing.
# MAGIC * On synthetic data we also measure agreement with the generator's hidden `segment`
# MAGIC   (Adjusted Rand Index) as a sanity check that the clusters recover real structure.

# COMMAND ----------

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..")))

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from lakehouse.config import Paths

dbutils.widgets.text("catalog", "workspace")
P = Paths(dbutils.widgets.get("catalog"))
F_T = P.t("ml", "fraud_features")

# COMMAND ----------

behaviour = spark.sql(f"""
    SELECT account_id,
           count(*)                                                            AS n_txn,
           count(*) / greatest(datediff(max(txn_ts), min(txn_ts)) + 1, 1)      AS txn_per_day,
           avg(amount)                                                         AS avg_amount,
           percentile_approx(amount, 0.9)                                      AS p90_amount,
           avg(CASE WHEN channel = 'online' THEN 1 ELSE 0 END)                 AS online_share,
           avg(is_night)                                                       AS night_share,
           count(DISTINCT merchant_category)                                   AS n_categories,
           count(DISTINCT merchant_id)                                         AS n_merchants,
           avg(is_cross_border)                                                AS cross_border_share,
           avg(dist_home_km)                                                   AS avg_dist_home_km,
           avg(CASE WHEN merchant_category = 'travel' THEN 1 ELSE 0 END)       AS travel_share,
           avg(CASE WHEN merchant_category IN ('grocery', 'gas') THEN 1 ELSE 0 END) AS essentials_share
    FROM {F_T}
    WHERE coalesce(is_fraud, 0) = 0
    GROUP BY account_id
    HAVING count(*) >= 10
""")
fraud_exposure = spark.sql(f"SELECT account_id, max(is_fraud) AS ever_fraud FROM {F_T} GROUP BY account_id")
seg = spark.table(P.t("silver", "accounts")).select("account_id", "segment")
pdf = behaviour.join(fraud_exposure, "account_id", "left").join(seg, "account_id", "left").toPandas()
print(f"accounts: {len(pdf):,}")

# COMMAND ----------

LOG_COLS = ["n_txn", "txn_per_day", "avg_amount", "p90_amount", "n_merchants", "avg_dist_home_km"]
RAW_COLS = ["online_share", "night_share", "n_categories", "cross_border_share", "travel_share", "essentials_share"]
X = pdf[LOG_COLS + RAW_COLS].astype(float).copy()
X = X.fillna(X.median())
X[LOG_COLS] = np.log1p(X[LOG_COLS].clip(lower=0))
Xs = StandardScaler().fit_transform(X)

sil = {}
fits = {}
for k in range(3, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Xs)
    sil[k] = silhouette_score(Xs, km.labels_, sample_size=min(5000, len(Xs)), random_state=42)
    fits[k] = km
best_k = max(sil, key=sil.get)
pdf["cluster"] = fits[best_k].labels_
print("silhouette by k:", {k: round(v, 3) for k, v in sil.items()}, "→ best k =", best_k)

# COMMAND ----------

profile = (
    pdf.groupby("cluster")
    .agg(accounts=("account_id", "size"), fraud_exposure=("ever_fraud", "mean"),
         **{c: (c, "mean") for c in ["txn_per_day", "avg_amount", "p90_amount", "online_share", "night_share",
                                     "travel_share", "essentials_share", "cross_border_share", "n_merchants"]})
    .round(3)
    .reset_index()
)
display(profile)

ari = None
if pdf["segment"].notna().any() and pdf["segment"].nunique() > 1 and set(pdf["segment"].unique()) != {"unknown"}:
    ari = adjusted_rand_score(pdf["segment"], pdf["cluster"])
    print(f"Adjusted Rand Index vs generator segment: {ari:.3f}")
    display(pd.crosstab(pdf["cluster"], pdf["segment"]))

# COMMAND ----------

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(list(sil), list(sil.values()), marker="o")
axes[0].set(xlabel="k", ylabel="silhouette", title="Choosing k")
xy = PCA(n_components=2, random_state=42).fit_transform(Xs)
sc = axes[1].scatter(xy[:, 0], xy[:, 1], c=pdf["cluster"], s=4, cmap="tab10")
axes[1].set(title=f"Accounts in PCA space (k={best_k})", xlabel="PC1", ylabel="PC2")
fig.tight_layout()
display(fig)

# COMMAND ----------

spark.createDataFrame(pdf.drop(columns=["segment"])).write.mode("overwrite").option("overwriteSchema", "true") \
    .saveAsTable(P.t("gold", "account_segments"))
spark.createDataFrame(profile).write.mode("overwrite").option("overwriteSchema", "true") \
    .saveAsTable(P.t("gold", "segment_profiles"))

user = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{user}/fin-lakehouse-fraud")
with mlflow.start_run(run_name="kmeans_segmentation"):
    mlflow.set_tag("model_family", "kmeans")
    mlflow.log_params({"best_k": best_k, "features": ",".join(LOG_COLS + RAW_COLS)})
    mlflow.log_metrics({f"silhouette_k{k}": v for k, v in sil.items()})
    if ari is not None:
        mlflow.log_metric("ari_vs_segment", ari)
    mlflow.log_figure(fig, "plots/segmentation.png")
