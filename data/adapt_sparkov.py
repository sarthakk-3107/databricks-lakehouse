# Databricks notebook source
# MAGIC %md
# MAGIC # (Optional) Use a public dataset instead of the synthetic generator
# MAGIC
# MAGIC Converts the Kaggle **Sparkov credit-card fraud dataset** (`fraudTrain.csv`, `fraudTest.csv`)
# MAGIC into this project's landing-zone layout, so the *same* bronze → silver → gold → ML
# MAGIC pipeline runs on data whose fraud labels were not created by our own generator.
# MAGIC
# MAGIC 1. Download both CSVs from Kaggle.
# MAGIC 2. Upload them to `/Volumes/<catalog>/fin_raw/files/public/` (Catalog Explorer → volume → *Upload*).
# MAGIC 3. Run `00_setup_catalog` with `reset=true`, then this notebook, then notebooks `01 → 08`.
# MAGIC
# MAGIC See `docs/using_public_data.md` for what changes (no device IDs, no credit limits, no scenarios).

# COMMAND ----------

import glob
import os
import sys

if "__file__" in globals():
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
else:
    ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.insert(0, ROOT)

import pandas as pd

from lakehouse.generator import write_landing
from lakehouse.public_data import from_sparkov

try:
    dbutils  # noqa: F821
    IN_DATABRICKS = True
except NameError:
    IN_DATABRICKS = False

# COMMAND ----------

if IN_DATABRICKS:
    from lakehouse.config import Paths

    dbutils.widgets.text("catalog", "workspace")
    P = Paths(dbutils.widgets.get("catalog"))
    src_dir, landing = P.public_data, P.landing
else:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="./public")
    ap.add_argument("--out", default="./landing")
    a = ap.parse_args()
    src_dir, landing = a.src, a.out

files = sorted(glob.glob(os.path.join(src_dir, "fraud*.csv")))
if not files:
    raise FileNotFoundError(f"No fraudTrain.csv / fraudTest.csv found in {src_dir}")
print("Reading:", *files, sep="\n  ")

raw = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
data = from_sparkov(raw)
summary = write_landing(data, landing)
print(summary)
print(f"Fraud rate: {data['transactions']['is_fraud'].mean():.4%}")
