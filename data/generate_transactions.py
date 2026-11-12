# Databricks notebook source
# MAGIC %md
# MAGIC # Generate synthetic transactions
# MAGIC
# MAGIC Writes raw files into the Unity Catalog volume that Auto Loader watches:
# MAGIC
# MAGIC | Path | Format | Contents |
# MAGIC |---|---|---|
# MAGIC | `landing/transactions/transactions_YYYY-MM-DD.json` | JSON lines, 1 file/day | card transactions incl. dirty rows |
# MAGIC | `landing/accounts/accounts.csv` | CSV | account dimension |
# MAGIC | `landing/merchants/merchants.csv` | CSV | merchant dimension |
# MAGIC
# MAGIC Fraud is generated from behavioural scenarios (card testing, account takeover,
# MAGIC cloned card, "mimic") with label noise. See `lakehouse/generator.py` for details.
# MAGIC
# MAGIC Runs in Databricks **or** locally: `python data/generate_transactions.py --out ./landing`

# COMMAND ----------

import os
import sys

if "__file__" in globals():
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
else:  # Databricks notebook in a Git folder: cwd is this notebook's folder
    ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.insert(0, ROOT)

from lakehouse.generator import GeneratorConfig, generate, write_landing

try:
    dbutils  # noqa: F821  (defined in Databricks)
    IN_DATABRICKS = True
except NameError:
    IN_DATABRICKS = False

# COMMAND ----------

if IN_DATABRICKS:
    from lakehouse.config import Paths

    dbutils.widgets.text("catalog", "workspace")
    dbutils.widgets.text("n_accounts", "5000")
    dbutils.widgets.text("days", "180")
    dbutils.widgets.text("seed", "42")
    dbutils.widgets.text("hold_back_days", "0")
    dbutils.widgets.dropdown("overwrite", "false", ["false", "true"])

    P = Paths(dbutils.widgets.get("catalog"))
    cfg = GeneratorConfig(
        n_accounts=int(dbutils.widgets.get("n_accounts")),
        days=int(dbutils.widgets.get("days")),
        seed=int(dbutils.widgets.get("seed")),
    )
    hold_back = int(dbutils.widgets.get("hold_back_days"))
    overwrite = dbutils.widgets.get("overwrite") == "true"
    landing, pending = P.landing, P.pending
else:
    import argparse

    ap = argparse.ArgumentParser(description="Generate synthetic card transactions")
    ap.add_argument("--out", default="./landing")
    ap.add_argument("--pending", default="./pending")
    ap.add_argument("--n-accounts", type=int, default=5000)
    ap.add_argument("--days", type=int, default=180)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hold-back-days", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    cfg = GeneratorConfig(n_accounts=args.n_accounts, days=args.days, seed=args.seed)
    hold_back, overwrite = args.hold_back_days, args.overwrite
    landing, pending = args.out, args.pending

# COMMAND ----------

tx_dir = os.path.join(landing, "transactions")
already = os.path.isdir(tx_dir) and any(f.endswith(".json") for f in os.listdir(tx_dir))

if already and not overwrite:
    print(f"Landing zone already has transaction files in {tx_dir}; skipping generation.")
    print("Set overwrite=true to regenerate (and run 00_setup_catalog with reset=true first,")
    print("otherwise Auto Loader will ignore rewritten files it has already ingested).")
else:
    print(f"Generating {cfg.n_accounts:,} accounts x {cfg.days} days (seed={cfg.seed}) ...")
    data = generate(cfg)
    summary = write_landing(data, landing, hold_back_days=hold_back, pending_dir=pending)
    for k, v in summary.items():
        print(f"  {k:>16}: {v:,}")
    tx = data["transactions"]
    print("\nFraud scenarios (ground truth, incl. unreported):")
    print(tx["fraud_scenario"].value_counts(dropna=True).to_string())
    print(f"\nLabelled fraud rate: {tx['is_fraud'].mean():.4%}")
