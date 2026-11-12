"""Evaluation metrics for imbalanced fraud classification.

Accuracy and ROC-AUC look great on a 0.3% fraud rate even for weak models, so
the headline metrics here are PR-AUC, recall at a fixed false-positive rate,
precision in the top-k alerts, and a dollar-cost curve used to pick the
operating threshold.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def _arr(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def recall_at_fpr(y_true, scores, max_fpr: float = 0.01) -> tuple[float, float]:
    """Highest recall achievable while keeping FPR <= max_fpr.

    Returns (recall, threshold).
    """
    fpr, tpr, thr = roc_curve(_arr(y_true), _arr(scores))
    i = int(np.searchsorted(fpr, max_fpr, side="right") - 1)
    i = max(i, 0)
    return float(tpr[i]), float(thr[i])


def precision_at_k(y_true, scores, k: int) -> float:
    """Precision among the k highest-scored transactions (an analyst's review queue)."""
    y = _arr(y_true)
    k = int(min(max(k, 1), len(y)))
    top = np.argsort(-_arr(scores), kind="stable")[:k]
    return float(y[top].mean())


def score_report(y_true, scores, prefix: str = "") -> dict[str, float]:
    """Standard metric bundle logged to MLflow for every candidate model."""
    y = _arr(y_true)
    s = _arr(scores)
    r1, _ = recall_at_fpr(y, s, 0.01)
    r01, _ = recall_at_fpr(y, s, 0.001)
    k = max(1, int(round(0.005 * len(y))))  # top 0.5% of traffic
    return {
        f"{prefix}pr_auc": float(average_precision_score(y, s)),
        f"{prefix}roc_auc": float(roc_auc_score(y, s)),
        f"{prefix}recall_at_fpr_1pct": r1,
        f"{prefix}recall_at_fpr_0p1pct": r01,
        f"{prefix}precision_at_top_0p5pct": precision_at_k(y, s, k),
        f"{prefix}fraud_rate": float(y.mean()),
        f"{prefix}n_rows": float(len(y)),
    }


def cost_curve(
    y_true,
    scores,
    amounts,
    fp_cost: float = 10.0,
    n_thresholds: int = 400,
) -> pd.DataFrame:
    """Expected business cost at each candidate threshold.

    Cost model (deliberately simple and explainable):
      * a missed fraud (FN) costs its transaction amount (charge-off loss)
      * a false alarm (FP) costs `fp_cost` dollars (customer friction, call-center
        contact, attrition risk from a declined legit purchase)
      * a caught fraud costs nothing
    """
    y = _arr(y_true).astype(bool)
    s = _arr(scores)
    amt = np.nan_to_num(_arr(amounts), nan=0.0)

    qs = np.linspace(0.0, 1.0, n_thresholds)
    thresholds = np.unique(np.quantile(s, qs))
    total_fraud_amt = float(amt[y].sum())
    n_fraud = int(y.sum())

    rows = []
    for t in thresholds:
        flag = s >= t
        tp = int((flag & y).sum())
        fp = int((flag & ~y).sum())
        caught_amt = float(amt[flag & y].sum())
        missed_amt = total_fraud_amt - caught_amt
        rows.append(
            {
                "threshold": float(t),
                "alerts": int(flag.sum()),
                "alert_rate": float(flag.mean()),
                "tp": tp,
                "fp": fp,
                "fn": n_fraud - tp,
                "precision": tp / max(tp + fp, 1),
                "recall": tp / max(n_fraud, 1),
                "fraud_amount_caught": caught_amt,
                "fraud_amount_missed": missed_amt,
                "dollar_recall": caught_amt / total_fraud_amt if total_fraud_amt else 0.0,
                "fp_cost_total": fp * fp_cost,
                "total_cost": missed_amt + fp * fp_cost,
            }
        )
    out = pd.DataFrame(rows)
    out["baseline_cost_no_model"] = total_fraud_amt
    out["savings_vs_no_model"] = total_fraud_amt - out["total_cost"]
    return out


def best_threshold(curve: pd.DataFrame) -> dict:
    """Row of the cost curve with minimum total cost."""
    return curve.loc[curve["total_cost"].idxmin()].to_dict()


def apply_threshold(y_true, scores, amounts, threshold: float, fp_cost: float = 10.0) -> dict:
    """Evaluate a *fixed* threshold (chosen on validation) on another split."""
    y = _arr(y_true).astype(bool)
    s = _arr(scores)
    amt = np.nan_to_num(_arr(amounts), nan=0.0)
    flag = s >= threshold
    tp = int((flag & y).sum())
    fp = int((flag & ~y).sum())
    fn = int((~flag & y).sum())
    total_fraud_amt = float(amt[y].sum())
    caught = float(amt[flag & y].sum())
    missed = total_fraud_amt - caught
    return {
        "threshold": float(threshold),
        "alerts": int(flag.sum()),
        "alert_rate": float(flag.mean()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "dollar_recall": caught / total_fraud_amt if total_fraud_amt else 0.0,
        "fraud_amount_caught": caught,
        "fraud_amount_missed": missed,
        "total_cost": missed + fp * fp_cost,
        "baseline_cost_no_model": total_fraud_amt,
        "savings_vs_no_model": total_fraud_amt - (missed + fp * fp_cost),
    }


def psi(expected, actual, bins: int = 10) -> float:
    """Population Stability Index between two score distributions.

    Rule of thumb: < 0.1 stable, 0.1-0.25 moderate shift, > 0.25 investigate.
    """
    e = _arr(expected)
    a = _arr(actual)
    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    e_pct = np.histogram(e, edges)[0] / len(e)
    a_pct = np.histogram(a, edges)[0] / len(a)
    eps = 1e-6
    e_pct = np.clip(e_pct, eps, None)
    a_pct = np.clip(a_pct, eps, None)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))
