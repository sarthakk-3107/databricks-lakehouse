import numpy as np
import pytest

from lakehouse.metrics import apply_threshold, best_threshold, cost_curve, precision_at_k, psi, recall_at_fpr, score_report


def test_perfect_scores():
    y = np.array([0] * 98 + [1] * 2)
    s = np.r_[np.linspace(0, 0.5, 98), [0.9, 0.95]]
    r = score_report(y, s)
    assert r["pr_auc"] == pytest.approx(1.0)
    assert r["roc_auc"] == pytest.approx(1.0)
    assert r["recall_at_fpr_1pct"] == pytest.approx(1.0)


def test_precision_at_k():
    y = np.array([1, 0, 1, 0, 0])
    s = np.array([0.9, 0.8, 0.7, 0.1, 0.0])
    assert precision_at_k(y, s, 1) == 1.0
    assert precision_at_k(y, s, 2) == 0.5
    assert precision_at_k(y, s, 3) == pytest.approx(2 / 3)


def test_recall_at_fpr_respects_budget():
    rng = np.random.default_rng(0)
    y = rng.random(5000) < 0.05
    s = y * 0.3 + rng.random(5000)
    rec, thr = recall_at_fpr(y, s, 0.01)
    fpr = ((s >= thr) & ~y).sum() / (~y).sum()
    assert fpr <= 0.0101
    assert 0 <= rec <= 1


def test_cost_curve_picks_sensible_threshold():
    y = np.array([0] * 990 + [1] * 10)
    s = np.r_[np.full(990, 0.1), np.full(10, 0.9)]
    amt = np.r_[np.full(990, 50.0), np.full(10, 500.0)]
    curve = cost_curve(y, s, amt, fp_cost=10)
    best = best_threshold(curve)
    assert best["total_cost"] == 0
    assert best["recall"] == 1.0 and best["fp"] == 0
    assert best["savings_vs_no_model"] == pytest.approx(5000)


def test_apply_threshold_counts():
    y = np.array([1, 1, 0, 0])
    s = np.array([0.9, 0.2, 0.8, 0.1])
    amt = np.array([100.0, 50.0, 10.0, 10.0])
    out = apply_threshold(y, s, amt, threshold=0.5, fp_cost=5)
    assert (out["tp"], out["fp"], out["fn"]) == (1, 1, 1)
    assert out["total_cost"] == pytest.approx(50 + 5)
    assert out["dollar_recall"] == pytest.approx(100 / 150)


def test_psi_zero_for_same_distribution_and_positive_for_shift():
    rng = np.random.default_rng(1)
    a = rng.normal(size=20000)
    assert psi(a, rng.normal(size=20000)) < 0.01
    assert psi(a, rng.normal(1.0, 1, size=20000)) > 0.25
