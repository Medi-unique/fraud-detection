"""Smoke tests for evaluation helpers."""

from __future__ import annotations

import numpy as np

from src.evaluation import compute_metrics, metrics_table


def test_compute_metrics_and_table():
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1, 0, 0])
    y_prob = np.array([0.1, 0.6, 0.9, 0.8, 0.2, 0.4])
    m = compute_metrics(y_true, y_pred, y_prob, name="demo")
    assert "auc_pr" in m
    assert "f1" in m
    assert m["confusion_matrix"].shape == (2, 2)
    table = metrics_table([m])
    assert list(table["model"]) == ["demo"]
    assert "AUC-PR" in table.columns
