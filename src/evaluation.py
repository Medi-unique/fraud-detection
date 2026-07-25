"""Evaluation metrics for imbalanced fraud detection."""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    y_true,
    y_pred,
    y_prob=None,
    name: str = "model",
) -> dict:
    """Compute AUC-PR, F1, precision, recall, ROC-AUC, and confusion matrix."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    cm = confusion_matrix(y_true, y_pred)
    result = {
        "model": name,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": cm,
        "tn": int(cm[0, 0]) if cm.shape == (2, 2) else None,
        "fp": int(cm[0, 1]) if cm.shape == (2, 2) else None,
        "fn": int(cm[1, 0]) if cm.shape == (2, 2) else None,
        "tp": int(cm[1, 1]) if cm.shape == (2, 2) else None,
    }
    if y_prob is not None:
        y_prob = np.asarray(y_prob).ravel()
        result["auc_pr"] = float(average_precision_score(y_true, y_prob))
        try:
            result["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            result["roc_auc"] = float("nan")
    else:
        result["auc_pr"] = float("nan")
        result["roc_auc"] = float("nan")
    return result


def metrics_table(metrics_list: Sequence[dict]) -> pd.DataFrame:
    """Side-by-side comparison table of model metrics."""
    rows = []
    for m in metrics_list:
        rows.append(
            {
                "model": m.get("model"),
                "AUC-PR": m.get("auc_pr"),
                "F1": m.get("f1"),
                "Precision": m.get("precision"),
                "Recall": m.get("recall"),
                "ROC-AUC": m.get("roc_auc"),
                "TP": m.get("tp"),
                "FP": m.get("fp"),
                "FN": m.get("fn"),
                "TN": m.get("tn"),
            }
        )
    return pd.DataFrame(rows)


def plot_confusion_matrix(
    cm: np.ndarray,
    title: str = "Confusion Matrix",
    ax=None,
    class_names: Sequence[str] = ("Legit", "Fraud"),
):
    """Heatmap of a 2x2 confusion matrix."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    return ax


def plot_pr_curve(y_true, y_prob, label: str = "model", ax=None):
    """Precision-Recall curve with AUC-PR in the legend."""
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    ax.plot(recall, precision, label=f"{label} (AUC-PR={ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="best")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    return ax
