"""SHAP explainability helpers for tree-based fraud models."""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import shap
except ImportError:  # pragma: no cover
    shap = None


def _require_shap():
    if shap is None:
        raise ImportError("shap is required for explainability. pip install shap")


def get_builtin_importance(
    model,
    feature_names: Sequence[str],
    top_n: int = 10,
) -> pd.DataFrame:
    """Extract built-in feature importance from an ensemble (XGBoost / RF)."""
    if hasattr(model, "feature_importances_"):
        imp = np.asarray(model.feature_importances_)
    elif hasattr(model, "named_steps") and "model" in model.named_steps:
        imp = np.asarray(model.named_steps["model"].feature_importances_)
    else:
        raise AttributeError("Model has no feature_importances_")

    names = list(feature_names)
    if len(names) != len(imp):
        names = [f"f{i}" for i in range(len(imp))]
    df = pd.DataFrame({"feature": names, "importance": imp})
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    return df.head(top_n)


def plot_builtin_importance(
    importance_df: pd.DataFrame,
    title: str = "Top Feature Importances",
    ax=None,
):
    """Horizontal bar chart of built-in feature importance."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    data = importance_df.iloc[::-1]
    ax.barh(data["feature"], data["importance"], color="steelblue")
    ax.set_xlabel("Importance")
    ax.set_title(title)
    plt.tight_layout()
    return ax


def make_explainer(model, X_background: pd.DataFrame | np.ndarray):
    """Create a SHAP TreeExplainer for an XGBoost (or tree) model."""
    _require_shap()
    estimator = model
    if hasattr(model, "named_steps") and "model" in model.named_steps:
        estimator = model.named_steps["model"]
    return shap.TreeExplainer(estimator)


def shap_values(explainer, X: pd.DataFrame | np.ndarray):
    """Compute SHAP values; returns array shaped (n_samples, n_features)."""
    _require_shap()
    values = explainer.shap_values(X)
    # Binary classifiers may return list [class0, class1]
    if isinstance(values, list):
        values = values[1]
    return values


def plot_shap_summary(
    shap_vals,
    X: pd.DataFrame | np.ndarray,
    feature_names: Sequence[str] | None = None,
    max_display: int = 15,
):
    """Global SHAP summary (beeswarm) plot."""
    _require_shap()
    if feature_names is not None and not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=list(feature_names))
    shap.summary_plot(shap_vals, X, max_display=max_display, show=True)


def plot_shap_force(
    explainer,
    shap_vals,
    X: pd.DataFrame | np.ndarray,
    index: int = 0,
    feature_names: Sequence[str] | None = None,
):
    """Force plot for a single prediction."""
    _require_shap()
    row = X.iloc[index] if isinstance(X, pd.DataFrame) else X[index]
    expected = explainer.expected_value
    if isinstance(expected, (list, np.ndarray)):
        expected = expected[1] if len(np.atleast_1d(expected)) > 1 else expected[0]
    sv = shap_vals[index]
    # Prefer matplotlib force plot for notebooks without JS
    return shap.force_plot(expected, sv, row, matplotlib=True, show=True)


def find_example_indices(
    y_true,
    y_pred,
    kind: str,
) -> np.ndarray:
    """
    Indices for TP / FP / FN examples.

    kind: 'tp' | 'fp' | 'fn'
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if kind == "tp":
        mask = (y_true == 1) & (y_pred == 1)
    elif kind == "fp":
        mask = (y_true == 0) & (y_pred == 1)
    elif kind == "fn":
        mask = (y_true == 1) & (y_pred == 0)
    else:
        raise ValueError("kind must be 'tp', 'fp', or 'fn'")
    return np.where(mask)[0]


def top_shap_drivers(
    shap_vals,
    feature_names: Sequence[str],
    top_n: int = 5,
) -> pd.DataFrame:
    """Mean |SHAP| global drivers."""
    mean_abs = np.abs(shap_vals).mean(axis=0)
    names = list(feature_names)
    if len(names) != len(mean_abs):
        names = [f"f{i}" for i in range(len(mean_abs))]
    df = pd.DataFrame({"feature": names, "mean_abs_shap": mean_abs})
    return df.sort_values("mean_abs_shap", ascending=False).head(top_n).reset_index(drop=True)
