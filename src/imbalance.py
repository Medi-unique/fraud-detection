"""Class-imbalance helpers (SMOTE / undersampling) — train set only."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from src.preprocessing import class_distribution


ResampleMethod = Literal["smote", "undersample", "none"]


def report_class_distribution(
    y: pd.Series | np.ndarray,
    stage: str = "before",
) -> pd.DataFrame:
    """Document class counts/rates before or after resampling."""
    dist = class_distribution(y)
    dist["stage"] = stage
    return dist


def resample_train(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    method: ResampleMethod = "smote",
    random_state: int = 42,
    sampling_strategy: str | float = "auto",
) -> tuple[pd.DataFrame | np.ndarray, pd.Series | np.ndarray, dict]:
    """
    Apply resampling to the **training** set only.

    Returns X_res, y_res, and a report dict with before/after distributions.

    Justification (SMOTE default):
      Fraud is rare; undersampling discards majority signal that helps
      calibrate decision boundaries. SMOTE synthesizes minority examples
      without throwing away legitimate transactions, improving recall while
      keeping AUC-PR / F1 meaningful on the untouched test set.
    """
    y_arr = np.asarray(y_train).ravel()
    before = report_class_distribution(y_arr, stage="before")

    if method == "none":
        return X_train, y_train, {"before": before, "after": before, "method": method}

    if method == "smote":
        sampler = SMOTE(random_state=random_state, sampling_strategy=sampling_strategy)
    elif method == "undersample":
        sampler = RandomUnderSampler(
            random_state=random_state, sampling_strategy=sampling_strategy
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    # SMOTE requires numeric matrix; convert DataFrame if needed
    is_df = isinstance(X_train, pd.DataFrame)
    columns = X_train.columns if is_df else None
    X_res, y_res = sampler.fit_resample(X_train, y_arr)

    if is_df:
        X_res = pd.DataFrame(X_res, columns=columns)
    y_res = pd.Series(y_res, name=getattr(y_train, "name", "class"))

    after = report_class_distribution(y_res, stage="after")
    report = {
        "before": before,
        "after": after,
        "method": method,
        "justification": (
            "SMOTE applied on training set only to avoid leakage. "
            "Preferred over undersampling to retain majority-class information "
            "under severe fraud imbalance."
            if method == "smote"
            else "Random undersampling applied on training set only."
        ),
    }
    return X_res, y_res, report
