"""Unit tests for SMOTE applied only on the training set."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from src.imbalance import report_class_distribution, resample_train


@pytest.fixture
def imbalanced_xy():
    rng = np.random.default_rng(42)
    n_maj, n_min = 200, 20
    X = pd.DataFrame(
        {
            "f1": np.concatenate(
                [rng.normal(0, 1, n_maj), rng.normal(3, 1, n_min)]
            ),
            "f2": np.concatenate(
                [rng.normal(0, 1, n_maj), rng.normal(-2, 1, n_min)]
            ),
        }
    )
    y = pd.Series([0] * n_maj + [1] * n_min, name="class")
    return X, y


def test_report_class_distribution(imbalanced_xy):
    _, y = imbalanced_xy
    dist = report_class_distribution(y, stage="before")
    assert dist.loc[0, "count"] == 200
    assert dist.loc[1, "count"] == 20
    assert dist["stage"].iloc[0] == "before"


def test_smote_balances_train_only(imbalanced_xy):
    X, y = imbalanced_xy
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
    test_len_before = len(X_test)
    test_pos_before = int((y_test == 1).sum())
    test_neg_before = int((y_test == 0).sum())

    X_res, y_res, report = resample_train(X_train, y_train, method="smote")

    # Train minority increased toward majority
    assert (y_res == 1).sum() == (y_res == 0).sum()
    assert len(X_res) > len(X_train)

    # Test set untouched
    assert len(X_test) == test_len_before
    assert int((y_test == 1).sum()) == test_pos_before
    assert int((y_test == 0).sum()) == test_neg_before

    assert report["method"] == "smote"
    assert report["before"] is not None
    assert report["after"] is not None


def test_resample_none_passthrough(imbalanced_xy):
    X, y = imbalanced_xy
    X_res, y_res, report = resample_train(X, y, method="none")
    assert len(X_res) == len(X)
    assert report["method"] == "none"
