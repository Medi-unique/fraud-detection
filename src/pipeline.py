"""Column transformers for scaling and encoding."""

from __future__ import annotations

from typing import Sequence

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler


def infer_column_types(
    X: pd.DataFrame,
    categorical: Sequence[str] | None = None,
    numerical: Sequence[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Infer numeric vs categorical columns if not provided."""
    if categorical is not None and numerical is not None:
        return list(numerical), list(categorical)

    cat: list[str] = []
    num: list[str] = []
    for col in X.columns:
        if categorical is not None and col in categorical:
            cat.append(col)
        elif numerical is not None and col in numerical:
            num.append(col)
        elif pd.api.types.is_numeric_dtype(X[col]):
            num.append(col)
        else:
            cat.append(col)
    return num, cat


def build_preprocessor(
    X: pd.DataFrame,
    categorical: Sequence[str] | None = None,
    numerical: Sequence[str] | None = None,
    scaler: str = "standard",
) -> ColumnTransformer:
    """
    Build a ColumnTransformer with scaler for numerics and OneHotEncoder for categoricals.

    scaler: 'standard' (StandardScaler) or 'minmax' (MinMaxScaler)
    """
    num_cols, cat_cols = infer_column_types(X, categorical=categorical, numerical=numerical)
    # Only use columns present in X (avoids errors when optional fields are absent)
    num_cols = [c for c in num_cols if c in X.columns]
    cat_cols = [c for c in cat_cols if c in X.columns]

    scale = StandardScaler() if scaler == "standard" else MinMaxScaler()
    transformers = []
    if num_cols:
        transformers.append(("num", scale, num_cols))
    if cat_cols:
        try:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:  # sklearn < 1.2
            encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
        transformers.append(("cat", encoder, cat_cols))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_model_pipeline(
    estimator,
    X: pd.DataFrame,
    categorical: Sequence[str] | None = None,
    numerical: Sequence[str] | None = None,
    scaler: str = "standard",
) -> Pipeline:
    """Wrap preprocessor + estimator in a sklearn Pipeline."""
    pre = build_preprocessor(
        X, categorical=categorical, numerical=numerical, scaler=scaler
    )
    return Pipeline([("preprocess", pre), ("model", estimator)])


# Default categorical columns for Fraud_Data after feature engineering
FRAUD_CATEGORICAL = ["source", "browser", "sex", "country"]

FRAUD_NUMERICAL = [
    "purchase_value",
    "age",
    "hour_of_day",
    "day_of_week",
    "time_since_signup",
    "user_tx_count",
    "device_tx_count",
    "user_tx_velocity",
]
