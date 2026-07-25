"""Behavioral and temporal feature engineering for Fraud_Data."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add hour_of_day, day_of_week, and time_since_signup (seconds).

    Requires signup_time and purchase_time as datetime columns.
    """
    out = df.copy()
    purchase = pd.to_datetime(out["purchase_time"])
    signup = pd.to_datetime(out["signup_time"])

    out["hour_of_day"] = purchase.dt.hour
    out["day_of_week"] = purchase.dt.dayofweek
    out["time_since_signup"] = (purchase - signup).dt.total_seconds()
    # Guard against negative / NaN
    out["time_since_signup"] = out["time_since_signup"].clip(lower=0).fillna(0)
    return out


def add_transaction_velocity(
    df: pd.DataFrame,
    user_col: str = "user_id",
    device_col: str = "device_id",
    time_col: str = "purchase_time",
) -> pd.DataFrame:
    """
    Transaction frequency / velocity features.

    - user_tx_count: total transactions per user in the dataset
    - device_tx_count: total transactions per device
    - user_tx_velocity: transactions per day for that user
      (count / span of activity in days; min span 1 day)
    """
    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col])

    out["user_tx_count"] = out.groupby(user_col)[user_col].transform("count")
    out["device_tx_count"] = out.groupby(device_col)[device_col].transform("count")

    # Velocity: user transaction count / activity span in days (min 1 day)
    span = out.groupby(user_col)[time_col].transform(
        lambda s: max((s.max() - s.min()).total_seconds() / 86400.0, 1.0)
    )
    out["user_tx_velocity"] = out["user_tx_count"] / span

    return out


def engineer_fraud_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all Fraud_Data feature engineering steps."""
    out = add_time_features(df)
    out = add_transaction_velocity(out)
    return out


def select_model_features_fraud(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separate features and target for Fraud_Data modeling.

    Drops identifiers and raw timestamps; keeps engineered + useful raw fields.
    """
    target_col = "class"
    drop_cols = [
        target_col,
        "user_id",
        "device_id",
        "signup_time",
        "purchase_time",
        "ip_address",
        "ip_int",
    ]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    return X, y


def select_model_features_creditcard(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate features and target for creditcard modeling."""
    target_col = "Class"
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()
    return X, y
