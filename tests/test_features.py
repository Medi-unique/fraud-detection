"""Unit tests for temporal and velocity features."""

from __future__ import annotations

import pandas as pd
import pytest

from src.features import (
    add_time_features,
    add_transaction_velocity,
    engineer_fraud_features,
)


@pytest.fixture
def sample_fraud() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": [1, 1, 2],
            "device_id": ["d1", "d1", "d2"],
            "signup_time": pd.to_datetime(
                ["2020-01-01 10:00:00", "2020-01-01 10:00:00", "2020-01-02 08:00:00"]
            ),
            "purchase_time": pd.to_datetime(
                ["2020-01-01 12:00:00", "2020-01-03 14:00:00", "2020-01-02 09:00:00"]
            ),
            "purchase_value": [10, 20, 30],
            "class": [0, 1, 0],
        }
    )


def test_time_features(sample_fraud):
    out = add_time_features(sample_fraud)
    assert "hour_of_day" in out.columns
    assert "day_of_week" in out.columns
    assert "time_since_signup" in out.columns
    # First purchase is 2 hours after signup
    assert out.loc[0, "time_since_signup"] == 2 * 3600
    assert out.loc[0, "hour_of_day"] == 12
    assert out["time_since_signup"].min() >= 0


def test_transaction_velocity(sample_fraud):
    out = add_transaction_velocity(sample_fraud)
    assert out.loc[0, "user_tx_count"] == 2
    assert out.loc[2, "user_tx_count"] == 1
    assert out.loc[0, "device_tx_count"] == 2
    assert out["user_tx_velocity"].notna().all()
    assert (out["user_tx_velocity"] > 0).all()


def test_engineer_fraud_features(sample_fraud):
    out = engineer_fraud_features(sample_fraud)
    for col in (
        "hour_of_day",
        "day_of_week",
        "time_since_signup",
        "user_tx_count",
        "device_tx_count",
        "user_tx_velocity",
    ):
        assert col in out.columns
