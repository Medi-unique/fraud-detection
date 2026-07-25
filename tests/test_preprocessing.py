"""Unit tests for IP conversion and country range merge."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    add_ip_integer,
    ip_to_int,
    merge_ip_country,
)


def test_ip_to_int_dotted_quad():
    assert ip_to_int("192.168.1.1") == (192 << 24) + (168 << 16) + (1 << 8) + 1
    assert ip_to_int("0.0.0.0") == 0
    assert ip_to_int("255.255.255.255") == 4294967295


def test_ip_to_int_numeric():
    assert ip_to_int(3232235777) == 3232235777
    assert ip_to_int(3232235777.0) == 3232235777
    assert np.isnan(ip_to_int(np.nan))


def test_add_ip_integer_column():
    df = pd.DataFrame({"ip_address": ["10.0.0.1", 167772161.0]})
    out = add_ip_integer(df)
    assert "ip_int" in out.columns
    assert out.loc[0, "ip_int"] == (10 << 24) + 1
    assert out.loc[1, "ip_int"] == 167772161


def test_merge_ip_country_range_lookup():
    fraud = pd.DataFrame(
        {
            "user_id": [1, 2, 3],
            "ip_int": [100, 250, 9999],
        }
    )
    ip_map = pd.DataFrame(
        {
            "lower_bound_ip_address": [0, 200, 500],
            "upper_bound_ip_address": [150, 300, 600],
            "country": ["A", "B", "C"],
        }
    )
    merged = merge_ip_country(fraud, ip_map)
    assert list(merged["country"]) == ["A", "B", "Unknown"]


def test_merge_ip_country_edge_bounds():
    fraud = pd.DataFrame({"ip_int": [150, 200]})
    ip_map = pd.DataFrame(
        {
            "lower_bound_ip_address": [0.0, 200.0],
            "upper_bound_ip_address": [150.0, 300.0],
            "country": ["Low", "High"],
        }
    )
    merged = merge_ip_country(fraud, ip_map)
    assert merged.loc[0, "country"] == "Low"
    assert merged.loc[1, "country"] == "High"
