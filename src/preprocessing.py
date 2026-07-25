"""Cleaning, IP conversion, and geolocation merge."""

from __future__ import annotations

import numpy as np
import pandas as pd


def drop_duplicates(df: pd.DataFrame, subset: list[str] | None = None) -> pd.DataFrame:
    """Remove duplicate rows; optionally keyed by subset columns."""
    return df.drop_duplicates(subset=subset).reset_index(drop=True)


def handle_missing(
    df: pd.DataFrame,
    strategy: str = "drop",
    fill_values: dict | None = None,
) -> pd.DataFrame:
    """
    Handle missing values.

    strategy:
      - 'drop': drop rows with any NA
      - 'fill': fill using fill_values (column -> value) or column medians/modes
    """
    out = df.copy()
    if strategy == "drop":
        return out.dropna().reset_index(drop=True)
    if strategy == "fill":
        fill_values = fill_values or {}
        for col in out.columns:
            if out[col].isna().any():
                if col in fill_values:
                    out[col] = out[col].fillna(fill_values[col])
                elif pd.api.types.is_numeric_dtype(out[col]):
                    out[col] = out[col].fillna(out[col].median())
                else:
                    mode = out[col].mode(dropna=True)
                    out[col] = out[col].fillna(mode.iloc[0] if len(mode) else "Unknown")
        return out.reset_index(drop=True)
    raise ValueError(f"Unknown strategy: {strategy}")


def correct_fraud_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure Fraud_Data columns have expected types."""
    out = df.copy()
    if "signup_time" in out.columns:
        out["signup_time"] = pd.to_datetime(out["signup_time"], errors="coerce")
    if "purchase_time" in out.columns:
        out["purchase_time"] = pd.to_datetime(out["purchase_time"], errors="coerce")
    if "purchase_value" in out.columns:
        out["purchase_value"] = pd.to_numeric(out["purchase_value"], errors="coerce")
    if "age" in out.columns:
        out["age"] = pd.to_numeric(out["age"], errors="coerce")
    if "class" in out.columns:
        out["class"] = out["class"].astype(int)
    if "ip_address" in out.columns:
        out["ip_address"] = pd.to_numeric(out["ip_address"], errors="coerce")
    return out


def correct_creditcard_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure creditcard columns have expected types."""
    out = df.copy()
    if "Class" in out.columns:
        out["Class"] = out["Class"].astype(int)
    if "Amount" in out.columns:
        out["Amount"] = pd.to_numeric(out["Amount"], errors="coerce")
    if "Time" in out.columns:
        out["Time"] = pd.to_numeric(out["Time"], errors="coerce")
    return out


def ip_to_int(ip) -> int | float:
    """
    Convert an IP address to a 32-bit integer.

    Accepts dotted-quad strings (e.g. '192.168.1.1') or numeric values
    already stored as floats/ints (as in Fraud_Data.csv).
    """
    if pd.isna(ip):
        return np.nan
    if isinstance(ip, (int, np.integer)):
        return int(ip)
    if isinstance(ip, float):
        if np.isnan(ip):
            return np.nan
        return int(ip)
    s = str(ip).strip()
    if s.replace(".", "", 1).isdigit() and s.count(".") == 0:
        return int(float(s))
    parts = s.split(".")
    if len(parts) != 4:
        # Already numeric string with decimal, e.g. "1.234e9"
        try:
            return int(float(s))
        except ValueError:
            return np.nan
    try:
        a, b, c, d = (int(p) for p in parts)
        return (a << 24) + (b << 16) + (c << 8) + d
    except ValueError:
        return np.nan


def add_ip_integer(df: pd.DataFrame, ip_col: str = "ip_address") -> pd.DataFrame:
    """Add `ip_int` column from `ip_col`."""
    out = df.copy()
    out["ip_int"] = out[ip_col].map(ip_to_int)
    return out


def merge_ip_country(
    fraud_df: pd.DataFrame,
    ip_country_df: pd.DataFrame,
    ip_col: str = "ip_int",
) -> pd.DataFrame:
    """
    Merge country via IP range lookup using vectorized searchsorted.

    For each IP, find the candidate range whose lower_bound is the largest
    value <= IP, then validate IP <= upper_bound.
    """
    fraud = fraud_df.copy()
    if ip_col not in fraud.columns:
        fraud = add_ip_integer(fraud)
        ip_col = "ip_int"

    ip_map = ip_country_df.copy()
    ip_map["lower_bound_ip_address"] = pd.to_numeric(
        ip_map["lower_bound_ip_address"], errors="coerce"
    )
    ip_map["upper_bound_ip_address"] = pd.to_numeric(
        ip_map["upper_bound_ip_address"], errors="coerce"
    )
    ip_map = ip_map.dropna(subset=["lower_bound_ip_address", "upper_bound_ip_address"])
    ip_map = ip_map.sort_values("lower_bound_ip_address").reset_index(drop=True)

    lowers = ip_map["lower_bound_ip_address"].to_numpy(dtype=np.float64)
    uppers = ip_map["upper_bound_ip_address"].to_numpy(dtype=np.float64)
    countries = ip_map["country"].to_numpy()

    ips = fraud[ip_col].to_numpy(dtype=np.float64)
    # searchsorted with side='right' - 1 → last lower bound <= ip
    idx = np.searchsorted(lowers, ips, side="right") - 1

    country = np.full(len(ips), "Unknown", dtype=object)
    valid = (idx >= 0) & (idx < len(lowers)) & np.isfinite(ips)
    # Check upper bound for candidates
    check = np.zeros(len(ips), dtype=bool)
    check[valid] = ips[valid] <= uppers[idx[valid]]
    country[check] = countries[idx[check]]

    fraud["country"] = country
    return fraud


def clean_fraud_data(
    fraud_df: pd.DataFrame,
    ip_country_df: pd.DataFrame | None = None,
    drop_na: bool = True,
) -> pd.DataFrame:
    """Full cleaning pipeline for Fraud_Data (+ optional country merge)."""
    df = correct_fraud_dtypes(fraud_df)
    df = drop_duplicates(df)
    if drop_na:
        df = handle_missing(df, strategy="drop")
    df = add_ip_integer(df)
    if ip_country_df is not None:
        df = merge_ip_country(df, ip_country_df)
    return df


def clean_creditcard_data(df: pd.DataFrame, drop_na: bool = True) -> pd.DataFrame:
    """Full cleaning pipeline for creditcard.csv."""
    out = correct_creditcard_dtypes(df)
    out = drop_duplicates(out)
    if drop_na:
        out = handle_missing(out, strategy="drop")
    return out


def class_distribution(y: pd.Series | np.ndarray, label: str = "class") -> pd.DataFrame:
    """Return counts and rates for binary class labels."""
    s = pd.Series(y)
    counts = s.value_counts().sort_index()
    rates = s.value_counts(normalize=True).sort_index()
    return pd.DataFrame(
        {"count": counts, "rate": rates, "label": label}
    )
