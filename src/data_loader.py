"""Load raw fraud-detection datasets with correct dtypes."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd

PathLike = Union[str, Path]

DEFAULT_RAW_DIR = Path("data") / "raw"


def _resolve(path: PathLike | None, filename: str, raw_dir: PathLike) -> Path:
    if path is not None:
        return Path(path)
    return Path(raw_dir) / filename


def load_fraud_data(
    path: PathLike | None = None,
    raw_dir: PathLike = DEFAULT_RAW_DIR,
) -> pd.DataFrame:
    """Load e-commerce Fraud_Data.csv with parsed timestamps."""
    file_path = _resolve(path, "Fraud_Data.csv", raw_dir)
    df = pd.read_csv(
        file_path,
        parse_dates=["signup_time", "purchase_time"],
    )
    return df


def load_ip_country(
    path: PathLike | None = None,
    raw_dir: PathLike = DEFAULT_RAW_DIR,
) -> pd.DataFrame:
    """Load IpAddress_to_Country.csv with numeric IP bounds."""
    file_path = _resolve(path, "IpAddress_to_Country.csv", raw_dir)
    df = pd.read_csv(file_path)
    for col in ("lower_bound_ip_address", "upper_bound_ip_address"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_creditcard(
    path: PathLike | None = None,
    raw_dir: PathLike = DEFAULT_RAW_DIR,
) -> pd.DataFrame:
    """Load bank creditcard.csv (PCA features + Amount + Class)."""
    file_path = _resolve(path, "creditcard.csv", raw_dir)
    df = pd.read_csv(file_path)
    return df


def load_all(raw_dir: PathLike = DEFAULT_RAW_DIR) -> dict[str, pd.DataFrame]:
    """Load all three raw datasets."""
    return {
        "fraud": load_fraud_data(raw_dir=raw_dir),
        "ip_country": load_ip_country(raw_dir=raw_dir),
        "creditcard": load_creditcard(raw_dir=raw_dir),
    }
