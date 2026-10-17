"""Data loading and cleaning for the UCI Heart Disease (Cleveland) dataset."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data" / "raw" / "processed.cleveland.data"
CLEAN_PATH = ROOT / "data" / "processed" / "heart_clean.csv"

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "target"


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Read the headerless raw file; '?' marks missing values."""
    return pd.read_csv(path, header=None, names=COLUMNS, na_values="?")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: drop duplicates, fix dtypes, binarise the target.

    Missing values are intentionally *kept* (only `ca` and `thal` have a few) and
    imputed inside the sklearn pipeline so that imputation statistics are learned
    on training folds only (no leakage).
    """
    out = df.drop_duplicates().copy()
    out[TARGET] = (out["num"] > 0).astype(int)  # 0 = no disease, 1 = disease
    out = out.drop(columns=["num"])
    for col in FEATURES:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.reset_index(drop=True)


def load_clean(path: Path = CLEAN_PATH) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/download_data.py` first."
        )
    return pd.read_csv(path)


def split_xy(df: pd.DataFrame):
    return df[FEATURES], df[TARGET].astype(int)


def validate_schema(df: pd.DataFrame) -> None:
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if df[FEATURES].dtypes.map(lambda t: not np.issubdtype(t, np.number)).any():
        raise ValueError("All features must be numeric")
