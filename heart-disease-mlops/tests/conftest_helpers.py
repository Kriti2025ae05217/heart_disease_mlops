"""Shared helpers for tests: a small synthetic dataset with the real schema."""
import numpy as np
import pandas as pd

from src.heart.data import FEATURES, TARGET


def synthetic_frame(n: int = 160, seed: int = 0, with_missing: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age": rng.integers(30, 77, n), "sex": rng.integers(0, 2, n),
        "cp": rng.integers(1, 5, n), "trestbps": rng.integers(95, 200, n),
        "chol": rng.integers(130, 400, n), "fbs": rng.integers(0, 2, n),
        "restecg": rng.integers(0, 3, n), "thalach": rng.integers(80, 200, n),
        "exang": rng.integers(0, 2, n), "oldpeak": rng.uniform(0, 5, n).round(1),
        "slope": rng.integers(1, 4, n), "ca": rng.integers(0, 4, n).astype(float),
        "thal": rng.choice([3.0, 6.0, 7.0], n),
    })
    score = (df["age"] / 80 + df["exang"] + df["oldpeak"] / 3 + df["ca"] / 2
             - df["thalach"] / 200 + rng.normal(0, 0.4, n))
    df[TARGET] = (score > score.median()).astype(int)
    if with_missing:
        df.loc[[3, 7], "ca"] = np.nan
        df.loc[[11], "thal"] = np.nan
    return df[FEATURES + [TARGET]]
