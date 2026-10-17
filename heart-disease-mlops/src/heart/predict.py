"""Inference helpers used by the API and by scripts."""
import json
from pathlib import Path

import joblib
import pandas as pd

from .data import FEATURES, ROOT

# Plausible clinical ranges used for input validation
RANGES = {
    "age": (1, 120), "sex": (0, 1), "cp": (1, 4), "trestbps": (50, 300),
    "chol": (50, 700), "fbs": (0, 1), "restecg": (0, 2), "thalach": (30, 250),
    "exang": (0, 1), "oldpeak": (0, 10), "slope": (1, 3), "ca": (0, 3), "thal": (3, 7),
}


class ValidationError(ValueError):
    pass


def validate_payload(payload) -> dict:
    if not isinstance(payload, dict):
        raise ValidationError("Body must be a JSON object")
    missing = [f for f in FEATURES if f not in payload]
    if missing:
        raise ValidationError(f"Missing fields: {missing}")
    clean = {}
    for f in FEATURES:
        try:
            v = float(payload[f])
        except (TypeError, ValueError):
            raise ValidationError(f"Field '{f}' must be numeric")
        lo, hi = RANGES[f]
        if not lo <= v <= hi:
            raise ValidationError(f"Field '{f}'={v} outside allowed range [{lo}, {hi}]")
        clean[f] = v
    return clean


class Predictor:
    def __init__(self, model_dir: Path = ROOT / "models"):
        self.model = joblib.load(Path(model_dir) / "model.joblib")
        meta_path = Path(model_dir) / "metadata.json"
        self.metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    def predict(self, payload: dict) -> dict:
        row = validate_payload(payload)
        X = pd.DataFrame([row], columns=FEATURES)
        proba = float(self.model.predict_proba(X)[0, 1])
        pred = int(proba >= 0.5)
        return {
            "prediction": pred,
            "label": "heart disease likely" if pred else "no heart disease likely",
            "confidence": round(proba if pred else 1 - proba, 4),
            "probability_disease": round(proba, 4),
        }
