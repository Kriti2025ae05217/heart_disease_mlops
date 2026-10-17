import tempfile
import unittest
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression

from api.app import create_app
from src.heart.data import split_xy
from src.heart.features import build_pipeline
from src.heart.predict import ValidationError, validate_payload
from tests.conftest_helpers import synthetic_frame

VALID = {"age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1,
         "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3,
         "ca": 0, "thal": 6}


class TestApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        X, y = split_xy(synthetic_frame())
        joblib.dump(build_pipeline(LogisticRegression(max_iter=500)).fit(X, y),
                    Path(cls.tmp.name) / "model.joblib")
        cls.client = create_app(cls.tmp.name).test_client()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_health(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["status"], "ok")

    def test_predict_returns_prediction_and_confidence(self):
        r = self.client.post("/predict", json=VALID)
        self.assertEqual(r.status_code, 200)
        body = r.get_json()
        self.assertIn(body["prediction"], (0, 1))
        self.assertTrue(0.5 <= body["confidence"] <= 1.0)

    def test_predict_missing_field_is_400(self):
        bad = {k: v for k, v in VALID.items() if k != "age"}
        r = self.client.post("/predict", json=bad)
        self.assertEqual(r.status_code, 400)
        self.assertIn("age", r.get_json()["error"])

    def test_predict_out_of_range_is_400(self):
        self.assertEqual(self.client.post("/predict", json={**VALID, "age": 500}).status_code, 400)

    def test_predict_non_json_is_400(self):
        self.assertEqual(self.client.post("/predict", data="nope").status_code, 400)

    def test_metrics_endpoint_counts_requests(self):
        self.client.post("/predict", json=VALID)
        text = self.client.get("/metrics").get_data(as_text=True)
        self.assertIn("http_requests_total", text)
        self.assertIn("predictions_total", text)

    def test_validate_payload_rejects_string(self):
        with self.assertRaises(ValidationError):
            validate_payload({**VALID, "chol": "abc"})


if __name__ == "__main__":
    unittest.main()
