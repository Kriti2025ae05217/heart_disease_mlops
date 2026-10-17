import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.heart import train
from src.heart.data import FEATURES
from src.heart.evaluate import classification_metrics
from tests.conftest_helpers import synthetic_frame


class TestMetrics(unittest.TestCase):
    def test_perfect_predictions(self):
        m = classification_metrics([0, 1, 1, 0], [0, 1, 1, 0], [0.1, 0.9, 0.8, 0.2])
        self.assertEqual(m["accuracy"], 1.0)
        self.assertEqual(m["roc_auc"], 1.0)


class TestTraining(unittest.TestCase):
    def test_end_to_end_training_writes_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            csv = tmp / "heart.csv"
            synthetic_frame(200).to_csv(csv, index=False)
            small = {k: (e, {kk: vv[:1] if kk != "model__max_depth" else [3]
                             for kk, vv in g.items()})
                     for k, (e, g) in train.candidate_models().items()}
            with mock.patch.object(train, "MODEL_DIR", tmp / "models"), \
                    mock.patch.object(train, "FIG_DIR", tmp / "figs"), \
                    mock.patch.object(train, "ROOT", tmp), \
                    mock.patch.object(train, "candidate_models", lambda seed=42: small):
                (tmp / "reports").mkdir()
                meta = train.run_training(csv, use_mlflow=False, outer_folds=3,
                                          inner_folds=2)
            self.assertTrue((tmp / "models" / "model.joblib").exists())
            saved = json.loads((tmp / "models" / "metadata.json").read_text())
            self.assertEqual(saved["features"], FEATURES)
            self.assertGreater(meta["metrics"]["cv_roc_auc"], 0.6)
            self.assertIn(meta["selected_model"], {"logistic_regression", "random_forest"})


if __name__ == "__main__":
    unittest.main()
