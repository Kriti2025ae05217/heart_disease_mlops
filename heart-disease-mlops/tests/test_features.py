import unittest

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.heart.data import split_xy
from src.heart.features import build_pipeline, build_preprocessor
from tests.conftest_helpers import synthetic_frame


class TestFeatures(unittest.TestCase):
    def setUp(self):
        self.X, self.y = split_xy(synthetic_frame())

    def test_preprocessor_outputs_no_nans(self):
        out = build_preprocessor().fit_transform(self.X)
        self.assertFalse(np.isnan(out.toarray() if hasattr(out, "toarray") else out).any())

    def test_numeric_columns_are_standardised(self):
        prep = build_preprocessor().fit(self.X)
        out = prep.transform(self.X)
        out = out.toarray() if hasattr(out, "toarray") else out
        self.assertAlmostEqual(float(out[:, 0].mean()), 0.0, places=6)  # age first
        self.assertAlmostEqual(float(out[:, 0].std()), 1.0, places=6)

    def test_unseen_category_does_not_crash(self):
        pipe = build_pipeline(LogisticRegression(max_iter=500)).fit(self.X, self.y)
        row = self.X.iloc[[0]].copy()
        row["cp"] = 99
        self.assertEqual(pipe.predict_proba(row).shape, (1, 2))

    def test_pipeline_handles_missing_at_inference(self):
        pipe = build_pipeline(LogisticRegression(max_iter=500)).fit(self.X, self.y)
        row = self.X.iloc[[0]].copy()
        row["ca"] = np.nan
        self.assertEqual(len(pipe.predict(row)), 1)


if __name__ == "__main__":
    unittest.main()
