import unittest

import numpy as np
import pandas as pd

from src.heart.data import COLUMNS, FEATURES, TARGET, clean, split_xy, validate_schema


def raw_frame():
    rows = [
        [63, 1, 1, 145, 233, 1, 2, 150, 0, 2.3, 3, 0.0, 6.0, 0],
        [67, 1, 4, 160, 286, 0, 2, 108, 1, 1.5, 2, 3.0, 3.0, 2],
        [41, 0, 2, 130, 204, 0, 2, 172, 0, 1.4, 1, np.nan, 3.0, 0],
        [41, 0, 2, 130, 204, 0, 2, 172, 0, 1.4, 1, np.nan, 3.0, 0],  # duplicate
    ]
    return pd.DataFrame(rows, columns=COLUMNS)


class TestData(unittest.TestCase):
    def test_clean_binarises_target_and_drops_num(self):
        out = clean(raw_frame())
        self.assertNotIn("num", out.columns)
        self.assertEqual(sorted(out[TARGET].unique()), [0, 1])
        self.assertEqual(out.loc[1, TARGET], 1)  # num=2 -> 1

    def test_clean_removes_duplicates(self):
        self.assertEqual(len(clean(raw_frame())), 3)

    def test_clean_keeps_missing_for_pipeline_imputation(self):
        self.assertEqual(int(clean(raw_frame())["ca"].isna().sum()), 1)

    def test_split_xy_shapes(self):
        X, y = split_xy(clean(raw_frame()))
        self.assertEqual(list(X.columns), FEATURES)
        self.assertEqual(len(X), len(y))

    def test_validate_schema_flags_missing_column(self):
        with self.assertRaises(ValueError):
            validate_schema(clean(raw_frame()).drop(columns=["chol"]))


if __name__ == "__main__":
    unittest.main()
