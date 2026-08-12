import unittest

import numpy as np
import pandas as pd

from qd.comprehensive_extensions import (
    classification_metrics,
    cluster_bootstrap_classification,
    fold_relative_metrics,
    ohlc_violation_count,
)


class ComprehensiveExtensionTests(unittest.TestCase):
    def test_ohlc_violations_are_detected(self):
        index = pd.date_range("2026-01-01", periods=2)
        open_ = pd.DataFrame({"A": [10, 10]}, index=index)
        close = pd.DataFrame({"A": [11, 9]}, index=index)
        high = pd.DataFrame({"A": [12, 9.5]}, index=index)
        low = pd.DataFrame({"A": [9, 8]}, index=index)
        self.assertEqual(ohlc_violation_count(open_, high, low, close), 1)

    def test_fold_metrics_use_compounded_geometric_excess(self):
        strategy = pd.Series([0.10, -0.05])
        benchmark = pd.Series([0.02, -0.01])
        result = fold_relative_metrics(strategy, benchmark)
        expected = ((1.10 * 0.95) / (1.02 * 0.99)) - 1
        self.assertAlmostEqual(result["geometric_excess_return"], expected)

    def test_classification_metrics_keep_all_three_classes(self):
        result = classification_metrics(np.array([0, 1, 2]), np.array([0, 1, 1]))
        self.assertAlmostEqual(result["accuracy"], 2 / 3)
        self.assertLess(result["macro_f1"], 1.0)

    def test_day_cluster_bootstrap_is_deterministic(self):
        frame = pd.DataFrame({"date": ["d1", "d1", "d2", "d2"], "true_label": [0, 1, 1, 2], "predicted_label": [0, 1, 0, 2]})
        first = cluster_bootstrap_classification(frame, iterations=10, seed=7)
        second = cluster_bootstrap_classification(frame, iterations=10, seed=7)
        pd.testing.assert_frame_equal(first, second)


if __name__ == "__main__":
    unittest.main()
