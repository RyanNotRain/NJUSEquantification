import unittest

import numpy as np
import pandas as pd

from qd.lstm_hybrid import (
    _assert_aligned,
    blend_model_probabilities,
    select_validation_blend,
)


class LSTMHybridTests(unittest.TestCase):
    def test_blend_is_normalized_and_includes_endpoints(self):
        lstm = np.array([[0.8, 0.1, 0.1], [0.1, 0.2, 0.7]])
        tree = np.array([[0.2, 0.7, 0.1], [0.6, 0.3, 0.1]])
        self.assertTrue(np.allclose(blend_model_probabilities(lstm, tree, 1.0), lstm))
        self.assertTrue(np.allclose(blend_model_probabilities(lstm, tree, 0.0), tree))
        self.assertTrue(
            np.allclose(blend_model_probabilities(lstm, tree, 0.4).sum(axis=1), 1.0)
        )

    def test_validation_selection_uses_requested_objective(self):
        labels = np.array([0, 0, 1, 1, 2, 2])
        lstm = np.array([
            [0.8, 0.1, 0.1], [0.4, 0.5, 0.1],
            [0.2, 0.7, 0.1], [0.2, 0.3, 0.5],
            [0.1, 0.2, 0.7], [0.5, 0.2, 0.3],
        ])
        tree = np.array([
            [0.4, 0.5, 0.1], [0.8, 0.1, 0.1],
            [0.2, 0.3, 0.5], [0.2, 0.7, 0.1],
            [0.5, 0.2, 0.3], [0.1, 0.2, 0.7],
        ])
        selection, probability = select_validation_blend(
            labels, lstm, tree, weights=[0.0, 0.5, 1.0]
        )
        self.assertIn(selection["lstm_weight"], {0.0, 0.5, 1.0})
        self.assertEqual(selection["objective"], "macro_f1_then_accuracy_then_nll")
        self.assertEqual(probability.shape, (6, 3))

    def test_alignment_rejects_mismatched_timestamps(self):
        metadata = pd.DataFrame({
            "stock": ["A"], "stock_id": [0], "date": ["2026-01-01"],
            "window_end": ["2026-01-01 10:00:00"],
            "target_time": ["2026-01-01 10:01:00"],
        })
        mismatched = metadata.copy()
        mismatched.loc[0, "target_time"] = "2026-01-01 10:02:00"
        with self.assertRaisesRegex(ValueError, "timestamps"):
            _assert_aligned(np.array([1]), metadata, np.array([1]), mismatched)


if __name__ == "__main__":
    unittest.main()

