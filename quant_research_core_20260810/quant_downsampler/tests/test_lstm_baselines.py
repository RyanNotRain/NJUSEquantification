from __future__ import annotations

import unittest

import numpy as np

from qd.lstm_baselines import apply_temperature, probability_metrics, sequence_summary


class LSTMBaselineTests(unittest.TestCase):
    def test_sequence_summary_is_fixed_past_window(self) -> None:
        x = np.arange(2 * 3 * 2, dtype=np.float32).reshape(2, 3, 2)
        summary = sequence_summary(x, np.array([0, 1]), 2)
        self.assertEqual(summary.shape, (2, 14))
        np.testing.assert_array_equal(summary[:, :2], x[:, -1, :])

    def test_probability_metrics_and_temperature_are_normalized(self) -> None:
        labels = np.array([0, 1, 2])
        probability = np.array([[0.8, 0.1, 0.1], [0.2, 0.6, 0.2], [0.1, 0.2, 0.7]])
        metrics = probability_metrics(labels, probability)
        self.assertEqual(metrics["accuracy"], 1.0)
        transformed = apply_temperature(probability, 1.5)
        np.testing.assert_allclose(transformed.sum(axis=1), 1.0)


if __name__ == "__main__":
    unittest.main()
