from __future__ import annotations

import unittest

import numpy as np
import torch

from qd.lstm_magnitude import (
    MagnitudeAwareLSTM,
    magnitude_loss,
    magnitude_thresholds,
    return_metrics,
    robust_return_transform,
    transform_return_target,
)


class LSTMMagnitudeTests(unittest.TestCase):
    def test_shared_model_has_aligned_classification_and_return_heads(self) -> None:
        model = MagnitudeAwareLSTM(4, hidden_size=8, num_layers=1, dropout=0.0)
        logits, predicted_return = model(torch.zeros(5, 6, 4))
        self.assertEqual(tuple(logits.shape), (5, 3))
        self.assertEqual(tuple(predicted_return.shape), (5,))

    def test_robust_transform_caps_outlier_and_keeps_zero_mean_anchor(self) -> None:
        returns = np.array([-0.001, 0.0, 0.001, 1.0])
        transform = robust_return_transform(returns)
        target = transform_return_target(returns, transform)
        self.assertEqual(float(transform["center"]), 0.0)
        self.assertGreater(transform["scale"], 0.0)
        self.assertLessEqual(abs(float(target[-1])), transform["cap"] / transform["scale"] + 1e-6)

    def test_loss_is_finite_and_backpropagates_both_heads(self) -> None:
        logits = torch.zeros(3, 3, requires_grad=True)
        predicted = torch.zeros(3, requires_grad=True)
        labels = torch.tensor([0, 1, 2])
        returns = torch.tensor([-1.0, 0.0, 2.0])
        total, classification, regression = magnitude_loss(
            logits, predicted, labels, returns, 0.25
        )
        total.backward()
        self.assertTrue(torch.isfinite(total))
        self.assertGreater(float(classification), 0.0)
        self.assertGreater(float(regression), 0.0)
        self.assertIsNotNone(logits.grad)
        self.assertIsNotNone(predicted.grad)

    def test_thresholds_are_validation_quantiles_and_nonnegative(self) -> None:
        thresholds = magnitude_thresholds(np.array([-2.0, -1.0, 1.0, 3.0, 4.0]))
        self.assertEqual(thresholds["all"], 0.0)
        self.assertGreaterEqual(thresholds["balanced"], 0.0)
        self.assertGreaterEqual(thresholds["strict"], thresholds["balanced"])

    def test_return_metrics_reward_correct_ranking(self) -> None:
        values = np.array([-2.0, 0.0, 1.0, 3.0])
        metrics = return_metrics(values, values)
        self.assertAlmostEqual(metrics["spearman_ic"], 1.0)
        self.assertAlmostEqual(metrics["nonflat_sign_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
