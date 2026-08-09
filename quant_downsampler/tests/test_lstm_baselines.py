from __future__ import annotations

import unittest

import numpy as np

from qd.lstm_baselines import (
    evaluate_probabilities,
    make_estimator,
    summarize_sequences,
)


class LSTMBaselineTests(unittest.TestCase):
    def test_sequence_summary_uses_fixed_past_window_statistics(self) -> None:
        sequences = np.arange(2 * 10 * 2, dtype=np.float32).reshape(2, 10, 2)
        matrix, names = summarize_sequences(
            sequences,
            base_feature_names=["a", "b"],
            stock_ids=np.array([0, 1]),
            stock_codes=["S0", "S1"],
        )
        self.assertEqual(matrix.shape, (2, 7 * 2 + 2))
        self.assertEqual(len(names), matrix.shape[1])
        self.assertAlmostEqual(matrix[0, names.index("last::a")], 18.0)
        self.assertAlmostEqual(matrix[0, names.index("mean_window::a")], 9.0)
        self.assertAlmostEqual(matrix[0, names.index("mean_last5::a")], 14.0)
        self.assertAlmostEqual(matrix[0, names.index("mean_last10::a")], 9.0)
        self.assertEqual(matrix[0, names.index("stock_id::S0")], 1.0)
        self.assertEqual(matrix[0, names.index("stock_id::S1")], 0.0)

    def test_logistic_preprocessing_is_fitted_on_training_only(self) -> None:
        train = np.array([
            [-2.0, 0.0], [-1.0, 1.0], [0.0, 2.0],
            [1.0, 3.0], [2.0, 4.0], [3.0, 5.0],
        ])
        labels = np.array([0, 1, 2, 0, 1, 2])
        validation = np.full((3, 2), 10_000.0)
        estimator = make_estimator(
            "logistic_regression", {"C": 0.1, "max_iter": 50}, seed=7
        )
        estimator.fit(train, labels)
        estimator.predict_proba(validation)
        scaler = estimator.named_steps["scaler"]
        self.assertTrue(np.allclose(scaler.mean_, train.mean(axis=0)))
        self.assertFalse(np.allclose(scaler.mean_, np.vstack([train, validation]).mean(axis=0)))

    def test_multiclass_probability_metrics_include_brier_and_nll(self) -> None:
        labels = np.array([0, 1, 2, 1])
        probability = np.array([
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.2, 0.7],
            [0.1, 0.6, 0.3],
        ])
        metrics = evaluate_probabilities(labels, probability)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["macro_f1"], 1.0)
        self.assertGreater(metrics["brier_score"], 0.0)
        self.assertGreater(metrics["negative_log_likelihood"], 0.0)
        self.assertEqual(metrics["n_samples"], 4)

    def test_tree_baseline_supports_three_class_probabilities(self) -> None:
        rng = np.random.default_rng(5)
        matrix = rng.normal(size=(90, 6))
        labels = np.repeat(np.arange(3), 30)
        matrix[:, 0] += labels
        estimator = make_estimator(
            "hist_gradient_boosting",
            {
                "learning_rate": 0.1,
                "max_iter": 5,
                "max_leaf_nodes": 7,
                "min_samples_leaf": 5,
                "l2_regularization": 1.0,
            },
            seed=11,
        )
        estimator.fit(matrix, labels)
        probability = estimator.predict_proba(matrix[:8])
        self.assertEqual(probability.shape, (8, 3))
        self.assertTrue(np.allclose(probability.sum(axis=1), 1.0))


if __name__ == "__main__":
    unittest.main()
