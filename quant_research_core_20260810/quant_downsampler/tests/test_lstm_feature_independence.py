from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from qd.lstm_feature_independence import (
    causal_summary_views,
    component_complementarity,
    select_feature_representatives,
    training_feature_statistics,
)


class LSTMFeatureIndependenceTests(unittest.TestCase):
    def test_causal_views_have_one_row_per_sequence(self) -> None:
        values = np.arange(4 * 5 * 3, dtype=np.float32).reshape(4, 5, 3)
        views = causal_summary_views(values)
        self.assertEqual(set(views), {"last", "mean", "std", "change"})
        self.assertTrue(all(view.shape == (4, 3) for view in views.values()))
        np.testing.assert_array_equal(views["change"], values[:, -1] - values[:, 0])

    def test_training_statistics_detect_redundant_features(self) -> None:
        rng = np.random.default_rng(3)
        values = rng.normal(size=(500, 6, 3)).astype(np.float32)
        values[:, :, 1] = values[:, :, 0] * 2
        target = values[:, -1, 0]
        correlation, quality, detail = training_feature_statistics(
            values, target, ["a", "b", "c"], max_rows=500
        )
        self.assertGreater(float(correlation.loc["a", "b"]), .99)
        self.assertGreater(float(quality.loc["a", "maximum_absolute_target_rank_ic"]), .9)
        self.assertEqual(len(detail), 12)

    def test_feature_representatives_use_train_only_score(self) -> None:
        quality = pd.DataFrame(
            {"maximum_absolute_target_rank_ic": [.1, .4, .2], "best_view": ["last"] * 3,
             "signed_rank_ic_at_best_view": [.1, -.4, .2], "sample_rows": [100] * 3},
            index=["a", "b", "c"],
        )
        selected, table = select_feature_representatives([["a", "b"], ["c"]], quality)
        self.assertEqual(selected, ["b", "c"])
        self.assertEqual(int(table["selected"].sum()), 2)

    def test_component_complementarity_reports_disagreement(self) -> None:
        frame = pd.DataFrame({
            "true_label": [0, 1, 2, 2],
            "joint_prob_down": [.8, .1, .1, .6], "joint_prob_flat": [.1, .8, .1, .2],
            "joint_prob_up": [.1, .1, .8, .2],
            "staged_prob_down": [.7, .1, .6, .1], "staged_prob_flat": [.2, .7, .2, .2],
            "staged_prob_up": [.1, .2, .2, .7],
            "prob_down": [.75, .1, .3, .3], "prob_flat": [.15, .75, .15, .2],
            "prob_up": [.1, .15, .55, .5],
        })
        metrics, summary = component_complementarity(frame)
        self.assertEqual(set(metrics["probability_source"]), {"joint", "staged", "blend"})
        self.assertAlmostEqual(summary["disagreement_rate"], .5)


if __name__ == "__main__":
    unittest.main()
