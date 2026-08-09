from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from qd.lstm_strategy import (
    _prepare_output_directory,
    attach_realised_returns,
    break_even_cost_bps,
    build_portfolio_path,
    generate_target_weights,
)


def _predictions() -> pd.DataFrame:
    return pd.DataFrame({
        "stock": ["A", "B", "A", "B"],
        "window_end": [
            "2025-01-02 10:00:00",
            "2025-01-02 10:00:00",
            "2025-01-02 10:01:00",
            "2025-01-02 10:01:00",
        ],
        "target_time": [
            "2025-01-02 10:01:00",
            "2025-01-02 10:01:00",
            "2025-01-02 10:02:00",
            "2025-01-02 10:02:00",
        ],
        "prob_down": [0.10, 0.60, 0.20, 0.30],
        "prob_flat": [0.20, 0.25, 0.60, 0.40],
        "prob_up": [0.70, 0.15, 0.20, 0.30],
        "selected_balanced": [True, True, True, False],
        "selected_strict": [True, False, False, False],
        "true_label": [2, 0, 1, 1],
        "realised_return": [0.01, -0.01, 0.0, 0.0],
    })


class LSTMStrategyTests(unittest.TestCase):
    def test_output_directory_requires_explicit_overwrite(self) -> None:
        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "result"
            (target / "positions").mkdir(parents=True)
            stale = target / "positions" / "stale.csv"
            stale.write_text("old\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                _prepare_output_directory(target, overwrite=False)
            _prepare_output_directory(target, overwrite=True)
            self.assertFalse(stale.exists())

    def test_weights_do_not_depend_on_labels_or_realised_returns(self) -> None:
        original = _predictions()
        mutated = original.copy()
        mutated["true_label"] = [0, 2, 2, 0]
        mutated["realised_return"] = [0.75, -0.50, 0.25, -0.10]
        first = generate_target_weights(
            original, tier="balanced", weighting="confidence"
        )
        second = generate_target_weights(
            mutated, tier="balanced", weighting="confidence"
        )
        np.testing.assert_allclose(first["target_weight"], second["target_weight"])
        np.testing.assert_array_equal(first["active_signal"], second["active_signal"])

    def test_flat_argmax_is_no_trade_by_default(self) -> None:
        weights = generate_target_weights(_predictions(), weighting="equal")
        second_time = weights[weights["window_end"].eq("2025-01-02 10:01:00")]
        self.assertTrue((second_time["target_weight"] == 0.0).all())
        first_time = weights[weights["window_end"].eq("2025-01-02 10:00:00")]
        self.assertEqual(first_time.set_index("stock").at["A", "target_weight"], 0.5)
        self.assertEqual(first_time.set_index("stock").at["B", "target_weight"], -0.5)

    def test_confidence_weights_use_directional_probability_gap(self) -> None:
        weights = generate_target_weights(_predictions(), weighting="confidence")
        first = weights[weights["window_end"].eq("2025-01-02 10:00:00")].set_index("stock")
        # Gaps are +0.60 and -0.45, normalised by total absolute gap 1.05.
        self.assertAlmostEqual(first.at["A", "target_weight"], 0.60 / 1.05)
        self.assertAlmostEqual(first.at["B", "target_weight"], -0.45 / 1.05)

    def test_selected_string_false_is_not_treated_as_true(self) -> None:
        data = _predictions()
        data["selected_balanced"] = ["True", "False", "False", "False"]
        weights = generate_target_weights(data, tier="balanced", weighting="equal")
        self.assertEqual(int(weights["active_signal"].sum()), 1)

    def test_attach_realised_returns_uses_exact_timestamps(self) -> None:
        data = _predictions().iloc[:2].drop(columns="realised_return")
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            table = pd.DataFrame(
                {"A": [100.0, 101.0], "B": [50.0, 49.0]},
                index=pd.to_datetime(["2025-01-02 10:00", "2025-01-02 10:01"]),
            )
            table.index.name = "datetime"
            table.to_csv(root / "20250102.csv")
            attached = attach_realised_returns(data, root)
        result = attached.set_index("stock")["realised_return"]
        self.assertAlmostEqual(result["A"], 0.01)
        self.assertAlmostEqual(result["B"], -0.02)

    def test_turnover_closes_positions_at_data_gaps(self) -> None:
        base = pd.DataFrame({
            "stock": ["A", "A"],
            "window_end": ["2025-01-02 10:00:00", "2025-01-02 13:59:00"],
            "target_time": ["2025-01-02 10:01:00", "2025-01-02 14:00:00"],
            "target_weight": [1.0, 1.0],
            "realised_return": [0.01, 0.01],
        })
        path = build_portfolio_path(base)
        # Entry + exit for each of the two disconnected one-minute holdings.
        self.assertEqual(path["turnover"].sum(), 4.0)

    def test_returns_are_cross_sectionally_aggregated_before_compounding(self) -> None:
        same_minute = pd.DataFrame({
            "stock": ["A", "B"],
            "window_end": ["2025-01-02 10:00:00"] * 2,
            "target_time": ["2025-01-02 10:01:00"] * 2,
            "target_weight": [0.5, 0.5],
            "realised_return": [0.10, -0.10],
        })
        path = build_portfolio_path(same_minute)
        # Cross-sectional return is 0.5*10% + 0.5*(-10%) = 0%, rather
        # than treating stock rows as periods and obtaining 1.1*0.9-1=-1%.
        self.assertEqual(len(path), 1)
        self.assertAlmostEqual(path.at[0, "gross_return"], 0.0)

    def test_break_even_cost_uses_compounded_net_return(self) -> None:
        path = pd.DataFrame({
            "gross_return": [0.01],
            "turnover": [2.0],
        })
        self.assertAlmostEqual(break_even_cost_bps(path), 50.0, places=8)


if __name__ == "__main__":
    unittest.main()
