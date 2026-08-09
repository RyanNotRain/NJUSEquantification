from __future__ import annotations

import unittest

import numpy as np

from qd.lstm_adjustment import _grid, select_validation_fusion


class LSTMAdjustmentTests(unittest.TestCase):
    def test_macro_f1_objective_can_prefer_two_stage(self) -> None:
        labels = np.array([0, 1, 2, 0, 1, 2])
        direction = np.array([[0.8, 0.2], [0.5, 0.5], [0.2, 0.8]] * 2)
        movement = np.array([[0.2, 0.8], [0.8, 0.2], [0.2, 0.8]] * 2)
        staged = np.array([
            [0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.1, 0.2, 0.7],
            [0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.1, 0.2, 0.7],
        ])
        # Joint collapses toward flat; a low joint weight should be selected.
        joint = np.tile([0.2, 0.6, 0.2], (6, 1))
        selection, probability = select_validation_fusion(
            labels,
            direction,
            movement,
            joint,
            move_biases=np.array([0.0]),
            joint_weights=np.array([0.0, 0.5, 1.0]),
        )
        # The hand-written staged matrix matches the probabilities implied by
        # direction/movement and documents the intended perfect prediction.
        self.assertTrue(np.array_equal(staged.argmax(axis=1), labels))
        self.assertLess(selection["joint_weight"], 1.0)
        self.assertTrue(np.array_equal(probability.argmax(axis=1), labels))

    def test_accuracy_objective_is_supported(self) -> None:
        labels = np.array([0, 1, 2])
        direction = np.array([[0.8, 0.2], [0.5, 0.5], [0.2, 0.8]])
        movement = np.array([[0.2, 0.8], [0.8, 0.2], [0.2, 0.8]])
        joint = np.eye(3) * 0.8 + 0.2 / 3
        selection, _ = select_validation_fusion(
            labels, direction, movement, joint,
            move_biases=np.array([0.0]), joint_weights=np.array([0.0, 1.0]),
            objective="accuracy_then_macro_f1",
        )
        self.assertIn(selection["joint_weight"], (0.0, 1.0))

    def test_grid_includes_both_endpoints(self) -> None:
        self.assertTrue(np.allclose(_grid(-0.3, 0.3, 0.1), np.linspace(-0.3, 0.3, 7)))
        with self.assertRaises(ValueError):
            _grid(1.0, 0.0, 0.1)

    def test_fusion_selection_rejects_invalid_component_probabilities(self) -> None:
        labels = np.array([0, 1, 2])
        direction = np.full((3, 2), 0.5)
        movement = np.full((3, 2), 0.5)
        joint = np.full((3, 3), 1.0 / 3.0)
        joint[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "joint probabilities"):
            select_validation_fusion(
                labels, direction, movement, joint,
                move_biases=np.array([0.0]), joint_weights=np.array([0.5]),
            )


if __name__ == "__main__":
    unittest.main()
