from __future__ import annotations

import unittest

import pandas as pd

from qd.task4_strict import buffered_selection


class Task4RobustnessTests(unittest.TestCase):
    def test_buffer_limits_replacements_and_keeps_locked_names(self) -> None:
        signal = pd.Series({f"s{i}": 100.0 - i for i in range(20)})
        previous = {"s8", "s9", "s10", "s11", "s12"}
        locked = {"s12"}
        target = buffered_selection(
            signal, previous, locked, set(signal.index), top_n=5,
            buffer_n=12, max_replacements=2,
        )
        self.assertIn("s12", target)
        self.assertEqual(len(target), 5)
        self.assertLessEqual(len(previous - set(target)), 2)

    def test_buffer_rejects_invalid_capacity(self) -> None:
        with self.assertRaises(ValueError):
            buffered_selection(pd.Series({"a": 1.0}), set(), set(), {"a"}, 2, buffer_n=1)


if __name__ == "__main__":
    unittest.main()
