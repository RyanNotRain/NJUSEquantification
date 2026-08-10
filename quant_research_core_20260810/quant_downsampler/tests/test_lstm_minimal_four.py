from __future__ import annotations

import unittest

from qd.lstm_components import ENHANCED_FEATURE_NAMES
from qd.lstm_minimal_four import PROMPT_FIELD_TO_FEATURE, PROMPT_FOUR_FEATURES


class LSTMMinimalFourTests(unittest.TestCase):
    def test_prompt_fields_map_to_four_unique_causal_features(self) -> None:
        self.assertEqual(set(PROMPT_FIELD_TO_FEATURE), {"Time", "Price", "Volume", "BSFlag"})
        self.assertEqual(len(PROMPT_FOUR_FEATURES), 4)
        self.assertEqual(len(set(PROMPT_FOUR_FEATURES)), 4)
        self.assertTrue(set(PROMPT_FOUR_FEATURES).issubset(ENHANCED_FEATURE_NAMES))


if __name__ == "__main__":
    unittest.main()
