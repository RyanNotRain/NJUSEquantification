from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from qd.t1_excess_return_research import add_market_adjusted_target


class T1ExcessReturnResearchTests(unittest.TestCase):
    def test_market_adjusted_target_has_zero_cross_sectional_mean(self) -> None:
        frame = pd.DataFrame({
            "target_time": pd.to_datetime([
                "2026-01-02 10:00", "2026-01-02 10:00", "2026-01-02 10:00",
                "2026-01-02 10:01", "2026-01-02 10:01", "2026-01-02 10:01",
            ]),
            "t1_same_minute_open": [0.03, 0.00, -0.03, 0.02, 0.01, 0.00],
        })
        result = add_market_adjusted_target(frame)
        means = result.groupby("target_time")[
            "t1_same_minute_open__market_excess"
        ].mean()
        np.testing.assert_allclose(means.to_numpy(), 0.0, atol=1e-12)
        self.assertAlmostEqual(
            result.iloc[3]["t1_same_minute_open__market_return"], 0.01
        )


if __name__ == "__main__":
    unittest.main()

