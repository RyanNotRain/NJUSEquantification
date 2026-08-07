from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from qd.adjfactor import code_mapping, full_stock_code, get_adjust_ratio
from qd.config import DATA_DIR, METRICS, TOTAL_BARS
from qd.daily import _fill_daily_prices, calc_daily_metrics
from qd.data_loader import load_one_stock
from qd.minute import build_minute_table_for_date, calc_minute_metrics
from qd.time_utils import MINUTE_BAR_LABELS, assign_minute_bar, parse_time_to_seconds


class CoreAcceptanceTests(unittest.TestCase):
    def test_workspace_paths_and_code_mapping(self) -> None:
        self.assertTrue(DATA_DIR.is_dir())
        self.assertEqual(len(code_mapping()), 300)
        self.assertEqual(full_stock_code("000012"), "000012.SZ")
        self.assertAlmostEqual(get_adjust_ratio("000012", "2025-04-01"), 1.0)

    def test_time_axis_matches_readme_example(self) -> None:
        self.assertEqual(TOTAL_BARS, 242)
        self.assertEqual(MINUTE_BAR_LABELS[0], "09:30:00")
        self.assertEqual(MINUTE_BAR_LABELS[120], "11:30:00")
        self.assertEqual(MINUTE_BAR_LABELS[121], "13:00:00")
        self.assertEqual(MINUTE_BAR_LABELS[-4:], [
            "14:57:00", "14:58:00", "14:59:00", "15:00:00"
        ])
        seconds = parse_time_to_seconds(np.array([92500740, 93000000, 150001110]))
        bars = assign_minute_bar(seconds)
        self.assertEqual(bars.tolist(), [-1, 0, 241])

    def test_daily_raw_aggregation(self) -> None:
        df = load_one_stock("20250401", "000012")
        self.assertIsNotNone(df)
        metrics = calc_daily_metrics(df, adjust_ratio=1.0)
        assert df is not None
        self.assertAlmostEqual(metrics["open"], df["Price"].iloc[0] / 100)
        self.assertAlmostEqual(metrics["close"], df["Price"].iloc[-1] / 100)
        self.assertEqual(metrics["volume"], float(df["Volume"].sum()))
        self.assertEqual(metrics["trade_count"], float(len(df)))
        self.assertEqual(
            metrics["buy_volume"], float(df.loc[df["BSFlag"] == 0, "Volume"].sum())
        )

    def test_daily_missing_ohlc_uses_previous_close(self) -> None:
        idx = ["2025-01-01", "2025-01-02"]
        raw = {
            m: pd.DataFrame(
                [[10.0], [np.nan]] if m in {"open", "high", "low", "close"}
                else [[1.0], [np.nan]],
                index=idx,
                columns=["000001.SZ"],
            )
            for m in METRICS
        }
        raw["open"].iloc[0, 0] = 9.5
        raw["high"].iloc[0, 0] = 10.5
        raw["low"].iloc[0, 0] = 9.0
        filled = _fill_daily_prices(raw)
        for metric in ("open", "high", "low", "close"):
            self.assertEqual(filled[metric].iloc[1, 0], 10.0)

    def test_minute_closing_auction_and_fill(self) -> None:
        tables = build_minute_table_for_date("20250401", all_stocks=["000012.SZ"])
        self.assertEqual(tables["close"].shape[0], 242)
        stock = "000012.SZ"
        self.assertEqual(tables["volume"][stock].iloc[-1], 9100.0)
        self.assertAlmostEqual(tables["close"][stock].iloc[-1], 61.22)
        self.assertTrue(tables["close"][stock].iloc[-4:-1].isna().all())
        self.assertTrue((tables["volume"][stock].iloc[-4:-1] == 0).all())

        zero_bars = tables["volume"][stock].iloc[:-4] == 0
        for i in np.flatnonzero(zero_bars.to_numpy()):
            if i == 0:
                continue
            expected = tables["close"][stock].iloc[i - 1]
            for metric in ("open", "high", "low", "close"):
                self.assertEqual(tables[metric][stock].iloc[i], expected)

    def test_minute_totals_exclude_only_opening_auction(self) -> None:
        df = load_one_stock("20250401", "000012")
        assert df is not None
        minute = calc_minute_metrics(df, adjust_ratio=1.0)
        opening = df[df["time_sec"] < 9 * 3600 + 30 * 60]
        self.assertEqual(
            minute["volume"].sum(),
            float(df["Volume"].sum() - opening["Volume"].sum()),
        )


if __name__ == "__main__":
    unittest.main()
