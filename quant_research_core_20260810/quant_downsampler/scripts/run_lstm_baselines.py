"""Fit Task 5 classical baselines and probability calibration."""

from __future__ import annotations

import json

from qd.lstm_baselines import run_lstm_baselines


if __name__ == "__main__":
    print(json.dumps(run_lstm_baselines(), ensure_ascii=False, indent=2, default=str))

