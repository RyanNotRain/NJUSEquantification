"""Run turnover-buffer and cost-stress analysis on the strict Task 4 ledger."""

from __future__ import annotations

import json

from qd.task4_robustness import run_task4_robustness


if __name__ == "__main__":
    print(json.dumps(run_task4_robustness(), ensure_ascii=False, indent=2, default=str))

