"""Run broad diagnostic extensions across Tasks 1--5."""

from __future__ import annotations

import argparse
import json

from qd.comprehensive_extensions import (
    run_data_quality_extension,
    run_factor_horizon_extension,
    run_task4_chronological_extension,
    run_task5_uncertainty_extension,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run broad diagnostic extensions")
    parser.add_argument(
        "--sections", nargs="+", choices=("data", "factor", "task4", "task5"),
        default=("data", "factor", "task4", "task5"),
    )
    args = parser.parse_args()
    runners = {
        "data": run_data_quality_extension,
        "factor": run_factor_horizon_extension,
        "task4": run_task4_chronological_extension,
        "task5": run_task5_uncertainty_extension,
    }
    results = {section: runners[section]() for section in args.sections}
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
