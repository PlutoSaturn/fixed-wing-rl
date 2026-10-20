"""Build a flat CSV dataset from scp_aircraft JSON results."""
from __future__ import annotations

import argparse
from pathlib import Path

from solvers.scp_aircraft_dataset import SCPAircraftDatasetParser


def main() -> int:
    project_root = Path(__file__).resolve().parent
    default_data_dir = project_root / "localstore"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=default_data_dir,
        help="directory containing solver result JSON files (default: localstore)",
    )
    parser.add_argument(
        "--environment",
        type=Path,
        default=default_data_dir / "c172_envs.json",
        help="environment JSON used by the solver (default: localstore/c172_envs.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_data_dir / "scp_aircraft_dataset.csv",
        help="output CSV path (default: localstore/scp_aircraft_dataset.csv)",
    )
    parser.add_argument("--nodes", type=int, default=100, help="resampled trajectory points")
    parser.add_argument("--recursive", action="store_true", help="search result folders recursively")
    args = parser.parse_args()

    if not args.results.is_dir():
        print(
            f"Results directory not found: {args.results}\n"
            "Run solvers\\scp_aircraft.py first, or pass the folder containing "
            "its result JSON files with --results."
        )
        return 1
    if not args.environment.is_file():
        print(
            f"Environment file not found: {args.environment}\n"
            "Pass the matching environment JSON with --environment. "
            "Geometry features will be unavailable without it."
        )
        return 1

    frame = SCPAircraftDatasetParser.parse_directory_to_dataframe(
        directory=args.results,
        node_count=args.nodes,
        environment_file=args.environment,
        recursive=args.recursive,
    )
    if frame.empty:
        print(
            f"No solver result JSON files found in {args.results}. "
            "Run the aircraft solver first or check --results."
        )
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"Saved {len(frame)} rows and {len(frame.columns)} columns to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
