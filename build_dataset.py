"""Build ML CSV tables from scp_aircraft JSON or solve_envs HDF5 results."""
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
        help="environment JSON for JSON solver results (not needed for solve_envs HDF5)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_data_dir / "scp_aircraft_dataset.csv",
        help="output CSV path (default: localstore/scp_aircraft_dataset.csv)",
    )
    parser.add_argument("--nodes", type=int, default=100, help="resampled trajectory points")
    parser.add_argument("--recursive", action="store_true", help="search result folders recursively")
    parser.add_argument(
        "--format",
        choices=("auto", "json", "hdf5"),
        default="auto",
        help="result format to scan (default: auto)",
    )
    args = parser.parse_args()

    if not args.results.is_dir():
        print(
            f"Results directory not found: {args.results}\n"
            "Run solvers\\scp_aircraft.py first, or pass the folder containing "
            "its result JSON files with --results."
        )
        return 1
    if args.format == "json" and not args.environment.is_file():
        print(
            f"Environment file not found: {args.environment}\n"
            "Pass the matching environment JSON with --environment. "
            "Geometry features will be unavailable without it."
        )
        return 1

    frame = SCPAircraftDatasetParser.parse_directory_to_dataframe(
        directory=args.results,
        node_count=args.nodes,
        environment_file=args.environment if args.environment.is_file() else None,
        recursive=args.recursive,
        file_format=args.format,
    )
    if frame.empty:
        print(
            f"No solver result records found in {args.results}. "
            "Check --results, --format, and --recursive."
        )
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    features, targets, metadata = SCPAircraftDatasetParser(
        node_count=args.nodes
    ).to_ml_tables(frame)
    feature_path = args.output.with_name(f"{args.output.stem}_features.csv")
    target_path = args.output.with_name(f"{args.output.stem}_targets.csv")
    metadata_path = args.output.with_name(f"{args.output.stem}_metadata.csv")
    features.to_csv(feature_path, index=False)
    targets.to_csv(target_path, index=False)
    metadata.to_csv(metadata_path, index=False)

    print(f"Saved {len(frame)} samples:")
    print(f"  full inspection table: {args.output}")
    print(f"  numeric model inputs X: {feature_path} ({features.shape[1]} columns)")
    print(f"  prediction targets y: {target_path} ({targets.shape[1]} columns)")
    print(f"  sample identifiers:    {metadata_path}")
    if "source_format" in frame and (frame["source_format"] == "hdf5").any():
        print("  HDF5 rows represent individual solver attempts.")
    if len(frame) < 100:
        print(
            "Warning: fewer than 100 solve samples; this is not enough to train "
            "or evaluate a reliable ML model."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
