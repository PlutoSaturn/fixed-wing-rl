"""Create summary and trajectory plots from an SCP parser inspection CSV."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


STATUS_COLORS = {
    "feasible": "#2ca02c",
    "feasible_unconverged": "#98df8a",
    "stalled": "#ff7f0e",
    "dense_check_failed": "#d62728",
    "converged_infeasible": "#9467bd",
    "solver_error": "#8c564b",
}


def _trajectory_columns(frame: pd.DataFrame, axis: str) -> list[str]:
    pattern = re.compile(rf"^trajectory_(\d+)_{axis}$")
    return sorted(
        (column for column in frame.columns if pattern.match(column)),
        key=lambda column: int(pattern.match(column).group(1)),
    )


def create_plots(dataset_path: str | Path, output_dir: str | Path | None = None) -> list[Path]:
    """Create outcome, per-environment, and path plots from the full CSV."""
    dataset_path = Path(dataset_path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset CSV not found: {dataset_path}")
    frame = pd.read_csv(dataset_path)
    if frame.empty:
        raise ValueError(f"Dataset CSV contains no samples: {dataset_path}")
    if "status" not in frame or "env_index" not in frame:
        raise ValueError("Expected parser inspection CSV with status and env_index columns")

    output = Path(output_dir) if output_dir else dataset_path.parent / "visualizations"
    output.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    status_counts = frame["status"].fillna("unknown").value_counts()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = [STATUS_COLORS.get(status, "#7f7f7f") for status in status_counts.index]
    bars = ax.bar(status_counts.index.astype(str), status_counts.values, color=colors)
    ax.bar_label(bars)
    ax.set_title(f"Solver outcomes ({len(frame)} attempts)")
    ax.set_ylabel("Attempts")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    path = output / "solver_outcomes.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(path)

    env_status = pd.crosstab(frame["env_index"], frame["status"]).sort_index()
    fig, ax = plt.subplots(figsize=(9, 5))
    bottom = np.zeros(len(env_status), dtype=float)
    for status in env_status.columns:
        values = env_status[status].to_numpy()
        ax.bar(
            env_status.index.astype(str),
            values,
            bottom=bottom,
            label=status,
            color=STATUS_COLORS.get(status, "#7f7f7f"),
        )
        bottom += values
    ax.set_title("Solver outcomes by environment")
    ax.set_xlabel("Environment index")
    ax.set_ylabel("Attempts")
    ax.legend(title="Status", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    path = output / "outcomes_by_environment.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(path)

    x_columns, y_columns = _trajectory_columns(frame, "x"), _trajectory_columns(frame, "y")
    if x_columns and y_columns:
        node_count = min(len(x_columns), len(y_columns))
        x_columns, y_columns = x_columns[:node_count], y_columns[:node_count]
        environments = sorted(frame["env_index"].dropna().unique())
        ncols = min(3, max(1, len(environments)))
        nrows = int(np.ceil(len(environments) / ncols))
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False
        )
        for ax, env_index in zip(axes.flat, environments):
            samples = frame[frame["env_index"] == env_index]
            for _, sample in samples.iterrows():
                x = pd.to_numeric(sample[x_columns], errors="coerce").to_numpy(dtype=float)
                y = pd.to_numeric(sample[y_columns], errors="coerce").to_numpy(dtype=float)
                valid = np.isfinite(x) & np.isfinite(y)
                if valid.sum() < 2:
                    continue
                status = str(sample.get("status", "unknown"))
                ax.plot(
                    x[valid], y[valid],
                    color=STATUS_COLORS.get(status, "#7f7f7f"),
                    alpha=0.8 if status.startswith("feasible") else 0.3,
                    linewidth=2 if status.startswith("feasible") else 1,
                    label=status,
                )
            if {"start_pos_x", "start_pos_y"}.issubset(samples.columns):
                ax.scatter(
                    samples["start_pos_x"].iloc[0], samples["start_pos_y"].iloc[0],
                    marker="o", color="black", s=32, label="start",
                )
                ax.scatter(
                    samples["goal_pos_x"].iloc[0], samples["goal_pos_y"].iloc[0],
                    marker="*", color="black", s=64, label="goal",
                )
            ax.set_title(f"Environment {env_index}")
            ax.set_xlabel("X position")
            ax.set_ylabel("Y position")
            ax.axis("equal")
            ax.grid(alpha=0.2)
        for ax in axes.flat[len(environments):]:
            ax.set_visible(False)
        statuses = sorted(frame["status"].dropna().astype(str).unique())
        legend_handles = [
            Line2D(
                [0], [0], color=STATUS_COLORS.get(status, "#7f7f7f"),
                linewidth=2, label=status,
            )
            for status in statuses
        ]
        legend_handles.extend([
            Line2D([0], [0], marker="o", color="black", linestyle="", label="start"),
            Line2D([0], [0], marker="*", color="black", linestyle="", label="goal"),
        ])
        fig.suptitle("Solver attempt paths (XY projection)", y=0.99)
        fig.legend(
            handles=legend_handles, loc="upper center",
            bbox_to_anchor=(0.5, 0.96), ncol=min(5, len(legend_handles)),
        )
        fig.tight_layout(rect=(0, 0, 1, 0.90))
        path = output / "attempt_trajectories_xy.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        written.append(path)

    summary = (
        f"Samples: {len(frame)}\n"
        f"Environments: {frame['env_index'].nunique()}\n"
        f"Successful: {int(frame['success'].fillna(False).astype(bool).sum()) if 'success' in frame else 'n/a'}\n"
        f"Statuses: {status_counts.to_dict()}\n"
    )
    (output / "summary.txt").write_text(summary, encoding="utf-8")
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("localstore/aircraft_shooting_attempts.csv"),
        help="full inspection CSV written by build_dataset.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="plot directory (default: <dataset folder>/visualizations)",
    )
    args = parser.parse_args()
    for path in create_plots(args.dataset, args.output_dir):
        print(f"Saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
