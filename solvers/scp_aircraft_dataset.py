"""Build flat supervised-learning rows from SCP aircraft result JSON files.

The current ``scp_aircraft.py`` output does not persist the initial guess or
raw solver log. Those features are populated only when a result/log explicitly
contains them or a matching ``.log`` file is available.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


class SCPAircraftDatasetParser:
    """Parse SCP aircraft results into one flattened row per solve result.

    Parameters
    ----------
    node_count:
        Number of uniformly time-spaced XYZ coordinates emitted for each
        successful trajectory.
    environment_file:
        Optional environment JSON file (the list written by ``envgen.py``).
        If omitted, the parser tries the result file's directory and the
        repository's ``localstore`` directory using the result's ``env_file``.
    environment_dir:
        Optional directory to search for environment files named in results.
    solver_constraints:
        Optional solver-configuration values to use when they are absent from
        older result JSON. New solver output records its runtime configuration.
    """

    FAILURE_PATTERNS = (
        ("infeasible", re.compile(r"\binfeasib(?:le|ility)\b", re.IGNORECASE)),
        (
            "max_iterations",
            re.compile(
                r"\b(?:maximum|max(?:imum)?)[ _-]*iterations?\b|"
                r"\biteration limit\b|\bmax_iterations\b",
                re.IGNORECASE,
            ),
        ),
        (
            "math_error",
            re.compile(
                r"\b(?:math(?:ematical)? error|nonfinite|non-finite|nan|inf(?:inity)?|"
                r"division by zero|overflow|underflow)\b",
                re.IGNORECASE,
            ),
        ),
        (
            "solver_error",
            re.compile(r"\bsolver error\b|\bsolver_error\b|\bexception\b", re.IGNORECASE),
        ),
    )
    CONSTRAINT_PATTERNS = (
        ("max_thrust", re.compile(r"\b(?:max(?:imum)? thrust|thrust limit|thrust bound)\b", re.I)),
        ("altitude_boundary", re.compile(r"\b(?:altitude|z[- ]?(?:bound|boundary|limit))\b", re.I)),
        ("workspace_boundary", re.compile(r"\b(?:workspace|position bounds?|box bounds?)\b", re.I)),
        ("speed_bound", re.compile(r"\b(?:speed|airspeed|velocity) bounds?\b", re.I)),
        ("stall", re.compile(r"\bstall(?: constraint| limit| bound)?\b", re.I)),
        ("load_factor", re.compile(r"\b(?:load factor|n_max)\b", re.I)),
        ("bank_angle", re.compile(r"\b(?:bank(?: angle)?|bank_max)\b", re.I)),
        ("obstacle", re.compile(r"\bobstacle(?: clearance| constraint| collision)?\b", re.I)),
        ("flight_path_angle", re.compile(r"\b(?:flight[- ]path angle|gamma(?:_max)?|climb angle)\b", re.I)),
    )
    FAILURE_CONTEXT = re.compile(
        r"\b(?:violat(?:e|ed|ion|ions)|infeasib(?:le|ility)|fail(?:ed|ure)?|"
        r"active|binding|residual|exceed(?:ed|s)?)\b",
        re.IGNORECASE,
    )
    ITERATION_PATTERNS = (
        re.compile(r"\biter(?:ation)?\s*(?:#|:|=)?\s*(\d+)\b", re.IGNORECASE),
        re.compile(r"\bit\s*(?:#|:|=)\s*(\d+)\b", re.IGNORECASE),
        re.compile(r"\bit\s+(\d+)\b", re.IGNORECASE),
    )
    STOP_EVENTS = {
        "converged",
        "stalled",
        "subproblem_failed",
        "rejected_nonfinite",
        "restored_best_valid",
    }
    OBSTACLE_KINDS = ("spheroid", "oriented_cylinder", "cylinder", "box", "wall")
    OCCUPANCY_RESOLUTION = (8, 8, 4)
    STATE_NAMES = ("x", "y", "z", "speed", "gamma", "heading")
    CONTROL_NAMES = ("thrust", "n_v", "n_h")
    BASE_FEATURE_COLUMNS = (
        "start_pos_x", "start_pos_y", "start_pos_z",
        "goal_pos_x", "goal_pos_y", "goal_pos_z",
        "start_vel_x", "start_vel_y", "start_vel_z",
        "goal_vel_x", "goal_vel_y", "goal_vel_z",
        "bounds_lo_x", "bounds_lo_y", "bounds_lo_z",
        "bounds_hi_x", "bounds_hi_y", "bounds_hi_z",
        "position_lo_x", "position_lo_y", "position_lo_z",
        "position_hi_x", "position_hi_y", "position_hi_z",
        "vehicle_radius", "margin", "clearance", "obstacle_count",
        "route_distance_m", "initial_guess_available", "solver_start_speed",
        "solver_start_gamma_rad", "solver_start_heading_rad",
        "solver_start_x", "solver_start_y", "solver_start_z",
        "solver_end_x", "solver_end_y", "solver_end_z",
        "solver_end_speed_is_free", "solver_end_heading_is_free",
        "solver_end_gamma_is_fixed",
    )
    TARGET_COLUMNS = (
        "success", "status", "failure_category", "failure_constraint",
        "stop_iteration", "stop_event", "iterations", "solve_time",
        "flight_time", "cost", "max_defect", "max_violation", "dense_min_margin",
    )
    METADATA_COLUMNS = (
        "source_file", "env_index", "env_file", "env_fingerprint", "solver",
        "best_start", "attempt_count", "initial_guess_source", "raw_log_available",
    )

    def __init__(
        self,
        node_count: int = 100,
        environment_file: str | Path | None = None,
        environment_dir: str | Path | None = None,
        solver_constraints: dict[str, Any] | None = None,
    ) -> None:
        if node_count < 2:
            raise ValueError("node_count must be at least 2")
        self.node_count = node_count
        self.environment_file = Path(environment_file) if environment_file else None
        self.environment_dir = Path(environment_dir) if environment_dir else None
        self.solver_constraints = solver_constraints or {}
        self._environment_cache: dict[Path, list[dict[str, Any]]] = {}

    @staticmethod
    def _records(payload: Any, source: Path) -> list[dict[str, Any]]:
        if isinstance(payload, dict):
            records = payload.get("results", [payload])
        else:
            records = payload
        if not isinstance(records, list) or not all(isinstance(row, dict) for row in records):
            raise ValueError(f"{source} must contain a JSON object or a list of objects")
        return [row for row in records if "x" in row or "solver" in row or "status" in row]

    @staticmethod
    def _safe_number(value: Any) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if np.isfinite(number) else None

    @staticmethod
    def _json_value(value: Any) -> str | None:
        return json.dumps(value, sort_keys=True) if value is not None else None

    @staticmethod
    def _read_json(path: Path) -> Any:
        with path.open(encoding="utf-8") as stream:
            return json.load(stream)

    def _environment_candidates(self, result_file: Path, result: dict[str, Any]) -> Iterable[Path]:
        if self.environment_file is not None:
            yield self.environment_file
            return

        env_name = result.get("env_file")
        if not env_name:
            return
        env_path = Path(env_name)
        candidates = []
        if env_path.is_absolute():
            candidates.append(env_path)
        else:
            candidates.extend((result_file.parent / env_path, Path.cwd() / env_path))
            if self.environment_dir is not None:
                candidates.append(self.environment_dir / env_path.name)
            candidates.append(result_file.parent / "localstore" / env_path.name)
            candidates.append(Path(__file__).resolve().parents[1] / "localstore" / env_path.name)
        seen: set[Path] = set()
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved not in seen:
                seen.add(resolved)
                yield resolved

    def _get_environment(
        self, result_file: Path, result: dict[str, Any]
    ) -> dict[str, Any] | None:
        env_index = result.get("env_index")
        if not isinstance(env_index, int) or env_index < 0:
            return None
        for candidate in self._environment_candidates(result_file, result):
            if not candidate.is_file():
                continue
            if candidate not in self._environment_cache:
                payload = self._read_json(candidate)
                if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
                    raise ValueError(f"Environment file {candidate} must contain a list of objects")
                self._environment_cache[candidate] = payload
            environments = self._environment_cache[candidate]
            if env_index >= len(environments):
                continue
            env = environments[env_index]
            expected = result.get("env_fingerprint")
            if expected and self._fingerprint(env) != expected:
                continue
            return env
        return None

    @staticmethod
    def _fingerprint(environment: dict[str, Any]) -> str | None:
        # Match Environment.fingerprint(), which hashes the environment without metrics.
        import hashlib

        value = dict(environment)
        value.pop("metrics", None)
        return hashlib.sha1(json.dumps(value, sort_keys=True).encode()).hexdigest()[:12]

    @staticmethod
    def _as_array(value: Any, columns: int | None = None) -> np.ndarray | None:
        try:
            array = np.asarray(value, dtype=float)
        except (TypeError, ValueError):
            return None
        if array.ndim != 2 or (columns is not None and array.shape[1] < columns):
            return None
        if not np.isfinite(array).all():
            return None
        return array

    @classmethod
    def _guess_array(cls, result: dict[str, Any]) -> np.ndarray | None:
        candidates = (
            result.get("guess_x"),
            result.get("initial_guess_x"),
            result.get("x_guess"),
            (result.get("initial_guess") or {}).get("x")
            if isinstance(result.get("initial_guess"), dict)
            else None,
        )
        for value in candidates:
            array = cls._as_array(value, columns=3)
            if array is not None:
                return array
        return None

    def _guess_differences(
        self, guess: np.ndarray | None, final: np.ndarray | None
    ) -> dict[str, float | None]:
        fields = {
            "guess_final_position_mean_error_m": None,
            "guess_final_position_rms_error_m": None,
            "guess_final_position_max_error_m": None,
            "guess_final_state_mean_abs_error": None,
        }
        if guess is None or final is None:
            return fields
        count = min(len(guess), len(final))
        if count < 2:
            return fields
        source = np.linspace(0.0, 1.0, len(guess))
        target = np.linspace(0.0, 1.0, len(final))
        aligned_guess = np.column_stack(
            [np.interp(target, source, guess[:, column]) for column in range(guess.shape[1])]
        )
        compared_columns = min(aligned_guess.shape[1], final.shape[1])
        state_error = np.abs(aligned_guess[:, :compared_columns] - final[:, :compared_columns])
        position_columns = min(3, compared_columns)
        position_error = np.linalg.norm(
            aligned_guess[:, :position_columns] - final[:, :position_columns], axis=1
        )
        fields.update(
            guess_final_position_mean_error_m=float(position_error.mean()),
            guess_final_position_rms_error_m=float(np.sqrt(np.mean(position_error**2))),
            guess_final_position_max_error_m=float(position_error.max()),
            guess_final_state_mean_abs_error=float(state_error.mean()),
        )
        return fields

    def _resample_xyz(self, trajectory: np.ndarray | None, flight_time: Any) -> dict[str, float | None]:
        values: dict[str, float | None] = {}
        for index in range(self.node_count):
            for axis in "xyz":
                values[f"trajectory_{index:03d}_{axis}"] = None
        if trajectory is None or trajectory.shape[1] < 3:
            return values
        tf = self._safe_number(flight_time)
        if tf is None or tf <= 0:
            return values
        old_time = np.linspace(0.0, tf, len(trajectory))
        new_time = np.linspace(0.0, tf, self.node_count)
        for axis, column in zip("xyz", range(3)):
            coordinates = np.interp(new_time, old_time, trajectory[:, column])
            for index, value in enumerate(coordinates):
                values[f"trajectory_{index:03d}_{axis}"] = float(value)
        return values

    @staticmethod
    def _resample_columns(
        array: np.ndarray | None, node_count: int, names: tuple[str, ...], prefix: str
    ) -> dict[str, float | None]:
        fields = {
            f"{prefix}_{index:03d}_{name}": None
            for index in range(node_count)
            for name in names
        }
        if array is None or array.shape[1] < len(names):
            return fields
        old_progress = np.linspace(0.0, 1.0, len(array))
        new_progress = np.linspace(0.0, 1.0, node_count)
        for column, name in enumerate(names):
            samples = np.interp(new_progress, old_progress, array[:, column])
            for index, value in enumerate(samples):
                fields[f"{prefix}_{index:03d}_{name}"] = float(value)
        return fields

    def _load_log(self, result_file: Path, result: dict[str, Any]) -> str | None:
        for key in ("raw_log", "log_text", "solver_log"):
            if isinstance(result.get(key), str):
                return result[key]
        log_path = result.get("log_file")
        candidates = []
        if log_path:
            path = Path(log_path)
            candidates.append(path if path.is_absolute() else result_file.parent / path)
        candidates.append(result_file.with_suffix(".log"))
        for candidate in candidates:
            if candidate.is_file():
                return candidate.read_text(encoding="utf-8", errors="replace")
        return None

    @classmethod
    def _failure_category(cls, status: Any, success: Any, raw_log: str | None) -> str:
        if success is True or str(status).startswith("feasible"):
            return "success"
        text = raw_log or ""
        for category, pattern in cls.FAILURE_PATTERNS:
            if pattern.search(text):
                return category
        status_text = str(status or "").lower()
        if "infeasible" in status_text:
            return "infeasible"
        if "iteration" in status_text or "max_iterations" in status_text:
            return "max_iterations"
        if "nonfinite" in status_text or "non-finite" in status_text or "math" in status_text:
            return "math_error"
        if "solver_error" in status_text or "solver error" in status_text:
            return "solver_error"
        if "stalled" in status_text or "trust_region_collapsed" in status_text:
            return "optimization_stalled"
        if "dense_check_failed" in status_text:
            return "constraint_failure"
        return "other_failure"

    @classmethod
    def _constraint_cause(cls, raw_log: str | None) -> str | None:
        if not raw_log:
            return None
        for line in raw_log.splitlines():
            if not cls.FAILURE_CONTEXT.search(line):
                continue
            for label, pattern in cls.CONSTRAINT_PATTERNS:
                if pattern.search(line):
                    return label
        return None

    @classmethod
    def _log_iteration(cls, raw_log: str | None) -> int | None:
        if not raw_log:
            return None
        matches: list[tuple[int, int]] = []
        for line_number, line in enumerate(raw_log.splitlines()):
            for pattern in cls.ITERATION_PATTERNS:
                match = pattern.search(line)
                if match:
                    matches.append((line_number, int(match.group(1))))
                    break
        return matches[-1][1] if matches else None

    @classmethod
    def _stop_iteration(cls, result: dict[str, Any], raw_log: str | None) -> tuple[int | None, str | None]:
        history = result.get("history")
        if isinstance(history, list):
            for record in reversed(history):
                if not isinstance(record, dict):
                    continue
                event = record.get("event")
                if event in cls.STOP_EVENTS and isinstance(record.get("iter"), int):
                    return record["iter"], event
            for record in reversed(history):
                if isinstance(record, dict) and isinstance(record.get("iter"), int):
                    return record["iter"], record.get("event")
        iteration = cls._log_iteration(raw_log)
        if iteration is None:
            iteration = result.get("iterations")
        return (iteration if isinstance(iteration, int) else None), None

    def _environment_features(self, environment: dict[str, Any] | None) -> dict[str, Any]:
        names = (
            "start_pos_x", "start_pos_y", "start_pos_z",
            "goal_pos_x", "goal_pos_y", "goal_pos_z",
            "start_vel_x", "start_vel_y", "start_vel_z",
            "goal_vel_x", "goal_vel_y", "goal_vel_z",
            "bounds_lo_x", "bounds_lo_y", "bounds_lo_z",
            "bounds_hi_x", "bounds_hi_y", "bounds_hi_z",
            "position_lo_x", "position_lo_y", "position_lo_z",
            "position_hi_x", "position_hi_y", "position_hi_z",
            "vehicle_radius", "margin", "clearance",
            "obstacle_count",
            "obstacles_json", "environment_metrics_json",
        )
        result: dict[str, Any] = {name: None for name in names}
        if environment is None:
            return result
        for key, prefix in (
            ("start_pos", "start_pos"),
            ("goal_pos", "goal_pos"),
            ("start_vel", "start_vel"),
            ("goal_vel", "goal_vel"),
            ("bounds_lo", "bounds_lo"),
            ("bounds_hi", "bounds_hi"),
        ):
            values = environment.get(key)
            if isinstance(values, (list, tuple)) and len(values) >= 3:
                for axis, value in zip("xyz", values[:3]):
                    result[f"{prefix}_{axis}"] = self._safe_number(value)
        radius = self._safe_number(environment.get("vehicle_radius"))
        margin = self._safe_number(environment.get("margin"))
        result["vehicle_radius"] = radius
        result["margin"] = margin
        if radius is not None and margin is not None:
            result["clearance"] = radius + margin
        for bounds_key, prefix in (("bounds_lo", "position_lo"), ("bounds_hi", "position_hi")):
            values = environment.get(bounds_key)
            if isinstance(values, (list, tuple)) and len(values) >= 3 and radius is not None:
                for axis, value in zip("xyz", values[:3]):
                    bound = self._safe_number(value)
                    if bound is not None:
                        result[f"{prefix}_{axis}"] = bound + radius if prefix == "position_lo" else bound - radius
        obstacles = environment.get("obstacles", [])
        result["obstacle_count"] = len(obstacles) if isinstance(obstacles, list) else None
        result["obstacles_json"] = self._json_value(obstacles)
        result["environment_metrics_json"] = self._json_value(environment.get("metrics"))
        for kind in self.OBSTACLE_KINDS:
            result[f"obstacle_count_{kind}"] = 0
        for obstacle in obstacles if isinstance(obstacles, list) else []:
            if not isinstance(obstacle, dict):
                continue
            kind = obstacle.get("type", obstacle.get("kind"))
            if kind:
                key = f"obstacle_count_{kind}"
                result[key] = result.get(key, 0) + 1
        try:
            from environment.envgen import Environment

            occupancy = Environment.from_dict(environment).occupancy_grid(
                self.OCCUPANCY_RESOLUTION
            ).reshape(-1)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Could not create occupancy features from environment: {exc}") from exc
        for index, occupied in enumerate(occupancy):
            result[f"occupancy_{index:03d}"] = int(occupied)
        metrics = environment.get("metrics")
        if isinstance(metrics, dict):
            for key, value in metrics.items():
                number = self._safe_number(value)
                if number is not None:
                    result[f"environment_metric_{key}"] = number
        return result

    def parse_result(self, result: dict[str, Any], result_file: str | Path = "result.json") -> dict[str, Any]:
        result_path = Path(result_file).resolve()
        final = self._as_array(result.get("x"), columns=3)
        controls = self._as_array(result.get("u"), columns=1)
        guess = self._guess_array(result)
        raw_log = self._load_log(result_path, result)
        stop_iteration, stop_event = self._stop_iteration(result, raw_log)
        success = bool(result.get("success", str(result.get("status", "")).startswith("feasible")))

        row: dict[str, Any] = {
            "source_file": str(result_path),
            "env_index": result.get("env_index"),
            "env_file": result.get("env_file"),
            "env_fingerprint": result.get("env_fingerprint"),
            "solver": result.get("solver"),
            "success": success,
            "status": result.get("status"),
            "failure_category": self._failure_category(result.get("status"), success, raw_log),
            "stop_iteration": stop_iteration,
            "stop_event": stop_event,
            "log_iteration": self._log_iteration(raw_log),
            "failure_constraint": self._constraint_cause(raw_log),
            "raw_log_available": raw_log is not None,
            "initial_guess_available": guess is not None,
            "initial_guess_source": (
                next(
                    (key for key in ("guess_x", "initial_guess_x", "x_guess")
                     if result.get(key) is not None),
                    "initial_guess.x" if isinstance(result.get("initial_guess"), dict)
                    and result["initial_guess"].get("x") is not None else None,
                )
            ),
            "iterations": result.get("iterations"),
            "solve_time": result.get("solve_time"),
            "flight_time": result.get("flight_time"),
            "cost": result.get("cost"),
            "max_defect": result.get("max_defect"),
            "max_violation": result.get("max_violation"),
            "dense_min_margin": result.get("dense_min_margin"),
            "best_start": result.get("best_start"),
            "attempt_count": len(result.get("attempts", []))
            if isinstance(result.get("attempts"), list) else None,
            "attempts_json": self._json_value(result.get("attempts")),
            "attempt_costs_json": self._json_value(result.get("attempt_costs")),
            "solver_config_json": self._json_value(result.get("solver_config") or self.solver_constraints),
            "constraint_config_json": self._json_value(result.get("solver_config") or self.solver_constraints),
            "trajectory_node_count": len(final) if final is not None else None,
            "control_node_count": len(controls) if controls is not None else None,
            "final_trajectory_json": self._json_value(final.tolist()) if final is not None else None,
            "controls_json": self._json_value(controls.tolist()) if controls is not None else None,
            "history_json": self._json_value(result.get("history")),
            "raw_log": raw_log,
        }
        environment = self._get_environment(result_path, result)
        solver_config = result.get("solver_config")
        if not isinstance(solver_config, dict):
            solver_config = self.solver_constraints
        row.update(self._environment_features(environment))
        for key, value in solver_config.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                row[f"solver_config_{key.lower()}"] = value
            else:
                row[f"solver_config_{key.lower()}_json"] = self._json_value(value)
        row["solver_start_speed"] = self._safe_number(solver_config.get("V_CRUISE"))
        row["solver_start_gamma_rad"] = 0.0 if environment is not None else None
        if environment is not None:
            start = environment.get("start_pos")
            goal = environment.get("goal_pos")
            if isinstance(start, (list, tuple)) and len(start) >= 2:
                row["solver_start_x"], row["solver_start_y"] = start[:2]
            if isinstance(start, (list, tuple)) and len(start) >= 3:
                row["solver_start_z"] = start[2]
            if isinstance(start, (list, tuple)) and len(start) >= 2 and isinstance(goal, (list, tuple)) and len(goal) >= 2:
                row["solver_start_heading_rad"] = float(
                    np.arctan2(goal[1] - start[1], goal[0] - start[0])
                )
            if isinstance(goal, (list, tuple)) and len(goal) >= 3:
                row["solver_end_x"], row["solver_end_y"], row["solver_end_z"] = goal[:3]
        row["solver_end_speed_is_free"] = True
        row["solver_end_heading_is_free"] = True
        row["solver_end_gamma_is_fixed"] = bool(solver_config.get("FINAL_LEVEL", False))
        if environment is not None:
            start = np.asarray(environment["start_pos"], dtype=float)
            goal = np.asarray(environment["goal_pos"], dtype=float)
            row["route_distance_m"] = float(np.linalg.norm(goal - start))
        row.update(self._guess_differences(guess, final))
        initial_guess = result.get("initial_guess")
        guess_controls = self._as_array(
            initial_guess.get("u") if isinstance(initial_guess, dict) else None,
            columns=len(self.CONTROL_NAMES),
        )
        guess_flight_time = (
            initial_guess.get("flight_time") if isinstance(initial_guess, dict) else None
        )
        row["initial_guess_flight_time"] = self._safe_number(guess_flight_time)
        row.update(
            self._resample_columns(guess, self.node_count, self.STATE_NAMES, "guess")
        )
        row.update(
            self._resample_columns(
                guess_controls, self.node_count, self.CONTROL_NAMES, "guess_control"
            )
        )
        row.update(self._resample_xyz(final, result.get("flight_time")) if success else
                   self._resample_xyz(None, None))
        for key, value in self.solver_constraints.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                row[f"constraint_{key}"] = value
            else:
                row[f"constraint_{key}_json"] = self._json_value(value)
        if isinstance(result.get("history"), list):
            iterations = [
                record.get("iter") for record in result["history"]
                if isinstance(record, dict) and isinstance(record.get("iter"), int)
            ]
            for key, value in {
                "history_first_iteration": min(iterations) if iterations else None,
                "history_last_iteration": max(iterations) if iterations else None,
            }.items():
                row[key] = value
        return row

    def to_ml_tables(
        self, frame: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split parsed rows into numeric input features, targets, and metadata.

        Post-solve trajectory, objective, status, and diagnostic fields are
        deliberately excluded from X to prevent target leakage.
        """
        feature_columns = list(self.BASE_FEATURE_COLUMNS)
        feature_columns.extend(f"obstacle_count_{kind}" for kind in self.OBSTACLE_KINDS)
        feature_columns.extend(
            f"occupancy_{index:03d}"
            for index in range(int(np.prod(self.OCCUPANCY_RESOLUTION)))
        )
        feature_columns.extend(
            f"guess_{index:03d}_{name}"
            for index in range(self.node_count)
            for name in self.STATE_NAMES
        )
        feature_columns.extend(
            f"guess_control_{index:03d}_{name}"
            for index in range(self.node_count)
            for name in self.CONTROL_NAMES
        )
        feature_columns.append("initial_guess_flight_time")
        if not frame.empty:
            feature_columns.extend(
                column for column in frame.columns
                if column.startswith("solver_config_")
                and not column.endswith("_json")
                and pd.api.types.is_numeric_dtype(frame[column])
            )
            feature_columns.extend(
                column for column in frame.columns
                if column.startswith("environment_metric_")
                and pd.api.types.is_numeric_dtype(frame[column])
            )
        feature_columns = list(dict.fromkeys(feature_columns))
        present_features = [column for column in feature_columns if column in frame.columns]
        X = frame.reindex(columns=present_features).copy()
        for column in X.columns:
            if X[column].dtype == bool:
                X[column] = X[column].astype("int8")
        X = X.apply(pd.to_numeric, errors="coerce")

        present_targets = [column for column in self.TARGET_COLUMNS if column in frame.columns]
        y = frame.reindex(columns=present_targets).copy()
        if "success" in y:
            y["success"] = y["success"].astype("int8")
        metadata_columns = [column for column in self.METADATA_COLUMNS if column in frame.columns]
        metadata = frame.reindex(columns=metadata_columns).copy()
        return X, y, metadata

    def parse_file(self, path: str | Path) -> list[dict[str, Any]]:
        result_path = Path(path).resolve()
        try:
            payload = self._read_json(result_path)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Could not read solver result JSON {result_path}: {exc}") from exc
        records = self._records(payload, result_path)
        return [self.parse_result(record, result_path) for record in records]

    def parse_directory(
        self, directory: str | Path, recursive: bool = False
    ) -> pd.DataFrame:
        """Read result JSON files in a directory and return flattened rows.

        JSON files that do not contain SCP result records (for example raw
        environment files) are ignored. Malformed JSON raises a contextual
        ``ValueError`` rather than silently dropping data.
        """
        root = Path(directory)
        if not root.is_dir():
            raise NotADirectoryError(f"Result directory does not exist: {root}")
        paths = sorted(root.rglob("*.json") if recursive else root.glob("*.json"))
        rows: list[dict[str, Any]] = []
        for path in paths:
            parsed = self.parse_file(path)
            rows.extend(parsed)
        return pd.DataFrame.from_records(rows)

    @classmethod
    def parse_directory_to_dataframe(
        cls,
        directory: str | Path,
        node_count: int = 100,
        environment_file: str | Path | None = None,
        environment_dir: str | Path | None = None,
        solver_constraints: dict[str, Any] | None = None,
        recursive: bool = False,
    ) -> pd.DataFrame:
        """Convenience batch API returning one DataFrame for a result directory."""
        parser = cls(
            node_count=node_count,
            environment_file=environment_file,
            environment_dir=environment_dir,
            solver_constraints=solver_constraints,
        )
        return parser.parse_directory(directory, recursive=recursive)


def parse_results_directory(
    directory: str | Path,
    node_count: int = 100,
    environment_file: str | Path | None = None,
    environment_dir: str | Path | None = None,
    solver_constraints: dict[str, Any] | None = None,
    recursive: bool = False,
) -> pd.DataFrame:
    """Functional wrapper for :meth:`SCPAircraftDatasetParser.parse_directory`."""
    return SCPAircraftDatasetParser.parse_directory_to_dataframe(
        directory=directory,
        node_count=node_count,
        environment_file=environment_file,
        environment_dir=environment_dir,
        solver_constraints=solver_constraints,
        recursive=recursive,
    )
