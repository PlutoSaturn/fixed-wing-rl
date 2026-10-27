import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from solvers.scp_aircraft_dataset import SCPAircraftDatasetParser


class SCPAircraftDatasetParserTests(unittest.TestCase):
    def test_parse_success_resamples_trajectory_and_compares_guess(self):
        parser = SCPAircraftDatasetParser(node_count=5)
        result = {
            "success": True,
            "status": "feasible",
            "iterations": 3,
            "flight_time": 4.0,
            "x": [[0, 0, 0, 15, 0, 0], [2, 0, 1, 15, 0, 0], [4, 0, 2, 15, 0, 0]],
            "initial_guess": {
                "x": [[0, 0, 0, 15, 0, 0], [4, 0, 0, 15, 0, 0]],
                "u": [[1, 1, 0], [1, 1, 0]],
                "flight_time": 4.0,
            },
            "u": [[1, 1, 0], [1, 1, 0], [1, 1, 0]],
            "history": [
                {"iter": 0, "event": "initial_guess"},
                {"iter": 3, "event": "converged"},
            ],
        }

        row = parser.parse_result(result)

        self.assertEqual(row["failure_category"], "success")
        self.assertEqual(row["stop_iteration"], 3)
        self.assertTrue(row["initial_guess_available"])
        self.assertEqual(row["trajectory_002_x"], 2.0)
        self.assertEqual(row["trajectory_004_z"], 2.0)
        self.assertAlmostEqual(row["guess_final_position_max_error_m"], 2.0)

    def test_parse_failure_uses_logs_for_category_iteration_and_constraint(self):
        parser = SCPAircraftDatasetParser(node_count=3)
        result = {
            "success": False,
            "status": "solver_error",
            "iterations": 8,
            "history": [{"iter": 8, "event": "subproblem_failed"}],
            "raw_log": "Iteration 7: maximum thrust constraint violation",
        }

        row = parser.parse_result(result)

        self.assertEqual(row["failure_category"], "solver_error")
        self.assertEqual(row["stop_iteration"], 8)
        self.assertEqual(row["log_iteration"], 7)
        self.assertEqual(row["failure_constraint"], "max_thrust")
        self.assertFalse(row["initial_guess_available"])
        self.assertIsNone(row["guess_final_position_max_error_m"])

    def test_parse_directory_skips_environment_json_and_loads_environment_features(self):
        environment = {
            "start_pos": [0, 1, 2],
            "start_vel": [3, 4, 5],
            "goal_pos": [10, 11, 12],
            "goal_vel": [0, 0, 0],
            "bounds_lo": [-1, -2, -3],
            "bounds_hi": [20, 21, 22],
            "vehicle_radius": 1,
            "margin": 2,
            "obstacles": [{"type": "box", "center": [5, 5, 5], "half_extents": [1, 1, 1]}],
            "metrics": {"occupancy_fraction": 0.2},
        }
        result = {
            "env_index": 0,
            "env_file": "environments.json",
            "env_fingerprint": SCPAircraftDatasetParser._fingerprint(environment),
            "solver_config": {"V_CRUISE": 15.0, "T_MAX": 20.0, "FINAL_LEVEL": False},
            "success": True,
            "status": "feasible",
            "flight_time": 2.0,
            "x": [[0, 1, 2], [10, 11, 12]],
            "u": [[0, 0, 0], [0, 0, 0]],
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "environments.json").write_text(json.dumps([environment]), encoding="utf-8")
            (root / "results.json").write_text(json.dumps([result]), encoding="utf-8")

            frame = SCPAircraftDatasetParser.parse_directory_to_dataframe(
                root, node_count=3
            )

        self.assertEqual(len(frame), 1)
        self.assertEqual(frame.loc[0, "start_pos_z"], 2.0)
        self.assertEqual(frame.loc[0, "position_lo_x"], 0.0)
        self.assertEqual(frame.loc[0, "clearance"], 3.0)
        self.assertEqual(frame.loc[0, "obstacle_count"], 1)
        self.assertEqual(frame.loc[0, "trajectory_002_x"], 10.0)
        self.assertEqual(frame.loc[0, "solver_config_v_cruise"], 15.0)
        self.assertEqual(frame.loc[0, "solver_start_heading_rad"], np.arctan2(10, 10))

        features, targets, metadata = SCPAircraftDatasetParser(
            node_count=3
        ).to_ml_tables(frame)
        self.assertNotIn("success", features.columns)
        self.assertNotIn("status", features.columns)
        self.assertNotIn("trajectory_000_x", features.columns)
        self.assertTrue(all(pd.api.types.is_numeric_dtype(dtype) for dtype in features.dtypes))
        self.assertEqual(targets.loc[0, "success"], 1)
        self.assertEqual(targets.loc[0, "failure_category"], "success")
        self.assertEqual(metadata.loc[0, "env_index"], 0)
        self.assertIn("occupancy_000", features.columns)
        self.assertIn("obstacle_count_box", features.columns)


if __name__ == "__main__":
    unittest.main()
