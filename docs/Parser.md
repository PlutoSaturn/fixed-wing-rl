# SCP Aircraft Dataset Parser & Pipeline

The SCP Aircraft Dataset pipeline converts raw optimization solver output JSON files into ML-ready tabular datasets (CSV / Parquet) for supervised learning workflows.

---

## Overview & Context

The raw SCP solver (`scp_aircraft.py`) saves individual runs as nested JSON files. However, raw outputs present several challenges for machine learning models:

* **Missing Problem Setup & Geometry:** Solver JSON files only store an environment hash/fingerprint. Obstacle geometries, workspace bounds, and start/goal coordinates must be dynamically matched from the original `envgen.py` files.
* **Missing Initial Guesses:** The solver does not persist the generated initial guess trajectory. "Guess-vs-final" improvement metrics cannot be computed without reproducibly regenerating the guess.
* **Aggregate-Only Diagnostics:** Failure metrics are limited to high-level aggregates (`max_defect`, `max_violation`, `status`). Named constraint residuals, specific obstacle IDs, and low-level solver exceptions are not saved.
* **Inconsistent & Unstructured Format:** Raw node trajectories, variable-length iteration histories, and obstacle lists are nested arrays that standard ML frameworks (e.g., PyTorch, XGBoost) cannot process directly.

---

### Model & Data Notes
* **Preprocessing:** Continuous numerical inputs require scaling, and categorical targets (`failure_category`) require encoding.
* **Not RL Experience Data:** Output rows represent completed optimization runs, not step-by-step $(s, a, r, s')$ transition sequences required for Reinforcement Learning.

---

## Quickstart & Commands

### 1. Run Complete Data Pipeline
```powershell
# Step 1: Generate 100 random environments (Occupancy target: 10% - 35%)
.\.venv\Scripts\python.exe .\environment\envgen.py --n 100 --seed 123 --out c172_envs.json --no-plot

# Step 2: Run the SCP aircraft solver across all generated environments
.\.venv\Scripts\python.exe -m solvers.scp_aircraft --envs c172_envs.json

# Step 3: Build structured dataset CSVs
.\.venv\Scripts\python.exe .\build_dataset.py