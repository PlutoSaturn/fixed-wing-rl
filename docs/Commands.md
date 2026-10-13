# fixed-wing-rl: Command Reference

Run every command from the `fixed-wing-rl` project folder. Unless noted, relative file names are read from and written to `localstore/`; the airframe file lives in `aircraftmodel/`.

---

## 1. Aircraft calibration (JSBSim)

| Command | What it does |
|---|---|
| `python -m sim.jsbsim_c172 calibrate` | Runs JSBSim trim and full-throttle sweeps on the C172, fits the drag polar, stall limit and thrust line, and writes `aircraftmodel/c172_params.json`. Every solver loads this file automatically. Rerun after changing any `PLAN_*` setting in `jsbsim_c172.py`. |

Options: `--out FILE` (default `c172_params.json`).

---

## 2. Environment generation

### Single file (`environment/envgen.py`)

| Command | What it does |
|---|---|
| `python -m environment.envgen --size 12000 2500 600 --vehicle-radius 6 --margin 30 --occupancy 0.05 0.12 --endpoints ends --n 10 --out c172_envs.json` | Generates 10 C172-scale environments and saves them to `localstore/c172_envs.json`. |
| `python -m environment.envgen ... --plot` | Same, then shows a 3D plot of the first environment. |

Main options:

| Option | Meaning |
|---|---|
| `--size LX LY LZ` | workspace size, m |
| `--n N` | number of environments |
| `--seed S` | random seed (reproducible output) |
| `--vehicle-radius R`, `--margin M` | clearance $d = R + M$ |
| `--occupancy LO HI` | target obstacle fill range (fraction of volume) |
| `--endpoints random\|ends` | start/goal anywhere, or at opposite ends of the long axis |
| `--types TYPE=WEIGHT ...` | obstacle mix, e.g. `spheroid=1 box=1 wall=0.01` |
| `--max-walls N`, `--wall-yaw DEG`, `--wall-pitch DEG` | wall settings |
| `--cylinder-rotation`, `--spheroid-rotation`, `--box-rotation` | orientation ranges (`M` or `LO:HI` degrees) |
| `--upright-frac F` | share of cylinders that are exactly vertical |
| `--min-size F` | drop obstacles smaller than this |
| `--out FILE` | output file |

### Large datasets (`supervised-learning-main/generate_envs.py`)

Output: `supervised-learning-main/datasets/<dataset>/` (shards, `index.csv`, `manifest.json`). Rerunning resumes; a larger `--n` extends the dataset.

| Command | What it does |
|---|---|
| `modal run supervised-learning-main/generate_envs.py --n 5000 --dataset c172_envs_v1` | Generates 5000 environments on Modal, in parallel shards. |
| `modal run supervised-learning-main/generate_envs.py --n 20 --shard-size 5 --dataset smoke_test_modal` | Small Modal smoke test. |
| `python supervised-learning-main/generate_envs.py --local --n 20 --dataset smoke_test` | Same thing on this computer (multiprocessing). |

Options: `--n`, `--dataset`, `--shard-size`, `--seed`; local only: `--local`, `--workers`. Environment settings (size, fill, clearance) are in `ENV_OVERRIDES` at the top of the file.

---

## 3. Full-trajectory solvers

| Command | What it does |
|---|---|
| `python -m solvers.scp_aircraft --envs c172_envs.json` | Multiple-shooting SCP on every environment (10 starts each). Writes `scp_aircraft_results.json` and `feasible_envs_aircraft.json`. |
| `python -m solvers.scp_aircraft --envs c172_envs.json --show 0` | Plots the **saved** result for environment 0 without re-solving (checks the environment fingerprint matches). |
| `python -m solvers.scp_aircraft --envs c172_envs.json --plot 0` | Solves everything, then re-solves and plots environment 0. |
| `python -m solvers.scp_birkhoff --envs c172_envs.json` | Birkhoff (pseudospectral collocation) SCP. Writes `scp_birkhoff_results.json` and `feasible_envs_birkhoff.json`, in the same format. |
| `python -m solvers.scp_birkhoff --envs c172_envs.json --show 0` | Plots the saved Birkhoff result for environment 0. |

Options (both): `--envs`, `--out`, `--feasible-out`, `--max-envs N` (first N only), `--show INDEX`, `--verbose` (print every iteration). `scp_aircraft` also has `--plot INDEX`.

To solve a dataset shard directly: `--envs supervised-learning-main/datasets/<dataset>/envs/shard_00000.json` (path from the project folder, since it is outside `localstore`).

---

## 4. Receding-horizon planning

| Command | What it does |
|---|---|
| `python -m solvers.receding_horizon --envs c172_envs.json` | Flies each environment by replanning a short horizon (default 25 s, replanning every 5 s) along a guide path, with both methods. Reports collision-free status, flight time and per-replan solve times. |
| `python -m solvers.receding_horizon --envs c172_envs.json --methods birkhoff --horizon 30 --execute 5` | Birkhoff only, 30 s horizon. |
| `python -m solvers.receding_horizon --envs c172_envs.json --baseline` | Also times full-horizon solves for comparison. |

Options: `--envs`, `--out`, `--methods shooting birkhoff`, `--max-envs`, `--horizon S`, `--execute S`, `--baseline`, `--verbose`.

---

## 5. Supervised-learning dataset (solving)

`supervised-learning-main/solve_envs.py`. Output: `supervised-learning-main/datasets/<dataset>/solutions/<run>/` (HDF5 parts with every start's trajectory, `results.csv`, `manifest.json`, `errors.json`). Rerunning resumes; changing solver settings requires a new `--run` name.

| Command | What it does |
|---|---|
| `modal run supervised-learning-main/solve_envs.py --dataset c172_envs_v1 --run birkhoff_v1` | Solves the whole dataset on Modal with Birkhoff, 10 starts each. |
| `modal run supervised-learning-main/solve_envs.py --dataset c172_envs_v1 --run shooting_v1 --method shooting` | Same with multiple shooting. |
| `modal run supervised-learning-main/solve_envs.py --dataset smoke_test_modal --run smoke --starts 2 --max-envs 4 --per-task 1` | Small Modal smoke test. |
| `python supervised-learning-main/solve_envs.py --local --dataset smoke_test --run test --max-envs 2` | Runs on this computer. |

Options: `--dataset`, `--run`, `--method birkhoff|shooting`, `--starts N`, `--max-envs N` (Modal: `-1` = all), `--per-task N`; local only: `--local`, `--workers`. Solver setting changes go in `SOLVER_OVERRIDES` at the top of the file.

Requires `pip install h5py` locally.

---

## 6. JSBSim tracking and visualization

These read the solver results file. Default is `scp_aircraft_results.json`; add `--results scp_birkhoff_results.json` for Birkhoff plans.

| Command | What it does |
|---|---|
| `python -m sim.jsbsim_c172 track` | Flies every planned trajectory in JSBSim with the guidance + autopilot loop; reports tracking error and whether the flown path stayed clear. Writes `c172_tracking.json`. |
| `python -m sim.jsbsim_c172 track --plot 0` | Same, then plots environment 0's plan vs flown path. |
| `python -m sim.jsbsim_c172 record 0` | Flies environment 0 and exports JSBSim commands and states (CSV), the plan (CSV), and the environment as an OBJ mesh + JSON to `localstore/flight_env0/`, for external rendering. |
| `python -m sim.jsbsim_c172 fgview 0` | Flies environment 0 in real time and streams it to FlightGear. Also writes `localstore/start_flightgear_env0.bat` and the obstacle scenery. |
| `python -m sim.jsbsim_c172 fgview 0 --speed 4 --no-wait` | 4× playback, without waiting for Enter. |

FlightGear: start it first with `localstore\start_flightgear_env0.bat`, wait until the aircraft view appears, then run `fgview`.

Options: `--envs`, `--results`, `--out` (track), `--out-dir` (record), `--speed`, `--no-wait` (fgview).

---

## 7. Modal

| Command | What it does |
|---|---|
| `pip install modal` then `modal setup` | Install and log in (once per computer). |
| `modal volume ls fixed-wing-rl-data /datasets` | List datasets mirrored to the Modal Volume. |
| `modal volume get fixed-wing-rl-data /datasets/<dataset> .` | Download a dataset from the Volume. |

---

## 8. Git

| Command | What it does |
|---|---|
| `git status` | Shows changed, staged and untracked files, and any merge in progress. |
| `git pull` | Downloads and merges other people's commits into your current branch. |
| `git checkout -b my-branch` | Creates a new branch and switches to it (uncommitted changes come with you). |
| `git add path/to/file` / `git add .` | Stages files for the next commit. |
| `git restore --staged path/to/file` | Unstages a file. |
| `git commit -m "message"` | Records the staged changes. |
| `git push` / `git push -u origin my-branch` | Uploads commits (first push of a new branch needs `-u origin`). |
| `git merge --abort` | Cancels an unfinished merge. |
| `git commit --no-edit` | Finishes a merge after resolving conflicts. |
| `git check-ignore -v path/to/file` | Shows which `.gitignore` rule is excluding a file. |
| `git add -f path/to/file` | Adds a file even though it is ignored. |
| `git log origin/main --oneline -3` | Shows the last three commits on GitHub's `main`. |

In `.gitignore`, a line starting with `!` re-includes a file (e.g. `*.json` then `!aircraftmodel/c172_params.json`). Ignore a folder's contents (`localstore/*`), not the folder itself (`localstore/`), if you want to re-include files inside it.

---

## Typical workflow

```
python -m sim.jsbsim_c172 calibrate
python -m environment.envgen --size 12000 2500 600 --vehicle-radius 6 --margin 30 --occupancy 0.05 0.12 --endpoints ends --n 10 --out c172_envs.json
python -m solvers.scp_birkhoff --envs c172_envs.json
python -m solvers.scp_birkhoff --envs c172_envs.json --show 0
python -m sim.jsbsim_c172 track --results scp_birkhoff_results.json --plot 0
```

Dataset pipeline:

```
modal run supervised-learning-main/generate_envs.py --n 5000 --dataset c172_envs_v1
modal run supervised-learning-main/solve_envs.py --dataset c172_envs_v1 --run birkhoff_v1
```
