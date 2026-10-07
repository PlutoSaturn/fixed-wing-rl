"""
Solve a generated environment dataset with the SCP solvers, on Modal or locally,
and store everything needed for supervised learning.

Reads the environments written by generate_envs.py and calls the existing
solvers (solvers/scp_birkhoff.py or solvers/scp_aircraft.py, unchanged).
Unlike the solvers' own CLI, every start's result is kept, not only the best,
because different starts often find different valid routes (left or right of
an obstacle), and a learned generator needs to see all of them.

Output (inside the environment dataset's folder, mirrored to the Modal Volume):

    supervised-learning-main/datasets/<DATASET>/solutions/<RUN>/
        manifest.json                 solver, every solver setting, code hashes; checked on resume
        parts/shard_00000_c000.h5     one HDF5 file per task (ENVS_PER_TASK environments)
        results.csv                   one row per environment (rebuilt at the end of every run)
        errors.json                   environments that crashed the solver (with traceback)

Inside each .h5 part, one group per environment, named by its fingerprint:

    /<fingerprint>/
        attrs: index, split, env_shard, status, success, cost, flight_time, iterations,
               solve_time, max_defect, max_violation, dense_min_margin, clearance_m,
               best_start, n_valid_starts, method
        env_json            the environment exactly as generated (string)
        start_state         (6,) initial state the solver used: px py pz V gamma chi
        goal                (3,) goal position
        best/x, best/u      best trajectory as the solver outputs it (states, controls)
        best/t              node times, s
        best/x_rs, best/u_rs  the same, resampled to RESAMPLE_NODES points uniform in time
        best/native/...     Birkhoff only: native LGL nodes X, V, U, node times, segments
        attempts/<k>/       every start k = 0..N_STARTS-1:
            attrs: status, success, cost, flight_time, iterations, solve_time,
                   max_defect, max_violation, dense_min_margin
            x_rs, u_rs      its final trajectory, resampled (stored for failed starts too;
                            check `success` before using one as a label)
            guess_p, guess_t  the initial guess's positions and normalized times
            history         per-iteration log (JSON string)

Run (from the fixed-wing-rl folder):
    modal run supervised-learning-main/solve_envs.py --dataset smoke_test_modal --run birkhoff_v1
    python supervised-learning-main/solve_envs.py --local --dataset smoke_test --run test --max-envs 2

Rerunning the same command resumes (finished parts are skipped).
"""
import os

# one thread per solver process: parallelism comes from many processes / containers
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

# ==========================================================================
# USER SETTINGS
# ==========================================================================
DATASET          = "c172_envs_v1"   # environment dataset made by generate_envs.py
RUN              = "birkhoff_v1"    # name for this set of solutions (new name = new settings)
DATASETS_DIR     = None             # None = supervised-learning-main/datasets
METHOD           = "birkhoff"       # "birkhoff" (scp_birkhoff.py) or "shooting" (scp_aircraft.py)
N_STARTS         = 10               # initial guesses per environment; all are kept
MAX_ENVS         = None             # solve only the first N environments (None = all)
ENVS_PER_TASK    = 5                # environments per Modal call / per output part (~5 min each)
RESAMPLE_NODES   = 64               # fixed-length copies of every trajectory (uniform in time)
LOCAL_WORKERS    = None             # --local: processes to use (None = all CPU cores)

# Overrides for solver settings, applied to the solver modules before solving.
# Use the setting names from scp_aircraft.py / scp_birkhoff.py, e.g.
#   {"W_TRUST": 1.0, "MAX_ITERS": 100}
# Every setting in force (overridden or not) is recorded in the manifest.
SOLVER_OVERRIDES = {}

# Modal
MODAL_APP_NAME   = "fixed-wing-solve"
MODAL_VOLUME     = "fixed-wing-rl-data"   # None = do not mirror results to a Modal Volume
MAX_CONTAINERS   = 200              # tasks solved at the same time
CPU_PER_TASK     = 1.0
MEMORY_MB        = 2048
TASK_TIMEOUT_S   = 3 * 3600
TASK_RETRIES     = 2
PYTHON_VERSION   = "3.12"
PIP_PACKAGES     = ["numpy", "scipy", "cvxpy", "clarabel", "h5py"]   # pin versions to match yours

# ==========================================================================
import argparse
import csv
import hashlib
import io
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

_HERE = Path(__file__).resolve().parent
PROJECT = _HERE.parent if (_HERE.parent / "solvers").is_dir() else Path("/root/fixed-wing-rl")
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

VOLUME_MOUNT = "/data"
CODE_DIRS = ("environment", "solvers", "aircraftmodel")      # copied into the Modal image


# --------------------------------------------------------------------------
# solver setup
# --------------------------------------------------------------------------
_SOLVER = {}


def get_solver(method, overrides, n_starts):
    """Import the solver module once per process, apply overrides, build the solver."""
    key = (method, json.dumps(overrides, sort_keys=True), n_starts)
    if key in _SOLVER:
        return _SOLVER[key]
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="cvxpy")
    from solvers import scp_aircraft as base
    mods = [base]
    if method == "birkhoff":
        from solvers import scp_birkhoff as bk
        mods.append(bk)
    elif method != "shooting":
        raise ValueError(f"METHOD must be 'birkhoff' or 'shooting', not {method!r}")
    for name, value in overrides.items():
        hits = [m for m in mods if hasattr(m, name)]
        if not hits:
            raise ValueError(f"SOLVER_OVERRIDES: no setting named {name!r} in the {method} solver")
        for m in hits:
            setattr(m, name, value)
    if base.AIRFRAME_NAME == "generic small UAV":
        raise RuntimeError("Airframe file not found: the solver fell back to the generic UAV. "
                           "Check that aircraftmodel/c172_params.json exists.")
    solver = (mods[-1].BirkhoffAircraftSCP if method == "birkhoff" else base.AircraftSCP)(
        verbose=False, n_starts=n_starts)
    _SOLVER[key] = (solver, mods)
    return _SOLVER[key]


def solver_settings(mods):
    """Every UPPER_CASE setting of the solver modules that can be written as JSON."""
    out = {}
    for m in mods:
        for k, v in vars(m).items():
            if k.isupper() and not k.startswith("_"):
                try:
                    json.dumps(v)
                except TypeError:
                    continue
                out[f"{m.__name__.split('.')[-1]}.{k}"] = v
    return json.loads(json.dumps(out))          # tuples -> lists, so it compares equal to the saved copy


# --------------------------------------------------------------------------
# solving one environment, keeping every start
# --------------------------------------------------------------------------
def _better(res, best):
    """The solvers' own best-of rule (SCPSolver.solve)."""
    if best is None:
        return True
    if res.success != best.success:
        return res.success
    if res.success:
        return res.cost < best.cost
    return res.max_violation + res.max_defect < best.max_violation + best.max_defect


def resample(x, m):
    """Linear interpolation of node values (uniform in normalized time) to m points."""
    import numpy as np
    x = np.asarray(x, float)
    s = np.linspace(0.0, 1.0, len(x))
    r = np.linspace(0.0, 1.0, m)
    return np.stack([np.interp(r, s, x[:, j]) for j in range(x.shape[1])], axis=1)


def solve_env(solver, env):
    """Run every start and return (best result, list of (guess, result))."""
    import numpy as np
    t0 = time.perf_counter()
    solver.n_nodes = solver.nodes_for(env)
    runs, best, best_k = [], None, -1
    for k, guess in enumerate(solver.initial_guesses(env)):
        X0 = np.asarray(guess[0])
        t_guess = (np.asarray(solver.r) if hasattr(solver, "r") and len(solver.r) == len(X0)
                   else np.linspace(0.0, 1.0, len(X0)))
        res = solver._solve_from(env, guess)
        runs.append((X0[:, :3].copy(), t_guess, res))
        if _better(res, best):
            best, best_k = res, k
    best.attempts = [r.status for _, _, r in runs]
    best.attempt_costs = [r.cost if r.success else None for _, _, r in runs]
    best.best_start = best_k
    best.solve_time = time.perf_counter() - t0
    return best, runs


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def write_env_group(h5, solver, env, info, best, runs, method, clearance):
    import numpy as np
    fp = env.fingerprint()
    g = h5.create_group(fp)
    a = g.attrs
    a.update({"index": info["index"], "split": info.get("split", ""), "env_shard": info["shard"],
              "method": method, "status": best.status, "success": bool(best.success),
              "cost": _f(best.cost), "flight_time": _f(best.flight_time),
              "iterations": int(sum(r.iterations for _, _, r in runs)),
              "solve_time": _f(best.solve_time), "max_defect": _f(best.max_defect),
              "max_violation": _f(best.max_violation), "dense_min_margin": _f(best.dense_min_margin),
              "clearance_m": _f(clearance), "best_start": int(best.best_start),
              "n_starts": len(runs), "n_valid_starts": int(sum(r.success for _, _, r in runs))})
    g.create_dataset("env_json", data=json.dumps(env.to_dict()))
    x0, goal = solver.boundary(env)
    g.create_dataset("start_state", data=np.asarray(x0, float))
    g.create_dataset("goal", data=np.asarray(goal, float))

    b = g.create_group("best")
    x, u = np.asarray(best.x, float), np.asarray(best.u, float)
    b.create_dataset("x", data=x, compression="gzip")
    b.create_dataset("u", data=u, compression="gzip")
    b.create_dataset("t", data=np.linspace(0.0, _f(best.flight_time), len(x)))
    b.create_dataset("x_rs", data=resample(x, RESAMPLE_NODES))
    b.create_dataset("u_rs", data=resample(u, RESAMPLE_NODES))
    bk = getattr(best, "birkhoff", None)
    if bk:
        n = b.create_group("native")
        for key in ("X", "V", "U", "node_times_normalized"):
            n.create_dataset(key, data=np.asarray(bk[key], float), compression="gzip")
        n.attrs.update({k: _f(bk[k]) for k in ("segments", "degree", "nodes", "verify_defect",
                                                 "open_loop_max_dev_m", "open_loop_end_error_m")})

    att = g.create_group("attempts")
    for k, (guess_p, guess_t, r) in enumerate(runs):
        s = att.create_group(str(k))
        s.attrs.update({"status": r.status, "success": bool(r.success), "cost": _f(r.cost),
                        "flight_time": _f(r.flight_time), "iterations": int(r.iterations),
                        "solve_time": _f(r.solve_time), "max_defect": _f(r.max_defect),
                        "max_violation": _f(r.max_violation),
                        "dense_min_margin": _f(r.dense_min_margin)})
        s.create_dataset("x_rs", data=resample(r.x, RESAMPLE_NODES))
        s.create_dataset("u_rs", data=resample(r.u, RESAMPLE_NODES))
        s.create_dataset("guess_p", data=guess_p, compression="gzip")
        s.create_dataset("guess_t", data=guess_t, compression="gzip")
        s.create_dataset("history", data=json.dumps(r.history, default=_f))
    return fp


def solve_task(task):
    """Solve one task's environments. Returns the .h5 part as bytes plus a summary."""
    import h5py
    from environment.envgen import Environment
    from solvers.scp_aircraft import plan_clearance

    solver, _ = get_solver(task["method"], task["overrides"], task["n_starts"])
    buf, errors, t_task = io.BytesIO(), [], time.perf_counter()
    with h5py.File(buf, "w") as h5:
        h5.attrs.update({"part": task["part"], "method": task["method"], "run": task["run"]})
        for info in task["envs"]:
            env = Environment.from_dict(info["env"])
            try:
                best, runs = solve_env(solver, env)
                try:
                    clearance = plan_clearance(env, best)          # independent re-check
                except Exception:
                    clearance = float("nan")
                write_env_group(h5, solver, env, info, best, runs, task["method"], clearance)
            except Exception:
                errors.append({"index": info["index"], "fingerprint": env.fingerprint(),
                               "traceback": traceback.format_exc()})
        h5.attrs["errors"] = json.dumps(errors)
        h5.attrs["complete"] = True                                  # written last
    return {"part": task["part"], "h5": buf.getvalue(), "n_envs": len(task["envs"]),
            "errors": errors, "time_s": round(time.perf_counter() - t_task, 1)}


# --------------------------------------------------------------------------
# planning, saving and bookkeeping (runs on your machine)
# --------------------------------------------------------------------------
def datasets_root():
    return Path(DATASETS_DIR) if DATASETS_DIR else _HERE / "datasets"


def write_bytes_atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def write_json_atomic(path, obj):
    write_bytes_atomic(path, json.dumps(obj, indent=1, default=str).encode())


def file_sha1(path):
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()[:12]


def git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT,
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


def part_done(path):
    if not path.exists():
        return False
    try:
        import h5py
        with h5py.File(path, "r") as h5:
            return bool(h5.attrs.get("complete", False))
    except Exception:
        return False


def load_split_lookup(env_root):
    index = env_root / "index.csv"
    if not index.exists():
        return {}
    with open(index, newline="") as f:
        return {int(r["index"]): r.get("split", "") for r in csv.DictReader(f)}


def plan_tasks(env_root, out_root, run, method, n_starts, overrides, max_envs, per_task):
    """Tasks for every environment shard, minus parts already finished."""
    shards = sorted((env_root / "envs").glob("shard_*.json"))
    if not shards:
        raise SystemExit(f"No environment shards in {env_root / 'envs'}. Run generate_envs.py first.")
    splits = load_split_lookup(env_root)
    todo, done, total = [], 0, 0
    for shard_path in shards:
        shard = int(shard_path.stem.split("_")[1])
        envs = json.loads(shard_path.read_text())
        items = []
        for d in envs:
            idx = int(d.get("metrics", {}).get("dataset", {}).get("index", -1))
            if max_envs is not None and idx >= max_envs:
                continue
            items.append({"index": idx, "shard": shard, "split": splits.get(idx, ""), "env": d})
        for c in range(0, len(items), per_task):
            chunk = items[c:c + per_task]
            part = f"shard_{shard:05d}_c{c // per_task:03d}"
            total += len(chunk)
            if part_done(out_root / "parts" / f"{part}.h5"):
                done += len(chunk)
                continue
            todo.append({"part": part, "run": run, "method": method, "n_starts": n_starts,
                         "overrides": overrides, "envs": chunk})
    return todo, done, total


_MUST_MATCH = ("method", "n_starts", "resample_nodes", "solver_overrides", "solver_settings",
               "env_dataset_manifest_sha1")


def prepare(dataset, run, method, n_starts, max_envs, per_task):
    env_root = datasets_root() / dataset
    if not (env_root / "manifest.json").exists():
        raise SystemExit(f"Environment dataset not found: {env_root}")
    out_root = env_root / "solutions" / run
    out_root.mkdir(parents=True, exist_ok=True)
    _, mods = get_solver(method, SOLVER_OVERRIDES, n_starts)        # also checks the airframe
    from solvers import scp_aircraft as base
    manifest = {
        "dataset": dataset, "run": run, "method": method, "n_starts": n_starts,
        "resample_nodes": RESAMPLE_NODES, "envs_per_task": per_task,
        "airframe": base.AIRFRAME_NAME,
        "solver_overrides": json.loads(json.dumps(SOLVER_OVERRIDES)),
        "solver_settings": solver_settings(mods),
        "env_dataset_manifest_sha1": file_sha1(env_root / "manifest.json"),
        "code_sha1": {f"{d}/{p.name}": file_sha1(p) for d in ("environment", "solvers", "aircraftmodel")
                      for p in sorted((PROJECT / d).glob("*.*")) if p.suffix in (".py", ".json")},
        "git_commit": git_commit(),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    path = out_root / "manifest.json"
    if path.exists():
        old = json.loads(path.read_text())
        bad = [k for k in _MUST_MATCH if old.get(k) != manifest[k]]
        if bad:
            raise SystemExit(f"Run '{run}' already exists with different {', '.join(bad)}.\n"
                             f"Use a new --run name (or delete {out_root}) rather than mixing settings.")
        if old.get("code_sha1") != manifest["code_sha1"]:
            print("WARNING: solver / envgen / airframe files changed since this run started. "
                  "Consider a new --run name.")
        manifest = {**old, "updated": manifest["created"]}
    write_json_atomic(path, manifest)
    todo, done, total = plan_tasks(env_root, out_root, run, method, n_starts, SOLVER_OVERRIDES,
                                   max_envs, per_task)
    print(f"Run '{run}' on '{dataset}' with {method} ({n_starts} starts, airframe: "
          f"{manifest['airframe']}): {total} environments, {done} already solved, "
          f"{total - done} to solve in {len(todo)} tasks.")
    return out_root, todo


def save_part(out_root, result):
    write_bytes_atomic(out_root / "parts" / f"{result['part']}.h5", result["h5"])


def report(result, n_done, n_total, t0):
    print(f"  {result['part']}: {result['n_envs']} envs, {len(result['errors'])} errors, "
          f"{result['time_s'] / 60:.1f} min   [{n_done}/{n_total} tasks, "
          f"{(time.time() - t0) / 60:.1f} min elapsed]", flush=True)


RESULT_COLUMNS = ("fingerprint", "index", "split", "env_shard", "part", "status", "success", "cost",
                  "flight_time", "solve_time", "iterations", "best_start", "n_valid_starts",
                  "n_starts", "dense_min_margin", "clearance_m", "max_defect", "max_violation")


def build_results(out_root):
    import h5py
    rows, errors = [], []
    for p in sorted((out_root / "parts").glob("*.h5")):
        with h5py.File(p, "r") as h5:
            if not h5.attrs.get("complete", False):
                continue
            errors += json.loads(h5.attrs.get("errors", "[]"))
            for fp, g in h5.items():
                a = dict(g.attrs)
                rows.append({c: (fp if c == "fingerprint" else p.stem if c == "part" else
                                 a.get(c, "")) for c in RESULT_COLUMNS})
    for r in rows:
        for k, v in r.items():
            if hasattr(v, "item"):
                r[k] = v.item()
    rows.sort(key=lambda r: r["index"])
    if rows:
        tmp = out_root / "results.csv.tmp"
        with open(tmp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(RESULT_COLUMNS))
            w.writeheader()
            w.writerows(rows)
        os.replace(tmp, out_root / "results.csv")
    write_json_atomic(out_root / "errors.json", errors)
    return rows, errors


def print_summary(out_root, rows, errors, wall_s):
    import numpy as np
    print(f"\nSolutions: {out_root}")
    print(f"  environments solved: {len(rows)}   solver errors: {len(errors)}")
    if not rows:
        return
    statuses = {}
    for r in rows:
        statuses[r["status"]] = statuses.get(r["status"], 0) + 1
    print("  status: " + ", ".join(f"{k} {v}" for k, v in sorted(statuses.items(), key=lambda kv: -kv[1])))
    ok = [r for r in rows if r["success"]]
    print(f"  feasible: {len(ok)}/{len(rows)} ({len(ok) / len(rows):.0%})")
    if ok:
        for key, label in (("cost", "cost"), ("flight_time", "flight time (s)"),
                           ("solve_time", "solve time (s)"), ("n_valid_starts", "valid starts"),
                           ("clearance_m", "clearance (m)")):
            v = np.array([r[key] for r in ok], float)
            v = v[np.isfinite(v)]
            if len(v):
                print(f"  {label:16s} min {v.min():9.2f}   median {np.median(v):9.2f}   max {v.max():9.2f}")
    print(f"  wall-clock this run: {wall_s / 60:.1f} min")
    if errors:
        print(f"  {len(errors)} environment(s) crashed the solver; see errors.json")


# --------------------------------------------------------------------------
# local mode
# --------------------------------------------------------------------------
def run_local(dataset, run, method, n_starts, max_envs, per_task, workers):
    from multiprocessing import Pool
    out_root, todo = prepare(dataset, run, method, n_starts, max_envs, per_task)
    t0 = time.time()
    if todo:
        workers = min(workers or os.cpu_count() or 1, len(todo))
        with Pool(workers) as pool:
            for k, result in enumerate(pool.imap_unordered(solve_task, todo), 1):
                save_part(out_root, result)
                report(result, k, len(todo), t0)
    rows, errors = build_results(out_root)
    print_summary(out_root, rows, errors, time.time() - t0)


# --------------------------------------------------------------------------
# Modal
# --------------------------------------------------------------------------
try:
    import modal
except ImportError:
    modal = None

if modal is not None:
    app = modal.App(MODAL_APP_NAME)
    image = (modal.Image.debian_slim(python_version=PYTHON_VERSION)
             .pip_install(*PIP_PACKAGES)
             .env({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                   "MKL_NUM_THREADS": "1", "RAYON_NUM_THREADS": "1"}))
    for _d in CODE_DIRS:
        image = image.add_local_dir(str(PROJECT / _d), f"/root/fixed-wing-rl/{_d}",
                                    ignore=["__pycache__", "*.pyc"])
    volume = (modal.Volume.from_name(MODAL_VOLUME, create_if_missing=True)
              if MODAL_VOLUME else None)

    @app.function(image=image, cpu=CPU_PER_TASK, memory=MEMORY_MB, timeout=TASK_TIMEOUT_S,
                  retries=TASK_RETRIES, max_containers=MAX_CONTAINERS,
                  volumes={VOLUME_MOUNT: volume} if volume is not None else {})
    def solve_task_remote(task, dataset):
        result = solve_task(task)
        if volume is not None:
            path = (Path(VOLUME_MOUNT) / "datasets" / dataset / "solutions" / task["run"]
                    / "parts" / f"{task['part']}.h5")
            write_bytes_atomic(path, result["h5"])
            volume.commit()
        return result

    @app.local_entrypoint()
    def main(dataset: str = DATASET, run: str = RUN, method: str = METHOD,
             starts: int = N_STARTS, max_envs: int = -1, per_task: int = ENVS_PER_TASK):
        max_envs = None if max_envs < 0 else max_envs
        if max_envs is None and MAX_ENVS is not None:
            max_envs = MAX_ENVS
        out_root, todo = prepare(dataset, run, method, starts, max_envs, per_task)
        t0, failed = time.time(), []
        calls = solve_task_remote.starmap([(t, dataset) for t in todo], order_outputs=False,
                                          return_exceptions=True)
        for k, result in enumerate(calls, 1):
            if isinstance(result, BaseException):
                failed.append(repr(result))
                print(f"  a task failed after retries: {result!r}", flush=True)
                continue
            save_part(out_root, result)
            report(result, k, len(todo), t0)
        rows, errors = build_results(out_root)
        if volume is not None:
            with volume.batch_upload(force=True) as batch:
                for name in ("manifest.json", "results.csv", "errors.json"):
                    if (out_root / name).exists():
                        batch.put_file(str(out_root / name),
                                       f"/datasets/{dataset}/solutions/{run}/{name}")
            print(f"Mirrored to Modal Volume '{MODAL_VOLUME}' at /datasets/{dataset}/solutions/{run}")
        print_summary(out_root, rows, errors, time.time() - t0)
        if failed:
            print(f"\n{len(failed)} task(s) failed; rerun the same command to retry them.")


# --------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Solve an environment dataset. Use 'modal run' "
                                 "for the cloud, or --local here.")
    ap.add_argument("--local", action="store_true", help="run on this computer (multiprocessing)")
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--run", default=RUN)
    ap.add_argument("--method", default=METHOD, choices=["birkhoff", "shooting"])
    ap.add_argument("--starts", type=int, default=N_STARTS)
    ap.add_argument("--max-envs", type=int, default=MAX_ENVS)
    ap.add_argument("--per-task", type=int, default=ENVS_PER_TASK)
    ap.add_argument("--workers", type=int, default=LOCAL_WORKERS)
    a = ap.parse_args()
    if not a.local:
        raise SystemExit("For Modal use:  modal run supervised-learning-main/solve_envs.py "
                         "--dataset ... --run ...\nFor this computer add --local.")
    run_local(a.dataset, a.run, a.method, a.starts, a.max_envs, a.per_task, a.workers)