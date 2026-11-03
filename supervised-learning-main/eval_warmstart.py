"""
Evaluate warm starts by what they do for the solver, not by how close the guess looks.

For every solved environment in the chosen split, the benchmark is the best of the
cold-start solves already in the dataset (the lowest-cost valid start of solve_envs.py).
Each warm-start method builds an initial guess from the environment alone, the same
solver (same settings as the benchmark run) is run from it, and we record:

    did it reach a valid trajectory, and with what status
    iterations to the first node-feasible iterate (dynamics and obstacles satisfied at the
        nodes / check points; the final dense and RK4 checks only run at the end)
    iterations to convergence, solve time (and the time to make the guess)
    how close the converged trajectory is to the benchmark: its flight time, as % above the
        benchmark's (negative = faster than the benchmark); the cost gap is kept in the CSV
    how good the guess itself was: its flight time (path length at cruise speed, as the
        solver initializes it) relative to the benchmark's, and % of it inside an inflated obstacle

Warm-start methods (--methods, comma-separated):
    model     the learned network (--model): its K routes, tried in confidence order
    straight  the straight line start -> goal (the solver's own first start)
    guide     the receding-horizon guide path (grid shortest path + line-of-sight shortcuts);
              no learning, a search baseline
    oracle    the benchmark path itself, as a polyline (the best any path-predicting warm start
              could do: it shows how many iterations a "perfect" path guess still needs)
The cold-start solves stored in the dataset are added for free as method "cold" (no re-solve):
its route 0 is the straight line, so it also checks that "straight" reproduces the dataset.

Every guess goes through the solver's own polyline guess (cruise speed, positions sampled
along the path at the solver's node times), so the methods differ only in the path.

Output: supervised-learning-main/evaluations/<NAME>/
    manifest.json       settings, benchmark run, model file hash; checked on resume
    parts/part_*.json   raw rows per task, including the guess and final paths (64 points)
    results.csv         one row per environment x method x route
    summary.json/.txt   the comparison table (also printed)

Run (from the fixed-wing-rl folder):
    modal run supervised-learning-main/eval_warmstart.py --model warmstart_v2
    modal run supervised-learning-main/eval_warmstart.py --model warmstart_v2 --max-envs 20 --name smoke
    python supervised-learning-main/eval_warmstart.py --local --dataset cache_test --model test_v2 --split train
The benchmark solutions and the model are read on this computer (model inference runs
here, on CPU, before anything is sent to Modal); Modal only runs the solver.
Rerunning the same command resumes (finished parts are skipped).
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

# ==========================================================================
# USER SETTINGS
# ==========================================================================
DATASET        = "c172_envs_v1"
RUN            = "birkhoff_v1"        # benchmark solutions (and the solver settings to reuse)
MODEL          = "warmstart_v2"       # supervised-learning-main/models/<MODEL>.pt
NAME           = None                 # evaluation name; None = "<MODEL>_<SPLIT>"
SPLIT          = "test"               # "test", "val", "train", or several: "val,test"
METHODS        = "model,straight,guide,oracle"
MAX_ROUTES     = None                 # model routes tried per environment (None = all K)
MAX_ENVS       = None                 # first N environments of the split (None = all)
ENVS_PER_TASK  = 4                    # environments per Modal call (~8 solves each)
NEAR_OPT_PCT   = 1.0                  # "near optimal" = flight time within this % of the benchmark
N_PTS          = 64                   # points per stored path (uniform in normalized time)
LOCAL_WORKERS  = None

# Modal
MODAL_APP_NAME = "fixed-wing-eval"
MODAL_VOLUME   = "fixed-wing-rl-data"
MAX_CONTAINERS = 200
CPU_PER_TASK   = 1.0
MEMORY_MB      = 2048
TASK_TIMEOUT_S = 3 * 3600
TASK_RETRIES   = 2
PYTHON_VERSION = "3.12"
PIP_PACKAGES   = ["numpy", "scipy", "cvxpy", "clarabel"]

# ==========================================================================
import argparse
import csv
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

_HERE = Path(__file__).resolve().parent
PROJECT = _HERE.parent if (_HERE.parent / "solvers").is_dir() else Path("/root/fixed-wing-rl")
for _p in (str(PROJECT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

VOLUME_MOUNT = "/data"
CODE_DIRS = ("environment", "solvers", "aircraftmodel")
ALL_METHODS = ("model", "straight", "guide", "oracle")


# --------------------------------------------------------------------------
# small helpers (used on both sides)
# --------------------------------------------------------------------------
def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def uniform(p, t, m=N_PTS):
    """Positions p (n, 3) at normalized times t -> m points uniform in normalized time."""
    import numpy as np
    p, t = np.asarray(p, float), np.asarray(t, float)
    r = np.linspace(t[0], t[-1], m)
    return np.stack([np.interp(r, t, p[:, k]) for k in range(3)], axis=1)


def resample_nodes(x, m=N_PTS):
    """Node values uniform in normalized time -> m points (as solve_envs.resample)."""
    import numpy as np
    x = np.asarray(x, float)
    s, r = np.linspace(0.0, 1.0, len(x)), np.linspace(0.0, 1.0, m)
    return np.stack([np.interp(r, s, x[:, j]) for j in range(x.shape[1])], axis=1)


def path_length(p):
    import numpy as np
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def first_feasible_iter(history, tol_defect, tol_violation):
    """Index (1-based, counting every iteration incl. failed subproblems and refinement
    rounds) of the first iterate that satisfies the dynamics and obstacles at the nodes."""
    k = 0
    for rec in history:
        if "iter" not in rec or rec.get("event") == "restored_best_valid":
            continue
        k += 1
        if rec.get("max_defect", float("inf")) <= tol_defect and \
                rec.get("max_violation", float("inf")) <= tol_violation:
            return k
    return None


def write_bytes_atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def write_json_atomic(path, obj):
    write_bytes_atomic(path, json.dumps(obj, indent=1, default=_f).encode())


# --------------------------------------------------------------------------
# solver side (runs on Modal or in a local worker)
# --------------------------------------------------------------------------
_SOLVER = {}


def get_solver(method, overrides):
    """Same solver and settings as the benchmark run (as in solve_envs.py)."""
    key = (method, json.dumps(overrides, sort_keys=True))
    if key in _SOLVER:
        return _SOLVER[key]
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="cvxpy")
    warnings.filterwarnings("ignore", message="Solution may be inaccurate")   # accepted steps are re-checked anyway
    from solvers import scp_aircraft as base
    mods = [base]
    if method == "birkhoff":
        from solvers import scp_birkhoff as bk
        mods.append(bk)
    for name, value in overrides.items():
        hits = [m for m in mods if hasattr(m, name)]
        if not hits:
            raise ValueError(f"no solver setting named {name!r}")
        for m in hits:
            setattr(m, name, value)
    if base.AIRFRAME_NAME == "generic small UAV":
        raise RuntimeError("Airframe file not found (aircraftmodel/c172_params.json).")
    solver = (mods[-1].BirkhoffAircraftSCP if method == "birkhoff" else base.AircraftSCP)(
        verbose=False, n_starts=1)
    _SOLVER[key] = (solver, base)
    return _SOLVER[key]


def guess_paths(env, rec, methods, max_routes):
    """(method, route rank, confidence, polyline (n, 3), seconds to make it) for every guess."""
    import numpy as np
    out = []
    for m in methods:
        t0 = time.perf_counter()
        if m == "model":
            K = len(rec["model_paths"]) if max_routes is None else min(max_routes, len(rec["model_paths"]))
            per = rec["model_time_s"] / max(len(rec["model_paths"]), 1)
            for k in range(K):
                out.append(("model", k, rec["model_conf"][k], np.asarray(rec["model_paths"][k], float), per))
            continue
        if m == "straight":
            pts = np.stack([env.start_pos, env.goal_pos])
        elif m == "guide":
            from solvers.receding_horizon import guide_path
            try:
                pts = guide_path(env)[0]
            except Exception:
                out.append(("guide", 0, 1.0, None, time.perf_counter() - t0))      # no connected route
                continue
        elif m == "oracle":
            pts = np.asarray(rec["bench_path"], float)
        else:
            raise ValueError(f"unknown method {m!r}")
        out.append((m, 0, 1.0, pts, time.perf_counter() - t0))
    return out


def evaluate_guess(solver, base, env, rec, pts, guess_fn=None):
    """Solve from one polyline (or from guess_fn(solver, env), a ready-made initial guess in the
    solver's own format); return the metrics row (without identification)."""
    import numpy as np
    solver.n_nodes = solver.nodes_for(env)
    if guess_fn is not None:
        guess = guess_fn(solver, env)
    else:
        lo, hi = env.position_bounds()
        pts = np.clip(np.asarray(pts, float), lo, hi)
        pts[0], pts[-1] = env.start_pos, env.goal_pos
        guess = solver._polyline_guess(pts, env)
    # the guess as the solver sees it
    Xg = np.asarray(guess[0])
    t_g = np.asarray(solver.r) if hasattr(solver, "r") and len(solver.r) == len(Xg) else np.linspace(0, 1, len(Xg))
    if len(guess) == 4 and hasattr(solver, "positions"):
        Pg = solver.positions(Xg, guess[1], solver.c_seg, solver.c_B)
    else:
        Pg = Xg[:, :3]
    hn = solver.obstacle_eval(env, Pg)[0]
    inside = float(np.mean(hn.min(axis=1) < 0)) * 100 if hn.size else 0.0
    g_sigma = float(guess[-1])

    solver.time_limit = rec.get("time_limit")          # optional wall-clock cap per solve (None = off)
    res = solver._solve_from(env, guess)
    x = np.asarray(res.x, float)
    f_path = uniform(x[:, :3], np.linspace(0, 1, len(x)))
    ok = bool(res.success)
    return {
        "status": res.status, "success": ok, "iterations": int(res.iterations),
        "iter_feasible": first_feasible_iter(res.history, base.TOL_DEFECT, base.TOL_VIOLATION),
        "subproblem_failures": sum(h.get("event") == "subproblem_failed" for h in res.history),
        "solve_time": _f(res.solve_time),
        "cost": _f(res.cost) if ok else float("nan"),
        "flight_time": _f(res.flight_time) if ok else float("nan"),
        "cost_gap_pct": 100 * (_f(res.cost) / rec["bench_cost"] - 1) if ok else float("nan"),
        "flight_time_gap_pct": 100 * (_f(res.flight_time) / rec["bench_flight_time"] - 1) if ok else float("nan"),
        "guess_flight_time": g_sigma,
        "guess_time_gap_pct": 100 * (g_sigma / rec["bench_flight_time"] - 1),
        "guess_inside_pct": inside,
        "max_defect": _f(res.max_defect), "max_violation": _f(res.max_violation),
        "dense_min_margin": _f(res.dense_min_margin),
        "verify_defect": _f((getattr(res, "birkhoff", None) or {}).get("verify_defect")),
        "guess_path": np.round(uniform(Xg[:, :3], t_g), 1).tolist(),
        "x_rs": resample_nodes(x).tolist() if ok else None,         # full trajectory (world units),
        "u_rs": resample_nodes(res.u).tolist() if ok else None,     #   for use as a training label
        "final_path": np.round(f_path, 1).tolist(),
    }


def eval_task(task):
    """All guesses for one task's environments. Returns the part's rows."""
    from environment.envgen import Environment
    solver, base = get_solver(task["solver_method"], task["overrides"])
    rows, errors, t0 = [], [], time.perf_counter()
    for rec in task["envs"]:
        env = Environment.from_dict(rec["env"])
        ident = {"index": rec["index"], "fingerprint": rec["fp"], "split": rec["split"],
                 "bench_cost": rec["bench_cost"], "bench_flight_time": rec["bench_flight_time"]}
        for method, route, conf, pts, t_guess in guess_paths(env, rec, task["methods"], task["max_routes"]):
            row = {**ident, "method": method, "route": route, "confidence": _f(conf),
                   "guess_time": _f(t_guess)}
            if pts is None:
                row.update(status="no_guess", success=False)
                rows.append(row)
                continue
            try:
                row.update(evaluate_guess(solver, base, env, rec, pts))
                if not task.get("keep_traj"):
                    row.pop("x_rs", None)
                    row.pop("u_rs", None)
            except Exception:
                row.update(status="error", success=False)
                errors.append({"index": rec["index"], "method": method, "route": route,
                               "traceback": traceback.format_exc()})
            rows.append(row)
    return {"part": task["part"], "rows": rows, "errors": errors, "n_envs": len(task["envs"]),
            "time_s": round(time.perf_counter() - t0, 1)}


# --------------------------------------------------------------------------
# local side: benchmark, model inference, tasks, summary
# --------------------------------------------------------------------------
def datasets_root():
    return _HERE / "datasets"


def file_sha1(path):
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()[:12]


def read_benchmark(sol_root, splits, max_envs, tol_defect, tol_violation, v_cruise, sigma_bounds):
    """Solved environments of the chosen splits: environment, benchmark, and the stored
    cold-start attempts (as rows of method "cold")."""
    import h5py
    import numpy as np
    parts = sorted((sol_root / "parts").glob("*.h5"))
    if not parts:
        raise SystemExit(f"No solution parts in {sol_root / 'parts'}")
    recs, cold, n_unsolved = [], [], 0
    print(f"Reading the benchmark from {len(parts)} parts in {sol_root} ...", flush=True)
    for i, p in enumerate(parts, 1):
        with h5py.File(p, "r") as h5:
            if not h5.attrs.get("complete", False):
                continue
            for fp, g in h5.items():
                a = g.attrs
                if str(a.get("split", "")) not in splits:
                    continue
                if not bool(a["success"]):
                    n_unsolved += 1
                    continue
                bench = g["best/x_rs"][:, :3]
                rec = {"fp": fp, "index": int(a["index"]), "split": str(a["split"]),
                       "env": json.loads(g["env_json"][()]), "start_state": g["start_state"][:].tolist(),
                       "bench_cost": float(a["cost"]), "bench_flight_time": float(a["flight_time"]),
                       "bench_path": uniform(bench, np.linspace(0, 1, len(bench))).tolist()}
                rows = []
                for k in sorted(g["attempts"], key=int):
                    s = g["attempts"][k]
                    sa = s.attrs
                    ok = bool(sa["success"])
                    hist = json.loads(s["history"][()])
                    xr = s["x_rs"][:, :3]
                    fpath = uniform(xr, np.linspace(0, 1, len(xr)))
                    gpath = uniform(s["guess_p"][:], s["guess_t"][:])
                    g_sigma = float(np.clip(path_length(gpath) / v_cruise, *sigma_bounds))
                    rows.append({
                        "index": rec["index"], "fingerprint": fp, "split": rec["split"],
                        "bench_cost": rec["bench_cost"], "bench_flight_time": rec["bench_flight_time"],
                        "method": "cold", "route": int(k), "confidence": float("nan"), "guess_time": 0.0,
                        "status": str(sa["status"]), "success": ok, "iterations": int(sa["iterations"]),
                        "iter_feasible": first_feasible_iter(hist, tol_defect, tol_violation),
                        "subproblem_failures": sum(h.get("event") == "subproblem_failed" for h in hist),
                        "solve_time": float(sa["solve_time"]),
                        "cost": float(sa["cost"]) if ok else float("nan"),
                        "flight_time": float(sa["flight_time"]) if ok else float("nan"),
                        "cost_gap_pct": 100 * (float(sa["cost"]) / rec["bench_cost"] - 1) if ok else float("nan"),
                        "flight_time_gap_pct": (100 * (float(sa["flight_time"]) / rec["bench_flight_time"] - 1)
                                                if ok else float("nan")),
                        "guess_flight_time": g_sigma,
                        "guess_time_gap_pct": 100 * (g_sigma / rec["bench_flight_time"] - 1),
                        "guess_inside_pct": float("nan"),
                        "max_defect": float(sa["max_defect"]), "max_violation": float(sa["max_violation"]),
                        "dense_min_margin": float(sa["dense_min_margin"])})
                recs.append(rec)
                cold.append(rows)
        if i % 250 == 0:
            print(f"  {i}/{len(parts)} parts", flush=True)
    order = np.argsort([r["index"] for r in recs], kind="stable")
    recs = [recs[i] for i in order]
    cold = [cold[i] for i in order]
    if max_envs is not None:
        recs, cold = recs[:max_envs], cold[:max_envs]
    print(f"  {len(recs)} solved environments in split(s) {sorted(splits)} "
          f"({n_unsolved} unsolved ones skipped: no benchmark)")
    return recs, [r for rows in cold for r in rows]


def add_model_guesses(recs, model_path):
    """Run the network on every environment (CPU, here) and attach its routes."""
    import numpy as np
    import torch
    import train_warmstart as tw
    from environment.envgen import Environment
    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
    model, ck = tw.load_model(model_path, "cpu")
    print(f"Model {model_path.name} (epoch {ck.get('epoch')}): predicting routes for {len(recs)} "
          f"environments ...", flush=True)
    for r in recs:
        env = Environment.from_dict(r["env"])
        t0 = time.perf_counter()
        paths, sigma, conf = tw.predict(model, ck, env, np.asarray(r["start_state"]))
        r["model_time_s"] = time.perf_counter() - t0
        r["model_paths"] = np.round(paths, 2).tolist()
        r["model_sigma"] = sigma.tolist()
        r["model_conf"] = conf.tolist()


def prepare(dataset, run, model, name, split, methods, max_routes, max_envs, per_task):
    env_root = datasets_root() / dataset
    sol_root = env_root / "solutions" / run
    bm = json.loads((sol_root / "manifest.json").read_text())
    splits = {s.strip() for s in split.split(",") if s.strip()}
    methods = [m.strip() for m in methods.split(",") if m.strip()]
    bad = [m for m in methods if m not in ALL_METHODS]
    if bad:
        raise SystemExit(f"Unknown method(s) {bad}; choose from {ALL_METHODS}")
    model_path = _HERE / "models" / f"{model}.pt"
    if "model" in methods and not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")
    name = name or f"{model if 'model' in methods else 'baselines'}_{split.replace(',', '+')}"
    out_root = _HERE / "evaluations" / name
    s = bm["solver_settings"]
    tol_d, tol_v = s["scp_aircraft.TOL_DEFECT"], s["scp_aircraft.TOL_VIOLATION"]

    manifest = {"dataset": dataset, "benchmark_run": run, "solver_method": bm["method"],
                "solver_overrides": bm.get("solver_overrides", {}), "split": sorted(splits),
                "methods": methods, "max_routes": max_routes, "envs_per_task": per_task,
                "model": model if "model" in methods else None,
                "model_sha1": file_sha1(model_path) if "model" in methods else None,
                "near_opt_pct": NEAR_OPT_PCT, "n_pts": N_PTS,
                "benchmark_manifest_sha1": file_sha1(sol_root / "manifest.json"),
                "code_sha1": {f"{d}/{p.name}": file_sha1(p) for d in CODE_DIRS
                              for p in sorted((PROJECT / d).glob("*.*")) if p.suffix in (".py", ".json")},
                "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    old_path = out_root / "manifest.json"
    if old_path.exists():
        old = json.loads(old_path.read_text())
        keys = ("dataset", "benchmark_run", "split", "methods", "max_routes", "envs_per_task",
                "model_sha1", "benchmark_manifest_sha1")
        diff = [k for k in keys if old.get(k) != json.loads(json.dumps(manifest[k]))]
        if diff:
            raise SystemExit(f"Evaluation '{name}' already exists with different {', '.join(diff)}. "
                             f"Use --name for a new one (or delete {out_root}).")
        if old.get("code_sha1") != manifest["code_sha1"]:
            print("WARNING: solver/envgen files changed since this evaluation started.")
        manifest = {**old, "updated": manifest["created"]}
    if manifest["code_sha1"] != bm.get("code_sha1"):
        print("WARNING: solver/envgen files differ from the benchmark run's; iteration counts may "
              "not be comparable with the stored cold starts.")
    write_json_atomic(old_path, manifest)

    recs, cold_rows = read_benchmark(sol_root, splits, max_envs, tol_d, tol_v,
                                     s["scp_aircraft.V_CRUISE"], s["scp_aircraft.SIGMA_BOUNDS"])
    if not recs:
        raise SystemExit("No solved environments in that split.")
    write_json_atomic(out_root / "cold_rows.json", cold_rows)

    chunks = [recs[i:i + per_task] for i in range(0, len(recs), per_task)]
    todo = [(k, c) for k, c in enumerate(chunks)
            if not (out_root / "parts" / f"part_{k:05d}.json").exists()]
    if "model" in methods and todo:
        add_model_guesses([r for _, c in todo for r in c], model_path)
    tasks = [{"part": f"part_{k:05d}", "solver_method": bm["method"],
              "overrides": manifest["solver_overrides"], "methods": methods,
              "max_routes": max_routes, "envs": c} for k, c in todo]
    print(f"Evaluation '{name}': {len(recs)} environments, methods {methods}; "
          f"{len(chunks) - len(todo)} parts done, {len(tasks)} to run.")
    return out_root, tasks


def save_part(out_root, result):
    write_json_atomic(out_root / "parts" / f"{result['part']}.json", result)


def report(result, k, n, t0):
    print(f"  {result['part']}: {result['n_envs']} envs, {len(result['rows'])} solves, "
          f"{len(result['errors'])} errors, {result['time_s'] / 60:.1f} min   "
          f"[{k}/{n}, {(time.time() - t0) / 60:.1f} min elapsed]", flush=True)


ROW_COLUMNS = ("index", "fingerprint", "split", "method", "route", "confidence", "status", "success",
               "iterations", "iter_feasible", "subproblem_failures", "guess_time", "solve_time",
               "flight_time", "bench_flight_time", "flight_time_gap_pct", "cost", "bench_cost",
               "cost_gap_pct", "guess_flight_time", "guess_time_gap_pct", "guess_inside_pct",
               "max_defect", "max_violation", "dense_min_margin")


def collect(out_root):
    rows, errors = [], []
    for p in sorted((out_root / "parts").glob("part_*.json")):
        d = json.loads(p.read_text())
        rows += d["rows"]
        errors += d["errors"]
    done = {r["index"] for r in rows}
    cold_path = out_root / "cold_rows.json"
    cold = [r for r in json.loads(cold_path.read_text()) if r["index"] in done] if cold_path.exists() else []
    rows = cold + rows
    rows.sort(key=lambda r: (r["index"], r["method"], r["route"]))
    with open(out_root / "results.csv.tmp", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ROW_COLUMNS), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    os.replace(out_root / "results.csv.tmp", out_root / "results.csv")
    write_json_atomic(out_root / "errors.json", errors)
    return rows, errors


def strategies(rows):
    """Per-environment outcomes of each way of using a method's guesses."""
    import numpy as np
    by = {}
    for r in rows:
        by.setdefault(r["method"], {}).setdefault(r["index"], []).append(r)
    out = {}
    for m, envs in by.items():
        multi = any(len(v) > 1 for v in envs.values())
        variants = [("top-1" if multi else "", "top1")] + ([("in order until valid", "seq"),
                                                            ("best of all routes", "best")] if multi else [])
        for label, kind in variants:
            res = []
            for idx, rr in envs.items():
                rr = sorted(rr, key=lambda r: r["route"])
                ok = [r for r in rr if r.get("success")]
                if kind == "top1":
                    r0 = rr[0]
                    used, pick = [r0], (r0 if r0.get("success") else None)
                elif kind == "seq":
                    used = []
                    pick = None
                    for r in rr:
                        used.append(r)
                        if r.get("success"):
                            pick = r
                            break
                else:
                    used = rr
                    pick = min(ok, key=lambda r: r["flight_time"]) if ok else None
                it_feas = None
                if pick is not None:
                    before = sum(r.get("iterations", 0) or 0 for r in used if r is not pick) \
                        if kind == "seq" else 0
                    it_feas = before + (pick.get("iter_feasible") or pick.get("iterations", 0))
                res.append({
                    "success": pick is not None,
                    "iterations": sum(r.get("iterations", 0) or 0 for r in used),
                    "iter_feasible": it_feas,
                    "time": sum((r.get("solve_time") or 0) + (r.get("guess_time") or 0) for r in used),
                    "solves": len(used),
                    "time_gap_pct": pick["flight_time_gap_pct"] if pick else np.nan,
                    "guess_time_gap_pct": rr[0].get("guess_time_gap_pct", np.nan),
                    "guess_inside_pct": rr[0].get("guess_inside_pct", np.nan)})
            out[f"{m} {label}".strip()] = res
    return out


def summarize(rows, out_root, wall_s=None):
    import warnings
    import numpy as np
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    strat = strategies(rows)
    table = {}
    for name, res in strat.items():
        n = len(res)
        ok = [r for r in res if r["success"]]
        v = lambda key, rr=ok: np.array([r[key] for r in rr if r[key] is not None], float)
        gap, itf, it, tm = v("time_gap_pct"), v("iter_feasible"), v("iterations"), v("time")
        med = lambda a: float(np.nanmedian(a)) if len(a) else float("nan")
        table[name] = {
            "envs": n, "valid_pct": 100 * len(ok) / n if n else float("nan"),
            "solves_per_env": float(np.mean([r["solves"] for r in res])),
            "iter_feasible_median": med(itf), "iter_feasible_mean": float(np.mean(itf)) if len(itf) else float("nan"),
            "iterations_median": med(it), "iterations_mean": float(np.mean(it)) if len(it) else float("nan"),
            "time_median_s": med(tm),
            "flight_time_gap_median_pct": med(gap),
            "flight_time_gap_mean_pct": float(np.mean(gap)) if len(gap) else float("nan"),
            "flight_time_gap_p90_pct": float(np.percentile(gap, 90)) if len(gap) else float("nan"),
            "near_optimal_pct": 100 * float(np.sum(gap <= NEAR_OPT_PCT)) / n if n else float("nan"),
            "better_than_benchmark_pct": 100 * float(np.sum(gap < -0.1)) / n if n else float("nan"),
            "guess_time_gap_median_pct": med(np.array([r["guess_time_gap_pct"] for r in res], float)),
            "guess_inside_median_pct": med(np.array([r["guess_inside_pct"] for r in res], float)),
        }
    lines = [f"Warm-start evaluation: {out_root.name}   ({len({r['index'] for r in rows})} environments; "
             f"benchmark = best of the stored cold starts)", "",
             "Rates are % of all environments; iterations, time and flight-time gap are over the valid ones.",
             "iter->feas: iterations until the first node-feasible iterate (for 'in order', including the "
             "failed routes before it).", "",
             f"{'strategy':32s} {'valid':>6s} {'solves':>6s} {'iter->feas':>10s} {'iters':>6s} "
             f"{'time s':>7s} {'tf gap med':>10s} {'tf gap p90':>10s} {'<=' + str(NEAR_OPT_PCT) + '%':>6s} "
             f"{'faster':>6s} {'guess tf':>8s} {'inside':>6s}"]
    for name, t in table.items():
        lines.append(f"{name:32s} {t['valid_pct']:5.1f}% {t['solves_per_env']:6.1f} "
                     f"{t['iter_feasible_median']:10.1f} {t['iterations_median']:6.1f} "
                     f"{t['time_median_s']:7.1f} {t['flight_time_gap_median_pct']:9.2f}% "
                     f"{t['flight_time_gap_p90_pct']:9.2f}% "
                     f"{t['near_optimal_pct']:5.1f}% {t['better_than_benchmark_pct']:5.1f}% "
                     f"{t['guess_time_gap_median_pct']:+7.1f}% "
                     f"{t['guess_inside_median_pct']:5.1f}%")
    lines += ["", "valid: reached a trajectory that passes every check.  solves: solver runs per environment.",
              "iters / time: medians (time = solver + making the guess).",
              "tf gap: converged flight time above the benchmark's (median, 90th percentile).",
              f"<={NEAR_OPT_PCT}%: flight time within {NEAR_OPT_PCT}% of the benchmark.  faster: >0.1% faster than the benchmark.",
              "guess tf / inside: the (first) guess's flight time vs the benchmark's, and % of its "
              "check points inside an inflated obstacle (medians)."]
    statuses = {}
    for r in rows:
        statuses.setdefault(r["method"], {}).setdefault(r.get("status", "?"), 0)
        statuses[r["method"]][r.get("status", "?")] += 1
    lines += ["", "status counts per method (all routes):"]
    lines += [f"  {m}: " + ", ".join(f"{k} {v}" for k, v in sorted(c.items(), key=lambda kv: -kv[1]))
              for m, c in statuses.items()]
    if wall_s is not None:
        lines.append(f"\nwall-clock this run: {wall_s / 60:.1f} min")
    text = "\n".join(lines)
    (out_root / "summary.txt").write_text(text)
    write_json_atomic(out_root / "summary.json", {"strategies": table, "status_counts": statuses})
    print("\n" + text)
    print(f"\nResults: {out_root}")


def finish(out_root, t0):
    rows, errors = collect(out_root)
    if errors:
        print(f"{len(errors)} solve(s) crashed; see errors.json")
    summarize(rows, out_root, time.time() - t0)


def run_local(a):
    from multiprocessing import Pool
    out_root, tasks = prepare(a.dataset, a.run, a.model, a.name, a.split, a.methods,
                              a.max_routes, a.max_envs, a.per_task)
    t0 = time.time()
    if tasks:
        workers = min(a.workers or os.cpu_count() or 1, len(tasks))
        with Pool(workers) as pool:
            for k, result in enumerate(pool.imap_unordered(eval_task, tasks), 1):
                save_part(out_root, result)
                report(result, k, len(tasks), t0)
    finish(out_root, t0)


# --------------------------------------------------------------------------
# Modal
# --------------------------------------------------------------------------
try:
    import modal
except ImportError:
    modal = None

if modal is not None and not os.environ.get("FW_NO_MODAL_DEFS"):   # skipped when imported by another Modal app
    app = modal.App(MODAL_APP_NAME)
    image = (modal.Image.debian_slim(python_version=PYTHON_VERSION)
             .pip_install(*PIP_PACKAGES)
             .env({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                   "MKL_NUM_THREADS": "1", "RAYON_NUM_THREADS": "1"}))
    for _d in CODE_DIRS:
        image = image.add_local_dir(str(PROJECT / _d), f"/root/fixed-wing-rl/{_d}",
                                    ignore=["__pycache__", "*.pyc"])
    volume = modal.Volume.from_name(MODAL_VOLUME, create_if_missing=True) if MODAL_VOLUME else None

    @app.function(image=image, cpu=CPU_PER_TASK, memory=MEMORY_MB, timeout=TASK_TIMEOUT_S,
                  retries=TASK_RETRIES, max_containers=MAX_CONTAINERS,
                  volumes={VOLUME_MOUNT: volume} if volume is not None else {})
    def eval_task_remote(task, name):
        result = eval_task(task)
        if volume is not None:
            write_bytes_atomic(Path(VOLUME_MOUNT) / "evaluations" / name / "parts" / f"{task['part']}.json",
                               json.dumps(result, default=_f).encode())
            volume.commit()
        return result

    @app.local_entrypoint()
    def main(dataset: str = DATASET, run: str = RUN, model: str = MODEL, name: str = "",
             split: str = SPLIT, methods: str = METHODS, max_routes: int = -1, max_envs: int = -1,
             per_task: int = ENVS_PER_TASK):
        max_routes = None if max_routes < 0 else max_routes
        max_envs = None if max_envs < 0 else max_envs
        if max_envs is None and MAX_ENVS is not None:
            max_envs = MAX_ENVS
        if max_routes is None and MAX_ROUTES is not None:
            max_routes = MAX_ROUTES
        out_root, tasks = prepare(dataset, run, model, name or NAME, split, methods,
                                  max_routes, max_envs, per_task)
        t0, failed = time.time(), 0
        for k, result in enumerate(eval_task_remote.starmap([(t, out_root.name) for t in tasks],
                                                           order_outputs=False, return_exceptions=True), 1):
            if isinstance(result, BaseException):
                failed += 1
                print(f"  a task failed after retries: {result!r}", flush=True)
                continue
            save_part(out_root, result)
            report(result, k, len(tasks), t0)
        finish(out_root, t0)
        if failed:
            print(f"\n{failed} task(s) failed; rerun the same command to retry them.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evaluate warm starts by solver iterations and "
                                 "closeness to the benchmark optimum.")
    ap.add_argument("--local", action="store_true", help="run on this computer")
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--run", default=RUN)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--name", default=NAME)
    ap.add_argument("--split", default=SPLIT)
    ap.add_argument("--methods", default=METHODS)
    ap.add_argument("--max-routes", type=int, default=MAX_ROUTES)
    ap.add_argument("--max-envs", type=int, default=MAX_ENVS)
    ap.add_argument("--per-task", type=int, default=ENVS_PER_TASK)
    ap.add_argument("--workers", type=int, default=LOCAL_WORKERS)
    ap.add_argument("--summary-only", action="store_true", help="rebuild results.csv and the summary")
    a = ap.parse_args()
    if a.summary_only:
        root = _HERE / "evaluations" / (a.name or f"{a.model}_{a.split.replace(',', '+')}")
        rows, _ = collect(root)
        summarize(rows, root)
    elif not a.local:
        raise SystemExit("For Modal use:  modal run supervised-learning-main/eval_warmstart.py --model ...\n"
                         "For this computer add --local.")
    else:
        run_local(a)