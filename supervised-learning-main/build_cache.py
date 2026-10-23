"""
Build a training cache for the learned warm start from solved datasets.

Reads the HDF5 parts written by solve_envs.py, keeps only environments that were
solved (at least one valid start), and writes plain NumPy arrays that a training
script can load in seconds. Nothing is re-solved; envgen and the solvers are not
modified.

Every environment is expressed in its own NORMALIZED FRAME:
    origin   = start position
    yaw      = rotation about z so the goal lies along +x (z stays up; gravity unchanged)
    scale    = LENGTH_SCALE metres per unit (default: cruise turn radius, ~422 m)
so the network sees "start at the origin, goal straight ahead" for every problem.
`to_world()` below converts predicted paths back for the solver.

Output: supervised-learning-main/datasets/<DATASET>/cache/<RUN>.npz  (+ <RUN>.json summary)

    per environment (E rows)
        env_fp, env_index, split            identification ("train" / "val" / "test")
        origin (E,3), yaw (E,)              frame of each environment (world metres, radians)
        ctx (E, 12)                         goal_n (3), start V/V_cruise, gamma, chi_n,
                                            floor_n, ceiling_n, y_lo_n, y_hi_n, x_lo_n, x_hi_n
        tok, tok_off                        obstacle tokens, ragged: env e has tok[tok_off[e]:tok_off[e+1]]
    per label (L rows: every distinct valid trajectory)
        lab_env (L,)                        row of its environment
        lab_x (L, 64, 6)                    px, py, pz (normalized), V/V_cruise, gamma, chi_n
        lab_u (L, 64, 3)                    T/T_max, n_v, n_h
        lab_sigma (L,)                      flight time, s
        lab_sigma_n (L,)                    flight time / (straight-line distance / V_cruise)
        lab_cost (L,), lab_best (L,)        solver cost; True for the environment's best label
        lab_off (E+1,)                      env e's labels are rows lab_off[e]:lab_off[e+1]

Obstacle token (TOKEN_DIM = 17), one per convex piece (walls are split into their boxes):
    [ is_spheroid, is_cylinder, is_box, is_wall_piece,
      center (3), size (3), R[:,0] (3), R[:,2] (3), equivalent_radius ]
    all positions and sizes normalized; R is the piece's rotation in the normalized frame.
    Spheroid size = semi-axes (a1, a2, a_polar); cylinder size = (radius, radius, half-length),
    R[:,2] = axis; box size = half-extents. Upright pillars become vertical cylinders.

Run (from the fixed-wing-rl folder):
    python supervised-learning-main/build_cache.py --dataset c172_envs_v1 --run birkhoff_v1
"""

# ==========================================================================
# USER SETTINGS
# ==========================================================================
DATASET        = "c172_envs_v1"
RUN            = "birkhoff_v1"
DATASETS_DIR   = None          # None = supervised-learning-main/datasets
LENGTH_SCALE_M = None          # metres per normalized unit; None = cruise turn radius from the airframe
DEDUP_M        = 50.0          # two valid starts whose paths differ by less than this on average
                               #   (m) are the same route; only the lower-cost one is kept
MAX_LABELS     = 10            # most distinct routes kept per environment
WORKERS        = None          # processes for reading parts (None = all cores, 1 = no multiprocessing)

# ==========================================================================
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
PROJECT = _HERE.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

TOKEN_DIM = 17
CTX_DIM = 12


# --------------------------------------------------------------------------
# airframe constants used for normalization
# --------------------------------------------------------------------------
def airframe():
    p = json.loads((PROJECT / "aircraftmodel" / "c172_params.json").read_text())
    g = 9.81
    turn_radius = p["V_CRUISE"] ** 2 / (g * np.tan(np.radians(p["BANK_MAX_DEG"])))
    return {"V_CRUISE": p["V_CRUISE"], "T_MAX": p["T_MAX"], "turn_radius_m": float(turn_radius)}


# --------------------------------------------------------------------------
# normalized frame
# --------------------------------------------------------------------------
def frame_of(start, goal):
    """(origin, yaw): rotate about z by -yaw so the goal is along +x."""
    d = np.asarray(goal, float) - np.asarray(start, float)
    return np.asarray(start, float), float(np.arctan2(d[1], d[0]))


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def to_frame(p, origin, yaw, scale):
    """World positions (..., 3) -> normalized frame."""
    return (np.asarray(p, float) - origin) @ rot_z(-yaw).T / scale


def to_world(p_n, origin, yaw, scale):
    """Normalized positions (..., 3) -> world metres (the inverse of to_frame)."""
    return np.asarray(p_n, float) * scale @ rot_z(yaw).T + origin


def wrap_heading(chi, yaw):
    """Heading relative to the frame, kept continuous along a path."""
    return np.unwrap(np.asarray(chi, float) - yaw)


# --------------------------------------------------------------------------
# obstacle tokens
# --------------------------------------------------------------------------
def tokens_for(env, origin, yaw, scale):
    from environment.envgen import Box, Cylinder, OrientedCylinder, Spheroid
    Rf = rot_z(-yaw)
    rows = []
    for o in env.obstacles:
        wall = o.kind == "wall"
        for q in o.pieces():
            if isinstance(q, Spheroid):
                onehot, c, size, R = (1, 0, 0), q.center, q.radii, q.rotation
            elif isinstance(q, OrientedCylinder):
                onehot, c, size, R = (0, 1, 0), q.center, (q.radius, q.radius, q.half_length), q.rotation
            elif isinstance(q, Cylinder):                  # upright pillar -> vertical cylinder
                floor = env.bounds_lo[2]
                top = env.bounds_hi[2] if q.full_height else q.z_top
                c = np.r_[q.center_xy, 0.5 * (floor + top)]
                onehot, size, R = (0, 1, 0), (q.radius, q.radius, 0.5 * (top - floor)), np.eye(3)
            elif isinstance(q, Box):
                onehot, c, size, R = (0, 0, 1), q.center, q.half_extents, q.rotation
            else:
                raise ValueError(f"unknown obstacle piece {type(q).__name__}")
            Rn = Rf @ R
            a = Rn[:, 2] * (1.0 if Rn[2, 2] >= 0 else -1.0)  # an axis and its negative are the same shape
            rows.append(np.r_[onehot, float(wall), to_frame(c, origin, yaw, scale),
                              np.asarray(size, float) / scale, Rn[:, 0], a,
                              q.equivalent_radius() / scale if np.isfinite(q.equivalent_radius()) else 0.0])
    return np.array(rows, np.float32).reshape(-1, TOKEN_DIM)


def context_for(env, start_state, origin, yaw, scale, v_cruise):
    goal_n = to_frame(env.goal_pos, origin, yaw, scale)
    corners = np.array([[x, y, 0.0] for x in (env.bounds_lo[0], env.bounds_hi[0])
                        for y in (env.bounds_lo[1], env.bounds_hi[1])])
    cn = to_frame(corners, origin, yaw, scale)
    floor_n = (env.bounds_lo[2] - origin[2]) / scale
    ceil_n = (env.bounds_hi[2] - origin[2]) / scale
    V, gam, chi = start_state[3], start_state[4], start_state[5]
    chi_n = (chi - yaw + np.pi) % (2 * np.pi) - np.pi
    return np.r_[goal_n, V / v_cruise, gam, chi_n, floor_n, ceil_n,
                 cn[:, 1].min(), cn[:, 1].max(), cn[:, 0].min(), cn[:, 0].max()].astype(np.float32)


# --------------------------------------------------------------------------
# labels
# --------------------------------------------------------------------------
def distinct_routes(cands, scale, dedup_m, max_labels):
    """cands: list of (cost, x_rs, u_rs, sigma). Lowest cost first; drop near-duplicates."""
    cands = sorted(cands, key=lambda c: c[0])
    kept = []
    for c in cands:
        p = c[1][:, :3]
        if all(np.mean(np.linalg.norm(p - k[1][:, :3], axis=1)) * scale >= dedup_m for k in kept):
            kept.append(c)
        if len(kept) >= max_labels:
            break
    return kept


def read_part(args):
    """Everything needed from one .h5 part: a list of per-environment records."""
    path, scale, af, dedup_m, max_labels, splits = args
    import h5py
    from environment.envgen import Environment
    out, skipped = [], 0
    with h5py.File(path, "r") as h5:
        if not h5.attrs.get("complete", False):
            return out, 0
        for fp, g in h5.items():
            if not bool(g.attrs["success"]):
                skipped += 1
                continue
            env = Environment.from_dict(json.loads(g["env_json"][()]))
            start_state = g["start_state"][:]
            origin, yaw = frame_of(env.start_pos, env.goal_pos)
            cands = []
            for k, a in g["attempts"].items():
                if not bool(a.attrs["success"]):
                    continue
                cands.append((float(a.attrs["cost"]), a["x_rs"][:], a["u_rs"][:], float(a.attrs["flight_time"])))
            if not cands:                                   # (best is always one of the attempts)
                cands = [(float(g.attrs["cost"]), g["best/x_rs"][:], g["best/u_rs"][:],
                          float(g.attrs["flight_time"]))]
            routes = distinct_routes(
                [(c, _norm_x(x, origin, yaw, scale, af), _norm_u(u, af), s) for c, x, u, s in cands],
                scale, dedup_m, max_labels)
            dist = np.linalg.norm(env.goal_pos - env.start_pos)
            idx = int(g.attrs["index"])
            out.append({
                "fp": fp, "index": idx, "split": splits.get(idx, str(g.attrs.get("split", ""))),
                "origin": origin, "yaw": yaw,
                "ctx": context_for(env, start_state, origin, yaw, scale, af["V_CRUISE"]),
                "tok": tokens_for(env, origin, yaw, scale),
                "n_valid": len(cands),
                "x": np.stack([r[1] for r in routes]).astype(np.float32),
                "u": np.stack([r[2] for r in routes]).astype(np.float32),
                "sigma": np.array([r[3] for r in routes], np.float32),
                "sigma_n": np.array([r[3] / (dist / af["V_CRUISE"]) for r in routes], np.float32),
                "cost": np.array([r[0] for r in routes], np.float32)})
    return out, skipped


def _norm_x(x, origin, yaw, scale, af):
    x = np.asarray(x, float)
    return np.column_stack([to_frame(x[:, :3], origin, yaw, scale), x[:, 3] / af["V_CRUISE"],
                            x[:, 4], wrap_heading(x[:, 5], yaw)])


def _norm_u(u, af):
    u = np.asarray(u, float)
    return np.column_stack([u[:, 0] / af["T_MAX"], u[:, 1], u[:, 2]])


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def datasets_root():
    return Path(DATASETS_DIR) if DATASETS_DIR else _HERE / "datasets"


def load_splits(env_root):
    p = env_root / "index.csv"
    if not p.exists():
        return {}
    with open(p, newline="") as f:
        return {int(r["index"]): r.get("split", "") for r in csv.DictReader(f)}


def build(dataset, run, workers=None):
    t0 = time.time()
    env_root = datasets_root() / dataset
    sol_root = env_root / "solutions" / run
    parts = sorted((sol_root / "parts").glob("*.h5"))
    if not parts:
        raise SystemExit(f"No solution parts in {sol_root / 'parts'}")
    af = airframe()
    scale = float(LENGTH_SCALE_M or af["turn_radius_m"])
    splits = load_splits(env_root)
    jobs = [(str(p), scale, af, DEDUP_M, MAX_LABELS, splits) for p in parts]
    print(f"Reading {len(parts)} parts from {sol_root} (length scale {scale:.1f} m)")

    recs, skipped = [], 0
    workers = workers if workers is not None else (os.cpu_count() or 1)
    if workers > 1:
        from multiprocessing import Pool
        with Pool(min(workers, len(jobs))) as pool:
            for k, (r, s) in enumerate(pool.imap(read_part, jobs, chunksize=8), 1):
                recs += r; skipped += s
                if k % 200 == 0:
                    print(f"  {k}/{len(jobs)} parts", flush=True)
    else:
        for k, j in enumerate(jobs, 1):
            r, s = read_part(j)
            recs += r; skipped += s
    recs.sort(key=lambda r: r["index"])
    if not recs:
        raise SystemExit("No solved environments found.")

    E = len(recs)
    n_tok = np.array([len(r["tok"]) for r in recs])
    n_lab = np.array([len(r["x"]) for r in recs])
    arrays = {
        "env_fp": np.array([r["fp"] for r in recs]),
        "env_index": np.array([r["index"] for r in recs], np.int64),
        "split": np.array([r["split"] for r in recs]),
        "origin": np.stack([r["origin"] for r in recs]).astype(np.float64),
        "yaw": np.array([r["yaw"] for r in recs], np.float64),
        "ctx": np.stack([r["ctx"] for r in recs]),
        "tok": np.concatenate([r["tok"] for r in recs]),
        "tok_off": np.r_[0, np.cumsum(n_tok)].astype(np.int64),
        "n_valid_starts": np.array([r["n_valid"] for r in recs], np.int64),
        "lab_env": np.repeat(np.arange(E), n_lab).astype(np.int64),
        "lab_x": np.concatenate([r["x"] for r in recs]),
        "lab_u": np.concatenate([r["u"] for r in recs]),
        "lab_sigma": np.concatenate([r["sigma"] for r in recs]),
        "lab_sigma_n": np.concatenate([r["sigma_n"] for r in recs]),
        "lab_cost": np.concatenate([r["cost"] for r in recs]),
        "lab_best": np.concatenate([np.r_[True, np.zeros(len(r["x"]) - 1, bool)] for r in recs]),
        "lab_off": np.r_[0, np.cumsum(n_lab)].astype(np.int64),
    }
    out_dir = env_root / "cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{run}.npz"
    tmp = out.with_name(out.stem + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, out)

    splits_count = {str(s): int(np.sum(arrays["split"] == s)) for s in sorted(set(arrays["split"]))}
    summary = {
        "dataset": dataset, "run": run, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "length_scale_m": scale, "v_cruise": af["V_CRUISE"], "t_max": af["T_MAX"],
        "dedup_m": DEDUP_M, "max_labels": MAX_LABELS, "token_dim": TOKEN_DIM, "ctx_dim": CTX_DIM,
        "environments": E, "unsolved_skipped": int(skipped), "labels": int(n_lab.sum()),
        "splits": splits_count,
        "valid_starts_per_env": _hist(arrays["n_valid_starts"]),
        "distinct_routes_per_env": _hist(n_lab),
        "tokens_per_env": {"min": int(n_tok.min()), "median": float(np.median(n_tok)), "max": int(n_tok.max())},
    }
    (out_dir / f"{run}.json").write_text(json.dumps(summary, indent=1))
    print(f"\nCache: {out}  ({out.stat().st_size / 1e6:.1f} MB, {time.time() - t0:.0f} s)")
    print(f"  solved environments: {E}   (unsolved skipped: {skipped})")
    print(f"  labels (distinct valid routes): {int(n_lab.sum())}   splits: {splits_count}")
    print(f"  valid starts per env:     {summary['valid_starts_per_env']}")
    print(f"  distinct routes per env:  {summary['distinct_routes_per_env']}")
    print(f"  obstacle tokens per env:  {summary['tokens_per_env']}")
    return out


def _hist(v):
    vals, counts = np.unique(v, return_counts=True)
    return {int(a): int(b) for a, b in zip(vals, counts)}


def load_cache(path):
    """Load a cache as a dict of arrays, plus its summary under "summary"."""
    path = Path(path)
    d = dict(np.load(path, allow_pickle=False))
    d["summary"] = json.loads(path.with_suffix(".json").read_text())
    return d


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build the warm-start training cache from solved datasets.")
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--run", default=RUN)
    ap.add_argument("--workers", type=int, default=WORKERS)
    a = ap.parse_args()
    build(a.dataset, a.run, a.workers)