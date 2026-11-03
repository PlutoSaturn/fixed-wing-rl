"""
Deep reinforcement learning pilot: an agent that FLIES the aircraft through the obstacle
field, step by step, and whose flown trajectory is the solver's warm start.

The simulator (vectorized in PyTorch, thousands of flights at once on one GPU)
    Same 3-DOF point-mass model and limits as the solvers (aircraftmodel/c172_params.json):
        V' = (T - D)/m - g sin(gamma),  gamma' = g (n_v - cos gamma)/V,  chi' = g n_h /(V cos gamma)
        D = a1 V^2 + a2 (n_v^2 + n_h^2) / V^2
    integrated with RK4 every DT seconds. The FLIGHT ENVELOPE IS ENFORCED: whatever the agent asks
    for is mapped into what the aircraft can do this step:
        n_v, n_h change at most N_V_RATE_MAX, N_H_RATE_MAX per second (the agent commands the change)
        |n_h| <= tan(bank max) n_v,  sqrt(n_v^2 + n_h^2) <= min(N_MAX, c_stall V^2)   (load and stall)
        n_v is limited so |gamma| never exceeds GAMMA_MAX next step
        0 <= T <= available thrust, and T is limited so V never exceeds V_MAX
    Speed below V_MIN (not enough thrust to hold a climb) ends the flight as a stall.
    Obstacles are the same convex pieces as envgen, inflated by the planning clearance (36 m);
    entering one, or leaving the box, ends the flight as a crash.

What the agent senses
    its speed, climb angle, current controls, height above the floor and below the ceiling,
    the direction / distance / elevation of the goal, the elapsed time, and
    "lidar": N_RAYS rays looking forward (a horizontal fan of +-FAN_DEG and two slightly raised /
        lowered fans of +-45 deg), each giving the exact distance along its own direction to the
        first inflated obstacle or box wall it hits, up to RAY_RANGE_M, plus
    a coarse map of the whole environment: occupancy of MAP_NX x MAP_NY x 3 cells (500 m x 500 m x
        200 m), with its own position and the goal marked, so it can choose a corridor early.

It also sees the projected miss at the goal plane (lateral and vertical, in MISS_SCALE_M), so it
can line up its arrival early, including climbing or descending in time.

What it controls each step (ACTION_MODE = "attitude"): thrust (share of what is available), a target
bank angle (+-BANK_MAX) and a target climb angle (+-GAMMA_MAX). Like an autopilot, the simulator moves
n_v and n_h toward what those targets need, as fast as the rate limits allow, and the envelope limits
still apply on top. Holding a turn is one steady action.

Departure and arrival areas are kept clear: obstacle pieces within CLEAR_RADIUS_M of the start or the
goal are removed, both in the simulator and in the environments the solver evaluates (so the solver
and the agent always see the same obstacles; the benchmark flight times were solved with them, so the
flight-time gap can be slightly optimistic for the affected environments).

Reward each step
    + PROG_W  * (progress toward the goal / straight-line distance)
    - TIME_W  * (dt / straight-line flight time)               (fly fast)
    - PROX_W  * dt * (how deep it is inside a PROX_M soft zone around the inflated obstacles)
    - TTC_W   * dt * (1 - time to impact / TTC_S)   when heading at an obstacle (central rays) with
                less than TTC_S seconds to impact: turn away early, not at the last moment
    - SMOOTH_W * dt * (commanded change of n_v and n_h)^2      (no porpoising / weaving)
    + AIM_K * (improvement of min(1, projected miss / MISS_SCALE_M))   within APPROACH_M of the goal
                (potential-based shaping: 0 ... +AIM_K per flight), where the projected miss is how
                far the current velocity would pass from the goal when it crosses the goal plane
                (the plane through the goal across the start-goal line)
    and at the end: + SUCCESS_R + PRECISION_R * (1 - aim error / GOAL_RADIUS_M) for arriving (coming
                    within GOAL_RADIUS_M), where the aim error is how close its line of flight passes
                    to the goal centre: dead centre earns the full bonus,
                    - MISS_R * min(1, miss / MISS_SCALE_M) for crossing the goal plane without arriving
                    (the flight ends there, so a near miss costs less than a wide one),
                    - CRASH_R for an obstacle / the box / a stall, - TIMEOUT_R for running out of time.

Learning: PPO (clipped, GAE), N_ENVS flights in parallel, separate actor and critic networks.
Multi-GPU: the N_ENVS flights are split across N_GPUS GPUs; each simulates its share and computes
gradients on its share of every minibatch with its own copy of the agent; the gradients are
averaged, one optimizer step is taken, and the copies are re-synchronized. The result is the
same update as on one GPU, computed in parallel.
Curriculum, in two parts; each step is taken when the success rate (last 10 updates) on the
current setting exceeds CURRICULUM_SUCCESS:
    1. obstacle thinning on the least cluttered CURRICULUM_START share of the training environments:
       every flight keeps a random KEEP_STAGES share of its environment's obstacles (a new subset
       each flight), starting with none - open sky, to learn to fly the envelope to the goal fast
    2. then all obstacles, adding more cluttered environments (by occupancy) CURRICULUM_STEP at a time.
The lidar, collisions and the coarse map always reflect the obstacles the flight actually has.
Solver evaluations always use the full environments.
Progress is judged on the agent's real flying: every CHECK_EVERY updates it flies CHECK_FLIGHTS
flights of the current stage with its mean action (no exploration noise); that success rate decides
the curriculum steps and plateaus. (The training flights' own success rate includes the noise.)
Against stagnation: exploration noise has a floor (LOG_STD_MIN) and a small entropy bonus; at a
plateau (no better noise-free success for PLATEAU_UPDATES updates) the noise is raised back to its
starting level, but only if it has collapsed below BOOST_BELOW of it. After MAX_BOOSTS plateaus in
a row at the last stage, training stops.

Evaluation against the solver (in the background, while training continues)
    Every EVAL_EVERY updates the current agent flies VAL_ENVS validation environments
    (deterministically). Each flight becomes a warm start: a successful flight gives the solver
    its full states and controls (re-integrated so they are consistent with the solver's own
    discretization), a failed one gives the flown path plus a line to the goal. The solves run on
    up to MAX_CONTAINERS Modal containers, with the benchmark run's solver settings, and are scored
    like train_rl.py: valid = -W_ITER * iterations - W_GAP * flight-time gap (%), failures below any
    valid result and graded by how close the solver got. The best agent by this score is kept.
    The first evaluation also scores the straight line, the baseline to beat.
    At the end the best agent is scored on the untouched test split (full-state warm start, path
    only, and the straight line, side by side).

Output on the Volume at /rl_pilot/<NAME>/ and copied to supervised-learning-main/rl_pilot/<NAME>/:
    updates.csv  per update: flight success / crash / stall / timeout %, return, flight time vs
                 the benchmark, curriculum pool, noise, losses, speed
    evals.csv    solver evaluations (valid %, iterations, flight-time gap, score)
    best.pt, checkpoint.pt, test_summary.json

Run (from the fixed-wing-rl folder):
    modal run supervised-learning-main/rl_pilot.py --name pilot_smoke --updates 6 --n-envs 64 --eval-every 3 --val-envs 20 --test-envs 20
    modal run --detach supervised-learning-main/rl_pilot.py --name pilot_v1
    modal run supervised-learning-main/rl_pilot.py --name pilot_v1 --download-only
Rerunning the same command resumes from the last checkpoint.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ["FW_NO_MODAL_DEFS"] = "1"

# ==========================================================================
# USER SETTINGS
# ==========================================================================
DATASET        = "c172_envs_v1"
RUN            = "birkhoff_v1"       # benchmark solutions (flight times) and solver settings
NAME           = "pilot_v1"

# simulator
DT             = 0.5                 # s per agent step
TIME_LIMIT_X   = 2.0                 # flight ends (timeout) after this x the straight-line time
GOAL_RADIUS_M  = 100.0
ACTION_MODE    = "attitude"          # "attitude": the agent commands target bank and climb angles (an autopilot
                                     #   moves n_v, n_h toward them within the rate limits); "rate": the agent
                                     #   commands the change of n_v, n_h directly (earlier runs)
TAU_GAMMA_S    = 3.0                 # attitude mode: climb-angle error is closed with this time constant
CLEAR_RADIUS_M = 500.0               # obstacle pieces within this distance of the start or the goal are
                                     #   removed, in the simulator AND in the solver evaluation (0 = keep all)
N_RAYS_FAN     = 15                  # horizontal fan, -FAN_DEG ... +FAN_DEG around the heading
FAN_DEG        = 75.0                #   (forward field of view only)
N_RAYS_TILT    = 7                   # raised and lowered fans (+-TILT_DEG), -45 ... +45 deg
TILT_DEG       = 8.0
RAY_RANGE_M    = 1500.0
MAP_NX, MAP_NY = 26, 10              # coarse map cells (500 m) along / across the start-goal line
MAP_X0, MAP_Y0 = -1000.0, -2500.0    # map corner in the flight frame, m
MAP_CELL       = 500.0

# reward
PROG_W         = 10.0
TIME_W         = 4.0
PROX_W         = 0.2
PROX_M         = 60.0
TTC_W          = 0.5                 # per second while heading at an obstacle with < TTC_S to impact
TTC_S          = 10.0                #   (time to impact = distance along the central rays / speed)
SMOOTH_W       = 0.05                # per second, on the squared change of the bank / climb targets per step
SUCCESS_R      = 10.0                # reaching the goal (within GOAL_RADIUS_M) ...
PRECISION_R    = 10.0                #   ... plus up to this much more, the closer to its centre
APPROACH_M     = 5000.0              # within this distance of the goal, aiming is shaped: each step earns
AIM_K          = 5.0                 #   AIM_K * (reduction in min(1, projected miss / MISS_SCALE_M)); over a
MISS_SCALE_M   = 500.0               #   flight this sums to 0 ... +AIM_K, so it can never reward giving up
MISS_R         = 10.0                # passing the goal plane without arriving: up to this penalty, graded
                                     #   by how far it missed (MISS_SCALE_M or more = the full penalty)
CRASH_R        = 25.0                # crash, leaving the box or stalling: always the worst outcome
TIMEOUT_R      = 15.0
# outcome order: arrive (+10 ... +20) > miss (0 ... -10) > timeout (-15) > crash / stall (-25)

# PPO
N_ENVS         = 1024                # flights in parallel
ROLLOUT        = 128                 # steps per flight per update
UPDATES        = 3000
EPOCHS         = 4
MINIBATCH      = 8192
LR             = 3e-4
GAMMA_RL       = 0.995
GAE_LAMBDA     = 0.95
CLIP           = 0.2
ENT_COEF       = 0.001               # small: lets the noise shrink once the agent knows what to do
VF_COEF        = 0.5
HIDDEN         = 256
LOG_STD_INIT   = -0.5                # exploration noise on the actions (in [-1, 1] units)
LOG_STD_MIN    = -2.5
SEED           = 0

# curriculum and stagnation
KEEP_STAGES        = (0.0, 0.25, 0.5, 0.75, 1.0)   # obstacle thinning first: each flight keeps this random
                                     #   share of its environment's obstacles (0 = open sky: learn to fly)
CURRICULUM_START   = 0.3             # least cluttered share of the training envs to start with
CURRICULUM_STEP    = 0.1
CURRICULUM_SUCCESS = 0.7             # NOISE-FREE success rate on the current stage needed to move on
CHECK_EVERY        = 10              # updates between noise-free check flights on the current stage
CHECK_FLIGHTS      = 256             # flights per check (the last 2 checks are averaged)
PLATEAU_UPDATES    = 100             # updates without a better noise-free success rate = a plateau
BOOST_BELOW        = 0.5             # at a plateau, boost exploration only if the noise has fallen below this
                                     #   share of its starting level (otherwise it is not what is missing)
MAX_BOOSTS         = 3               # plateaus in a row at the last stage -> stop

# evaluation with the solver
EVAL_EVERY     = 100                 # updates between solver evaluations
VAL_ENVS       = 200
TEST_ENVS      = None                # None = all
W_ITER, W_GAP, GAP_CLIP, W_CLOSE, D_MAX = 0.1, 0.5, 20.0, 1.0, 10.0
CHECKPOINT_EVERY = 10

# Modal
MODAL_APP_NAME = "fixed-wing-pilot"
MODAL_VOLUME   = "fixed-wing-rl-data"
GPU            = "H100"
N_GPUS         = 1                   # GPUs in the training container (Modal allows up to 8). The flights are
                                     #   split across them, each GPU holds a copy of the agent, and their
                                     #   gradients are averaged into one update (same learning, faster).
LOOP_TIMEOUT_S = 24 * 3600
MAX_CONTAINERS = 1000
SOLVES_PER_CALL = 2
TASK_TIMEOUT_S = 1800
PYTHON_VERSION = "3.12"
SOLVER_PACKAGES = ["numpy", "scipy", "cvxpy", "clarabel"]

# ==========================================================================
import csv
import io
import json
import math
import sys
import time
from functools import lru_cache
from pathlib import Path

_HERE = Path(__file__).resolve().parent
PROJECT = _HERE.parent if (_HERE.parent / "solvers").is_dir() else Path("/root/fixed-wing-rl")
for _p in (str(PROJECT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

VOLUME_MOUNT = "/data"
CODE_DIRS = ("environment", "solvers", "aircraftmodel")
_CFG_KEYS = ("dataset", "run", "n_envs", "seed")
N_RAYS = N_RAYS_FAN + 2 * N_RAYS_TILT
OBS_VERSION = 3                      # 1: old lidar; 2: forward exact lidar; 3: + projected miss at the goal
OBS_DIM = 15 + N_RAYS
STATUS = {0: "flying", 1: "goal", 2: "crash", 3: "out_of_bounds", 4: "stall", 5: "timeout", 6: "missed"}


def log(msg):
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


def write_bytes_atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def append_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    old = []
    if path.exists():
        with open(path, newline="") as f:
            old = list(csv.DictReader(f))
    allrows = old + rows
    cols = list(dict.fromkeys(c for r in allrows for c in r))
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, restval="")
        w.writeheader()
        w.writerows(allrows)
    os.replace(tmp, path)


def airframe(project=PROJECT):
    p = json.loads((Path(project) / "aircraftmodel" / "c172_params.json").read_text())
    g = 9.81
    a = {k: p[k] for k in ("MASS", "WING_AREA", "RHO", "C_D0", "K_INDUCED", "C_L_MAX", "T_MAX",
                           "THRUST_SLOPE", "THRUST_PLAN_FRACTION", "V_MIN", "V_MAX", "V_CRUISE",
                           "N_MAX", "BANK_MAX_DEG", "GAMMA_MAX_DEG", "N_H_RATE_MAX", "N_V_RATE_MAX")}
    a["G"] = g
    a["a1"] = 0.5 * a["RHO"] * a["WING_AREA"] * a["C_D0"]
    a["a2"] = 2.0 * a["K_INDUCED"] * (a["MASS"] * g) ** 2 / (a["RHO"] * a["WING_AREA"])
    a["c_stall"] = a["RHO"] * a["WING_AREA"] * a["C_L_MAX"] / (2.0 * a["MASS"] * g)
    return a


# --------------------------------------------------------------------------
# solver-side: score and warm starts (CPU containers)
# --------------------------------------------------------------------------
def score(row, consts):
    """Same score as train_rl.py."""
    worst_valid = -(W_ITER * consts["max_iters"] + W_GAP * GAP_CLIP)
    if row.get("success"):
        gap = max(-GAP_CLIP, min(GAP_CLIP, row["flight_time_gap_pct"]))
        return -W_ITER * row["iterations"] - W_GAP * gap
    f = lambda k: row.get(k) if isinstance(row.get(k), (int, float)) and math.isfinite(row.get(k)) else None
    if row.get("status") == "error":
        d = D_MAX
    else:
        d = 0.0
        d += math.log10(max(f("max_defect") / consts["tol_defect"], 1.0)) if f("max_defect") is not None else D_MAX
        d += (max(f("max_violation") - consts["tol_violation"], 0.0) / consts["clearance"]
              if f("max_violation") is not None else D_MAX)
        if f("dense_min_margin") is not None:
            d += max(-f("dense_min_margin") - consts["dense_tol"], 0.0) / consts["dense_tol"]
        if f("verify_defect") is not None:
            d += math.log10(max(f("verify_defect") / consts["verify_tol"], 1.0))
        d = min(d, D_MAX)
    return worst_valid - 1.0 - W_CLOSE * d


def trajectory_guess(t, X, U):
    """guess_fn for eval_warmstart.evaluate_guess: a flown trajectory (world frame) sampled at the
    solver's node times, its derivatives taken from the dynamics, re-integrated with the solver's
    own matrices and bent linearly to end exactly at the goal."""
    import numpy as np

    def make(solver, env):
        t_ = np.asarray(t, float)
        Xf, Uf = np.asarray(X, float), np.asarray(U, float)
        Xf[:, 5] = np.unwrap(Xf[:, 5])
        sigma = float(t_[-1])
        r = np.asarray(solver.r) * sigma
        Xn = np.stack([np.interp(r, t_, Xf[:, k]) for k in range(6)], axis=1)
        Un = np.stack([np.interp(r, t_, Uf[:, k]) for k in range(3)], axis=1)
        x0, p_goal = solver.boundary(env)
        Xn[0] = x0
        Xn[0, 5] = Xf[0, 5] + np.angle(np.exp(1j * (x0[5] - Xf[0, 5])))
        V = sigma * solver.model.f(Xn, Un)                       # dX/dtau, tau in [0, 1]
        Xg = solver.integrate_segments(Xn[0], V)
        V[:, :3] += p_goal - Xg[-1, :3]                          # integrates to a linear ramp in tau
        Xg = solver.integrate_segments(Xn[0], V)
        return Xg, V, Un, sigma
    return make


@lru_cache(maxsize=16)
def _shard(dataset, shard):
    path = Path(VOLUME_MOUNT) / "datasets" / dataset / "envs" / f"shard_{shard:05d}.json"
    return {int(d.get("metrics", {}).get("dataset", {}).get("index", -1)): d
            for d in json.loads(path.read_text())}


def solve_item(item):
    import traceback
    import numpy as np
    import eval_warmstart as ew
    from environment.envgen import Environment
    t0 = time.perf_counter()
    try:
        solver, base = ew.get_solver(item["solver_method"], item["overrides"])
        env = Environment.from_dict(_shard(item["dataset"], item["shard"])[item["index"]])
        if item.get("drop_pieces"):                           # same cleared start / goal areas as the simulator
            gone = set(item["drop_pieces"])
            kept = [q for i, q in enumerate(Environment.pieces(env)) if i not in gone]
            env.pieces = lambda kept=kept: kept
        if item["kind"] == "full":
            row = ew.evaluate_guess(solver, base, env, item, None,
                                    guess_fn=trajectory_guess(item["t"], item["X"], item["U"]))
        elif item["kind"] == "path":
            row = ew.evaluate_guess(solver, base, env, item, np.asarray(item["path"]))
        else:                                                    # straight line
            row = ew.evaluate_guess(solver, base, env, item, np.stack([env.start_pos, env.goal_pos]))
        for k in ("guess_path", "final_path", "x_rs", "u_rs"):
            row.pop(k, None)
    except Exception:
        row = {"status": "error", "success": False, "traceback": traceback.format_exc()[-2000:]}
    row.update({"index": item["index"], "kind": item["kind"], "wall_s": time.perf_counter() - t0})
    return row


def solve_items(items):
    return [solve_item(it) for it in items]


# --------------------------------------------------------------------------
# GPU side: simulator, agent, PPO
# --------------------------------------------------------------------------
def build_world(cache, scale, bounds_lo, bounds_hi, vehicle_radius, clearance, device):
    """Everything the simulator needs, as tensors: obstacle tokens in metres (flight frame:
    start at the origin, goal along +x, z up relative to the start), frames, goals, box."""
    import numpy as np
    import torch
    E = len(cache["env_index"])
    n_tok = np.diff(cache["tok_off"])
    T = int(n_tok.max())
    tok = np.zeros((E, T, 17), np.float32)
    mask = np.ones((E, T), bool)
    for e in range(E):
        t0, t1 = cache["tok_off"][e], cache["tok_off"][e + 1]
        tk = cache["tok"][t0:t1].copy()
        tk[:, 4:10] *= scale                                   # centre and size -> metres
        tk[:, 16] *= scale
        tok[e, :t1 - t0] = tk
        mask[e, :t1 - t0] = False
    goal = cache["ctx"][:, :3].astype(np.float64) * scale
    tt = lambda a, dt=torch.float32: torch.as_tensor(np.asarray(a), dtype=dt, device=device)
    W = {"tok": tt(tok), "mask": tt(mask, torch.bool), "goal": tt(goal),
         "origin": tt(cache["origin"]), "yaw": tt(cache["yaw"]),
         "lo": tt(np.asarray(bounds_lo) + vehicle_radius), "hi": tt(np.asarray(bounds_hi) - vehicle_radius),
         "clearance": float(clearance), "E": E}
    W["removed"] = clear_start_goal(W, CLEAR_RADIUS_M)          # departure / arrival areas kept clear
    return W


def piece_h(p, tok, d):
    """Clearance-like value (m) of points p (N, M, 3) from every inflated piece tok (N, T, 17): (N, M, T)."""
    import torch
    c, s = tok[..., 4:7], tok[..., 7:10].clamp_min(1e-3)
    r0, a = tok[..., 10:13], tok[..., 13:16]
    R = torch.stack([r0, torch.cross(a, r0, dim=-1), a], -1)
    sph, cyl, box = tok[..., 0], tok[..., 1], tok[..., 2]
    smin = s.min(-1).values
    q = torch.einsum("nmi,ntij->nmtj", p, R) - torch.einsum("nti,ntij->ntj", c, R)[:, None]
    h_sph = ((q / s[:, None]).norm(dim=-1) - (1 + d / smin[:, None])) * smin[:, None]
    h_cyl = torch.maximum(q[..., :2].norm(dim=-1) - (s[..., 0] + d)[:, None],
                          q[..., 2].abs() - (s[..., 2] + d)[:, None])
    h_box = (q.abs() - (s + d)[:, None]).max(-1).values
    return sph[:, None] * h_sph + cyl[:, None] * h_cyl + box[:, None] * h_box


def clear_start_goal(W, radius):
    """Remove (mask) the pieces within `radius` of each environment's start (flight-frame origin) or goal.
    Returns, per environment, the list of removed piece indices (same order as env.pieces())."""
    import torch
    removed = [[] for _ in range(W["E"])]
    if radius <= 0:
        return removed
    for e0 in range(0, W["E"], 256):
        ids = torch.arange(e0, min(e0 + 256, W["E"]), device=W["tok"].device)
        pts = torch.stack([torch.zeros_like(W["goal"][ids]), W["goal"][ids]], 1)       # (n, 2, 3)
        h = piece_h(pts, W["tok"][ids], W["clearance"]).min(1).values                  # (n, T)
        near = (h < radius) & ~W["mask"][ids]
        W["mask"][ids] |= near
        for k, row in enumerate(near.cpu()):
            removed[e0 + k] = torch.nonzero(row).squeeze(1).tolist()
    return removed


def sdf(p, env_ids, W, chunk=None, drop=None):
    """Distance-like clearance (m) of points p (N, M, 3, flight frame) from the inflated obstacles and
    the box walls of each flight's environment; negative inside. Conservative (never larger than
    the true distance), which makes it safe for sphere tracing."""
    import torch
    tok, mask = W["tok"][env_ids], W["mask"][env_ids]                # (N, T, 17), (N, T)
    if drop is not None:
        mask = mask | drop                                           # thinned-out obstacles are ignored
    d = W["clearance"]
    c, s = tok[..., 4:7], tok[..., 7:10].clamp_min(1e-3)
    r0, a = tok[..., 10:13], tok[..., 13:16]
    R = torch.stack([r0, torch.cross(a, r0, dim=-1), a], -1)         # (N, T, 3, 3)
    sph, cyl, box = tok[..., 0], tok[..., 1], tok[..., 2]
    smin = s.min(-1).values
    out = []
    M = p.shape[1]
    step = chunk or M
    for m0 in range(0, M, step):
        pp = p[:, m0:m0 + step]
        h = piece_h(pp, tok, d).masked_fill(mask[:, None], 1e9).min(-1).values   # (N, m)
        # box walls (world frame)
        cy, sy = torch.cos(W["yaw"][env_ids]), torch.sin(W["yaw"][env_ids])
        xw = pp[..., 0] * cy[:, None] - pp[..., 1] * sy[:, None] + W["origin"][env_ids, None, 0]
        yw = pp[..., 0] * sy[:, None] + pp[..., 1] * cy[:, None] + W["origin"][env_ids, None, 1]
        zw = pp[..., 2] + W["origin"][env_ids, None, 2]
        pw = torch.stack([xw, yw, zw], -1)
        hb = torch.minimum(pw - W["lo"], W["hi"] - pw).min(-1).values
        out.append(torch.minimum(h, hb))
    return torch.cat(out, 1)


def ray_dirs(chi, device):
    """(N, N_RAYS, 3) unit ray directions around the heading chi (level frame)."""
    import torch
    az = [math.radians(a) for a in torch.linspace(-FAN_DEG, FAN_DEG, N_RAYS_FAN).tolist()]
    azt = [math.radians(a) for a in torch.linspace(-45, 45, N_RAYS_TILT).tolist()]
    el = math.radians(TILT_DEG)
    dirs = [(a, 0.0) for a in az] + [(a, el) for a in azt] + [(a, -el) for a in azt]
    A = torch.tensor([d[0] for d in dirs], device=device)
    Eel = torch.tensor([d[1] for d in dirs], device=device)
    ang = chi[:, None] + A[None]
    v = torch.stack([torch.cos(Eel) * torch.cos(ang), torch.cos(Eel) * torch.sin(ang),
                     torch.sin(Eel).expand_as(ang)], -1)
    return v


def ray_hits(pos, dirs, env_ids, W, drop=None):
    """Exact distance (m) along each ray to the first inflated obstacle piece or box wall it hits.
    pos (N, 3), dirs (N, R, 3) unit, flight frame. 0 if the ray starts inside an obstacle; RAY_RANGE_M
    if nothing is hit within range. Same shapes as envgen/sdf(): inflated spheroids are ellipsoids
    with semi-axes s (1 + d / s_min); cylinders are finite with flat caps (radius r + d, half-length
    L + d); boxes have half-extents s + d."""
    import torch
    big = torch.tensor(1e9, device=pos.device)
    tok, mask = W["tok"][env_ids], W["mask"][env_ids]
    if drop is not None:
        mask = mask | drop
    d = W["clearance"]
    c, s = tok[..., 4:7], tok[..., 7:10].clamp_min(1e-3)
    r0, a = tok[..., 10:13], tok[..., 13:16]
    R = torch.stack([r0, torch.cross(a, r0, dim=-1), a], -1)                    # (N, T, 3, 3)
    o = torch.einsum("ni,ntij->ntj", pos, R) - torch.einsum("nti,ntij->ntj", c, R)   # (N, T, 3) local origin
    u = torch.einsum("nri,ntij->nrtj", dirs, R)                                 # (N, R, T, 3) local direction
    o = o[:, None].expand_as(u)
    eps = 1e-9

    def interval_quadratic(qa, qb, qc):
        """[t0, t1] where qa t^2 + qb t + qc <= 0 (qa >= 0); empty -> (big, -big)."""
        disc = qb * qb - 4 * qa * qc
        ok = disc >= 0
        sq = torch.sqrt(torch.clamp(disc, min=0.0))
        qa_ = torch.clamp(qa, min=eps)
        t0 = (-qb - sq) / (2 * qa_)
        t1 = (-qb + sq) / (2 * qa_)
        flat = qa < eps                                              # ray parallel: inside everywhere or nowhere
        t0 = torch.where(flat, torch.where(qc <= 0, -big, big), t0)
        t1 = torch.where(flat, torch.where(qc <= 0, big, -big), t1)
        return torch.where(ok, t0, big), torch.where(ok, t1, -big)

    def slab(oo, uu, h):
        """Per-axis slab interval for |oo + t uu| <= h, combined over the given axes."""
        inv = 1.0 / torch.where(uu.abs() < eps, torch.full_like(uu, eps), uu)
        ta, tb = (-h - oo) * inv, (h - oo) * inv
        lo_, hi_ = torch.minimum(ta, tb), torch.maximum(ta, tb)
        par = uu.abs() < eps
        inside = oo.abs() <= h
        lo_ = torch.where(par, torch.where(inside, -big, big), lo_)
        hi_ = torch.where(par, torch.where(inside, big, -big), hi_)
        return lo_.max(-1).values, hi_.min(-1).values

    S = s[:, None].expand_as(u)
    # boxes
    b0, b1 = slab(o, u, S + d)
    # cylinders: radial quadratic in local x, y and a slab in z
    rr = (S[..., 0] + d)
    c0, c1 = interval_quadratic(u[..., 0] ** 2 + u[..., 1] ** 2, 2 * (o[..., 0] * u[..., 0] + o[..., 1] * u[..., 1]),
                                o[..., 0] ** 2 + o[..., 1] ** 2 - rr ** 2)
    z0, z1 = slab(o[..., 2:], u[..., 2:], S[..., 2:] + d)
    c0, c1 = torch.maximum(c0, z0), torch.minimum(c1, z1)
    # spheroids: unit sphere after scaling by the inflated semi-axes
    A = S * (1 + d / S.min(-1, keepdim=True).values)
    oe, ue = o / A, u / A
    e0, e1 = interval_quadratic((ue * ue).sum(-1), 2 * (oe * ue).sum(-1), (oe * oe).sum(-1) - 1)
    kind = tok[..., 0:3][:, None].expand(*u.shape[:3], 3)
    t0 = kind[..., 0] * e0 + kind[..., 1] * c0 + kind[..., 2] * b0
    t1 = kind[..., 0] * e1 + kind[..., 1] * c1 + kind[..., 2] * b1
    hit = (t1 >= t0) & (t1 >= 0)
    t_hit = torch.where(hit, torch.clamp(t0, min=0.0), big)
    t_hit = t_hit.masked_fill(mask[:, None], 1e9).min(-1).values                  # (N, R)
    # box walls (world frame)
    cy, sy = torch.cos(W["yaw"][env_ids]), torch.sin(W["yaw"][env_ids])
    rot = lambda v: torch.stack([v[..., 0] * cy[:, None] - v[..., 1] * sy[:, None],
                                 v[..., 0] * sy[:, None] + v[..., 1] * cy[:, None], v[..., 2]], -1)
    pw = rot(pos[:, None]) + W["origin"][env_ids, None]                         # (N, 1, 3)
    dw = rot(dirs)
    inv = 1.0 / torch.where(dw.abs() < eps, torch.full_like(dw, eps), dw)
    tw = torch.where(dw > 0, (W["hi"] - pw) * inv, (W["lo"] - pw) * inv)
    tw = torch.where(dw.abs() < eps, big, tw).clamp(min=0.0).min(-1).values
    return torch.clamp(torch.minimum(t_hit, tw), max=RAY_RANGE_M)


def cast_rays(pos, chi, env_ids, W, drop=None):
    dirs = ray_dirs(chi, pos.device)
    return ray_hits(pos, dirs, env_ids, W, drop) / RAY_RANGE_M


def build_maps(W, device, batch=16, ids_all=None, drop=None, progress=False):
    """Coarse occupancy (len(ids), 3, MAP_NX, MAP_NY): fraction of 8 sample points per cell that are
    inside an inflated obstacle or outside the box. Layers = world altitude bands of 1/3 height.
    ids_all: environment ids (default: all); drop: (len(ids), T) obstacles to leave out."""
    import torch
    if ids_all is None:
        ids_all = torch.arange(W["E"], device=device)
    xs = MAP_X0 + MAP_CELL * (torch.arange(MAP_NX, device=device) + 0.5)
    ys = MAP_Y0 + MAP_CELL * (torch.arange(MAP_NY, device=device) + 0.5)
    off = torch.tensor([-0.25, 0.25], device=device) * MAP_CELL
    H = (W["hi"][2] - W["lo"][2]).item()
    zoff = torch.tensor([-1 / 12, 1 / 12], device=device) * H
    maps = torch.zeros((len(ids_all), 3, MAP_NX, MAP_NY), device=device)
    for e0 in range(0, len(ids_all), batch):
        ids = ids_all[e0:e0 + batch]
        pts = []
        for layer in range(3):
            zc = W["lo"][2] + H * (layer + 0.5) / 3
            for dx in off:
                for dy in off:
                    for dz in zoff:
                        X, Y = torch.meshgrid(xs + dx, ys + dy, indexing="ij")
                        Z = torch.full_like(X, float(zc + dz))
                        pts.append(torch.stack([X, Y, Z], -1).reshape(-1, 3))
        P = torch.stack(pts, 0).reshape(3, 8, MAP_NX * MAP_NY, 3)       # world z
        P = P.reshape(-1, 3)[None].expand(len(ids), -1, -1).clone()
        P[..., 2] = P[..., 2] - W["origin"][ids, None, 2]               # -> flight frame z
        dr = drop[e0:e0 + batch] if drop is not None else None
        occ = (sdf(P, ids, W, chunk=2048, drop=dr) < 0).float().reshape(len(ids), 3, 8, MAP_NX, MAP_NY).mean(2)
        maps[e0:e0 + len(ids)] = occ
        if progress and (e0 // batch) % 50 == 0:
            log(f"  maps: {e0 + len(ids)}/{len(ids_all)}")
    return maps


class Sim:
    """N parallel flights. State per flight: x (6: px py pz V gamma chi, flight frame), controls
    (T, n_v, n_h), time, environment id."""

    def __init__(self, W, maps, bench_tf, af, n, device, seed=0):
        import torch
        self.W, self.maps, self.af, self.n, self.dev = W, maps, af, n, device
        self.bench_tf = bench_tf
        self.gen = torch.Generator(device="cpu").manual_seed(seed)
        self.x = torch.zeros((n, 6), device=device)
        self.u = torch.zeros((n, 3), device=device)
        self.t = torch.zeros(n, device=device)
        self.env = torch.zeros(n, dtype=torch.long, device=device)
        self.d0 = torch.ones(n, device=device)
        self.tref = torch.ones(n, device=device)
        self.prev_d = torch.ones(n, device=device)
        self.prev_phi = torch.full((n,), -AIM_K, device=device)
        self.prev_act = torch.zeros((n, 3), device=device)
        self.pool = None
        self.keep_frac = 1.0                                  # share of obstacles each new flight keeps
        T = W["tok"].shape[1]
        self.drop = torch.zeros((n, T), dtype=torch.bool, device=device)
        self.mapocc = torch.zeros((n, 3, MAP_NX, MAP_NY), device=device)
        self.rays = None

    # ---- aircraft model ----
    def T_avail(self, V):
        a = self.af
        return a["THRUST_PLAN_FRACTION"] * (a["T_MAX"] + a["THRUST_SLOPE"] * V)

    def f(self, x, u):
        import torch
        a = self.af
        V, ga, ch = x[:, 3], x[:, 4], x[:, 5]
        T, nv, nh = u[:, 0], u[:, 1], u[:, 2]
        cg, sg = torch.cos(ga), torch.sin(ga)
        D = a["a1"] * V ** 2 + a["a2"] * (nv ** 2 + nh ** 2) / V ** 2
        return torch.stack([V * cg * torch.cos(ch), V * cg * torch.sin(ch), V * sg,
                            (T - D) / a["MASS"] - a["G"] * sg,
                            a["G"] * (nv - cg) / V, a["G"] * nh / (V * cg)], -1)

    def apply_envelope(self, act):
        """Agent action in [-1, 1]^3 -> controls the aircraft can actually fly this step."""
        import torch
        a = self.af
        V, ga = self.x[:, 3], self.x[:, 4]
        gmax = math.radians(a["GAMMA_MAX_DEG"])
        tanb = math.tan(math.radians(a["BANK_MAX_DEG"]))
        g = a["G"]
        n_cap = torch.clamp(a["c_stall"] * V ** 2, max=a["N_MAX"])
        if ACTION_MODE == "attitude":
            # autopilot: climb-angle target -> n_v that closes the error in TAU_GAMMA_S; bank target -> n_h
            g_cmd = act[:, 1] * gmax
            phi_cmd = act[:, 2] * math.radians(a["BANK_MAX_DEG"])
            nv_des = torch.cos(ga) + (g_cmd - ga) * V / (g * TAU_GAMMA_S)
            dv = a["N_V_RATE_MAX"] * DT
            nv = self.u[:, 1] + torch.clamp(nv_des - self.u[:, 1], -dv, dv)
            nh_des = torch.tan(phi_cmd) * nv
            dh = a["N_H_RATE_MAX"] * DT
            nh = self.u[:, 2] + torch.clamp(nh_des - self.u[:, 2], -dh, dh)
        else:
            nv = self.u[:, 1] + act[:, 1] * a["N_V_RATE_MAX"] * DT
            nh = self.u[:, 2] + act[:, 2] * a["N_H_RATE_MAX"] * DT
        nv_hi = torch.cos(ga) + (gmax - ga) * V / (g * DT)              # keeps gamma <= GAMMA_MAX
        nv_lo = torch.cos(ga) + (-gmax - ga) * V / (g * DT)             # keeps gamma >= -GAMMA_MAX
        nv = torch.minimum(torch.maximum(nv, torch.clamp(nv_lo, min=0.0)), torch.minimum(nv_hi, n_cap))
        nv = torch.clamp(nv, min=0.0)
        nh_lim = torch.minimum(tanb * nv, torch.sqrt(torch.clamp(n_cap ** 2 - nv ** 2, min=0.0)))
        nh = torch.maximum(torch.minimum(nh, nh_lim), -nh_lim)
        D = a["a1"] * V ** 2 + a["a2"] * (nv ** 2 + nh ** 2) / V ** 2
        T_hi = torch.minimum(self.T_avail(V),
                             D + a["MASS"] * g * torch.sin(ga) + a["MASS"] * (a["V_MAX"] - V) / DT)
        T_hi = torch.clamp(T_hi, min=0.0)
        T = (act[:, 0] + 1) / 2 * T_hi
        return torch.stack([T, nv, nh], -1)

    # ---- episodes ----
    def reset(self, idx, env_ids):
        import torch
        a = self.af
        self.env[idx] = env_ids
        g = self.W["goal"][env_ids]
        self.d0[idx] = g.norm(dim=-1)
        self.tref[idx] = self.d0[idx] / a["V_CRUISE"]
        self.prev_d[idx] = self.d0[idx]
        self.prev_phi[idx] = -AIM_K
        self.prev_act[idx] = 0.0
        x = torch.zeros((len(idx), 6), device=self.dev)
        x[:, 3] = a["V_CRUISE"]
        self.x[idx] = x
        D = a["a1"] * a["V_CRUISE"] ** 2 + a["a2"] / a["V_CRUISE"] ** 2
        self.u[idx] = torch.tensor([D, 1.0, 0.0], device=self.dev)
        self.t[idx] = 0.0
        if self.keep_frac >= 1.0:
            self.drop[idx] = False
            self.mapocc[idx] = self.maps[env_ids]
        else:
            r = torch.rand((len(idx), self.drop.shape[1]), generator=self.gen).to(self.dev)
            self.drop[idx] = r >= self.keep_frac
            self.mapocc[idx] = build_maps(self.W, self.dev, batch=64, ids_all=env_ids, drop=self.drop[idx])

    def sample_envs(self, k):
        import torch
        j = torch.randint(len(self.pool), (k,), generator=self.gen)
        return self.pool[j].to(self.dev)

    def obs(self):
        import torch
        a = self.af
        x, u = self.x, self.u
        g = self.W["goal"][self.env]
        rel = g - x[:, :3]
        dh = rel[:, :2].norm(dim=-1).clamp_min(1.0)
        bearing = torch.atan2(rel[:, 1], rel[:, 0]) - x[:, 5]
        gmax = math.radians(a["GAMMA_MAX_DEG"])
        zw = x[:, 2] + self.W["origin"][self.env, 2]
        H = (self.W["hi"][2] - self.W["lo"][2])
        self.rays = cast_rays(x[:, :3], x[:, 5], self.env, self.W, drop=self.drop)
        v = torch.stack([
            (x[:, 3] - a["V_CRUISE"]) / (a["V_MAX"] - a["V_MIN"]), x[:, 4] / gmax,
            torch.sin(bearing), torch.cos(bearing),
            torch.clamp(torch.atan2(rel[:, 2], dh) / gmax, -3, 3),
            rel.norm(dim=-1) / self.d0, (zw - self.W["lo"][2]) / H, (self.W["hi"][2] - zw) / H,
            u[:, 0] / self.T_avail(x[:, 3]), u[:, 1] - 1.0, u[:, 2] / math.tan(math.radians(a["BANK_MAX_DEG"])),
            torch.sin(x[:, 5]), self.t / self.tref], -1)
        lat, vert, _ = self.projected_miss(x, g)
        obs = torch.cat([v, torch.stack([lat, vert], -1).clamp(-3, 3) / 1.0, self.rays], -1)
        # map with own position and goal marked
        m = torch.zeros((self.n, 5, MAP_NX, MAP_NY), device=self.dev)
        m[:, :3] = self.mapocc
        ar = torch.arange(self.n, device=self.dev)
        for ch, p in ((3, x[:, :3]), (4, g)):
            ix = ((p[:, 0] - MAP_X0) / MAP_CELL).long().clamp(0, MAP_NX - 1)
            iy = ((p[:, 1] - MAP_Y0) / MAP_CELL).long().clamp(0, MAP_NY - 1)
            m[ar, ch, ix, iy] = 1.0
        return obs, m

    def smooth_term(self, act):
        """attitude mode: squared change of the climb / bank targets since the last step;
        rate mode: squared commanded change of n_v, n_h."""
        if ACTION_MODE == "attitude":
            d = act[:, 1:] - self.prev_act[:, 1:]
            self.prev_act = act.detach().clone()
            return (d ** 2).sum(-1)
        return act[:, 1] ** 2 + act[:, 2] ** 2

    def aim_shaping(self, dist, miss):
        """Potential-based: phi = -AIM_K * min(1, miss / scale) inside APPROACH_M, -AIM_K outside;
        the reward is phi(new) - phi(old), so a whole flight earns phi(end) + AIM_K in [0, AIM_K]."""
        import torch
        phi = torch.where(dist < APPROACH_M, -AIM_K * torch.clamp(miss / MISS_SCALE_M, max=1.0),
                          torch.full_like(dist, -AIM_K))
        r = phi - self.prev_phi
        self.prev_phi = phi
        return r

    @staticmethod
    def projected_miss(x, g):
        """Where the current velocity would cross the goal plane (x = goal x in the flight frame):
        (lateral, vertical) offsets from the goal in MISS_SCALE_M, and the miss distance (m).
        Flying away from the plane counts as a full miss."""
        import torch
        V, ga, ch = x[:, 3], x[:, 4], x[:, 5]
        vel = torch.stack([V * torch.cos(ga) * torch.cos(ch), V * torch.cos(ga) * torch.sin(ch), V * torch.sin(ga)], -1)
        ahead = vel[:, 0] > 1.0
        tc = (g[:, 0] - x[:, 0]) / vel[:, 0].clamp_min(1.0)
        p = x[:, :3] + vel * tc.clamp(min=0.0)[:, None]
        lat = (p[:, 1] - g[:, 1]) / MISS_SCALE_M
        vert = (p[:, 2] - g[:, 2]) / MISS_SCALE_M
        miss = torch.where(ahead, torch.hypot(p[:, 1] - g[:, 1], p[:, 2] - g[:, 2]),
                           torch.full_like(lat, 10 * MISS_SCALE_M))
        lat = torch.where(ahead, lat, torch.full_like(lat, 3.0))
        vert = torch.where(ahead, vert, torch.zeros_like(vert))
        return lat, vert, miss

    def step(self, act):
        """-> reward (N,), done (N,) bool, status (N,) int, controls used (N, 3)."""
        import torch
        a = self.af
        u = self.apply_envelope(act)
        x = self.x
        k1 = self.f(x, u); k2 = self.f(x + 0.5 * DT * k1, u)
        k3 = self.f(x + 0.5 * DT * k2, u); k4 = self.f(x + DT * k3, u)
        mid = x + 0.5 * DT * k2
        xn = x + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        gmax = math.radians(a["GAMMA_MAX_DEG"])
        xn[:, 4] = xn[:, 4].clamp(-gmax, gmax)
        xn[:, 3] = torch.clamp(xn[:, 3], max=a["V_MAX"])
        self.x, self.u, self.t = xn, u, self.t + DT
        h = sdf(torch.stack([mid[:, :3], xn[:, :3]], 1), self.env, self.W, drop=self.drop)   # (N, 2)
        h_min = h.min(1).values
        g = self.W["goal"][self.env]
        dist = (g - xn[:, :3]).norm(dim=-1)
        # closest approach to the goal during this step (segment x -> xn), for arrival and precision
        seg = xn[:, :3] - x[:, :3]
        s_ = (((g - x[:, :3]) * seg).sum(-1) / (seg * seg).sum(-1).clamp_min(1e-6)).clamp(0, 1)
        d_close = (x[:, :3] + s_[:, None] * seg - g).norm(dim=-1)
        # time to impact along the central horizontal rays (from the lidar seen this step)
        c0 = N_RAYS_FAN // 2
        ahead = self.rays[:, max(0, c0 - 1):c0 + 2].min(-1).values * RAY_RANGE_M if self.rays is not None \
            else torch.full_like(dist, RAY_RANGE_M)
        ttc = ahead / x[:, 3].clamp_min(1.0)
        _, _, miss_now = self.projected_miss(xn, g)
        r = PROG_W * (self.prev_d - dist) / self.d0 - TIME_W * DT / self.tref \
            - PROX_W * DT * torch.clamp(1 - h_min / PROX_M, 0, 1) \
            - TTC_W * DT * torch.clamp(1 - ttc / TTC_S, 0, 1) \
            - SMOOTH_W * DT * self.smooth_term(act) \
            + self.aim_shaping(dist, miss_now)
        self.prev_d = dist
        # precision: how close the current line of flight passes to the goal centre
        Vn, gan, chn = xn[:, 3], xn[:, 4], xn[:, 5]
        vhat = torch.stack([torch.cos(gan) * torch.cos(chn), torch.cos(gan) * torch.sin(chn), torch.sin(gan)], -1)
        dv = g - xn[:, :3]
        perp = (dv - (dv * vhat).sum(-1, keepdim=True) * vhat).norm(dim=-1)
        crossed = (x[:, 0] < g[:, 0]) & (xn[:, 0] >= g[:, 0])                # passed the goal plane this step
        miss_at_plane = torch.hypot(xn[:, 1] - g[:, 1], xn[:, 2] - g[:, 2])
        status = torch.zeros(self.n, dtype=torch.long, device=self.dev)
        status = torch.where(self.t > TIME_LIMIT_X * self.tref, torch.full_like(status, 5), status)
        status = torch.where(xn[:, 3] < a["V_MIN"], torch.full_like(status, 4), status)
        cy, sy = torch.cos(self.W["yaw"][self.env]), torch.sin(self.W["yaw"][self.env])
        pw = torch.stack([xn[:, 0] * cy - xn[:, 1] * sy, xn[:, 0] * sy + xn[:, 1] * cy, xn[:, 2]], -1) \
            + self.W["origin"][self.env]
        out = ((pw < self.W["lo"]) | (pw > self.W["hi"])).any(-1)
        status = torch.where(h_min < 0, torch.full_like(status, 2), status)
        status = torch.where(out, torch.full_like(status, 3), status)
        status = torch.where(crossed & (status == 0), torch.full_like(status, 6), status)
        status = torch.where(d_close < GOAL_RADIUS_M, torch.full_like(status, 1), status)
        r = r + torch.where(status == 1, SUCCESS_R + PRECISION_R * torch.clamp(1 - perp / GOAL_RADIUS_M, 0, 1), 0.0) \
            - torch.where((status >= 2) & (status <= 4), CRASH_R, 0.0) \
            - torch.where(status == 5, TIMEOUT_R, 0.0) \
            - torch.where(status == 6, MISS_R * torch.clamp(miss_at_plane / MISS_SCALE_M, max=1.0), 0.0)
        self.last_close = d_close
        return r, status > 0, status, u


def make_agent(device):
    import torch
    import torch.nn as nn

    class MapEnc(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Conv2d(5, 16, 3, padding=1), nn.ReLU(),
                                     nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(), nn.Flatten())
            n = 32 * ((MAP_NX + 1) // 2) * ((MAP_NY + 1) // 2)
            self.fc = nn.Sequential(nn.Linear(n, 128), nn.ReLU())

        def forward(self, m):
            return self.fc(self.net(m))

    def trunk(out):
        return nn.Sequential(nn.Linear(OBS_DIM + 128, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, HIDDEN), nn.Tanh(),
                             nn.Linear(HIDDEN, out))

    class Agent(nn.Module):
        def __init__(self):
            super().__init__()
            self.map_a, self.map_c = MapEnc(), MapEnc()
            self.actor, self.critic = trunk(3), trunk(1)
            nn.init.zeros_(self.actor[-1].weight)
            with torch.no_grad():
                self.actor[-1].bias.copy_(torch.tensor([0.4, 0.0, 0.0]))   # start near trim thrust, no turn
            self.log_std = nn.Parameter(torch.full((3,), LOG_STD_INIT))

        def mean(self, o, m):
            return self.actor(torch.cat([o, self.map_a(m)], -1))

        def value(self, o, m):
            return self.critic(torch.cat([o, self.map_c(m)], -1)).squeeze(-1)

    return Agent().to(device)


def fly(agent, sim_template, W, maps, bench_tf, af, env_rows, device, max_mult=TIME_LIMIT_X, keep_frac=1.0, seed=0):
    """Fly the agent's mean action on the given environments; record every step (flight frame)."""
    import numpy as np
    import torch
    n = len(env_rows)
    sim = Sim(W, maps, bench_tf, af, n, device, seed=seed)
    sim.keep_frac = keep_frac
    ids = torch.as_tensor(np.asarray(env_rows), device=device)
    sim.reset(torch.arange(n, device=device), ids)
    done = torch.zeros(n, dtype=torch.bool, device=device)
    status = torch.zeros(n, dtype=torch.long, device=device)
    rec_x, rec_u, rec_t = [sim.x.clone()], [sim.u.clone()], [sim.t.clone()]
    steps = int(max_mult * sim.tref.max().item() / DT) + 2
    with torch.no_grad():
        for _ in range(steps):
            o, m = sim.obs()
            act = agent.mean(o, m).clamp(-1, 1)
            x_prev, u_prev, t_prev = sim.x.clone(), sim.u.clone(), sim.t.clone()
            _, d, s, _ = sim.step(act)
            # finished flights stay frozen
            sim.x = torch.where(done[:, None], x_prev, sim.x)
            sim.u = torch.where(done[:, None], u_prev, sim.u)
            sim.t = torch.where(done, t_prev, sim.t)
            status = torch.where(~done & d, s, status)
            done = done | d
            rec_x.append(sim.x.clone()); rec_u.append(sim.u.clone()); rec_t.append(sim.t.clone())
            if done.all():
                break
    X = torch.stack(rec_x, 1).cpu().numpy()
    U = torch.stack(rec_u, 1).cpu().numpy()
    Tm = torch.stack(rec_t, 1).cpu().numpy()
    out = []
    for i in range(n):
        k = int(np.searchsorted(Tm[i], Tm[i, -1])) + 1                 # last distinct time
        out.append({"status": STATUS[int(status[i])], "t": Tm[i, :k], "X": X[i, :k], "U": U[i, :k]})
    return out


def to_world(fl, origin, yaw, goal_w):
    """Flight-frame trajectory -> world frame; the controls at node k are the ones flown after it."""
    import numpy as np
    c, s = np.cos(yaw), np.sin(yaw)
    X = fl["X"].astype(np.float64).copy()
    p = X[:, :3]
    pw = np.stack([p[:, 0] * c - p[:, 1] * s, p[:, 0] * s + p[:, 1] * c, p[:, 2]], 1) + origin
    X[:, :3] = pw
    X[:, 5] = X[:, 5] + yaw
    U = fl["U"].astype(np.float64).copy()
    U[:-1] = U[1:]                                                       # u stored after each step
    t = fl["t"].astype(np.float64)
    return t, X, U


# --------------------------------------------------------------------------
# training (one long GPU container)
# --------------------------------------------------------------------------
def run(cfg, submit_solves, commit):
    import numpy as np
    import torch
    import build_cache as bc
    V = Path(VOLUME_MOUNT)
    ds, run_name, name = cfg["dataset"], cfg["run"], cfg["name"]
    root = V / "rl_pilot" / name
    root.mkdir(parents=True, exist_ok=True)
    cpath = root / "config.json"
    if cpath.exists():
        old = json.loads(cpath.read_text())
        diff = [k for k in _CFG_KEYS if old.get(k) != cfg.get(k)]
        if diff:
            raise SystemExit(f"Run '{name}' exists with different {', '.join(diff)}; use a new --name.")
    cpath.write_text(json.dumps(cfg, indent=1))
    if cfg.get("devices"):
        devices = list(cfg["devices"])                      # (testing: e.g. ["cpu", "cpu"])
    elif torch.cuda.is_available():
        devices = [f"cuda:{i}" for i in range(min(torch.cuda.device_count(), cfg.get("n_gpus") or 1))]
    else:
        devices = ["cpu"]
    device = devices[0]
    G = len(devices)
    torch.manual_seed(cfg["seed"])
    log(f"started on {G} device(s): {[torch.cuda.get_device_name(i) for i in range(G)] if device.startswith('cuda') else devices}")
    log("loading the training cache from the Volume ...")

    cache = bc.load_cache(V / "datasets" / ds / "cache" / f"{run_name}.npz")
    scale = cache["summary"]["length_scale_m"]
    first = cache["lab_off"][:-1]
    bench_tf, bench_cost = cache["lab_sigma"][first].astype(np.float64), cache["lab_cost"][first].astype(np.float64)
    with open(V / "datasets" / ds / "index.csv", newline="") as f:
        meta = {int(r["index"]): r for r in csv.DictReader(f)}
    shard_of = {i: int(r["shard"]) for i, r in meta.items()}
    occ = np.array([float(meta[int(i)].get("occupancy", 0) or 0) for i in cache["env_index"]])
    bm = json.loads((V / "datasets" / ds / "solutions" / run_name / "manifest.json").read_text())
    s = bm["solver_settings"]
    e0 = int(cache["env_index"][0])
    d0 = _shard(ds, shard_of[e0])[e0]
    consts = {"tol_defect": s["scp_aircraft.TOL_DEFECT"], "tol_violation": s["scp_aircraft.TOL_VIOLATION"],
              "max_iters": s["scp_aircraft.MAX_ITERS"], "verify_tol": s.get("scp_birkhoff.VERIFY_DEFECT_TOL") or 2e-3,
              "dense_tol": s.get("scp_aircraft.DENSE_TOLERANCE") or 0.25 * d0["margin"],
              "clearance": d0["vehicle_radius"] + d0["margin"]}
    af = airframe()
    log("building the simulator world (obstacles of every environment on the GPU) ...")
    W = build_world(cache, scale, d0["bounds_lo"], d0["bounds_hi"], d0["vehicle_radius"],
                    consts["clearance"], device)
    log(f"{W['E']} environments, up to {W['tok'].shape[1]} obstacle pieces each, devices {devices}")
    n_rm = [len(r) for r in W["removed"]]
    log(f"cleared start / goal areas ({CLEAR_RADIUS_M:.0f} m): removed {sum(n_rm)} pieces in "
        f"{sum(1 for k in n_rm if k)} of {W['E']} environments")
    mpath = root / "maps.pt"
    if mpath.exists():
        maps = torch.load(mpath, map_location=device)
    else:
        t0 = time.time()
        log(f"building coarse obstacle maps for {W['E']} environments (first run only, a few minutes) ...")
        maps = build_maps(W, device, progress=True)
        buf = io.BytesIO(); torch.save(maps.cpu(), buf); write_bytes_atomic(mpath, buf.getvalue())
        log(f"coarse maps built in {time.time() - t0:.0f} s")
        maps = maps.to(device)
    Ws = [W] + [{k: (v.to(d) if hasattr(v, "to") else v) for k, v in W.items()} for d in devices[1:]]
    maps_g = [maps] + [maps.to(d) for d in devices[1:]]

    splits = cache["split"]
    train_rows = np.nonzero(splits == "train")[0]
    train_rows = train_rows[np.argsort(occ[train_rows], kind="stable")]        # least cluttered first
    val_rows = np.sort(np.random.default_rng(cfg["seed"] + 1).permutation(np.nonzero(splits == "val")[0])[:cfg["val_envs"]])
    test_rows = np.nonzero(splits == "test")[0]
    if cfg["test_envs"] is not None:
        test_rows = test_rows[:cfg["test_envs"]]

    import copy
    agent = make_agent(device)                                 # the master copy (optimizer, evaluation, saving)
    opt = torch.optim.Adam(agent.parameters(), lr=LR, eps=1e-5)
    reps = [agent] + [copy.deepcopy(agent).to(d) for d in devices[1:]]

    def sync_replicas():
        with torch.no_grad():
            for r in reps[1:]:
                for pr, p0 in zip(r.parameters(), agent.parameters()):
                    pr.copy_(p0)
    st = {"update": 0, "keep_idx": 0, "pool_frac": CURRICULUM_START, "best_success": -1.0, "since_best": 0, "boosts": 0,
          "best_eval": -float("inf"), "best_eval_update": -1, "evals_done": [], "stopped": False}
    ck = root / "checkpoint.pt"
    if ck.exists():
        d = torch.load(ck, map_location=device, weights_only=False)
        agent.load_state_dict(d["agent"]); opt.load_state_dict(d["opt"]); st = d["state"]
        log(f"resumed at update {st['update']}")
        if "det_hist" not in st:                       # checkpoint from before the noise-free checks
            st.update(best_success=-1.0, since_best=0, boosts=0, det_hist=[])
    sync_replicas()
    log("starting PPO" + (f" (resuming at update {st['update']})" if st["update"] else ""))

    n = cfg["n_envs"]
    n_g = [n // G + (1 if g < n % G else 0) for g in range(G)]           # flights per GPU
    sims = [Sim(Ws[g], maps_g[g], bench_tf, af, n_g[g], devices[g], seed=cfg["seed"] * 1000 + 97 * g + st["update"])
            for g in range(G)]

    def set_pool():
        k = max(1, int(round(st["pool_frac"] * len(train_rows))))
        for sm in sims:
            sm.pool = torch.as_tensor(train_rows[:k])

    def set_keep(frac):
        for sm in sims:
            sm.keep_frac = frac

    set_pool()
    set_keep(KEEP_STAGES[st.get("keep_idx", len(KEEP_STAGES) - 1)])
    for sm in sims:
        sm.reset(torch.arange(sm.n, device=sm.dev), sm.sample_envs(sm.n))
    ep_ret = [torch.zeros(sm.n, device=sm.dev) for sm in sims]
    recent = []                                                    # finished episodes (status, return, tf ratio)

    def save(best=False):
        buf = io.BytesIO()
        torch.save({"agent": agent.state_dict(), "opt": opt.state_dict(), "state": st, "cfg": cfg,
                    "settings": {k: v for k, v in globals().items() if k.isupper()
                                 and isinstance(v, (int, float, str, bool, type(None)))}}, buf)
        write_bytes_atomic(ck, buf.getvalue())
        if best:
            write_bytes_atomic(root / "best.pt", buf.getvalue())
        commit()

    # ---- solver evaluations (background) ----
    pending = []

    def items_from_flights(rows, flights, split, kinds):
        items = []
        for e, fl in zip(rows, flights):
            idx = int(cache["env_index"][e])
            base_item = {"dataset": ds, "solver_method": bm["method"], "overrides": bm.get("solver_overrides", {}),
                         "index": idx, "shard": shard_of[idx], "split": split, "time_limit": None,
                         "bench_cost": float(bench_cost[e]), "bench_flight_time": float(bench_tf[e]),
                         "flight": fl["status"], "drop_pieces": W["removed"][e]}
            goal_w = bc.to_world(W["goal"][e].cpu().numpy() / scale, cache["origin"][e], float(cache["yaw"][e]), scale)
            t, X, U = to_world(fl, cache["origin"][e], float(cache["yaw"][e]), goal_w)
            for kind in kinds:
                it = dict(base_item, kind=kind, label={"pilot": "pilot", "path": "path_only",
                                                       "straight": "straight"}[kind])
                if kind == "pilot":                                 # full state if it arrived, else path + line
                    if fl["status"] == "goal":
                        it.update(kind="full", t=t.tolist(), X=X.tolist(), U=U.tolist())
                    else:
                        it.update(kind="path", path=np.vstack([X[:, :3], goal_w]).tolist())
                elif kind == "path":
                    it.update(path=np.vstack([X[:, :3], goal_w]).tolist())
                items.append(it)
        return items

    def summarize_solves(rows):
        ok = [r for r in rows if r.get("success")]
        med = lambda v: float(np.median(v)) if len(v) else float("nan")
        sc = [score(r, consts) for r in rows]
        return {"n": len(rows), "score_mean": float(np.mean(sc)) if sc else float("nan"),
                "valid_pct": 100 * len(ok) / max(len(rows), 1),
                "iterations_median": med([r["iterations"] for r in ok]),
                "iter_feasible_median": med([r["iter_feasible"] for r in ok if r.get("iter_feasible")]),
                "flight_time_gap_median_pct": med([r["flight_time_gap_pct"] for r in ok]),
                "flight_time_gap_p90_pct": float(np.percentile([r["flight_time_gap_pct"] for r in ok], 90)) if ok else float("nan"),
                "solve_time_median_s": med([r["solve_time"] for r in ok])}

    def start_eval(update, rows, split, kinds, tag):
        flights = fly(agent, None, W, maps, bench_tf, af, rows, device)
        fstat = {k: 100 * sum(f["status"] == k for f in flights) / len(flights) for k in STATUS.values() if k != "flying"}
        items = items_from_flights(rows, flights, split, kinds)
        pending.append({"update": update, "split": split, "tag": tag, "fstat": fstat,
                        "snapshot": {k: v.detach().cpu().clone() for k, v in agent.state_dict().items()},
                        "get": submit_solves(items), "t0": time.time()})
        log(f"{tag} update {update}: flew {len(rows)} {split} envs (goal {fstat['goal']:.0f}%, crash "
            f"{fstat['crash']:.0f}%, stall {fstat['stall']:.0f}%, timeout {fstat['timeout']:.0f}%); "
            f"{len(items)} solves sent")

    def poll_evals(block=False):
        for p in list(pending):
            res = p["get"](block)
            if res is None:
                continue
            pending.remove(p)
            rows = []
            for label, rr in (("pilot", [r for r in res if r.get("label") == "pilot"]),
                              ("path_only", [r for r in res if r.get("label") == "path_only"]),
                              ("straight", [r for r in res if r.get("label") == "straight"])):
                if rr:
                    sm = summarize_solves(rr)
                    rows.append({"update": p["update"], "split": p["split"], "warm_start": label, **sm,
                                 **({f"flight_{k}_pct": v for k, v in p["fstat"].items()} if label != "straight" else {}),
                                 "time_s": round(time.time() - p["t0"], 1)})
            append_csv(root / "evals.csv", rows)
            for r in rows:
                log(f"{p['tag']} RESULT update {p['update']} {r['warm_start']:9s}: score {r['score_mean']:7.2f}  "
                    f"valid {r['valid_pct']:5.1f}%  iters {r['iterations_median']:5.1f}  "
                    f"tf gap {r['flight_time_gap_median_pct']:6.2f}%")
            if p["split"] == "val":
                pr = next(r for r in rows if r["warm_start"] == "pilot")
                if pr["score_mean"] > st["best_eval"] + 0.05:
                    st["best_eval"], st["best_eval_update"] = pr["score_mean"], p["update"]
                    buf = io.BytesIO()
                    torch.save({"agent": p["snapshot"], "state": dict(st), "cfg": cfg}, buf)
                    write_bytes_atomic(root / "best.pt", buf.getvalue())
                    log(f"  new best agent (update {p['update']})")
            else:
                write_bytes_atomic(root / "test_summary.json", json.dumps(rows, indent=1).encode())
            commit()

    # ---- PPO ----
    t_start = time.time()
    while st["update"] < cfg["updates"] and not st["stopped"]:
        k = st["update"]
        if k % cfg["eval_every"] == 0 and k not in st["evals_done"]:
            start_eval(k, val_rows, "val", ["pilot", "straight"] if k == 0 else ["pilot"], "VAL")
            st["evals_done"].append(k)
        poll_evals()
        t0 = time.time()
        buf = [{"O": [], "M": [], "A": [], "LP": [], "R": [], "D": [], "V": []} for _ in range(G)]
        with torch.no_grad():
            for _ in range(ROLLOUT):
                outs = []
                for g, sm in enumerate(sims):                 # launch every GPU's work first (runs in parallel)
                    ag = reps[g]
                    o, m = sm.obs()
                    mu = ag.mean(o, m)
                    std = ag.log_std.clamp(min=LOG_STD_MIN).exp()
                    a_ = mu + std * torch.randn_like(mu)
                    lp = (-0.5 * ((a_ - mu) / std) ** 2 - torch.log(std) - 0.5 * math.log(2 * math.pi)).sum(-1)
                    v = ag.value(o, m)
                    r, done, status, _ = sm.step(a_.clamp(-1, 1))
                    ep_ret[g] += r
                    bg = buf[g]
                    bg["O"].append(o); bg["M"].append(m); bg["A"].append(a_); bg["LP"].append(lp)
                    bg["R"].append(r); bg["D"].append(done.float()); bg["V"].append(v)
                    outs.append((done, status))
                for g, sm in enumerate(sims):                 # then the bookkeeping that waits for results
                    done, status = outs[g]
                    idx = torch.nonzero(done).squeeze(1)
                    if len(idx):
                        stt, tt = status[idx].cpu().numpy(), sm.t[idx].cpu().numpy()
                        ee, rr = sm.env[idx].cpu().numpy(), ep_ret[g][idx].cpu().numpy()
                        for s_, t_, e_, r_ in zip(stt, tt, ee, rr):
                            recent.append((int(s_), float(r_), float(t_ / bench_tf[int(e_)]) if s_ == 1 else float("nan")))
                        ep_ret[g][idx] = 0.0
                        sm.reset(idx, sm.sample_envs(len(idx)))
            data = []
            for g, sm in enumerate(sims):
                bg = buf[g]
                o, m = sm.obs()
                v_last = reps[g].value(o, m)
                R, D, Vv = torch.stack(bg["R"]), torch.stack(bg["D"]), torch.stack(bg["V"])
                adv = torch.zeros_like(R)
                gae = torch.zeros(sm.n, device=sm.dev)
                for i in reversed(range(ROLLOUT)):
                    nv = v_last if i == ROLLOUT - 1 else Vv[i + 1]
                    delta = R[i] + GAMMA_RL * nv * (1 - D[i]) - Vv[i]
                    gae = delta + GAMMA_RL * GAE_LAMBDA * (1 - D[i]) * gae
                    adv[i] = gae
                data.append({"O": torch.cat(bg["O"]), "M": torch.cat(bg["M"]), "A": torch.cat(bg["A"]),
                             "LP": torch.cat(bg["LP"]), "adv": adv.reshape(-1), "ret": (adv + Vv).reshape(-1)})
            del buf
        info = {"pl": 0.0, "vl": 0.0, "ent": 0.0, "clip": 0.0}
        nb = 0
        mb_g = max(1, MINIBATCH // G)                       # each GPU's share of a minibatch
        n_mb = max(1, math.ceil(len(data[0]["adv"]) / mb_g))
        for _ in range(EPOCHS):
            perms = [torch.randperm(len(dg["adv"]), device=dg["adv"].device) for dg in data]
            for j in range(n_mb):
                losses = []
                for g in range(G):                            # forward/backward on every GPU (in parallel)
                    ag, dg = reps[g], data[g]
                    bidx = perms[g][j * mb_g:(j + 1) * mb_g]
                    if len(bidx) == 0:
                        continue
                    mu = ag.mean(dg["O"][bidx], dg["M"][bidx])
                    ls = ag.log_std.clamp(min=LOG_STD_MIN)
                    std = ls.exp()
                    lp = (-0.5 * ((dg["A"][bidx] - mu) / std) ** 2 - ls - 0.5 * math.log(2 * math.pi)).sum(-1)
                    ratio = (lp - dg["LP"][bidx]).exp()
                    ab = dg["adv"][bidx]
                    ab = (ab - ab.mean()) / (ab.std() + 1e-8)
                    pl = -torch.min(ratio * ab, ratio.clamp(1 - CLIP, 1 + CLIP) * ab).mean()
                    vl = ((ag.value(dg["O"][bidx], dg["M"][bidx]) - dg["ret"][bidx]) ** 2).mean()
                    ent = (ls + 0.5 * (1 + math.log(2 * math.pi))).sum()
                    loss = pl + VF_COEF * vl - ENT_COEF * ent
                    ag.zero_grad(set_to_none=True)
                    loss.backward()
                    losses.append((pl.detach(), vl.detach(), ent.detach(), ((ratio - 1).abs() > CLIP).float().mean().detach()))
                if G > 1:                                     # average the gradients into the master copy
                    with torch.no_grad():
                        for ps in zip(*[r.parameters() for r in reps]):
                            grads = [p_.grad.to(device) for p_ in ps if p_.grad is not None]
                            if grads:
                                ps[0].grad = torch.stack(grads).mean(0)
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                opt.step()
                with torch.no_grad():
                    agent.log_std.clamp_(min=LOG_STD_MIN)
                sync_replicas()
                for pl_, vl_, en_, cl_ in losses:
                    info["pl"] += pl_.item() / len(losses); info["vl"] += vl_.item() / len(losses)
                    info["ent"] += en_.item() / len(losses); info["clip"] += cl_.item() / len(losses)
                nb += 1
        del data
        info = {kk: vv / max(nb, 1) for kk, vv in info.items()}

        # ---- stats, curriculum, stagnation ----
        recent = recent[-4000:]
        last = recent[-max(1, min(len(recent), 2 * n)):]
        cnt = lambda c: 100 * sum(s_ == c for s_, _, _ in last) / max(len(last), 1)
        succ = cnt(1)
        ratios = [r_ for s_, _, r_ in last if s_ == 1]
        st["update"] = k + 1
        action = ""
        det = ""
        at_end = st["keep_idx"] >= len(KEEP_STAGES) - 1 and st["pool_frac"] >= 1.0
        if (k + 1) % CHECK_EVERY == 0:
            # noise-free check flights on the current stage
            pool = sims[0].pool.numpy()
            rows_c = np.random.default_rng(cfg["seed"] * 7919 + k).choice(pool, CHECK_FLIGHTS, replace=len(pool) < CHECK_FLIGHTS)
            fl = fly(agent, None, W, maps, bench_tf, af, rows_c, device, keep_frac=KEEP_STAGES[st["keep_idx"]], seed=k)
            det = 100 * sum(f["status"] == "goal" for f in fl) / len(fl)
            hist = st.setdefault("det_hist", [])
            hist.append(det)
            score2 = float(np.mean(hist[-2:]))
            if len(hist) >= 2 and score2 >= 100 * CURRICULUM_SUCCESS and not at_end:
                if st["keep_idx"] < len(KEEP_STAGES) - 1:
                    st["keep_idx"] += 1
                    set_keep(KEEP_STAGES[st["keep_idx"]])
                    action = f"curriculum -> keep {KEEP_STAGES[st['keep_idx']]:.0%} of the obstacles"
                else:
                    st["pool_frac"] = min(1.0, st["pool_frac"] + CURRICULUM_STEP)
                    set_pool()
                    action = f"curriculum -> {st['pool_frac']:.0%} of training envs"
                st["best_success"], st["since_best"] = -1.0, 0
                hist.clear()
                recent.clear()
            else:
                if st["best_success"] < 0 or score2 > st["best_success"] + 1.0:
                    st["best_success"], st["since_best"] = score2, 0
                    if not at_end:
                        st["boosts"] = 0
                else:
                    st["since_best"] += CHECK_EVERY
                if st["since_best"] >= PLATEAU_UPDATES:
                    st["since_best"] = 0
                    st["boosts"] += 1
                    std_now = agent.log_std.clamp(min=LOG_STD_MIN).exp()
                    if bool((std_now < BOOST_BELOW * math.exp(LOG_STD_INIT)).all()):
                        with torch.no_grad():
                            agent.log_std.copy_(torch.maximum(agent.log_std, torch.full_like(agent.log_std, LOG_STD_INIT)))
                        sync_replicas()
                        action = f"plateau {st['boosts']}: exploration boosted"
                    else:
                        action = f"plateau {st['boosts']}: noise not collapsed, no boost"
                    if at_end and st["boosts"] >= MAX_BOOSTS:
                        st["stopped"] = True
                        action += "; stop: no further improvement"
        sig = agent.log_std.clamp(min=LOG_STD_MIN).exp().tolist()
        row = {"update": k + 1, "obstacles_kept": KEEP_STAGES[st["keep_idx"]], "pool_frac": round(st["pool_frac"], 2),
               "episodes": len(last),
               "goal_pct": round(succ, 2), "goal_noise_free_pct": round(det, 2) if det != "" else "", "crash_pct": round(cnt(2) + cnt(3), 2), "stall_pct": round(cnt(4), 2),
               "timeout_pct": round(cnt(5), 2), "missed_pct": round(cnt(6), 2), "return_mean": round(float(np.mean([r_ for _, r_, _ in last])) if last else 0.0, 3),
               "tf_vs_benchmark": round(float(np.median(ratios)), 4) if ratios else "",
               "std_T": round(sig[0], 3), "std_nv": round(sig[1], 3), "std_nh": round(sig[2], 3),
               **{kk: round(vv, 4) for kk, vv in info.items()},
               "steps_per_s": round(ROLLOUT * n / (time.time() - t0)), "action": action,
               "elapsed_min": round((time.time() - t_start) / 60, 1)}
        append_csv(root / "updates.csv", [row])
        if (k + 1) % 5 == 0 or action:
            log(f"update {k + 1:5d}: goal {succ:5.1f}%" + (f" (noise-free {det:5.1f}%)" if det != "" else "") + f"  crash {row['crash_pct']:5.1f}%  stall {row['stall_pct']:4.1f}%  "
                f"timeout {row['timeout_pct']:4.1f}%  missed {row['missed_pct']:4.1f}%  tf/bench {row['tf_vs_benchmark'] or float('nan'):.3f}  "
                f"obst {KEEP_STAGES[st['keep_idx']]:.0%} pool {st['pool_frac']:.0%}  std {sig[1]:.2f}  [{row['steps_per_s']} steps/s] {action}")
        if (k + 1) % CHECKPOINT_EVERY == 0 or action:
            save()

    save()
    log("training finished; waiting for outstanding evaluations")
    while pending:
        poll_evals(block=True)

    # ---- test: best agent ----
    if len(test_rows) and not (root / "test_summary.json").exists():
        if (root / "best.pt").exists():
            agent.load_state_dict(torch.load(root / "best.pt", map_location=device, weights_only=False)["agent"])
        start_eval(st["best_eval_update"], test_rows, "test", ["pilot", "path", "straight"], "TEST")
        while pending:
            poll_evals(block=True)
    if (root / "test_summary.json").exists():
        for r in json.loads((root / "test_summary.json").read_text()):
            log(f"TEST {r['warm_start']:9s}: score {r['score_mean']:7.2f}  valid {r['valid_pct']:5.1f}%  "
                f"iters {r['iterations_median']:5.1f}  tf gap {r['flight_time_gap_median_pct']:6.2f}%")


# --------------------------------------------------------------------------
# Modal
# --------------------------------------------------------------------------
try:
    import modal
except ImportError:
    modal = None

if modal is not None:
    app = modal.App(MODAL_APP_NAME)
    volume = modal.Volume.from_name(MODAL_VOLUME, create_if_missing=True)
    _env = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
            "RAYON_NUM_THREADS": "1", "FW_NO_MODAL_DEFS": "1"}
    solver_image = modal.Image.debian_slim(python_version=PYTHON_VERSION).pip_install(*SOLVER_PACKAGES).env(_env)
    gpu_image = (modal.Image.debian_slim(python_version=PYTHON_VERSION)
                 .pip_install("numpy", "torch").env({"FW_NO_MODAL_DEFS": "1"}))
    for _d in CODE_DIRS:
        solver_image = solver_image.add_local_dir(str(PROJECT / _d), f"/root/fixed-wing-rl/{_d}",
                                                  ignore=["__pycache__", "*.pyc"])
    gpu_image = gpu_image.add_local_dir(str(PROJECT / "aircraftmodel"), "/root/fixed-wing-rl/aircraftmodel")
    for _f in ("eval_warmstart.py", "build_cache.py"):
        solver_image = solver_image.add_local_file(str(_HERE / _f), f"/root/{_f}")
        gpu_image = gpu_image.add_local_file(str(_HERE / _f), f"/root/{_f}")

    @app.function(image=solver_image, cpu=1.0, memory=2048, timeout=TASK_TIMEOUT_S, retries=1,
                  max_containers=MAX_CONTAINERS, volumes={VOLUME_MOUNT: volume})
    def solve_remote(items):
        return solve_items(items)

    @app.function(image=gpu_image, gpu=GPU if N_GPUS <= 1 else f"{GPU}:{N_GPUS}", cpu=4.0 * max(1, N_GPUS),
                  memory=32768 * max(1, N_GPUS) // 2, timeout=LOOP_TIMEOUT_S,
                  volumes={VOLUME_MOUNT: volume})
    def train_remote(cfg):
        volume.reload()

        def submit(items):
            chunks = [items[i:i + SOLVES_PER_CALL] for i in range(0, len(items), SOLVES_PER_CALL)]
            calls = [solve_remote.spawn(c) for c in chunks]
            state = {"out": None}

            def get(block=False):
                if state["out"] is not None:
                    return state["out"]
                out = []
                for call, chunk in zip(calls, chunks):
                    try:
                        out += call.get(timeout=None if block else 0)
                    except TimeoutError:
                        return None
                    except Exception as exc:
                        if "timeout" in type(exc).__name__.lower() or "Timeout" in str(type(exc)):
                            return None
                        out += [{"status": "error", "success": False, "traceback": repr(exc),
                                 "label": it.get("label"), "kind": it["kind"], "index": it["index"]} for it in chunk]
                for r, it in zip(out, items):
                    r.setdefault("label", it.get("label"))
                state["out"] = out
                return out
            return get
        return run(cfg, submit, volume.commit)

    def _ensure_inputs(dataset, run_name):
        root = _HERE / "datasets" / dataset

        def have(path):
            try:
                return {e.path.lstrip("/"): e.size for e in volume.listdir(path)}
            except Exception:
                return {}
        uploads = []
        cache = root / "cache" / f"{run_name}.npz"
        if have(f"/datasets/{dataset}/cache").get(f"datasets/{dataset}/cache/{run_name}.npz") is None:
            uploads += [(cache, f"/datasets/{dataset}/cache/{run_name}.npz"),
                        (cache.with_suffix(".json"), f"/datasets/{dataset}/cache/{run_name}.json")]
        if not have(f"/datasets/{dataset}/envs"):
            uploads += [(p, f"/datasets/{dataset}/envs/{p.name}") for p in sorted((root / "envs").glob("shard_*.json"))]
        if f"datasets/{dataset}/index.csv" not in have(f"/datasets/{dataset}"):
            uploads.append((root / "index.csv", f"/datasets/{dataset}/index.csv"))
        if f"datasets/{dataset}/solutions/{run_name}/manifest.json" not in have(f"/datasets/{dataset}/solutions/{run_name}"):
            uploads.append((root / "solutions" / run_name / "manifest.json",
                            f"/datasets/{dataset}/solutions/{run_name}/manifest.json"))
        missing = [str(p) for p, _ in uploads if not Path(p).exists()]
        if missing:
            raise SystemExit("Needed but not found locally or on the Volume:\n  " + "\n  ".join(missing))
        if uploads:
            print(f"Uploading {len(uploads)} file(s) to Volume '{MODAL_VOLUME}'...", flush=True)
            with volume.batch_upload(force=True) as batch:
                for p, r in uploads:
                    batch.put_file(str(p), r)

    def _download(name):
        out = _HERE / "rl_pilot" / name
        n = 0
        for e in volume.listdir(f"/rl_pilot/{name}"):
            fname = e.path.rsplit("/", 1)[-1]
            if e.type != modal.volume.FileEntryType.FILE or fname == "maps.pt":
                continue
            write_bytes_atomic(out / fname, b"".join(volume.read_file(e.path)))
            n += 1
        print(f"Downloaded {n} files to {out}")
        if (out / "evals.csv").exists():
            print("\nevals.csv:\n" + (out / "evals.csv").read_text())

    @app.local_entrypoint()
    def main(dataset: str = DATASET, run: str = RUN, name: str = NAME, updates: int = UPDATES,
             n_envs: int = N_ENVS, eval_every: int = EVAL_EVERY, val_envs: int = VAL_ENVS,
             test_envs: int = -1, seed: int = SEED, download_only: bool = False):
        if not download_only:
            _ensure_inputs(dataset, run)
            cfg = {"dataset": dataset, "run": run, "name": name, "updates": updates, "n_envs": n_envs,
                   "eval_every": eval_every, "val_envs": val_envs,
                   "test_envs": None if test_envs < 0 else test_envs, "seed": seed, "n_gpus": N_GPUS}
            print(f"Starting pilot training '{name}' on Modal: {json.dumps(cfg)}", flush=True)
            train_remote.remote(cfg)
        _download(name)


if __name__ == "__main__":
    raise SystemExit("Run with Modal:  modal run --detach supervised-learning-main/rl_pilot.py --name NAME")