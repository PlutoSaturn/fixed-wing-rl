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

What the agent senses (only what its own scans have shown it; nothing else about the obstacles)
    its speed, climb angle, current controls, height above the floor and below the ceiling,
    the direction / distance / elevation of the goal, the elapsed time, and
    a SHORT SCAN every step: N_RAYS lidar rays looking forward (a horizontal fan of +-FAN_DEG and two
        slightly raised / lowered fans of +-45 deg), each giving the exact distance to the first
        inflated obstacle or box wall it hits, up to RAY_RANGE_M (cheap: done every step), and
    a FAR SCAN every FAR_EVERY_S seconds (and at the start): FAR_AZ x FAR_EL rays in a narrow cone
        (+-FAR_AZ[1] deg x +-FAR_EL[1] deg) centred on the line to the goal, out to FAR_RANGE_M, to
        see how the course opens up kilometres ahead (costlier: done now and then). The strategist
        (stage 2) sees the last FAR_KEEP far scans as depth images, each with where it was taken
        relative to the aircraft now, the heading change since and its age: its memory of the course
        ahead. The flier sees only the short scan.
No map is built (it made each step too slow): nothing about the obstacles is remembered except
those last far scans.
No rewinds: a crash ends the flight.

Two learning systems (hierarchical RL), trained one after the other (STAGE):
    1. "flier": the agent above flies to a TARGET, a waypoint 1-3 km ahead or, at the end, the goal.
       It sees the target's direction / distance / height and the projected miss of its current
       line of flight at the target (not the goal's), whether the target is the goal, and how far
       the goal still is. In training it is given a series of random waypoints (WP_DIST_M ahead,
       within +-WP_ANG_DEG of the line to the goal, +-WP_DZ_M in height, in free air) that end at
       the goal, so it learns to fly anywhere it is sent.
    2. "strategy": a second agent picks the waypoints; the flier from stage 1 is FROZEN and flies to
       them. The strategist decides when a flight starts, when a waypoint is reached or passed, and
       at least every STRAT_EVERY_S seconds (right after the far scan). It sees its own state
       relative to the GOAL, the short scan, and its last FAR_KEEP far scans (with where / when they
       were taken), and outputs the next waypoint: a heading offset from the line to the goal
       (+-STRAT_ANG_DEG), a distance (WP_DIST_M) and a height change (+-STRAT_DZ_M); a waypoint
       beyond the goal means "go to the goal". Each decision is one PPO step; its reward is what the
       flight earns until the next decision: progress toward the goal, time, and the outcome.
    Alternating rounds (each round is one run): a basic flier (--stop-at-keep 0.5 ends it once it
    has mastered half the obstacles) -> a strategist on it -> the flier again, continuing from its
    weights (--init-from), now practising on that frozen strategist's waypoints (--strategy; a
    WP_MIX_RANDOM share of its flights keep random practice waypoints so it does not forget them)
    -> the strategist again on the new flier (--init-from) -> ... Each continued run starts from the
    previous run's weights and curriculum position.

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

Reward each step (flier)
    + PROG_W  * (reduction in distance to the target / starting distance)   (progress)
    + WP_R for reaching a waypoint (within WP_RADIUS_M); - WP_MISS_R x min(1, distance / MISS_SCALE_M)
      for passing one (it ends up behind the aircraft, within WP_PASS_M); either way the next comes
    - TIME_W  * (dt / straight-line flight time)               (fly fast)
    - PROX_W  * dt * (how deep it is inside a PROX_M soft zone around the inflated obstacles)
    + danger shaping: the change in -TTC_K * (1 - time to impact / TTC_S) (central rays, 0 beyond TTC_S
                seconds): turning away from an obstacle pays, heading at one costs; potential-based, so
                it adds up to the same total however the flight ends (no reason to end it early)
    - SMOOTH_W * dt * (commanded change of n_v and n_h)^2      (no porpoising / weaving)
    + AIM_K * (improvement of min(1, projected miss / MISS_SCALE_M))   within APPROACH_M of the goal
                (potential-based shaping: 0 ... +AIM_K per flight), where the projected miss is how
                far the current velocity would pass from the goal when it crosses the goal plane
                (the plane through the goal across the start-goal line)
    and at the end: + SUCCESS_R + PRECISION_R * (1 - closest approach / GOAL_RADIUS_M) ^ PRECISION_POW
                    for arriving. Entering the arrival sphere (GOAL_RADIUS_M around the goal) while the
                    goal is the current target makes the flight a success (flying through it while still
                    on waypoints does not count); it then keeps flying until it is past its closest point to the goal
                    (or leaves the sphere), and the bonus is graded by how close it actually got (0 m:
                    +PRECISION_R, 50 m: +6.4, 125 m: +2.5, 250 m: 0 with the defaults). For the solver, the
                    flight is cut at that closest point.
                    - MISS_R * min(1, miss / MISS_SCALE_M) for crossing the goal plane without arriving
                    (the flight ends there, so a near miss costs less than a wide one),
                    - CRASH_R for an obstacle / the box / a stall, - TIMEOUT_R for running out of time.

Learning: PPO (clipped, GAE), N_ENVS flights in parallel, separate actor and critic networks.
Multi-GPU: the N_ENVS flights are split across N_GPUS GPUs; each simulates its share and computes
gradients on its share of every minibatch with its own copy of the agent; the gradients are
averaged, one optimizer step is taken, and the copies are re-synchronized. The result is the
same update as on one GPU, computed in parallel.
Curriculum: a smooth difficulty dial, in two parts:
    1. obstacle share: every flight keeps a random share of its environment's obstacles (a new subset
       each flight), rising continuously from KEEP_START (0 = open sky, to learn to fly the envelope)
       to all of them, on the least cluttered CURRICULUM_START share of the training environments;
    2. then the environment pool widens continuously toward the more cluttered ones (by occupancy).
The dial is driven by the training flights themselves (no extra flying): once CURRICULUM_MIN_FLIGHTS
flights that STARTED at the current difficulty have finished (every update or two), it moves up by
CURRICULUM_GAIN x the margin by which their success beats CURRICULUM_SUCCESS: at 85 % success,
1/3 x 15 points = 5 points more obstacles (or 5 % more of the environments); at 100 %, 10 points; at
or below the threshold it holds. For the flier the margin is the smaller of its goal margin and its
waypoint margin (above CURRICULUM_WP). Flights started before a bump never count toward the next one.
The scans and collisions always reflect the obstacles the flight actually has.
Solver evaluations always use the full environments.
Every CHECK_EVERY updates it also flies CHECK_FLIGHTS flights at the current difficulty with its mean
action (no exploration noise), for the log and to detect a stall (plateau).
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
    modal run --detach supervised-learning-main/rl_pilot.py --name flier_v1                    stage 1
    modal run --detach supervised-learning-main/rl_pilot.py --name strat_v1 --stage strategy --flier flier_v1   stage 2
    modal run supervised-learning-main/rl_pilot.py --name flier_v1 --download-only
  alternating rounds:
    modal run --detach supervised-learning-main/rl_pilot.py --name flier_r1 --stop-at-keep 0.5
    modal run --detach supervised-learning-main/rl_pilot.py --name strat_r1 --stage strategy --flier flier_r1
    modal run --detach supervised-learning-main/rl_pilot.py --name flier_r2 --init-from flier_r1 --strategy strat_r1
    modal run --detach supervised-learning-main/rl_pilot.py --name strat_r2 --stage strategy --flier flier_r2 --init-from strat_r1
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
GOAL_RADIUS_M  = 250.0               # arrival sphere: passing within this of the goal = reaching it
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
FAR_EVERY_S    = 10.0                # a far scan every this many seconds (and at the start): a cone ...
FAR_AZ         = (31, 15.0)          #   ... of this many rays across +- this azimuth (deg) ...
FAR_EL         = (11, 5.0)           #   ... x this many across +- this elevation (deg), around the line
FAR_RANGE_M    = 5000.0              #   to the goal, out to this range
FAR_EL_MAX_DEG = 10.0                # the cone's centre is tilted toward the goal by at most this
FAR_CHUNK      = 24                  # rays per batch in the far scan (memory)
FAR_KEEP       = 3                   # the strategist sees this many most recent far scans (30 s)


# targets: waypoints, then the goal (both stages)
WP_RADIUS_M    = 150.0               # a waypoint is reached when the flight passes within this
WP_PASS_M      = 800.0               # ... and passed (missed) when it is behind the aircraft and this close
WP_R           = 2.0                 # flier: reaching a waypoint
WP_MISS_R      = 2.0                 # flier: passing one, up to this (graded by the miss distance)
WP_NORM_M      = 3000.0              # distance scale of the target inputs
WP_DIST_M      = (1000.0, 3000.0)    # waypoint distance from the aircraft (both stages)
WP_ALT_MARGIN_M = 80.0               # waypoints stay this far inside the floor and ceiling
# stage 1 (flier): random waypoints to practise on
WP_ANG_DEG     = 60.0                # within +- this of the line to the goal
WP_DZ_M        = 150.0               # +- this from the current height
WP_CLEAR_M     = 100.0               # in free air: at least this far from the inflated obstacles, and so is
                                     #   the straight path to it (sampled at WP_PATH_PTS points)
WP_PATH_PTS    = 12
WP_RUNOUT_M    = 600.0               # and the air beyond it, along the approach, for this far (room to turn
                                     #   after reaching it: no waypoint just in front of an obstacle)
WP_AROUND_FRAC = 0.3                 # ... except this share of practice waypoints: in free air, but with an
                                     #   obstacle across the straight path (first hit >= WP_AROUND_MIN_M away)
                                     #   and a one-turn detour that is clear: they must be flown AROUND
WP_AROUND_MIN_M = 500.0
WP_AROUND_OFF_M = (400.0, 800.0)     # detour check: via the midpoint, offset sideways by these (both sides)
WP_TRIES       = 32                  # candidates tried per waypoint (else the clearest one)
WP_GOAL_PROB   = 0.2                 # share of practice flights that fly straight to the goal instead
WP_MIX_RANDOM  = 0.15                # flier rounds with a strategist: this share of flights get random
                                     #   practice waypoints, the rest the strategist's ...
PRACTICE_KEEP_MAX = 0.4              # ... and those practice flights keep at most this obstacle share (in
                                     #   heavy clutter random waypoints are near-certain crashes); the
                                     #   curriculum then judges the strategist's flights only (goal rate)
WP_LINE_M      = 2000.0              # "line" mode (stage-1 evaluations): waypoints this far apart on the
                                     #   straight line to the goal
# stage 2 (strategy): the strategist picks the waypoints
STRAT_EVERY_S  = 10.0                # decides at least this often (also at the start and at each waypoint)
STRAT_ANG_DEG  = 75.0                # heading offset from the line to the goal, +- this
STRAT_DZ_M     = 200.0               # height change, +- this
STRAT_ROLLOUT  = 256                 # simulator steps per update (each flight makes ~13+ decisions)
GAMMA_S        = 0.97                # discount per decision (~10 s)
LAMBDA_S       = 0.9
STRAT_PROG_W   = 10.0                # strategist reward: progress toward the GOAL / starting distance ...
STRAT_TIME_W   = 6.0                 #   ... minus time (the route's speed is the strategist's job), plus the
                                     #   flight's outcome (same terms as the flier's)
STAGE          = "flier"             # "flier" (stage 1) or "strategy" (stage 2, needs a trained flier run)

# reward
PROG_W         = 10.0
TIME_W         = 1.0                 # flier: light time pressure, so following the waypoints is not punished
PROX_W         = 0.2
PROX_M         = 60.0
TTC_W          = 0.0                 # (old per-second form: it piled up in heavy clutter, and ending the flight
                                     #   with a stall became the cheapest way out - replaced by TTC_K)
TTC_K          = 10.0                # danger shaping, potential-based: reward = change in -TTC_K x danger,
                                     #   danger = (1 - time to impact / TTC_S), 0 ... 1. Turning away pays up
                                     #   to +TTC_K, heading in costs it; summed over a flight it is the same
                                     #   however and whenever the flight ends, so ending early never pays
TTC_S          = 20.0                #   (time to impact = distance along the central rays / speed); ~1.1 km,
                                     #   early enough to turn gently around it
SMOOTH_W       = 0.05                # per second, on the squared change of the bank / climb targets per step
SUCCESS_R      = 10.0                # reaching the goal (entering the GOAL_RADIUS_M sphere) ...
PRECISION_R    = 10.0                #   ... plus up to this much more, the closer it actually gets to the centre:
PRECISION_POW  = 2.0                 #   PRECISION_R x (1 - closest approach / GOAL_RADIUS_M) ^ this
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
ENT_COEF       = 0.01                # keeps exploring (0.001 let the bank noise collapse before it found
                                     #   that turning away from obstacles pays)
VF_COEF        = 0.5
HIDDEN         = 256
LOG_STD_INIT   = -0.5                # exploration noise on the actions (in [-1, 1] units)
LOG_STD_MIN    = -1.6                # noise floor ~0.2 (in [-1, 1] action units): large enough to try evasive turns
ENT_COEF_END   = 0.001               # flier: over each run's first STD_ANNEAL_UPDATES updates, the entropy bonus
LOG_STD_MIN_END = -3.0               #   and the noise floor fall from ENT_COEF / LOG_STD_MIN to these (floor
STD_ANNEAL_UPDATES = 300             #   0.2 -> 0.05): explore early in a round, settle late
STD_LR_MULT_S  = 10.0                # strategist: its noise learns this much faster than the weights (it makes
                                     #   few decisions -> few optimizer steps per update: at the base rate
                                     #   the noise took ~300 updates to halve)
ENT_COEF_S     = 0.001               # strategist: small, so its noise can shrink (a heading noise of 0.6 =
LOG_STD_MIN_S  = -2.5                #   +-45 deg per decision held its training flights far below its
                                     #   noise-free ones, and the curriculum judges the training flights)
SEED           = 0

# curriculum and stagnation
KEEP_START         = 0.0             # obstacle share a new flier starts with (0 = open sky); a new strategist
                                     #   starts at the share its flier reached, and climbs from there
CURRICULUM_START   = 0.3             # least cluttered share of the training envs to start with
CURRICULUM_SUCCESS = 0.7             # NOISE-FREE goal success above which the difficulty rises ...
CURRICULUM_WP      = 0.8             # ... and for the flier, NOISE-FREE share of waypoints reached above which
CURRICULUM_GAIN    = 1.0 / 3.0       # rise per check = this x the margin above the threshold (smaller of the
                                     #   two for the flier), in obstacle share, then in share of environments
CURRICULUM_MIN_FLIGHTS = 256         # finished flights at the current difficulty needed for each decision
_OLD_KEEP_STAGES   = (0.0, 0.25, 0.5, 0.75, 1.0)   # (stage-based checkpoints: their obstacle shares)
CHECK_EVERY        = 50              # updates between noise-free check flights (logging and stall detection
                                     #   only; the curriculum uses the training flights)
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
N_GPUS         = 1                 # GPUs in the training container (Modal allows up to 8). The flights are
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
_CFG_KEYS = ("dataset", "run", "n_envs", "seed", "stage", "flier", "strategy", "init_from")
N_RAYS = N_RAYS_FAN + 2 * N_RAYS_TILT
OBS_VERSION = 11                      # 1: old lidar; 2: forward exact lidar; 3: + projected miss; 5: + difficulty field;
                                     # 6: + look-ahead manoeuvres; 7: learned penalty map; 8: maps it builds from
                                     #   its own scans (short + far scan; local view + overview); 10: + waypoint targets
                                     #   (flier) and the strategist; 11: no map (flier: short scan only;
                                     #   strategist: + the last FAR_KEEP far scans)
OBS_DIM = 17 + N_RAYS                 # flier: state, target (6), miss at the target (2), rays, final flag, goal dist
STRAT_DIM = 10 + N_RAYS + 6 * FAR_KEEP   # strategist: state relative to the goal, rays, where / when
                                         #   each kept far scan was taken
FAR_SHAPE = (FAR_KEEP, FAR_EL[0], FAR_AZ[0])   # the kept far scans as depth images (elevation x azimuth)
MAP_FLAT = math.prod(FAR_SHAPE)        # image inputs per flight (the strategist's far scans; the flier has none)
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


def sdf(p, env_ids, W, chunk=None, drop=None, walls=True):
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
        if not walls:
            out.append(h)
            continue
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


def ray_hits(pos, dirs, env_ids, W, drop=None, rng=None):
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
    return torch.clamp(torch.minimum(t_hit, tw), max=RAY_RANGE_M if rng is None else rng)


def cast_rays(pos, chi, env_ids, W, drop=None):
    dirs = ray_dirs(chi, pos.device)
    return ray_hits(pos, dirs, env_ids, W, drop) / RAY_RANGE_M


class Sim:
    """N parallel flights. State per flight: x (6: px py pz V gamma chi, flight frame), controls
    (T, n_v, n_h), time, environment id, target, and its last far scans."""

    def __init__(self, W, maps, bench_tf, af, n, device, seed=0):
        import torch
        self.W, self.af, self.n, self.dev = W, af, n, device
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
        self.rows = torch.arange(n, device=device)
        # the last FAR_KEEP far scans: depth images (newest first), where / which way / when taken
        self.far = torch.ones((n,) + FAR_SHAPE, device=device)
        self.far_pos = torch.zeros((n, FAR_KEEP, 3), device=device)
        self.far_dir = torch.zeros((n, FAR_KEEP), device=device)
        self.far_t = torch.full((n, FAR_KEEP), -1e3, device=device)
        self.steps = torch.zeros(n, dtype=torch.long, device=device)            # steps since the flight started
        self.far_every = max(1, int(round(FAR_EVERY_S / DT)))
        self.crashes = torch.zeros(n, dtype=torch.long, device=device)           # (always 0: no rewinds)
        self.ended = torch.zeros(n, dtype=torch.bool, device=device)            # set by step: flight finished
        self.frozen = torch.zeros(n, dtype=torch.bool, device=device)           # fly(): finished flights stay put
        self.last_status = torch.zeros(n, dtype=torch.long, device=device)
        self.L = int(TIME_LIMIT_X * float(W["goal"].norm(dim=-1).max()) / af["V_CRUISE"] / DT) + 8
        self.hist_x = torch.zeros((n, self.L, 6), device=device)
        self.hist_u = torch.zeros((n, self.L, 3), device=device)
        self.hist_t = torch.zeros((n, self.L), device=device)
        self.hist_len = torch.ones(n, dtype=torch.long, device=device)
        self.record = False                                   # fly(record=True): lidar along the path
        self.hist_r = None
        self.pool = None
        self.keep_frac = 1.0                                  # share of obstacles each new flight keeps
        self.level = 0                                        # curriculum difficulty counter (bumps so far) ...
        self.start_level = torch.zeros(n, dtype=torch.long, device=device)   # ... when each flight started
        self.clock = 0                                        # simulator steps so far (all flights step together)
        self.wp_around = torch.zeros(n, dtype=torch.bool, device=device)     # current waypoint must be flown around
        self.start_step = torch.zeros(n, dtype=torch.long, device=device)    # clock when each flight started
        T = W["tok"].shape[1]
        self.drop = torch.zeros((n, T), dtype=torch.bool, device=device)
        self.rays = None
        # far-scan directions (relative to the line to the goal): azimuth x elevation
        az = torch.linspace(-math.radians(FAR_AZ[1]), math.radians(FAR_AZ[1]), FAR_AZ[0], device=device)
        el = torch.linspace(-math.radians(FAR_EL[1]), math.radians(FAR_EL[1]), FAR_EL[0], device=device)
        E_, A_ = torch.meshgrid(el, az, indexing="ij")                             # image order: elevation x azimuth
        self.far_az, self.far_el = A_.reshape(-1), E_.reshape(-1)
        self.far_hits = None                                  # fly(record=True): what each far scan hit
        # targets: a waypoint or the goal. wp_mode: "goal" (always the goal), "random" (stage 1
        # practice waypoints), "external" (stage 2: set by the strategist when need_wp)
        self.wp_mode = "goal"
        self.mix_random = WP_MIX_RANDOM                       # "mixed" mode: share of flights on random waypoints
        self.ext = torch.zeros(n, dtype=torch.bool, device=device)   # this flight's waypoints come from the strategist
        self.tgt = torch.zeros((n, 3), device=device)
        self.tgt_final = torch.ones(n, dtype=torch.bool, device=device)
        self.need_wp = torch.zeros(n, dtype=torch.bool, device=device)
        self.since_wp = torch.zeros(n, dtype=torch.long, device=device)
        self.prev_dt = torch.ones(n, device=device)
        self.prev_danger = torch.zeros(n, device=device)            # danger potential at the previous step
        self.r_goal = torch.zeros(n, device=device)          # the strategist's reward this step
        self.wp_reached = torch.zeros(n, dtype=torch.long, device=device)
        self.in_goal = torch.zeros(n, dtype=torch.bool, device=device)        # inside the arrival sphere (a success)
        self.goal_best = torch.full((n,), 1e9, device=device)                 # closest approach to the goal so far
        self.wp_missed = torch.zeros(n, dtype=torch.long, device=device)
        self.strat_every = max(1, int(round(STRAT_EVERY_S / DT)))
        self.wp_log = None                                    # fly(record=True): (flight, time, waypoint)

    # ---- targets ----
    def set_targets(self, rows, p, final, log_it=True):
        self.tgt[rows] = p
        self.tgt_final[rows] = final
        self.need_wp[rows] = False
        self.since_wp[rows] = 0
        self.prev_dt[rows] = (p - self.x[rows, :3]).norm(dim=-1)
        self.prev_phi[rows] = -AIM_K
        if log_it and self.record and self.wp_log is not None:
            for b in range(len(rows)):
                self.wp_log.append((int(rows[b]), float(self.t[rows[b]]), p[b].cpu().numpy(), bool(final[b])))

    def z_band(self, rows):
        """Flight-frame heights a waypoint may have (inside the box, WP_ALT_MARGIN_M from floor / ceiling)."""
        oz = self.W["origin"][self.env[rows], 2]
        return self.W["lo"][2] - oz + WP_ALT_MARGIN_M, self.W["hi"][2] - oz - WP_ALT_MARGIN_M

    def sample_wp(self, rows):
        """Stage 1: a random waypoint in free air ahead, or the goal when it is close. In "line" mode:
        the next point WP_LINE_M along the straight line to the goal."""
        import torch
        if len(rows) == 0:
            return
        if self.wp_mode == "line":
            pos = self.x[rows, :3]
            g = self.W["goal"][self.env[rows]]
            rel = g - pos
            dist = rel.norm(dim=-1)
            final = dist <= WP_LINE_M * 1.2
            wp = pos + rel * (WP_LINE_M / dist.clamp_min(1.0))[:, None]
            self.set_targets(rows, torch.where(final[:, None], g, wp), final)
            return
        B, K = len(rows), WP_TRIES
        pos = self.x[rows, :3]
        g = self.W["goal"][self.env[rows]]
        rel = g - pos
        beta = torch.atan2(rel[:, 1], rel[:, 0])
        rnd = torch.rand((3, B, K), generator=self.gen).to(self.dev)
        th = beta[:, None] + (2 * rnd[0] - 1) * math.radians(WP_ANG_DEG)
        d = WP_DIST_M[0] + rnd[1] * (WP_DIST_M[1] - WP_DIST_M[0])
        lo, hi = self.z_band(rows)
        z = torch.minimum(torch.maximum(pos[:, 2:3] + (2 * rnd[2] - 1) * WP_DZ_M, lo[:, None]), hi[:, None])
        cand = torch.stack([pos[:, 0:1] + d * torch.cos(th), pos[:, 1:2] + d * torch.sin(th), z], -1)   # (B, K, 3)
        cand[..., 0] = torch.minimum(cand[..., 0], g[:, 0:1] - 300.0)
        # clearance of the waypoint AND of the straight path to it (a waypoint behind an obstacle it
        # cannot get around teaches crashing, not waypoint flying) ...
        M = WP_PATH_PTS
        env_r, drop_r = self.env[rows], self.drop[rows]

        def path_clr(a_, b_, t0):                                   # (B, K, 3) x2 -> (B, K, M) clearances
            tt = torch.linspace(t0, 1.0, M, device=self.dev)
            p_ = a_[:, :, None, :] + tt[None, None, :, None] * (b_ - a_)[:, :, None, :]
            return sdf(p_.reshape(B, K * M, 3), env_r, self.W, drop=drop_r).reshape(B, K, M)
        pos_k = pos[:, None, :].expand(B, K, 3)
        cp = path_clr(pos_k, cand, 0.15)
        clr = cp.amin(-1)                                                                              # (B, K)
        dvec = cand - pos_k
        u_ = dvec / dvec.norm(dim=-1, keepdim=True).clamp_min(1.0)
        ro = path_clr(cand, cand + WP_RUNOUT_M * u_, 0.1).amin(-1)                                    # beyond it
        ro_ok = ro > 0.5 * WP_CLEAR_M
        ok = (clr > WP_CLEAR_M) & ro_ok
        pick = torch.where(ok.any(-1), ok.float().argmax(-1), torch.minimum(clr, 2 * ro).argmax(-1))
        # ... except some that must be flown around: clear at the waypoint, an obstacle across the
        # straight path far enough ahead to turn, and a clear one-turn detour via a sideways offset
        around = torch.rand(B, generator=self.gen).to(self.dev) < WP_AROUND_FRAC
        if bool(around.any()):
            d_k = (cand - pos_k).norm(dim=-1)                                                          # (B, K)
            tt = torch.linspace(0.15, 1.0, M, device=self.dev)
            hit = cp < 0.0
            first = torch.where(hit, tt[None, None, :] * d_k[..., None], torch.full_like(cp, 1e9)).amin(-1)
            blocked = hit.any(-1) & (first >= WP_AROUND_MIN_M) & (cp[..., -1] > WP_CLEAR_M) & ro_ok
            perp = torch.stack([-dvec[..., 1], dvec[..., 0], torch.zeros_like(dvec[..., 0])], -1)
            perp = perp / perp.norm(dim=-1, keepdim=True).clamp_min(1.0)
            mid = pos_k + 0.5 * dvec
            detour = torch.zeros_like(blocked)
            for off in WP_AROUND_OFF_M:
                for sgn in (1.0, -1.0):
                    via = mid + sgn * off * perp
                    c1 = path_clr(pos_k, via, 0.15).amin(-1)
                    c2 = path_clr(via, cand, 0.0).amin(-1)
                    detour |= (c1 > 0.5 * WP_CLEAR_M) & (c2 > 0.5 * WP_CLEAR_M)
            good = blocked & detour
            use = around & good.any(-1)
            pick = torch.where(use, good.float().argmax(-1), pick)
            self.wp_around[rows] = use
        wp = cand[torch.arange(B, device=self.dev), pick]
        final = rel.norm(dim=-1) < WP_DIST_M[1]
        self.set_targets(rows, torch.where(final[:, None], g, wp), final)

    def wp_from_action(self, rows, a):
        """Stage 2: the strategist's action in [-1, 1]^3 -> the next waypoint (heading offset from the
        line to the goal, distance, height change); beyond the goal = the goal."""
        import torch
        pos = self.x[rows, :3]
        g = self.W["goal"][self.env[rows]]
        rel = g - pos
        th = torch.atan2(rel[:, 1], rel[:, 0]) + a[:, 0] * math.radians(STRAT_ANG_DEG)
        d = WP_DIST_M[0] + (a[:, 1] + 1) / 2 * (WP_DIST_M[1] - WP_DIST_M[0])
        lo, hi = self.z_band(rows)
        z = torch.minimum(torch.maximum(pos[:, 2] + a[:, 2] * STRAT_DZ_M, lo), hi)
        wp = torch.stack([pos[:, 0] + d * torch.cos(th), pos[:, 1] + d * torch.sin(th), z], -1)
        final = rel.norm(dim=-1) <= d
        self.set_targets(rows, torch.where(final[:, None], g, wp), final)

    def decisions_due(self):
        """Stage 2: flights that need a waypoint from the strategist now."""
        return (self.need_wp | ((self.since_wp >= self.strat_every) & ~self.tgt_final)) & ~self.frozen & self.ext

    def strat_obs(self, rows):
        """The strategist's inputs for flights rows: state relative to the GOAL, a fresh short scan,
        where / when its kept far scans were taken; and those far scans as images."""
        import torch
        a = self.af
        x = self.x[rows]
        env = self.env[rows]
        g = self.W["goal"][env]
        rel = g - x[:, :3]
        dh = rel[:, :2].norm(dim=-1).clamp_min(1.0)
        bearing = torch.atan2(rel[:, 1], rel[:, 0]) - x[:, 5]
        gmax = math.radians(a["GAMMA_MAX_DEG"])
        zw = x[:, 2] + self.W["origin"][env, 2]
        H = (self.W["hi"][2] - self.W["lo"][2])
        rays = ray_hits(x[:, :3], ray_dirs(x[:, 5], self.dev), env, self.W, drop=self.drop[rows]) / RAY_RANGE_M
        v = torch.stack([
            (x[:, 3] - a["V_CRUISE"]) / (a["V_MAX"] - a["V_MIN"]), x[:, 4] / gmax,
            torch.sin(bearing), torch.cos(bearing), torch.clamp(torch.atan2(rel[:, 2], dh) / gmax, -3, 3),
            rel.norm(dim=-1) / self.d0[rows], (zw - self.W["lo"][2]) / H, (self.W["hi"][2] - zw) / H,
            torch.sin(x[:, 5]), self.t[rows] / self.tref[rows]], -1)
        return torch.cat([v, rays, self.far_inputs(rows)], -1), self.far[rows].reshape(len(rows), -1).half()

    def far_inputs(self, rows):
        """For each kept far scan: where it was taken relative to the aircraft now (heading frame,
        km), the change of heading since (sin, cos of cone centre minus heading), its age (/ FAR_EVERY_S)."""
        import torch
        x = self.x[rows]
        d = self.far_pos[rows] - x[:, None, :3]                                   # (B, K, 3)
        c, s_ = torch.cos(x[:, 5:6]), torch.sin(x[:, 5:6])
        rel = self.far_dir[rows] - x[:, 5:6]
        age = ((self.t[rows, None] - self.far_t[rows]) / FAR_EVERY_S).clamp(max=10.0)
        f = torch.stack([(c * d[..., 0] + s_ * d[..., 1]) / 1000.0, (-s_ * d[..., 0] + c * d[..., 1]) / 1000.0,
                         d[..., 2] / 1000.0, torch.sin(rel), torch.cos(rel), age], -1)
        return f.reshape(len(rows), -1)

    @staticmethod
    def closest_miss(x, p):
        """How the current line of flight passes the point p: (lateral, vertical) offsets of the
        closest approach in the heading frame (in MISS_SCALE_M, clipped +-3), the miss distance (m)
        and how far ahead along the line it is (m; negative = behind)."""
        import torch
        ga, ch = x[:, 4], x[:, 5]
        vhat = torch.stack([torch.cos(ga) * torch.cos(ch), torch.cos(ga) * torch.sin(ch), torch.sin(ga)], -1)
        d = p - x[:, :3]
        along = (d * vhat).sum(-1)
        perp = d - torch.clamp(along, min=0.0)[:, None] * vhat
        lat = (-torch.sin(ch) * perp[:, 0] + torch.cos(ch) * perp[:, 1]) / MISS_SCALE_M
        vert = perp[:, 2] / MISS_SCALE_M
        return lat.clamp(-3, 3), vert.clamp(-3, 3), perp.norm(dim=-1), along

    def far_scan(self, rows):
        """The cone toward the goal for flights rows: becomes the newest of their kept far scans."""
        import torch
        if len(rows) == 0:
            return
        x = self.x[rows]
        pos = x[:, :3]
        rel = self.W["goal"][self.env[rows]] - pos
        az0 = torch.atan2(rel[:, 1], rel[:, 0])
        el0 = torch.atan2(rel[:, 2], rel[:, :2].norm(dim=-1)).clamp(-math.radians(FAR_EL_MAX_DEG), math.radians(FAR_EL_MAX_DEG))
        ang = az0[:, None] + self.far_az[None]
        el = el0[:, None] + self.far_el[None]
        dirs = torch.stack([torch.cos(el) * torch.cos(ang), torch.cos(el) * torch.sin(ang), torch.sin(el)], -1)
        env, drop = self.env[rows], self.drop[rows]
        dists = []
        for r0 in range(0, dirs.shape[1], FAR_CHUNK):
            d_ = dirs[:, r0:r0 + FAR_CHUNK]
            dist = ray_hits(pos, d_, env, self.W, drop=drop, rng=FAR_RANGE_M)
            dists.append(dist)
            if self.record:
                hit = dist < FAR_RANGE_M * 0.999
                pts = pos[:, None] + d_ * dist[..., None]
                for b in range(len(rows)):
                    self.far_hits.append((int(rows[b]), pts[b][hit[b]].cpu().numpy()))
                    self.far_hits_t.append((int(rows[b]), float(self.t[rows[b]])))        # when (for videos)
        img = (torch.cat(dists, 1) / FAR_RANGE_M).view(len(rows), 1, FAR_EL[0], FAR_AZ[0])
        self.far[rows] = torch.cat([img, self.far[rows, :-1]], 1)                  # newest first
        self.far_pos[rows] = torch.cat([pos[:, None], self.far_pos[rows, :-1]], 1)
        self.far_dir[rows] = torch.cat([az0[:, None], self.far_dir[rows, :-1]], 1)
        self.far_t[rows] = torch.cat([self.t[rows, None], self.far_t[rows, :-1]], 1)

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
        self.prev_danger[idx] = 0.0
        self.prev_phi[idx] = -AIM_K
        self.prev_act[idx] = 0.0
        x = torch.zeros((len(idx), 6), device=self.dev)
        x[:, 3] = a["V_CRUISE"]
        self.x[idx] = x
        D = a["a1"] * a["V_CRUISE"] ** 2 + a["a2"] / a["V_CRUISE"] ** 2
        self.u[idx] = torch.tensor([D, 1.0, 0.0], device=self.dev)
        self.t[idx] = 0.0
        self.crashes[idx] = 0
        self.ended[idx] = False
        self.frozen[idx] = False
        self.hist_x[idx, 0] = x
        self.hist_u[idx, 0] = self.u[idx]
        self.hist_t[idx, 0] = 0.0
        self.hist_len[idx] = 1
        if self.record:
            if self.hist_r is None:
                self.hist_r = torch.zeros((self.n, self.L, N_RAYS), device=self.dev)
                self.far_hits = []
                self.far_hits_t = []
            self.hist_r[idx] = 0.0
        # who sets this flight's waypoints: "external" the strategist, "random" practice waypoints,
        # "mixed" the strategist for a (1 - mix_random) share of flights, practice waypoints for the rest
        if self.wp_mode == "mixed":
            ext = torch.rand(len(idx), generator=self.gen).to(self.dev) >= self.mix_random
        else:
            ext = torch.full((len(idx),), self.wp_mode == "external", dtype=torch.bool, device=self.dev)
        kf = torch.full((len(idx),), float(self.keep_frac), device=self.dev)
        if self.wp_mode == "mixed":                               # practice flights: lighter clutter
            kf = torch.where(ext, kf, kf.clamp(max=PRACTICE_KEEP_MAX))
        if bool((kf >= 1.0).all()):
            self.drop[idx] = False
        else:
            r = torch.rand((len(idx), self.drop.shape[1]), generator=self.gen).to(self.dev)
            self.drop[idx] = r >= kf[:, None]
        self.far[idx] = 1.0                                      # no far scans yet (all "clear", age large)
        self.far_pos[idx] = 0.0
        self.far_dir[idx] = 0.0
        self.far_t[idx] = -1e3
        self.steps[idx] = 0
        self.start_level[idx] = self.level
        self.start_step[idx] = self.clock
        self.wp_around[idx] = False
        self.far_scan(idx)                                       # first look down the course at the start
        self.wp_reached[idx] = 0
        self.wp_missed[idx] = 0
        self.in_goal[idx] = False
        self.goal_best[idx] = 1e9
        if self.record and self.wp_log is None:
            self.wp_log = []
        fin = torch.ones(len(idx), dtype=torch.bool, device=self.dev)
        self.ext[idx] = ext
        if self.wp_mode in ("random", "mixed", "external"):
            ri, ei = idx[~ext], idx[ext]
            if len(ri):
                direct = torch.rand(len(ri), generator=self.gen).to(self.dev) < WP_GOAL_PROB
                self.sample_wp(ri[~direct])
                self.set_targets(ri[direct], g[~ext][direct], fin[:len(ri)][direct])   # some fly straight to the goal
            if len(ei):                                          # the strategist decides before the first step
                self.set_targets(ei, g[ext], ~fin[:len(ei)], log_it=False)
                self.need_wp[ei] = True
        elif self.wp_mode == "line":
            self.sample_wp(idx)
        else:
            self.set_targets(idx, g, fin)

    def sample_envs(self, k):
        import torch
        j = torch.randint(len(self.pool), (k,), generator=self.gen)
        return self.pool[j].to(self.dev)

    def obs(self):
        """-> (flier inputs (N, OBS_DIM), empty image (N, 0)). Runs the short scan (every step) and
        the far scan (flights that are due)."""
        import torch
        a = self.af
        x, u = self.x, self.u
        g = self.W["goal"][self.env]
        rel = self.tgt - x[:, :3]                                 # the target: waypoint or goal
        dh = rel[:, :2].norm(dim=-1).clamp_min(1.0)
        bearing = torch.atan2(rel[:, 1], rel[:, 0]) - x[:, 5]
        gmax = math.radians(a["GAMMA_MAX_DEG"])
        zw = x[:, 2] + self.W["origin"][self.env, 2]
        H = (self.W["hi"][2] - self.W["lo"][2])
        dirs = ray_dirs(x[:, 5], self.dev)
        dist = ray_hits(x[:, :3], dirs, self.env, self.W, drop=self.drop)        # short scan
        self.rays = dist / RAY_RANGE_M
        due = torch.nonzero((self.steps % self.far_every == 0) & (self.steps > 0) & ~self.frozen).squeeze(1)
        self.far_scan(due)
        v = torch.stack([
            (x[:, 3] - a["V_CRUISE"]) / (a["V_MAX"] - a["V_MIN"]), x[:, 4] / gmax,
            torch.sin(bearing), torch.cos(bearing),
            torch.clamp(torch.atan2(rel[:, 2], dh) / gmax, -3, 3),
            (rel.norm(dim=-1) / WP_NORM_M).clamp(0, 4), (zw - self.W["lo"][2]) / H, (self.W["hi"][2] - zw) / H,
            u[:, 0] / self.T_avail(x[:, 3]), u[:, 1] - 1.0, u[:, 2] / math.tan(math.radians(a["BANK_MAX_DEG"])),
            torch.sin(x[:, 5]), self.t / self.tref], -1)
        lat, vert, _, _ = self.closest_miss(x, self.tgt)
        extra = torch.stack([self.tgt_final.float(), (g - x[:, :3]).norm(dim=-1) / self.d0], -1)
        obs = torch.cat([v, torch.stack([lat, vert], -1), self.rays, extra], -1)
        return obs, torch.zeros((self.n, 0), dtype=torch.half, device=self.dev)

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
        phi = torch.where((dist < APPROACH_M) & self.tgt_final, -AIM_K * torch.clamp(miss / MISS_SCALE_M, max=1.0),
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
        self.clock += 1
        self.u_before = self.u
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
        # closest approach to the goal during this step (segment x -> xn): arrival
        seg = xn[:, :3] - x[:, :3]
        s_ = (((g - x[:, :3]) * seg).sum(-1) / (seg * seg).sum(-1).clamp_min(1e-6)).clamp(0, 1)
        d_close = (x[:, :3] + s_[:, None] * seg - g).norm(dim=-1)
        # time to impact along the central horizontal rays (from the lidar seen this step)
        c0 = N_RAYS_FAN // 2
        ahead = self.rays[:, max(0, c0 - 1):c0 + 2].min(-1).values * RAY_RANGE_M if self.rays is not None \
            else torch.full_like(dist, RAY_RANGE_M)
        ttc = ahead / x[:, 3].clamp_min(1.0)
        _, _, miss_now = self.projected_miss(xn, g)
        # the target (waypoint or goal): progress, closest approach this step, passed?
        tg = self.tgt
        dt_ = (tg - xn[:, :3]).norm(dim=-1)
        st_ = (((tg - x[:, :3]) * seg).sum(-1) / (seg * seg).sum(-1).clamp_min(1e-6)).clamp(0, 1)
        dct = (x[:, :3] + st_[:, None] * seg - tg).norm(dim=-1)
        _, _, _, along_t = self.closest_miss(xn, tg)
        wp_hit = ~self.tgt_final & (dct < WP_RADIUS_M)
        wp_pass = ~self.tgt_final & ~wp_hit & (along_t < 0) & (dt_ < WP_PASS_M)
        r_goal = STRAT_PROG_W * (self.prev_d - dist) / self.d0 - STRAT_TIME_W * DT / self.tref
        r = PROG_W * (self.prev_dt - dt_) / self.d0 - TIME_W * DT / self.tref \
            + WP_R * wp_hit.float() - WP_MISS_R * wp_pass.float() * torch.clamp(dt_ / MISS_SCALE_M, max=1.0) \
            - PROX_W * DT * torch.clamp(1 - h_min / PROX_M, 0, 1) \
            - TTC_W * DT * torch.clamp(1 - ttc / TTC_S, 0, 1) \
            - SMOOTH_W * DT * self.smooth_term(act) \
            + self.aim_shaping(dist, miss_now)
        self.prev_d = dist
        self.prev_dt = dt_
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
        # arrival: entering the sphere makes it a success; it keeps flying until past its closest point
        # (or out of the sphere), then ends; anything that ends it meanwhile still ends it as a success
        newly = (d_close < GOAL_RADIUS_M) & ~self.in_goal & ~self.frozen & self.tgt_final   # only once the goal
                                                                                            #   is the target
        self.in_goal = self.in_goal | newly
        self.goal_best = torch.where(self.in_goal, torch.minimum(self.goal_best, d_close), self.goal_best)
        past = (dist > self.goal_best + 1.0) | (dist > GOAL_RADIUS_M)
        goal_end = self.in_goal & (past | ((status >= 2) & (status <= 5)))     # (crossing the goal plane inside is fine)
        status = torch.where(goal_end, torch.full_like(status, 1), torch.where(self.in_goal, torch.zeros_like(status), status))
        prec = torch.clamp(1 - self.goal_best / GOAL_RADIUS_M, 0, 1) ** PRECISION_POW
        term = torch.where(status == 1, SUCCESS_R + PRECISION_R * prec, 0.0) \
            - torch.where((status >= 2) & (status <= 4), CRASH_R, 0.0) \
            - torch.where(status == 5, TIMEOUT_R, 0.0) \
            - torch.where(status == 6, MISS_R * torch.where(self.tgt_final, torch.clamp(miss_at_plane / MISS_SCALE_M, max=1.0),
                                                            torch.ones_like(miss_at_plane)), 0.0)   # skipped its waypoints: full penalty
        r = r + term
        r_goal = r_goal + term
        # potential-based danger shaping: gamma * phi(now) - phi(before); at the end phi = 0
        phi = -TTC_K * torch.clamp(1 - ttc / TTC_S, 0, 1)
        r = r + torch.where(status != 0, -self.prev_danger, GAMMA_RL * phi - self.prev_danger)
        self.prev_danger = torch.where(self.frozen, self.prev_danger, phi)
        self.last_close = d_close
        # ---- finished flights (fly) stay put ----
        live = ~self.frozen
        status = torch.where(live, status, torch.zeros_like(status))
        r = torch.where(live, r, torch.zeros_like(r))
        if bool(self.frozen.any()):
            fz = self.frozen[:, None]
            self.x = torch.where(fz, x, self.x)
            self.u = torch.where(fz, self.u_before, self.u)
            self.t = torch.where(self.frozen, self.t - DT, self.t)
        self.ended = status > 0
        self.last_status = status
        self.steps = self.steps + live.long()
        self.r_goal = torch.where(live, r_goal, torch.zeros_like(r_goal))
        self.since_wp = self.since_wp + live.long()
        nxt = (wp_hit | wp_pass) & live & (status == 0)                     # next target
        self.wp_reached = self.wp_reached + (wp_hit & live).long()
        self.wp_missed = self.wp_missed + (wp_pass & live).long()
        if bool(nxt.any()):
            rows_ = torch.nonzero(nxt).squeeze(1)
            if self.wp_mode == "line":
                self.sample_wp(rows_)
            else:
                ex = self.ext[rows_]
                self.sample_wp(rows_[~ex])                                  # practice waypoints
                self.need_wp[rows_[ex]] = True                              # the strategist picks the next
        ia = torch.nonzero(live).squeeze(1)                                 # record the path
        if len(ia):
            L_ = self.hist_len[ia].clamp(max=self.L - 1)
            self.hist_x[ia, L_] = self.x[ia]
            self.hist_u[ia, L_] = self.u[ia]
            self.hist_t[ia, L_] = self.t[ia]
            if self.record and self.rays is not None:
                self.hist_r[ia, L_ - 1] = self.rays[ia]
            self.hist_len[ia] = L_ + 1
        return r, status > 0, status, u


def make_agent(device, obs_dim=None, bias=(0.4, 0.0, 0.0), img_shape=None):
    """Actor-critic. The flier: make_agent(device) (OBS_DIM inputs, no images, starts near trim thrust,
    no turn). The strategist: make_agent(device, STRAT_DIM, (0, 0, 0), FAR_SHAPE) (its far scans go
    through a small CNN; starts with: toward the goal, 2 km, level)."""
    import torch
    import torch.nn as nn
    obs_dim = OBS_DIM if obs_dim is None else obs_dim

    class ImgEnc(nn.Module):
        def __init__(self, ch, nx, ny):
            super().__init__()
            self.net = nn.Sequential(nn.Conv2d(ch, 16, 3, padding=1), nn.ReLU(),
                                     nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(), nn.Flatten())
            self.fc = nn.Sequential(nn.Linear(32 * ((nx + 1) // 2) * ((ny + 1) // 2), 128), nn.ReLU())

        def forward(self, m):
            return self.fc(self.net(m))

    n_img = 128 if img_shape else 0

    def trunk(out):
        return nn.Sequential(nn.Linear(obs_dim + n_img, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, HIDDEN), nn.Tanh(),
                             nn.Linear(HIDDEN, out))

    class Agent(nn.Module):
        def __init__(self):
            super().__init__()
            if img_shape:
                self.img_a, self.img_c = ImgEnc(*img_shape), ImgEnc(*img_shape)
            self.actor, self.critic = trunk(3), trunk(1)
            nn.init.zeros_(self.actor[-1].weight)
            with torch.no_grad():
                self.actor[-1].bias.copy_(torch.tensor(list(bias)))
            self.log_std = nn.Parameter(torch.full((3,), LOG_STD_INIT))

        def mean(self, o, m):
            if img_shape:
                o = torch.cat([o, self.img_a(m.float().view(-1, *img_shape))], -1)
            return self.actor(o)

        def value(self, o, m):
            if img_shape:
                o = torch.cat([o, self.img_c(m.float().view(-1, *img_shape))], -1)
            return self.critic(o).squeeze(-1)

    return Agent().to(device)


def strategy_decide(sim, strat, rows, explore):
    """The strategist picks the next waypoint for flights rows -> (inputs, far-scan images, action, log-prob, value)."""
    import torch
    o, m = sim.strat_obs(rows)
    mu = strat.mean(o, m)
    if explore:
        std = strat.log_std.clamp(min=LOG_STD_MIN_S).exp()
        a_ = mu + std * torch.randn_like(mu)
        lp = (-0.5 * ((a_ - mu) / std) ** 2 - torch.log(std) - 0.5 * math.log(2 * math.pi)).sum(-1)
        v = strat.value(o, m)
    else:
        a_, lp, v = mu, None, None
    sim.wp_from_action(rows, a_.clamp(-1, 1))
    return o, m, a_, lp, v


def fly(agent, sim_template, W, maps, bench_tf, af, env_rows, device, max_mult=TIME_LIMIT_X, keep_frac=1.0, seed=0,
        record=False, strategy=None, wp_mode=None):
    """Fly the agent (the flier) with its mean action on the given environments; its path (flight
    frame). Targets: strategy given -> the strategist's waypoints (mean action); else wp_mode
    ("goal": straight to the goal, "line": waypoints along the line to it, "random": practice waypoints). record=True also returns
    the lidar along the path, what the far scans hit and the waypoints."""
    import numpy as np
    import torch
    n = len(env_rows)
    sim = Sim(W, None, bench_tf, af, n, device, seed=seed)
    sim.keep_frac = keep_frac
    sim.record = record
    sim.wp_mode = wp_mode or ("external" if strategy is not None else "goal")
    ids = torch.as_tensor(np.asarray(env_rows), device=device)
    sim.reset(torch.arange(n, device=device), ids)
    status = torch.zeros(n, dtype=torch.long, device=device)
    steps = int(max_mult * sim.tref.max().item() / DT) + 2
    with torch.no_grad():
        for _ in range(steps):
            if strategy is not None:
                due = torch.nonzero(sim.decisions_due()).squeeze(1)
                if len(due):
                    strategy_decide(sim, strategy, due, explore=False)
            o, m = sim.obs()
            act = agent.mean(o, m).clamp(-1, 1)
            sim.step(act)
            status = torch.where(sim.ended, sim.last_status, status)
            sim.frozen = sim.frozen | sim.ended
            if sim.frozen.all():
                break
    L = sim.hist_len.cpu().numpy()
    X, U, Tm = sim.hist_x.cpu().numpy(), sim.hist_u.cpu().numpy(), sim.hist_t.cpu().numpy()
    out = []
    gl = W["goal"][ids].cpu().numpy()
    for i in range(n):
        k = int(L[i])
        if int(status[i]) == 1:                                   # cut at the closest point to the goal
            k = int(np.argmin(np.linalg.norm(X[i, :k, :3] - gl[i], axis=1))) + 1
        f = {"status": STATUS[int(status[i])], "t": Tm[i, :k], "X": X[i, :k], "U": U[i, :k],
             "crashes": 0,
             "wp_reached": int(sim.wp_reached[i]), "wp_missed": int(sim.wp_missed[i])}
        if record:
            f["rays"] = sim.hist_r[i, :k].cpu().numpy() * RAY_RANGE_M
            f["far_hits"] = [h for j, h in sim.far_hits if j == i]
            f["far_hits_t"] = [t_ for j, t_ in sim.far_hits_t if j == i]
            f["drop"] = sim.drop[i].cpu()
            f["waypoints"] = [(t_, p_, fin_) for j, t_, p_, fin_ in (sim.wp_log or []) if j == i]
        out.append(f)
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
    Ws = [W] + [{k: (v.to(d) if hasattr(v, "to") else v) for k, v in W.items()} for d in devices[1:]]
    maps = None
    maps_g = [None] * len(devices)

    splits = cache["split"]
    train_rows = np.nonzero(splits == "train")[0]
    train_rows = train_rows[np.argsort(occ[train_rows], kind="stable")]        # least cluttered first
    val_rows = np.sort(np.random.default_rng(cfg["seed"] + 1).permutation(np.nonzero(splits == "val")[0])[:cfg["val_envs"]])
    test_rows = np.nonzero(splits == "test")[0]
    if cfg["test_envs"] is not None:
        test_rows = test_rows[:cfg["test_envs"]]

    import copy
    stage = cfg.get("stage") or "flier"
    cur = {"ent": ENT_COEF_S, "ls_min": LOG_STD_MIN_S} if stage == "strategy" else \
        {"ent": ENT_COEF, "ls_min": LOG_STD_MIN}                 # current exploration settings (flier: annealed)

    def set_exploration():
        if stage == "strategy":
            return
        f_ = min(1.0, max(0.0, (st["update"] - st.setdefault("anneal_start", st["update"])) / max(STD_ANNEAL_UPDATES, 1)))
        cur["ent"] = ENT_COEF * (ENT_COEF_END / ENT_COEF) ** f_
        cur["ls_min"] = LOG_STD_MIN + f_ * (LOG_STD_MIN_END - LOG_STD_MIN)

    def load_run(rname, want_stage, what, prefer="best.pt"):
        """Another run's checkpoint from the Volume (prefer, else the other one), checked for this code.
        Frozen partners and --init-from use checkpoint.pt (the latest weights: the solver evaluations fly
        all obstacles, which says little about a partner still partway up the curriculum)
        (the latest weights)."""
        rdir = V / "rl_pilot" / rname
        other = "checkpoint.pt" if prefer == "best.pt" else "best.pt"
        rpath = rdir / prefer if (rdir / prefer).exists() else rdir / other
        if not rpath.exists():
            raise SystemExit(f"{what} run not found on the Volume: {rdir}")
        ck_ = torch.load(rpath, map_location=device, weights_only=False)
        ver = (ck_.get("settings") or {}).get("OBS_VERSION")
        if ver != OBS_VERSION:
            raise SystemExit(f"{what} run {rname} has OBS_VERSION {ver}, this code needs {OBS_VERSION}")
        rst = ((ck_.get("cfg") or {}).get("stage") or "flier")
        if rst != want_stage:
            raise SystemExit(f"{what} run {rname} is a '{rst}' run, a '{want_stage}' run is needed")
        if rpath.name == "best.pt" and (rdir / "checkpoint.pt").exists():   # curriculum position: the latest
            ck_["state"] = torch.load(rdir / "checkpoint.pt", map_location="cpu", weights_only=False).get("state", ck_.get("state"))
        return ck_, rpath.name

    def frozen(net):
        net.eval()
        for p_ in net.parameters():
            p_.requires_grad_(False)
        return [net] + [copy.deepcopy(net).to(d) for d in devices[1:]]

    flier, flier_reps, strat_reps, fkeep = None, [], [], 1.0
    if stage == "strategy":                                    # the frozen flier
        if not cfg.get("flier"):
            raise SystemExit("stage 'strategy' needs --flier NAME (a trained flier run)")
        fck, fname_ = load_run(cfg["flier"], "flier", "flier", prefer="checkpoint.pt")
        flier = make_agent(device)
        flier.load_state_dict(fck["agent"])
        flier_reps = frozen(flier)
        fst_ = fck.get("state") or {}                            # start where the flier got to
        fkeep = fst_.get("keep", _OLD_KEEP_STAGES[fst_["keep_idx"]] if "keep_idx" in fst_ else 1.0)
        log(f"strategist training; flier = {cfg['flier']} ({fname_}, frozen; it reached {fkeep:.0%} of the obstacles)")
        agent = make_agent(device, STRAT_DIM, (0.0, 0.0, 0.0), FAR_SHAPE)   # the strategist (trained)
    else:
        agent = make_agent(device)                             # the master copy (optimizer, evaluation, saving)
        if cfg.get("strategy"):                                # a frozen strategist sets most waypoints
            sck, sname_ = load_run(cfg["strategy"], "strategy", "strategist", prefer="checkpoint.pt")
            strat = make_agent(device, STRAT_DIM, (0.0, 0.0, 0.0), FAR_SHAPE)
            strat.load_state_dict(sck["agent"])
            strat_reps = frozen(strat)
            log(f"flier training on strategist {cfg['strategy']}'s waypoints ({sname_}, frozen; "
                f"{WP_MIX_RANDOM:.0%} of flights on random practice waypoints)")
        else:
            log("flier training on random practice waypoints")
    if stage == "strategy":                                    # the noise on its own, faster learning rate
        rest_ = [p_ for n_, p_ in agent.named_parameters() if n_ != "log_std"]
        opt = torch.optim.Adam([{"params": rest_}, {"params": [agent.log_std], "lr": LR * STD_LR_MULT_S}],
                               lr=LR, eps=1e-5)
    else:
        opt = torch.optim.Adam(agent.parameters(), lr=LR, eps=1e-5)
    reps = [agent] + [copy.deepcopy(agent).to(d) for d in devices[1:]]

    def sync_replicas():
        with torch.no_grad():
            for r in reps[1:]:
                for pr, p0 in zip(r.parameters(), agent.parameters()):
                    pr.copy_(p0)
    st = {"update": 0, "keep": fkeep if stage == "strategy" else KEEP_START, "pool_frac": CURRICULUM_START, "best_success": -1.0, "since_best": 0, "boosts": 0,
          "best_eval": -float("inf"), "best_eval_update": -1, "evals_done": [], "stopped": False}
    ck = root / "checkpoint.pt"
    if cfg.get("init_from") and not ck.exists():               # a new round: start from another run
        ick, iname_ = load_run(cfg["init_from"], stage, "init-from", prefer="checkpoint.pt")
        agent.load_state_dict(ick["agent"])
        ist = ick.get("state") or {}
        st["keep"] = ist.get("keep", _OLD_KEEP_STAGES[ist["keep_idx"]] if "keep_idx" in ist else st["keep"])
        st["pool_frac"] = ist.get("pool_frac", st["pool_frac"])
        log(f"starting from {cfg['init_from']} ({iname_}): its weights and curriculum position "
            f"(keep {st['keep']:.0%} of the obstacles, {st['pool_frac']:.0%} of the envs)")
    if cfg.get("start_std") is not None and not ck.exists():   # --start-std: a new run explores this much at first
        with torch.no_grad():
            agent.log_std.fill_(math.log(max(1e-3, float(cfg["start_std"]))))
        log(f"exploration noise set to {float(cfg['start_std']):.2f} (--start-std)")
    if cfg.get("start_keep") is not None and not ck.exists():  # --start-keep: a new run starts at this share
        st["keep"] = min(1.0, max(0.0, float(cfg["start_keep"])))
        log(f"starting at {st['keep']:.0%} of the obstacles (--start-keep)")
    if cfg.get("start_pool") is not None and not ck.exists():  # --start-pool: ... and this share of the envs
        st["pool_frac"] = min(1.0, max(0.0, float(cfg["start_pool"])))
        log(f"starting with the least cluttered {st['pool_frac']:.0%} of the training environments (--start-pool)")
    if cfg.get("fixed_difficulty"):
        log("difficulty held fixed (--fixed-difficulty): the curriculum will not change it")
    if cfg.get("crash_penalty") is not None:                   # --crash-penalty: post-training on safety
        globals()["CRASH_R"] = float(cfg["crash_penalty"])
        log(f"crash / out-of-bounds / stall penalty {CRASH_R:g} (default 25; a missed goal costs up to "
            f"{MISS_R:g}, a timeout {TIMEOUT_R:g}) (--crash-penalty)")
    if ck.exists():
        d = torch.load(ck, map_location=device, weights_only=False)
        agent.load_state_dict(d["agent"]); st = d["state"]
        try:
            opt.load_state_dict(d["opt"])
        except ValueError:                                     # (saved before the separate noise rate)
            log("optimizer state from the checkpoint does not match this version: starting it fresh")
        log(f"resumed at update {st['update']}")
        if "det_hist" not in st:                       # checkpoint from before the noise-free checks
            st.update(best_success=-1.0, since_best=0, boosts=0, det_hist=[])
        if "keep" not in st:                           # checkpoint from the stage-based curriculum
            st["keep"] = _OLD_KEEP_STAGES[st.get("keep_idx", len(_OLD_KEEP_STAGES) - 1)]
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
    set_keep(st["keep"])
    for sm in sims:
        sm.level = st.setdefault("level", 0)
    for sm in sims:
        sm.wp_mode = "external" if stage == "strategy" else ("mixed" if strat_reps else "random")
        sm.reset(torch.arange(sm.n, device=sm.dev), sm.sample_envs(sm.n))
    # stage 2: each flight's open strategist decision (kept across updates until it is complete)
    pend = [{"has": torch.zeros(sm.n, dtype=torch.bool, device=sm.dev),
             "O": torch.zeros((sm.n, STRAT_DIM), device=sm.dev), "M": torch.zeros((sm.n, MAP_FLAT), dtype=torch.half, device=sm.dev),
             "A": torch.zeros((sm.n, 3), device=sm.dev), "LP": torch.zeros(sm.n, device=sm.dev),
             "V": torch.zeros(sm.n, device=sm.dev), "R": torch.zeros(sm.n, device=sm.dev)} for sm in sims]
    order = [0]

    def fly_eval(rows, keep_frac=1.0, seed=0, check=False):
        """Noise-free flights: a flier on practice waypoints (checks) or on waypoints along the straight
        line to the goal (solver evaluations); a flier round with a strategist, or a strategist run -> the
        pair (the strategist's waypoints)."""
        if stage == "strategy":
            return fly(flier, None, W, maps, bench_tf, af, rows, device, keep_frac=keep_frac, seed=seed, strategy=agent)
        if strat_reps:                                          # flier round with a strategist: fly the pair
            return fly(agent, None, W, maps, bench_tf, af, rows, device, keep_frac=keep_frac, seed=seed,
                       strategy=strat_reps[0])
        return fly(agent, None, W, maps, bench_tf, af, rows, device, keep_frac=keep_frac, seed=seed,
                   wp_mode="random" if check else "line")
    ep_ret = [torch.zeros(sm.n, device=sm.dev) for sm in sims]
    recent = []                                                    # finished flights (status, return, tf ratio, crashes)

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
        flights = fly_eval(rows)
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
                    torch.save({"agent": p["snapshot"], "state": dict(st), "cfg": cfg,
                                "settings": {k_: v_ for k_, v_ in globals().items() if k_.isupper()
                                             and isinstance(v_, (int, float, str, bool, type(None)))}}, buf)
                    write_bytes_atomic(root / "best.pt", buf.getvalue())
                    log(f"  new best agent (update {p['update']})")
            else:
                write_bytes_atomic(root / "test_summary.json", json.dumps(rows, indent=1).encode())
            commit()

    def flier_rollout():
        """Stage 1: ROLLOUT steps of every flight with exploration noise -> per-GPU PPO data."""
        nonlocal recent
        buf = [{"O": [], "M": [], "A": [], "LP": [], "R": [], "D": [], "V": []} for _ in range(G)]
        with torch.no_grad():
            for _ in range(ROLLOUT):
                outs = []
                for g, sm in enumerate(sims):                 # launch every GPU's work first (runs in parallel)
                    ag = reps[g]
                    if strat_reps:                            # the frozen strategist's waypoints, where due
                        due = torch.nonzero(sm.decisions_due()).squeeze(1)
                        if len(due):
                            strategy_decide(sm, strat_reps[g], due, explore=False)
                    o, m = sm.obs()
                    mu = ag.mean(o, m)
                    std = ag.log_std.clamp(min=cur['ls_min']).exp()
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
                    idx = torch.nonzero(sm.ended).squeeze(1)      # flights that finished
                    if len(idx):
                        stt, tt = status[idx].cpu().numpy(), sm.t[idx].cpu().numpy()
                        ee, rr = sm.env[idx].cpu().numpy(), ep_ret[g][idx].cpu().numpy()
                        wps = list(zip(sm.wp_reached[idx].tolist(), sm.wp_missed[idx].tolist()))
                        lvs, sts, exs = sm.start_level[idx].tolist(), sm.start_step[idx].tolist(), sm.ext[idx].tolist()
                        for ii, (s_, t_, e_, r_) in enumerate(zip(stt, tt, ee, rr)):
                            recent.append((int(s_), float(r_), float(t_ / bench_tf[int(e_)]) if s_ == 1 else float("nan"), wps[ii], lvs[ii], sts[ii], exs[ii]))
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
        return data

    def strategy_rollout():
        """Stage 2: STRAT_ROLLOUT simulator steps; the frozen flier flies every step, the strategist
        decides when a flight needs a waypoint. One PPO sample per completed decision: its reward is
        everything the flight earned until the next decision (or the end of the flight)."""
        nonlocal recent
        recs = [[] for _ in range(G)]

        def close(g, rows, done, nv):
            pg = pend[g]
            recs[g].append({"fid": rows.clone(), "ord": torch.full_like(rows, order[0]), "O": pg["O"][rows].clone(),
                            "M": pg["M"][rows].clone(), "A": pg["A"][rows].clone(), "LP": pg["LP"][rows].clone(),
                            "V": pg["V"][rows].clone(), "R": pg["R"][rows].clone(),
                            "D": torch.full_like(pg["R"][rows], float(done)), "NV": nv})
            order[0] += 1
            pg["has"][rows] = False

        with torch.no_grad():
            for _ in range(STRAT_ROLLOUT):
                for g, sm in enumerate(sims):
                    pg = pend[g]
                    due = torch.nonzero(sm.decisions_due()).squeeze(1)
                    if len(due):
                        o_s, m_s, a_s, lp_s, v_s = strategy_decide(sm, reps[g], due, explore=True)
                        prev = pg["has"][due]
                        if bool(prev.any()):
                            close(g, due[prev], False, v_s[prev])
                        pg["O"][due], pg["M"][due], pg["A"][due] = o_s, m_s, a_s
                        pg["LP"][due], pg["V"][due], pg["R"][due] = lp_s, v_s, 0.0
                        pg["has"][due] = True
                    o, m = sm.obs()
                    act = flier_reps[g].mean(o, m).clamp(-1, 1)
                    sm.step(act)
                    pg["R"] += sm.r_goal * pg["has"].float()
                    ep_ret[g] += sm.r_goal
                    idx = torch.nonzero(sm.ended).squeeze(1)
                    if len(idx):
                        h_ = pg["has"][idx]
                        if bool(h_.any()):
                            close(g, idx[h_], True, torch.zeros(int(h_.sum()), device=sm.dev))
                        stt, tt = sm.last_status[idx].cpu().numpy(), sm.t[idx].cpu().numpy()
                        ee, rr = sm.env[idx].cpu().numpy(), ep_ret[g][idx].cpu().numpy()
                        wps = list(zip(sm.wp_reached[idx].tolist(), sm.wp_missed[idx].tolist()))
                        lvs, sts, exs = sm.start_level[idx].tolist(), sm.start_step[idx].tolist(), sm.ext[idx].tolist()
                        for ii, (s_, t_, e_, r_) in enumerate(zip(stt, tt, ee, rr)):
                            recent.append((int(s_), float(r_), float(t_ / bench_tf[int(e_)]) if s_ == 1 else float("nan"), wps[ii], lvs[ii], sts[ii], exs[ii]))
                        ep_ret[g][idx] = 0.0
                        sm.reset(idx, sm.sample_envs(len(idx)))
        data = []
        for g in range(G):
            rg = recs[g]
            if not rg:
                data.append(None)
                continue
            cat = {k_: torch.cat([r_[k_] for r_ in rg]) for k_ in rg[0]}
            fid, od = cat["fid"].cpu().numpy(), cat["ord"].cpu().numpy()
            R, D, Vv, NV = (cat[k_].cpu().numpy() for k_ in ("R", "D", "V", "NV"))
            srt = np.lexsort((od, fid))
            adv = np.zeros(len(R), np.float32)
            gae, last_f = 0.0, -1
            for i in srt[::-1]:                                   # backward in time, per flight
                if fid[i] != last_f:
                    gae, last_f = 0.0, fid[i]
                delta = R[i] + GAMMA_S * NV[i] * (1 - D[i]) - Vv[i]
                gae = delta + GAMMA_S * LAMBDA_S * (1 - D[i]) * gae
                adv[i] = gae
            adv_t = torch.as_tensor(adv, device=cat["R"].device)
            data.append({"O": cat["O"], "M": cat["M"], "A": cat["A"], "LP": cat["LP"],
                         "adv": adv_t, "ret": adv_t + cat["V"]})
        return data if any(d_ is not None for d_ in data) else None      # None entries: GPUs with no
                                                                            #   completed decision yet

    def ppo_update(data):
        """Clipped PPO epochs over per-GPU data (O, M, A, LP, adv, ret) -> loss info."""
        info = {"pl": 0.0, "vl": 0.0, "ent": 0.0, "clip": 0.0}
        nb = 0
        mb_g = max(1, MINIBATCH // G)                       # each GPU's share of a minibatch
        n_mb = max(1, math.ceil(max(len(dg["adv"]) for dg in data if dg is not None) / mb_g))
        for _ in range(EPOCHS):
            perms = [None if dg is None else torch.randperm(len(dg["adv"]), device=dg["adv"].device) for dg in data]
            for j in range(n_mb):
                losses = []
                for g in range(len(data)):                    # forward/backward on every GPU (in parallel)
                    ag, dg = reps[g], data[g]
                    if dg is None:
                        continue
                    bidx = perms[g][j * mb_g:(j + 1) * mb_g]
                    if len(bidx) == 0:
                        continue
                    mu = ag.mean(dg["O"][bidx], dg["M"][bidx])
                    ls = ag.log_std.clamp(min=cur['ls_min'])
                    std = ls.exp()
                    lp = (-0.5 * ((dg["A"][bidx] - mu) / std) ** 2 - ls - 0.5 * math.log(2 * math.pi)).sum(-1)
                    ratio = (lp - dg["LP"][bidx]).exp()
                    ab = dg["adv"][bidx]
                    ab = (ab - ab.mean()) / (ab.std() + 1e-8)
                    pl = -torch.min(ratio * ab, ratio.clamp(1 - CLIP, 1 + CLIP) * ab).mean()
                    vl = ((ag.value(dg["O"][bidx], dg["M"][bidx]) - dg["ret"][bidx]) ** 2).mean()
                    ent = (ls + 0.5 * (1 + math.log(2 * math.pi))).sum()
                    loss = pl + VF_COEF * vl - cur['ent'] * ent
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
                    agent.log_std.clamp_(min=cur['ls_min'])
                sync_replicas()
                if not losses:
                    continue
                for pl_, vl_, en_, cl_ in losses:
                    info["pl"] += pl_.item() / len(losses); info["vl"] += vl_.item() / len(losses)
                    info["ent"] += en_.item() / len(losses); info["clip"] += cl_.item() / len(losses)
                nb += 1
        info = {kk: vv / max(nb, 1) for kk, vv in info.items()}
        return info

    # ---- PPO ----
    t_start = time.time()
    while st["update"] < cfg["updates"] and not st["stopped"]:
        set_exploration()
        k = st["update"]
        if k % cfg["eval_every"] == 0 and k not in st["evals_done"]:
            start_eval(k, val_rows, "val", ["pilot", "straight"] if k == 0 else ["pilot"], "VAL")
            st["evals_done"].append(k)
        poll_evals()
        t0 = time.time()
        if stage == "strategy":
            data = strategy_rollout()
        else:
            data = flier_rollout()
        info = ppo_update(data) if data else {"pl": 0.0, "vl": 0.0, "ent": 0.0, "clip": 0.0}

        # ---- stats, curriculum, stagnation ----
        recent = recent[-8000:]
        last = recent[-max(1, min(len(recent), 2 * n)):]
        cnt = lambda c: 100 * sum(e_[0] == c for e_ in last) / max(len(last), 1)
        succ = cnt(1)
        ratios = [e_[2] for e_ in last if e_[0] == 1]
        wr = sum(e_[3][0] for e_ in last)
        wm = sum(e_[3][1] for e_ in last)
        wp_pct = 100 * wr / max(wr + wm, 1)
        steps_done = (STRAT_ROLLOUT if stage == "strategy" else ROLLOUT) * n
        st["update"] = k + 1
        action = ""
        det = det_wp = det_crash = ""
        at_end = (st["keep"] >= 1.0 - 1e-9 and st["pool_frac"] >= 1.0 - 1e-9) or bool(cfg.get("fixed_difficulty"))
        # curriculum: the training flights that started at the current difficulty
        # Judged on the FIRST CURRICULUM_MIN_FLIGHTS flights to start at it, once all of them have
        # finished: taking whichever finish first would over-count crashes (they end early).
        mixed_ = stage != "strategy" and bool(strat_reps)       # flier round with a strategist: judge its flights
        win = [e_ for e_ in recent if len(e_) > 6 and e_[4] == st["level"] and (e_[6] or not mixed_)]
        fly_now = torch.cat([sm.start_step[(sm.start_level == st["level"]) & (sm.ext | (not mixed_))].cpu()
                             for sm in sims]).tolist()
        stamps = sorted([e_[5] for e_ in win] + fly_now)
        ready = len(stamps) >= CURRICULUM_MIN_FLIGHTS
        if ready:
            cut = stamps[CURRICULUM_MIN_FLIGHTS - 1]
            ready = not any(s_ <= cut for s_ in fly_now)            # the whole cohort has finished
            win = [e_ for e_ in win if e_[5] <= cut]
        if ready:
            sw = sum(e_[0] == 1 for e_ in win) / len(win)
            wrw, wmw = sum(e_[3][0] for e_ in win), sum(e_[3][1] for e_ in win)
            margin = sw - CURRICULUM_SUCCESS
            if stage != "strategy" and not mixed_:
                margin = min(margin, wrw / max(wrw + wmw, 1) - CURRICULUM_WP)
            bump = CURRICULUM_GAIN * margin
            if cfg.get("stop_at_keep") is not None and st["keep"] >= cfg["stop_at_keep"] - 1e-9 and bump > 0:
                st["stopped"] = True                             # mastered the requested obstacle share
                action = f"stop: above the thresholds at {st['keep']:.0%} of the obstacles (--stop-at-keep)"
            elif bump > 0 and not at_end:
                if st["keep"] < 1.0 - 1e-9:
                    st["keep"] = min(1.0, st["keep"] + bump)
                    set_keep(st["keep"])
                    action = f"curriculum +{100 * bump:.1f} -> keep {st['keep']:.0%} of the obstacles"
                else:
                    st["pool_frac"] = min(1.0, st["pool_frac"] + bump)
                    set_pool()
                    action = f"curriculum +{100 * bump:.1f} -> {st['pool_frac']:.0%} of training envs"
                st["best_success"], st["since_best"] = -1.0, 0
            elif not at_end:
                action = "curriculum hold"
            # either way, a fresh group of flights is judged next (a hold must not keep re-reading the
            # same group: the agent keeps improving)
            st["level"] += 1
            for sm in sims:
                sm.level = st["level"]
            st["cohort"] = (round(100 * sw, 1), round(100 * wrw / max(wrw + wmw, 1), 1))
        if (k + 1) % CHECK_EVERY == 0:
            # noise-free check flights at the current difficulty (log, stall detection)
            pool = sims[0].pool.numpy()
            rows_c = np.random.default_rng(cfg["seed"] * 7919 + k).choice(pool, CHECK_FLIGHTS, replace=len(pool) < CHECK_FLIGHTS)
            fl = fly_eval(rows_c, keep_frac=st["keep"], seed=k, check=True)
            det = 100 * sum(f["status"] == "goal" for f in fl) / len(fl)
            wr_, wm_ = sum(f["wp_reached"] for f in fl), sum(f["wp_missed"] for f in fl)
            det_wp = 100 * wr_ / max(wr_ + wm_, 1)
            det_crash = 100 * sum(f["status"] in ("crash", "out_of_bounds", "stall") for f in fl) / len(fl)
            if cfg.get("stop_crash_pct") is not None and det_crash <= cfg["stop_crash_pct"] + 1e-9:
                st["stopped"] = True                                 # --stop-crash: target reached
                action = (f"stop: noise-free crash rate {det_crash:.1f}% <= {cfg['stop_crash_pct']:g}% "
                          f"({len(fl)} check flights, --stop-crash)")
            hist = st.setdefault("det_hist", [])
            hist.append(det)
            score2 = float(np.mean(hist[-2:]))
            # watch for a stall (each bump resets the count)
            if st["best_success"] < 0 or score2 > st["best_success"] + 1.0:
                st["best_success"], st["since_best"] = score2, 0
                if not at_end:
                    st["boosts"] = 0
            else:
                st["since_best"] += CHECK_EVERY
            if st["since_best"] >= PLATEAU_UPDATES:
                st["since_best"] = 0
                st["boosts"] += 1
                std_now = agent.log_std.clamp(min=cur['ls_min']).exp()
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
        sig = agent.log_std.clamp(min=cur['ls_min']).exp().tolist()
        row = {"update": k + 1, "obstacles_kept": round(st["keep"], 3), "pool_frac": round(st["pool_frac"], 3),
               "episodes": len(last),
               "goal_pct": round(succ, 2), "goal_noise_free_pct": round(det, 2) if det != "" else "",
               "waypoints_noise_free_pct": round(det_wp, 2) if det_wp != "" else "",
               "crash_noise_free_pct": round(det_crash, 2) if det != "" else "",
               "crash_pct": round(cnt(2) + cnt(3), 2), "stall_pct": round(cnt(4), 2),
               "timeout_pct": round(cnt(5), 2), "missed_pct": round(cnt(6), 2),
               "waypoints_reached_pct": round(wp_pct, 2), "return_mean": round(float(np.mean([e_[1] for e_ in last])) if last else 0.0, 3),
               "tf_vs_benchmark": round(float(np.median(ratios)), 4) if ratios else "",
               "std_T": round(sig[0], 3), "std_nv": round(sig[1], 3), "std_nh": round(sig[2], 3),
               **{kk: round(vv, 4) for kk, vv in info.items()},
               "steps_per_s": round(steps_done / (time.time() - t0)), "action": action,
               "elapsed_min": round((time.time() - t_start) / 60, 1)}
        append_csv(root / "updates.csv", [row])
        quiet = action.startswith("curriculum +") or action == "curriculum hold"
        if (k + 1) % 5 == 0 or (action and not quiet) or (k + 1) % CHECK_EVERY == 0:
            log(f"update {k + 1:5d}: goal {succ:5.1f}%" + (f" (noise-free {det:5.1f}%, crash {det_crash:4.1f}%, waypoints {det_wp:5.1f}%)" if det != "" else "")
                + f"  crash {row['crash_pct']:5.1f}%  stall {row['stall_pct']:4.1f}%  "
                f"timeout {row['timeout_pct']:4.1f}%  missed {row['missed_pct']:4.1f}%  waypoints reached {wp_pct:4.1f}%  tf/bench {row['tf_vs_benchmark'] or float('nan'):.3f}  "
                f"obst {st['keep']:.0%} pool {st['pool_frac']:.0%}  std {sig[1]:.2f} (floor {math.exp(cur['ls_min']):.2f})  [{row['steps_per_s']} steps/s] {'' if quiet else action}"
                + (f"  (last judged group: goal {st['cohort'][0]:.0f}%, waypoints {st['cohort'][1]:.0f}%)" if st.get("cohort") else ""))
        if (k + 1) % CHECKPOINT_EVERY == 0 or (action and action != "curriculum hold"):
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
             test_envs: int = -1, seed: int = SEED, download_only: bool = False, stage: str = STAGE,
             flier: str = "", strategy: str = "", init_from: str = "", stop_at_keep: float = -1.0,
             start_keep: float = -1.0, start_std: float = -1.0, start_pool: float = -1.0,
             fixed_difficulty: bool = False, stop_crash: float = -1.0, crash_penalty: float = -1.0):
        if not download_only:
            _ensure_inputs(dataset, run)
            cfg = {"dataset": dataset, "run": run, "name": name, "updates": updates, "n_envs": n_envs,
                   "eval_every": eval_every, "val_envs": val_envs,
                   "test_envs": None if test_envs < 0 else test_envs, "seed": seed, "n_gpus": N_GPUS,
                   "stage": stage, "flier": flier or None, "strategy": strategy or None,
                   "init_from": init_from or None, "stop_at_keep": None if stop_at_keep < 0 else stop_at_keep,
                   "start_keep": None if start_keep < 0 else start_keep,
                   "start_std": None if start_std < 0 else start_std,
                   "start_pool": None if start_pool < 0 else start_pool,
                   "fixed_difficulty": fixed_difficulty, "stop_crash_pct": None if stop_crash < 0 else stop_crash,
                   "crash_penalty": None if crash_penalty < 0 else crash_penalty}
            if stage == "flier" and flier:
                raise SystemExit("--flier is for strategist runs; a flier run takes --strategy NAME (optional)")
            if stage == "strategy" and (strategy or stop_at_keep >= 0):
                raise SystemExit("--strategy and --stop-at-keep are for flier runs")
            if stage not in ("flier", "strategy"):
                raise SystemExit("--stage must be 'flier' or 'strategy'")
            if stage == "strategy" and not flier:
                raise SystemExit("--stage strategy needs --flier NAME (the stage-1 run to fly the waypoints)")
            print(f"Starting {stage} training '{name}' on Modal: {json.dumps(cfg)}", flush=True)
            train_remote.remote(cfg)
        _download(name)


if __name__ == "__main__":
    raise SystemExit("Run with Modal:  modal run --detach supervised-learning-main/rl_pilot.py --name NAME")