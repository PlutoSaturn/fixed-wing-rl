#!/usr/bin/env python3
"""
ardupilot_mission.py - Fly the C172 in ArduPilot (ArduPlane SITL with the JSBSim
Cessna 172) and check the flights.

  box      Tier 1: writes a simple test mission (take off, fly a rectangle at
           cruise speed, return home) and prints the exact commands to fly it.
  summary  Tier 1: reads that flight's telemetry log and reports how well the
           autopilot holds the C172's cruise airspeed, altitude and course, and
           the throttle it needs for level flight (to set TRIM_THROTTLE).
  make     Tier 2: turns one SCP plan into an ArduPilot mission (.waypoints file)
           and prints the exact commands to fly it.
  analyze  Tier 2: reads the telemetry log of that flight, converts the flown path
           back into the workspace frame, and reports tracking error and obstacle
           clearance (ArduPilot does not know the obstacles; only this check does).

Run from the project root (in PyCharm's terminal); files are read from and written
to localstore/. Step-by-step commands: docs/ArduPilot_Commands.md.
    python -m ardupilot.ardupilot_mission box
    python -m ardupilot.ardupilot_mission summary --mission box.waypoints --log box.tlog --plot
    python -m ardupilot.ardupilot_mission make --envs tfr_env.json --results tfr_results.json --index 0
    python -m ardupilot.ardupilot_mission analyze --mission tfr_env0.waypoints --log tfr_env0.tlog --plot

--------------------------------------------------------------------------
The SCP mission (make)
--------------------------------------------------------------------------
  1. Take off from a home point placed upstream of the workspace (outside it),
  2. set the plan's airspeed,
  3. climb in a loiter, still outside the workspace, to the plan's start altitude,
  4. fly a straight lead-in leg (LEAD_IN_TURN_RADII turn radii) lined up with the
     plan's start heading, so the aircraft reaches the start level, at speed and
     pointed the way the planner assumed,
  5. fly the plan as waypoints: closer together where it turns, further apart on
     straight parts,
  6. continue straight out of the workspace and loiter there.

Frames: the workspace is local East-North-Up (x east, y north, z up) about
ORIGIN_LAT/LON (from sim/jsbsim_c172.py). Altitudes are sent relative to home:
    altitude above home = z + FLOOR_AGL_M
so the workspace floor is FLOOR_AGL_M above the simulated ground (300 m, the same
as ALT_BASE_M in the JSBSim tracking runs). Start SITL with its home at the point
the script prints, at altitude 0.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import struct
from pathlib import Path

import numpy as np

import sim.jsbsim_c172 as fly
from environment.envgen import find_input, localstore_path, project_root

# ==========================================================================
# USER SETTINGS
# ==========================================================================
FLOOR_AGL_M          = fly.ALT_BASE_M   # workspace z = 0 is this far above SITL ground (home)
LEAD_IN_TURN_RADII   = 3.0              # straight lead-in before the plan's start
LOITER_RADIUS_M      = 600.0            # climb / final loiter radius (= WP_LOITER_RAD in c172_sitl.parm)
LOITER_GAP_M         = 300.0            # gap between a loiter circle and the workspace
TAKEOFF_ALT_M        = 150.0            # above home
TAKEOFF_PITCH_DEG    = 10.0
MAX_WP_SPACING_M     = 600.0            # plan waypoints: at most this far apart ...
MIN_WP_SPACING_M     = 200.0            #   ... at least this far apart (keep > 1.5 x WP_RADIUS)
MAX_WP_TURN_DEG      = 12.0             #   ... and a new one whenever the heading has turned this much
CHORD_TOL_M          = 3.0              #   ... or when the straight leg would stray more than this from
                                        #   the plan (on curves the leg cuts inside the curve; on the
                                        #   TFR arc 600 m legs cut 8 m into the clearance band)
EXIT_MAX_TURN_DEG    = 90.0             # exit after the goal: turn at most this much to find a clear way out
AP_BANK_DEG          = 30.0             # bank used to model the autopilot's turns on the exit
TECS_CLMB_MAX        = 5.0              # m/s, as in c172_sitl.parm (used for a warning only)
TECS_SINK_MAX        = 6.0
SITL_PARAM_FILE      = "ardupilot/c172_sitl.parm"           # relative to the project root
AIRFRAME_FILE        = "aircraftmodel/c172_params.json"     # the planner's C172 (speeds, bank limit)
TRIM_THROTTLE_PCT    = 71.0             # as in c172_sitl.parm (summary compares the flight with it)
AP_ROLL_LIMIT_DEG    = 45.0             # ROLL_LIMIT_DEG in c172_sitl.parm (most bank the autopilot may use)
SPEED_PROFILE        = "constant"       # "constant": one airspeed for the whole mission, the plan's mean.
                                        # "plan": before each waypoint leg, command the plan's mean airspeed
                                        #   over that leg (DO_CHANGE_SPEED). Tried in the Tier 3 pilot
                                        #   (batch tier3_v2_speed): ArduPilot followed the speeds 5-10 s late,
                                        #   the throttle swung 0-100%, and the hardest plan tracked worse,
                                        #   so "constant" is the default (tracking setup v2)
SPEED_CMD_STEP_MS    = 0.5              # "plan": only add a speed command when it changes by at least this
SPEED_HEADROOM_MS    = 1.5              # "plan": stay this far below the C172's V_MAX (TECS needs some margin)
SPEED_SPREAD_WARN_MS = 12.0             # "constant": warn when the plan's speed range is wider than this
                                        #   (ArduPilot flies one airspeed, so the timing will differ)
FLY_MIN_CLEARANCE_FRAC = 0.5            # a plan is "not flyable as waypoints" when the expected flown
                                        #   path (legs + the autopilot's turns) comes closer to an obstacle
                                        #   than this fraction of the planner's margin

# box (Tier 1 test mission), relative to home; the runway points along --heading
BOX_START_M          = 2000.0           # first corner this far ahead of home (room to climb)
BOX_LENGTH_M         = 4000.0           # long legs (along the runway heading) ...
BOX_WIDTH_M          = 2000.0           # ... and short legs (to the right)
BOX_ALT_M            = 300.0            # above home
BOX_LAPS             = 2
# summary: a sample counts as "settled on a leg" when it is within LEG_TOL_M of a
# leg, at least SETTLE_M past the corner it came from and at least LEG_END_M before
# the next one, from the first leg the aircraft starts after reaching the box altitude
LEG_TOL_M            = 150.0
SETTLE_M             = 800.0
LEG_END_M            = 300.0

# analyze: the plan segment of the flight starts/ends where the aircraft passes
# within this distance of the plan's start/goal
PASS_TOL_M           = 300.0
# analyze: oscillation (weaving) of roll, pitch and altitude. Each signal minus its
# moving average over OSC_WINDOW_S leaves the fast wobble; its RMS and dominant
# period are reported. The first OSC_SKIP_S of the plan (the turn onto the first
# leg) are left out so the start transient does not count as oscillation.
OSC_WINDOW_S         = 20.0
OSC_SKIP_S           = 30.0
# analyze: every analyzed flight is archived in localstore/runs/<log name>/ (log, results,
# plot, and the autopilot parameters it flew with), and localstore/runs/runs.csv lists all
# runs in the order they were flown with the parameters changed since the previous run.
# Parameters that change by themselves (statistics, barometer calibration, telemetry
# rates) are left out of that comparison.
RUNS_DIR             = "runs"

# ---- Tier 3: batches of generated environments (batch-make, run_batch.py, batch-analyze) ----
BATCH_DIR            = "batches"         # localstore/batches/<batch name>/
BATCH_CHORD_TOL_M    = 1.0               # waypoint chord tolerance for batch missions (tracking setup v1)
BATCH_SPEEDUP        = 5                 # SITL --speedup used by run_batch.py (JSBSim manages about 2x)
# pass / fail for one flight (see evaluate_flight); every check must pass
PASS_MAX_ERROR_FRAC  = 1.0               # max distance from the plan < this x the planner's margin
PASS_RMS_M           = 10.0              # rms distance from the plan
PASS_GOAL_M          = 50.0              # closest approach to the plan's goal
PASS_TIME_FRAC       = 0.10              # flight time within this fraction of the plan's
NEAR_MISS_FRAC       = 0.5               # flagged (not failed): flown clearance below this x margin
PARAM_IGNORE         = ("STAT_", "BARO", "MAV", "SR0_", "SR1_", "SR2_", "SR3_", "MIS_")

R_EARTH = fly.R_EARTH
GRAVITY = 9.81


# --------------------------------------------------------------------------
# frames
# --------------------------------------------------------------------------
def enu_to_geo(x, y):
    return fly.enu_to_geo(np.asarray(x, float), np.asarray(y, float))


def geo_to_enu(lat, lon):
    lat0 = np.radians(fly.ORIGIN_LAT_DEG)
    x = np.radians(np.asarray(lon, float) - fly.ORIGIN_LON_DEG) * R_EARTH * np.cos(lat0)
    y = np.radians(np.asarray(lat, float) - fly.ORIGIN_LAT_DEG) * R_EARTH
    return x, y


def compass_deg(chi):
    """Planner course (from east, counter-clockwise) -> compass heading (from north, clockwise)."""
    return float(np.degrees(np.pi / 2 - chi) % 360)


def wsl_path(path):
    """A path on this computer as WSL sees it: C:\\Users\\... -> /mnt/c/Users/..."""
    p = str(path)
    if p.startswith("/"):                        # already a Linux/WSL path
        return p
    if not re.match(r"^[A-Za-z]:[\\/]", p):
        p = str(Path(p).resolve())
    m = re.match(r"^([A-Za-z]):[\\/](.*)$", p)
    return f"/mnt/{m.group(1).lower()}/{m.group(2).replace(chr(92), '/')}" if m else p


def load_airframe():
    """The planner's C172 parameters (aircraftmodel/c172_params.json)."""
    path = project_root() / AIRFRAME_FILE
    if not path.exists():
        raise SystemExit(f"{path} not found; it holds the C172's speeds and bank limit.")
    return json.load(open(path))


def sitl_commands(home, mission_path):
    """The WSL and MAVProxy commands to fly a mission file."""
    parm = wsl_path(project_root() / SITL_PARAM_FILE)
    return f"""In WSL (Ubuntu terminal):
  cd ~/ardupilot/ArduPlane
  sim_vehicle.py -v ArduPlane -f jsbsim:c172x --custom-location={home} \\
      --add-param-file={parm} --console --map -w
  (-w starts from clean parameters; keep it while you are changing c172_sitl.parm)

In the MAVProxy console, once it reports the EKF is using GPS (about a minute):
  wp load {wsl_path(mission_path)}
  mode auto
  arm throttle
"""


# --------------------------------------------------------------------------
# plan helpers
# --------------------------------------------------------------------------
def dense_plan(result, steps=20):
    """Positions, airspeed and course along the plan, integrated with the planner's model."""
    from solvers.scp_aircraft import AircraftSCP
    solver = AircraftSCP()
    X, U = np.array(result["x"]), np.array(result["u"])
    d = solver.propagate(X, U, result["flight_time"], steps, sens=False)
    xs = np.vstack([d["x_sub"].reshape(-1, 6), X[-1]])
    t = np.linspace(0.0, result["flight_time"], len(xs))
    return t, xs


def turn_radius():
    p = load_airframe()
    return p["V_CRUISE"] ** 2 / (GRAVITY * np.tan(np.radians(p["BANK_MAX_DEG"]))), p


def chord_deviation(P, i, k):
    """Largest distance of the plan points P[i..k] from the straight leg P[i] -> P[k]."""
    a, b = P[i], P[k]
    ab = b - a
    L2 = float(ab @ ab)
    if L2 < 1e-12:
        return 0.0
    q = P[i:k + 1] - a
    t = np.clip(q @ ab / L2, 0.0, 1.0)
    return float(np.linalg.norm(q - t[:, None] * ab, axis=1).max())


def resample(P, chi, chord_tol=CHORD_TOL_M):
    """Indices of dense plan points to use as waypoints. A new waypoint is placed
    when the leg from the last one reaches MAX_WP_SPACING_M, when the heading has
    turned MAX_WP_TURN_DEG, or when the straight leg would stray more than
    chord_tol (default CHORD_TOL_M) from the plan; never closer than MIN_WP_SPACING_M (so the
    autopilot cannot skip one)."""
    keep, last, s = [0], 0, 0.0
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    cum = np.r_[0.0, np.cumsum(seg)]
    for k in range(1, len(P) - 1):
        s += seg[k - 1]
        if s < MIN_WP_SPACING_M:
            continue
        turned = abs(np.degrees(chi[k] - chi[last]))
        if s >= MAX_WP_SPACING_M or turned >= MAX_WP_TURN_DEG:
            keep.append(k); last, s = k, 0.0
        elif chord_deviation(P, last, k) > chord_tol:
            j = k - 1 if cum[k - 1] - cum[last] >= MIN_WP_SPACING_M else k   # the last point still within tolerance
            keep.append(j); last, s = j, cum[k] - cum[j]
    end = len(P) - 1
    if cum[end] - cum[keep[-1]] < MIN_WP_SPACING_M and len(keep) > 1 \
            and chord_deviation(P, keep[-2], end) <= chord_tol:
        keep.pop()                                   # do not crowd the goal, if the merged leg still fits
    keep.append(end)
    return keep


def point_to_polyline(P, Q):
    """Distance from each point of P (n, 3) to the polyline Q (m, 3), and the
    nearest point on the polyline."""
    A, B = Q[:-1], Q[1:]
    AB = B - A
    L2 = np.maximum((AB**2).sum(1), 1e-12)
    out = np.empty(len(P)); near = np.empty_like(P, dtype=float)
    for i0 in range(0, len(P), 500):
        p = P[i0:i0 + 500, None, :]
        t = np.clip(((p - A) * AB).sum(-1) / L2, 0, 1)
        C = A + t[..., None] * AB
        d = np.linalg.norm(p - C, axis=-1)
        k = d.argmin(1)
        out[i0:i0 + 500] = d[np.arange(len(k)), k]
        near[i0:i0 + 500] = C[np.arange(len(k)), k]
    return out, near


def body_clearance(env, P):
    """Distance from each point to the nearest obstacle surface, minus the vehicle
    radius (negative = the aircraft would touch it), as in jsbsim_c172.evaluate."""
    h = env.constraint_values(P, d=env.vehicle_radius)
    A, _ = env.linearized_constraints(P)
    gn = np.maximum(np.linalg.norm(A, axis=-1), 1e-9).T
    return (h / gn).min(axis=0)


def ray_box_exit(p, u, lo, hi):
    """Distance along unit direction u (2D) from p until leaving the box [lo, hi] (2D)."""
    t = np.inf
    for k in range(2):
        if u[k] > 1e-9:
            t = min(t, (hi[k] - p[k]) / u[k])
        elif u[k] < -1e-9:
            t = min(t, (lo[k] - p[k]) / u[k])
    return max(t, 0.0)


def exit_route(env, g, chi_end, radius):
    """Way out of the workspace after the goal: turn by delta (smallest |delta| first)
    at the given radius, then fly straight until LOITER_GAP_M past the boundary.
    Returns (exit waypoint, final loiter centre, delta in deg, clearance in m)."""
    lo, hi = env.bounds_lo, env.bounds_hi
    best = None
    for d in sorted(np.arange(-EXIT_MAX_TURN_DEG, EXIT_MAX_TURN_DEG + 1e-9, 5.0), key=abs):
        dr = np.radians(d)
        side = np.sign(dr) if dr else 1.0
        n = np.array([-np.sin(chi_end), np.cos(chi_end)]) * side              # toward the turn centre
        c = g[:2] + radius * n
        a = np.linspace(0, abs(dr), max(2, int(abs(d)) + 2))
        start_ang = np.arctan2(*(g[:2] - c)[::-1])
        arc = c + radius * np.c_[np.cos(start_ang + side * a), np.sin(start_ang + side * a)]
        h = chi_end + dr
        u = np.array([np.cos(h), np.sin(h)])
        p1 = arc[-1]
        t_out = ray_box_exit(p1, u, lo[:2], hi[:2]) + LOITER_GAP_M
        line = p1 + np.linspace(0, t_out, 200)[:, None] * u
        pts2 = np.vstack([arc, line])
        pts = np.c_[pts2, np.full(len(pts2), g[2])]
        inside = np.all((pts >= lo) & (pts <= hi), axis=1)
        clear = body_clearance(env, pts[inside]).min() if inside.any() else np.inf
        exit_wp = np.r_[line[-1], g[2]]
        # final loiter centre: further out along u until the whole circle is outside the workspace
        ang = np.linspace(0, 2 * np.pi, 180)
        for k in range(1, 40):
            centre = line[-1] + (LOITER_RADIUS_M + (k - 1) * 200.0) * u
            ring = centre + LOITER_RADIUS_M * np.c_[np.cos(ang), np.sin(ang)]
            if not np.any(np.all((ring >= lo[:2]) & (ring <= hi[:2]), axis=1)):
                break
        cand = (exit_wp, np.r_[centre, g[2]], float(d), float(clear))
        if clear >= env.margin:
            return cand
        if best is None or clear > best[3]:
            best = cand
    return best


# --------------------------------------------------------------------------
# make
# --------------------------------------------------------------------------
CMD_WAYPOINT, CMD_LOITER_UNLIM, CMD_TAKEOFF, CMD_LOITER_TO_ALT, CMD_CHANGE_SPEED = 16, 17, 22, 31, 178
CMD_RTL, CMD_DO_JUMP = 20, 177
FRAME_GLOBAL, FRAME_REL_ALT = 0, 3


def write_waypoints(path, home_lat, home_lon, items):
    """QGC WPL 110 mission file. items: (frame, command, p1, p2, p3, p4, x, y, z_rel),
    x, y in the local frame (None = no position)."""
    lines = ["QGC WPL 110",
             f"0\t1\t{FRAME_GLOBAL}\t{CMD_WAYPOINT}\t0\t0\t0\t0\t{home_lat:.8f}\t{home_lon:.8f}\t0.000\t1"]
    for k, (fr, cmd, p1, p2, p3, p4, x, y, z) in enumerate(items, start=1):
        la, lo_ = (0.0, 0.0) if x is None else enu_to_geo(x, y)
        lines.append(f"{k}\t0\t{fr}\t{cmd}\t{p1:g}\t{p2:g}\t{p3:g}\t{p4:g}\t{la:.8f}\t{lo_:.8f}\t{z:.3f}\t1")
    with open(path, "w", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


def autopilot_path(W, radius, max_bank_deg, speed):
    """Estimate of the path ArduPilot flies through the waypoints W: straight legs
    joined at each waypoint by a turn of `radius` (the autopilot starts turning
    before the waypoint, so it cuts inside each corner). Where a leg is too short
    for that turn, the turn is tightened to fit. Returns (points, steepest bank the
    tightened turns need in deg)."""
    out, worst_bank = [W[0]], 0.0
    for i in range(1, len(W) - 1):
        a, b, c = W[i - 1], W[i], W[i + 1]
        u1, u2 = (b - a)[:2], (c - b)[:2]
        l1, l2 = np.linalg.norm(u1), np.linalg.norm(u2)
        if l1 < 1e-6 or l2 < 1e-6:
            out.append(b); continue
        u1, u2 = u1 / l1, u2 / l2
        theta = float(np.arccos(np.clip(u1 @ u2, -1.0, 1.0)))       # heading change at this waypoint
        if theta < 1e-4:
            out.append(b); continue
        tangent = radius * np.tan(theta / 2)
        tangent = min(tangent, 0.5 * l1, 0.5 * l2)                    # both turns must fit on a leg
        r_eff = tangent / np.tan(theta / 2)
        worst_bank = max(worst_bank, float(np.degrees(np.arctan(speed**2 / (GRAVITY * max(r_eff, 1e-6))))))
        p1, p2 = b[:2] - tangent * u1, b[:2] + tangent * u2
        side = np.sign(u1[0] * u2[1] - u1[1] * u2[0])                 # +1 left turn, -1 right turn
        centre = p1 + side * r_eff * np.array([-u1[1], u1[0]])
        a0 = np.arctan2(*(p1 - centre)[::-1])
        ang = a0 + side * np.linspace(0.0, theta, max(3, int(np.degrees(theta)) + 2))
        arc = centre + r_eff * np.c_[np.cos(ang), np.sin(ang)]
        out.extend(np.c_[arc, np.full(len(arc), b[2])])
    out.append(W[-1])
    return np.array(out), worst_bank


def densify(Q, spacing=10.0):
    """Points along the polyline Q at most `spacing` apart."""
    pts = [Q[:1]]
    for a, b in zip(Q[:-1], Q[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / spacing)))
        pts.append(a + np.linspace(0, 1, n + 1)[1:, None] * (b - a))
    return np.vstack(pts)


def leg_speeds(V, idx, v_lo, v_hi):
    """The plan's mean airspeed over each waypoint leg idx[k] -> idx[k+1], clipped to [v_lo, v_hi]."""
    return np.array([np.clip(np.mean(V[a:b + 1]), v_lo, v_hi) for a, b in zip(idx[:-1], idx[1:])])


def build_mission(env, r, chord_tol=None, speed_profile=None):
    """Everything `make` writes for one SCP plan: the mission items, where SITL's home
    goes, and the checks that say whether the plan can be flown as waypoints.

    Rejections (the mission is still built, but the instance should not be flown):
      * the waypoint path, with the autopilot's turns, passes closer than
        FLY_MIN_CLEARANCE_FRAC x margin to an obstacle (inside the planner's margin by
        more than the autopilot can be trusted to hold);
      * a corner needs more bank than the autopilot may use (ROLL_LIMIT_DEG in
        c172_sitl.parm, AP_ROLL_LIMIT_DEG here) because the legs around it are short.
    Warnings (flown anyway): lead-in / exit near an obstacle (outside the scored
    segment; obstacles are not physical in SITL), climb or sink beyond TECS limits,
    and a wide speed range (ArduPilot flies one airspeed, so timing drifts)."""
    chord_tol = CHORD_TOL_M if chord_tol is None else chord_tol
    speed_profile = SPEED_PROFILE if speed_profile is None else speed_profile
    if speed_profile not in ("plan", "constant"):
        raise ValueError(f"speed_profile must be 'plan' or 'constant', not {speed_profile!r}")
    R, params = turn_radius()
    t, xs = dense_plan(r)
    P, V, gam, chi = xs[:, :3], xs[:, 3], xs[:, 4], xs[:, 5]
    lo, hi = env.bounds_lo, env.bounds_hi
    warnings_, rejects = [], []

    # ---- plan waypoints
    idx = resample(P, chi, chord_tol)
    W = P[idx]
    chord, _ = point_to_polyline(P, W)

    # ---- lead-in, climb loiter and home, upstream of the start
    s0, u0 = P[0], np.array([np.cos(chi[0]), np.sin(chi[0]), 0.0])
    lead = s0 - LEAD_IN_TURN_RADII * R * u0
    t_out = ray_box_exit(lead[:2], -u0[:2], lo[:2], hi[:2])          # back out of the workspace
    loiter_c = lead - (t_out + LOITER_GAP_M + LOITER_RADIUS_M) * u0
    loiter_c[2] = s0[2]
    home_xy = loiter_c[:2]
    home_heading = compass_deg(chi[0])

    # ---- exit: leave the workspace after the goal by the smallest turn that clears every obstacle
    speed = float(np.clip(np.mean(V), params["V_MIN"] + 2, params["V_MAX"] - 4))
    ap_radius = speed**2 / (GRAVITY * np.tan(np.radians(AP_BANK_DEG)))
    exit_wp, end_c, exit_turn, exit_clear = exit_route(env, P[-1], chi[-1], ap_radius)
    if exit_clear < env.margin:
        warnings_.append(f"no fully clear way out after the goal; best exit passes {exit_clear:.0f} m from an "
                         f"obstacle (outside the plan; in SITL the obstacles are not physical)")

    # ---- checks on the added legs (the planner never saw them)
    def leg_clear(a, b, name):
        pts = np.linspace(a, b, 200)
        inside = np.all((pts >= lo) & (pts <= hi), axis=1)
        if inside.any():
            c = body_clearance(env, pts[inside]).min()
            if c < env.margin:
                warnings_.append(f"{name} passes {c:.0f} m from an obstacle (plan keeps {env.margin:.0f} m)")
    leg_clear(lead, s0, "lead-in leg")
    ang = np.linspace(0, 2 * np.pi, 180)
    for c, name in ((loiter_c, "climb loiter"), (end_c, "final loiter")):
        ring = c[:2] + LOITER_RADIUS_M * np.c_[np.cos(ang), np.sin(ang)]
        if np.any(np.all((ring >= lo[:2]) & (ring <= hi[:2]), axis=1)):
            warnings_.append(f"{name} circle overlaps the workspace")
    climb = (V * np.sin(gam)).max(); sink = -(V * np.sin(gam)).min()
    if climb > TECS_CLMB_MAX or sink > TECS_SINK_MAX:
        warnings_.append(f"plan climbs {climb:.1f} m/s / sinks {sink:.1f} m/s, beyond TECS_CLMB_MAX "
                         f"{TECS_CLMB_MAX} / TECS_SINK_MAX {TECS_SINK_MAX}: expect altitude lag")
    if speed_profile == "constant" and V.max() - V.min() > SPEED_SPREAD_WARN_MS:
        warnings_.append(f"plan speed ranges {V.min():.1f}-{V.max():.1f} m/s but ArduPilot flies one airspeed "
                         f"({speed:.1f} m/s): expect the flight time to differ from the plan's")

    # ---- can the plan be flown as waypoints? (the scored part: start to goal)
    flown_est, worst_bank = autopilot_path(W, ap_radius, AP_ROLL_LIMIT_DEG, speed)
    est = densify(flown_est)
    inside = np.all((est >= lo) & (est <= hi), axis=1)
    est_clear = float(body_clearance(env, est[inside]).min()) if inside.any() else float("inf")
    est_offset = float(point_to_polyline(est, P)[0].max())
    plan_clear = float(body_clearance(env, densify(P, 20.0)).min())
    min_clear = FLY_MIN_CLEARANCE_FRAC * env.margin
    if est_clear < min_clear:
        rejects.append(f"waypoint path with the autopilot's turns passes {est_clear:.1f} m from an obstacle "
                       f"(needs >= {min_clear:.0f} m = {FLY_MIN_CLEARANCE_FRAC:g} x margin; the plan itself "
                       f"keeps {plan_clear:.1f} m)")
    if worst_bank > AP_ROLL_LIMIT_DEG:
        rejects.append(f"a waypoint corner needs {worst_bank:.0f} deg of bank (legs too short for the turn); "
                       f"the autopilot may use {AP_ROLL_LIMIT_DEG:.0f}")
    checks = {"plan_min_clearance_m": plan_clear, "est_min_clearance_m": est_clear,
              "est_max_offset_from_plan_m": est_offset, "chord_max_m": float(chord.max()),
              "worst_corner_bank_deg": worst_bank, "min_clearance_required_m": min_clear}

    # ---- mission items: (frame, command, p1, p2, p3, p4, x, y, z_rel)
    #      Speeds ("plan" profile): ArduPilot runs the DO commands that follow a NAV command while that
    #      NAV command is active, i.e. while flying TO that waypoint. So the speed for the leg
    #      W[k-1] -> W[k] goes right after waypoint W[k]; the lead-in uses the plan's start speed.
    alt = lambda z: float(z + FLOOR_AGL_M)
    v_lo, v_hi = params["V_MIN"] + 2, params["V_MAX"] - SPEED_HEADROOM_MS
    if speed_profile == "plan":
        v_leg = leg_speeds(V, idx, v_lo, v_hi)
        v_start = float(np.clip(V[0], v_lo, v_hi))
    else:
        v_leg, v_start = np.full(len(W) - 1, speed), speed
    items = [(FRAME_REL_ALT, CMD_TAKEOFF, TAKEOFF_PITCH_DEG, 0, 0, 0, home_xy[0], home_xy[1], TAKEOFF_ALT_M),
             (FRAME_REL_ALT, CMD_CHANGE_SPEED, 0, v_start, -1, 0, None, None, 0),
             (FRAME_REL_ALT, CMD_LOITER_TO_ALT, 1, LOITER_RADIUS_M, 0, 0, loiter_c[0], loiter_c[1], alt(s0[2])),
             (FRAME_REL_ALT, CMD_WAYPOINT, 0, 0, 0, 0, lead[0], lead[1], alt(lead[2]))]
    first_plan_item = len(items) + 1                         # +1: item 0 is home
    plan_items, last_v, n_speed_cmds = [], v_start, 0
    for k, w in enumerate(W):
        plan_items.append(len(items) + 1)
        items.append((FRAME_REL_ALT, CMD_WAYPOINT, 0, 0, 0, 0, w[0], w[1], alt(w[2])))
        if k >= 1 and abs(v_leg[k - 1] - last_v) >= SPEED_CMD_STEP_MS:
            items.append((FRAME_REL_ALT, CMD_CHANGE_SPEED, 0, float(v_leg[k - 1]), -1, 0, None, None, 0))
            last_v, n_speed_cmds = float(v_leg[k - 1]), n_speed_cmds + 1
    items += [(FRAME_REL_ALT, CMD_WAYPOINT, 0, 0, 0, 0, exit_wp[0], exit_wp[1], alt(exit_wp[2])),
              (FRAME_REL_ALT, CMD_LOITER_UNLIM, 0, 0, LOITER_RADIUS_M, 0, end_c[0], end_c[1], alt(end_c[2]))]
    hlat, hlon = enu_to_geo(*home_xy)
    # rough simulated duration: takeoff and climb, lead-in, the plan, the exit (for the batch runner's timeout)
    expected_s = (alt(s0[2]) / (0.6 * TECS_CLMB_MAX) + 120.0 + LEAD_IN_TURN_RADII * R / speed
                  + float(r["flight_time"]) + 2 * LOITER_GAP_M / speed + 120.0)
    return {"items": items, "W": W, "P": P, "chord": chord, "speed": speed, "R": R,
            "speed_profile": speed_profile, "leg_speeds": [float(v) for v in v_leg], "plan_items": plan_items,
            "n_speed_commands": n_speed_cmds + 1,
            "commanded_mean_ms": float(np.mean(v_leg)) if len(v_leg) else speed,
            "home_lat_lon": (float(hlat), float(hlon)), "home_heading_deg": home_heading,
            "first_plan_item": first_plan_item, "start_alt_rel_m": alt(s0[2]),
            "exit_turn_deg": exit_turn, "exit_clearance_m": exit_clear,
            "warnings": warnings_, "rejects": rejects, "checks": checks, "expected_sim_s": expected_s}


def write_mission(mission, out, env_file, results_file, index, chord_tol, flight_time):
    """Write the .waypoints file and its .json companion (read by analyze)."""
    hlat, hlon = mission["home_lat_lon"]
    write_waypoints(out, hlat, hlon, mission["items"])
    meta = {"envs": env_file, "results": results_file, "index": index,
            "floor_agl_m": FLOOR_AGL_M, "origin": [fly.ORIGIN_LAT_DEG, fly.ORIGIN_LON_DEG],
            "home_lat_lon": [hlat, hlon], "home_heading_deg": mission["home_heading_deg"],
            "first_plan_item": mission["first_plan_item"], "n_plan_waypoints": len(mission["W"]),
            "chord_tol_m": chord_tol, "commanded_airspeed_ms": mission["commanded_mean_ms"],
            "speed_profile": mission["speed_profile"], "leg_speeds_ms": mission["leg_speeds"],
            "plan_items": mission["plan_items"],
            "plan_flight_time_s": flight_time, "flyable": not mission["rejects"],
            "rejects": mission["rejects"], "warnings": mission["warnings"], "checks": mission["checks"],
            "kind": "scp_plan"}
    with open(f"{out}.json", "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def sitl_home(mission):
    hlat, hlon = mission["home_lat_lon"]
    return f"{hlat:.7f},{hlon:.7f},0,{mission['home_heading_deg']:.0f}"


def make(args):
    from solvers.scp_aircraft import load_plan
    env, r, res = load_plan(args.envs, args.results, args.index)
    if not r["success"]:
        raise SystemExit(f"Environment {args.index} has no feasible plan (status {r['status']}).")
    chord_tol = CHORD_TOL_M if getattr(args, "chord_tol", None) is None else args.chord_tol
    m = build_mission(env, r, chord_tol, getattr(args, "speed_profile", None))
    name = args.out or f"{os.path.splitext(os.path.basename(args.envs))[0]}{args.index}.waypoints"
    out = localstore_path(name)
    write_mission(m, out, args.envs, args.results, args.index, chord_tol, r["flight_time"])

    # ---- report
    P, W, R, c = m["P"], m["W"], m["R"], m["checks"]
    print(f"Wrote {out} ({len(m['items']) + 1} items) and {out}.json")
    print(f"  plan: {len(W)} waypoints over {np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1)) / 1000:.2f} km, "
          f"items {m['first_plan_item']}-{m['first_plan_item'] + len(W) - 1}; "
          f"straight-line legs stay within {m['chord'].max():.1f} m of the plan")
    if m["speed_profile"] == "plan":
        print(f"  airspeed follows the plan: {m['n_speed_commands']} speed commands, "
              f"{min(m['leg_speeds']):.1f}-{max(m['leg_speeds']):.1f} m/s")
    print(f"  airspeed {m['commanded_mean_ms']:.1f} m/s (mean), lead-in {LEAD_IN_TURN_RADII * R:.0f} m, "
          f"climb loiter to {m['start_alt_rel_m']:.0f} m above home")
    print(f"  exit after the goal: turn {m['exit_turn_deg']:+.0f} deg, then straight out "
          f"(closest obstacle {m['exit_clearance_m']:.0f} m)")
    print(f"  expected flown path: closest obstacle {c['est_min_clearance_m']:.1f} m (plan "
          f"{c['plan_min_clearance_m']:.1f} m), at most {c['est_max_offset_from_plan_m']:.1f} m from the plan")
    for w in m["warnings"]:
        print("  WARNING:", w)
    for w in m["rejects"]:
        print("  NOT FLYABLE AS WAYPOINTS:", w)
    log_name = os.path.splitext(name)[0] + ".tlog"
    print(f"""
------------------------------------------------------------------------------
To fly it
------------------------------------------------------------------------------
{sitl_commands(sitl_home(m), out)}
When it is loitering after the last waypoint, copy the telemetry log (in WSL):
  cp ~/ardupilot/ArduPlane/mav.tlog {wsl_path(localstore_path(log_name))}
  (if it is not there:  find ~/ardupilot -name "*.tlog" -mmin -60)
then in PyCharm's terminal:
  python -m ardupilot.ardupilot_mission analyze --mission {name} --log {log_name} --plot
""")


# --------------------------------------------------------------------------
# box (Tier 1 test mission)
# --------------------------------------------------------------------------
def box_corners(heading_deg, start, length, width):
    """Corners A, B, C, D of the rectangle in the local frame (home = origin):
    A is `start` ahead of home along the runway heading, A -> B along it, then a
    right turn to C and back down to D."""
    h = np.radians(heading_deg)
    fwd, right = np.array([np.sin(h), np.cos(h)]), np.array([np.cos(h), -np.sin(h)])
    A = start * fwd
    return np.array([A, A + length * fwd, A + length * fwd + width * right, A + width * right])


def box(args):
    params = load_airframe()
    speed = round(float(args.speed or params["V_CRUISE"]), 1)
    if not params["V_MIN"] <= speed <= params["V_MAX"]:
        raise SystemExit(f"--speed {speed} is outside the C172's {params['V_MIN']:.0f}-{params['V_MAX']:.0f} m/s.")
    if args.laps < 1:
        raise SystemExit("--laps must be at least 1.")
    C = box_corners(args.heading, BOX_START_M, args.length, args.width)
    R45 = speed**2 / (GRAVITY * np.tan(np.radians(45)))
    if min(args.length, args.width) < SETTLE_M + LEG_END_M + 2 * R45:
        raise SystemExit("The box is too small to settle on each leg; use --length/--width of at least "
                         f"{SETTLE_M + LEG_END_M + 2 * R45:.0f} m.")
    wp = lambda p: (FRAME_REL_ALT, CMD_WAYPOINT, 0, 0, 0, 0, p[0], p[1], args.alt)
    items = [(FRAME_REL_ALT, CMD_TAKEOFF, TAKEOFF_PITCH_DEG, 0, 0, 0, 0.0, 0.0, TAKEOFF_ALT_M),
             (FRAME_REL_ALT, CMD_CHANGE_SPEED, 0, speed, -1, 0, None, None, 0),
             wp(C[0]), wp(C[1]), wp(C[2]), wp(C[3]), wp(C[0])]
    seq_B = 4                                          # item numbers: 0 home, 1 takeoff, 2 speed, 3 A, 4 B ...
    if args.laps > 1:
        items.append((FRAME_REL_ALT, CMD_DO_JUMP, seq_B, args.laps - 1, 0, 0, None, None, 0))
    items.append((FRAME_REL_ALT, CMD_RTL, 0, 0, 0, 0, None, None, 0))

    out = localstore_path(args.out)
    write_waypoints(out, fly.ORIGIN_LAT_DEG, fly.ORIGIN_LON_DEG, items)
    meta = {"kind": "box", "origin": [fly.ORIGIN_LAT_DEG, fly.ORIGIN_LON_DEG], "heading_deg": args.heading,
            "corners_enu": C.tolist(), "alt_rel_m": args.alt, "commanded_airspeed_ms": speed,
            "laps": args.laps, "takeoff_alt_m": TAKEOFF_ALT_M}
    with open(f"{out}.json", "w") as f:
        json.dump(meta, f, indent=2)
    lap_km = 2 * (args.length + args.width) / 1000
    print(f"Wrote {out} ({len(items) + 1} items) and {out}.json")
    print(f"  take off heading {args.heading:.0f} deg, climb to {args.alt:.0f} m above home, "
          f"{args.laps} lap(s) of a {args.length / 1000:.1f} x {args.width / 1000:.1f} km rectangle at "
          f"{speed:.1f} m/s (about {args.laps * lap_km * 1000 / speed / 60:.0f} min), then return home")
    home = f"{fly.ORIGIN_LAT_DEG:.7f},{fly.ORIGIN_LON_DEG:.7f},0,{args.heading:.0f}"
    log_name = os.path.splitext(args.out)[0] + ".tlog"
    print(f"""
------------------------------------------------------------------------------
To fly it
------------------------------------------------------------------------------
{sitl_commands(home, out)}
When it is circling home after the last lap, copy the telemetry log (in WSL):
  cp ~/ardupilot/ArduPlane/mav.tlog {wsl_path(localstore_path(log_name))}
  (if it is not there:  find ~/ardupilot -name "*.tlog" -mmin -60)
then in PyCharm's terminal:
  python -m ardupilot.ardupilot_mission summary --mission {args.out} --log {log_name} --plot
""")


# --------------------------------------------------------------------------
# telemetry log reading (no extra packages needed)
# --------------------------------------------------------------------------
# GLOBAL_POSITION_INT, VFR_HUD, ATTITUDE, NAV_CONTROLLER_OUTPUT, PARAM_VALUE, HEARTBEAT
_CRC_EXTRA = {33: 104, 74: 20, 30: 39, 62: 183, 22: 220, 0: 50}
_LEN = {33: 28, 74: 20, 30: 28, 62: 26, 22: 25, 0: 9}
# ArduPlane flight modes (HEARTBEAT custom_mode)
PLANE_MODES = {0: "MANUAL", 1: "CIRCLE", 2: "STABILIZE", 3: "TRAINING", 4: "ACRO", 5: "FBWA", 6: "FBWB",
               7: "CRUISE", 8: "AUTOTUNE", 10: "AUTO", 11: "RTL", 12: "LOITER", 13: "TAKEOFF", 15: "GUIDED"}


def _crc(data, extra):
    crc = 0xFFFF
    for b in bytes(data) + bytes([extra]):
        tmp = b ^ (crc & 0xFF)
        tmp = (tmp ^ (tmp << 4)) & 0xFF
        crc = ((crc >> 8) ^ (tmp << 8) ^ (tmp << 3) ^ (tmp >> 4)) & 0xFFFF
    return crc


def read_tlog(path, sysid=1):
    """GLOBAL_POSITION_INT, VFR_HUD, ATTITUDE, NAV_CONTROLLER_OUTPUT and
    PARAM_VALUE from a MAVProxy .tlog (8-byte timestamp + MAVLink 1/2 packet per
    record), CRC-checked. Returns a dict of arrays; "attitude" (t, roll, pitch in deg)
    and "nav" (t, commanded roll, commanded pitch in deg, altitude error in m) are on
    the same clock as "t" when the log has them. "params" holds the autopilot's
    parameters (the last value of each in the log: MAVProxy fetches them all when it
    connects, so with several sessions the latest one's values win), "log_start" the
    log (wall-clock) time of the first position, in seconds since 1970."""
    data = open(path, "rb").read()
    pos, hud, att, nav, params, modes, pchanges = [], [], [], [], {}, [], []
    i, n = 0, len(data)
    while i + 8 < n:
        t_us = int.from_bytes(data[i:i + 8], "big")
        j = i + 8
        stx = data[j]
        if stx == 0xFD and j + 10 <= n:                       # MAVLink 2
            ln, incompat = data[j + 1], data[j + 2]
            sid, mid = data[j + 5], int.from_bytes(data[j + 7:j + 10], "little")
            end = j + 10 + ln + 2 + (13 if incompat & 1 else 0)
            body, pl = data[j + 1:j + 10 + ln], data[j + 10:j + 10 + ln]
            crc_at = j + 10 + ln
        elif stx == 0xFE and j + 6 <= n:                      # MAVLink 1
            ln = data[j + 1]
            sid, mid = data[j + 3], data[j + 5]
            end = j + 6 + ln + 2
            body, pl = data[j + 1:j + 6 + ln], data[j + 6:j + 6 + ln]
            crc_at = j + 6 + ln
        else:
            i += 1
            continue
        if end > n:
            break
        if mid in _CRC_EXTRA and sid == sysid:
            if int.from_bytes(data[crc_at:crc_at + 2], "little") == _crc(body, _CRC_EXTRA[mid]):
                pl = pl + bytes(_LEN[mid] - len(pl))           # MAVLink 2 trims trailing zeros
                if mid == 33:
                    tb, lat, lon, a_msl, a_rel, vx, vy, vz, hdg = struct.unpack("<IiiiihhhH", pl[:28])
                    pos.append((t_us * 1e-6, tb * 1e-3, lat * 1e-7, lon * 1e-7, a_rel * 1e-3))
                elif mid == 74:
                    aspd, gspd, a, climb, hdg, thr = struct.unpack("<ffffhH", pl[:20])
                    hud.append((t_us * 1e-6, aspd, thr, climb))
                elif mid == 30:
                    roll, pitch = struct.unpack("<ff", pl[4:12])     # rad
                    att.append((t_us * 1e-6, np.degrees(roll), np.degrees(pitch)))
                elif mid == 0:
                    if pl[4] == 1:                                   # MAV_TYPE_FIXED_WING (not a GCS or camera)
                        modes.append((t_us * 1e-6, struct.unpack("<I", pl[:4])[0]))
                elif mid == 22:
                    value = struct.unpack("<f", pl[:4])[0]
                    pname = pl[8:24].split(b"\0")[0].decode("ascii", "replace")
                    if pname in params and params[pname] != value and not pname.startswith(PARAM_IGNORE):
                        pchanges.append((t_us * 1e-6, pname, params[pname], value))   # e.g. a param set in flight
                    params[pname] = value
                else:
                    nav_roll, nav_pitch, alt_err = struct.unpack("<fff", pl[:12])  # deg, deg, m
                    nav.append((t_us * 1e-6, nav_roll, nav_pitch, alt_err))
        i = end
    if not pos:
        raise SystemExit(f"No GLOBAL_POSITION_INT messages from system {sysid} in {path}.")
    pos = np.array(pos)
    pos = pos[(pos[:, 2] != 0) | (pos[:, 3] != 0)]              # no position yet (before the GPS fix)
    if not len(pos):
        raise SystemExit(f"{path} has no valid positions (the GPS never got a fix).")
    pos = pos[np.r_[True, np.diff(pos[:, 1]) != 0]]            # drop repeated samples
    # a log can hold several SITL sessions; the autopilot clock restarts at each one
    starts = np.nonzero(np.diff(pos[:, 1]) < -1.0)[0] + 1
    if len(starts):
        print(f"  {path}: {len(starts) + 1} sessions in this log; using the most recent one")
        keep = slice(starts[-1], None)
        pos = pos[keep]
        if hud:
            hud = [h for h in hud if h[0] >= pos[0, 0]]
    out = {"t": pos[:, 1] - pos[0, 1], "lat": pos[:, 2], "lon": pos[:, 3], "alt_rel": pos[:, 4],
           "params": params, "log_start": float(pos[0, 0]),
           # parameters changed during this session: (time on the clock of "t", name, old, new)
           "param_changes": [(float(np.interp(c[0], pos[:, 0], pos[:, 1]) - pos[0, 1]),) + c[1:]
                             for c in pchanges if pos[0, 0] <= c[0] <= pos[-1, 0]]}
    if hud:
        hud = np.array(hud)
        out["airspeed"] = np.interp(pos[:, 0], hud[:, 0], hud[:, 1])
        out["throttle"] = np.interp(pos[:, 0], hud[:, 0], hud[:, 2])
        out["climb"] = np.interp(pos[:, 0], hud[:, 0], hud[:, 3])
    for key, rows in (("attitude", att), ("nav", nav), ("mode", modes)):
        if not rows:
            continue
        a = np.array(rows)
        a = a[(a[:, 0] >= pos[0, 0]) & (a[:, 0] <= pos[-1, 0])]   # this session only
        if len(a):                                  # log time -> the autopilot clock used for "t"
            a[:, 0] = np.interp(a[:, 0], pos[:, 0], pos[:, 1]) - pos[0, 1]
            out[key] = a
    return out


def oscillation(t, y, window=OSC_WINDOW_S):
    """RMS and dominant period of the fast part of y(t): y minus its moving average
    over `window` seconds. Returns (None, None) if the record is too short."""
    if len(t) < 4:
        return None, None
    dt = float(np.median(np.diff(t)))
    tu = np.arange(t[0], t[-1], dt)
    w = int(round(window / dt))
    if w < 2 or len(tu) < 4 * w:
        return None, None
    yu = np.interp(tu, t, y)
    fast = (yu - np.convolve(yu, np.ones(w) / w, "same"))[w:-w]   # drop the edges of the average
    freq = np.fft.rfftfreq(len(fast), dt)
    power = np.abs(np.fft.rfft(fast))
    power[freq < 1.0 / window] = 0.0
    rms = float(np.sqrt(np.mean(fast ** 2)))
    if not power.any():
        return rms, None                                 # a constant signal has no period
    return rms, float(1.0 / freq[np.argmax(power)])


def follow_lag(t_cmd, cmd, t_out, out, window=OSC_WINDOW_S, max_lag=2.0):
    """How far `out` trails `cmd` (s) and how big its fast part is relative to the
    command's: the delay that maximizes the correlation of the two fast parts."""
    t0, t1 = max(t_cmd[0], t_out[0]), min(t_cmd[-1], t_out[-1])
    dt = 0.05
    tu = np.arange(t0, t1, dt)
    w = int(round(window / dt))
    if len(tu) < 4 * w:
        return None, None
    hp = lambda t, y: (np.interp(tu, t, y) - np.convolve(np.interp(tu, t, y), np.ones(w) / w, "same"))[w:-w]
    c, o = hp(t_cmd, cmd), hp(t_out, out)
    if c.std() == 0 or o.std() == 0:
        return None, None
    lags = np.arange(0, int(max_lag / dt) + 1)
    corr = [np.corrcoef(c[:len(c) - L], o[L:])[0, 1] for L in lags]
    return float(lags[int(np.argmax(corr))] * dt), float(o.std() / c.std())


def read_bin(path):
    """DataFlash .BIN log via pymavlink (pip install pymavlink), using POS messages."""
    try:
        from pymavlink import DFReader
    except ImportError:
        raise SystemExit("Reading .BIN logs needs pymavlink (pip install pymavlink), or use the .tlog.")
    log = DFReader.DFReader_binary(path)
    rows = []
    while True:
        m = log.recv_match(type=["POS"])
        if m is None:
            break
        rows.append((m.TimeUS * 1e-6, m.Lat, m.Lng, m.RelHomeAlt))
    if not rows:
        raise SystemExit(f"No POS messages in {path}.")
    a = np.array(rows)
    return {"t": a[:, 0] - a[0, 0], "lat": a[:, 1], "lon": a[:, 2], "alt_rel": a[:, 3]}


def read_flight(path):
    path = str(find_input(path))
    ext = os.path.splitext(path)[1].lower()
    if ext == ".bin":
        return read_bin(path)
    if ext == ".csv":
        import csv
        rows = list(csv.DictReader(open(path)))
        return {k: np.array([float(r[k]) for r in rows]) for k in ("t", "lat", "lon", "alt_rel")}
    return read_tlog(path)


# --------------------------------------------------------------------------
# analyze
# --------------------------------------------------------------------------
def analyze(args):
    from solvers.scp_aircraft import load_plan
    meta = json.load(open(find_input(args.mission + ".json")))
    env, r, _ = load_plan(meta["envs"], meta["results"], meta["index"])
    t_plan, xs = dense_plan(r)
    Pp = xs[:, :3]

    f = read_flight(args.log)
    x, y = geo_to_enu(f["lat"], f["lon"])
    F = np.c_[x, y, f["alt_rel"] - meta["floor_agl_m"]]
    t = f["t"]

    # ---- the plan segment: from the aircraft's closest approach to the plan's start (not the first
    #      pass within PASS_TOL_M: the takeoff can pass under the start, hundreds of metres too low)
    #      to its first pass near the goal after that
    d0 = np.linalg.norm(F - Pp[0], axis=1)
    near0 = np.nonzero(d0 < PASS_TOL_M)[0]
    if not len(near0):
        raise SystemExit(f"The aircraft never came within {PASS_TOL_M:.0f} m of the plan's start "
                         f"(closest {d0.min():.0f} m). Did the mission run, and is SITL's home where "
                         f"'make' said?")
    i0 = int(np.argmin(d0))
    d1 = np.linalg.norm(F[i0:] - Pp[-1], axis=1)
    goal_miss = float(d1.min())                                  # closest the aircraft came to the goal
    near1 = np.nonzero(d1 < PASS_TOL_M)[0]
    reached = len(near1) > 0
    i1 = i0 + (near1[0] + int(np.argmin(d1[near1[0]:near1[0] + 200])) if reached else int(np.argmin(d1)))
    seg = slice(i0, i1 + 1)
    Fs, ts = F[seg], t[seg] - t[i0]

    # ---- metrics
    err, near = point_to_polyline(Fs, Pp)
    errh = np.linalg.norm((Fs - near)[:, :2], axis=1)          # split of the same error vector
    errv = np.abs((Fs - near)[:, 2])
    clear = body_clearance(env, Fs)
    lo, hi = env.bounds_lo, env.bounds_hi
    out_of_box = float(np.max(np.r_[lo - Fs.min(0), Fs.max(0) - hi, 0.0]))
    plan_clear = body_clearance(env, Pp)
    summary = {
        "log": args.log, "mission": args.mission, "reached_goal": bool(reached), "goal_miss_m": goal_miss,
        "flight_time_s": float(ts[-1]), "plan_flight_time_s": float(r["flight_time"]),
        "max_error_m": float(err.max()), "rms_error_m": float(np.sqrt(np.mean(err**2))),
        "max_horizontal_error_m": float(errh.max()),
        "max_vertical_error_m": float(errv.max()),
        "min_clearance_m": float(clear.min()), "plan_min_clearance_m": float(plan_clear.min()),
        "planned_margin_m": env.margin, "samples_in_collision": int((clear < 0).sum()),
        "left_workspace_m": out_of_box,
        "collision_free": bool(clear.min() >= 0 and out_of_box <= 0)}
    v_plan_near = None
    if "airspeed" in f:
        summary["mean_airspeed_ms"] = float(np.mean(f["airspeed"][seg]))
        # the plan's airspeed at the nearest plan point, to see whether ArduPilot follows the speed profile
        k_near = np.concatenate([np.argmin(((Fs[i:i + 400, None, :] - Pp[None]) ** 2).sum(-1), axis=1)
                                 for i in range(0, len(Fs), 400)])
        v_plan_near = xs[k_near, 3]
        summary["airspeed_vs_plan_rms_ms"] = float(np.sqrt(np.mean((f["airspeed"][seg] - v_plan_near) ** 2)))

    # ---- oscillation (weaving), after the first OSC_SKIP_S of the plan
    osc = {}
    k = ts >= OSC_SKIP_S
    osc["altitude"] = oscillation(ts[k], (Fs - near)[k, 2])          # height above/below the plan
    t_from, t_to = t[i0] + OSC_SKIP_S, t[i1]
    if "attitude" in f:
        A = f["attitude"][(f["attitude"][:, 0] >= t_from) & (f["attitude"][:, 0] <= t_to)]
        osc["roll"] = oscillation(A[:, 0], A[:, 1])
        osc["pitch"] = oscillation(A[:, 0], A[:, 2])
        if len(A):
            summary["max_abs_roll_deg"] = float(np.abs(A[:, 1]).max())
        if "nav" in f:
            Nv = f["nav"][(f["nav"][:, 0] >= t_from) & (f["nav"][:, 0] <= t_to)]
            osc["commanded_roll"] = oscillation(Nv[:, 0], Nv[:, 1])
            if len(A) and len(Nv) > 1:                             # how closely roll follows the autopilot's command
                cmd = np.interp(A[:, 0], Nv[:, 0], Nv[:, 1])
                summary["roll_minus_command_rms_deg"] = float(np.sqrt(np.mean((A[:, 1] - cmd) ** 2)))
    for name, (rms, period) in osc.items():
        unit = "m" if name == "altitude" else "deg"
        summary[f"{name}_oscillation_{unit}"] = rms
        summary[f"{name}_oscillation_period_s"] = period

    print(f"Flight log {args.log}: {len(t)} samples, plan segment {ts[-1]:.0f} s "
          f"(plan {r['flight_time']:.0f} s){'' if reached else '  -- goal NOT reached'}")
    print(f"  distance from the planned path: max {summary['max_error_m']:.1f} m, "
          f"rms {summary['rms_error_m']:.1f} m (horizontal max {summary['max_horizontal_error_m']:.1f} m, "
          f"vertical max {summary['max_vertical_error_m']:.1f} m)")
    print(f"  closest approach to an obstacle: flown {summary['min_clearance_m']:.1f} m, "
          f"planned {summary['plan_min_clearance_m']:.1f} m (plan keeps {env.margin:.0f} m + vehicle radius)")
    if "mean_airspeed_ms" in summary:
        print(f"  mean airspeed {summary['mean_airspeed_ms']:.1f} m/s (commanded {meta['commanded_airspeed_ms']:.1f}"
              f"{', following the plan' if meta.get('speed_profile') == 'plan' else ''}); "
              f"{summary['airspeed_vs_plan_rms_ms']:.1f} m/s rms from the plan's airspeed")
    def osc_text(name, label, unit):
        rms, period = osc.get(name, (None, None))
        if rms is None:
            return f"{label} n/a"
        return f"{label} {rms:.1f} {unit} rms" + (f", period {period:.1f} s" if period is not None else "")
    print(f"  oscillation after the first {OSC_SKIP_S:.0f} s (part faster than {OSC_WINDOW_S:.0f} s):")
    print(f"    {osc_text('roll', 'roll', 'deg')}; {osc_text('commanded_roll', 'commanded roll', 'deg')}"
          + (f"; max |roll| {summary['max_abs_roll_deg']:.1f} deg" if "max_abs_roll_deg" in summary else ""))
    print(f"    {osc_text('pitch', 'pitch', 'deg')}; {osc_text('altitude', 'altitude', 'm')}"
          + (f"; roll vs command {summary['roll_minus_command_rms_deg']:.1f} deg rms"
             if "roll_minus_command_rms_deg" in summary else ""))
    print(f"  {'COLLISION-FREE' if summary['collision_free'] else 'COLLISION or left the workspace'}"
          + (f"  ({summary['samples_in_collision']} samples inside an obstacle)" if summary['samples_in_collision'] else ""))

    if "log_start" in f:
        import datetime
        summary["flown_at"] = datetime.datetime.fromtimestamp(f["log_start"]).isoformat(timespec="seconds")
    if getattr(args, "note", None):
        summary["note"] = args.note
    summary["n_plan_waypoints"] = meta.get("n_plan_waypoints")
    summary["chord_tol_m"] = meta.get("chord_tol_m", 3.0)        # missions made before this was recorded used 3 m
    log_path = str(find_input(args.log))
    base = os.path.splitext(log_path)[0]                              # outputs go next to the log
    with open(base + "_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    import csv
    lat, lon = f["lat"][seg], f["lon"][seg]
    with open(base + "_flight.csv", "w", newline="") as fh:                 # same columns as `record`
        w = csv.writer(fh)
        w.writerow(["t_s", "x_east_m", "y_north_m", "z_up_m", "lat_deg", "lon_deg", "alt_rel_m"]
                   + (["airspeed_ms"] if "airspeed" in f else []))
        for k in range(len(Fs)):
            w.writerow([f"{v:.8g}" for v in (ts[k], *Fs[k], lat[k], lon[k], f["alt_rel"][seg][k])]
                       + ([f"{f['airspeed'][seg][k]:.4g}"] if "airspeed" in f else []))
    print(f"  wrote {base}_summary.json and {base}_flight.csv")
    plot(env, Pp, F, Fs, err, ts, base, f.get("attitude"), f.get("nav"), t[i0], ts[-1],
         (t, f["throttle"]) if "throttle" in f else None, summary,
         (ts, f["airspeed"][seg], v_plan_near) if v_plan_near is not None else None)
    archive_run(log_path, base, summary, f.get("params", {}), str(find_input(args.mission)),
                root=getattr(args, "runs_dir", None))
    import matplotlib.pyplot as plt
    if args.plot:
        plt.show()
    plt.close("all")
    return summary


# --------------------------------------------------------------------------
# run archive (analyze)
# --------------------------------------------------------------------------
def _file_hash(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_params(path, params):
    with open(path, "w") as fh:
        for k in sorted(params):
            fh.write(f"{k:<17} {params[k]:.7g}\n")


def read_params(path):
    out = {}
    if os.path.exists(path):
        for line in open(path):
            parts = line.split()
            if len(parts) >= 2 and not parts[0].startswith("#"):
                out[parts[0]] = float(parts[1])
    return out


def param_changes(old, new):
    "'NAME old->new' for every parameter that differs, leaving out PARAM_IGNORE."
    keys = sorted(k for k in set(old) | set(new)
                  if not k.startswith(PARAM_IGNORE) and old.get(k) != new.get(k))
    fmt = lambda v: "-" if v is None else f"{v:.6g}"
    return [f"{k} {fmt(old.get(k))}->{fmt(new.get(k))}" for k in keys]


def archive_run(log_path, base, summary, params, mission_path=None, root=None):
    "Copy this flight's log, results, parameters and mission to <log folder>/runs/<log name>/ (or root/<log name>/)."
    import shutil
    name = os.path.basename(base)
    root = root or os.path.join(os.path.dirname(log_path), RUNS_DIR)
    os.makedirs(root, exist_ok=True)
    digest = _file_hash(log_path)
    run, k = os.path.join(root, name), 1
    while os.path.exists(os.path.join(run, "log_sha256.txt")) and \
            open(os.path.join(run, "log_sha256.txt")).read().strip() != digest:
        k += 1                                  # a different flight was already archived under this name
        run = os.path.join(root, f"{name}_{k}")
    if k > 1 and not os.path.exists(run):
        print(f"  NOTE: a different flight is already archived as runs/{name}; was {os.path.basename(log_path)} "
              f"overwritten? This one is archived as runs/{os.path.basename(run)}.")
    os.makedirs(run, exist_ok=True)
    ext = os.path.splitext(log_path)[1]
    if not os.path.exists(os.path.join(run, "flight" + ext)):
        shutil.copy2(log_path, os.path.join(run, "flight" + ext))
    open(os.path.join(run, "log_sha256.txt"), "w").write(digest + "\n")
    for suffix in ("_summary.json", "_flight.csv", "_check.png"):
        if os.path.exists(base + suffix):
            shutil.copy2(base + suffix, os.path.join(run, suffix[1:]))
    if params:
        write_params(os.path.join(run, "params.parm"), params)
    if mission_path:
        shutil.copy2(mission_path, os.path.join(run, "mission.waypoints"))
    write_runs_table(root)
    print(f"  archived in {run} (all runs: {os.path.join(root, 'runs.csv')})")


def write_runs_table(root):
    "runs.csv: one row per archived run, in the order flown, with the parameters changed since the previous run."
    import csv
    runs = []
    for d in sorted(os.listdir(root)):
        sp = os.path.join(root, d, "summary.json")
        if os.path.exists(sp):
            s = json.load(open(sp))
            runs.append((s.get("flown_at", ""), d, s, read_params(os.path.join(root, d, "params.parm"))))
    runs.sort(key=lambda r: (r[0], r[1]))
    cols = [("rms_error_m", 1), ("max_error_m", 1), ("min_clearance_m", 1), ("collision_free", None),
            ("reached_goal", None), ("roll_oscillation_deg", 2), ("commanded_roll_oscillation_deg", 2),
            ("pitch_oscillation_deg", 2), ("altitude_oscillation_m", 2), ("roll_oscillation_period_s", 1),
            ("max_abs_roll_deg", 1), ("mean_airspeed_ms", 1)]
    with open(os.path.join(root, "runs.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "flown_at", "mission", "n_plan_waypoints", "chord_tol_m"] + [c for c, _ in cols]
                   + ["changed_since_previous_run", "note"])
        prev = None
        for flown, d, s, p in runs:
            v = [s.get(c) for c, _ in cols]
            v = ["" if x is None else (f"{x:.{nd}f}" if nd is not None else x) for x, (_, nd) in zip(v, cols)]
            changed = "; ".join(param_changes(prev, p)) if prev is not None and p else ""
            w.writerow([d, flown, os.path.basename(str(s.get("mission", ""))), s.get("n_plan_waypoints", ""),
                        s.get("chord_tol_m", "")] + v + [changed, s.get("note", "")])
            prev = p if p else prev


def plot(env, Pp, F, Fs, err, ts, base, att=None, nav=None, t0=0.0, t_end=None, thr=None, summary=None,
         aspd=None):
    import matplotlib.pyplot as plt
    from environment.envgen import plot_environment
    ax = plot_environment(env)
    ax.plot(*Pp.T, "k--", lw=1.5, label="SCP plan")
    ax.plot(*Fs.T, "m-", lw=1.5, label="ArduPilot SITL")
    ax.legend(loc="upper left")
    fig, axs = plt.subplots(2, 3, figsize=(19, 9))
    axs = axs.ravel()
    summary = summary or {}

    def osc_title(name, key, unit):
        v = summary.get(key)
        return name if v is None else f"{name}: oscillation {v:.2f} {unit} rms (after {OSC_SKIP_S:.0f} s)"
    axs[0].plot(F[:, 0], F[:, 1], "-", color="0.75", lw=1, label="whole flight")
    axs[0].plot(Pp[:, 0], Pp[:, 1], "k--", lw=1.5, label="plan")
    axs[0].plot(Fs[:, 0], Fs[:, 1], "m-", lw=1.2, label="flown (plan segment)")
    b0, b1 = env.bounds_lo, env.bounds_hi
    axs[0].plot([b0[0], b1[0], b1[0], b0[0], b0[0]], [b0[1], b0[1], b1[1], b1[1], b0[1]], "b:", lw=1, label="workspace")
    axs[0].set_aspect("equal"); axs[0].set_xlabel("east [m]"); axs[0].set_ylabel("north [m]")
    axs[0].set_title("top view"); axs[0].legend(fontsize=8)
    sp = np.r_[0, np.cumsum(np.linalg.norm(np.diff(Pp[:, :2], axis=0), axis=1))]
    sf = np.r_[0, np.cumsum(np.linalg.norm(np.diff(Fs[:, :2], axis=0), axis=1))]
    axs[1].plot(sp / 1000, Pp[:, 2], "k--", label="plan")
    axs[1].plot(sf / 1000, Fs[:, 2], "m-", label="flown")
    axs[1].set_xlabel("distance flown [km]"); axs[1].set_ylabel("z (above workspace floor) [m]")
    axs[1].set_title("altitude"); axs[1].legend(fontsize=8)
    axs[2].plot(ts, err, "m-")
    axs[2].set_xlabel("time since plan start [s]"); axs[2].set_ylabel("distance from planned path [m]")
    axs[2].set_title("tracking error"); axs[2].set_ylim(bottom=0)
    if att is not None:                                        # roll and pitch over the plan segment
        A = att[(att[:, 0] >= t0) & (att[:, 0] <= t0 + t_end)]
        Nv = None if nav is None else nav[(nav[:, 0] >= t0) & (nav[:, 0] <= t0 + t_end)]
        for k, (col, name, key) in enumerate(((1, "roll", "roll_oscillation_deg"),
                                               (2, "pitch", "pitch_oscillation_deg"))):
            a = axs[3 + k]
            a.plot(A[:, 0] - t0, A[:, col], "m-", lw=0.8, label=name)
            if Nv is not None:
                a.plot(Nv[:, 0] - t0, Nv[:, col], "k-", lw=0.6, alpha=0.6, label=f"commanded {name}")
            a.axvline(OSC_SKIP_S, color="0.6", ls=":", lw=1)
            a.set_xlabel("time since plan start [s]"); a.set_ylabel(f"{name} [deg]")
            a.set_title(osc_title(name, key, "deg"), fontsize=10); a.legend(fontsize=8)
    else:
        axs[3].set_visible(False); axs[4].set_visible(False)
    if thr is not None:
        k = (thr[0] >= t0) & (thr[0] <= t0 + t_end)
        axs[5].plot(thr[0][k] - t0, thr[1][k], "r-", lw=0.8, label="throttle")
        axs[5].set_xlabel("time since plan start [s]"); axs[5].set_ylabel("throttle [%]", color="r")
        axs[5].set_title("throttle and airspeed")
        if aspd is not None:                                   # airspeed against the plan's, right axis
            a2 = axs[5].twinx()
            a2.plot(aspd[0], aspd[1], "m-", lw=0.9, label="airspeed")
            a2.plot(aspd[0], aspd[2], "k--", lw=1.0, label="plan airspeed")
            a2.set_ylabel("airspeed [m/s]")
            a2.legend(fontsize=8, loc="lower right")
            v = summary.get("airspeed_vs_plan_rms_ms")
            if v is not None:
                axs[5].set_title(f"throttle and airspeed (airspeed {v:.1f} m/s rms from the plan's)")
    else:
        axs[5].set_visible(False)
    v = summary.get("altitude_oscillation_m")
    if v is not None:
        axs[1].set_title(f"altitude: oscillation {v:.2f} m rms")
    for a in axs:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(base + "_check.png", dpi=130)
    print(f"  plot saved to {base}_check.png")


# --------------------------------------------------------------------------
# summary (Tier 1)
# --------------------------------------------------------------------------
def legs_position(P, corners):
    """For each point: nearest leg of the closed rectangle, distance from it,
    distance along it from its start corner, and the leg length."""
    A = corners
    B = np.roll(corners, -1, axis=0)
    AB = B - A
    L = np.linalg.norm(AB, axis=1)
    u = AB / L[:, None]
    rel = P[:, None, :] - A[None]                           # (n, 4, 2)
    along = (rel * u[None]).sum(-1)
    t = np.clip(along / L[None], 0, 1)
    d = np.linalg.norm(rel - t[..., None] * AB[None], axis=-1)
    k = d.argmin(1)
    i = np.arange(len(P))
    cross = rel[i, k, 0] * u[k, 1] - rel[i, k, 1] * u[k, 0]  # signed: + = right of the leg
    return k, d[i, k], along[i, k], L[k], cross


def summary(args):
    meta = json.load(open(find_input(args.mission + ".json")))
    if meta.get("kind") != "box":
        raise SystemExit(f"{args.mission} is not a box mission (for SCP missions use analyze).")
    f = read_flight(args.log)
    x, y = geo_to_enu(f["lat"], f["lon"])
    P = np.c_[x, y]
    z, t = f["alt_rel"], f["t"]
    corners = np.array(meta["corners_enu"])
    target_alt, v_cmd = meta["alt_rel_m"], meta["commanded_airspeed_ms"]

    leg, dist, along, L, cross = legs_position(P, corners)
    on_leg = (dist < LEG_TOL_M) & (along > SETTLE_M) & (along < L - LEG_END_M)
    at_alt = np.nonzero(on_leg & (np.abs(z - target_alt) < 10.0))[0]
    if z.max() < 50:
        raise SystemExit(f"The aircraft never got above 50 m (max {z.max():.0f} m above home): it did not "
                         f"take off. Check the arming messages in MAVProxy.")
    if not len(at_alt):
        raise SystemExit("The aircraft never settled on a leg of the box at the box altitude "
                         f"(closest approach to the box {dist.min():.0f} m). Did the mission run, and is "
                         "SITL's home where 'box' said?")
    # start measuring on the first leg the aircraft enters after reaching the box altitude,
    # so the end of the climb is not counted as an altitude error
    entries = np.nonzero(np.diff(np.r_[0, on_leg.astype(int)]) == 1)[0]
    later = entries[entries > at_alt[0]]
    if not len(later):
        raise SystemExit("The aircraft reached the box altitude but did not fly a full leg afterwards; "
                         "let the mission run longer (at least one lap after the climb).")
    settled = on_leg & (np.arange(len(t)) >= later[0])
    # number of straight legs measured (each separate stretch of settled samples)
    n_legs = int(np.sum(np.diff(np.r_[0, settled.astype(int)]) == 1))

    alt_err = z[settled] - target_alt
    xt = cross[settled]
    res = {"log": args.log, "mission": args.mission, "settled_samples": int(settled.sum()),
           "settled_time_s": float(np.sum(np.diff(t, append=t[-1])[settled])),
           "legs_measured": n_legs, "max_alt_rel_m": float(z.max()),
           "altitude_error_mean_m": float(alt_err.mean()),
           "altitude_error_rms_m": float(np.sqrt(np.mean(alt_err**2))),
           "altitude_error_max_m": float(np.abs(alt_err).max()),
           "cross_track_rms_m": float(np.sqrt(np.mean(xt**2))),
           "cross_track_max_m": float(np.abs(xt).max()),
           "commanded_airspeed_ms": v_cmd, "target_alt_rel_m": target_alt}
    if "airspeed" in f:
        a = f["airspeed"][settled]
        res.update(airspeed_mean_ms=float(a.mean()), airspeed_std_ms=float(a.std()),
                   airspeed_min_ms=float(a.min()), airspeed_max_ms=float(a.max()))
    if "throttle" in f:
        th = f["throttle"][settled]
        res.update(throttle_mean_pct=float(th.mean()), throttle_std_pct=float(th.std()))

    print(f"Flight log {args.log}: {len(t)} samples over {t[-1] / 60:.1f} min; "
          f"{res['settled_time_s'] / 60:.1f} min settled on {n_legs} straight legs")
    if "airspeed_mean_ms" in res:
        print(f"  airspeed   mean {res['airspeed_mean_ms']:.1f} m/s (commanded {v_cmd:.1f}), "
              f"spread +-{res['airspeed_std_ms']:.1f}, range {res['airspeed_min_ms']:.1f}-{res['airspeed_max_ms']:.1f}")
    print(f"  altitude   error mean {res['altitude_error_mean_m']:+.1f} m, rms {res['altitude_error_rms_m']:.1f} m, "
          f"max {res['altitude_error_max_m']:.1f} m (target {target_alt:.0f} m above home)")
    print(f"  course     distance from the leg: rms {res['cross_track_rms_m']:.1f} m, max {res['cross_track_max_m']:.1f} m")
    if "throttle_mean_pct" in res:
        print(f"  throttle   mean {res['throttle_mean_pct']:.0f}% in level flight (TRIM_THROTTLE is {TRIM_THROTTLE_PCT:.0f})")

    print("\nTier 1 checks")
    verdicts = []
    def judge(level, ok, text, fix=""):
        verdicts.append(ok or level == "WARN")
        print(f"  {'PASS' if ok else level}  {text}" + ("" if ok or not fix else f"\n        -> {fix}"))
    judge("FAIL", n_legs >= 4, f"took off and flew at least one full lap of settled legs ({n_legs} legs measured)")
    if "airspeed_mean_ms" in res:
        judge("WARN", abs(res["airspeed_mean_ms"] - v_cmd) <= 2.0,
              f"holds cruise airspeed within 2 m/s ({res['airspeed_mean_ms'] - v_cmd:+.1f} m/s)",
              "check AIRSPEED_CRUISE / TRIM_THROTTLE; if the throttle is at 100% the C172 cannot reach that speed")
    judge("WARN", res["altitude_error_rms_m"] <= 10.0,
          f"holds altitude within 10 m rms ({res['altitude_error_rms_m']:.1f} m)",
          "TECS tuning: set TRIM_THROTTLE to the measured value first; then consider AUTOTUNE")
    judge("WARN", res["cross_track_rms_m"] <= 20.0,
          f"tracks the straight legs within 20 m rms ({res['cross_track_rms_m']:.1f} m)",
          "lower NAVL1_PERIOD a little (e.g. 17) for tighter tracking, or AUTOTUNE the roll loop")
    if "throttle_mean_pct" in res:
        judge("WARN", abs(res["throttle_mean_pct"] - TRIM_THROTTLE_PCT) <= 5.0,
              f"TRIM_THROTTLE matches the throttle needed for level cruise "
              f"({res['throttle_mean_pct']:.0f}% vs {TRIM_THROTTLE_PCT:.0f}%)",
              f"set TRIM_THROTTLE {res['throttle_mean_pct']:.0f} in ardupilot/c172_sitl.parm "
              f"(and TRIM_THROTTLE_PCT in ardupilot_mission.py) and fly again")
    print(f"\nOVERALL: {'PASS' if all(verdicts) else 'FAIL'}")

    base = os.path.splitext(str(find_input(args.log)))[0]           # outputs go next to the log
    with open(base + "_summary.json", "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"  wrote {base}_summary.json")
    if args.plot:
        plot_box(P, z, t, f, settled, corners, meta, base)


def plot_box(P, z, t, f, settled, corners, meta, base):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 2, figsize=(14, 8))
    loop = np.vstack([corners, corners[:1]])
    ax = axs[0, 0]
    ax.plot(loop[:, 0] / 1000, loop[:, 1] / 1000, "k--", lw=1.2, label="box (commanded)")
    ax.plot(P[:, 0] / 1000, P[:, 1] / 1000, "-", color="0.6", lw=1, label="flown")
    ax.plot(P[settled, 0] / 1000, P[settled, 1] / 1000, "m.", ms=2, label="settled on a leg (measured)")
    ax.plot(0, 0, "k^", ms=8, label="home")
    ax.set_aspect("equal"); ax.set_xlabel("east [km]"); ax.set_ylabel("north [km]")
    ax.set_title("top view"); ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    tm = t / 60
    def shade(a):
        a.fill_between(tm, 0, 1, where=settled, color="m", alpha=0.08, transform=a.get_xaxis_transform(),
                       label="settled")
    a = axs[0, 1]
    if "airspeed" in f:
        a.plot(tm, f["airspeed"], "b-", lw=1, label="airspeed")
    a.axhline(meta["commanded_airspeed_ms"], color="k", ls="--", lw=1, label="commanded")
    a.set_ylabel("airspeed [m/s]"); a.set_title("airspeed"); shade(a)
    a = axs[1, 0]
    a.plot(tm, z, "g-", lw=1, label="altitude above home")
    a.axhline(meta["alt_rel_m"], color="k", ls="--", lw=1, label="target")
    a.set_ylabel("altitude above home [m]"); a.set_title("altitude"); shade(a)
    a = axs[1, 1]
    if "throttle" in f:
        a.plot(tm, f["throttle"], "r-", lw=1, label="throttle")
    a.axhline(TRIM_THROTTLE_PCT, color="k", ls="--", lw=1, label="TRIM_THROTTLE")
    a.set_ylabel("throttle [%]"); a.set_title("throttle"); shade(a)
    for a in (axs[0, 1], axs[1, 0], axs[1, 1]):
        a.set_xlabel("time [min]"); a.grid(alpha=0.3); a.legend(fontsize=8)
    axs[0, 0].grid(alpha=0.3)
    fig.suptitle("ArduPilot SITL, JSBSim C172: box test flight", fontsize=11)
    fig.tight_layout()
    fig.savefig(base + "_summary.png", dpi=120)
    print(f"  plot saved to {base}_summary.png")
    plt.show()


# --------------------------------------------------------------------------
# wobble: oscillation in each flight-mode segment of any flight
# --------------------------------------------------------------------------
WOBBLE_MIN_S         = 40.0             # report flight-mode segments at least this long ...
WOBBLE_SETTLE_S      = 10.0             # ... leaving out this much after each mode change


def wobble(args):
    """Roll, pitch, altitude and throttle oscillation (same measure as analyze) for
    every stretch of the flight spent in one flight mode with the same parameters:
    e.g. FBWA with the sticks centred against AUTO, or several values of a parameter
    tried in flight with `param set` while circling in the final loiter."""
    f = read_tlog(str(find_input(args.log)))
    if "mode" not in f or "attitude" not in f:
        raise SystemExit("This log has no HEARTBEAT or ATTITUDE messages.")
    md, A = f["mode"], f["attitude"]
    # segment boundaries: every mode change and every parameter change
    cuts = [(float(md[0, 0]), "")]
    cuts += [(float(md[k, 0]), "") for k in np.nonzero(np.diff(md[:, 1]) != 0)[0] + 1]
    for tc, pname, old, new in f.get("param_changes", []):
        cuts.append((tc, f"{pname} {old:g}->{new:g}"))
    cuts.sort(key=lambda c: c[0])
    merged = []                                          # changes within a few seconds form one boundary
    for tc, label in cuts:
        if merged and tc - merged[-1][0] < 3.0:
            merged[-1] = (merged[-1][0], "; ".join(x for x in (merged[-1][1], label) if x))
        else:
            merged.append((tc, label))
    print(f"Flight log {args.log}: oscillation = rms of the part faster than {OSC_WINDOW_S:.0f} s "
          f"(first {WOBBLE_SETTLE_S:.0f} s after each mode or parameter change left out)")
    print(f"  {'mode':<9} {'from':>7} {'length':>7} | {'roll':>9} {'pitch':>9} {'altitude':>9} "
          f"{'throttle':>9} | mean roll, throttle | pitch lag behind command, size | parameters changed at the start")
    rows = []
    for (ta, label), tb in zip(merged, [c[0] for c in merged[1:]] + [float(f["t"][-1])]):
        t0, t1 = ta + WOBBLE_SETTLE_S, tb
        if t1 - t0 < WOBBLE_MIN_S - WOBBLE_SETTLE_S:
            continue
        mode_now = md[max(0, np.searchsorted(md[:, 0], t0) - 1), 1]           # the mode after the settle time
        name = PLANE_MODES.get(int(mode_now), str(int(mode_now)))
        sa = (A[:, 0] >= t0) & (A[:, 0] <= t1)
        sp = (f["t"] >= t0) & (f["t"] <= t1)
        if "airspeed" in f and sp.any() and np.mean(f["airspeed"][sp]) < 15.0:
            continue                                     # still on the ground
        osc = {"roll": oscillation(A[sa, 0], A[sa, 1]), "pitch": oscillation(A[sa, 0], A[sa, 2]),
               "altitude": oscillation(f["t"][sp], f["alt_rel"][sp])}
        if "throttle" in f:
            osc["throttle"] = oscillation(f["t"][sp], f["throttle"][sp])
        lag, ratio = (None, None)
        if "nav" in f:                                   # how closely pitch follows the autopilot's pitch command
            Nv = f["nav"][(f["nav"][:, 0] >= t0) & (f["nav"][:, 0] <= t1)]
            if len(Nv) > 10 and np.ptp(Nv[:, 2]) > 0.2:
                lag, ratio = follow_lag(Nv[:, 0], Nv[:, 2], A[sa, 0], A[sa, 2])
        cell = lambda k, u: (f"{osc[k][0]:.2f}{u}" if k in osc and osc[k][0] is not None else "n/a")
        thr = f"{np.mean(f['throttle'][sp]):.0f}%" if "throttle" in f else "n/a"
        print(f"  {name:<9} {t0 - WOBBLE_SETTLE_S:6.0f}s {t1 - t0 + WOBBLE_SETTLE_S:6.0f}s | {cell('roll', ' deg'):>9} "
              f"{cell('pitch', ' deg'):>9} {cell('altitude', ' m'):>9} {cell('throttle', ' %'):>9} | "
              f"{np.mean(A[sa, 1]):+.1f} deg, {thr} | "
              + (f"{lag:.2f} s x{ratio:.2f}" if lag is not None else "-") + f" | {label}")
        rows.append({"mode": name, "start_s": float(t0 - WOBBLE_SETTLE_S), "length_s": float(t1 - t0 + WOBBLE_SETTLE_S),
                     "params_changed": label, "pitch_lag_s": lag, "pitch_vs_command": ratio,
                     **{f"{k}_oscillation": v[0] for k, v in osc.items()},
                     **{f"{k}_period_s": v[1] for k, v in osc.items()}})
    base = os.path.splitext(str(find_input(args.log)))[0]
    with open(base + "_wobble.json", "w") as fh:
        json.dump(rows, fh, indent=2)
    print(f"  wrote {base}_wobble.json")


# --------------------------------------------------------------------------
# Tier 3: batches of generated environments
# --------------------------------------------------------------------------
def batch_path(name, *parts):
    """localstore/batches/<name>/<parts...> (folders created as needed)."""
    from environment.envgen import localstore_dir
    d = localstore_dir() / BATCH_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    if not parts:
        return d
    p = d.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def batch_make(args):
    """Missions for every solved environment in one or more (environments, results)
    pairs, a flyability check for each, and the manifest run_batch.py flies from."""
    from environment.envgen import load_environments
    from solvers.scp_aircraft import load_plan
    import datetime
    chord_tol = args.chord_tol
    if (batch_path(args.name) / "manifest.json").exists() and not args.overwrite:
        raise SystemExit(f"Batch '{args.name}' already exists ({batch_path(args.name)}). Pick a new --name, "
                         f"or add --overwrite to rebuild its missions (flown logs are kept).")
    instances = []
    for envs_file, results_file in args.pair:
        envs = load_environments(envs_file)
        by_index = {r["env_index"]: r for r in json.load(open(find_input(results_file)))}
        stem = os.path.splitext(os.path.basename(envs_file))[0]
        for i in range(len(envs)):
            iid = f"{stem}_{i}"
            rec = {"id": iid, "envs": envs_file, "results": results_file, "index": i}
            r0 = by_index.get(i)
            if r0 is None or not r0.get("success"):
                rec.update(status="not_solved",
                           reason="no result" if r0 is None else f"SCP status {r0.get('status')}")
                instances.append(rec); print(f"  {iid}: not solved ({rec['reason']})"); continue
            env, r, _ = load_plan(envs_file, results_file, i)
            m = build_mission(env, r, chord_tol, getattr(args, "speed_profile", None))
            out = batch_path(args.name, "missions", f"{iid}.waypoints")
            write_mission(m, out, envs_file, results_file, i, chord_tol, r["flight_time"])
            log = batch_path(args.name, "logs", f"{iid}.tlog")
            rec.update(status="flyable" if not m["rejects"] else "not_flyable",
                       rejects=m["rejects"], warnings=m["warnings"], checks=m["checks"],
                       mission=str(out), mission_wsl=wsl_path(out), home=sitl_home(m),
                       log=str(log), log_wsl=wsl_path(log), margin_m=env.margin,
                       plan_flight_time_s=float(r["flight_time"]), expected_sim_s=m["expected_sim_s"],
                       n_obstacles=len(env.obstacles))
            instances.append(rec)
            c = m["checks"]
            print(f"  {iid}: {rec['status']:<11} {len(m['W'])} waypoints, expected path keeps "
                  f"{c['est_min_clearance_m']:.1f} m (plan {c['plan_min_clearance_m']:.1f} m)"
                  + "".join(f"\n      NOT FLYABLE: {x}" for x in m["rejects"]))
    manifest = {"name": args.name, "created": datetime.datetime.now().isoformat(timespec="seconds"),
                "note": args.note or "", "chord_tol_m": chord_tol, "speedup": args.speedup,
                "speed_profile": getattr(args, "speed_profile", None) or SPEED_PROFILE,
                "param_overrides": parse_param_overrides(getattr(args, "param", None) or []),
                "parm_wsl": wsl_path(project_root() / SITL_PARAM_FILE), "instances": instances}
    mf = batch_path(args.name) / "manifest.json"
    with open(mf, "w") as fh:
        json.dump(manifest, fh, indent=2)
    n = {k: sum(x["status"] == k for x in instances) for k in ("flyable", "not_flyable", "not_solved")}
    if manifest["param_overrides"]:
        print("  parameters for this batch (set before arming, on top of c172_sitl.parm): "
              + ", ".join(f"{k}={v:g}" for k, v in manifest["param_overrides"].items()))
    print(f"""
Batch '{args.name}': {len(instances)} environments: {n['flyable']} flyable, {n['not_flyable']} not flyable as
waypoints, {n['not_solved']} not solved. Wrote {mf}

To fly the flyable ones (WSL, a terminal with your ArduPilot Python environment active;
close any running SITL first). It runs them one after another, unattended:
  python3 {wsl_path(project_root() / "ardupilot" / "run_batch.py")} {wsl_path(mf)}
Then, in PyCharm's terminal:
  python -m ardupilot.ardupilot_mission batch-analyze --name {args.name}
""")


def parse_param_overrides(items):
    """["TECS_SINK_MAX=7", ...] -> {"TECS_SINK_MAX": 7.0, ...}"""
    out = {}
    for it in items:
        name, sep, value = it.partition("=")
        if not sep or not name.strip():
            raise SystemExit(f"--param must look like NAME=VALUE, not {it!r}")
        try:
            out[name.strip().upper()] = float(value)
        except ValueError:
            raise SystemExit(f"--param {it!r}: {value!r} is not a number")
    return out


def evaluate_flight(s, margin):
    """Pass/fail for one analyzed flight (the PASS_* settings). Returns (passed, reasons, near_miss)."""
    fails = []
    if s["samples_in_collision"] > 0:
        fails.append(f"collision ({s['min_clearance_m']:.1f} m clearance)")
    if s["left_workspace_m"] > 0:
        fails.append(f"left the workspace by {s['left_workspace_m']:.0f} m")
    if s["goal_miss_m"] > PASS_GOAL_M:
        fails.append(f"missed the goal by {s['goal_miss_m']:.0f} m")
    if s["max_error_m"] >= PASS_MAX_ERROR_FRAC * margin:
        fails.append(f"max {s['max_error_m']:.1f} m from the plan (limit {PASS_MAX_ERROR_FRAC * margin:.0f})")
    if s["rms_error_m"] > PASS_RMS_M:
        fails.append(f"rms {s['rms_error_m']:.1f} m from the plan (limit {PASS_RMS_M:.0f})")
    if s["reached_goal"]:
        ratio = s["flight_time_s"] / s["plan_flight_time_s"]
        if abs(ratio - 1) > PASS_TIME_FRAC:
            fails.append(f"flight time {100 * (ratio - 1):+.0f}% vs the plan (limit +-{100 * PASS_TIME_FRAC:.0f}%)")
    return not fails, fails, s["min_clearance_m"] < NEAR_MISS_FRAC * margin


def batch_analyze(args):
    """Analyze every flown instance of a batch, score it, and summarize the batch."""
    from types import SimpleNamespace
    import csv
    bd = batch_path(args.name)
    mf = bd / "manifest.json"
    if not mf.exists():
        raise SystemExit(f"No batch '{args.name}' ({mf} not found). Run batch-make first.")
    manifest = json.load(open(mf))
    rows = []
    for inst in manifest["instances"]:
        row = {"id": inst["id"], "status": inst["status"], "outcome": "", "reasons": "", "near_miss": "",
               "warnings": len(inst.get("warnings", []))}
        if inst["status"] == "not_solved":
            row.update(outcome="NOT_SOLVED", reasons=inst.get("reason", ""))
        elif inst["status"] == "not_flyable":
            row.update(outcome="REJECTED", reasons="; ".join(inst["rejects"]))
        else:
            log = Path(inst["log"])
            run_info = {}
            rj = log.with_suffix(".run.json")
            if rj.exists():
                run_info = json.load(open(rj))
            row["run_status"] = run_info.get("status", "")
            if not log.exists() or run_info.get("status") in ("sitl_failed", "upload_failed", "arm_failed", "param_failed"):
                row.update(outcome="NOT_FLOWN", reasons=run_info.get("detail", "no log"))
            else:
                print(f"\n=== {inst['id']} ===")
                try:
                    s = analyze(SimpleNamespace(mission=inst["mission"], log=str(log), plot=False,
                                                note=f"batch {args.name}", runs_dir=str(bd / "runs")))
                except SystemExit as e:                             # e.g. never reached the plan's start
                    row.update(outcome="FAIL", reasons=str(e)); rows.append(row); continue
                ok, why, near = evaluate_flight(s, inst["margin_m"])
                if run_info.get("status") not in (None, "completed"):
                    why.append(f"run {run_info.get('status')}: {run_info.get('detail', '')}"); ok = False
                row.update(outcome="PASS" if ok else "FAIL", reasons="; ".join(why), near_miss="yes" if near else "")
                for k in ("rms_error_m", "max_error_m", "min_clearance_m", "plan_min_clearance_m", "goal_miss_m",
                          "flight_time_s", "plan_flight_time_s", "roll_oscillation_deg", "pitch_oscillation_deg",
                          "altitude_oscillation_m", "max_horizontal_error_m", "max_vertical_error_m",
                          "airspeed_vs_plan_rms_ms"):
                    row[k] = s.get(k)
        rows.append(row)

    cols = ["id", "outcome", "reasons", "near_miss", "status", "run_status", "rms_error_m", "max_error_m",
            "min_clearance_m", "plan_min_clearance_m", "goal_miss_m", "flight_time_s", "plan_flight_time_s",
            "roll_oscillation_deg", "pitch_oscillation_deg", "altitude_oscillation_m", "max_horizontal_error_m",
            "max_vertical_error_m", "airspeed_vs_plan_rms_ms", "warnings"]
    fmt = lambda v: f"{v:.2f}" if isinstance(v, float) else ("" if v is None else v)
    with open(bd / "results.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r_ in rows:
            w.writerow([fmt(r_.get(c)) for c in cols])
    counts = {k: sum(r_["outcome"] == k for r_ in rows) for k in ("PASS", "FAIL", "REJECTED", "NOT_FLOWN", "NOT_SOLVED")}
    flown = counts["PASS"] + counts["FAIL"]
    summary = {"name": args.name, "n_instances": len(rows), **{k.lower(): v for k, v in counts.items()},
               "pass_rate_of_flown": counts["PASS"] / flown if flown else None,
               "near_misses": sum(r_["near_miss"] == "yes" for r_ in rows),
               "thresholds": {"max_error_frac_of_margin": PASS_MAX_ERROR_FRAC, "rms_m": PASS_RMS_M,
                              "goal_m": PASS_GOAL_M, "time_frac": PASS_TIME_FRAC, "near_miss_frac": NEAR_MISS_FRAC}}
    with open(bd / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    plot_batch(rows, bd / "summary.png", args.name)

    print(f"\nBatch '{args.name}': {len(rows)} environments")
    print(f"  {'id':<22} {'outcome':<10} {'rms':>6} {'max':>6} {'clear':>6}  reasons")
    for r_ in rows:
        num = lambda k: f"{r_[k]:6.1f}" if isinstance(r_.get(k), float) else f"{'':>6}"
        print(f"  {r_['id']:<22} {r_['outcome']:<10} {num('rms_error_m')} {num('max_error_m')} "
              f"{num('min_clearance_m')}  {r_['reasons']}{'  [near miss]' if r_['near_miss'] else ''}")
    rate = f"{100 * summary['pass_rate_of_flown']:.0f}%" if flown else "n/a"
    print(f"  PASS {counts['PASS']}, FAIL {counts['FAIL']}, REJECTED (not flyable as waypoints) {counts['REJECTED']}, "
          f"NOT FLOWN {counts['NOT_FLOWN']}, NOT SOLVED {counts['NOT_SOLVED']}; pass rate of flown: {rate}")
    print(f"  wrote {bd / 'results.csv'}, {bd / 'summary.json'} and {bd / 'summary.png'}; "
          f"per-flight plots and logs in {bd / 'runs'}")
    import matplotlib.pyplot as plt
    if args.plot:
        plt.show()
    plt.close("all")


def plot_batch(rows, path, name):
    import matplotlib.pyplot as plt
    order = ["PASS", "FAIL", "REJECTED", "NOT_FLOWN", "NOT_SOLVED"]
    colours = {"PASS": "tab:green", "FAIL": "tab:red", "REJECTED": "tab:orange", "NOT_FLOWN": "0.6",
               "NOT_SOLVED": "0.8"}
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
    counts = [sum(r_["outcome"] == k for r_ in rows) for k in order]
    axs[0].bar(order, counts, color=[colours[k] for k in order])
    axs[0].set_title(f"batch {name}: outcomes ({len(rows)} environments)"); axs[0].set_ylabel("environments")
    for x, c in enumerate(counts):
        axs[0].text(x, c, str(c), ha="center", va="bottom")
    flown = [r_ for r_ in rows if isinstance(r_.get("rms_error_m"), float)]
    for r_ in flown:
        axs[1].scatter(r_["rms_error_m"], r_["min_clearance_m"], color=colours[r_["outcome"]], s=40)
        axs[1].annotate(r_["id"], (r_["rms_error_m"], r_["min_clearance_m"]), fontsize=7,
                        xytext=(3, 3), textcoords="offset points")
    axs[1].axvline(PASS_RMS_M, color="k", ls="--", lw=1, label=f"rms limit {PASS_RMS_M:.0f} m")
    axs[1].axhline(0, color="tab:red", lw=1, label="touching an obstacle")
    axs[1].set_xlabel("rms distance from the plan [m]"); axs[1].set_ylabel("closest approach to an obstacle [m]")
    axs[1].set_title("flown instances"); axs[1].legend(fontsize=8)
    for a in axs:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Fly the C172 in ArduPilot SITL and check the flights.")
    sub = ap.add_subparsers(dest="command", required=True)
    b = sub.add_parser("box", help="Tier 1: write a rectangle test mission around home")
    b.add_argument("--heading", type=float, default=0.0, help="runway / long-leg heading, deg from north")
    b.add_argument("--alt", type=float, default=BOX_ALT_M, help="box altitude above home, m")
    b.add_argument("--length", type=float, default=BOX_LENGTH_M)
    b.add_argument("--width", type=float, default=BOX_WIDTH_M)
    b.add_argument("--laps", type=int, default=BOX_LAPS)
    b.add_argument("--speed", type=float, default=None, help="airspeed, m/s (default: the C172's cruise)")
    b.add_argument("--out", default="box.waypoints")
    s_ = sub.add_parser("summary", help="Tier 1: check a flown box mission")
    s_.add_argument("--mission", default="box.waypoints", help="the .waypoints file written by box")
    s_.add_argument("--log", required=True, help="MAVProxy .tlog (or DataFlash .BIN with pymavlink)")
    s_.add_argument("--plot", action="store_true")
    m = sub.add_parser("make", help="Tier 2: write an ArduPilot mission (.waypoints) for one SCP plan")
    m.add_argument("--envs", default="tfr_env.json")
    m.add_argument("--results", default="tfr_results.json")
    m.add_argument("--index", type=int, default=0)
    m.add_argument("--out", default=None, help="mission file name (default <envs><index>.waypoints)")
    m.add_argument("--speed-profile", choices=["plan", "constant"], default=None,
                   help=f"airspeed: follow the plan per leg, or one constant speed (default {SPEED_PROFILE})")
    m.add_argument("--chord-tol", type=float, default=None,
                   help=f"how far a straight leg may stray from the plan, m (default CHORD_TOL_M = {CHORD_TOL_M:g})")
    a = sub.add_parser("analyze", help="Tier 2: check a flown SCP mission against the plan and the obstacles")
    a.add_argument("--mission", required=True, help="the .waypoints file written by make")
    a.add_argument("--log", required=True, help="MAVProxy .tlog (or DataFlash .BIN with pymavlink)")
    a.add_argument("--plot", action="store_true", help="show the plots (they are saved either way)")
    a.add_argument("--note", default=None, help='what was different about this run, e.g. "TECS_TIME_CONST 8"')
    w_ = sub.add_parser("wobble", help="oscillation in each flight mode of any flight (e.g. an FBWA test)")
    w_.add_argument("--log", required=True, help="MAVProxy .tlog")
    bm = sub.add_parser("batch-make", help="Tier 3: missions + flyability checks for many solved environments")
    bm.add_argument("--name", required=True, help="batch name (folder localstore/batches/<name>/)")
    bm.add_argument("--pair", nargs=2, action="append", required=True, metavar=("ENVS", "RESULTS"),
                    help="an environments file and the SCP results solved on it; repeat for several")
    bm.add_argument("--chord-tol", type=float, default=BATCH_CHORD_TOL_M)
    bm.add_argument("--speed-profile", choices=["plan", "constant"], default=None,
                    help=f"airspeed: follow the plan per leg, or one constant speed (default {SPEED_PROFILE})")
    bm.add_argument("--speedup", type=int, default=BATCH_SPEEDUP, help="SITL speedup for run_batch.py")
    bm.add_argument("--note", default=None, help='e.g. "tracking setup v1"')
    bm.add_argument("--param", action="append", default=[], metavar="NAME=VALUE",
                    help="autopilot parameter for this batch only, set by run_batch.py before arming "
                         "(on top of c172_sitl.parm); repeat for several")
    bm.add_argument("--overwrite", action="store_true", help="rebuild the missions of an existing batch")
    ba = sub.add_parser("batch-analyze", help="Tier 3: score every flown instance of a batch and summarize")
    ba.add_argument("--name", required=True)
    ba.add_argument("--plot", action="store_true", help="show the summary plot (saved either way)")
    args = ap.parse_args()
    {"box": box, "summary": summary, "make": make, "analyze": analyze, "wobble": wobble,
     "batch-make": batch_make, "batch-analyze": batch_analyze}[args.command](args)


if __name__ == "__main__":
    main()
