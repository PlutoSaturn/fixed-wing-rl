#!/usr/bin/env python3
"""
jsbsim_c172.py - Fly SCP trajectories with a JSBSim Cessna 172.

Four commands:

  calibrate   Measures the JSBSim C172 (mass, wing area, drag polar, stall,
              full-throttle thrust vs speed) with trim sweeps and writes a
              parameter file that scp_aircraft.py loads automatically, so the
              planner and the simulator describe the same airplane.

  track       Flies each trajectory in an scp_aircraft.py results file with
              JSBSim, using a guidance + autopilot loop, and reports how
              closely the aircraft followed the plan and whether the flown
              path stayed clear of the obstacles.

  record N    Flies environment N and writes, to flight_envN/, the JSBSim
              control commands and aircraft state (CSV), the SCP plan (CSV),
              and the environment as an OBJ mesh plus JSON, for rendering in
              an external program.

  fgview N    Flies environment N in real time and streams the aircraft to
              FlightGear, which acts as the 3D display (cockpit/chase views,
              scenery). JSBSim here stays the flight model; FlightGear runs
              with --fdm=null and just draws what it receives.

Typical workflow
    python jsbsim_c172.py calibrate
    python envgen.py --size 12000 2500 600 --vehicle-radius 6 --margin 30 \\
        --occupancy 0.02 0.06 --endpoints ends --n 10 --out c172_envs.json
    python scp_aircraft.py --envs c172_envs.json
    python jsbsim_c172.py track --plot 0
    python jsbsim_c172.py record 0      # export commands, trajectory, environment mesh
    python jsbsim_c172.py fgview 0      # watch the flight live in FlightGear

--------------------------------------------------------------------------
Tracking controller (classic successive loop closure)
--------------------------------------------------------------------------
Outer guidance loop, every GUIDANCE_DT seconds, compares the aircraft with
the plan at the same time t:
    airspeed command     V_cmd   = V_ref + K_ALONG * along-track error
    course command       chi_cmd = chi_ref + atan(K_CROSS * cross-track error)
    bank command         phi_cmd = phi_ref + K_COURSE * (chi_cmd - chi)
                         (sign flipped for JSBSim, where positive bank turns right)
    flight-path command  gam_cmd = gam_ref + atan(K_VERT * altitude error)
(phi_ref and gam_ref come straight from the plan as feedforward.)

Inner loops, every JSBSim step (120 Hz):
    aileron   <- bank error, roll-rate damping
    elevator  <- pitch attitude error (theta_cmd = gam_cmd + alpha + integral), pitch-rate damping
    throttle  <- airspeed PI with thrust feedforward from the plan
    rudder    <- sideslip feedback (turn coordination)

Coordinates: the workspace is a local East-North-Up frame (x east, y north,
z up) centred at ORIGIN_LAT/LON, with z = 0 at ALT_BASE_M above sea level.
Planner heading chi is measured from east, counter-clockwise; JSBSim heading
psi is from north, clockwise: psi = 90 deg - chi.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import time

import numpy as np


# ==========================================================================
# USER SETTINGS
# ==========================================================================

# ---- files ---------------------------------------------------------------
PARAMS_FILE        = "c172_params.json"            # written by `calibrate`
ENV_FILE           = "c172_envs.json"
SCP_RESULTS_FILE   = "scp_aircraft_results.json"
TRACK_RESULTS_FILE = "c172_tracking.json"

# ---- simulation ------------------------------------------------------------
AIRCRAFT           = "c172x"
ORIGIN_LAT_DEG     = 37.0
ORIGIN_LON_DEG     = -120.0
ALT_BASE_M         = 300.0             # altitude (MSL) of the workspace floor z = 0
EXTRA_TIME_S       = 0.0               # keep flying this long after the plan ends

# ---- calibration: measured ---------------------------------------------------
CAL_ALT_M          = ALT_BASE_M + 300  # altitude for the trim sweeps (mid workspace)
CAL_SPEEDS_KTS     = range(45, 126, 5)
CAL_POLAR_FIT_KTS  = (60, 125)         # fit the drag polar over this range
CAL_THRUST_KTS     = range(60, 121, 10)
CAL_THRUST_SETTLE_S = 5.0

# ---- calibration: planning choices written into the parameter file ----------
CL_MAX_FACTOR      = 0.85              # plan with this share of the measured max C_L
V_MIN_FACTOR       = 1.3               # V_MIN = factor * stall speed
V_CRUISE_KTS       = 95.0
V_MAX_KTS          = 115.0
PLAN_BANK_MAX_DEG  = 30.0              # autopilot may use up to BANK_CMD_MAX_DEG
PLAN_N_MAX         = 1.5
PLAN_GAMMA_MAX_DEG = 6.0
PLAN_THRUST_FRACTION = 0.8             # leave 20% thrust for the autopilot
PLAN_N_H_RATE_MAX  = 0.12              # 1/s, about 10 deg/s of roll at cruise
PLAN_N_V_RATE_MAX  = 0.15              # 1/s
PLAN_W_TIME        = 0.1
PLAN_W_SPEED       = 100.0             # keep speed near cruise
PLAN_W_SMOOTH      = 5.0
PLAN_N_NODES       = 60
PLAN_RK4_STEPS     = 8

# ---- guidance gains --------------------------------------------------------
GUIDANCE_DT        = 0.1               # s
K_ALONG            = 0.05              # (m/s) per m of along-track error
MAX_DV_CMD         = 5.0               # m/s
K_CROSS            = 0.01              # 1/m; course correction = atan(K_CROSS * e)
MAX_COURSE_CORR_DEG = 30.0
K_COURSE           = 1.2               # bank per course error (rad/rad)
BANK_CMD_MAX_DEG   = 45.0
K_VERT             = 0.01              # 1/m; flight-path correction = atan(K_VERT * e)
MAX_GAMMA_CORR_DEG = 5.0

# ---- inner-loop gains --------------------------------------------------------
K_PHI, K_P         = 2.0, 0.5          # aileron per rad bank error, per rad/s roll rate
K_THETA, K_Q, K_THETA_I = 4.0, 1.5, 0.5
K_V, K_V_I         = 0.08, 0.02        # throttle per m/s airspeed error (and integral)
K_BETA             = 3.0               # rudder per rad sideslip

# ---- evaluation ------------------------------------------------------------
PLOT_INDEX         = None

# ---- FlightGear visual (the `fgview` command) --------------------------------
FG_HOST            = "127.0.0.1"       # machine running FlightGear
FG_PORT            = 5550              # UDP port FlightGear listens on
FG_RATE_HZ         = 60                # packets per second sent to FlightGear
FG_AIRCRAFT        = "c172p"           # FlightGear's Cessna 172 (visual model only)
FG_EXE             = r"C:\Program Files\FlightGear 2024.1\bin\fgfs.exe"   # for the .bat file
FG_SCENERY_DIR     = "fg_scenery"      # obstacle scenery is written here (one subfolder per environment)
FG_SHOW_OBSTACLES  = True
FG_CYLINDER_EXTEND_DOWN_M = 400.0      # extend pillars below the workspace floor to reach the terrain
FG_OBSTACLE_SEGMENTS = 24              # mesh resolution of curved obstacles
FG_OBSTACLE_ALPHA  = 0.0               # transparency 0 (opaque) .. 1

FT, KT, LBF = 0.3048, 0.514444, 4.448222
R_EARTH = 6371000.0


# --------------------------------------------------------------------------
# JSBSim helpers
# --------------------------------------------------------------------------
def make_fdm():
    import jsbsim
    fdm = jsbsim.FGFDMExec(None)
    fdm.set_debug_level(0)
    with contextlib.redirect_stdout(io.StringIO()):
        fdm.load_model(AIRCRAFT)
    return fdm


def init_state(fdm, pos_enu, v_ms, chi_rad, gamma_rad=0.0):
    """Set initial conditions from a local ENU state and trim for steady flight."""
    lat, lon = enu_to_geo(pos_enu[0], pos_enu[1])
    fdm["ic/lat-geod-deg"] = lat
    fdm["ic/long-gc-deg"] = lon
    fdm["ic/h-sl-ft"] = (pos_enu[2] + ALT_BASE_M) / FT
    fdm["ic/vt-kts"] = v_ms / KT
    fdm["ic/psi-true-deg"] = np.degrees(np.pi / 2 - chi_rad) % 360
    fdm["ic/gamma-deg"] = np.degrees(gamma_rad)
    fdm.run_ic()
    fdm["propulsion/set-running"] = -1
    fdm["fcs/mixture-cmd-norm"] = 1.0
    fdm["propulsion/magneto_cmd"] = 3          # both magnetos on, so the engine reports "running"
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            fdm["simulation/do_simple_trim"] = 1
        ok = True
    except Exception:
        ok = False
    fdm["propulsion/magneto_cmd"] = 3
    return ok


def enu_to_geo(x, y):
    lat0 = np.radians(ORIGIN_LAT_DEG)
    return (ORIGIN_LAT_DEG + np.degrees(y / R_EARTH),
            ORIGIN_LON_DEG + np.degrees(x / (R_EARTH * np.cos(lat0))))


def read_state(fdm):
    """Aircraft state in the planner's local frame and units."""
    lat0 = np.radians(ORIGIN_LAT_DEG)
    x = np.radians(fdm["position/long-gc-deg"] - ORIGIN_LON_DEG) * R_EARTH * np.cos(lat0)
    y = np.radians(fdm["position/lat-geod-deg"] - ORIGIN_LAT_DEG) * R_EARTH
    z = fdm["position/h-sl-ft"] * FT - ALT_BASE_M
    ve, vn, vd = (fdm["velocities/v-east-fps"] * FT, fdm["velocities/v-north-fps"] * FT,
                  fdm["velocities/v-down-fps"] * FT)
    return {"p": np.array([x, y, z]),
            "V": fdm["velocities/vt-fps"] * FT,
            "chi": np.arctan2(vn, ve),                          # course, from east CCW
            "gamma": np.arctan2(-vd, np.hypot(ve, vn)),
            "phi": fdm["attitude/phi-rad"], "theta": fdm["attitude/theta-rad"],
            "p_rate": fdm["velocities/p-rad_sec"], "q_rate": fdm["velocities/q-rad_sec"],
            "psi": fdm["attitude/psi-rad"],
            "alpha": fdm["aero/alpha-rad"], "beta": fdm["aero/beta-rad"],
            "thrust": fdm["propulsion/engine/thrust-lbs"] * LBF,
            "lat": fdm["position/lat-geod-deg"], "lon": fdm["position/long-gc-deg"],
            "alt_msl": fdm["position/h-sl-ft"] * FT}


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# --------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------
def calibrate(out_file=PARAMS_FILE):
    rows = []
    for v_kts in CAL_SPEEDS_KTS:
        fdm = make_fdm()
        fdm["ic/h-sl-ft"] = CAL_ALT_M / FT
        if not init_state(fdm, np.array([0, 0, CAL_ALT_M - ALT_BASE_M]), v_kts * KT, np.pi / 2):
            continue
        q, S = fdm["aero/qbar-psf"], fdm["metrics/Sw-sqft"]
        rows.append({"v_kts": v_kts, "CL": fdm["forces/fwz-aero-lbs"] / (q * S),
                     "CD": fdm["forces/fwx-aero-lbs"] / (q * S)})
    mass = fdm["inertia/weight-lbs"] * LBF / 9.80665
    S = fdm["metrics/Sw-sqft"] * FT**2
    rho = fdm["atmosphere/rho-slugs_ft3"] * 515.379
    span = fdm["metrics/bw-ft"] * FT

    fit = [r for r in rows if CAL_POLAR_FIT_KTS[0] <= r["v_kts"] <= CAL_POLAR_FIT_KTS[1]]
    CL2 = np.array([r["CL"] ** 2 for r in fit])
    CD = np.array([r["CD"] for r in fit])
    K, CD0 = np.polyfit(CL2, CD, 1)
    CL_max_meas = max(r["CL"] for r in rows)
    CL_max = CL_MAX_FACTOR * CL_max_meas
    v_stall = np.sqrt(2 * mass * 9.81 / (rho * S * CL_max))

    # full-throttle thrust, allowed to settle, vs airspeed
    Vs, Ts = [], []
    for v_kts in CAL_THRUST_KTS:
        fdm = make_fdm()
        init_state(fdm, np.array([0, 0, CAL_ALT_M - ALT_BASE_M]), v_kts * KT, np.pi / 2)
        fdm["fcs/throttle-cmd-norm"] = 1.0
        for _ in range(int(CAL_THRUST_SETTLE_S / fdm.get_delta_t())):
            fdm.run()
        Vs.append(fdm["velocities/vt-fps"] * FT)
        Ts.append(fdm["propulsion/engine/thrust-lbs"] * LBF)
    slope, T0 = np.polyfit(Vs, Ts, 1)

    params = {
        "name": f"JSBSim {AIRCRAFT} (calibrated)",
        "MASS": mass, "WING_AREA": S, "RHO": rho, "C_D0": CD0, "K_INDUCED": K,
        "C_L_MAX": CL_max, "T_MAX": T0, "THRUST_SLOPE": slope,
        "THRUST_PLAN_FRACTION": PLAN_THRUST_FRACTION,
        "V_MIN": V_MIN_FACTOR * v_stall, "V_MAX": V_MAX_KTS * KT, "V_CRUISE": V_CRUISE_KTS * KT,
        "N_MAX": PLAN_N_MAX, "BANK_MAX_DEG": PLAN_BANK_MAX_DEG,
        "GAMMA_MAX_DEG": PLAN_GAMMA_MAX_DEG,
        "N_H_RATE_MAX": PLAN_N_H_RATE_MAX, "N_V_RATE_MAX": PLAN_N_V_RATE_MAX,
        "W_TIME": PLAN_W_TIME, "W_SPEED": PLAN_W_SPEED, "W_SMOOTH": PLAN_W_SMOOTH,
        "N_NODES": PLAN_N_NODES, "RK4_STEPS": PLAN_RK4_STEPS,
        "_measured": {"wingspan_m": span, "CL_max_trimmed": CL_max_meas,
                      "stall_speed_ms": float(v_stall), "trim_sweep": rows,
                      "thrust_sweep": {"V_ms": Vs, "T_N": Ts}},
    }
    with open(out_file, "w") as f:
        json.dump(params, f, indent=2)
    R = params["V_CRUISE"] ** 2 / (9.81 * np.tan(np.radians(PLAN_BANK_MAX_DEG)))
    print(f"Wrote {out_file}")
    print(f"  mass {mass:.0f} kg, wing area {S:.2f} m^2, span {span:.1f} m, rho {rho:.3f}")
    print(f"  drag polar C_D = {CD0:.4f} + {K:.4f} C_L^2   (fit {CAL_POLAR_FIT_KTS} kts)")
    print(f"  C_L max measured {CL_max_meas:.2f}, planning {CL_max:.2f} -> stall {v_stall:.1f} m/s, "
          f"V_MIN {params['V_MIN']:.1f} m/s")
    print(f"  full-throttle thrust {T0:.0f} + {slope:.2f} V  N   (x{PLAN_THRUST_FRACTION} for planning)")
    print(f"  turn radius at cruise, {PLAN_BANK_MAX_DEG:.0f} deg bank: {R:.0f} m")
    return params


# --------------------------------------------------------------------------
# reference trajectory from an SCP result
# --------------------------------------------------------------------------
class Reference:
    """Dense, time-indexed reference from an scp_aircraft result."""

    def __init__(self, res, steps=20):
        from solvers.scp_aircraft import AircraftSCP
        solver = AircraftSCP()
        X, U, tf = np.array(res["x"]), np.array(res["u"]), res["flight_time"]
        dense = solver.propagate(X, U, tf, steps, sens=False)
        xs = np.vstack([dense["x_sub"].reshape(-1, 6), X[-1]])
        K = len(X) - 1
        s = np.r_[np.repeat(np.arange(K), steps) + np.tile(np.arange(steps) / steps, K), K]
        self.t = s / K * tf
        self.x = xs
        self.u = np.stack([np.interp(s, np.arange(K + 1), U[:, j]) for j in range(3)], axis=1)
        self.x[:, 5] = np.unwrap(self.x[:, 5])
        self.tf = tf

    def at(self, t):
        t = min(max(t, 0.0), self.tf)
        x = np.array([np.interp(t, self.t, self.x[:, j]) for j in range(6)])
        u = np.array([np.interp(t, self.t, self.u[:, j]) for j in range(3)])
        return x, u


# --------------------------------------------------------------------------
# tracking
# --------------------------------------------------------------------------
def fly(env, res, fg=False, speed=None):
    """Fly one planned trajectory in JSBSim. Returns a log dict.
    fg:    also stream the aircraft state to FlightGear over UDP.
    speed: pace the simulation against the wall clock (1.0 = real time);
           None runs as fast as possible."""
    ref = Reference(res)
    fdm = make_fdm()
    if fg:
        import os
        import tempfile
        xml = os.path.join(tempfile.gettempdir(), "jsbsim_fg_output.xml")
        with open(xml, "w") as f:
            f.write(f'<?xml version="1.0"?>\n<output name="{FG_HOST}" type="FLIGHTGEAR" '
                    f'port="{FG_PORT}" protocol="UDP" rate="{FG_RATE_HZ}"/>\n')
        fdm.set_output_directive(xml)
    dt = fdm.get_delta_t()
    wall0 = time.perf_counter()
    x0, _ = ref.at(0.0)
    if not init_state(fdm, x0[:3], x0[3], x0[5], x0[4]):
        return {"status": "trim_failed"}
    trim_ail, trim_rud = fdm["fcs/aileron-cmd-norm"], fdm["fcs/rudder-cmd-norm"]
    thr_trim = fdm["fcs/throttle-cmd-norm"]
    T_trim = fdm["propulsion/engine/thrust-lbs"] * LBF

    cmd = {"V": x0[3], "phi": 0.0, "gamma": 0.0}
    int_theta, int_v = 0.0, 0.0
    n_guid = max(int(round(GUIDANCE_DT / dt)), 1)
    log = {k: [] for k in ("t", "p", "p_ref", "V", "V_ref", "V_cmd", "phi", "phi_cmd", "theta", "psi",
                           "chi", "gamma", "gamma_ref", "gamma_cmd", "alpha", "beta", "thrust",
                           "lat", "lon", "alt_msl", "thr", "ail", "elev", "rud")}
    t, step = 0.0, 0
    t_end = ref.tf + EXTRA_TIME_S
    while t <= t_end:
        s = read_state(fdm)
        xr, ur = ref.at(t)
        if step % n_guid == 0:
            e = xr[:3] - s["p"]
            chi_r = xr[5]
            t_hat = np.array([np.cos(chi_r), np.sin(chi_r)])
            n_hat = np.array([-np.sin(chi_r), np.cos(chi_r)])
            e_along, e_cross = e[:2] @ t_hat, e[:2] @ n_hat
            cmd["V"] = xr[3] + np.clip(K_ALONG * e_along, -MAX_DV_CMD, MAX_DV_CMD)
            chi_cmd = chi_r + np.clip(np.arctan(K_CROSS * e_cross), *np.radians([-1, 1]) * MAX_COURSE_CORR_DEG)
            # planner: positive n_h turns left (chi increases); JSBSim: positive bank turns right
            mu_ff = np.arctan2(ur[2], ur[1])
            cmd["phi"] = -np.clip(mu_ff + K_COURSE * wrap(chi_cmd - s["chi"]),
                                  *np.radians([-1, 1]) * BANK_CMD_MAX_DEG)
            cmd["gamma"] = xr[4] + np.clip(np.arctan(K_VERT * e[2]),
                                           *np.radians([-1, 1]) * MAX_GAMMA_CORR_DEG)

        # inner loops
        ail = trim_ail + K_PHI * (cmd["phi"] - s["phi"]) - K_P * s["p_rate"]
        theta_cmd = cmd["gamma"] + s["alpha"] + int_theta
        int_theta += K_THETA_I * (cmd["gamma"] - s["gamma"]) * dt
        # positive elevator command pitches the nose DOWN in JSBSim
        elev = -(K_THETA * (theta_cmd - s["theta"]) - K_Q * s["q_rate"])
        thr_ff = thr_trim * ur[0] / max(T_trim, 1.0)
        thr = thr_ff + K_V * (cmd["V"] - s["V"]) + int_v
        int_v = np.clip(int_v + K_V_I * (cmd["V"] - s["V"]) * dt, -0.5, 0.5)
        rud = trim_rud - K_BETA * s["beta"]

        ail, elev = float(np.clip(ail, -1, 1)), float(np.clip(elev, -1, 1))
        thr, rud = float(np.clip(thr, 0, 1)), float(np.clip(rud, -1, 1))
        fdm["fcs/aileron-cmd-norm"] = ail
        fdm["fcs/elevator-cmd-norm"] = elev
        fdm["fcs/throttle-cmd-norm"] = thr
        fdm["fcs/rudder-cmd-norm"] = rud

        if step % 12 == 0:                                  # log at 10 Hz (commands as sent)
            for k, v in (("t", t), ("p", s["p"]), ("p_ref", xr[:3]), ("V", s["V"]), ("V_ref", xr[3]),
                         ("V_cmd", cmd["V"]), ("phi", s["phi"]), ("phi_cmd", cmd["phi"]),
                         ("theta", s["theta"]), ("psi", s["psi"]), ("chi", s["chi"]),
                         ("gamma", s["gamma"]), ("gamma_ref", xr[4]), ("gamma_cmd", cmd["gamma"]),
                         ("alpha", s["alpha"]), ("beta", s["beta"]), ("thrust", s["thrust"]),
                         ("lat", s["lat"]), ("lon", s["lon"]), ("alt_msl", s["alt_msl"]),
                         ("thr", thr), ("ail", ail), ("elev", elev), ("rud", rud)):
                log[k].append(v)
        fdm.run()
        t += dt
        step += 1
        if speed:                                           # real-time pacing
            ahead = t / speed - (time.perf_counter() - wall0)
            if ahead > 0:
                time.sleep(ahead)
        if s["p"][2] < -ALT_BASE_M + 5:                    # hit the real ground
            break
    return {k: np.array(v) for k, v in log.items()}


def evaluate(env, log, t_plan):
    """Tracking error and obstacle clearance of the flown path."""
    during = log["t"] <= t_plan
    err = np.linalg.norm(log["p"] - log["p_ref"], axis=1)[during]
    P = log["p"][during]
    # clearance to collision (vehicle radius only), and to the planning envelope (radius + margin)
    h_body = env.constraint_values(P, d=env.vehicle_radius)
    A, _ = env.linearized_constraints(P)
    gn = np.maximum(np.linalg.norm(A, axis=-1), 1e-9).T
    clear_body = float((h_body / gn).min())
    lo, hi = env.bounds_lo, env.bounds_hi
    out_of_box = float(np.max(np.r_[lo - P.min(0), P.max(0) - hi, 0.0]))
    sat = lambda a, lo_, hi_: float(np.mean((a <= lo_) | (a >= hi_)))
    return {"max_error_m": float(err.max()), "rms_error_m": float(np.sqrt(np.mean(err**2))),
            "final_error_m": float(err[-1]),
            "min_clearance_m": clear_body,                 # >= 0: no collision
            "planned_clearance_m": env.margin,             # what the plan kept (radius + margin)
            "left_workspace_m": out_of_box,
            "collision_free": bool(clear_body >= 0 and out_of_box <= 0),
            "throttle_saturated_frac": sat(log["thr"][during], 0, 1),
            "aileron_saturated_frac": sat(log["ail"][during], -1, 1),
            "elevator_saturated_frac": sat(log["elev"][during], -1, 1)}


# --------------------------------------------------------------------------
# plotting
# --------------------------------------------------------------------------
def plot_tracking(env, log, title=""):
    import matplotlib.pyplot as plt
    from benchmarks.envgen import plot_environment

    ax = plot_environment(env)
    ax.plot(*log["p_ref"].T, "k--", lw=1.5, label="plan (SCP)")
    ax.plot(*log["p"].T, "m-", lw=1.5, label="flown (JSBSim)")
    ax.legend(loc="upper left")
    ax.set_title(title)

    t = log["t"]
    fig, axs = plt.subplots(2, 2, figsize=(11, 6), sharex=True)
    axs[0, 0].plot(t, np.linalg.norm(log["p"] - log["p_ref"], axis=1))
    axs[0, 0].set_ylabel("position error [m]")
    axs[0, 1].plot(t, log["V_ref"], "k--", t, log["V"], "m-"); axs[0, 1].set_ylabel("airspeed [m/s]")
    axs[1, 0].plot(t, np.degrees(log["phi_cmd"]), "k--", t, np.degrees(log["phi"]), "m-")
    axs[1, 0].set_ylabel("bank [deg]"); axs[1, 0].set_xlabel("t [s]")
    axs[1, 1].plot(t, np.degrees(log["gamma_ref"]), "k--", t, np.degrees(log["gamma"]), "m-")
    axs[1, 1].set_ylabel("flight-path angle [deg]"); axs[1, 1].set_xlabel("t [s]")
    for a in axs.flat:
        a.grid(alpha=0.3)
    fig.suptitle("dashed: plan / command, solid: JSBSim")
    fig.tight_layout()
    plt.show()


# --------------------------------------------------------------------------
# export: commands CSV + environment
# --------------------------------------------------------------------------
def export_flight(env, res, log, out_dir):
    """Write the JSBSim command/state log, the environment, and the SCP plan."""
    import csv
    import os
    os.makedirs(out_dir, exist_ok=True)
    deg = np.degrees
    cols = [("t_s", log["t"]),
            ("aileron_cmd_norm", log["ail"]), ("elevator_cmd_norm", log["elev"]),
            ("throttle_cmd_norm", log["thr"]), ("rudder_cmd_norm", log["rud"]),
            ("x_east_m", log["p"][:, 0]), ("y_north_m", log["p"][:, 1]), ("z_up_m", log["p"][:, 2]),
            ("lat_deg", log["lat"]), ("lon_deg", log["lon"]), ("alt_msl_m", log["alt_msl"]),
            ("airspeed_ms", log["V"]), ("roll_deg", deg(log["phi"])), ("pitch_deg", deg(log["theta"])),
            ("heading_deg", deg(log["psi"]) % 360), ("flight_path_deg", deg(log["gamma"])),
            ("alpha_deg", deg(log["alpha"])), ("beta_deg", deg(log["beta"])), ("thrust_N", log["thrust"]),
            ("ref_x_m", log["p_ref"][:, 0]), ("ref_y_m", log["p_ref"][:, 1]), ("ref_z_m", log["p_ref"][:, 2]),
            ("ref_airspeed_ms", log["V_ref"]), ("ref_flight_path_deg", deg(log["gamma_ref"])),
            ("cmd_airspeed_ms", log["V_cmd"]), ("cmd_roll_deg", deg(log["phi_cmd"])),
            ("cmd_flight_path_deg", deg(log["gamma_cmd"]))]
    with open(os.path.join(out_dir, "jsbsim_commands.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([c for c, _ in cols])
        for row in zip(*[v for _, v in cols]):
            w.writerow([f"{v:.8g}" for v in row])
    env_d = env.to_dict()
    env_d["frame"] = {"type": "local ENU (x east, y north, z up), metres",
                      "origin_lat_deg": ORIGIN_LAT_DEG, "origin_lon_deg": ORIGIN_LON_DEG,
                      "z0_alt_msl_m": ALT_BASE_M}
    with open(os.path.join(out_dir, "environment.json"), "w") as f:
        json.dump(env_d, f, indent=1)
    with open(os.path.join(out_dir, "constraints.txt"), "w") as f:
        f.write(env.describe() + "\n")
    with open(os.path.join(out_dir, "scp_plan.json"), "w") as f:
        json.dump(res, f)


# --------------------------------------------------------------------------
# obstacle mesh export (Wavefront OBJ)
# --------------------------------------------------------------------------
MESH_SEGMENTS = 32          # resolution of curved obstacle surfaces


def _box_mesh(lo, hi):
    V = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    F = [[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1], [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]]   # outward
    return V, F


def obstacle_meshes(env, segments=MESH_SEGMENTS, floor=None):
    """(name, vertices, faces) for every obstacle, in the local ENU frame, metres."""
    from benchmarks.envgen import clip_polytope
    floor = env.bounds_lo[2] if floor is None else floor
    out = []
    for i, o in enumerate(env.obstacles):
        if o.kind == "spheroid":
            nu, nv = segments, segments // 2
            V = [np.array([0, 0, -1.0])]
            for j in range(1, nv):
                th = np.pi * j / nv - np.pi / 2
                for k in range(nu):
                    ph = 2 * np.pi * k / nu
                    V.append([np.cos(th) * np.cos(ph), np.cos(th) * np.sin(ph), np.sin(th)])
            V.append([0, 0, 1.0])
            V = (np.array(V) * o.radii) @ o.rotation.T + o.center
            ring = lambda j, k: 1 + (j - 1) * nu + k % nu
            F = [[0, ring(1, k + 1), ring(1, k)] for k in range(nu)]
            for j in range(1, nv - 1):
                F += [[ring(j, k), ring(j, k + 1), ring(j + 1, k + 1), ring(j + 1, k)] for k in range(nu)]
            top = len(V) - 1
            F += [[top, ring(nv - 1, k), ring(nv - 1, k + 1)] for k in range(nu)]
            out.append((f"spheroid_{i}", V, F))
        elif o.kind == "oriented_cylinder":
            R = o.rotation                                     # columns: rotated x, y, axis
            ang = 2 * np.pi * np.arange(segments) / segments
            rim = o.radius * (np.cos(ang)[:, None] * R[:, 0] + np.sin(ang)[:, None] * R[:, 1])
            bottom, top = o.endpoints()
            V = np.vstack([bottom + rim, top + rim])
            n = segments
            F = [[k, (k + 1) % n, n + (k + 1) % n, n + k] for k in range(n)]
            F += [list(range(n - 1, -1, -1)), list(range(n, 2 * n))]          # end caps
            out.append((f"ocylinder_{i}", V, F))
        elif o.kind == "cylinder":
            ang = 2 * np.pi * np.arange(segments) / segments
            ring = np.c_[o.center_xy[0] + o.radius * np.cos(ang), o.center_xy[1] + o.radius * np.sin(ang)]
            V = np.vstack([np.c_[ring, np.full(segments, floor)], np.c_[ring, np.full(segments, o.z_top)]])
            n = segments
            F = [[k, (k + 1) % n, n + (k + 1) % n, n + k] for k in range(n)]
            F += [list(range(n - 1, -1, -1)), list(range(n, 2 * n))]          # bottom, top caps
            out.append((f"cylinder_{i}", V, F))
        elif o.kind in ("box", "wall"):
            # boxes may be rotated and wall pieces reach far past the workspace, so
            # mesh only the part inside it (down to `floor`, so walls meet the terrain)
            lo = np.r_[env.bounds_lo[:2], min(floor, env.bounds_lo[2])]
            parts = [(f"box_{i}", o)] if o.kind == "box" else \
                    [(f"wall_{i}_{j}", bx) for j, bx in enumerate(o.boxes)]
            for name, bx in parts:
                mesh = clip_polytope(*bx.halfspaces(), lo, env.bounds_hi)
                if mesh is not None:
                    out.append((name, mesh[0], mesh[1]))
    return out


def export_obj(env, path, segments=MESH_SEGMENTS):
    """Obstacles as a Wavefront OBJ (one named object each) plus the workspace
    box as an edge-only object. Units metres, local ENU, Z up."""
    lines = ["# obstacles exported by jsbsim_c172.py",
             "# units: metres; frame: local ENU (x east, y north, z up)",
             f"# origin: lat {ORIGIN_LAT_DEG} lon {ORIGIN_LON_DEG}, z = 0 at {ALT_BASE_M} m MSL"]
    base = 1
    for name, V, F in obstacle_meshes(env, segments):
        lines.append(f"o {name}")
        lines += [f"v {x:.3f} {y:.3f} {z:.3f}" for x, y, z in V]
        lines += ["f " + " ".join(str(base + i) for i in f) for f in F]
        base += len(V)
    V, _ = _box_mesh(env.bounds_lo, env.bounds_hi)
    lines.append("o workspace_bounds")
    lines += [f"v {x:.3f} {y:.3f} {z:.3f}" for x, y, z in V]
    for a, b in [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]:
        lines.append(f"l {base + a} {base + b}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def export_plan_csv(res, path, steps=10):
    """The SCP plan, densely sampled with the planner's own dynamics."""
    import csv
    ref = Reference(res, steps=steps)
    bank = np.degrees(np.arctan2(ref.u[:, 2], ref.u[:, 1]))
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "x_east_m", "y_north_m", "z_up_m", "airspeed_ms", "flight_path_deg",
                    "course_deg_from_east", "bank_deg_right_positive", "thrust_N", "load_factor"])
        for k in range(len(ref.t)):
            x, u = ref.x[k], ref.u[k]
            w.writerow([f"{v:.6g}" for v in (ref.t[k], *x[:4], np.degrees(x[4]), np.degrees(x[5]),
                                              -bank[k], u[0], np.hypot(u[1], u[2]))])


def _sg_bucket(lon, lat):
    """FlightGear scenery tile (SimGear bucket): directory path and tile index."""
    def span(l):
        for lim, s in ((89, 12), (86, 4), (83, 2), (76, 1), (62, 0.5), (22, 0.25), (-22, 0.125),
                       (-62, 0.25), (-76, 0.5), (-83, 1), (-86, 2), (-89, 4)):
            if l >= lim:
                return s
        return 12
    ilon, ilat = int(np.floor(lon)), int(np.floor(lat))
    x = int((lon - ilon) / span(lat))
    y = int((lat - ilat) * 8)
    index = ((ilon + 180) << 14) + ((ilat + 90) << 6) + (y << 3) + x

    def top(v):                                    # C-style truncation, then floor to 10 degrees
        t = int(v / 10)
        if v < 0 and t * 10 != v:
            t -= 1
        return t * 10
    tlon, tlat = top(ilon), top(ilat)
    fmt = lambda lo, la: (f"{'e' if lo >= 0 else 'w'}{abs(lo):03d}"
                          f"{'n' if la >= 0 else 's'}{abs(la):02d}")
    return f"{fmt(tlon, tlat)}/{fmt(ilon, ilat)}", index


def _write_ac(path, name, V_local, faces, rgb):
    """AC3D model. V_local is in local ENU metres relative to the model origin;
    FlightGear shows AC3D's +y as up, so (east, north, up) -> (-north, up, -east)."""
    V = np.column_stack([-V_local[:, 1], V_local[:, 2], -V_local[:, 0]])
    r, g, b = rgb
    lines = ["AC3Db",
             f'MATERIAL "{name}" rgb {r} {g} {b}  amb {r * .5:.2f} {g * .5:.2f} {b * .5:.2f}  '
             f"emis 0 0 0  spec 0.1 0.1 0.1  shi 16  trans {FG_OBSTACLE_ALPHA}",
             "OBJECT world", "kids 1", "OBJECT poly", f'name "{name}"', f"numvert {len(V)}"]
    lines += [f"{x:.3f} {y:.3f} {z:.3f}" for x, y, z in V]
    lines.append(f"numsurf {len(faces)}")
    for f in faces:
        lines += ["SURF 0x30", "mat 0", f"refs {len(f)}"] + [f"{i} 0 0" for i in f]
    lines.append("kids 0")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def export_flightgear_scenery(env, out_dir):
    """Write every obstacle as a FlightGear scenery object (AC3D model placed
    by a .stg file in the right scenery tile). Returns the scenery folder."""
    import os
    import shutil
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    colours = {"spheroid": (1.0, 0.55, 0.1), "cylinder": (0.2, 0.45, 0.85),
               "ocylinder": (0.55, 0.3, 0.8),
               "box": (0.35, 0.65, 0.35), "wall": (0.55, 0.55, 0.55)}
    floor = env.bounds_lo[2] - FG_CYLINDER_EXTEND_DOWN_M
    stg = {}
    for name, V, F in obstacle_meshes(env, FG_OBSTACLE_SEGMENTS, floor=floor):
        kind = name.split("_")[0]
        if kind == "cylinder":                                  # origin at the pillar's base
            origin = np.r_[V[:, :2].mean(axis=0), floor]
        else:
            origin = V.mean(axis=0)
        lat, lon = enu_to_geo(origin[0], origin[1])
        tile_dir, index = _sg_bucket(lon, lat)
        d = os.path.join(out_dir, "Objects", tile_dir)
        os.makedirs(d, exist_ok=True)
        _write_ac(os.path.join(d, f"{name}.ac"), name, V - origin, F, colours[kind])
        stg.setdefault((d, index), []).append(
            f"OBJECT_STATIC {name}.ac {lon:.8f} {lat:.8f} {origin[2] + ALT_BASE_M:.2f} 0")
    for (d, index), lines in stg.items():
        with open(os.path.join(d, f"{index}.stg"), "w") as fh:
            fh.write("\n".join(lines) + "\n")
    return os.path.abspath(out_dir)


def _load_checked(envs_file, results_file, index):
    """Load an environment and its SCP plan, verifying they belong together and
    that the plan itself is collision-free before anything is flown."""
    from solvers.scp_aircraft import load_plan, plan_clearance
    env, r, res = load_plan(envs_file, results_file, index)
    if not r["success"]:
        raise SystemExit(f"Environment {index} has no feasible SCP plan in {results_file} "
                         f"(status: {r['status']}).")
    c = plan_clearance(env, res)
    print(f"Plan for environment {index}: closest approach to an obstacle {c:.1f} m"
          + ("" if c >= 0 else "  <-- WARNING: the plan itself intersects an obstacle"))
    return env, r


def flightgear_args(env, res):
    """Command-line options that start FlightGear as a display for JSBSim."""
    x = np.array(res["x"])[0]
    lat, lon = enu_to_geo(x[0], x[1])
    heading = np.degrees(np.pi / 2 - x[5]) % 360
    return [f"--aircraft={FG_AIRCRAFT}", "--fdm=null",
            f"--native-fdm=socket,in,{FG_RATE_HZ},,{FG_PORT},udp",
            f"--lat={lat:.6f}", f"--lon={lon:.6f}",
            f"--altitude={(x[2] + ALT_BASE_M) / FT:.0f}", f"--heading={heading:.1f}",
            "--in-air", "--timeofday=noon", "--disable-real-weather-fetch",
            "--disable-ai-traffic", "--enable-terrasync", "--prop:/sim/freeze/fuel=true"]


def fgview(args):
    """Fly one environment in JSBSim, in real time, displayed in FlightGear."""
    from benchmarks.envgen import load_environments
    env, r = _load_checked(args.envs, args.results, args.index)
    envs = {args.index: env}
    fg_args = flightgear_args(envs[args.index], r)
    if FG_SHOW_OBSTACLES:
        import os
        scen = export_flightgear_scenery(envs[args.index],
                                         os.path.join(FG_SCENERY_DIR, f"env{args.index}"))
        fg_args.insert(0, f'--fg-scenery="{scen}"')
        print(f"Wrote {len(envs[args.index].obstacles)} obstacles as FlightGear scenery in {scen}\n")
    bat = f"start_flightgear_env{args.index}.bat"
    with open(bat, "w") as f:
        f.write(f'"{FG_EXE}" ' + " ".join(fg_args) + "\n")
    print("1. Start FlightGear with these options, either by running", bat)
    print("   (check FG_EXE points at your fgfs.exe), or by pasting them into the")
    print("   launcher's Settings > Additional Settings box:\n")
    print("   " + " ".join(fg_args) + "\n")
    print("2. Wait until FlightGear has finished loading and shows the aircraft.")
    if not args.no_wait:
        input("3. Press Enter here to start the flight ... ")
    print(f"Flying environment {args.index} at {args.speed}x real time "
          f"({r['flight_time'] / args.speed:.0f} s). Ctrl+C to stop.")
    try:
        log = fly(envs[args.index], r, fg=True, speed=args.speed)
    except KeyboardInterrupt:
        print("\nStopped.")
        return
    if "t" in log:
        ev = evaluate(envs[args.index], log, r["flight_time"])
        print(f"Done: error rms {ev['rms_error_m']:.1f} m, max {ev['max_error_m']:.1f} m, "
              f"min clearance {ev['min_clearance_m']:.1f} m")


def record(args):
    """Fly one environment and export everything for external rendering."""
    import os
    from benchmarks.envgen import load_environments
    env, r = _load_checked(args.envs, args.results, args.index)
    envs = {args.index: env}
    out_dir = args.out_dir or f"flight_env{args.index}"
    env = envs[args.index]
    print(f"Flying environment {args.index} in JSBSim ...")
    log = fly(env, r)
    if "t" not in log:
        raise SystemExit(log["status"])
    ev = evaluate(env, log, r["flight_time"])
    export_flight(env, r, log, out_dir)
    with open(os.path.join(out_dir, "tracking_summary.json"), "w") as f:
        json.dump(ev, f, indent=2)
    print(f"  error rms {ev['rms_error_m']:.1f} m, max {ev['max_error_m']:.1f} m, "
          f"min clearance {ev['min_clearance_m']:.1f} m")
    export_obj(env, os.path.join(out_dir, "environment.obj"))
    export_plan_csv(r, os.path.join(out_dir, "scp_plan.csv"))
    print(f"  wrote {out_dir}/")
    for name, what in (("jsbsim_commands.csv", "JSBSim control commands + aircraft state, 10 Hz"),
                       ("scp_plan.csv", "planned trajectory, densely sampled"),
                       ("environment.obj", "obstacle meshes + workspace box (metres, ENU, Z up)"),
                       ("environment.json", "obstacle parameters, start/goal, frame origin"),
                       ("constraints.txt", "the constraint set written out"),
                       ("scp_plan.json", "raw SCP result (nodes, controls)"),
                       ("tracking_summary.json", "tracking error and clearance")):
        print(f"    {name:24s} {what}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="JSBSim Cessna 172: calibrate the planner model "
                                             "or fly SCP trajectories.")
    sub = ap.add_subparsers(dest="command", required=True)
    c = sub.add_parser("calibrate")
    c.add_argument("--out", default=PARAMS_FILE)
    t = sub.add_parser("track")
    t.add_argument("--envs", default=ENV_FILE)
    t.add_argument("--results", default=SCP_RESULTS_FILE)
    t.add_argument("--out", default=TRACK_RESULTS_FILE)
    t.add_argument("--plot", type=int, default=PLOT_INDEX, metavar="INDEX",
                   help="plot this environment index")
    rec = sub.add_parser("record", help="fly one environment and export the JSBSim commands, "
                                        "trajectory and environment for external rendering")
    rec.add_argument("index", type=int, help="environment index")
    rec.add_argument("--envs", default=ENV_FILE)
    rec.add_argument("--results", default=SCP_RESULTS_FILE)
    rec.add_argument("--out-dir", default=None, help="default: flight_env<INDEX>")
    fgv = sub.add_parser("fgview", help="fly one environment in real time, displayed in FlightGear")
    fgv.add_argument("index", type=int, help="environment index")
    fgv.add_argument("--envs", default=ENV_FILE)
    fgv.add_argument("--results", default=SCP_RESULTS_FILE)
    fgv.add_argument("--speed", type=float, default=1.0, help="playback speed (1 = real time)")
    fgv.add_argument("--no-wait", action="store_true", help="start without waiting for Enter")
    args = ap.parse_args()

    if args.command == "fgview":
        fgview(args)
        return

    if args.command == "calibrate":
        calibrate(args.out)
        return

    if args.command == "record":
        record(args)
        return

    from benchmarks.envgen import load_environments
    envs = load_environments(args.envs)
    with open(args.results) as f:
        results = json.load(f)
    out = []
    print(f"Flying {sum(r['success'] for r in results)} feasible SCP trajectories in JSBSim ({AIRCRAFT})")
    for r in results:
        if not r["success"]:
            continue
        i = r["env_index"]
        fp = r.get("env_fingerprint")
        if i >= len(envs) or (fp is not None and fp != envs[i].fingerprint()):
            raise SystemExit(f"MISMATCH: {args.results} was not solved on {args.envs} "
                             f"(it came from {r.get('env_file', '?')}). Use the same --envs file "
                             f"as when solving, or re-run scp_aircraft.py.")
        t0 = time.perf_counter()
        log = fly(envs[i], r)
        if "t" not in log:
            print(f"  env {i:3d}  {log['status']}")
            out.append({"env_index": i, "status": log["status"]})
            continue
        ev = evaluate(envs[i], log, r["flight_time"])
        print(f"  env {i:3d}  flight {r['flight_time']:6.1f}s  "
              f"error rms {ev['rms_error_m']:6.1f} m  max {ev['max_error_m']:6.1f} m  "
              f"clearance {ev['min_clearance_m']:7.1f} m  "
              f"{'collision-free' if ev['collision_free'] else 'COLLISION'}  "
              f"({time.perf_counter() - t0:.1f}s)")
        out.append({"env_index": i, "status": "flown", **ev})
        if args.plot == i:
            plot_tracking(envs[i], log, f"environment {i}")
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    n_ok = sum(o.get("collision_free", False) for o in out)
    print(f"\n{n_ok}/{len(out)} flown collision-free. Results -> {args.out}")


if __name__ == "__main__":
    main()