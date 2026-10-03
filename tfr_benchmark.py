#!/usr/bin/env python3
"""
tfr_benchmark.py - Known-solution test: fly a Cessna 172 around a stadium TFR.

The problem
-----------
The FAA's standing sporting-event Temporary Flight Restriction (FDC NOTAM
4/3621) closes a cylinder of 3 NM radius, from the surface to 3,000 ft AGL,
around major stadiums during events. Here the aircraft must get from A to B,
staying below the workspace ceiling (600 m, under the TFR's 914 m top), with
the stadium right next to the straight line A-B. So it has to fly around.

The known solution
------------------
Seen from above, the restricted area grown by the aircraft's clearance d is a
circle of radius R = 3 NM + d. The shortest path from A to B that stays
outside a circle is pure geometry: a straight line tangent to the circle, an
arc along it, and a second tangent line to B. Its length L* is a hard LOWER
BOUND for any path that avoids the circle, whatever the dynamics:

  * an SCP plan shorter than L* means the obstacle constraints are leaking;
  * a plan slightly longer is expected (turn-radius and heading limits, the
    speed and smoothness terms in the cost, discretization);
  * here R (~5.6 km) is about 13 turn radii of the C172, so the bound is
    nearly tight: the real optimum is longer only by a short turn at the
    start, because the planner starts pointed straight at the goal.

Workflow (run in the project folder)
------------------------------------
    python tfr_benchmark.py make
    python scp_aircraft.py --envs tfr_env.json --out tfr_results.json --feasible-out tfr_feasible.json --verbose
    python tfr_benchmark.py check --plot
  optional, fly it in JSBSim and check the flown path too:
    python jsbsim_c172.py record 0 --envs tfr_env.json --results tfr_results.json --out-dir tfr_flight
    python tfr_benchmark.py check --flight tfr_flight/jsbsim_commands.csv --plot
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from envgen import Cylinder, Environment, load_environments, save_environments

NM = 1852.0                       # metres per nautical mile
FT = 0.3048

# ---- scenario defaults -------------------------------------------------------
TFR_RADIUS_NM     = 3.0           # FDC NOTAM 4/3621: 3 NM radius ...
TFR_TOP_FT_AGL    = 3000.0        # ... surface to 3,000 ft AGL
DISTANCE_KM       = 24.0          # start to goal
GOAL_OFFSET_M     = 100.0         # sideways offset of the goal, so one side of the TFR is
                                  #   strictly shorter (an exactly symmetric problem has two
                                  #   equal optima and puts the straight line through the
                                  #   cylinder's axis, where its gradient is undefined)
ALTITUDE_M        = 300.0         # flight altitude above the workspace floor
CEILING_M         = 600.0         # workspace top; below the TFR top, so going over is not an option
VEHICLE_RADIUS_M  = 6.0           # same clearance as the C172 benchmark sets
MARGIN_M          = 30.0
SIDE_ROOM_M       = 2500.0        # free space beyond the detour, so the workspace walls never bind
END_ROOM_M        = 1000.0        # free space behind the start and past the goal

# ---- pass / fail thresholds on the path-length gap (L - L*) / L* --------------
GAP_PASS          = 0.02
GAP_WARN          = 0.05
LEAK_TOL          = 1e-3          # a plan more than 0.1% shorter than L* is a constraint leak


# --------------------------------------------------------------------------
# geometry: shortest path around a circle
# --------------------------------------------------------------------------
def _cross2(a, b):
    """z-component of the cross product of two planar vectors."""
    return float(a[0] * b[1] - a[1] * b[0])


def shortest_path_around_circle(A, B, C, R, n_arc=200):
    """Shortest planar path from A to B staying outside the disc (C, R).
    Returns a dict with the length, the polyline, and both sides' lengths.
    A and B must be outside the disc."""
    A, B, C = (np.asarray(v, float)[:2] for v in (A, B, C))
    dA, dB = np.linalg.norm(A - C), np.linalg.norm(B - C)
    if min(dA, dB) <= R:
        raise ValueError("start or goal is inside the restricted circle")

    AB = B - A
    s = np.clip((C - A) @ AB / (AB @ AB), 0.0, 1.0)
    if np.linalg.norm(A + s * AB - C) >= R:                     # straight line is already clear
        L = float(np.linalg.norm(AB))
        return {"length": L, "straight": L, "side": "straight", "path": np.stack([A, B]),
                "lengths": {"left": L, "right": L}, "tangent_m": [L, 0.0], "arc_m": 0.0, "arc_deg": 0.0}

    tA, tB = np.sqrt(dA**2 - R**2), np.sqrt(dB**2 - R**2)
    aA, aB = np.arccos(R / dA), np.arccos(R / dB)              # center angle: radius vs. line to point
    angA = np.arctan2(*(A - C)[::-1])
    angB = np.arctan2(*(B - C)[::-1])
    phi = (angB - angA) % (2 * np.pi)                         # counter-clockwise angle C->A to C->B
    out = {}
    for side, wrap, ta, tb, sgn in (("ccw", phi - aA - aB, angA + aA, angB - aB, 1.0),
                                    ("cw", 2 * np.pi - phi - aA - aB, angA - aA, angB + aB, -1.0)):
        th = ta + sgn * np.linspace(0.0, wrap, n_arc)
        arc = C + R * np.c_[np.cos(th), np.sin(th)]
        out[side] = {"length": float(tA + tB + R * wrap), "wrap": float(wrap),
                     "path": np.vstack([A, arc, B])}
    # name each side as seen flying from A to B, from where its path actually goes
    names = {k: side_of(out[k]["path"], A, B, C) for k in out}
    best = min(out, key=lambda k: out[k]["length"])
    return {"length": out[best]["length"], "straight": float(np.linalg.norm(AB)),
            "side": names[best], "path": out[best]["path"],
            "lengths": {names[k]: out[k]["length"] for k in out},
            "tangent_m": [float(tA), float(tB)], "arc_m": float(R * out[best]["wrap"]),
            "arc_deg": float(np.degrees(out[best]["wrap"]))}


def side_of(path_xy, A, B, C):
    """Which side of the circle a path passes: 'left' or 'right' of A->B, judged
    at the path point closest to the circle's center."""
    k = int(np.argmin(np.linalg.norm(path_xy - C, axis=1)))
    A = np.asarray(A, float)[:2]
    return "left" if _cross2(np.asarray(B, float)[:2] - A, path_xy[k] - A) > 0 else "right"


def turn_radius_m(params_file="c172_params.json"):
    """Planning turn radius at cruise (V^2 / (g tan(bank_max))), for context."""
    if not os.path.exists(params_file):
        return None
    p = json.load(open(params_file))
    return p["V_CRUISE"] ** 2 / (9.81 * np.tan(np.radians(p["BANK_MAX_DEG"])))


# --------------------------------------------------------------------------
# make: environment + reference
# --------------------------------------------------------------------------
def make_environment(radius_nm=TFR_RADIUS_NM, distance_km=DISTANCE_KM, offset_m=GOAL_OFFSET_M,
                     altitude_m=ALTITUDE_M, ceiling_m=CEILING_M, vehicle_radius=VEHICLE_RADIUS_M,
                     margin=MARGIN_M):
    rho = radius_nm * NM
    d = vehicle_radius + margin
    R = rho + d
    D = distance_km * 1000.0
    half_w = R + abs(offset_m) + SIDE_ROOM_M
    lo = np.array([0.0, 0.0, 0.0])
    hi = np.array([D + 2 * END_ROOM_M, 2 * half_w, ceiling_m])
    yc = half_w
    start = np.array([END_ROOM_M, yc, altitude_m])
    goal = np.array([END_ROOM_M + D, yc + offset_m, altitude_m])
    center = np.array([END_ROOM_M + D / 2, yc])
    if not 0 < altitude_m < ceiling_m:
        raise ValueError("altitude must be inside the workspace")

    tfr = Cylinder(center, rho, ceiling_m, full_height=True)   # floor to ceiling: must go around
    ref = shortest_path_around_circle(start, goal, center, R)
    if np.any(ref["path"][:, 1] < lo[1] + vehicle_radius) or np.any(ref["path"][:, 1] > hi[1] - vehicle_radius):
        raise ValueError("workspace too narrow for the detour; raise SIDE_ROOM_M")

    env = Environment(lo, hi, [tfr], start, np.zeros(3), goal, np.zeros(3), vehicle_radius, margin)
    reference = {k: v for k, v in ref.items() if k != "path"}
    reference.update({
        "description": f"Stadium TFR detour: {radius_nm:g} NM radius (FDC NOTAM 4/3621), "
                       f"surface to {TFR_TOP_FT_AGL:.0f} ft AGL; flight below {ceiling_m:.0f} m",
        "tfr_radius_m": rho, "inflated_radius_m": R, "clearance_m": d,
        "lower_bound_length_m": ref["length"],
        "note": "Shortest path outside the inflated circle (tangent-arc-tangent). A hard lower "
                "bound on the length of any collision-free path, whatever the dynamics.",
    })
    env.metrics = {"tfr_reference": reference}
    return env, ref


# --------------------------------------------------------------------------
# check: SCP plan (and optionally a JSBSim flight) against the reference
# --------------------------------------------------------------------------
def path_length(P):
    return float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1)))


def scp_path(result, steps=40):
    """Dense positions of an SCP plan, integrated with the planner's own model."""
    from scp_aircraft import AircraftSCP
    solver = AircraftSCP()
    X, U = np.array(result["x"]), np.array(result["u"])
    dense = solver.propagate(X, U, result["flight_time"], steps, sens=False)
    return solver.check_points(X, dense), X


def assess(name, P, env, ref, extra="", is_plan=True):
    """Compare one path (N, 3) to the reference. Returns (verdict, lines).
    For a plan, being shorter than the optimum means the constraints leak. A
    flown path may legitimately cut a little inside the planning clearance, so
    for flights only intrusion into the TFR itself fails."""
    tfr = env.obstacles[0]
    C, rho = tfr.center_xy, tfr.radius
    R = rho + env.clearance
    L, Lstar = path_length(P), ref["length"]
    gap = (L - Lstar) / Lstar
    dist_tfr = float(np.min(np.linalg.norm(P[:, :2] - C, axis=1)) - rho)   # to the TFR boundary
    side = side_of(P[:, :2], env.start_pos[:2], env.goal_pos[:2], C)
    L_side = ref["lengths"][side]
    gap_side = (L - L_side) / L_side

    lines = [f"{name}{extra}",
             f"  path length           {L / 1000:9.3f} km   (optimum {Lstar / 1000:.3f} km, "
             f"gap {100 * gap:+.2f}%)",
             f"  passes on the         {side} side" + ("" if side == ref["side"] else
                                                       f"  (the {ref['side']} side is shorter; "
                                                       f"vs the {side}-side optimum: {100 * gap_side:+.2f}%)"),
             f"  closest to the TFR    {dist_tfr:9.1f} m outside its boundary "
             f"(required: {env.clearance:.0f} m)"]
    if dist_tfr < env.vehicle_radius:
        return "FAIL", lines + ["  FAIL: the aircraft enters the restricted area."]
    plan_tol = 0.25 * env.margin            # scp_aircraft's DENSE_TOLERANCE default
    if is_plan and dist_tfr < env.clearance - plan_tol:
        return "FAIL", lines + [f"  FAIL: the plan cuts {env.clearance - dist_tfr:.1f} m into its own "
                                f"{env.clearance:.0f} m clearance (allowed: {plan_tol:.1f} m). The obstacle "
                                f"constraints are leaking."]
    if is_plan and gap_side < -LEAK_TOL:
        return "FAIL", lines + ["  FAIL: shorter than the geometric optimum, which is impossible for a "
                                "path that respects the clearance. The obstacle constraints are leaking."]
    if dist_tfr < env.clearance - 1.0:
        lines.append(f"  note: inside the {env.clearance:.0f} m planning clearance but clear of the TFR itself.")
    if gap_side <= GAP_PASS:
        verdict = "PASS"
    elif gap_side <= GAP_WARN:
        verdict = "WARN"
        lines.append(f"  WARN: {100 * gap_side:.1f}% longer than optimal (pass threshold {100 * GAP_PASS:.0f}%).")
    else:
        verdict = "FAIL"
        lines.append(f"  FAIL: {100 * gap_side:.1f}% longer than optimal; the solver stopped far from the optimum.")
    if side != ref["side"]:
        lines.append("  note: went around the longer side, a local optimum. The gap above is measured "
                     "against that side's optimum; the overall gap is the first line.")
        if verdict == "PASS":
            verdict = "WARN"
    return verdict, lines


def check(args):
    env = load_environments(args.envs)[0]
    if len(env.obstacles) != 1 or env.obstacles[0].kind != "cylinder":
        raise SystemExit(f"{args.envs} is not a TFR environment (run `python tfr_benchmark.py make`).")
    tfr = env.obstacles[0]
    R = tfr.radius + env.clearance
    ref = shortest_path_around_circle(env.start_pos, env.goal_pos, tfr.center_xy, R)
    rt = turn_radius_m()

    print("Stadium TFR detour: known-solution check")
    print(f"  TFR radius {tfr.radius:.0f} m + clearance {env.clearance:.0f} m = {R:.0f} m"
          + (f"  (= {R / rt:.1f} C172 turn radii)" if rt else ""))
    print(f"  straight line {ref['straight'] / 1000:.3f} km (blocked); optimum {ref['length'] / 1000:.3f} km "
          f"on the {ref['side']} side = tangent {ref['tangent_m'][0] / 1000:.2f} km + arc "
          f"{ref['arc_m'] / 1000:.2f} km ({ref['arc_deg']:.1f} deg) + tangent {ref['tangent_m'][1] / 1000:.2f} km")
    print(f"  other side {ref['lengths']['left' if ref['side'] == 'right' else 'right'] / 1000:.3f} km\n")

    verdicts, plot_paths = [], []
    if os.path.exists(args.results):
        results = json.load(open(args.results))
        r = next((r for r in results if r["env_index"] == 0), None)
        if r is None:
            raise SystemExit(f"No result for environment 0 in {args.results}.")
        fp = r.get("env_fingerprint")
        if fp is not None and fp != env.fingerprint():
            raise SystemExit(f"{args.results} was not solved on {args.envs}; re-run scp_aircraft.py with "
                             f"--envs {args.envs}.")
        print(f"SCP: status {r['status']}, {r['iterations']} iterations on the last start, "
              f"starts tried {len(r.get('attempts', []))}, solve time {r['solve_time']:.1f} s")
        if not r["success"]:
            verdicts.append("FAIL")
            print("  FAIL: SCP did not return a feasible plan. If the status is max_iterations, the detour "
                  "(many turn radii wide) may need more iterations: try MAX_ITERS = 150 in scp_aircraft.py.")
        P, X = scp_path(r)
        V = X[:, 3]
        v, lines = assess("SCP plan", P, env, ref)
        print("\n".join(lines))
        t_ref = ref["length"] / np.mean(V)
        print(f"  flight time           {r['flight_time']:9.1f} s    (optimum path at the plan's mean speed "
              f"{np.mean(V):.1f} m/s: {t_ref:.1f} s)")
        print(f"  verdict: {v}\n")
        verdicts.append(v)
        plot_paths.append(("SCP plan", P, "k"))
    else:
        print(f"No SCP results yet ({args.results}). Run:\n  python scp_aircraft.py --envs {args.envs} "
              f"--out {args.results} --feasible-out tfr_feasible.json --verbose\n")

    if args.flight:
        import csv
        with open(args.flight) as f:
            rows = list(csv.DictReader(f))
        F = np.array([[float(q["x_east_m"]), float(q["y_north_m"]), float(q["z_up_m"])] for q in rows])
        v, lines = assess("JSBSim flight", F, env, ref, f" ({args.flight})", is_plan=False)
        print("\n".join(lines))
        print(f"  verdict: {v}  (a flown path may be a little shorter or longer than the plan;\n"
              f"           what matters most here is that it stays out of the TFR)\n")
        verdicts.append(v)
        plot_paths.append(("JSBSim flight", F, "m"))

    if verdicts:
        overall = "FAIL" if "FAIL" in verdicts else ("WARN" if "WARN" in verdicts else "PASS")
        print(f"OVERALL: {overall}")
    if args.plot:
        plot(env, ref, plot_paths, args.save)


def plot(env, ref, paths, save):
    import matplotlib.pyplot as plt
    tfr = env.obstacles[0]
    C, rho, R = tfr.center_xy, tfr.radius, tfr.radius + env.clearance
    th = np.linspace(0, 2 * np.pi, 400)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.fill(C[0] + rho * np.cos(th), C[1] + rho * np.sin(th), color="tab:red", alpha=0.15, label="TFR (3 NM)")
    ax.plot(C[0] + R * np.cos(th), C[1] + R * np.sin(th), "r:", lw=1, label="TFR + clearance")
    ax.plot(*ref["path"].T, "g-", lw=3, alpha=0.5, label=f"optimum ({ref['length'] / 1000:.3f} km)")
    for name, P, col in paths:
        ax.plot(P[:, 0], P[:, 1], col + "-", lw=1.2, label=f"{name} ({path_length(P) / 1000:.3f} km)")
    ax.plot(*env.start_pos[:2], "go", ms=8, label="start")
    ax.plot(*env.goal_pos[:2], "ro", ms=8, label="goal")
    ax.set_xlim(env.bounds_lo[0], env.bounds_hi[0])
    ax.set_ylim(env.bounds_lo[1], env.bounds_hi[1])
    ax.set_aspect("equal")
    ax.set_xlabel("east [m]"); ax.set_ylabel("north [m]")
    ax.set_title("Stadium TFR detour: SCP vs. analytic optimum (top view)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save, dpi=130)
    print(f"Plot saved to {save}")
    plt.show()


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Known-solution test: C172 detour around a stadium TFR.")
    sub = ap.add_subparsers(dest="command", required=True)
    m = sub.add_parser("make", help="write the TFR environment and print the analytic optimum")
    m.add_argument("--out", default="tfr_env.json")
    m.add_argument("--radius-nm", type=float, default=TFR_RADIUS_NM)
    m.add_argument("--distance-km", type=float, default=DISTANCE_KM)
    m.add_argument("--offset-m", type=float, default=GOAL_OFFSET_M)
    m.add_argument("--altitude-m", type=float, default=ALTITUDE_M)
    m.add_argument("--plot", action="store_true", help="plot the environment and the optimum")
    c = sub.add_parser("check", help="compare an SCP plan (and optionally a JSBSim flight) to the optimum")
    c.add_argument("--envs", default="tfr_env.json")
    c.add_argument("--results", default="tfr_results.json")
    c.add_argument("--flight", default=None, help="jsbsim_commands.csv from `jsbsim_c172.py record`")
    c.add_argument("--plot", action="store_true")
    c.add_argument("--save", default="tfr_check.png", help="where --plot saves its figure")
    args = ap.parse_args()

    if args.command == "make":
        env, ref = make_environment(args.radius_nm, args.distance_km, args.offset_m, args.altitude_m)
        save_environments([env], args.out)
        rt = turn_radius_m()
        R = env.obstacles[0].radius + env.clearance
        print(f"Wrote {args.out}: workspace {env.bounds_hi[0] / 1000:.1f} x {env.bounds_hi[1] / 1000:.1f} km "
              f"x {env.bounds_hi[2]:.0f} m, one full-height TFR cylinder")
        print(f"  TFR radius {env.obstacles[0].radius:.0f} m + clearance {env.clearance:.0f} m = {R:.0f} m"
              + (f"  (= {R / rt:.1f} C172 turn radii)" if rt else ""))
        print(f"  straight line {ref['straight'] / 1000:.3f} km, blocked")
        print(f"  OPTIMUM {ref['length'] / 1000:.3f} km on the {ref['side']} side: tangent "
              f"{ref['tangent_m'][0] / 1000:.2f} km + arc {ref['arc_m'] / 1000:.2f} km ({ref['arc_deg']:.1f} deg)"
              f" + tangent {ref['tangent_m'][1] / 1000:.2f} km")
        if rt and R < 3 * rt:
            print("  note: the circle is only a few turn radii wide, so the true optimum is noticeably "
                  "longer than this geometric bound (turns limit it). Expect a larger gap.")
        print(f"\nNext:  python scp_aircraft.py --envs {args.out} --out tfr_results.json "
              f"--feasible-out tfr_feasible.json --verbose")
        if args.plot:
            plot(env, ref, [], "tfr_env.png")
    else:
        check(args)


if __name__ == "__main__":
    main()
