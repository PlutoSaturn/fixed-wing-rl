#!/usr/bin/env python3
"""
guided_test.py - Tests of the GUIDED follower (guided_follower.py): the plan streamed from a
"companion computer" in GUIDED mode instead of uploaded as waypoints, compared with tracking setup v4
(batch tier3_v4, waypoints).

Round 1 (2026-11-07; batches g100, g150, g200 = lookahead 100/150/200 m, on easy_0, nominal_2, dense_1):
all pass; max sideways error 9.5-9.9 -> 1.0-1.5 m on the curvy plans with no tracking correction,
dense_1 rms 4.2 -> 1.7 m; altitude no better (it now sets the max error); throttle travel +20-40%,
single spikes at each speed command (sent as "change at once"). Lookahead 150 m kept.

Round 2 (batches h0, h1, h2; dense_1): speed changes ramped, altitude lead 0/30/60 m. Worse: ArduPlane
restarts a speed ramp at every command, so the target barely moved (airspeed flat at ~51 m/s, the climb
17 m behind without a lead); throttle bumps about once a second came from the ALTITUDE commands
(clipped to "rate x ms since the last update" around the current altitude), not from the speed.

Round 3 (current; dense_1): speed changed at once again; altitude sent with a 100000 m/s rate (no
clipping) and only when it changes by 0.5 m; altitude lead:
  k0  none      k1  30 m      k2  60 m

Usage (PyCharm terminal, project root):
  python -m ardupilot.guided_test make       # writes localstore/batches/k0, k1, k2
  (WSL)  bash /mnt/c/Users/isaia/fixed-wing-rl/ardupilot/fly_guided_test.sh
  python -m ardupilot.guided_test analyze    # compares with tier3_v4, including the throttle
"""
import csv
import json
import sys
from types import SimpleNamespace

import numpy as np

from ardupilot import ardupilot_mission as M

PAIRS = [["tier3_easy_envs.json", "tier3_easy_results.json"],
         ["tier3_nominal_envs.json", "tier3_nominal_results.json"],
         ["tier3_dense_envs.json", "tier3_dense_results.json"]]
FLY = ["tier3_dense_envs_1"]                     # round 1: also tier3_easy_envs_0, tier3_nominal_envs_2
CONFIGS = {"k0": dict(lookahead=150.0, alt_lead=0.0), "k1": dict(lookahead=150.0, alt_lead=30.0),
           "k2": dict(lookahead=150.0, alt_lead=60.0)}
BASELINE = "tier3_v4"
ROUND1 = ["g150", "h2"]                           # earlier rounds, shown in the comparison too
COLS = [("outcome", "result"), ("max_error_m", "max err"), ("rms_error_m", "rms"),
        ("max_horizontal_error_m", "max side"), ("max_vertical_error_m", "max vert"),
        ("min_clearance_m", "clear"), ("goal_miss_m", "goal"), ("airspeed_vs_plan_rms_ms", "spd rms"),
        ("roll_oscillation_deg", "roll osc")]


def make():
    for name, c in CONFIGS.items():
        L = c["lookahead"]
        note = (f"{name}: GUIDED follower, lookahead {L:g} m, altitude correction on, altitude lead "
                f"{c['alt_lead']:g} m, plan airspeed changed at once, altitude rate 100000 m/s, step 0.5 m")
        print(f"\n##### {note}")
        M.batch_make(SimpleNamespace(name=name, pair=PAIRS, chord_tol=M.BATCH_CHORD_TOL_M, speed_profile=None,
                                     speed_lead=0.0, speedup=M.BATCH_SPEEDUP, note=note, param=[], overwrite=True,
                                     max_waypoints=None, fly_all=False, alt_lead=0.0, min_spacing=None,
                                     wp_spacing=None, track_comp=None, follower="guided", lookahead=L,
                                     guided_alt_comp="on", guided_alt_lead=c["alt_lead"], guided_speed_ramp="off",
                                     guided_alt_rate=100000.0, guided_alt_step=0.5))
    bad, n = [], 0
    for name in CONFIGS:
        for inst in json.load(open(M.batch_path(name) / "manifest.json"))["instances"]:
            if inst["id"] in FLY:
                n += 1
                if inst["status"] != "flyable":
                    bad.append(f"  {name} {inst['id']}: {inst['status']} {'; '.join(inst.get('rejects', []))}")
    print("\n" + "=" * 70)
    if bad:
        print(f"NOT READY: {len(bad)} of {n} flights are not flyable. Do not fly yet; send this:")
        print("\n".join(bad))
    else:
        print(f"READY: all {n} flights are flyable. Now fly them (WSL):")
        print("  bash " + M.wsl_path(M.project_root() / "ardupilot" / "fly_guided_test.sh"))


def read(name):
    f = M.batch_path(name) / "results.csv"
    return {r["id"]: r for r in csv.DictReader(open(f))} if f.exists() else {}


def throttle_activity(name, iid):
    """Throttle travel (% per minute) and share of time at 0 or 100% over the plan segment."""
    run = M.batch_path(name, "runs", iid)
    log, flight = run / "flight.tlog", run / "flight.csv"
    if not (log.exists() and flight.exists()):
        return None
    f = M.read_tlog(str(log))
    c = list(csv.DictReader(open(flight)))
    if "throttle" not in f or not c:
        return None
    ll = np.c_[f["lat"], f["lon"]]
    g = lambda r: np.array([float(r["lat_deg"]), float(r["lon_deg"])])
    i0, i1 = (int(np.argmin(np.linalg.norm(ll - g(r), axis=1))) for r in (c[0], c[-1]))
    th, t = np.asarray(f["throttle"][i0:i1], float), np.asarray(f["t"][i0:i1], float)
    if len(th) < 10:
        return None
    travel = np.abs(np.diff(th)).sum() / ((t[-1] - t[0]) / 60.0)
    return travel, 100.0 * np.mean((th <= 1) | (th >= 99))


def analyze():
    for name in CONFIGS:
        M.batch_analyze(SimpleNamespace(name=name, plot=False))
    names = [BASELINE] + ROUND1 + list(CONFIGS)
    res = {n: read(n) for n in names}
    print(f"\n\n========== GUIDED follower test ({BASELINE} = tracking setup v4, waypoints) ==========")
    for iid in FLY:
        print(f"\n{iid}")
        print(f"  {'':10}" + "".join(f"{lab:>10}" for _, lab in COLS) + f"{'thr %/min':>11}{'thr 0/100':>10}")
        for n in names:
            r = res[n].get(iid, {})
            th = throttle_activity(n, iid)
            print(f"  {n:10}" + "".join(f"{r.get(k, '-') or '-':>10}" for k, _ in COLS)
                  + (f"{th[0]:>11.0f}{th[1]:>9.0f}%" if th else f"{'-':>11}{'-':>10}"))
    print("\nPer-flight charts: localstore/batches/<batch>/runs/<id>/check.png; "
          "the follower's own log: localstore/batches/<batch>/logs/<id>.guided.csv")


if __name__ == "__main__":
    {"make": make, "analyze": analyze}.get(sys.argv[1] if len(sys.argv) > 1 else "", lambda: print(__doc__))()
