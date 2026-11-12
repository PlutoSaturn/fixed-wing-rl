#!/usr/bin/env python3
"""
fix_sweep.py - Build, then score, candidate fixes for the remaining ArduPilot tracking
problems (the steady pitch/throttle wobble and the ~30 m altitude lag on steep climbs),
each flown on three representative plans.

Round 2 (current). Round 1 (fixA-D, 2026-11-05) showed: slowing the pitch loop
(PTCH2SRV_TCONST 0.5) made the wobble slower and larger (2.2 s -> 4.5 s), so the wobble
is TECS (outer loop) reacting faster than the pitch loop can follow; and the altitude
lead (--alt-lead) put waypoints ~30 m off the plan, so the flyability check rejected the
nominal/dense missions, and on easy_0 it made the vertical error worse. Both dropped.

  fixS1  TECS_PTCH_DAMP 0.1 (was 0.3): less climb-rate feedback into the pitch command
  fixS2  TECS_PTCH_DAMP 0.0
  fixS3  fixS1 + TECS_SPDWEIGHT 0: pitch holds height only, so the plane trades speed for
         height on steep climbs (as the plan does) instead of maxing the throttle
  fixS4  fixS3 + fewer waypoints where the plan is nearly straight: chord tolerance 5 m
         (was 1 m), waypoints at least 300 m apart (was 200 m)

Usage (PyCharm terminal, project root):
  python -m ardupilot.fix_sweep make        # writes localstore/batches/fixS{1,2,3,4}/; checks all are flyable
  (WSL)  bash /mnt/c/Users/isaia/fixed-wing-rl/ardupilot/fly_fix_sweep.sh
  python -m ardupilot.fix_sweep analyze     # scores all four and compares with tier3_v2
"""
import csv
import sys
from types import SimpleNamespace

from ardupilot import ardupilot_mission as M

PAIRS = [["tier3_easy_envs.json", "tier3_easy_results.json"],
         ["tier3_nominal_envs.json", "tier3_nominal_results.json"],
         ["tier3_dense_envs.json", "tier3_dense_results.json"]]
FLY = ["tier3_easy_envs_0", "tier3_nominal_envs_2", "tier3_dense_envs_1"]
CHORD = M.BATCH_CHORD_TOL_M
CONFIGS = {
    "fixS1": dict(param=["TECS_PTCH_DAMP=0.1"], chord_tol=CHORD, min_spacing=None),
    "fixS2": dict(param=["TECS_PTCH_DAMP=0"], chord_tol=CHORD, min_spacing=None),
    "fixS3": dict(param=["TECS_PTCH_DAMP=0.1", "TECS_SPDWEIGHT=0"], chord_tol=CHORD, min_spacing=None),
    "fixS4": dict(param=["TECS_PTCH_DAMP=0.1", "TECS_SPDWEIGHT=0"], chord_tol=5.0, min_spacing=300.0),
}
BASELINE = "tier3_v2"
COLS = [("outcome", "result"), ("max_error_m", "max err"), ("rms_error_m", "rms"),
        ("max_vertical_error_m", "max vert"), ("min_clearance_m", "clear"),
        ("pitch_oscillation_deg", "pitch osc"), ("roll_oscillation_deg", "roll osc"),
        ("altitude_oscillation_m", "alt osc")]


def make():
    for name, c in CONFIGS.items():
        note = f"{name}: " + ", ".join(c["param"]) + f", chord tol {c['chord_tol']:g} m" \
            + (f", min spacing {c['min_spacing']:g} m" if c["min_spacing"] else "")
        print(f"\n##### {note}")
        M.batch_make(SimpleNamespace(name=name, pair=PAIRS, chord_tol=c["chord_tol"], speed_profile="constant",
                                     speedup=M.BATCH_SPEEDUP, note=note, param=c["param"], overwrite=True,
                                     max_waypoints=None, fly_all=False, alt_lead=0.0,
                                     min_spacing=c["min_spacing"], wp_spacing=0, track_comp="none"))
    import json
    bad, n = [], 0
    for name in CONFIGS:
        for inst in json.load(open(M.batch_path(name) / "manifest.json"))["instances"]:
            if inst["id"] in FLY:
                n += 1
                if inst["status"] != "flyable":
                    bad.append(f"  {name} {inst['id']}: {inst['status']} {'; '.join(inst.get('rejects', []))}")
    print("\n" + "=" * 70)
    if bad:
        print(f"NOT READY: {len(bad)} of {n} flights would be skipped (not flyable). Do not fly yet; send this:")
        print("\n".join(bad))
    else:
        print(f"READY: all {n} flights are flyable. Now fly them (WSL):")
        print("  bash " + M.wsl_path(M.project_root() / "ardupilot" / "fly_fix_sweep.sh"))


def read(name):
    f = M.batch_path(name) / "results.csv"
    return {r["id"]: r for r in csv.DictReader(open(f))} if f.exists() else {}


def analyze():
    for name in CONFIGS:
        M.batch_analyze(SimpleNamespace(name=name, plot=False))
    names = [BASELINE] + list(CONFIGS)
    res = {n: read(n) for n in names}
    print("\n\n================ comparison (tier3_v2 = current setup) ================")
    for iid in FLY:
        print(f"\n{iid}")
        print(f"  {'':10}" + "".join(f"{lab:>11}" for _, lab in COLS))
        for n in names:
            r = res[n].get(iid, {})
            print(f"  {n:10}" + "".join(f"{r.get(k, '-') or '-':>11}" for k, _ in COLS))
    print("\nPer-flight charts: localstore/batches/<fix>/runs/<id>/check.png")


if __name__ == "__main__":
    {"make": make, "analyze": analyze}.get(sys.argv[1] if len(sys.argv) > 1 else "", lambda: print(__doc__))()
