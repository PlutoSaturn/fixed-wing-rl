#!/usr/bin/env python3
"""
speed_lead_test.py - Does commanding the plan's airspeed a few seconds EARLY help?

Background: the manager's suggestion (fly the plan's speed on each leg) was tried in
batch v2_dense_speed and made dense_1 worse (RMS 14.0 -> 18.3 m), because ArduPilot reached
each commanded speed 5-10 s late. This test sends each leg's speed command early, on top of
tracking setup v3 (a waypoint every 30 m + tracking correction), and flies dense_1:

  spd0  plan speeds, no lead       (the old idea, on v3)
  spd4  plan speeds, 4 s early
  spd8  plan speeds, 8 s early
Baseline: tier3_v3b (v3, one constant airspeed).

The hope: on dense_1 the plan speeds up to ~58 m/s before its climb at ~110 s and trades that
speed for height; at a constant ~52 m/s the throttle hits 100 % and the altitude falls 24 m behind.

Usage (PyCharm terminal, project root):
  python -m ardupilot.speed_lead_test make       # writes localstore/batches/spd{0,4,8}/
  (WSL)  bash /mnt/c/Users/isaia/fixed-wing-rl/ardupilot/fly_speed_lead.sh
  python -m ardupilot.speed_lead_test analyze    # compares with tier3_v3b
"""
import csv
import json
import sys
from types import SimpleNamespace

from ardupilot import ardupilot_mission as M

PAIRS = [["tier3_dense_envs.json", "tier3_dense_results.json"]]
FLY = ["tier3_dense_envs_1"]
CONFIGS = {"spd0": 0.0, "spd4": 4.0, "spd8": 8.0}
BASELINE = "tier3_v3b"
COLS = [("outcome", "result"), ("max_error_m", "max err"), ("rms_error_m", "rms"),
        ("max_vertical_error_m", "max vert"), ("max_horizontal_error_m", "max side"),
        ("min_clearance_m", "clear"), ("goal_miss_m", "goal"), ("airspeed_vs_plan_rms_ms", "spd rms"),
        ("flight_time_s", "time")]


def make():
    for name, lead in CONFIGS.items():
        note = f"{name}: v3 + plan airspeed per leg, commanded {lead:g} s early"
        print(f"\n##### {note}")
        M.batch_make(SimpleNamespace(name=name, pair=PAIRS, chord_tol=M.BATCH_CHORD_TOL_M,
                                     speed_profile="plan", speed_lead=lead, speedup=M.BATCH_SPEEDUP,
                                     note=note, param=[], overwrite=True, max_waypoints=None,
                                     fly_all=False, alt_lead=0.0, min_spacing=None,
                                     wp_spacing=None, track_comp=None))
    bad, n = [], 0
    print()
    for name in CONFIGS:
        for inst in json.load(open(M.batch_path(name) / "manifest.json"))["instances"]:
            if inst["id"] in FLY:
                n += 1
                print(f"  {name} {inst['id']}: {inst['status']}, {inst.get('n_mission_items')} mission items, "
                      f"{inst.get('n_speed_commands')} speed commands")
                if inst["status"] != "flyable":
                    bad.append(f"  {name} {inst['id']}: {inst['status']} {'; '.join(inst.get('rejects', []))}")
    print("\n" + "=" * 70)
    if bad:
        print(f"NOT READY: {len(bad)} of {n} flights are not flyable. Do not fly yet; send this:")
        print("\n".join(bad))
    else:
        print(f"READY: all {n} flights are flyable. Now fly them (WSL):")
        print("  bash " + M.wsl_path(M.project_root() / "ardupilot" / "fly_speed_lead.sh"))


def read(name):
    f = M.batch_path(name) / "results.csv"
    return {r["id"]: r for r in csv.DictReader(open(f))} if f.exists() else {}


def analyze():
    for name in CONFIGS:
        M.batch_analyze(SimpleNamespace(name=name, plot=False))
    names = [BASELINE] + list(CONFIGS)
    res = {n: read(n) for n in names}
    print(f"\n\n========== speed lead test ({BASELINE} = v3, constant airspeed) ==========")
    for iid in FLY:
        print(f"\n{iid}")
        print(f"  {'':10}" + "".join(f"{lab:>10}" for _, lab in COLS))
        for n in names:
            r = res[n].get(iid, {})
            print(f"  {n:10}" + "".join(f"{r.get(k, '-') or '-':>10}" for k, _ in COLS))
    print("\nPer-flight charts: localstore/batches/<spd0|spd4|spd8>/runs/tier3_dense_envs_1/check.png")


if __name__ == "__main__":
    {"make": make, "analyze": analyze}.get(sys.argv[1] if len(sys.argv) > 1 else "", lambda: print(__doc__))()
