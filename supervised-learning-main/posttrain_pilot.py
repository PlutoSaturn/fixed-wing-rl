#!/usr/bin/env python3
"""
posttrain_pilot.py - post-training for safety: continue a trained pilot at a FIXED difficulty with
crashes punished much harder than misses, until the crash rate is (near) zero.

It starts the same Modal training job as rl_pilot.py (same code path, same outputs under
/rl_pilot/<name>/ on the Volume), with these post-training settings:

    --init-from RUN     start from that run's weights (required; a new --name)
    --keep / --pool     the difficulty, held fixed: obstacle share and environment pool (default 0.5, 1.0)
    --crash-penalty     crash / out of bounds / stall penalty (default 100; training uses 25, a missed
                        goal costs up to 10, a timeout 15)
    --stop-crash        stop once a noise-free check (256 flights, every 50 updates) has at most this
                        percentage of crashes / out of bounds / stalls (default 0)
    --updates           at most this many updates (default 1500); a plateau also ends the run

A strategist (default --stage strategy) needs --flier (the frozen flier); a flier needs --strategy.

    python supervised-learning-main/posttrain_pilot.py --name strat_r5 --flier flier_r4b --init-from strat_r3a
    python supervised-learning-main/posttrain_pilot.py --name flier_r5 --stage flier --strategy strat_r5 \\
        --init-from flier_r4b --keep 0.75
    add --dry-run to print the Modal command without running it

Watch the noise-free crash rate in the log, "(noise-free 85.2%, crash 3.1%, ...)", or in updates.csv
(crash_noise_free_pct). Download and plot as usual with rl_pilot.py --download-only / plot_pilot.py.
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description="Post-train a pilot for safety at a fixed difficulty.")
    ap.add_argument("--name", required=True, help="new run name")
    ap.add_argument("--init-from", required=True, help="run to start from (its weights)")
    ap.add_argument("--stage", default="strategy", choices=["strategy", "flier"])
    ap.add_argument("--flier", default="", help="strategist post-training: the frozen flier run")
    ap.add_argument("--strategy", default="", help="flier post-training: the frozen strategist run")
    ap.add_argument("--keep", type=float, default=0.5, help="obstacle share, held fixed")
    ap.add_argument("--pool", type=float, default=1.0, help="share of the training environments, held fixed")
    ap.add_argument("--crash-penalty", type=float, default=100.0)
    ap.add_argument("--stop-crash", type=float, default=0.0, help="stop at this noise-free crash rate (%%)")
    ap.add_argument("--updates", type=int, default=1500)
    ap.add_argument("--start-std", type=float, default=None, help="exploration noise to start with (optional)")
    ap.add_argument("--no-detach", action="store_true", help="keep the run attached to this terminal")
    ap.add_argument("--dry-run", action="store_true", help="print the command only")
    a, extra = ap.parse_known_args()                 # anything else is passed on to rl_pilot.py unchanged

    if a.stage == "strategy" and not a.flier:
        raise SystemExit("a strategist post-training needs --flier NAME (the frozen flier)")
    if a.stage == "flier" and a.flier:
        raise SystemExit("a flier post-training takes --strategy NAME, not --flier")
    cmd = ["modal", "run"] + ([] if a.no_detach else ["--detach"]) + [str(_HERE / "rl_pilot.py"),
           "--name", a.name, "--stage", a.stage, "--init-from", a.init_from,
           "--start-keep", str(a.keep), "--start-pool", str(a.pool), "--fixed-difficulty",
           "--crash-penalty", str(a.crash_penalty), "--stop-crash", str(a.stop_crash),
           "--updates", str(a.updates)]
    if a.flier:
        cmd += ["--flier", a.flier]
    if a.strategy:
        cmd += ["--strategy", a.strategy]
    if a.start_std is not None:
        cmd += ["--start-std", str(a.start_std)]
    cmd += extra
    print("post-training:", f"{a.stage} '{a.name}' from '{a.init_from}' at {a.keep:.0%} of the obstacles, "
          f"{a.pool:.0%} of the environments (fixed); crash penalty {a.crash_penalty:g}; "
          f"stop at <= {a.stop_crash:g}% noise-free crashes or {a.updates} updates")
    print(" ".join(shlex.quote(c) for c in cmd))
    if a.dry_run:
        return
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()