#!/usr/bin/env python3
"""
wobble_sweep.py - Try several autopilot settings in one SITL flight, straight and level,
to find what removes the remaining ~2.2 s pitch wobble.

What we know (2026-11-05): the wobble only appears when TECS steers pitch (FBWA with a
fixed pitch target is calm); pitch then moves ~2x more than TECS commands. Ruled out:
waypoints, speed commands, airspeed in pitch (TECS_SPDWEIGHT), the throttle loop
(throttle locked), pitch-rate P and I, PTCH2SRV_TCONST, TECS_PTCH_DAMP, SITL speedup.
Left: how jumpy TECS's pitch commands are, and how much damping the pitch loop has.

The script flies CRUISE (straight, holding altitude and heading; sticks centred, throttle
stick at --throttle), and holds each setting for --each simulated seconds, putting the
original values back before the next one:
  1  baseline (the current c172_sitl.parm)
  2  TECS_INTEG_GAIN 0.1          (the one setting that ever shrank it: 0.3->0.15 cut 1.7->1.3 deg)
  3  TECS_HGT_OMEGA 1.5           (smoother height / climb-rate estimate, so smoother commands)
  4  PTCH_RATE_D 0.03             (more damping in the pitch loop; P and I were lowered before, D never raised)
  5  TECS_INTEG_GAIN 0.1 + TECS_HGT_OMEGA 1.5
then puts everything back and switches to LOITER. `ardupilot_mission wobble` gives one row
per setting.

Run it in WSL (a second Ubuntu terminal, venv-ardupilot active) while SITL is running and
the aircraft is flying at altitude. First, in MAVProxy:
    output add 127.0.0.1:14552
then:
    python3 /mnt/c/Users/isaia/fixed-wing-rl/ardupilot/wobble_sweep.py
Do not use MAVProxy's `rc` command during the run. Ctrl-C stops early (it still restores
the parameters and switches to LOITER).
"""
import argparse
import sys
import time

from pymavlink import mavutil

PLANE_MODE = {"CRUISE": 7, "LOITER": 12}
IGNORE, RELEASE = 65535, 0
SEND_HZ = 10
STEPS = [
    ("baseline", {}),
    ("TECS_INTEG_GAIN 0.1", {"TECS_INTEG_GAIN": 0.1}),
    ("TECS_HGT_OMEGA 1.5", {"TECS_HGT_OMEGA": 1.5}),
    ("PTCH_RATE_D 0.03", {"PTCH_RATE_D": 0.03}),
    ("TECS_INTEG_GAIN 0.1 + TECS_HGT_OMEGA 1.5", {"TECS_INTEG_GAIN": 0.1, "TECS_HGT_OMEGA": 1.5}),
]
# --preset l6: the pitch gains AUTOTUNE found at AUTOTUNE_LEVEL 6 (2026-11-05, autotune_l6.tlog). With them
# pitch follows its command 1:1 (the 2x amplification is gone), but the time constant it chose (0.75 s)
# made TECS cycle at 7.4 s, +-4 deg. These steps pair the new rate gains with faster time constants, or
# slow TECS to match. Loaded in flight because SITL (-w) forgets AUTOTUNE's gains on restart.
L6_GAINS = {"PTCH2SRV_TCONST": 0.75, "PTCH2SRV_RMAX_UP": 75.0, "PTCH2SRV_RMAX_DN": 75.0,
            "PTCH_RATE_P": 2.0031814575195312, "PTCH_RATE_I": 1.5023860931396484,
            "PTCH_RATE_D": 0.19382575154304504, "PTCH_RATE_FF": 1.2623021602630615,
            "PTCH_RATE_FLTT": 2.122066020965576}
STEPS_L6 = [
    ("L6 gains as tuned (PTCH2SRV_TCONST 0.75)", {}),
    ("L6 gains + PTCH2SRV_TCONST 0.4", {"PTCH2SRV_TCONST": 0.4}),
    ("L6 gains + PTCH2SRV_TCONST 0.25", {"PTCH2SRV_TCONST": 0.25}),
    ("L6 gains + TECS_TIME_CONST 10 (slower TECS)", {"TECS_TIME_CONST": 10.0}),
]
PRESETS = {"v2": ({}, STEPS), "l6": (L6_GAINS, STEPS_L6)}


def main():
    ap = argparse.ArgumentParser(description="In-flight sweep of settings against the pitch wobble.")
    ap.add_argument("--connect", default="udpin:127.0.0.1:14552",
                    help="where MAVProxy sends to (MAVProxy: output add 127.0.0.1:14552)")
    ap.add_argument("--each", type=float, default=150.0, help="simulated seconds per setting")
    ap.add_argument("--throttle", type=int, default=1846, help="throttle stick PWM in CRUISE (sets the airspeed)")
    ap.add_argument("--preset", choices=sorted(PRESETS), default="v2",
                    help="v2: the TECS / pitch-damping sweep on the current gains; l6: the AUTOTUNE level-6 "
                         "pitch gains with several time constants")
    ap.add_argument("--baseline-only", action="store_true",
                    help="only fly straight in CRUISE with the current settings and change nothing "
                         "(e.g. to measure the wobble right after AUTOTUNE)")
    ap.add_argument("--sysid", type=int, default=255,
                    help="our MAVLink system id; ArduPilot only takes stick overrides from its GCS id (255)")
    args = ap.parse_args()

    m = mavutil.mavlink_connection(args.connect, source_system=args.sysid)
    print(f"Waiting for the aircraft on {args.connect} ...")
    st = {"mode": None, "rc3": None, "boot_s": None, "alt": None, "params": {}}
    end = time.time() + 20
    while time.time() < end and st["mode"] is None:
        hb = m.recv_match(type="HEARTBEAT", blocking=True, timeout=1)
        if hb is not None and hb.type == mavutil.mavlink.MAV_TYPE_FIXED_WING:
            st["mode"] = hb.custom_mode
            m.target_system, m.target_component = hb.get_srcSystem(), 1
    if st["mode"] is None:
        sys.exit("No heartbeat from the aircraft. In MAVProxy, run:  output add 127.0.0.1:14552")
    print(f"Connected to system {m.target_system}.")

    def drain():
        while True:
            msg = m.recv_msg()
            if msg is None:
                return
            t = msg.get_type()
            if t == "HEARTBEAT" and msg.type == mavutil.mavlink.MAV_TYPE_FIXED_WING:
                st["mode"] = msg.custom_mode
            elif t == "RC_CHANNELS":
                st["rc3"] = msg.chan3_raw
            elif t == "PARAM_VALUE":
                pid = msg.param_id.decode(errors="replace") if isinstance(msg.param_id, bytes) else msg.param_id
                st["params"][pid.strip("\x00")] = msg.param_value
            elif t == "GLOBAL_POSITION_INT":
                st["alt"] = msg.relative_alt / 1000.0
            if hasattr(msg, "time_boot_ms") and t in ("ATTITUDE", "GLOBAL_POSITION_INT", "SYSTEM_TIME"):
                st["boot_s"] = msg.time_boot_ms / 1000.0

    sticks = {1: 1500, 2: 1500, 3: args.throttle}

    def override(values):
        chans = [IGNORE] * 8
        for ch, pwm in values.items():
            chans[ch - 1] = pwm
        m.mav.rc_channels_override_send(m.target_system, m.target_component, *chans)

    def wait_wall(seconds):
        end = time.time() + seconds
        while time.time() < end:
            override(sticks)
            drain()
            time.sleep(1.0 / SEND_HZ)

    def wait_sim(seconds):
        "Hold the sticks for this many SIMULATED seconds (SITL may run faster than real time)."
        wait_wall(0.5)
        t0 = st["boot_s"]
        while st["boot_s"] is None or t0 is None or st["boot_s"] - t0 < seconds:
            wait_wall(0.5)
            t0 = st["boot_s"] if t0 is None else t0

    def param_get(name):
        for _ in range(5):
            st["params"].pop(name, None)
            m.mav.param_request_read_send(m.target_system, m.target_component, name.encode(), -1)
            end = time.time() + 2
            while time.time() < end and name not in st["params"]:
                wait_wall(0.1)
            if name in st["params"]:
                return st["params"][name]
        sys.exit(f"Could not read {name} from the aircraft.")

    def param_set(name, value):
        for _ in range(5):
            st["params"].pop(name, None)
            m.mav.param_set_send(m.target_system, m.target_component, name.encode(), float(value),
                                 mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
            end = time.time() + 2
            while time.time() < end and name not in st["params"]:
                wait_wall(0.1)
            if name in st["params"] and abs(st["params"][name] - value) < 1e-4:
                print(f"    {name} = {value:g}")
                return
        raise RuntimeError(f"the aircraft did not accept {name} = {value:g}")

    def set_mode(name):
        m.mav.command_long_send(m.target_system, m.target_component, mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0,
                                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, PLANE_MODE[name], 0, 0, 0, 0, 0)

    base, steps = PRESETS[args.preset]
    steps = steps[:1] if args.baseline_only else steps
    names = sorted(set(base) | {k for _, ch in steps for k in ch})
    original = {n: param_get(n) for n in names}
    if original:
        print("Current values: " + ", ".join(f"{n}={v:g}" for n, v in original.items()))

    wait_wall(3.0)
    if st["rc3"] is None or abs(st["rc3"] - args.throttle) > 20:
        override({1: RELEASE, 2: RELEASE, 3: RELEASE})
        sys.exit(f"The aircraft is not taking this script's stick inputs (throttle input {st['rc3']}, sent "
                 f"{args.throttle}). Check MAV_GCS_SYSID matches --sysid ({args.sysid}).")
    if st["alt"] is not None and st["alt"] < 100:
        override({1: RELEASE, 2: RELEASE, 3: RELEASE})
        sys.exit(f"The aircraft is only {st['alt']:.0f} m above home; start this once it is flying at altitude.")
    set_mode("CRUISE")
    wait_wall(3.0)
    if st["mode"] != PLANE_MODE["CRUISE"]:
        override({1: RELEASE, 2: RELEASE, 3: RELEASE})
        sys.exit("The aircraft did not switch to CRUISE (is it armed and flying?).")
    total = len(steps) * args.each
    print(f"CRUISE, straight and level. {len(steps)} settings x {args.each:.0f} s simulated "
          f"({total / 60:.1f} min simulated).")
    try:
        for k, (label, change) in enumerate(steps, 1):
            print(f"[{k}/{len(steps)}] {label}")
            target = {**original, **base, **change}           # this step = the preset's base + its change
            for n in names:
                if abs(st["params"].get(n, original[n]) - target[n]) > 1e-6:
                    param_set(n, target[n])
            wait_sim(args.each)
    except KeyboardInterrupt:
        print("Stopped early.")
    finally:
        if original:
            print("Restoring the original values:")
        for n, v in original.items():
            try:
                param_set(n, v)
            except RuntimeError as e:
                print(f"    WARNING: {e}")
        set_mode("LOITER")
        wait_wall(2.0)
        override({1: RELEASE, 2: RELEASE, 3: RELEASE})
        print("Done; switched to LOITER. Now copy the log (see the docs) and run wobble on it.")


if __name__ == "__main__":
    main()
