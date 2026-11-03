#!/usr/bin/env python3
"""
autotune_inputs.py - Run ArduPlane AUTOTUNE in SITL without a transmitter.

AUTOTUNE learns the pitch (and/or roll) controller gains from sharp stick inputs.
In SITL there is no stick, so this script sends them as RC overrides through
MAVProxy: it sets the AUTOTUNE options, holds a fixed throttle, switches to
AUTOTUNE, sends repeated stick doublets (up, centre, down, centre), and finally
switches to LOITER, which ends AUTOTUNE and keeps the new gains (ArduPilot saves
them when the mode is left). The gains it found appear in the flight log, so
`python -m ardupilot.ardupilot_mission wobble --log <log>` lists them.

Run it in WSL (a second Ubuntu terminal, with venv-ardupilot active) while SITL
is running and the aircraft is circling at altitude. First, in MAVProxy:
    output add 127.0.0.1:14552
then, in the second Ubuntu terminal:
    python3 /mnt/c/Users/isaia/fixed-wing-rl/ardupilot/autotune_inputs.py --axis pitch

Do not use MAVProxy's `rc` command during the run (its overrides would fight
this script's). Ctrl-C stops early (it still switches to LOITER).
"""
import argparse
import sys
import time

from pymavlink import mavutil

CHANNEL = {"roll": 1, "pitch": 2}
THROTTLE_CHANNEL = 3
PLANE_MODE = {"AUTOTUNE": 8, "LOITER": 12}   # ArduPlane custom_mode numbers
IGNORE = 65535                 # RC override: leave this channel alone
RELEASE = 0                    # RC override: give this channel back to the RC input
AXES_BITS = {"roll": 1, "pitch": 2, "both": 3}
SEND_HZ = 10


def vehicle_heartbeat(m, timeout):
    "The next HEARTBEAT from the aircraft itself (not from a ground station)."
    end = time.time() + timeout
    while time.time() < end:
        hb = m.recv_match(type="HEARTBEAT", blocking=True, timeout=1)
        if hb is not None and hb.type == mavutil.mavlink.MAV_TYPE_FIXED_WING:
            return hb
    return None


def main():
    ap = argparse.ArgumentParser(description="Stick inputs for ArduPlane AUTOTUNE in SITL.")
    ap.add_argument("--connect", default="udpin:127.0.0.1:14552",
                    help="where MAVProxy sends to (MAVProxy: output add 127.0.0.1:14552)")
    ap.add_argument("--axis", choices=["pitch", "roll", "both"], default="pitch")
    ap.add_argument("--level", type=int, default=0,
                    help="AUTOTUNE_LEVEL; 0 keeps the current time constants and rate limits")
    ap.add_argument("--cycles", type=int, default=30, help="doublets per axis")
    ap.add_argument("--amplitude", type=int, default=400, help="stick deflection, PWM from centre (400 = 80%%)")
    ap.add_argument("--hold", type=float, default=0.8, help="seconds at each deflection")
    ap.add_argument("--pause", type=float, default=1.5, help="seconds centred between deflections")
    ap.add_argument("--throttle", type=int, default=1800, help="throttle PWM held during the run (1800 = 80%%)")
    ap.add_argument("--sysid", type=int, default=255,
                    help="our MAVLink system id; ArduPilot only takes stick overrides from its GCS id (MAV_GCS_SYSID, 255)")
    args = ap.parse_args()

    m = mavutil.mavlink_connection(args.connect, source_system=args.sysid)
    print(f"Waiting for the aircraft on {args.connect} ...")
    hb = vehicle_heartbeat(m, 20)
    if hb is None:
        sys.exit("No heartbeat from the aircraft. In MAVProxy, run:  output add 127.0.0.1:14552")
    m.target_system, m.target_component = hb.get_srcSystem(), 1
    print(f"Connected to system {m.target_system}.")
    state = {"mode": hb.custom_mode, "rc": {}}

    def drain():
        "Read everything waiting (keeps the socket from backing up); note the aircraft's mode and stick inputs."
        while True:
            msg = m.recv_msg()
            if msg is None:
                return
            if msg.get_type() == "HEARTBEAT" and msg.type == mavutil.mavlink.MAV_TYPE_FIXED_WING:
                state["mode"] = msg.custom_mode
            elif msg.get_type() == "RC_CHANNELS":
                state["rc"] = {2: msg.chan2_raw, 3: msg.chan3_raw}

    def set_mode(name):
        m.mav.command_long_send(m.target_system, m.target_component, mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0,
                                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, PLANE_MODE[name], 0, 0, 0, 0, 0)
        print(f"  mode {name}")

    def param_set(name, value):
        m.mav.param_set_send(m.target_system, m.target_component, name.encode(), float(value),
                             mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
        print(f"  set {name} = {value}")

    def override(values):
        chans = [IGNORE] * 8
        for ch, pwm in values.items():
            chans[ch - 1] = pwm
        m.mav.rc_channels_override_send(m.target_system, m.target_component, *chans)

    def hold(values, seconds):
        end = time.time() + seconds
        while time.time() < end:
            override(values)
            drain()
            time.sleep(1.0 / SEND_HZ)

    param_set("AUTOTUNE_AXES", AXES_BITS[args.axis])
    param_set("AUTOTUNE_LEVEL", args.level)
    hold({THROTTLE_CHANNEL: args.throttle}, 3.0)          # throttle first: AUTOTUNE flies like FBWA
    seen = state["rc"].get(THROTTLE_CHANNEL)
    if seen is None or abs(seen - args.throttle) > 20:    # the aircraft must actually see our sticks
        override({THROTTLE_CHANNEL: RELEASE})
        sys.exit(f"The aircraft is not taking this script's stick inputs (throttle input {seen}, sent "
                 f"{args.throttle}); AUTOTUNE not started, the aircraft stays in its current mode. "
                 f"Check MAV_GCS_SYSID (older ArduPilot: SYSID_MYGCS) matches --sysid ({args.sysid}).")
    print(f"  stick inputs reach the aircraft (throttle input {seen})")
    set_mode("AUTOTUNE")
    hold({THROTTLE_CHANNEL: args.throttle}, 3.0)
    if state["mode"] != PLANE_MODE["AUTOTUNE"]:
        set_mode("LOITER")
        sys.exit("The aircraft did not switch to AUTOTUNE (is it armed and flying?). Switched to LOITER.")
    print("AUTOTUNE on. MAVProxy prints its progress (AutoTune: ...).")

    axes = ["roll", "pitch"] if args.axis == "both" else [args.axis]
    try:
        for axis in axes:
            ch, c, a = CHANNEL[axis], 1500, args.amplitude
            for i in range(args.cycles):
                for pwm, sec in ((c + a, args.hold), (c, args.pause), (c - a, args.hold), (c, args.pause)):
                    hold({ch: pwm, THROTTLE_CHANNEL: args.throttle}, sec)
                if state["mode"] != PLANE_MODE["AUTOTUNE"]:
                    print("The aircraft left AUTOTUNE; stopping.")
                    return
                print(f"  {axis}: {i + 1}/{args.cycles} doublets")
    except KeyboardInterrupt:
        print("Stopped early.")
    finally:
        set_mode("LOITER")                                # leaving AUTOTUNE keeps the new gains
        hold({THROTTLE_CHANNEL: args.throttle}, 2.0)
        override({1: RELEASE, 2: RELEASE, 3: RELEASE})
        print("Switched to LOITER (AUTOTUNE gains kept). In MAVProxy:  param show PTCH_RATE*")


if __name__ == "__main__":
    main()
