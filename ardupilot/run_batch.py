#!/usr/bin/env python3
"""
run_batch.py - Tier 3: fly every flyable instance of a batch in ArduPilot SITL,
one after another, unattended.

For each instance in the batch's manifest (written by
`python -m ardupilot.ardupilot_mission batch-make`), this script:
  1. starts SITL (sim_vehicle.py, JSBSim C172, the instance's home location,
     ardupilot/c172_sitl.parm, -w, --speedup) without MAVProxy,
  2. connects to it directly and records everything it receives to the
     instance's .tlog (the same format MAVProxy writes),
  3. fetches the parameters (so the log records them), uploads the mission,
     switches to AUTO and arms (retrying until the EKF is ready),
  4. waits until the aircraft has been in the final loiter for DONE_HOLD_S of
     simulated time, or stops early on a timeout, a crash or a dead simulator,
  5. stops SITL and writes <id>.run.json (status, times, details).

Instances already completed are skipped, so an interrupted batch can simply be
started again. Run in WSL, in a terminal where sim_vehicle.py works and your
ArduPilot Python environment (pymavlink) is active:
    python3 /mnt/c/Users/isaia/fixed-wing-rl/ardupilot/run_batch.py \\
        /mnt/c/Users/isaia/fixed-wing-rl/localstore/batches/<name>/manifest.json
Options: --only ID [ID ...], --redo, --speedup N, --ardupilot DIR.
Needs no packages besides pymavlink (no numpy), so it runs in ArduPilot's environment.
"""
import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("MAVLINK20", "1")                 # talk MAVLink 2, like MAVProxy
from pymavlink import mavutil  # noqa: E402

SITL_HOST, SITL_PORT = "127.0.0.1", 5760   # SITL's first serial port (what MAVProxy normally connects to)
AUTO_MODE = 10                           # ArduPlane custom_mode for AUTO
DONE_HOLD_S = 30.0                       # simulated seconds in the final loiter before a flight is done
TIMEOUT_FACTOR = 2.0                     # give up after this x the expected simulated duration ...
TIMEOUT_EXTRA_S = 600.0                  # ... plus this many simulated seconds
MIN_SIM_RATE = 0.5                       # wall-clock cap assumes SITL runs at least this fast vs real time
CONNECT_TIMEOUT_S = 120.0               # wall seconds to wait for SITL to start
ARM_TIMEOUT_S = 300.0                   # wall seconds to keep trying to arm (EKF / GPS warm-up)
SILENCE_TIMEOUT_S = 60.0                # wall seconds without any message = the simulator died
CRASH_ALT_M = 5.0                        # below this (above home) after having flown = crashed


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def read_wpl(path):
    """QGC WPL 110 file -> list of (seq, current, frame, command, p1..p4, lat, lon, alt, autocontinue)."""
    lines = open(path).read().split("\n")
    if not lines or not lines[0].startswith("QGC WPL 110"):
        raise ValueError(f"{path} is not a QGC WPL 110 mission")
    items = []
    for ln in lines[1:]:
        f = ln.strip().split("\t")
        if len(f) < 12:
            continue
        items.append((int(f[0]), int(f[1]), int(f[2]), int(f[3]), float(f[4]), float(f[5]), float(f[6]),
                      float(f[7]), float(f[8]), float(f[9]), float(f[10]), int(f[11])))
    return items


def sitl_already_running():
    try:
        out = subprocess.run(["pgrep", "-f", "arduplane"], capture_output=True, text=True).stdout.strip()
    except FileNotFoundError:
        return False
    return bool(out)


def start_sitl(inst, parm, speedup, ardupilot_dir, out_path):
    exe = shutil.which("sim_vehicle.py") or str(ardupilot_dir / "Tools" / "autotest" / "sim_vehicle.py")
    cmd = [exe, "-v", "ArduPlane", "-f", "jsbsim:c172x", f"--custom-location={inst['home']}",
           f"--add-param-file={parm}", "-w", "--no-mavproxy", "--no-rebuild", f"--speedup={speedup}"]
    fh = open(out_path, "w")
    fh.write(" ".join(cmd) + "\n\n"); fh.flush()
    proc = subprocess.Popen(cmd, cwd=str(ardupilot_dir / "ArduPlane"), stdout=fh, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, start_new_session=True)
    return proc, fh


def stop_sitl(proc, fh):
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            break
        try:
            proc.wait(timeout=8)
            break
        except subprocess.TimeoutExpired:
            continue
    for name in ("arduplane", "JSBSim"):                 # anything the launcher left behind
        subprocess.run(["pkill", "-9", "-f", name], capture_output=True)
    fh.close()


def connect(tlog, proc, timeout):
    """Connect to SITL as soon as it listens (it waits for this connection before it starts)."""
    end = time.time() + timeout
    while time.time() < end:
        if proc.poll() is not None:
            return None                                          # the launcher exited: SITL failed
        try:
            return Link(tlog)
        except (OSError, ConnectionError):
            time.sleep(2)
    return None


class Link:
    """The MAVLink connection to SITL plus what we know about the aircraft."""

    def __init__(self, tlog_path):
        self.m = mavutil.mavlink_connection(f"tcp:{SITL_HOST}:{SITL_PORT}", source_system=255,
                                            source_component=190, autoreconnect=False)
        self.m.setup_logfile(str(tlog_path), mode="wb")    # every received message, MAVProxy .tlog format
        self.mode = self.armed = self.boot_s = self.alt_rel = self.seq = None
        self.max_alt = 0.0
        self.last_msg = time.time()
        self.last_hb_sent = 0.0
        self.texts = []

    def heartbeat(self):
        if time.time() - self.last_hb_sent >= 1.0:
            self.m.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
            self.last_hb_sent = time.time()

    def handle(self, msg):
        t = msg.get_type()
        if t == "BAD_DATA":
            return
        self.last_msg = time.time()
        if msg.get_srcSystem() != getattr(self.m, "target_system", 1):
            return
        if t == "HEARTBEAT" and msg.type == mavutil.mavlink.MAV_TYPE_FIXED_WING:
            self.mode = msg.custom_mode
            self.armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
        elif t == "GLOBAL_POSITION_INT":
            self.boot_s = msg.time_boot_ms / 1000.0
            if msg.lat or msg.lon:
                self.alt_rel = msg.relative_alt / 1000.0
                self.max_alt = max(self.max_alt, self.alt_rel)
        elif t == "MISSION_CURRENT":
            self.seq = msg.seq
        elif t == "STATUSTEXT":
            text = msg.text if isinstance(msg.text, str) else msg.text.decode(errors="replace")
            self.texts.append(text)
            self.texts = self.texts[-20:]

    def pump(self, seconds):
        """Read (and log) everything for `seconds` of wall time, keeping our heartbeat going."""
        end = time.time() + seconds
        while True:
            self.heartbeat()
            msg = self.m.recv_match(blocking=True, timeout=0.2)
            if msg is not None:
                self.handle(msg)
            if time.time() >= end:
                return

    def wait(self, types, timeout):
        """The next message of one of `types` (others are still handled and logged)."""
        end = time.time() + timeout
        while time.time() < end:
            self.heartbeat()
            msg = self.m.recv_match(blocking=True, timeout=0.2)
            if msg is None:
                continue
            self.handle(msg)
            if msg.get_type() in types and msg.get_srcSystem() == self.m.target_system:
                return msg
        return None

    def command(self, cmd, *params):
        p = list(params) + [0] * (7 - len(params))
        self.m.mav.command_long_send(self.m.target_system, self.m.target_component, cmd, 0, *p)

    def upload(self, items):
        n = len(items)
        for attempt in range(3):
            self.m.mav.mission_count_send(self.m.target_system, self.m.target_component, n, 0)
            sent = set()
            while True:
                msg = self.wait(("MISSION_REQUEST", "MISSION_REQUEST_INT", "MISSION_ACK"), 5.0)
                if msg is None:
                    break                                     # no answer: send the count again
                if msg.get_type() == "MISSION_ACK":
                    if msg.type == mavutil.mavlink.MAV_MISSION_ACCEPTED and len(sent) == n:
                        return True, ""
                    if msg.type != mavutil.mavlink.MAV_MISSION_ACCEPTED:
                        return False, f"mission rejected (MISSION_ACK type {msg.type})"
                    continue
                s = msg.seq
                if not 0 <= s < n:
                    continue
                (seq, cur, frame, cmd, p1, p2, p3, p4, lat, lon, alt, auto) = items[s]
                self.m.mav.mission_item_int_send(self.m.target_system, self.m.target_component, seq, frame, cmd,
                                                 cur, auto, p1, p2, p3, p4, int(round(lat * 1e7)),
                                                 int(round(lon * 1e7)), alt, 0)
                sent.add(s)
        return False, "no mission handshake from the aircraft"


def set_params(link, overrides):
    """Set the batch's parameter overrides and wait for the aircraft to confirm each one."""
    for name, value in overrides.items():
        for attempt in range(3):
            link.m.mav.param_set_send(1, 1, name.encode(), float(value), mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
            end = time.time() + 5.0
            while time.time() < end:
                msg = link.wait(("PARAM_VALUE",), end - time.time())
                pid = msg.param_id if msg is not None else ""
                pid = pid.decode(errors="replace") if isinstance(pid, bytes) else pid
                if msg is not None and pid.strip("\x00") == name and abs(msg.param_value - value) < 1e-3:
                    log(f"  set {name} = {value:g}")
                    break
            else:
                continue
            break
        else:
            return False, f"the aircraft did not accept {name} = {value:g}"
    return True, ""


def fly(inst, link, items, overrides=None):
    """Upload, arm, and follow the flight. Returns (status, detail)."""
    link.m.target_system, link.m.target_component = 1, 1
    link.m.mav.request_data_stream_send(1, 1, mavutil.mavlink.MAV_DATA_STREAM_ALL, 10, 1)
    link.m.mav.param_request_list_send(1, 1)               # the log records every parameter
    link.pump(5.0)
    if overrides:
        ok, detail = set_params(link, overrides)
        if not ok:
            return "param_failed", detail
    ok, detail = link.upload(items)
    if not ok:
        return "upload_failed", detail
    log(f"  mission uploaded ({len(items)} items); arming")

    end = time.time() + ARM_TIMEOUT_S
    while not link.armed:
        if time.time() > end:
            reasons = [t for t in link.texts if "arm" in t.lower() or "prearm" in t.lower()]
            return "arm_failed", "; ".join(reasons[-3:]) or "the aircraft never armed"
        link.command(mavutil.mavlink.MAV_CMD_DO_SET_MODE, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                     AUTO_MODE)
        link.pump(1.0)
        link.command(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 1)
        link.pump(4.0)
    arm_sim = link.boot_s or 0.0
    log("  armed in AUTO; flying")

    last_seq = len(items) - 1                                # the final loiter
    sim_limit = TIMEOUT_FACTOR * inst["expected_sim_s"] + TIMEOUT_EXTRA_S
    wall_end = time.time() + sim_limit / MIN_SIM_RATE
    in_loiter_since, last_report = None, 0.0
    while True:
        link.pump(1.0)
        now_sim = (link.boot_s or arm_sim) - arm_sim
        if time.time() - link.last_msg > SILENCE_TIMEOUT_S:
            return "sitl_died", f"no messages for {SILENCE_TIMEOUT_S:.0f} s (at {now_sim:.0f} s simulated)"
        if link.max_alt > 50.0 and link.alt_rel is not None and link.alt_rel < CRASH_ALT_M:
            return "crashed", f"altitude {link.alt_rel:.1f} m above home at {now_sim:.0f} s simulated"
        if link.max_alt > 50.0 and link.armed is False:
            return "crashed", f"disarmed in flight at {now_sim:.0f} s simulated"
        if link.seq == last_seq:
            in_loiter_since = now_sim if in_loiter_since is None else in_loiter_since
            if now_sim - in_loiter_since >= DONE_HOLD_S:
                return "completed", f"{now_sim:.0f} s simulated"
        if now_sim > sim_limit or time.time() > wall_end:
            return "timeout", f"at mission item {link.seq}/{last_seq} after {now_sim:.0f} s simulated"
        if time.time() - last_report > 30:
            alt = f"{link.alt_rel:.0f} m" if link.alt_rel is not None else "?"
            log(f"  item {link.seq}/{last_seq}, {now_sim:.0f} s simulated, {alt} above home")
            last_report = time.time()


def main():
    ap = argparse.ArgumentParser(description="Fly a Tier 3 batch in ArduPilot SITL, unattended.")
    ap.add_argument("manifest", help="the batch's manifest.json (WSL path)")
    ap.add_argument("--only", nargs="+", default=None, metavar="ID", help="fly only these instances")
    ap.add_argument("--redo", action="store_true", help="fly instances again even if already completed")
    ap.add_argument("--speedup", type=int, default=None, help="SITL speedup (default: the manifest's)")
    ap.add_argument("--ardupilot", default=str(Path.home() / "ardupilot"), help="ArduPilot source folder")
    args = ap.parse_args()

    manifest = json.load(open(args.manifest))
    speedup = args.speedup or manifest.get("speedup", 5)
    parm = manifest["parm_wsl"]
    ardupilot_dir = Path(args.ardupilot).expanduser()
    if not (ardupilot_dir / "ArduPlane").is_dir():
        sys.exit(f"{ardupilot_dir}/ArduPlane not found; pass --ardupilot <your ArduPilot folder>.")
    if not os.path.exists(parm):
        sys.exit(f"{parm} not found (the batch was made on another computer or the project moved).")
    if sitl_already_running():
        sys.exit("SITL (arduplane) is already running. Close it first (Ctrl-C in its MAVProxy window, "
                 "or: pkill -9 -f arduplane; pkill -9 -f JSBSim).")

    todo = [i for i in manifest["instances"] if i["status"] == "flyable"
            and (args.only is None or i["id"] in args.only)]
    log(f"Batch '{manifest['name']}': {len(todo)} instances to fly at --speedup {speedup}")
    if manifest.get("param_overrides"):
        log("  parameters for this batch: " + ", ".join(f"{k}={v:g}" for k, v in manifest["param_overrides"].items()))
    results = []
    for k, inst in enumerate(todo, 1):
        tlog = Path(inst["log_wsl"])
        run_json = tlog.with_suffix(".run.json")
        if run_json.exists() and not args.redo:
            prev = json.load(open(run_json))
            if prev.get("status") == "completed":
                log(f"[{k}/{len(todo)}] {inst['id']}: already completed, skipped (--redo to fly again)")
                results.append(prev)
                continue
        tlog.parent.mkdir(parents=True, exist_ok=True)
        log(f"[{k}/{len(todo)}] {inst['id']}: starting SITL at {inst['home']}")
        t0 = time.time()
        proc, fh = start_sitl(inst, parm, speedup, ardupilot_dir, tlog.with_suffix(".sitl.txt"))
        link = None
        try:
            link = connect(tlog, proc, CONNECT_TIMEOUT_S)
            if link is None:
                status, detail = "sitl_failed", f"SITL did not start; see {tlog.with_suffix('.sitl.txt').name}"
            else:
                hb = link.wait(("HEARTBEAT",), 60.0)
                if hb is None:
                    status, detail = "sitl_failed", "no heartbeat from SITL"
                else:
                    status, detail = fly(inst, link, read_wpl(inst["mission_wsl"]),
                                         manifest.get("param_overrides") or {})
        except KeyboardInterrupt:
            status, detail = "interrupted", "stopped by the user"
        except Exception as e:                                   # keep going with the next instance
            status, detail = "error", f"{type(e).__name__}: {e}"
        finally:
            if link is not None:
                try:
                    link.m.close()
                except Exception:
                    pass
            stop_sitl(proc, fh)
        rec = {"id": inst["id"], "status": status, "detail": detail, "wall_s": round(time.time() - t0, 1),
               "speedup": speedup, "param_overrides": manifest.get("param_overrides") or {},
               "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}
        json.dump(rec, open(run_json, "w"), indent=2)
        results.append(rec)
        log(f"[{k}/{len(todo)}] {inst['id']}: {status} ({detail}), {rec['wall_s']:.0f} s wall")
        if status == "interrupted":
            break
        time.sleep(3)

    log("Done: " + ", ".join(f"{r['id']} {r['status']}" for r in results))
    print(f"\nNext, in PyCharm's terminal:\n  python -m ardupilot.ardupilot_mission batch-analyze --name {manifest['name']}")


if __name__ == "__main__":
    main()
