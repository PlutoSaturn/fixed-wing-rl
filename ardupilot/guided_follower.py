#!/usr/bin/env python3
"""
guided_follower.py - the "companion computer": flies an SCP trajectory in ArduPlane GUIDED mode by
streaming setpoints, instead of uploading it as a waypoint mission.

How it works (run_batch.py calls on_position() for every GLOBAL_POSITION_INT, ~20 Hz simulated):
  * The aircraft takes off and climbs in AUTO (the usual mission items) and flies the lead-in
    towards the plan's start. Within ENGAGE_MARGIN_M + lookahead of the start it switches to GUIDED.
  * Each update it finds the aircraft's place on the plan (along-path distance s, a moving window so
    it never jumps ahead or back), and then sends:
      - roll: the L1 guidance law aimed at the plan point `lookahead_m` further along the CURVED path,
        a = 2 V^2 sin(eta) / L, roll = atan(a / g). On a curve of radius R this asks for V^2 / R, the
        bank the turn needs, so it does not fly outside curves the way straight waypoint legs do.
        Sent as SET_ATTITUDE_TARGET with only the roll used: pitch and throttle stay with TECS.
      - altitude: the plan's altitude at s (+ the climb lead), MAV_CMD_GUIDED_CHANGE_ALTITUDE, with a very
        large climb rate. ArduPlane moves the altitude target at most "rate x time since the last update"
        from the CURRENT altitude, and every command restarts that clock; with "as fast as possible"
        (1000 m/s) a command arriving a few ms before an update snapped the target back towards the
        aircraft, and TECS answered with a throttle bump about once a second (batches h0-h2).
      - airspeed: the plan's airspeed at s, MAV_CMD_GUIDED_CHANGE_SPEED, in small continuous steps,
        changed at once. (A ramp, speed_ramp, was tried in batches h0-h2: ArduPlane restarts the ramp at
        every command, so with commands several times a second the target barely moved and the
        aircraft stopped following the plan's speed.)
  * At the plan's goal it sets the mission to the exit waypoint and switches back to AUTO, which
    flies the exit and the final loiter as before (so run_batch.py's end-of-flight logic is unchanged).
If the stream stops, ArduPlane drops the forced roll after GUIDED_TIMEOUT (3 s) and circles the point
where GUIDED began; the batch then times out.

The reference trajectory (one sample every few metres, written by `batch-make --follower guided`)
holds lat, lon, the commanded altitude above home, the plan's altitude, airspeed and s. Before the
start and after the goal it is extended along straight lines so the lookahead point always exists.

Pure Python (no numpy), like run_batch.py, so it runs in ArduPilot's environment.
"""
import csv
import json
import math

GUIDED_MODE, AUTO_MODE = 15, 10                     # ArduPlane custom_mode numbers
# MAVLink numbers (written out so this file needs no pymavlink, e.g. for the unit tests)
MAV_CMD_DO_SET_MODE, MAV_MODE_FLAG_CUSTOM_MODE_ENABLED = 176, 1
MAV_CMD_GUIDED_CHANGE_SPEED, MAV_CMD_GUIDED_CHANGE_ALTITUDE = 43000, 43001
MAV_FRAME_GLOBAL_RELATIVE_ALT, MAV_RESULT_ACCEPTED = 3, 0
GRAVITY = 9.80665
R_EARTH = 6371000.0
ENGAGE_MARGIN_M = 100.0                            # switch to GUIDED this far + lookahead before the start
SEARCH_BACK, SEARCH_AHEAD = 20, 120                # projection window, in reference samples
MAX_ROLL_DEG = 40.0                                # below ROLL_LIMIT_DEG (45) so the command never saturates
SPEED_STEP_MS = 0.05                               # send a new airspeed when it changes by this much ...
SPEED_ACCEL_MIN, SPEED_ACCEL_MAX = 0.3, 2.0        # m/s^2: the ramp is 1.25 x the plan's acceleration, in these bounds
ALT_STEP_M = 0.1                                   # ... or a new altitude by this much (alt_step_m overrides) ...
ALT_RATE_MS = 0.0                                  # climb rate sent with each altitude (alt_rate_ms overrides; 0 = ArduPlane's 1000 m/s)
RESEND_S = 2.0                                     # ... or this long (simulated) after the last one
ROLL_TYPE_MASK = 0b11000110                        # SET_ATTITUDE_TARGET: use the roll only (see below)
# ArduPlane reads the roll from the quaternion when the roll-rate bit is "used" and the attitude bit is
# "ignored"; with the attitude bit used it would also force pitch and yaw. Ignored: pitch rate, yaw rate,
# throttle, attitude (bits 1, 2, 6, 7).


def wrap_pi(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class GuidedFollower:
    def __init__(self, ref_path, params, start_item, exit_item, log_path):
        ref = json.load(open(ref_path))
        self.lat0, self.lon0 = ref["lat"][ref["i_start"]], ref["lon"][ref["i_start"]]
        self.coslat0 = math.cos(math.radians(self.lat0))
        self.x, self.y = zip(*(self.enu(la, lo) for la, lo in zip(ref["lat"], ref["lon"])))
        self.alt_cmd, self.alt_plan, self.v, self.s = ref["alt_cmd"], ref["alt_plan"], ref["v"], ref["s"]
        self.i_start, self.i_end = ref["i_start"], ref["i_end"]
        self.n = len(self.s)
        self.L = float(params.get("lookahead_m", 150.0))
        self.speed_lead = float(params.get("speed_lead_m", 0.0))
        self.alt_lead = float(params.get("alt_lead_m", 0.0))
        self.speed_ramp = params.get("speed_ramp", False)       # False/absent: change the speed at once
        self.alt_rate = float(params.get("alt_rate_ms", ALT_RATE_MS))
        self.alt_step = float(params.get("alt_step_m", ALT_STEP_M))
        self.start_item, self.exit_item = start_item, exit_item
        self.state = "wait"                             # wait -> engaging -> active -> leaving -> done | failed
        self.k = None                                    # reference index of the last projection
        self.detail = ""
        self.last_v = self.last_alt = None
        self.t_v = self.t_alt = -1e9
        self.t_switch = -1e9
        self.n_rejected = 0
        self.log_fh = open(log_path, "w", newline="")
        self.log = csv.writer(self.log_fh)
        self.log.writerow(["t_boot_s", "state", "s_m", "xtrack_m", "eta_deg", "roll_cmd_deg", "v_cmd_ms",
                           "alt_cmd_m", "alt_m", "groundspeed_ms"])

    # ---- geometry
    def enu(self, lat, lon):
        return (math.radians(lon - self.lon0) * R_EARTH * self.coslat0, math.radians(lat - self.lat0) * R_EARTH)

    def project(self, px, py):
        """Nearest point on the reference polyline within the search window: (index, fraction, s, xtrack)."""
        if self.k is None:
            lo, hi = 0, min(self.n - 1, self.i_start + 100)     # first time: anywhere before or near the start
        else:
            lo, hi = max(0, self.k - SEARCH_BACK), min(self.n - 1, self.k + SEARCH_AHEAD)
        best = None
        for i in range(lo, hi):
            ax, ay, bx, by = self.x[i], self.y[i], self.x[i + 1], self.y[i + 1]
            dx, dy = bx - ax, by - ay
            l2 = dx * dx + dy * dy or 1e-9
            u = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / l2))
            qx, qy = ax + u * dx, ay + u * dy
            d2 = (px - qx) ** 2 + (py - qy) ** 2
            if best is None or d2 < best[0]:
                side = dx * (py - ay) - dy * (px - ax)          # > 0: aircraft left of the path
                best = (d2, i, u, math.copysign(math.sqrt(d2), side))
        _, i, u, xt = best
        self.k = i
        return i, u, self.s[i] + u * (self.s[i + 1] - self.s[i]), xt

    def at(self, s, arr):
        """Linear interpolation of a reference column at along-path distance s."""
        s = max(self.s[0], min(self.s[-1], s))
        i = max(0, min(self.n - 2, self.k if self.k is not None else 0))
        while i > 0 and self.s[i] > s:
            i -= 1
        while i < self.n - 2 and self.s[i + 1] < s:
            i += 1
        f = (s - self.s[i]) / ((self.s[i + 1] - self.s[i]) or 1e-9)
        return arr[i] + f * (arr[i + 1] - arr[i])

    # ---- commands
    def send_roll(self, link, roll_rad, t_boot_ms):
        q = [math.cos(roll_rad / 2.0), math.sin(roll_rad / 2.0), 0.0, 0.0]
        link.m.mav.set_attitude_target_send(t_boot_ms, 1, 1, ROLL_TYPE_MASK, q, 0, 0, 0, 0)

    def send_speed(self, link, v, accel=0.0):
        link.m.mav.command_int_send(1, 1, 0, MAV_CMD_GUIDED_CHANGE_SPEED, 0, 0,
                                    0, v, accel, 0, 0, 0, 0)   # airspeed, ramp (0 = change at once)

    def plan_accel(self, s, v):
        """The plan's acceleration magnitude at s, m/s^2 (dV/ds x V over +-25 m)."""
        dv = self.at(s + 25.0, self.v) - self.at(s - 25.0, self.v)
        return abs(dv / 50.0) * v

    def send_alt(self, link, alt_rel):
        link.m.mav.command_int_send(1, 1, MAV_FRAME_GLOBAL_RELATIVE_ALT, MAV_CMD_GUIDED_CHANGE_ALTITUDE, 0, 0,
                                    0, 0, self.alt_rate, 0, 0, 0, alt_rel)  # param3: climb rate limit, m/s

    def set_mode(self, link, mode):
        link.command(MAV_CMD_DO_SET_MODE, MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, mode)

    def on_ack(self, msg):
        if msg.command in (MAV_CMD_GUIDED_CHANGE_SPEED, MAV_CMD_GUIDED_CHANGE_ALTITUDE) \
                and msg.result != MAV_RESULT_ACCEPTED and self.state == "active":
            self.n_rejected += 1
            if self.n_rejected >= 5:
                self.state, self.detail = "failed", (f"ArduPlane rejected GUIDED speed/altitude commands "
                                                     f"(command {msg.command}, result {msg.result}); needs ArduPlane 4.3+")

    # ---- the loop, once per GLOBAL_POSITION_INT
    def on_position(self, link, msg):
        if self.state in ("done", "failed") or not (msg.lat or msg.lon):
            return
        t = msg.time_boot_ms / 1000.0
        px, py = self.enu(msg.lat * 1e-7, msg.lon * 1e-7)
        ve, vn = msg.vy / 100.0, msg.vx / 100.0
        gs = math.hypot(ve, vn)

        if self.state == "wait":
            if link.seq is not None and link.seq > self.start_item:
                self.state, self.detail = "failed", "passed the plan's start without engaging GUIDED"
                return
            d_start = math.hypot(px - self.x[self.i_start], py - self.y[self.i_start])
            if link.seq == self.start_item and d_start < self.L + ENGAGE_MARGIN_M:
                self.state, self.t_switch = "engaging", t
                self.set_mode(link, GUIDED_MODE)
            return

        if self.state == "engaging":
            if link.mode != GUIDED_MODE:
                if t - self.t_switch > 1.0:
                    self.set_mode(link, GUIDED_MODE)
                    self.t_switch = t
                return
            self.state = "active"

        if self.state == "active" and link.mode != GUIDED_MODE:
            self.state, self.detail = "failed", f"left GUIDED unexpectedly (mode {link.mode}) at {t:.0f} s"
            return

        i, u, s, xt = self.project(px, py)
        s_end = self.s[self.i_end]

        if self.state == "leaving":
            if link.mode == AUTO_MODE and link.seq == self.exit_item:
                self.state = "done"
                self.log_fh.close()
                return
            if t - self.t_switch > 1.0:
                link.m.mav.mission_set_current_send(1, 1, self.exit_item)
                self.set_mode(link, AUTO_MODE)
                self.t_switch = t
            return

        if s >= s_end:                                       # the goal: hand back to the mission
            self.state, self.t_switch = "leaving", t
            link.m.mav.mission_set_current_send(1, 1, self.exit_item)
            self.set_mode(link, AUTO_MODE)
            return

        # lateral: L1 law towards the point lookahead_m further along the curved path
        s_aim = s + self.L
        ax, ay = self.at(s_aim, self.x), self.at(s_aim, self.y)
        dist = max(math.hypot(ax - px, ay - py), 1.0)
        course = math.atan2(vn, ve) if gs > 5.0 else math.atan2(ay - py, ax - px)
        eta = wrap_pi(math.atan2(ay - py, ax - px) - course)          # > 0: aim point to the left
        eta = max(-math.pi / 2, min(math.pi / 2, eta))
        a_lat = 2.0 * max(gs, 20.0) ** 2 * math.sin(eta) / dist
        roll = -math.atan(a_lat / GRAVITY)                            # left turn = negative roll
        lim = math.radians(MAX_ROLL_DEG)
        roll = max(-lim, min(lim, roll))
        self.send_roll(link, roll, msg.time_boot_ms)

        # vertical and speed: the plan's values here (plus optional leads), sent when they change
        alt_cmd = self.at(s + self.alt_lead, self.alt_cmd)
        if self.last_alt is None or abs(alt_cmd - self.last_alt) >= self.alt_step or t - self.t_alt > RESEND_S:
            self.send_alt(link, alt_cmd)
            self.last_alt, self.t_alt = alt_cmd, t
        v_cmd = self.at(s + self.speed_lead, self.v)
        if self.last_v is None or abs(v_cmd - self.last_v) >= SPEED_STEP_MS or t - self.t_v > RESEND_S:
            accel = 0.0
            if self.speed_ramp:
                accel = max(SPEED_ACCEL_MIN, min(SPEED_ACCEL_MAX, 1.25 * self.plan_accel(s + self.speed_lead, v_cmd)))
            self.send_speed(link, v_cmd, accel)
            self.last_v, self.t_v = v_cmd, t

        self.log.writerow([f"{t:.2f}", self.state, f"{s:.1f}", f"{xt:.2f}", f"{math.degrees(eta):.2f}",
                           f"{math.degrees(roll):.2f}", f"{v_cmd:.2f}", f"{alt_cmd:.1f}",
                           f"{msg.relative_alt / 1000.0:.1f}", f"{gs:.1f}"])

    def close(self):
        try:
            self.log_fh.close()
        except Exception:
            pass
