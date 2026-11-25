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
      - airspeed: the plan's airspeed at s (+ the speed loop's correction, if on), MAV_CMD_GUIDED_CHANGE_SPEED, sent every update (round 4:
        sending it only on 0.05 m/s changes made the throttle bump), changed at once. (A ramp, speed_ramp, was tried in batches h0-h2: ArduPlane restarts the ramp at
        every command, so with commands several times a second the target barely moved and the
        aircraft stopped following the plan's speed.)
  * Speed loop (speed_ctrl, guided_test round 9): from the plan's start the follower keeps the plan's
    SCHEDULE, not only its airspeed. The plan's time to each point is T(s) = integral ds / v (its airspeed
    is its groundspeed: it was planned in still air). The schedule error e = s_sched(t) - s (m, > 0 =
    behind) adds a PI correction to the airspeed: v_cmd = v_plan(s) + kp e + ki integral(e dt), clamped to
    [speed_min_ms, speed_max_ms] (the integral stops growing while clamped). In a steady wind the integral
    settles at minus the along-track wind, so the aircraft flies the plan's groundspeed; it also removes
    the ~4% indicated-vs-true airspeed offset at altitude. Without it (round 8) a 10 m/s wind put the
    aircraft 39 s off the plan's schedule, and a tailwind left it below the plan on climbs.
    Round 9 ("pi"): timing fixed, but the aircraft ended up to 50 m ABOVE the plan in dense_1's steep
    descent: the loop asked for 38 m/s there, a steeper glide than the C172 can fly at idle while also
    slowing. Round 10 compares two safer forms (speed_ctrl "ff" and "hybrid"):
      - wind feedforward: the along-track wind is estimated as groundspeed - ArduPlane's airspeed (VFR_HUD,
        via on_airspeed), smoothed over WIND_TAU_S, and subtracted from the plan's airspeed. This also
        absorbs the indicated-vs-true airspeed offset, since it compares with the airspeed ArduPlane flies;
      - a small trim on the schedule error, at most +-TRIM_MAX_MS: proportional only ("ff"), or
        proportional + integral ("hybrid": the integral removes what the wind estimate misses, e.g. a
        biased airspeed sensor, and is a fallback if the estimate is lost);
      - altitude first: above the plan by more than ALT_GUARD_M the correction may not slow the aircraft,
        below it may not speed it up (faded out over the next ALT_GUARD_FADE_M); the integral holds meanwhile;
      - floor SPEED_MIN_MS 42 m/s (was 38).
    Round 10: timing fixed in every wind, but the aircraft still ended up 17-25 m above the plan in dense_1's
    steep descent (grade -10.5%) with no wind or a tailwind: the guard above reacts too late. Round 11 adds
    descent_guard: never command below the plan's airspeed while the plan ahead (DESCENT_LOOK_M) descends
    steeper than DESCENT_GRADE.
    Round 12 (hybrid + TECS_SINK_MAX 8): still 13-21 m high in the descent, and the logs showed the "altitude
    first" guard causing it: once > 3 m high it stepped the airspeed demand UP by 4-11 m/s in ~2 s, and asking
    for more speed while descending at idle sends TECS above the path. Round 13: alt_guard False (no guard)
    and speed_rate_ms2: the loop's correction (not the plan's own speed profile) may change at most this fast.
    Round 13: better, but still 10-21 m off in still air / tailwinds whenever the loop SLOWS the aircraft.
    Round 14: speed_up_only - the correction may only add airspeed (never below the plan's), the part that
    worked (headwind).
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
SPEED_STEP_MS = 0.0                                # send a new airspeed when it changes by this much (0: every
                                                   # update; 0.05 until guided_test round 4: 31% less throttle travel)
SPEED_ACCEL_MIN, SPEED_ACCEL_MAX = 0.3, 2.0        # m/s^2: the ramp is 1.25 x the plan's acceleration, in these bounds
ALT_STEP_M = 0.1                                   # ... or a new altitude by this much (alt_step_m overrides) ...
ALT_RATE_MS = 0.0                                  # climb rate sent with each altitude (alt_rate_ms overrides; 0 = ArduPlane's 1000 m/s)
SPEED_KP, SPEED_KI = 0.1, 0.005                    # speed loop "pi" (round 9): (m/s) per m of schedule error, and per m*s
TRIM_KP, TRIM_KI = 0.02, 0.0005                    # "ff"/"hybrid" trim on the schedule error (100 m -> 2 m/s)
TRIM_MAX_MS = 3.0                                  #   ... at most this much
WIND_TAU_S = 20.0                                  # wind estimate smoothing (simulated seconds)
ALT_GUARD_M, ALT_GUARD_FADE_M = 3.0, 5.0           # altitude first: see the docstring
DESCENT_GRADE, DESCENT_LOOK_M = 0.05, 500.0       # descent_guard: no slowing ahead of a descent steeper than 5%
SPEED_MIN_MS, SPEED_MAX_MS = 42.0, 59.0            # speed loop clamp (round 9 used 38); AIRSPEED_MIN 33, AIRSPEED_MAX 59
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
        self.speed_step = float(params.get("speed_step_ms", SPEED_STEP_MS))   # 0: send every update
        # speed_hold (test only, guided_test round 4): one constant airspeed, the plan's mean, so no speed
        # changes are sent during the plan
        seg = self.v[self.i_start:self.i_end + 1]
        self.v_hold = sum(seg) / len(seg) if params.get("speed_hold", False) and seg else None
        # robustness tests (guided_test round 6), both in simulated seconds, 0 = off:
        #   cmd_delay_s - every roll/altitude/airspeed setpoint reaches ArduPlane this much later
        #   pos_delay_s - the follower steers from the position reported this long ago
        self.cmd_delay = float(params.get("cmd_delay_s", 0.0))
        self.pos_delay = float(params.get("pos_delay_s", 0.0))
        self.pending = []                                # delayed setpoints: (due time, send function, args)
        # speed loop (see the docstring): plan time to each reference sample, 0 at the plan's start
        mode = params.get("speed_ctrl", False)               # False/"off", True/"pi" (round 9), "ff", "hybrid"
        self.speed_ctrl = {True: "pi", False: None, "off": None}.get(mode, mode)
        pi = self.speed_ctrl == "pi"
        self.kp = float(params.get("speed_kp", SPEED_KP if pi else TRIM_KP))
        self.ki = float(params.get("speed_ki", SPEED_KI if pi else (TRIM_KI if self.speed_ctrl == "hybrid" else 0.0)))
        self.v_min = float(params.get("speed_min_ms", 38.0 if pi else SPEED_MIN_MS))
        self.descent_guard = bool(params.get("descent_guard", False))
        self.alt_guard = bool(params.get("alt_guard", True))        # round 13: False
        self.corr_rate = float(params.get("speed_rate_ms2", 0.0))   # 0 = no limit (rounds 10-12)
        self.up_only = bool(params.get("speed_up_only", False))     # round 14: never below the plan's airspeed
        self.airspeed = None                             # ArduPlane's airspeed (VFR_HUD), for the wind estimate
        self.wind = None                                 # along-track wind estimate, m/s (> 0 = tailwind)
        self.v_corr = 0.0
        self.v_max = float(params.get("speed_max_ms", SPEED_MAX_MS))
        self.T = [0.0] * self.n
        for k in range(self.i_start + 1, self.n):
            self.T[k] = self.T[k - 1] + (self.s[k] - self.s[k - 1]) / max(self.v[k], 1.0)
        for k in range(self.i_start - 1, -1, -1):
            self.T[k] = self.T[k + 1] - (self.s[k + 1] - self.s[k]) / max(self.v[k + 1], 1.0)
        self.t0 = None                                   # time the aircraft passed the plan's start
        self.e_int, self.t_prev, self.e_sched = 0.0, None, 0.0
        self.pos_buf = []                                # recent positions, for pos_delay_s
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
                           "alt_cmd_m", "alt_m", "groundspeed_ms", "v_plan_ms", "sched_err_m", "airspeed_ms",
                           "wind_est_ms"])

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

    def s_sched(self, t_plan):
        """Along-path distance the plan reaches at plan time t_plan (bisection on T)."""
        lo, hi = 0, self.n - 1
        if t_plan <= self.T[lo]:
            return self.s[lo]
        if t_plan >= self.T[hi]:
            return self.s[hi]
        while hi - lo > 1:
            mid = (lo + hi) // 2
            lo, hi = (mid, hi) if self.T[mid] <= t_plan else (lo, mid)
        f = (t_plan - self.T[lo]) / ((self.T[hi] - self.T[lo]) or 1e-9)
        return self.s[lo] + f * (self.s[hi] - self.s[lo])

    def on_airspeed(self, airspeed):
        """VFR_HUD airspeed from run_batch.py (the airspeed ArduPlane controls)."""
        self.airspeed = airspeed

    def speed_correction(self, t, s, gs=None, alt=None):
        """Speed loop, from the plan's start; returns (correction m/s, v_plan). "pi": PI on the schedule
        error (round 9). "ff"/"hybrid": wind feedforward + a limited trim, altitude first (round 10)."""
        v_plan = self.at(s + self.speed_lead, self.v)
        dt = 0.0 if self.t_prev is None else max(0.0, min(1.0, t - self.t_prev))
        self.t_prev = t
        if self.speed_ctrl != "pi":
            return self.speed_correction_ff(t, s, dt, v_plan, gs, alt)
        if self.t0 is None:
            if s < 0.0:
                return 0.0, v_plan
            self.t0 = t - self.at(s, self.T)                 # plan clock starts as the aircraft passes s = 0
        self.e_sched = self.s_sched(t - self.t0) - s
        corr = self.kp * self.e_sched + self.ki * self.e_int
        v = v_plan + corr
        if self.v_min < v < self.v_max or (v <= self.v_min and self.e_sched > 0) or (v >= self.v_max and self.e_sched < 0):
            self.e_int += self.e_sched * dt                  # anti-windup: only integrate towards the range
        return corr, v_plan

    def steep_descent_ahead(self, s, step=50.0):
        """True if the plan descends steeper than DESCENT_GRADE anywhere in the next DESCENT_LOOK_M."""
        a0, x = self.at(s, self.alt_plan), s
        while x < s + DESCENT_LOOK_M:
            a1 = self.at(x + step, self.alt_plan)
            if (a1 - a0) / step < -DESCENT_GRADE:
                return True
            a0, x = a1, x + step
        return False

    def speed_correction_ff(self, t, s, dt, v_plan, gs, alt):
        if self.airspeed is not None and gs is not None:     # wind estimate, from GUIDED on (converged by s = 0)
            w = gs - self.airspeed
            self.wind = w if self.wind is None else self.wind + min(1.0, dt / WIND_TAU_S) * (w - self.wind)
        if self.t0 is None:
            if s < 0.0:
                return 0.0, v_plan
            self.t0 = t - self.at(s, self.T)
        self.e_sched = self.s_sched(t - self.t0) - s
        trim = self.kp * self.e_sched + self.ki * self.e_int
        trim_c = max(-TRIM_MAX_MS, min(TRIM_MAX_MS, trim))
        corr = trim_c - (self.wind or 0.0)
        # altitude first: no slowing while above the plan, no speeding up while below
        guard = 1.0
        if alt is not None and self.alt_guard:
            dz = alt - self.at(s, self.alt_plan)
            if (dz > ALT_GUARD_M and corr < 0) or (dz < -ALT_GUARD_M and corr > 0):
                guard = max(0.0, 1.0 - (abs(dz) - ALT_GUARD_M) / ALT_GUARD_FADE_M)
        corr *= guard
        if self.descent_guard and corr < 0 and self.steep_descent_ahead(s):
            corr, guard = 0.0, 0.0                       # holds the integral too
        limited = False
        if self.up_only and corr < 0:
            corr, limited = 0.0, True                        # integral only moves towards "less negative"
        if self.corr_rate > 0:                               # no jumps: the correction moves at most this fast
            step = self.corr_rate * dt
            if abs(corr - self.v_corr) > step:
                corr, limited = self.v_corr + math.copysign(step, corr - self.v_corr), True
            self.v_corr = corr
        v = v_plan + corr
        free = self.v_min < v < self.v_max and trim == trim_c and guard == 1.0 and not limited
        toward = (trim > 0) != (self.e_sched > 0)                # integrating would shrink |trim|
        if free or toward:
            self.e_int += self.e_sched * dt
        return corr, v_plan

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

    def emit(self, link, t, fn, *args):
        """Send a setpoint now, or cmd_delay_s later (flushed by on_position, simulated time)."""
        if self.cmd_delay > 0:
            self.pending.append((t + self.cmd_delay, fn, args))
        else:
            fn(link, *args)

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
        if self.pending:                                 # cmd_delay_s: send what is due (only while active)
            due = [p for p in self.pending if p[0] <= t + 1e-9]
            self.pending = [p for p in self.pending if p[0] > t + 1e-9]
            if self.state == "active":
                for _, fn, args in due:
                    fn(link, *args)
        if self.pos_delay > 0:                           # pos_delay_s: steer from an old position
            self.pos_buf.append((t, px, py, ve, vn))
            while len(self.pos_buf) > 1 and self.pos_buf[1][0] <= t - self.pos_delay:
                self.pos_buf.pop(0)
            _, px, py, ve, vn = self.pos_buf[0]
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
            self.pending = []
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
        self.emit(link, t, self.send_roll, roll, msg.time_boot_ms)

        # vertical and speed: the plan's values here (plus optional leads), sent when they change
        alt_cmd = self.at(s + self.alt_lead, self.alt_cmd)
        if self.last_alt is None or abs(alt_cmd - self.last_alt) >= self.alt_step or t - self.t_alt > RESEND_S:
            self.emit(link, t, self.send_alt, alt_cmd)
            self.last_alt, self.t_alt = alt_cmd, t
        v_cmd = self.v_hold if self.v_hold is not None else self.at(s + self.speed_lead, self.v)
        v_plan = v_cmd
        if self.speed_ctrl and self.v_hold is None:
            corr, v_plan = self.speed_correction(t, s, gs, msg.relative_alt / 1000.0)
            v_cmd = max(self.v_min, min(self.v_max, v_plan + corr))
        if self.last_v is None or abs(v_cmd - self.last_v) >= self.speed_step or t - self.t_v > RESEND_S:
            accel = 0.0
            if self.speed_ramp:
                accel = max(SPEED_ACCEL_MIN, min(SPEED_ACCEL_MAX, 1.25 * self.plan_accel(s + self.speed_lead, v_cmd)))
            self.emit(link, t, self.send_speed, v_cmd, accel)
            self.last_v, self.t_v = v_cmd, t

        self.log.writerow([f"{t:.2f}", self.state, f"{s:.1f}", f"{xt:.2f}", f"{math.degrees(eta):.2f}",
                           f"{math.degrees(roll):.2f}", f"{v_cmd:.2f}", f"{alt_cmd:.1f}",
                           f"{msg.relative_alt / 1000.0:.1f}", f"{gs:.1f}", f"{v_plan:.2f}",
                           f"{self.e_sched:.1f}", "" if self.airspeed is None else f"{self.airspeed:.2f}",
                           "" if self.wind is None else f"{self.wind:.2f}"])

    def close(self):
        try:
            self.log_fh.close()
        except Exception:
            pass
