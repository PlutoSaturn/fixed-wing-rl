#!/usr/bin/env python3
"""
test_ardupilot_mission.py - Checks for ardupilot_mission.py without ArduPilot.

Tier 1 (always run):
  * box writes a valid mission: home, takeoff, cruise speed, the four corners at
    the right place and altitude, the lap repeat (DO_JUMP) and return home;
  * the telemetry-log reader decodes MAVLink 1 and 2 packets (including MAVLink 2
    zero-trimmed payloads), skips other systems and corrupted packets, and keeps
    only the latest session of a log; attitude and commanded roll share its clock;
  * oscillation() recovers the size and period of a known wobble;
  * summary recovers a known airspeed, altitude offset, course offset, throttle
    and lap count from a synthetic box flight.
Tier 2 (run when localstore/ has tfr_env.json and tfr_results.json, i.e. after
the TFR plan has been solved):
  * make writes a geometrically sound mission for the SCP plan;
  * analyze recovers a known tracking offset and a known roll/altitude wobble,
    uses the latest session of a log, and detects a flight through the TFR;
  * every analyzed flight is archived once in runs/, a different flight saved
    under an old name gets its own run folder, and runs.csv lists the runs in
    the order flown with the parameters changed between them.
Tier 3 (batches of generated environments):
  * the expected autopilot path cuts corners by the right amount and flags
    corners whose legs are too short for the turn;
  * the pass/fail rules (evaluate_flight) catch each kind of failure;
  * with the TFR plan: build_mission rejects a plan when the required clearance
    is raised above what the waypoint path keeps, and a whole batch (batch-make,
    a flown log, batch-analyze) gives PASS / NOT_SOLVED / NOT_FLOWN as expected.

Synthetic flights are written as .tlog files exactly as MAVProxy records them
(8-byte big-endian microsecond timestamp + raw MAVLink packet).

Run from the project root:  python -m ardupilot.test_ardupilot_mission
"""
import json
import os
import struct
import tempfile
from types import SimpleNamespace

import numpy as np

from ardupilot import ardupilot_mission as M
from environment.envgen import find_input, load_environments

CASE = ("tfr_env.json", "tfr_results.json", 0)


def have_tfr_plan():
    try:
        find_input(CASE[0]); find_input(CASE[1])
        return True
    except FileNotFoundError:
        return False


# --------------------------------------------------------------------------
# MAVLink encoding (test only)
# --------------------------------------------------------------------------
def packet(msgid, payload, sysid=1, compid=1, seq=0, v2=True, extra=None):
    extra = {0: 50, 33: 104, 74: 20, 30: 39, 62: 183, 22: 220}[msgid] if extra is None else extra
    if v2:
        body = bytes([len(payload), 0, 0, seq & 255, sysid, compid]) + msgid.to_bytes(3, "little") + payload
        return b"\xfd" + body + M._crc(body, extra).to_bytes(2, "little")
    body = bytes([len(payload), seq & 255, sysid, compid, msgid]) + payload
    return b"\xfe" + body + M._crc(body, extra).to_bytes(2, "little")


def gpi(t_ms, lat, lon, rel_alt, v2=True):
    p = struct.pack("<IiiiihhhH", int(t_ms), int(round(lat * 1e7)), int(round(lon * 1e7)),
                    int(round((rel_alt + 0) * 1000)), int(round(rel_alt * 1000)), 0, 0, 0, 0)
    if v2:
        p = p.rstrip(b"\x00") or b"\x00"            # MAVLink 2 trims trailing zero bytes
    return packet(33, p, v2=v2)


def write_tlog(path, F, t, airspeed=50.0, corrupt_at=None, rel_alt=None, throttle=55,
               roll=None, pitch=None, nav_roll=None, log_start_s=0.0, params=None, mode=None,
               param_sets=None):
    """F: local x, y, z (z above the workspace floor unless rel_alt is given).
    roll, pitch, nav_roll (deg, one per sample) add ATTITUDE and
    NAV_CONTROLLER_OUTPUT messages. log_start_s shifts the log (wall-clock) time,
    as for a later SITL session appended to the same log. params ({name: value})
    are sent first as PARAM_VALUE messages, as MAVProxy fetches them on connecting.
    mode (ArduPlane mode number per sample) adds the aircraft's HEARTBEAT;
    param_sets ({sample: (name, value)}) a parameter changed in flight."""
    lat, lon = M.enu_to_geo(F[:, 0], F[:, 1])
    rel = F[:, 2] + M.FLOOR_AGL_M if rel_alt is None else rel_alt
    t0 = 1_700_000_000_000_000 + int(log_start_s * 1e6)
    with open(path, "wb") as f:
        for n_, (name, value) in enumerate(sorted((params or {}).items())):
            pv = struct.pack("<fHH16sB", value, len(params), n_, name.encode(), 9)
            f.write((t0 - 1000 + n_).to_bytes(8, "big") + packet(22, pv))
        for k in range(len(F)):
            us = t0 + int(t[k] * 1e6)
            v2 = k % 3 != 0                                   # mix MAVLink 1 and 2
            pk = gpi(t[k] * 1000, lat[k], lon[k], rel[k], v2=v2)
            if corrupt_at is not None and k == corrupt_at:
                pk = pk[:-1] + bytes([pk[-1] ^ 0xFF])          # bad CRC: must be skipped
            f.write(us.to_bytes(8, "big") + pk)
            if param_sets and k in param_sets:                  # `param set` in flight: the new value is broadcast
                name, value = param_sets[k]
                pv = struct.pack("<fHH16sB", value, 1, 65535, name.encode(), 9)
                f.write((us + 2).to_bytes(8, "big") + packet(22, pv))
            if mode is not None:                                # the aircraft's heartbeat: fixed wing, its mode
                f.write((us + 5).to_bytes(8, "big") + packet(0, struct.pack("<IBBBBB", int(mode[k]), 1, 3, 0, 4, 3)))
            if k % 5 == 0:                                     # GCS heartbeat from system 255
                hb = packet(0, struct.pack("<IBBBBB", 0, 6, 8, 0, 0, 3), sysid=255)
                f.write((us + 10).to_bytes(8, "big") + hb)
                f.write((us + 20).to_bytes(8, "big") +        # a GLOBAL_POSITION_INT from another system
                        packet(33, struct.pack("<IiiiihhhH", 1, 0, 0, 0, 0, 0, 0, 0, 0), sysid=7))
            hud = packet(74, struct.pack("<ffffhH", airspeed, airspeed, rel[k], 0.0, 90, int(throttle)))
            f.write((us + 30).to_bytes(8, "big") + hud)
            if roll is not None:
                p_ = 0.0 if pitch is None else np.radians(pitch[k])
                att = struct.pack("<I6f", int(t[k] * 1000) + 3, np.radians(roll[k]), p_, 0, 0, 0, 0)
                f.write((us + 40).to_bytes(8, "big") + packet(30, att, v2=v2))
            if nav_roll is not None:
                nv = struct.pack("<5fhhH", nav_roll[k], 0, 0, 0, 0, 0, 0, 0)
                f.write((us + 50).to_bytes(8, "big") + packet(62, nv))


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def make_mission(tmp):
    envs, res, idx = CASE
    out = os.path.join(tmp, "m.waypoints")
    M.make(SimpleNamespace(envs=envs, results=res, index=idx, out=out))
    return out


def read_waypoints(path):
    lines = open(path).read().splitlines()
    assert lines[0] == "QGC WPL 110"
    rows = [l.split("\t") for l in lines[1:]]
    assert all(len(r) == 12 for r in rows)
    return [dict(seq=int(r[0]), frame=int(r[2]), cmd=int(r[3]), p=[float(v) for v in r[4:8]],
                 lat=float(r[8]), lon=float(r[9]), alt=float(r[10])) for r in rows]


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------
def tier2_mission_structure_and_geometry():
    with tempfile.TemporaryDirectory() as tmp:
        path = make_mission(tmp)
        meta = json.load(open(path + ".json"))
        items = read_waypoints(path)
    env = load_environments(CASE[0])[CASE[2]]
    r = next(x for x in json.load(open(find_input(CASE[1]))) if x["env_index"] == CASE[2])
    _, xs = M.dense_plan(r)
    P = xs[:, :3]
    assert [it["seq"] for it in items] == list(range(len(items)))
    assert items[0]["cmd"] == 16 and items[0]["frame"] == 0                        # home
    assert [it["cmd"] for it in items[1:5]] == [22, 178, 31, 16]                    # takeoff, speed, climb, lead-in
    assert items[-1]["cmd"] == 17 and all(it["frame"] == 3 for it in items[1:])
    lo, hi = env.bounds_lo, env.bounds_hi
    xy = lambda it: np.array(M.geo_to_enu(it["lat"], it["lon"]))
    # home, climb loiter and lead-in are outside the workspace; the loiter circle too
    for it in (items[0], items[3], items[4]):
        p = xy(it)
        assert not (lo[0] <= p[0] <= hi[0] and lo[1] <= p[1] <= hi[1]), it
    c = xy(items[3]); ang = np.linspace(0, 2 * np.pi, 360)
    ring = c + M.LOITER_RADIUS_M * np.c_[np.cos(ang), np.sin(ang)]
    assert not np.any(np.all((ring >= lo[:2]) & (ring <= hi[:2]), axis=1))
    # lead-in is aligned with the start heading
    d = P[0][:2] - xy(items[4])
    assert abs(np.degrees(np.arctan2(d[1], d[0]) - xs[0, 5])) < 0.5
    # plan waypoints: first = start, last = goal, altitudes = z + floor, spacing
    plan = [items[i] for i in meta["plan_items"]]                         # speed commands sit between them
    assert len(plan) == meta["n_plan_waypoints"] and all(it["cmd"] == 16 for it in plan)
    W = np.array([np.r_[xy(it), it["alt"] - meta["floor_agl_m"]] for it in plan])
    assert np.linalg.norm(W[0] - P[0]) < 0.01 and np.linalg.norm(W[-1] - P[-1]) < 0.01
    gaps = np.linalg.norm(np.diff(W, axis=0), axis=1)
    assert gaps[:-1].min() >= M.MIN_WP_SPACING_M - 1e-6 and gaps.max() <= M.MAX_WP_SPACING_M + 60
    dist, _ = M.point_to_polyline(W, P)
    assert dist.max() < 0.01                                               # waypoints lie on the plan
    chord, _ = M.point_to_polyline(P, W)
    assert chord.max() < 50                                                # legs stay near the plan
    # on the arc around the TFR the straight legs cut inside the curve by at most CHORD_TOL_M
    tfr = env.obstacles[0]
    legs = np.vstack([np.linspace(W[i], W[i + 1], 200) for i in range(len(W) - 1)])
    edge = np.linalg.norm(legs[:, :2] - tfr.center_xy, axis=1) - tfr.radius
    assert edge.min() >= env.clearance - M.CHORD_TOL_M - 0.5, edge.min()
    # climb-loiter altitude = start altitude
    assert abs(items[3]["alt"] - (P[0, 2] + meta["floor_agl_m"])) < 1e-3


# --------------------------------------------------------------------------
# Tier 1 tests
# --------------------------------------------------------------------------
def make_box(tmp, **kw):
    out = os.path.join(tmp, "box.waypoints")
    a = dict(heading=30.0, alt=300.0, length=M.BOX_LENGTH_M, width=M.BOX_WIDTH_M, laps=2, speed=None, out=out)
    a.update(kw)
    M.box(SimpleNamespace(**a))
    return out


def test_box_mission():
    with tempfile.TemporaryDirectory() as tmp:
        path = make_box(tmp)
        meta = json.load(open(path + ".json"))
        items = read_waypoints(path)
    assert [it["seq"] for it in items] == list(range(len(items)))
    assert [it["cmd"] for it in items] == [16, 22, 178, 16, 16, 16, 16, 16, 177, 20]
    assert abs(items[0]["lat"] - 37.0) < 1e-9 and abs(items[0]["lon"] + 120.0) < 1e-9     # home
    assert items[1]["alt"] == M.TAKEOFF_ALT_M
    assert abs(items[2]["p"][1] - meta["commanded_airspeed_ms"]) < 1e-9
    C = np.array(meta["corners_enu"])
    xy = lambda it: np.array(M.geo_to_enu(it["lat"], it["lon"]))
    W = np.array([xy(it) for it in items[3:8]])
    assert np.abs(W[:4] - C).max() < 0.01 and np.abs(W[4] - C[0]).max() < 0.01           # A B C D A
    assert all(it["alt"] == 300.0 for it in items[3:8])
    assert items[8]["p"][0] == 4 and items[8]["p"][1] == 1                                # jump to B, once more
    # geometry: A ahead of home along 30 deg, A->B along 30 deg, B->C to the right (120 deg)
    brg = lambda d: np.degrees(np.arctan2(d[0], d[1])) % 360
    assert abs(brg(C[0]) - 30) < 1e-6 and abs(np.linalg.norm(C[0]) - M.BOX_START_M) < 1e-6
    assert abs(brg(C[1] - C[0]) - 30) < 1e-6 and abs(brg(C[2] - C[1]) - 120) < 1e-6
    assert abs(np.linalg.norm(C[1] - C[0]) - M.BOX_LENGTH_M) < 1e-6
    assert abs(np.linalg.norm(C[2] - C[1]) - M.BOX_WIDTH_M) < 1e-6
    with tempfile.TemporaryDirectory() as tmp:
        assert [it["cmd"] for it in read_waypoints(make_box(tmp, laps=1))][-2:] == [16, 20]  # no jump


def synthetic_box_flight(meta, laps=2, offset=12.0, dz=6.0, dt=0.25, speed=49.5):
    """Take off, climb to the box, fly `laps` laps `offset` m to the right of each
    leg and `dz` m high, then head home. Returns local x, y, altitude above home, t."""
    C = np.array(meta["corners_enu"])
    # build the horizontal path, then altitude by phase
    path = [np.zeros(2), C[0]]
    for _ in range(laps):
        for k in range(4):
            a, b = C[k], C[(k + 1) % 4]
            u = (b - a) / np.linalg.norm(b - a)
            r = np.array([u[1], -u[0]])                                   # right of the leg
            path += [a + offset * r, b + offset * r]
    path.append(np.zeros(2))
    P = [path[0]]
    for a, b in zip(path[:-1], path[1:]):
        n = max(1, int(np.linalg.norm(b - a) / (speed * dt)))
        P += [a + s * (b - a) for s in np.linspace(0, 1, n + 1)[1:]]
    P = np.array(P)
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    z = np.minimum(s * 0.1, meta["alt_rel_m"] + dz)                       # climb out, then level
    return P, z, np.arange(len(P)) * dt


def test_summary_recovers_known_flight():
    with tempfile.TemporaryDirectory() as tmp:
        mission = make_box(tmp, heading=30.0)
        meta = json.load(open(mission + ".json"))
        P, z, t = synthetic_box_flight(meta, laps=2, offset=12.0, dz=6.0)
        log = os.path.join(tmp, "box.tlog")
        write_tlog(log, np.c_[P, z], t, airspeed=49.5, rel_alt=z, throttle=58)
        M.summary(SimpleNamespace(mission=mission, log=log, plot=False))
        s = json.load(open(os.path.join(tmp, "box_summary.json")))
    assert s["legs_measured"] == 7, s                                       # 2 laps, minus the climb leg
    assert abs(s["altitude_error_mean_m"] - 6.0) < 0.01 and abs(s["altitude_error_rms_m"] - 6.0) < 0.01, s
    assert abs(s["cross_track_rms_m"] - 12.0) < 0.05 and abs(s["cross_track_max_m"] - 12.0) < 0.05, s
    assert abs(s["airspeed_mean_ms"] - 49.5) < 1e-4 and abs(s["throttle_mean_pct"] - 58) < 1e-9, s
    assert 240 < s["settled_time_s"] < 255, s                               # 7 legs, settled parts only


def test_summary_rejects_no_takeoff():
    with tempfile.TemporaryDirectory() as tmp:
        mission = make_box(tmp)
        t = np.arange(400) * 0.25
        F = np.c_[np.zeros(400), np.zeros(400), np.zeros(400)]
        log = os.path.join(tmp, "ground.tlog")
        write_tlog(log, F, t, rel_alt=np.zeros(400))
        try:
            M.summary(SimpleNamespace(mission=mission, log=log, plot=False))
        except SystemExit as e:
            assert "did not" in str(e) and "take off" in str(e)
        else:
            raise AssertionError("expected a clear error for a flight that never took off")


def test_tlog_reader():
    rng = np.random.default_rng(1)
    F = np.c_[np.cumsum(rng.normal(0, 10, 300)), np.cumsum(rng.normal(0, 10, 300)), 100 + rng.normal(0, 5, 300)]
    t = np.arange(len(F)) * 0.5
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "f.tlog")
        write_tlog(p, F, t, airspeed=51.5, corrupt_at=100, throttle=61)
        f = M.read_tlog(p)
    assert len(f["t"]) == len(F) - 1                                      # the corrupted one is skipped
    keep = np.r_[0:100, 101:len(F)]
    x, y = M.geo_to_enu(f["lat"], f["lon"])
    assert np.abs(x - F[keep, 0]).max() < 0.02 and np.abs(y - F[keep, 1]).max() < 0.02
    assert np.abs(f["alt_rel"] - (F[keep, 2] + M.FLOOR_AGL_M)).max() < 0.002
    assert np.allclose(f["airspeed"], 51.5) and np.allclose(f["throttle"], 61)


def test_tlog_reader_attitude_on_same_clock():
    """ATTITUDE / NAV_CONTROLLER_OUTPUT come back in degrees, on the clock of "t",
    from the latest session only."""
    F = np.c_[np.arange(200) * 25.0, np.zeros(200), np.full(200, 100.0)]
    t = np.arange(200) * 0.25
    roll = 5 * np.sin(2 * np.pi * t / 4)
    with tempfile.TemporaryDirectory() as tmp:
        a, b, log = (os.path.join(tmp, n) for n in ("a.tlog", "b.tlog", "both.tlog"))
        write_tlog(a, F, t, roll=np.full(200, 40.0), nav_roll=np.full(200, 40.0))
        write_tlog(b, F, t, roll=roll, pitch=-roll / 2, nav_roll=roll / 2, log_start_s=600)
        open(log, "wb").write(open(a, "rb").read() + open(b, "rb").read())
        f = M.read_tlog(log)
    A, Nv = f["attitude"], f["nav"]
    assert np.abs(A[:, 1]).max() < 5.01                                     # old session dropped
    assert np.allclose(np.interp(A[:, 0], f["t"], f["t"]), A[:, 0])
    assert np.abs(A[:, 1] - 5 * np.sin(2 * np.pi * A[:, 0] / 4)).max() < 0.05   # same clock as t
    assert np.allclose(A[:, 2], -A[:, 1] / 2, atol=1e-4)
    assert np.abs(Nv[:, 1] - 2.5 * np.sin(2 * np.pi * Nv[:, 0] / 4)).max() < 0.05


def test_wobble_per_flight_mode():
    """A roll wobble only while in AUTO: wobble reports it for AUTO, not for FBWA,
    and splits FBWA where a parameter was changed in flight."""
    t = np.arange(0, 400, 0.25)
    F = np.c_[t * 50.0, np.zeros_like(t), np.full(len(t), 300.0)]
    auto = t < 200
    roll = np.where(auto, 5 * np.sin(2 * np.pi * t / 5.0), 0.0)
    with tempfile.TemporaryDirectory() as tmp:
        log = os.path.join(tmp, "w.tlog")
        write_tlog(log, F, t, roll=roll, mode=np.where(auto, 10, 5), params={"TECS_TIME_CONST": 5.0},
                   param_sets={1200: ("TECS_TIME_CONST", 8.0)})                       # at t = 300 s
        M.wobble(SimpleNamespace(log=log))
        rows = json.load(open(os.path.join(tmp, "w_wobble.json")))
    assert [r["mode"] for r in rows] == ["AUTO", "FBWA", "FBWA"], rows
    assert rows[2]["params_changed"] == "TECS_TIME_CONST 5->8" and abs(rows[2]["start_s"] - 300) < 1
    by = {r["mode"]: r for r in rows[:2]}
    assert abs(by["AUTO"]["roll_oscillation"] - 5 / np.sqrt(2)) < 0.3 and abs(by["AUTO"]["roll_period_s"] - 5) < 0.2
    assert by["FBWA"]["roll_oscillation"] < 0.05


def test_oscillation_measures_known_wobble():
    """A 4-unit, 5.3 s wobble on top of a slow trend: RMS 4/sqrt(2), period 5.3 s."""
    t = np.arange(0, 400, 0.25)
    y = 0.05 * t + 30 * np.sin(2 * np.pi * t / 300) + 4 * np.sin(2 * np.pi * t / 5.3)
    rms, period = M.oscillation(t, y)
    assert abs(rms - 4 / np.sqrt(2)) < 0.2, rms
    assert abs(period - 5.3) < 0.15, period
    assert M.oscillation(t[:20], y[:20]) == (None, None)                     # too short to tell


def test_resample_keeps_legs_close_to_curves():
    """On a gentle arc the legs are limited by CHORD_TOL_M; on straight parts by
    MAX_WP_SPACING_M; waypoints are never closer than MIN_WP_SPACING_M."""
    R = 5000.0
    ang = np.linspace(0, np.radians(60), 3000)
    arc = np.c_[R * np.sin(ang), R * (1 - np.cos(ang)), np.full(len(ang), 300.0)]
    line = arc[-1] + np.linspace(0, 6000, 2500)[1:, None] * np.r_[np.cos(ang[-1]), np.sin(ang[-1]), 0.0]
    P = np.vstack([arc, line])
    d = np.diff(P[:, :2], axis=0)
    chi = np.unwrap(np.r_[np.arctan2(d[:, 1], d[:, 0]), np.arctan2(d[-1, 1], d[-1, 0])])
    idx = M.resample(P, chi)
    W = P[idx]
    dev, _ = M.point_to_polyline(P, W)
    gaps = np.linalg.norm(np.diff(W, axis=0), axis=1)
    assert dev.max() <= M.CHORD_TOL_M + 0.05, dev.max()
    assert gaps[:-1].min() >= M.MIN_WP_SPACING_M - 1e-6
    assert gaps[:-1].max() <= M.MAX_WP_SPACING_M + 5
    assert gaps[-1] <= M.MAX_WP_SPACING_M + M.MIN_WP_SPACING_M            # a short final leg may be merged
    on_arc = gaps[: np.searchsorted(idx, len(arc)) - 1]
    assert on_arc.max() < np.sqrt(8 * R * M.CHORD_TOL_M) + 10                 # chord rule binds on the arc
    tight = M.resample(P, chi, chord_tol=1.0)                                # make --chord-tol 1
    dev1, _ = M.point_to_polyline(P, P[tight])
    assert dev1.max() <= 1.05 and len(tight) > len(idx), (dev1.max(), len(tight), len(idx))


def test_wsl_path():
    assert M.wsl_path(r"C:\Users\isaia\fixed-wing-rl\localstore\box.waypoints") == \
        "/mnt/c/Users/isaia/fixed-wing-rl/localstore/box.waypoints"
    assert M.wsl_path("/home/x/a.txt") == "/home/x/a.txt"


# --------------------------------------------------------------------------
# Tier 2 tests (need the solved TFR plan)
# --------------------------------------------------------------------------
def test_geo_round_trip():
    rng = np.random.default_rng(0)
    x, y = rng.uniform(-5e3, 30e3, 200), rng.uniform(-5e3, 20e3, 200)
    lat, lon = M.enu_to_geo(x, y)
    lat8, lon8 = np.round(lat, 8), np.round(lon, 8)                        # .waypoints precision
    x2, y2 = M.geo_to_enu(lat8, lon8)
    assert np.abs(x2 - x).max() < 0.01 and np.abs(y2 - y).max() < 0.01     # metres


def synthetic_flight(lateral=15.0, vertical=-8.0, wobble_m=0.0):
    """Loiter far away, approach on the lead-in, fly the plan offset by a known
    amount (plus an optional 5 s altitude wobble), then exit: what a real mission
    looks like in the log."""
    env = load_environments(CASE[0])[CASE[2]]
    r = next(x for x in json.load(open(find_input(CASE[1]))) if x["env_index"] == CASE[2])
    t_p, xs = M.dense_plan(r)
    P, chi = xs[:, :3], xs[:, 5]
    nrm = np.c_[-np.sin(chi), np.cos(chi), np.zeros(len(chi))]
    plan_part = P + lateral * nrm + np.array([0, 0, vertical])
    plan_part[:, 2] += wobble_m * np.sin(2 * np.pi * (t_p - t_p[0]) / 5.0)
    u0 = np.r_[np.cos(chi[0]), np.sin(chi[0]), 0]
    approach = P[0] - np.linspace(3000, 0, 60, endpoint=False)[:, None] * u0 + lateral * nrm[0] + [0, 0, vertical]
    ang = np.linspace(0, 2 * np.pi, 120)
    loiter = P[0] - 4000 * u0 + 600 * np.c_[np.cos(ang), np.sin(ang), np.zeros_like(ang)]
    ug = np.r_[np.cos(chi[-1]), np.sin(chi[-1]), 0]
    exit_ = P[-1] + np.linspace(50, 3000, 60)[:, None] * ug
    keep = np.searchsorted(t_p, np.arange(t_p[0], t_p[-1], 0.5))          # plan sampled every 0.5 s
    F = np.vstack([loiter, approach, plan_part[keep], exit_])
    t = np.arange(len(F)) * 0.5
    return env, r, F, t


def tier2_analyze_recovers_known_offset():
    env, r, F, t = synthetic_flight(lateral=15.0, vertical=-8.0)
    with tempfile.TemporaryDirectory() as tmp:
        mission = make_mission(tmp)
        log = os.path.join(tmp, "flight.tlog")
        write_tlog(log, F, t)
        M.analyze(SimpleNamespace(mission=mission, log=log, plot=False))
        s = json.load(open(os.path.join(tmp, "flight_summary.json")))
        csv_head = open(os.path.join(tmp, "flight_flight.csv")).readline().strip().split(",")
    assert s["reached_goal"]
    assert abs(s["max_horizontal_error_m"] - 15.0) < 1.5, s
    assert abs(s["max_vertical_error_m"] - 8.0) < 0.5, s
    assert abs(s["rms_error_m"] - np.hypot(15, 8)) < 1.5, s
    assert s["collision_free"] and s["min_clearance_m"] > 0
    assert s["altitude_oscillation_m"] < 0.3, s                          # a steady offset is not a wobble
    assert "roll_oscillation_deg" not in s                                # no ATTITUDE in this log
    assert {"t_s", "x_east_m", "y_north_m", "z_up_m"} <= set(csv_head)   # works with tfr_benchmark --flight


def tier2_analyze_recovers_known_wobble():
    """A 3 m, 5 s altitude wobble and a 6 deg, 5 s roll wobble (commanded 2 deg)."""
    env, r, F, t = synthetic_flight(lateral=0.0, vertical=0.0, wobble_m=3.0)
    roll = 6 * np.sin(2 * np.pi * t / 5.0)
    with tempfile.TemporaryDirectory() as tmp:
        mission = make_mission(tmp)
        log = os.path.join(tmp, "wob.tlog")
        write_tlog(log, F, t, roll=roll, pitch=roll / 3, nav_roll=roll / 3)
        M.analyze(SimpleNamespace(mission=mission, log=log, plot=False))
        s = json.load(open(os.path.join(tmp, "wob_summary.json")))
    assert abs(s["altitude_oscillation_m"] - 3 / np.sqrt(2)) < 0.25, s
    assert abs(s["altitude_oscillation_period_s"] - 5.0) < 0.2, s
    assert abs(s["roll_oscillation_deg"] - 6 / np.sqrt(2)) < 0.3, s
    assert abs(s["roll_oscillation_period_s"] - 5.0) < 0.2, s
    assert abs(s["commanded_roll_oscillation_deg"] - 2 / np.sqrt(2)) < 0.15, s
    assert abs(s["roll_minus_command_rms_deg"] - 4 / np.sqrt(2)) < 0.3, s
    assert 5.5 < s["max_abs_roll_deg"] <= 6.0 + 1e-6, s                  # 0.5 s samples may miss the peak


def tier2_analyze_archives_runs():
    """Re-analyzing a flight updates its run folder; a different flight saved under
    the same log name gets a new one; runs.csv shows what changed between runs."""
    env, r, F, t = synthetic_flight(lateral=5.0, vertical=0.0)
    base_params = {"TECS_TIME_CONST": 5.0, "RLL2SRV_TCONST": 0.25, "STAT_FLTTIME": 100.0}
    new_params = dict(base_params, TECS_TIME_CONST=8.0, STAT_FLTTIME=900.0)
    with tempfile.TemporaryDirectory() as tmp:
        mission = make_mission(tmp)
        log = os.path.join(tmp, "run.tlog")
        write_tlog(log, F, t, params=base_params)
        for _ in range(2):                                    # analyzing twice: still one run
            M.analyze(SimpleNamespace(mission=mission, log=log, plot=False, note="first"))
        runs = os.path.join(tmp, M.RUNS_DIR)
        assert sorted(os.listdir(runs)) == ["run", "runs.csv"]
        assert M.read_params(os.path.join(runs, "run", "params.parm"))["TECS_TIME_CONST"] == 5.0
        write_tlog(log, F, t, params=new_params, log_start_s=3600)   # a new flight over the same file
        M.analyze(SimpleNamespace(mission=mission, log=log, plot=False, note="second"))
        assert sorted(os.listdir(runs)) == ["run", "run_2", "runs.csv"]
        for d in ("run", "run_2"):
            assert {"flight.tlog", "summary.json", "flight.csv", "check.png", "params.parm"} <= \
                set(os.listdir(os.path.join(runs, d)))
        import csv
        rows = list(csv.DictReader(open(os.path.join(runs, "runs.csv"))))
    assert [x["run"] for x in rows] == ["run", "run_2"] and [x["note"] for x in rows] == ["first", "second"]
    assert rows[0]["changed_since_previous_run"] == ""
    assert rows[1]["changed_since_previous_run"] == "TECS_TIME_CONST 5->8"     # STAT_ ignored


def test_autopilot_path_cuts_corners():
    """A 90 deg corner between long legs is cut by R (1/cos 45 - 1); with legs too
    short for the turn, the turn tightens and needs more bank than allowed."""
    R, V = 500.0, 50.0
    W = np.array([[0, 0, 300], [2000, 0, 300], [2000, 2000, 300]], float)
    path, bank = M.autopilot_path(W, R, 45.0, V)
    cut = np.linalg.norm(path[:, :2] - W[1, :2], axis=1).min()
    assert abs(cut - R * (1 / np.cos(np.radians(45)) - 1)) < 2.0, cut
    assert abs(bank - np.degrees(np.arctan(V**2 / (9.81 * R)))) < 0.5
    short = np.array([[0, 0, 300], [300, 0, 300], [300, 300, 300]], float)
    _, bank_short = M.autopilot_path(short, R, 45.0, V)
    assert bank_short > 45.0, bank_short                                    # 150 m turn radius at 50 m/s


def test_param_overrides_parsing():
    assert M.parse_param_overrides(["TECS_SINK_MAX=7", "navl1_period = 15"]) == \
        {"TECS_SINK_MAX": 7.0, "NAVL1_PERIOD": 15.0}
    for bad in ("TECS_SINK_MAX", "TECS_SINK_MAX=fast", "=3"):
        try:
            M.parse_param_overrides([bad])
        except SystemExit:
            continue
        raise AssertionError(f"accepted {bad!r}")


def test_evaluate_flight_rules():
    good = {"samples_in_collision": 0, "min_clearance_m": 25.0, "left_workspace_m": 0.0, "goal_miss_m": 8.0,
            "max_error_m": 18.0, "rms_error_m": 4.0, "reached_goal": True, "flight_time_s": 470.0,
            "plan_flight_time_s": 458.0}
    ok, why, near = M.evaluate_flight(good, 30.0)
    assert ok and not why and not near
    bad = dict(good, samples_in_collision=3, min_clearance_m=-2.0, goal_miss_m=120.0, max_error_m=35.0,
               rms_error_m=12.0, flight_time_s=560.0)
    ok, why, near = M.evaluate_flight(bad, 30.0)
    assert not ok and near and len(why) == 5, why
    _, _, near = M.evaluate_flight(dict(good, min_clearance_m=10.0), 30.0)
    assert near                                                             # below half the margin


def tier2_speed_commands_follow_the_plan():
    """"plan" profile: after each plan waypoint W[k] (k >= 1) the speed command, when there is one, is the
    plan's mean airspeed over the leg W[k-1] -> W[k] (ArduPilot applies it while flying to W[k]);
    "constant": a single speed command."""
    from solvers.scp_aircraft import load_plan
    env, r, _ = load_plan(*CASE)
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "p.waypoints")
        m = M.build_mission(env, r, 1.0, "plan")
        M.write_mission(m, out, CASE[0], CASE[1], CASE[2], 1.0, r["flight_time"])
        items = read_waypoints(out)
        m_c = M.build_mission(env, r, 1.0, "constant")
    p = M.load_airframe()
    speeds = [it["p"][1] for it in items if it["cmd"] == 178]
    assert len(speeds) > 3 and all(p["V_MIN"] + 2 - 1e-6 <= v <= p["V_MAX"] - M.SPEED_HEADROOM_MS + 1e-6 for v in speeds)
    pi = m["plan_items"]
    checked = 0
    for k in range(1, len(pi)):
        nxt = items[pi[k] + 1] if pi[k] + 1 < len(items) else None
        if nxt is not None and nxt["cmd"] == 178:
            assert abs(nxt["p"][1] - m["leg_speeds"][k - 1]) < 1e-3
            checked += 1
    assert checked == len(speeds) - 1                                     # all but the one before takeoff
    assert sum(it[1] == 178 for it in m_c["items"]) == 1


def tier2_build_mission_rejects_unsafe_waypoint_path():
    from solvers.scp_aircraft import load_plan
    env, r, _ = load_plan(*CASE)
    m = M.build_mission(env, r, 1.0)
    assert not m["rejects"], m["rejects"]
    assert m["checks"]["est_min_clearance_m"] > 0.5 * env.margin
    old = M.FLY_MIN_CLEARANCE_FRAC
    try:
        M.FLY_MIN_CLEARANCE_FRAC = 1.5                                      # demand more than the path keeps
        m = M.build_mission(env, r, 1.0)
    finally:
        M.FLY_MIN_CLEARANCE_FRAC = old
    assert m["rejects"] and "passes" in m["rejects"][0], m["rejects"]


def tier2_batch_end_to_end():
    """batch-make on two copies of the TFR plus an unsolved one, a synthetic flight for
    the first, a failed arm for the second; batch-analyze scores them."""
    import csv
    env_list = json.load(open(find_input(CASE[0])))
    res = [x for x in json.load(open(find_input(CASE[1]))) if x["env_index"] == 0][0]
    with tempfile.TemporaryDirectory() as tmp:
        envs_f, res_f = os.path.join(tmp, "envs.json"), os.path.join(tmp, "res.json")
        json.dump([env_list[0]] * 3, open(envs_f, "w"))
        json.dump([dict(res, env_index=0), dict(res, env_index=1),
                   dict(res, env_index=2, success=False, status="infeasible")], open(res_f, "w"))
        old = M.BATCH_DIR
        M.BATCH_DIR = os.path.join(tmp, "batches")
        try:
            M.batch_make(SimpleNamespace(name="t", pair=[[envs_f, res_f]], chord_tol=1.0, speedup=5,
                                         note="test", overwrite=False, param=["tecs_sink_max=7"]))
            man = json.load(open(os.path.join(M.BATCH_DIR, "t", "manifest.json")))
            assert man["param_overrides"] == {"TECS_SINK_MAX": 7.0} and man["speed_profile"] == "constant"
            st = {i["id"]: i for i in man["instances"]}
            assert [st[k]["status"] for k in ("envs_0", "envs_1", "envs_2")] == ["flyable", "flyable", "not_solved"]
            env, r, F, t = synthetic_flight(lateral=5.0, vertical=0.0)
            write_tlog(st["envs_0"]["log"], F, t)
            json.dump({"status": "completed"}, open(st["envs_0"]["log"].replace(".tlog", ".run.json"), "w"))
            json.dump({"status": "arm_failed", "detail": "PreArm: test"},
                      open(st["envs_1"]["log"].replace(".tlog", ".run.json"), "w"))
            M.batch_analyze(SimpleNamespace(name="t", plot=False))
            rows = {x["id"]: x for x in csv.DictReader(open(os.path.join(M.BATCH_DIR, "t", "results.csv")))}
            summ = json.load(open(os.path.join(M.BATCH_DIR, "t", "summary.json")))
            assert os.path.exists(os.path.join(M.BATCH_DIR, "t", "summary.png"))
            assert os.path.exists(os.path.join(M.BATCH_DIR, "t", "runs", "envs_0", "flight.tlog"))
        finally:
            M.BATCH_DIR = old
    assert rows["envs_0"]["outcome"] == "PASS", rows["envs_0"]
    assert abs(float(rows["envs_0"]["rms_error_m"]) - 5.0) < 1.0
    assert rows["envs_1"]["outcome"] == "NOT_FLOWN" and "PreArm" in rows["envs_1"]["reasons"]
    assert rows["envs_2"]["outcome"] == "NOT_SOLVED"
    assert summ["pass"] == 1 and summ["not_flown"] == 1 and summ["not_solved"] == 1 and summ["pass_rate_of_flown"] == 1.0


def tier2_log_with_two_sessions_uses_latest():
    """An older session with a bad flight, then a good one: the good one is analyzed."""
    env, r, F, t = synthetic_flight(lateral=15.0, vertical=-8.0)
    _, xs = M.dense_plan(r)
    P = xs[:, :3]
    bad = np.vstack([P[0] - np.linspace(2000, 10, 40)[:, None] * [1, 0, 0], np.linspace(P[0], P[-1], 300)])
    with tempfile.TemporaryDirectory() as tmp:
        mission = make_mission(tmp)
        a, b, log = (os.path.join(tmp, n) for n in ("a.tlog", "b.tlog", "both.tlog"))
        write_tlog(a, bad, np.arange(len(bad)) * 0.5)
        write_tlog(b, F, t)                                   # its clock restarts from zero
        open(log, "wb").write(open(a, "rb").read() + open(b, "rb").read())
        M.analyze(SimpleNamespace(mission=mission, log=log, plot=False))
        s = json.load(open(os.path.join(tmp, "both_summary.json")))
    assert s["collision_free"] and abs(s["max_horizontal_error_m"] - 15.0) < 1.5


def tier2_analyze_ignores_low_pass_under_the_start():
    """The takeoff passes 270 m directly below the plan's start, then the aircraft
    loiters far away before flying the plan: the plan segment must start at the real
    pass, not the low one (this happened in the Tier 3 pilot batch)."""
    env, r, F, t = synthetic_flight(lateral=5.0, vertical=0.0)
    _, xs = M.dense_plan(r)
    P, chi = xs[:, :3], xs[:, 5]
    u0 = np.r_[np.cos(chi[0]), np.sin(chi[0]), 0]
    low = P[0] - [0, 0, 270] + np.linspace(-1500, 1500, 60)[:, None] * u0          # under the start
    away = P[0] - 4000 * u0 + 600 * np.c_[np.cos(np.linspace(0, 6.3, 120)), np.sin(np.linspace(0, 6.3, 120)),
                                         np.zeros(120)]
    F2 = np.vstack([low, away, F])
    with tempfile.TemporaryDirectory() as tmp:
        mission = make_mission(tmp)
        log = os.path.join(tmp, "low.tlog")
        write_tlog(log, F2, np.arange(len(F2)) * 0.5)
        M.analyze(SimpleNamespace(mission=mission, log=log, plot=False))
        s = json.load(open(os.path.join(tmp, "low_summary.json")))
    assert s["max_horizontal_error_m"] < 6.5 and s["rms_error_m"] < 6.0, s


def tier2_analyze_detects_collision():
    """A flight straight through the TFR must be reported as a collision."""
    env, r, F, t = synthetic_flight()
    _, xs = M.dense_plan(r)
    P = xs[:, :3]
    straight = np.linspace(P[0], P[-1], 400)
    F2 = np.vstack([P[0] - np.linspace(2000, 10, 40)[:, None] * [1, 0, 0], straight])
    with tempfile.TemporaryDirectory() as tmp:
        mission = make_mission(tmp)
        log = os.path.join(tmp, "bad.tlog")
        write_tlog(log, F2, np.arange(len(F2)) * 0.5)
        M.analyze(SimpleNamespace(mission=mission, log=log, plot=False))
        s = json.load(open(os.path.join(tmp, "bad_summary.json")))
    assert not s["collision_free"] and s["samples_in_collision"] > 0 and s["min_clearance_m"] < -1000


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    if have_tfr_plan():
        tests += [(n, f) for n, f in sorted(globals().items()) if n.startswith("tier2_") and callable(f)]
    else:
        print("  (Tier 2 tests skipped: solve the TFR plan first; see docs/Known_Solutions_Commands.md)")
    for name, fn in tests:
        fn()
        print(f"  ok  {name}")
    print(f"\nAll {len(tests)} tests passed.")
