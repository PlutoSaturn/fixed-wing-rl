#!/usr/bin/env python3
"""
test_ardupilot_mission.py - Checks for ardupilot_mission.py without ArduPilot.

Tier 1 (always run):
  * box writes a valid mission: home, takeoff, cruise speed, the four corners at
    the right place and altitude, the lap repeat (DO_JUMP) and return home;
  * the telemetry-log reader decodes MAVLink 1 and 2 packets (including MAVLink 2
    zero-trimmed payloads), skips other systems and corrupted packets, and keeps
    only the latest session of a log;
  * summary recovers a known airspeed, altitude offset, course offset, throttle
    and lap count from a synthetic box flight.
Tier 2 (run when localstore/ has tfr_env.json and tfr_results.json, i.e. after
the TFR plan has been solved):
  * make writes a geometrically sound mission for the SCP plan;
  * analyze recovers a known tracking offset, uses the latest session of a log,
    and detects a flight through the TFR.

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
    extra = {0: 50, 33: 104, 74: 20}[msgid] if extra is None else extra
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


def write_tlog(path, F, t, airspeed=50.0, corrupt_at=None, rel_alt=None, throttle=55):
    """F: local x, y, z (z above the workspace floor unless rel_alt is given)."""
    lat, lon = M.enu_to_geo(F[:, 0], F[:, 1])
    rel = F[:, 2] + M.FLOOR_AGL_M if rel_alt is None else rel_alt
    t0 = 1_700_000_000_000_000
    with open(path, "wb") as f:
        for k in range(len(F)):
            us = t0 + int(t[k] * 1e6)
            v2 = k % 3 != 0                                   # mix MAVLink 1 and 2
            pk = gpi(t[k] * 1000, lat[k], lon[k], rel[k], v2=v2)
            if corrupt_at is not None and k == corrupt_at:
                pk = pk[:-1] + bytes([pk[-1] ^ 0xFF])          # bad CRC: must be skipped
            f.write(us.to_bytes(8, "big") + pk)
            if k % 5 == 0:                                     # GCS heartbeat from system 255
                hb = packet(0, struct.pack("<IBBBBB", 0, 6, 8, 0, 0, 3), sysid=255)
                f.write((us + 10).to_bytes(8, "big") + hb)
                f.write((us + 20).to_bytes(8, "big") +        # a GLOBAL_POSITION_INT from another system
                        packet(33, struct.pack("<IiiiihhhH", 1, 0, 0, 0, 0, 0, 0, 0, 0), sysid=7))
            hud = packet(74, struct.pack("<ffffhH", airspeed, airspeed, rel[k], 0.0, 90, int(throttle)))
            f.write((us + 30).to_bytes(8, "big") + hud)


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
    plan = items[meta["first_plan_item"]:meta["first_plan_item"] + meta["n_plan_waypoints"]]
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


def synthetic_flight(lateral=15.0, vertical=-8.0):
    """Loiter far away, approach on the lead-in, fly the plan offset by a known
    amount, then exit: what a real mission looks like in the log."""
    env = load_environments(CASE[0])[CASE[2]]
    r = next(x for x in json.load(open(find_input(CASE[1]))) if x["env_index"] == CASE[2])
    t_p, xs = M.dense_plan(r)
    P, chi = xs[:, :3], xs[:, 5]
    nrm = np.c_[-np.sin(chi), np.cos(chi), np.zeros(len(chi))]
    plan_part = P + lateral * nrm + np.array([0, 0, vertical])
    u0 = np.r_[np.cos(chi[0]), np.sin(chi[0]), 0]
    approach = P[0] - np.linspace(3000, 0, 60, endpoint=False)[:, None] * u0 + lateral * nrm[0] + [0, 0, vertical]
    ang = np.linspace(0, 2 * np.pi, 120)
    loiter = P[0] - 4000 * u0 + 600 * np.c_[np.cos(ang), np.sin(ang), np.zeros_like(ang)]
    ug = np.r_[np.cos(chi[-1]), np.sin(chi[-1]), 0]
    exit_ = P[-1] + np.linspace(50, 3000, 60)[:, None] * ug
    F = np.vstack([loiter, approach, plan_part[::4], exit_])
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
    assert {"t_s", "x_east_m", "y_north_m", "z_up_m"} <= set(csv_head)   # works with tfr_benchmark --flight


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
