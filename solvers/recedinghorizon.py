#!/usr/bin/env python3
"""
receding_horizon.py - short-horizon (receding-horizon / MPC-style) planning
with either SCP discretization, benchmarked against the full-horizon solves.

How it flies one environment
----------------------------
1. Guide path: a shortest route through a coarse 3D grid (obstacles grown by
   the planning clearance plus GUIDE_EXTRA_CLEARANCE_M) whose altitude levels
   are spaced so the route can never climb or descend faster than the
   aircraft can, shortcut by line of sight. It only tells the short-horizon
   planner which way to go around obstacles; it is not flown.
2. Every EXECUTE_S seconds, solve a HORIZON_S-second problem from the current
   state: same aircraft model, limits and obstacle constraints as the
   full-horizon solvers, but with a fixed duration and a free end position
   pulled toward the point on the guide path HORIZON_S seconds ahead
   (terminal cost TERMINAL_WEIGHT * ||p_end - target||^2 / POS_SCALE^2).
   The previous plan, shifted forward, is the warm start; the guide path is
   the fallback guess.
3. Fly the first EXECUTE_S seconds of the plan (the true dynamics integrated
   with the plan's controls), then replan.
4. Final approach: once the goal is within FINAL_APPROACH_S of flying along
   the guide, solve the normal problem (free final time, end exactly at the
   goal) from the current state and fly it to the end.
If a replan fails, the aircraft keeps flying the previous plan.

Methods
-------
  shooting  multiple shooting (scp_aircraft.py), SHOOT_NODES nodes per horizon
  birkhoff  Birkhoff, one global segment of degree BK_HORIZON_DEGREE per horizon

Reported per environment and method: reached goal, collision-free (body) and
safety-margin intrusion along the FLOWN path, flight time vs the full-horizon
solution, number of replans and fallbacks, and solve time per replan (mean,
95th percentile, max) and in total.

Usage (from the project root)
    python -m solvers.receding_horizon --envs bench_envs.json
    python -m solvers.receding_horizon --envs bench_envs.json --baseline   # also time full-horizon solves
"""
from __future__ import annotations

import argparse
import json
import time
import warnings

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra

from solvers import scp_aircraft as base
from solvers import scp_birkhoff as bkmod
from solvers.scp_aircraft import AircraftSCP
from solvers.scp_birkhoff import BirkhoffAircraftSCP, BirkhoffBasis
from environment.envgen import load_environments, localstore_path


# ==========================================================================
# USER SETTINGS
# ==========================================================================
ENV_FILE          = "c172_envs.json"
RESULTS_FILE      = "receding_horizon_results.json"
METHODS           = ("shooting", "birkhoff")

HORIZON_S         = 25.0      # length of each short-horizon plan, s
EXECUTE_S         = 5.0       # flown before replanning, s
FINAL_APPROACH_S  = 32.0      # switch to the to-the-goal solve when this close (flying time along the guide)
TERMINAL_WEIGHT   = 20.0      # pull of the horizon end toward the guide-path target
TARGET_SPEED      = None      # the target is TARGET_SPEED * HORIZON_S ahead along the guide
                              #   (None = halfway between V_CRUISE and V_MAX, so the planner is pushed
                              #   to make progress, as the time cost does in the full-horizon problem)
ANGLE_WEIGHT      = 0.0       # pull of the horizon-end heading / climb angle toward the guide direction
                              #   there (0 = off). Ending each plan pointed along the obstacle-free guide
                              #   leaves room to keep going; without it a plan can end aimed at an obstacle
                              #   too close to turn away from (a fixed-wing cannot stop).
SHOOT_NODES       = 11        # shooting nodes per horizon (2.5 s intervals)
BK_HORIZON_DEGREE = 12        # Birkhoff: one segment of this degree per horizon (13 nodes)
RH_MAX_ITERS      = 30        # SCP iteration cap per replan
RH_W_TRUST        = 1.0       # PTR step penalty for horizon solves. Small values (e.g. 0.03) let steps
                              #   hit the cap every iteration and the solver can oscillate between two
                              #   trajectories without converging
MAX_REPLANS       = 200

GUIDE_RES_M       = 100.0     # guide-path grid resolution (horizontal; vertical follows the climb limit)
GUIDE_EXTRA_CLEARANCE_M = 0.0   # extra room around obstacles for the guide; the grid already adds
                                #   half a cell, and the planner enforces the real clearance
GUIDE_SLOPE_FRACTION = 0.8    # guide shortcuts may climb/descend at most this share of GAMMA_MAX
EXEC_DT           = 0.05      # s, integration step when flying a plan


# --------------------------------------------------------------------------
# guide path: grid shortest path + line-of-sight shortcutting
# --------------------------------------------------------------------------
def _max_slope():
    return np.tan(GUIDE_SLOPE_FRACTION * np.radians(base.GAMMA_MAX_DEG))


def _segment_free(env, a, b, d, step):
    dxy = np.linalg.norm((b - a)[:2])
    if abs(b[2] - a[2]) > _max_slope() * max(dxy, 1e-9):
        return False                                     # steeper than the aircraft can climb
    n = max(int(np.ceil(np.linalg.norm(b - a) / step)), 2)
    P = np.linspace(a, b, n)
    return bool(np.all(env.constraint_values(P, d=d) >= 0))


def guide_path(env, res=GUIDE_RES_M, extra=GUIDE_EXTRA_CLEARANCE_M):
    """Try the guide with GUIDE_EXTRA_CLEARANCE_M, relaxing it if the corridor
    is blocked at that clearance."""
    for ex in dict.fromkeys((extra, extra / 2, 0.0)):         # relax, without repeating
        try:
            return _guide_path(env, res, ex)
        except RuntimeError:
            continue
    raise RuntimeError("guide path: start and goal are not connected")


def _guide_path(env, res, extra):
    """Shortest 3D route on a climb-limited grid: altitude levels are spaced so
    that one horizontal step can change altitude by at most one level, i.e. the
    route never climbs or descends faster than GUIDE_SLOPE_FRACTION * GAMMA_MAX.
    Returns (points, arclength)."""
    lo, hi = env.position_bounds()
    z_res = res * _max_slope()
    gx = np.arange(lo[0] + res / 2, hi[0], res)
    gy = np.arange(lo[1] + res / 2, hi[1], res)
    gz = np.arange(lo[2] + z_res / 2, hi[2], z_res)
    shape = (len(gx), len(gy), len(gz))
    P = np.stack(np.meshgrid(gx, gy, gz, indexing="ij"), axis=-1).reshape(-1, 3)
    d_grid = env.clearance + extra + np.hypot(res * np.sqrt(2) / 2, z_res / 2)
    free = np.ones(len(P), bool)
    for i in range(0, len(P), 20000):
        free[i:i + 20000] = np.all(env.constraint_values(P[i:i + 20000], d=d_grid) >= 0, axis=0)
    idx = np.arange(len(P)).reshape(shape)
    rows, cols, w = [], [], []
    for ox, oy in ((1, 0), (0, 1), (1, 1), (1, -1)):
        for oz in (-1, 0, 1):
            off = (ox, oy, oz)
            sl0 = tuple(slice(max(0, -o), n - max(0, o)) for o, n in zip(off, shape))
            sl1 = tuple(slice(max(0, o), n - max(0, -o)) for o, n in zip(off, shape))
            i0, i1 = idx[sl0].ravel(), idx[sl1].ravel()
            ok = free[i0] & free[i1]
            rows.append(i0[ok]); cols.append(i1[ok])
            w.append(np.full(ok.sum(), np.hypot(res * np.hypot(ox, oy), z_res * oz)))
    Gr = sp.csr_matrix((np.concatenate(w), (np.concatenate(rows), np.concatenate(cols))),
                       shape=(len(P), len(P)))
    free_idx = np.nonzero(free)[0]
    # nearest free node, counting altitude differences as the distance needed to climb them
    scale = np.array([1.0, 1.0, 1.0 / _max_slope()])
    near = lambda p: free_idx[np.argmin(np.linalg.norm((P[free_idx] - p) * scale, axis=1))]
    g, s0 = near(env.goal_pos), near(env.start_pos)
    dist, pred = dijkstra(Gr, directed=False, indices=g, return_predecessors=True)
    if not np.isfinite(dist[s0]):
        raise RuntimeError("guide path: start and goal are not connected")
    path = [s0]
    while path[-1] != g:
        path.append(pred[path[-1]])
    pts = np.vstack([env.start_pos, P[path], env.goal_pos])
    # shortcut: from each point jump to the furthest point in line of sight (and not too steep)
    d_los = env.clearance + extra / 2
    out, i = [pts[0]], 0
    while i < len(pts) - 1:
        j = len(pts) - 1
        while j > i + 1 and not _segment_free(env, pts[i], pts[j], d_los, res / 3):
            j -= 1
        out.append(pts[j])
        i = j
    pts = np.array(out)
    return pts, np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]


def project_on_path(pts, s_cum, p, s_min=0.0):
    """Arclength of the closest point of the polyline to p (not behind s_min - 200 m)."""
    best_s, best_d = s_min, np.inf
    for k in range(len(pts) - 1):
        a, b = pts[k], pts[k + 1]
        L = s_cum[k + 1] - s_cum[k]
        t = np.clip((p - a) @ (b - a) / max(L * L, 1e-9), 0, 1)
        d = np.linalg.norm(a + t * (b - a) - p)
        s = s_cum[k] + t * L
        if d < best_d and s >= s_min - 200:
            best_d, best_s = d, s
    return best_s


def point_at(pts, s_cum, s):
    s = np.clip(s, 0, s_cum[-1])
    return np.array([np.interp(s, s_cum, pts[:, k]) for k in range(3)])


def direction_at(pts, s_cum, s, chi_now):
    """(gamma, chi) of the guide path at arclength s, chi unwrapped next to chi_now."""
    k = int(np.clip(np.searchsorted(s_cum, s) - 1, 0, len(pts) - 2))
    d = pts[k + 1] - pts[k]
    gam = np.clip(np.arctan2(d[2], np.linalg.norm(d[:2])),
                  -np.radians(base.GAMMA_MAX_DEG), np.radians(base.GAMMA_MAX_DEG))
    chi = np.arctan2(d[1], d[0])
    return float(gam), float(chi + 2 * np.pi * np.round((chi_now - chi) / (2 * np.pi)))


def path_between(pts, s_cum, s0, s1, p0):
    """Polyline from p0 along the guide from arclength s0 to s1."""
    inner = pts[(s_cum > s0) & (s_cum < s1)]
    return np.vstack([p0, inner, point_at(pts, s_cum, s1)])


# --------------------------------------------------------------------------
# dense plans, flying, and turning trajectories into guesses
# --------------------------------------------------------------------------
def dense_plan(solver, res, kind):
    """(t, x, u) arrays along a solved plan, t from 0 to its flight time."""
    sigma = res.flight_time
    if kind == "birkhoff":
        X, V, U = (np.array(res.birkhoff[k]) for k in ("X", "V", "U"))
        Xo, Uo = solver.resample(X, V, U, 400)
        return np.linspace(0, sigma, 400), Xo, Uo
    X, U = np.asarray(res.x), np.asarray(res.u)
    N, steps = len(X), 10
    d = solver.propagate(X, U, sigma, steps, sens=False)
    xs = np.vstack([d["x_sub"].reshape(-1, 6), X[-1]])
    s = np.r_[np.repeat(np.arange(N - 1), steps) + np.tile(np.arange(steps) / steps, N - 1), N - 1]
    us = np.stack([np.interp(s, np.arange(N), U[:, j]) for j in range(3)], axis=1)
    return s / (N - 1) * sigma, xs, us


def fly(model, x0, plan, duration, dt=EXEC_DT):
    """Integrate the true dynamics from x0 with the plan's controls (RK4)."""
    t_p, _, u_p = plan
    uat = lambda t: np.array([np.interp(t, t_p, u_p[:, j]) for j in range(3)])
    K = max(int(round(duration / dt)), 1)
    h = duration / K
    x = np.array(x0, float)
    out = [x.copy()]
    for i in range(K):
        t = i * h
        u0, um, u1 = uat(t), uat(t + h / 2), uat(t + h)
        k1 = model.f(x, u0); k2 = model.f(x + h / 2 * k1, um)
        k3 = model.f(x + h / 2 * k2, um); k4 = model.f(x + h * k3, u1)
        x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        out.append(x.copy())
    return np.linspace(0, duration, K + 1), np.array(out)


def samples_from_plan(plan, shift, t_nodes, x_now):
    """States/controls at t_nodes taken from a plan shifted by `shift` seconds;
    beyond the plan's end, keep flying straight at the last state."""
    t_p, x_p, u_p = plan
    t = t_nodes + shift
    X = np.stack([np.interp(t, t_p, x_p[:, j]) for j in range(6)], axis=1)
    U = np.stack([np.interp(t, t_p, u_p[:, j]) for j in range(3)], axis=1)
    late = t > t_p[-1]
    if np.any(late):
        xe = x_p[-1]
        dirv = np.array([np.cos(xe[4]) * np.cos(xe[5]), np.cos(xe[4]) * np.sin(xe[5]), np.sin(xe[4])])
        X[late, :3] = xe[:3] + np.outer(t[late] - t_p[-1], xe[3] * dirv)
    X[0] = x_now
    return X, U


def samples_from_polyline(model, pts, t_nodes, T, x_now):
    """Fly a polyline at the speed needed to cover it in T seconds."""
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s_pts = np.r_[0.0, np.cumsum(seg)]
    s = t_nodes / T * s_pts[-1]
    p = np.stack([np.interp(s, s_pts, pts[:, k]) for k in range(3)], axis=1)
    d = np.gradient(p, t_nodes, axis=0)
    gam = np.clip(np.arctan2(d[:, 2], np.linalg.norm(d[:, :2], axis=1)),
                  -np.radians(base.GAMMA_MAX_DEG), np.radians(base.GAMMA_MAX_DEG))
    chi = np.arctan2(d[:, 1], d[:, 0])
    chi = x_now[5] + np.unwrap(np.r_[x_now[5], chi])[1:] - x_now[5]   # continuous with current heading
    X = np.column_stack([p, np.full(len(p), np.clip(s_pts[-1] / T, base.V_MIN, base.V_MAX)), gam, chi])
    X[0] = x_now
    return X, model.trim_controls(X)


# --------------------------------------------------------------------------
# horizon problems
# --------------------------------------------------------------------------
class _HorizonMixin:
    """Fixed duration, start = current state, free end pulled toward a target."""

    def set_horizon(self, x_now, target, T, guesses, angles=None):
        self.start_state, self.terminal_target, self.fixed_sigma = np.asarray(x_now, float), target, T
        self.terminal_weight = TERMINAL_WEIGHT
        self.terminal_angles, self.angle_weight = angles, ANGLE_WEIGHT
        self.guesses = guesses

    def initial_guesses(self, env):
        yield from self.guesses

    def solve_first(self, env):
        """Try the guesses in order and return the first feasible plan (or the last attempt)."""
        t0 = time.perf_counter()
        self.n_nodes = self.nodes_for(env)
        res, iters = None, 0
        for g in self.initial_guesses(env):
            res = self._solve_from(env, g)
            iters += res.iterations
            if res.success:
                break
        res.solve_time, res.total_iters = time.perf_counter() - t0, iters
        return res


class ShootingHorizon(_HorizonMixin, AircraftSCP):
    name = "rh_shooting"

    def __init__(self, nodes=SHOOT_NODES):
        super().__init__(verbose=False, n_starts=1)
        self.nodes = nodes
        self.start_state = self.terminal_target = self.fixed_sigma = None
        self.terminal_weight = 0.0

    def nodes_for(self, env):
        return self.nodes

    def boundary(self, env):
        x0, pg = super().boundary(env)
        return (x0 if self.start_state is None else self.start_state), pg

    def node_times(self, T):
        return np.linspace(0, T, self.nodes)

    def make_guess(self, X, U, T):
        return X, U, T

    def subproblem_constraints(self, env, X, U, sig, X_bar, U_bar, s_bar, N):
        import cvxpy as cp
        cons = super().subproblem_constraints(env, X, U, sig, X_bar, U_bar, s_bar, N)
        # scp_aircraft builds: [start, goal, pos>=lo, pos<=hi, V>=min, V<=max, |gamma|<=max, ...];
        # node 0 is the aircraft's actual state, so its state limits are dropped
        lo, hi = env.position_bounds()
        gmax = np.radians(base.GAMMA_MAX_DEG)
        cons[2:7] = [X[1:, :3] >= lo, X[1:, :3] <= hi, X[1:, 3] >= base.V_MIN,
                     X[1:, 3] <= base.V_MAX, cp.abs(X[1:, 4]) <= gmax]
        if self.terminal_target is not None:
            cons.pop(1)                                   # drop "end at the goal"
        if self.fixed_sigma is not None:
            cons.append(sig == self.fixed_sigma)
        return cons

    def cost(self, X, U, sigma):
        c = super().cost(X, U, sigma)
        if self.terminal_target is not None:
            e = (X[-1, :3] - self.terminal_target) / base.POS_SCALE
            c += self.terminal_weight * float(e @ e)
        if getattr(self, "terminal_angles", None) is not None:
            da = X[-1, 4:6] - np.asarray(self.terminal_angles)
            c += self.angle_weight * float(da @ da)
        return c

    def cost_expr(self, X, U, sig, N):
        import cvxpy as cp
        c = super().cost_expr(X, U, sig, N)
        if self.terminal_target is not None:
            c = c + self.terminal_weight * cp.sum_squares((X[-1, :3] - self.terminal_target) / base.POS_SCALE)
        if getattr(self, "terminal_angles", None) is not None:
            c = c + self.angle_weight * cp.sum_squares(X[-1, 4:6] - np.asarray(self.terminal_angles))
        return c


class BirkhoffHorizon(_HorizonMixin, BirkhoffAircraftSCP):
    name = "rh_birkhoff"

    def __init__(self, degree=BK_HORIZON_DEGREE):
        super().__init__(verbose=False, n_starts=1)
        self.bb = BirkhoffBasis(degree)
        self.setup_segments(1, HORIZON_S)
        self.D = self.bb.derivative()

    def nodes_for(self, env):
        self.setup_segments(1, self.fixed_sigma or getattr(self, "t_est", HORIZON_S))
        return self.G

    def node_times(self, T):
        return self.r * T

    def make_guess(self, X, U, T):
        V = self.D @ X                                    # one segment: V = dX/dtau
        X = self.integrate_segments(X[0], V)              # exactly consistent with X = X0 + B V
        return X, V, U, T


# --------------------------------------------------------------------------
# flying one environment
# --------------------------------------------------------------------------
def fly_environment(env, method, verbose=False):
    t_start = time.perf_counter()
    pts, s_cum = guide_path(env)
    guide_time = time.perf_counter() - t_start
    horizon = ShootingHorizon() if method == "shooting" else BirkhoffHorizon()
    model = horizon.model
    x_now = AircraftSCP.boundary(horizon, env)[0]           # the environment's start state
    plan, plan_age = None, 0.0               # last good plan and how long it has been flown
    flown_t, flown_x = [0.0], [x_now.copy()]
    replans, fails, solve_times, iters = 0, 0, [], []
    s_prog = 0.0
    status = "max_replans"
    saved = base.MAX_ITERS
    saved_wt = base.W_TRUST
    base.MAX_ITERS, base.W_TRUST = RH_MAX_ITERS, RH_W_TRUST
    try:
        while replans < MAX_REPLANS:
            s_prog = project_on_path(pts, s_cum, x_now[:3], s_prog)
            remaining_time = (s_cum[-1] - s_prog) / base.V_CRUISE
            if remaining_time <= FINAL_APPROACH_S:
                break
            T = HORIZON_S
            v_tgt = TARGET_SPEED or 0.5 * (base.V_CRUISE + base.V_MAX)
            target = point_at(pts, s_cum, s_prog + v_tgt * T)
            t_nodes = horizon.node_times(T)
            guesses = []
            if plan is not None:
                guesses.append(horizon.make_guess(*samples_from_plan(plan, plan_age, t_nodes, x_now), T))
            poly = path_between(pts, s_cum, s_prog, s_prog + x_now[3] * T, x_now[:3])   # at current speed
            guesses.append(horizon.make_guess(*samples_from_polyline(model, poly, t_nodes, T, x_now), T))
            angles = direction_at(pts, s_cum, s_prog + v_tgt * T, x_now[5]) if ANGLE_WEIGHT > 0 else None
            horizon.set_horizon(x_now, target, T, guesses, angles)
            res = horizon.solve_first(env)
            replans += 1
            solve_times.append(res.solve_time)
            iters.append(res.total_iters)
            if res.success:
                plan, plan_age = dense_plan(horizon, res, method), 0.0
            else:
                fails += 1
                if plan is None or plan_age + EXECUTE_S > plan[0][-1] - 1e-6:
                    status = "replan_failed_no_fallback"
                    break
            shifted = (plan[0] - plan_age, plan[1], plan[2])
            tt, xx = fly(model, x_now, (shifted[0], shifted[1], shifted[2]), EXECUTE_S)
            flown_t += list(flown_t[-1] + tt[1:]); flown_x += list(xx[1:])
            x_now, plan_age = xx[-1].copy(), plan_age + EXECUTE_S
            if verbose:
                print(f"    replan {replans:3d}  {res.status:20s} {res.solve_time:5.2f}s  "
                      f"progress {s_prog / s_cum[-1]:5.1%}")
        else:
            status = "max_replans"
        if status not in ("replan_failed_no_fallback",):
            # final approach: free final time, end exactly at the goal, from the current state
            base.MAX_ITERS = saved
            t_est = max((s_cum[-1] - s_prog) / base.V_CRUISE, 5.0)
            final = ShootingHorizon(nodes=max(6, int(np.ceil(t_est / (HORIZON_S / (SHOOT_NODES - 1)))) + 1)) \
                if method == "shooting" else BirkhoffHorizon()
            final.t_est = t_est
            final.start_state = np.asarray(x_now, float)
            final.n_nodes = final.nodes_for(env)
            poly = path_between(pts, s_cum, s_prog, s_cum[-1], x_now[:3])
            straight = np.vstack([x_now[:3], env.goal_pos])
            guesses = [final._polyline_guess(poly, env), final._polyline_guess(straight, env)]
            final.set_horizon(x_now, None, None, guesses)
            res = final.solve_first(env)
            solve_times.append(res.solve_time)
            iters.append(res.total_iters)
            replans += 1
            if res.success:
                fplan = dense_plan(final, res, method)
                tt, xx = fly(model, x_now, fplan, fplan[0][-1])
                flown_t += list(flown_t[-1] + tt[1:]); flown_x += list(xx[1:])
                x_now = xx[-1]
                status = "reached_goal"
            else:
                fails += 1
                status = f"final_approach_failed ({res.status})"
    finally:
        base.MAX_ITERS, base.W_TRUST = saved, saved_wt

    flown_x = np.array(flown_x)
    P = flown_x[:, :3]
    hn, _ = AircraftSCP.obstacle_eval(horizon, env, P)
    margin_min = float(hn.min())                                     # vs the 36 m planning clearance
    body_clear = margin_min + env.margin                             # vs the aircraft's own size
    lo, hi = env.position_bounds()                                  # planning box (shrunk by the radius)
    overshoot = float(max(np.max(lo - P), np.max(P - hi), 0.0))      # past the planning box, m
    st = np.array(solve_times)
    return {"method": method, "status": status,
            "goal_error_m": float(np.linalg.norm(P[-1] - env.goal_pos)),
            "flight_time_s": float(flown_t[-1]),
            "path_length_m": float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1))),
            "collision_free": bool(body_clear >= 0 and overshoot <= env.vehicle_radius),   # vs the real walls
            "box_overshoot_m": overshoot,
            "min_body_clearance_m": body_clear,
            "worst_margin_intrusion_m": float(max(0.0, -margin_min)),
            "replans": replans, "failed_replans": fails,
            "solve_time_total_s": float(st.sum()),
            "solve_time_mean_s": float(st[:-1].mean()) if len(st) > 1 else float(st.mean()),
            "solve_time_p95_s": float(np.percentile(st[:-1], 95)) if len(st) > 1 else float(st.max()),
            "solve_time_max_s": float(st[:-1].max()) if len(st) > 1 else float(st.max()),
            "final_solve_s": float(st[-1]),
            "iters_mean": float(np.mean(iters)),
            "guide_path_s": guide_time,
            "wall_time_s": time.perf_counter() - t_start,
            "flown_x": flown_x[::20].tolist()}


# --------------------------------------------------------------------------
# full-horizon baseline (time to the first feasible full plan, and best of N_STARTS)
# --------------------------------------------------------------------------
def full_horizon(env, method):
    solver = AircraftSCP() if method == "shooting" else BirkhoffAircraftSCP()
    saved = base.PICK_BEST_START
    base.PICK_BEST_START = False
    try:
        t0 = time.perf_counter()
        r = solver.solve(env)
        t_first = time.perf_counter() - t0
    finally:
        base.PICK_BEST_START = saved
    return {"first_feasible_time_s": t_first, "first_status": r.status,
            "first_flight_time_s": float(r.flight_time) if r.success else None}


def main():
    ap = argparse.ArgumentParser(description="Benchmark receding-horizon SCP against full-horizon SCP.")
    ap.add_argument("--envs", default=ENV_FILE)
    ap.add_argument("--out", default=RESULTS_FILE)
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=["shooting", "birkhoff"])
    ap.add_argument("--max-envs", type=int, default=None)
    ap.add_argument("--horizon", type=float, default=None, help="horizon length, s")
    ap.add_argument("--execute", type=float, default=None, help="time flown between replans, s")
    ap.add_argument("--baseline", action="store_true", help="also time full-horizon solves")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=UserWarning, module="cvxpy")
    global HORIZON_S, EXECUTE_S
    HORIZON_S = args.horizon or HORIZON_S
    EXECUTE_S = args.execute or EXECUTE_S

    envs = load_environments(args.envs)[: args.max_envs]
    out = []
    print(f"Receding horizon: {HORIZON_S:.0f} s horizon, replan every {EXECUTE_S:.0f} s "
          f"[{base.AIRFRAME_NAME}]")
    for i, env in enumerate(envs):
        for m in args.methods:
            r = fly_environment(env, m, verbose=args.verbose)
            r["env_index"] = i
            if args.baseline:
                r["full_horizon"] = full_horizon(env, m)
            out.append(r)
            print(f"  env {i}  {m:8s}  {r['status']:24s} flight {r['flight_time_s']:6.1f}s  "
                  f"{'collision-free' if r['collision_free'] else 'COLLISION'} "
                  f"(margin intrusion {r['worst_margin_intrusion_m']:4.1f} m)  "
                  f"replans {r['replans']:3d} (failed {r['failed_replans']})  "
                  f"solve mean {r['solve_time_mean_s']:.2f}s / p95 {r['solve_time_p95_s']:.2f}s / "
                  f"max {r['solve_time_max_s']:.2f}s  total {r['solve_time_total_s']:5.1f}s"
                  + (f"  | full horizon: first plan in {r['full_horizon']['first_feasible_time_s']:.1f}s"
                     if args.baseline else ""), flush=True)
    path = localstore_path(args.out)
    with open(path, "w") as f:
        json.dump(out, f)
    print(f"\nResults -> {path}")


if __name__ == "__main__":
    main()