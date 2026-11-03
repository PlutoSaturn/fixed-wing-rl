#!/usr/bin/env python3
"""
scp_birkhoff.py - SCP for the 3-DOF fixed-wing aircraft with a Birkhoff
pseudospectral discretization instead of multiple shooting.

Same problem, same SCP loop as scp_aircraft.py (aircraft model, limits,
boundary conditions, cost, obstacles, PTR trust region, virtual control,
obstacle slacks, restarts, stall detection, result format). Only the
discretization differs, so the two can be benchmarked against each other.

--------------------------------------------------------------------------
Birkhoff discretization
--------------------------------------------------------------------------
Time is split into S segments of equal length. Inside each segment, with
local time tau in [-1, 1] at Legendre-Gauss-Lobatto (LGL) nodes tau_0..tau_n,
the state is a polynomial written through its value at the segment start and
its derivative at every node (Birkhoff interpolation):

    x(tau) = x_0 + sum_j B_j(tau) V_j,        B_j(tau) = integral_{-1}^{tau} l_j(s) ds

(l_j = Lagrange polynomials on the nodes). At the nodes this is the matrix
equation  X_i = X_0 + sum_j B_ij V_j, which is LINEAR and EXACT, so it is
never linearized. The physics is imposed pointwise at every node:

    V_i = (sigma / (2 S)) f(X_i, U_i)          (collocation; sigma = flight time)

Decision variables: X (state), V (state derivative), U (controls) at every
global node (segments share their end nodes), plus sigma.

Each SCP iteration linearizes only the collocation equations, using the
model's Jacobians at the nodes. No integration happens inside the loop.
Obstacle check points are placed at any local times you like; their
positions p(tau) = p_0 + sum_j B_j(tau) V_j,pos are exactly linear in the
variables.

Verification: the converged polynomial trajectory is (1) checked for
obstacles on a dense grid, (2) re-checked by RK4 multiple shooting over
VERIFY_INTERVALS equal intervals using its own polynomial controls, and (3)
simulated open loop from the start; the deviation is reported.

Output: results are saved in the same format as scp_aircraft.py, resampled
onto OUTPUT_NODES uniformly spaced nodes, so --show, jsbsim_c172.py track /
record / fgview all work unchanged (pass --results scp_birkhoff_results.json).
The native Birkhoff nodes are stored too, under "birkhoff".

Usage (from the project root)
    python -m solvers.scp_birkhoff --envs c172_envs.json
    python -m solvers.scp_birkhoff --envs c172_envs.json --show 0
    python -m sim.jsbsim_c172 track --results scp_birkhoff_results.json
"""
from __future__ import annotations

import argparse
import json
import time
import warnings

import cvxpy as cp
import numpy as np
import scipy.sparse as sp
from numpy.polynomial import legendre as npleg

from solvers import scp_aircraft as base
from solvers.scp_aircraft import AircraftSCP, SolveResult, load_plan, plan_clearance, plot_result
from environment.envgen import load_environments, localstore_path, save_environments


# ==========================================================================
# USER SETTINGS (everything not listed here is shared with scp_aircraft.py:
# airframe, limits, cost weights, penalties, tolerances, restarts, early stopping)
# ==========================================================================
ENV_FILE          = "c172_envs.json"
RESULTS_FILE      = "scp_birkhoff_results.json"
FEASIBLE_FILE     = "feasible_envs_birkhoff.json"

BK_DEGREE         = 12        # polynomial degree per segment (nodes per segment = BK_DEGREE + 1)
BK_SEGMENTS       = None      # fixed number of segments; None = choose from BK_SEGMENT_S
BK_SEGMENT_S      = 40.0      # target segment length in seconds (estimated flight time / this)
                              #   1 segment = a single global polynomial (classic Birkhoff PS)
BK_CHECK_DT_S     = 0.5       # spacing of obstacle check points, s (uniform in time)
BK_DENSE_DT_S     = 0.05      # spacing for the final dense obstacle check, s
FLIGHT_TIME_PAD   = 1.3       # flight time assumed for spacing = straight-line time x this
OUTPUT_NODES      = 241       # uniform nodes in the saved result (for tracking / plotting)
VERIFY_INTERVALS  = 120       # the plan is re-checked over this many equal intervals with RK4 ...
VERIFY_SUBSTEPS   = 8         #   ... and this many substeps each, using the plan's own polynomial controls
VERIFY_DEFECT_TOL = 2e-3      # max scaled RK4 defect over those intervals for "feasible"
VERIFY_DT         = 0.1       # s, step of the open-loop verification simulation

# ---- speed --------------------------------------------------------------------
BK_MAX_OBS_PER_CHECK = 3      # obstacle constraints per check point: only its K nearest obstacles
                              #   (None = every obstacle within reach). Skipped ones still count in
                              #   the merit, so one that becomes close is added next iteration.
BK_ADAPTIVE_REACH = True      # shrink the obstacle search radius as SCP steps get small
DIRECT_CLARABEL   = True      # build the subproblem matrices directly and call Clarabel, instead of
                              #   rebuilding a cvxpy model every iteration (same problem, less overhead)
SOLVER_OPTS       = {}        # extra cvxpy/Clarabel settings, e.g. {"tol_gap_rel": 1e-6}

# ---- adaptive mesh refinement -------------------------------------------------
# Off: equal segments of BK_SEGMENT_S. On: each start is first solved on coarse segments of
# BK_REFINE_START_S; segments where the plan is hard to fly accurately are then split in half
# and the start is re-solved, warm-started from its own solution (exact on the finer mesh).
BK_REFINE           = False   # adaptive segment refinement on/off
BK_REFINE_START_S   = 60.0    # coarse segment length for the first solve, s
BK_REFINE_ROUNDS    = 3       # at most this many split-and-resolve rounds per start
BK_REFINE_DEFECT    = 0.5     # split a segment whose RK4 re-check defect exceeds this fraction of
                              #   VERIFY_DEFECT_TOL (dynamics not resolved between its nodes)
BK_REFINE_MARGIN_M  = -2.0    # where the dense obstacle check dips deeper than this into the
                              #   safety margin (m) between check points, halve that segment's
                              #   check spacing (no split: it is the checks, not the nodes, that miss it)
BK_MIN_SEGMENT_S    = 5.0     # never split a segment shorter than this, s
BK_MAX_SEGMENTS     = 48      # never use more segments than this


# --------------------------------------------------------------------------
# LGL nodes and Birkhoff matrices
# --------------------------------------------------------------------------
def lgl_nodes(n):
    """Legendre-Gauss-Lobatto nodes on [-1, 1]: the ends plus the roots of P_n'."""
    inner = np.sort(npleg.Legendre.basis(n).deriv().roots().real) if n > 1 else np.array([])
    return np.r_[-1.0, inner, 1.0]


class BirkhoffBasis:
    """Matrices for degree-n Birkhoff interpolation on LGL nodes, built in a
    Legendre basis (accurate for n up to a few hundred)."""

    def __init__(self, n):
        self.n = n
        self.tau = lgl_nodes(n)
        Vm = np.stack([self._P(k, self.tau) for k in range(n + 1)], axis=1)
        self.C = np.linalg.inv(Vm)                     # Legendre coefficients of the Lagrange basis
        self.B = self.integral(self.tau)               # (n+1, n+1): X_i = X_0 + B[i] @ V
        self.w = self.B[-1].copy()                     # quadrature weights (row at tau = 1)

    @staticmethod
    def _P(k, t):
        return npleg.legval(t, np.eye(k + 1)[k]) if k >= 0 else np.zeros_like(t)

    def integral(self, t):
        """Rows B_j(t) = integral_{-1}^{t} l_j(s) ds for each t."""
        t = np.atleast_1d(np.asarray(t, float))
        cols = [t + 1.0] + [(self._P(k + 1, t) - self._P(k - 1, t)) / (2 * k + 1)
                            for k in range(1, self.n + 1)]
        return np.stack(cols, axis=1) @ self.C

    def lagrange(self, t):
        """Rows l_j(t): polynomial interpolation through the node values."""
        t = np.atleast_1d(np.asarray(t, float))
        return np.stack([self._P(k, t) for k in range(self.n + 1)], axis=1) @ self.C

    def derivative(self):
        """Differentiation matrix D (used only to build consistent initial guesses)."""
        dP = np.stack([npleg.legval(self.tau, npleg.legder(np.eye(self.n + 1)[k]))
                       for k in range(self.n + 1)], axis=1)
        return dP @ self.C


def _better_result(res, best):
    """The solvers' best-of rule: valid beats invalid; then lower cost; else less infeasible."""
    if best is None:
        return True
    if res.success != best.success:
        return res.success
    if res.success:
        return res.cost < best.cost
    return res.max_violation + res.max_defect < best.max_violation + best.max_defect


def _refine_score(res):
    """Ranking of one start's results across refinement rounds: valid plans by cost; then plans
    converged at the nodes that failed only the between-node checks, by how badly; then the rest."""
    if res.success:
        return (2, -res.cost)
    if res.max_defect <= base.TOL_DEFECT and res.max_violation <= base.TOL_VIOLATION:
        verify = res.birkhoff["verify_defect"] / VERIFY_DEFECT_TOL
        return (1, -max(verify, -res.dense_min_margin / 7.5))
    return (0, -(res.max_violation + res.max_defect))


# --------------------------------------------------------------------------
# solver
# --------------------------------------------------------------------------
class BirkhoffAircraftSCP(AircraftSCP):
    """Same aircraft problem as AircraftSCP, discretized with Birkhoff
    interpolation on LGL nodes (optionally in several segments)."""
    name = "birkhoff_scp"

    def __init__(self, verbose=base.VERBOSE, n_starts=base.N_STARTS):
        super().__init__(verbose=verbose, n_starts=n_starts)
        self.bb = BirkhoffBasis(BK_DEGREE)
        self.setup_segments(1)
        # short-horizon (receding-horizon) options; all off for the normal full-flight problem
        self.start_state = None      # start from this state instead of the environment's start
        self.fixed_sigma = None      # fixed horizon length in seconds instead of free final time
        self.terminal_target = None  # free end position, pulled toward this point instead of
        self.terminal_weight = 0.0   #   being fixed to the goal (cost w * ||p_end - target||^2 / POS_SCALE^2)
        self.terminal_angles = None  # optional (gamma, chi) the end state should point along ...
        self.angle_weight = 0.0      #   (cost w * ((gamma_end - g)^2 + (chi_end - c)^2), radians)

    def boundary(self, env):
        x0, p_goal = super().boundary(env)
        return (x0 if self.start_state is None else np.asarray(self.start_state, float)), p_goal

    def _terminal_cost(self, X):
        c = 0.0
        if self.terminal_target is not None:
            e = (X[-1, :3] - self.terminal_target) / base.POS_SCALE
            c += self.terminal_weight * float(e @ e)
        if self.terminal_angles is not None:
            da = X[-1, 4:6] - np.asarray(self.terminal_angles)
            c += self.angle_weight * float(da @ da)
        return c

    # ---- grid ----------------------------------------------------------------
    def setup_segments(self, S, t_est=200.0, bounds=None, check_dt=None):
        """Segments in normalized time. Equal segments by default; `bounds` (S+1 increasing
        values from 0 to 1) gives unequal ones, and `check_dt` (one value per segment) the
        obstacle check spacing in each, s.
        The derivative variables V are dX/dtau over the WHOLE flight (tau in [0, 1]), so a
        node shared by two segments of different lengths has one consistent V:
            within segment s (half-length h_s in tau):  X(t) = X_0 + h_s * sum_j B_j(t) V_j
            collocation at every node:                  V_i = sigma * f(X_i, U_i)"""
        n = self.bb.n
        tb = np.linspace(0.0, 1.0, S + 1) if bounds is None else np.asarray(bounds, float)
        S = len(tb) - 1
        self.S, self.tb, self.t_est = S, tb, t_est
        self.hs = np.diff(tb) / 2                                  # half-length of each segment, tau
        self.check_dt = (np.full(S, BK_CHECK_DT_S) if check_dt is None
                         else np.asarray(check_dt, float))
        self.G = S * n + 1                                         # global nodes (shared ends)
        self.idx = np.arange(S)[:, None] * n + np.arange(n + 1)[None, :]     # (S, n+1)
        # normalized time [0, 1] of every global node
        self.r = np.r_[np.concatenate([tb[s] + (self.bb.tau[:-1] + 1) * self.hs[s] for s in range(S)]), 1.0]
        self.dr = np.diff(self.r)                                  # normalized node spacing
        self.hloc = 2.0 / n                                        # mean local spacing (for scaling)
        # per-node segment half-length (averaged at shared nodes), for scaling only
        self.h_node = np.zeros(self.G)
        cnt = np.zeros(self.G)
        # quadrature weights over normalized time [0, 1]
        self.q = np.zeros(self.G)
        for s in range(S):
            self.q[self.idx[s]] += self.bb.w * self.hs[s]
            self.h_node[self.idx[s]] += self.hs[s]
            cnt[self.idx[s]] += 1
        self.h_node /= cnt
        # obstacle check points: uniform in time within each segment, plus the very end
        seg_s = t_est * FLIGHT_TIME_PAD * 2 * self.hs              # seconds per segment (estimated)
        self.c_seg, self.c_B = self._check_grid(np.maximum(8, np.ceil(seg_s / self.check_dt)).astype(int))
        self.d_seg, self.d_B = self._check_grid(np.maximum(40, np.ceil(seg_s / BK_DENSE_DT_S)).astype(int))
        self.n_nodes = self.G

    def _check_grid(self, ms):
        """ms[s] points in segment s, uniform in time, plus the final point. The rows already
        include the segment's half-length, so p = p_0(seg) + Brows @ V_pos(seg)."""
        segs, rows = [], []
        for s, m in enumerate(np.broadcast_to(ms, (self.S,))):
            t = -1 + 2 * np.arange(m) / m
            segs.append(np.full(m, s))
            rows.append(self.hs[s] * self.bb.integral(t))
        seg = np.r_[np.concatenate(segs), self.S - 1]
        Brows = np.vstack(rows + [self.hs[-1] * self.bb.B[-1:]])
        return seg, Brows

    def locate(self, r):
        """Segment index and local time t in [-1, 1] of normalized times r."""
        r = np.clip(np.asarray(r, float), 0.0, 1.0)
        seg = np.clip(np.searchsorted(self.tb, r, side="right") - 1, 0, self.S - 1)
        return seg, (r - self.tb[seg]) / self.hs[seg] - 1.0

    def nodes_for(self, env):
        t_est = np.linalg.norm(env.goal_pos - env.start_pos) / base.V_CRUISE
        seg_s = BK_REFINE_START_S if self.refining() else BK_SEGMENT_S
        S = int(BK_SEGMENTS) if BK_SEGMENTS is not None else max(1, int(np.ceil(t_est / seg_s)))
        self.setup_segments(S, t_est)
        self._start_mesh = (self.tb.copy(), self.t_est, self.check_dt.copy())
        return self.G

    allow_refine = True          # subclasses (e.g. the receding-horizon solver) can turn it off

    def refining(self):
        return BK_REFINE and self.allow_refine

    def positions(self, X, V, seg, Brows):
        """Positions at check points: p = p_0(seg) + Brows @ V_pos(seg)."""
        p0 = X[self.idx[seg, 0], :3]
        Vp = V[self.idx[seg], :3]                                  # (C, n+1, 3)
        return p0 + np.einsum("cj,cjk->ck", Brows, Vp)

    def integrate_segments(self, X0, V):
        """X at every node from the start state and V (chained segment by segment)."""
        X = np.zeros((self.G, 6))
        X[0] = X0
        for s in range(self.S):
            i = self.idx[s]
            X[i[1:]] = X[i[0]] + self.hs[s] * self.bb.B[1:] @ V[i]
        return X

    # ---- initial guesses ---------------------------------------------------
    def _polyline_guess(self, pts, env):
        """A polyline flown at cruise speed, sampled at the LGL node times, with
        V chosen so the guess satisfies X = X_0 + B V exactly."""
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        s_pts = np.r_[0.0, np.cumsum(seg)]
        p = np.stack([np.interp(self.r * s_pts[-1], s_pts, pts[:, k]) for k in range(3)], axis=1)
        d = np.gradient(p, self.r, axis=0)
        gam = np.clip(np.arctan2(d[:, 2], np.linalg.norm(d[:, :2], axis=1)),
                      -np.radians(base.GAMMA_MAX_DEG), np.radians(base.GAMMA_MAX_DEG))
        chi = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
        X = np.column_stack([p, np.full(self.G, base.V_CRUISE), gam, chi])
        x0, p_goal = self.boundary(env)
        X[0] = x0
        X[0, 5] = chi[0] + np.angle(np.exp(1j * (x0[5] - chi[0])))      # keep heading continuous
        sigma = float(np.clip(s_pts[-1] / base.V_CRUISE, *base.SIGMA_BOUNDS))
        # V = dX/dtau from differentiating each segment's polynomial; shared nodes averaged
        D = self.bb.derivative()
        V = np.zeros_like(X)
        cnt = np.zeros(self.G)
        for s in range(self.S):
            i = self.idx[s]
            V[i] += D @ X[i] / self.hs[s]
            cnt[i] += 1
        V /= cnt[:, None]
        # re-integrate so X = X_0 + B V holds exactly, then shift V so the end is at the goal
        X = self.integrate_segments(X[0], V)
        V[:, :3] += p_goal - X[-1, :3]                                  # a constant c integrates to c over tau in [0, 1]
        X = self.integrate_segments(X[0], V)
        # controls that roughly produce the guessed climb and turn rates (dX/dt = V / sigma)
        k = sigma
        U = self.model.trim_controls(X)
        g, Vv, cg = base.GRAVITY, X[:, 3], np.cos(X[:, 4])
        U[:, 1] = cg + Vv * (V[:, 4] / k) / g
        U[:, 2] = Vv * cg * (V[:, 5] / k) / g
        U[:, 1] = np.clip(U[:, 1], 0.2, base.N_MAX)
        lim = np.minimum(np.tan(np.radians(base.BANK_MAX_DEG)) * U[:, 1],
                         np.sqrt(np.maximum(base.N_MAX**2 - U[:, 1]**2, 0)))
        U[:, 2] = np.clip(U[:, 2], -lim, lim)
        return X, V, U, sigma

    # ---- problem pieces on the non-uniform grid -------------------------------
    def _smooth_weights(self):
        h_ref = 1.0 / (self.G - 1)
        return h_ref / self.dr                     # makes sum (du)^2 h_ref/dr ~ uniform-grid scaling

    def cost(self, X, U, sigma):
        wts = self._smooth_weights()
        return (self._terminal_cost(X) + base.W_TIME * sigma
                + base.W_SPEED * float(np.sum(self.q * ((X[:, 3] - base.V_CRUISE) / base.V_CRUISE) ** 2))
                + base.W_SMOOTH * float(np.sum(wts[:, None] * (np.diff(U, axis=0) / self.us) ** 2)))

    def cost_expr(self, X, U, sig, N):
        wts = np.sqrt(self._smooth_weights())
        term = 0
        if self.terminal_target is not None:
            term = self.terminal_weight * cp.sum_squares((X[-1, :3] - self.terminal_target) / base.POS_SCALE)
        if self.terminal_angles is not None:
            term = term + self.angle_weight * cp.sum_squares(X[-1, 4:6] - np.asarray(self.terminal_angles))
        return (term + base.W_TIME * sig
                + base.W_SPEED * cp.sum(cp.multiply(self.q, cp.square((X[:, 3] - base.V_CRUISE) / base.V_CRUISE)))
                + base.W_SMOOTH * cp.sum_squares(cp.multiply(cp.diff(U, axis=0), wts[:, None] / self.us[None, :])))

    def subproblem_constraints(self, env, X, U, sig, X_bar, U_bar, s_bar, N):
        x0, p_goal = self.boundary(env)
        lo, hi = env.position_bounds()
        tanb = np.tan(np.radians(base.BANK_MAX_DEG))
        gmax = np.radians(base.GAMMA_MAX_DEG)
        cons = [X[0] == x0,                        # state limits from node 1: node 0 is the given start
                X[1:, :3] >= lo, X[1:, :3] <= hi,
                X[1:, 3] >= base.V_MIN, X[1:, 3] <= base.V_MAX,
                cp.abs(X[1:, 4]) <= gmax,
                U[:, 0] >= 0,
                U[:, 0] <= base.THRUST_PLAN_FRACTION * (base.T_MAX + base.THRUST_SLOPE * X[:, 3]),
                cp.norm(U[:, 1:], 2, axis=1) <= base.N_MAX,
                cp.abs(U[:, 2]) <= tanb * U[:, 1],
                sig >= base.SIGMA_BOUNDS[0], sig <= base.SIGMA_BOUNDS[1]]
        if self.terminal_target is None:
            cons.append(X[-1, :3] == p_goal)
        if self.fixed_sigma is not None:
            cons.append(sig == self.fixed_sigma)
        if base.FINAL_LEVEL:
            cons.append(X[-1, 4] == 0)
        for col, rate in ((1, base.N_V_RATE_MAX), (2, base.N_H_RATE_MAX)):
            if rate is not None:                   # |du| <= rate * dt, with dt = sigma * dr (non-uniform)
                cons.append(cp.abs(cp.diff(U[:, col])) <= rate * sig * self.dr)
        cons.append(cp.norm(U[:, 1:], 2, axis=1) <=
                    self.model.c_stall * (2 * cp.multiply(X_bar[:, 3], X[:, 3]) - X_bar[:, 3] ** 2))
        return cons

    # ---- nonlinear merit -------------------------------------------------------
    def bk_merit(self, env, X, V, U, sigma):
        with np.errstate(all="ignore"):
            f = self.model.f(X, U)
            if not np.all(np.isfinite(f)):
                return np.inf, np.inf, np.inf
            coll = (np.abs(V - sigma * f) * (self.h_node * self.hloc)[:, None]
                    / self.xs)                                               # scaled collocation defect
            integ = np.abs(self.integrate_segments(X[0], V) - X) / self.xs           # ~0 by construction
            defect = coll + integ
            if defect.max() > base.BLOWUP_DEFECT:
                return np.inf, np.inf, np.inf
            P = self.positions(X, V, self.c_seg, self.c_B)
            hn, Gn = self.obstacle_eval(env, P)
            self._obs_cache = (P, hn, Gn)              # reused by the next subproblem
            viol = np.maximum(0.0, -hn).max(axis=1)
            J = (self.cost(X, U, sigma) + base.LAMBDA_DYN * defect.sum()
                 + base.LAMBDA_OBS * viol.sum() / base.POS_SCALE)
        if not np.isfinite(J):
            return np.inf, np.inf, np.inf
        return J, float(defect.max()), float(viol.max())

    # ---- direct Clarabel subproblem -----------------------------------------------
    def _direct_subproblem(self, env, X_bar, V_bar, U_bar, s_bar, zbar, zs, cap,
                           Mc_s, rhs_c, Mi_s, rhs_i, Gm, Es, h_rows):
        """The same convex subproblem the cvxpy path builds, assembled as
        min 1/2 y'Py + q'y  s.t.  Ay + s = b, s in K  and solved by Clarabel.
        y = [dz, nu+, nu-, slack]. Returns (dz, nu_l1, slack_sum) or None."""
        import clarabel
        G, nx, nu = self.G, 6, 3
        nzv = len(zbar)
        oV, oU, isg = G * nx, 2 * G * nx, nzv - 1
        Nn = G * nx
        C = Es.shape[1] if Es is not None else 0
        ny = nzv + 2 * Nn + C
        iX = lambda i, a: i * nx + a
        iU = lambda i, c: oU + i * nu + c

        A_blocks, b_parts, cones = [], [], []

        def raw_rows(rows, cols, vals, const, n_rows, kind):
            """Rows written on raw variables z (expr = R z + const), converted to dz."""
            R = sp.csr_matrix((vals, (rows, cols)), shape=(n_rows, nzv))
            e = R @ zbar + const
            Rdz = R @ sp.diags(zs)
            pad = sp.csr_matrix((n_rows, ny - nzv))
            if kind == "soc":                       # s = b - Ay = expr
                A_blocks.append(sp.hstack([-Rdz, pad]))
                b_parts.append(e)
            else:                                   # eq: expr = 0 / ineq: expr <= 0
                A_blocks.append(sp.hstack([Rdz, pad]))
                b_parts.append(-e)

        # ---- equalities
        x0, p_goal = self.boundary(env)
        eq_r, eq_c, eq_v, eq_k = [], [], [], []
        for a in range(nx):
            eq_r.append(len(eq_k)); eq_c.append(iX(0, a)); eq_v.append(1.0); eq_k.append(-x0[a])
        if self.terminal_target is None:
            for a in range(3):
                eq_r.append(len(eq_k)); eq_c.append(iX(G - 1, a)); eq_v.append(1.0); eq_k.append(-p_goal[a])
        if self.fixed_sigma is not None:
            eq_r.append(len(eq_k)); eq_c.append(isg); eq_v.append(1.0); eq_k.append(-self.fixed_sigma)
        if base.FINAL_LEVEL:
            eq_r.append(len(eq_k)); eq_c.append(iX(G - 1, 4)); eq_v.append(1.0); eq_k.append(0.0)
        raw_rows(eq_r, eq_c, eq_v, np.array(eq_k), len(eq_k), "eq")
        n_eq = len(eq_k)
        A_blocks.append(sp.hstack([Mc_s, -sp.eye(Nn), sp.eye(Nn), sp.csr_matrix((Nn, C))]))
        b_parts.append(rhs_c)
        A_blocks.append(sp.hstack([Mi_s, sp.csr_matrix((Mi_s.shape[0], ny - nzv))]))
        b_parts.append(rhs_i)
        n_eq += Nn + Mi_s.shape[0]
        cones.append(clarabel.ZeroConeT(n_eq))

        # ---- linear inequalities (expr <= 0), built vectorized: each entry is
        #      (list of (column array, coefficient array)) and a constant array, one row per node
        lo, hi = env.position_bounds()
        gmax = np.radians(base.GAMMA_MAX_DEG)
        tanb = np.tan(np.radians(base.BANK_MAX_DEG))
        frac = base.THRUST_PLAN_FRACTION
        nodes = np.arange(G)
        one = np.ones(G)
        sn = nodes[1:]                                   # state limits from node 1 (node 0 = given start)
        s1 = np.ones(G - 1)
        groups = []
        for a in range(3):
            groups += [([(iX(sn, a), -s1)], np.full(G - 1, lo[a])),
                       ([(iX(sn, a), s1)], np.full(G - 1, -hi[a]))]
        groups += [([(iX(sn, 3), -s1)], np.full(G - 1, base.V_MIN)),
                   ([(iX(sn, 3), s1)], np.full(G - 1, -base.V_MAX)),
                   ([(iX(sn, 4), s1)], np.full(G - 1, -gmax)),
                   ([(iX(sn, 4), -s1)], np.full(G - 1, -gmax)),
                   ([(iU(nodes, 0), -one)], np.zeros(G)),
                   ([(iU(nodes, 0), one), (iX(nodes, 3), -frac * base.THRUST_SLOPE * one)],
                    np.full(G, -frac * base.T_MAX)),
                   ([(iU(nodes, 2), one), (iU(nodes, 1), -tanb * one)], np.zeros(G)),
                   ([(iU(nodes, 2), -one), (iU(nodes, 1), -tanb * one)], np.zeros(G)),
                   ([(np.array([isg]), np.array([-1.0]))], np.array([base.SIGMA_BOUNDS[0]])),
                   ([(np.array([isg]), np.array([1.0]))], np.array([-base.SIGMA_BOUNDS[1]]))]
        m = nodes[:-1]
        for col, rate in ((1, base.N_V_RATE_MAX), (2, base.N_H_RATE_MAX)):
            if rate is None:
                continue
            o1 = np.ones(G - 1)
            sg = np.full(G - 1, isg)
            for sgn in (1.0, -1.0):
                groups.append(([(iU(m + 1, col), sgn * o1), (iU(m, col), -sgn * o1),
                                (sg, -rate * self.dr)], np.zeros(G - 1)))
        r, c, v, kconst = [], [], [], []
        n_in = 0
        for entries, const in groups:
            nr = len(const)
            for cols, vals in entries:
                r.append(n_in + np.arange(nr)); c.append(np.asarray(cols)); v.append(np.asarray(vals, float))
            kconst.append(const)
            n_in += nr
        raw_rows(np.concatenate(r), np.concatenate(c), np.concatenate(v), np.concatenate(kconst),
                 n_in, "ineq")
        # trust region |dz| <= cap
        I_dz = sp.eye(nzv, format="csr")
        pad = sp.csr_matrix((nzv, ny - nzv))
        A_blocks += [sp.hstack([I_dz, pad]), sp.hstack([-I_dz, pad])]
        b_parts += [np.full(nzv, cap), np.full(nzv, cap)]
        n_in += 2 * nzv
        # obstacles: -(Gm dz + Es slack) <= h
        if Gm is not None:
            A_blocks.append(sp.hstack([-Gm, sp.csr_matrix((Gm.shape[0], 2 * Nn)), -Es]))
            b_parts.append(h_rows)
            n_in += Gm.shape[0]
        # nu+, nu-, slack >= 0
        A_blocks.append(sp.hstack([sp.csr_matrix((ny - nzv, nzv)), -sp.eye(ny - nzv)]))
        b_parts.append(np.zeros(ny - nzv))
        n_in += ny - nzv
        cones.append(clarabel.NonnegativeConeT(n_in))

        # ---- second-order cones, per node: load factor ||n|| <= N_MAX and
        #      (linearized) stall ||n|| <= c (2 Vbar V - Vbar^2); 6 rows per node
        cs = self.model.c_stall
        Vb = X_bar[:, 3]
        base_row = 6 * nodes
        r = np.concatenate([base_row + 1, base_row + 2,                         # load cone: U_v, U_h
                            base_row + 3, base_row + 4, base_row + 5])          # stall cone: t, U_v, U_h
        c = np.concatenate([iU(nodes, 1), iU(nodes, 2), iX(nodes, 3), iU(nodes, 1), iU(nodes, 2)])
        v = np.concatenate([one, one, 2 * cs * Vb, one, one])
        const = np.zeros(6 * G)
        const[base_row] = base.N_MAX
        const[base_row + 3] = -cs * Vb ** 2
        raw_rows(r, c, v, const, 6 * G, "soc")
        cones += [clarabel.SecondOrderConeT(3)] * (2 * G)

        A = sp.vstack(A_blocks).tocsc()
        b = np.concatenate(b_parts)

        # ---- objective
        Pd = np.zeros(ny)
        q = np.zeros(ny)
        Poff_r, Poff_c, Poff_v = [], [], []
        Pd[:nzv] += 2 * base.W_TRUST
        q[isg] += base.W_TIME * s_bar                                # sigma = s_bar (1 + dz)
        if self.terminal_target is not None:                         # w ||(p_end - target)/POS_SCALE||^2
            wt = self.terminal_weight / base.POS_SCALE ** 2
            for a in range(3):
                j = iX(G - 1, a)
                Pd[j] += 2 * wt * zs[j] ** 2
                q[j] += 2 * wt * (X_bar[-1, a] - self.terminal_target[a]) * zs[j]
        if self.terminal_angles is not None:                         # w ((gamma, chi)_end - ref)^2
            for a, ref in zip((4, 5), self.terminal_angles):
                j = iX(G - 1, a)
                Pd[j] += 2 * self.angle_weight * zs[j] ** 2
                q[j] += 2 * self.angle_weight * (X_bar[-1, a] - ref) * zs[j]
        xsV = self.xs[3]
        ws = base.W_SPEED * self.q                                   # W_SPEED sum q_i ((V - Vc)/Vc)^2
        a_s = (X_bar[:, 3] - base.V_CRUISE) / base.V_CRUISE
        b_s = xsV / base.V_CRUISE
        Pd[iX(np.arange(G), 3)] += 2 * ws * b_s ** 2
        q[iX(np.arange(G), 3)] += 2 * ws * a_s * b_s
        wts = self._smooth_weights()                                 # W_SMOOTH sum w_i (dU/us)^2
        w = np.repeat(base.W_SMOOTH * wts, nu)                       # per (interval, control)
        a0 = (np.diff(U_bar, axis=0) / self.us).ravel()
        j0 = (oU + m[:, None] * nu + np.arange(nu)).ravel()          # dz coefficient -1
        j1 = j0 + nu                                                 # dz coefficient +1
        np.add.at(Pd, j1, 2 * w); np.add.at(Pd, j0, 2 * w)
        Poff_r, Poff_c, Poff_v = list(j0), list(j1), list(-2 * w)
        np.add.at(q, j1, 2 * w * a0); np.add.at(q, j0, -2 * w * a0)
        q[nzv:nzv + 2 * Nn] = base.LAMBDA_DYN
        q[nzv + 2 * Nn:] = base.LAMBDA_OBS / base.POS_SCALE
        P = sp.triu(sp.diags(Pd) + sp.csr_matrix((Poff_v, (Poff_r, Poff_c)), shape=(ny, ny))).tocsc()

        settings = clarabel.DefaultSettings()
        settings.verbose = False
        for key, val in SOLVER_OPTS.items():
            setattr(settings, key, val)
        sol = clarabel.DefaultSolver(P, q, A, b, cones, settings).solve()
        if str(sol.status) not in ("Solved", "AlmostSolved"):
            return None
        y = np.asarray(sol.x)
        nu_l1 = float(np.sum(y[nzv:nzv + 2 * Nn]))
        return y[:nzv], nu_l1, float(np.sum(y[nzv + 2 * Nn:]))

    # ---- one start ---------------------------------------------------------------
    def _solve_mesh(self, env, guess):
        """SCP from one guess on the current segment mesh."""
        t0 = time.perf_counter()
        G, S, n = self.G, self.S, self.bb.n
        nx, nu = 6, 3
        nz = 2 * G * nx + G * nu + 1
        oX, oV, oU, isg = 0, G * nx, 2 * G * nx, nz - 1
        xs, us = self.xs, self.us
        vs = xs[None, :] / (self.h_node * self.hloc)[:, None]       # V moves ~ X / local spacing (per node)

        X_bar, V_bar, U_bar, s_bar = guess
        J_bar, def_bar, viol_bar = self.bk_merit(env, X_bar, V_bar, U_bar, s_bar)
        history, status, converged = [], "max_iterations", False
        cap = base.TRUST_CAP
        last_step = cap
        infeas_hist, cost_hist, feasible_at, best_valid = [], [], None, None

        # fixed sparsity pieces
        gi = np.arange(G)
        rX = (gi[:, None, None] * nx + np.arange(nx)[None, :, None]).repeat(nx, 2)
        cX = (oX + gi[:, None, None] * nx + np.arange(nx)[None, None, :]).repeat(nx, 1)
        rU = (gi[:, None, None] * nx + np.arange(nx)[None, :, None]).repeat(nu, 2)
        cU = (oU + gi[:, None, None] * nu + np.arange(nu)[None, None, :]).repeat(nx, 1)
        row_scale = ((self.h_node * self.hloc)[:, None] / xs[None, :]).ravel()
        # integration rows: X_j - X_0 - B V = 0 per segment / state (exact, linear)
        ir, ic, iv = [], [], []
        row = 0
        for s in range(S):
            i = self.idx[s]
            for j in range(1, n + 1):
                for a in range(nx):
                    ir += [row, row]; ic += [oX + i[j] * nx + a, oX + i[0] * nx + a]; iv += [1.0, -1.0]
                    ir += [row] * (n + 1); ic += list(oV + i * nx + a); iv += list(-self.hs[s] * self.bb.B[j])
                    row += 1
        Mint = sp.csr_matrix((iv, (ir, ic)), shape=(row, nz))
        int_scale = np.tile(1 / xs, S * n)

        it = 0
        limit = getattr(self, "time_limit", None)       # optional wall-clock limit per start, s (None = off)
        for it in range(1, base.MAX_ITERS + 1):
            if limit is not None and time.perf_counter() - t0 > limit:
                status = "time_limit"
                it -= 1
                break
            k = s_bar
            f_bar = self.model.f(X_bar, U_bar)
            A, Bu = self.model.jacobians(X_bar, U_bar)
            zbar = np.r_[X_bar.ravel(), V_bar.ravel(), U_bar.ravel(), s_bar]
            zs = np.r_[np.tile(xs, G), vs.ravel(), np.tile(us, G), s_bar]

            # linearized collocation: dV - sigma(A dX + B dU) - f dsigma - nu = sigma f - V
            Mc = sp.csr_matrix((np.r_[np.ones(G * nx), (-k * A).ravel(), (-k * Bu).ravel(),
                                      (-f_bar).ravel()],
                                (np.r_[np.arange(G * nx), rX.ravel(), rU.ravel(), np.arange(G * nx)],
                                 np.r_[oV + np.arange(G * nx), cX.ravel(), cU.ravel(),
                                       np.full(G * nx, isg)])), shape=(G * nx, nz))
            Mc_s = (sp.diags(row_scale) @ Mc @ sp.diags(zs)).tocsr()
            rhs = (k * f_bar - V_bar).ravel() * row_scale

            # exact Birkhoff integration X = X_0 + B V
            int_res = (X_bar[np.r_[[self.idx[s][1:] for s in range(S)]].ravel()]
                       - np.repeat(X_bar[self.idx[:, 0]], n, axis=0)
                       - np.concatenate([self.hs[s] * self.bb.B[1:] @ V_bar[self.idx[s]]
                                         for s in range(S)])).ravel()
            Mi_s = (sp.diags(int_scale) @ Mint @ sp.diags(zs)).tocsr()
            rhs_i = -int_res * int_scale

            # obstacles at check points: n . (dp0 + Bc dVpos) + slack >= -h
            P_bar = self.positions(X_bar, V_bar, self.c_seg, self.c_B)
            C = len(P_bar)
            cache = getattr(self, "_obs_cache", None)
            if cache is not None and cache[0].shape == P_bar.shape and np.array_equal(cache[0], P_bar):
                hn, Gn = cache[1], cache[2]
            else:
                hn, Gn = self.obstacle_eval(env, P_bar)
            step_bound = min(cap, max(2.0 * last_step, 0.02)) if BK_ADAPTIVE_REACH else cap
            reach = np.sqrt(3) * step_bound * base.POS_SCALE + base.PRUNE_BUFFER
            near = hn <= reach
            if BK_MAX_OBS_PER_CHECK is not None and hn.shape[1] > BK_MAX_OBS_PER_CHECK:
                K = BK_MAX_OBS_PER_CHECK
                order = np.argpartition(np.where(near, hn, np.inf), K - 1, axis=1)[:, :K]
                keep = np.zeros_like(near)
                np.put_along_axis(keep, order, True, axis=1)
                near &= keep
            rc, rm = np.nonzero(near)
            Gm = Es = None
            if len(rc):
                R = len(rc)
                nv = Gn[rc, rm]                                        # (R, 3)
                seg = self.c_seg[rc]
                i0 = self.idx[seg, 0]
                rows = np.r_[np.repeat(np.arange(R), 3), np.repeat(np.arange(R), (n + 1) * 3)]
                cols = np.r_[(oX + i0[:, None] * nx + np.arange(3)).ravel(),
                             (oV + self.idx[seg][:, :, None] * nx + np.arange(3)[None, None, :]).ravel()]
                vals = np.r_[nv.ravel(), (self.c_B[rc][:, :, None] * nv[:, None, :]).ravel()]
                Gm = (sp.csr_matrix((vals, (rows, cols)), shape=(R, nz)) @ sp.diags(zs)).tocsr()
                Es = sp.csr_matrix((np.ones(R), (np.arange(R), rc)), shape=(R, C))

            if DIRECT_CLARABEL and base.SOLVER == "CLARABEL":
                try:
                    out = self._direct_subproblem(env, X_bar, V_bar, U_bar, s_bar, zbar, zs, cap,
                                                  Mc_s, rhs, Mi_s, rhs_i, Gm, Es,
                                                  hn[rc, rm] if Gm is not None else None)
                except Exception:
                    out = None
                ok = out is not None
                if ok:
                    dz_val = out[0]
            else:
                dz = cp.Variable(nz)
                nu_v = cp.Variable(G * nx)
                X = X_bar + cp.multiply(xs[None, :], cp.reshape(dz[oX:oV], (G, nx), order="C"))
                U = U_bar + cp.multiply(us[None, :], cp.reshape(dz[oU:isg], (G, nu), order="C"))
                sig = s_bar * (1 + dz[isg])
                cons = self.subproblem_constraints(env, X, U, sig, X_bar, U_bar, s_bar, G)
                cons += [cp.abs(dz) <= cap, Mc_s @ dz - nu_v == rhs, Mi_s @ dz == rhs_i]
                L_expr = self.cost_expr(X, U, sig, G) + base.LAMBDA_DYN * cp.norm1(nu_v)
                if Gm is not None:
                    slack = cp.Variable(C, nonneg=True)
                    cons.append(Gm @ dz + Es @ slack >= -hn[rc, rm])
                    L_expr = L_expr + base.LAMBDA_OBS * cp.sum(slack) / base.POS_SCALE
                prob = cp.Problem(cp.Minimize(L_expr + base.W_TRUST * cp.sum_squares(dz)), cons)
                try:
                    prob.solve(solver=base.SOLVER, canon_backend=cp.SCIPY_CANON_BACKEND, **SOLVER_OPTS)
                    ok = prob.status in ("optimal", "optimal_inaccurate")
                except (cp.error.SolverError, ValueError):
                    ok = False
                if ok:
                    dz_val = dz.value
            if not ok:
                cap /= 2.0
                history.append({"iter": it, "cap": cap, "event": "subproblem_failed"})
                if cap < base.ETA_MIN:
                    status = "solver_error"
                    break
                continue

            zv = zbar + zs * dz_val
            X_new = zv[oX:oV].reshape(G, nx)
            V_new = zv[oV:oU].reshape(G, nx)
            U_new = zv[oU:isg].reshape(G, nu)
            s_new = float(zv[isg])
            J_new, def_new, viol_new = self.bk_merit(env, X_new, V_new, U_new, s_new)
            if not np.isfinite(J_new):
                cap /= 2.0
                history.append({"iter": it, "cap": cap, "event": "rejected_nonfinite"})
                if self.verbose:
                    print(f"    it {it:2d}  step rejected: non-finite; step cap now {cap:.3g}")
                if cap < base.ETA_MIN:
                    status = "nonfinite_step"
                    break
                continue
            step = max(np.max(np.abs(X_new - X_bar) / xs), np.max(np.abs(V_new - V_bar) / vs),
                       np.max(np.abs(U_new - U_bar) / us), abs(s_new - s_bar) / s_bar)
            improvement = J_bar - J_new
            last_step = float(step)
            cap = min(2.0 * cap, base.TRUST_CAP)
            X_bar, V_bar, U_bar, s_bar, J_bar, def_bar, viol_bar = X_new, V_new, U_new, s_new, J_new, def_new, viol_new
            rec = {"iter": it, "J": J_new, "flight_time": s_new, "max_defect": def_new,
                   "max_violation": viol_new, "step": float(step), "n_obstacle_rows": int(len(rc)),
                   "event": "accepted"}
            history.append(rec)
            if self.verbose:
                print(f"    it {it:2d}  J {J_new:10.3f}  tf {s_new:6.1f}s  defect {def_new:.1e}  "
                      f"viol {viol_new:.1e}  step {step:.1e}")

            small = step <= base.TOL_STEP or \
                0 <= improvement <= base.TOL_DECREASE_ABS + base.TOL_DECREASE_REL * abs(J_new)
            if def_new <= base.TOL_DEFECT and viol_new <= base.TOL_VIOLATION:
                feasible_at = feasible_at or it
                infeas_hist = []
                c_new = self.cost(X_new, U_new, s_new)
                if best_valid is None or c_new < best_valid[0]:
                    best_valid = (c_new, X_new, V_new, U_new, s_new, def_new, viol_new)
                cost_hist.append(c_new)
                stalled_cost = (len(cost_hist) > base.COST_WINDOW and
                                cost_hist[-base.COST_WINDOW - 1] - cost_hist[-1]
                                <= base.COST_IMPROVEMENT * abs(cost_hist[-base.COST_WINDOW - 1]))
                polished = base.POLISH_ITERS is not None and it - feasible_at >= base.POLISH_ITERS
                if small or stalled_cost or polished:
                    rec["event"] = "converged"
                    converged = True
                    break
            else:
                cost_hist = []
                infeas_hist.append(def_new + viol_new / base.POS_SCALE)
                if len(infeas_hist) > base.STALL_WINDOW and \
                        infeas_hist[-1] > (1 - base.STALL_IMPROVEMENT) * infeas_hist[-base.STALL_WINDOW - 1]:
                    rec["event"] = status = "stalled"
                    if self.verbose:
                        print("    stalled; moving to the next start")
                    break

        # best valid iterate of this start, if better than (or instead of an invalid) last one
        cur_valid = def_bar <= base.TOL_DEFECT and viol_bar <= base.TOL_VIOLATION
        if best_valid is not None and (not cur_valid or best_valid[0] < self.cost(X_bar, U_bar, s_bar)):
            _, X_bar, V_bar, U_bar, s_bar, def_bar, viol_bar = best_valid
            history.append({"iter": it, "event": "restored_best_valid"})
        return self._finish(env, X_bar, V_bar, U_bar, s_bar, def_bar, viol_bar, converged, status,
                            it, history, time.perf_counter() - t0)

    # ---- adaptive mesh refinement ---------------------------------------------------
    def _solve_from(self, env, guess):
        """One start. Without refinement: SCP on the current mesh. With refinement:
        solve on the coarse mesh, then repeatedly split the segments where the plan is
        hard to resolve and re-solve from the previous solution, keeping the best result."""
        if not self.refining():
            return self._solve_mesh(env, guess)
        t0 = time.perf_counter()
        start_mesh = getattr(self, "_start_mesh", (self.tb.copy(), self.t_est, self.check_dt.copy()))
        res = self._solve_mesh(env, guess)
        rounds, total_iters, history = [], res.iterations, list(res.history)
        best = res
        for rnd in range(1, BK_REFINE_ROUNDS + 1):
            if res.max_defect > base.TOL_DEFECT or res.max_violation > base.TOL_VIOLATION:
                break                                       # not converged at the nodes: refining won't help
            split, why = self._segments_to_split(env, res)
            if not split.any():
                break
            new_tb, new_dt = self._split(split, why)
            if new_tb is None:
                break
            warm = self._remesh(res, new_tb, new_dt)
            rounds.append({"round": rnd, "segments_before": int(len(split)),
                           "split": int(np.sum(split & (why & 1 > 0))),
                           "denser_checks": int(np.sum(split & (why & 2 > 0))),
                           "segments_after": self.S, "nodes": self.G})
            history.append({"event": "refined", **rounds[-1]})
            res = self._solve_mesh(env, warm)
            rounds[-1].update(status=res.status, cost=float(res.cost),
                              verify_defect=float(res.birkhoff["verify_defect"]),
                              dense_margin=float(res.dense_min_margin))
            total_iters += res.iterations
            history += res.history
            if _refine_score(res) > _refine_score(best):
                best = res
        best.iterations, best.history = total_iters, history
        best.solve_time = time.perf_counter() - t0
        best.birkhoff["refinement"] = rounds
        # back to the coarse mesh, so the next start's guess is built on it
        self.setup_segments(len(start_mesh[0]) - 1, start_mesh[1], bounds=start_mesh[0], check_dt=start_mesh[2])
        return best

    def use_mesh_of(self, res):
        """Switch to the mesh a result was solved on (needed before resample() on its native nodes)."""
        bk = res.birkhoff
        self.setup_segments(len(bk["segment_bounds"]) - 1, res.flight_time,
                            bounds=bk["segment_bounds"], check_dt=bk.get("check_dt"))

    def _segments_to_split(self, env, res):
        """Which segments to split, and why (1 = dynamics error, 2 = obstacle margin, 3 = both)."""
        self.use_mesh_of(res)
        bk = res.birkhoff
        X, V, U = (np.asarray(bk[k]) for k in ("X", "V", "U"))
        sigma = res.flight_time
        why = np.zeros(self.S, int)
        r0, d = self.verify_rk4(X, V, U, sigma, per_interval=True)
        seg, _ = self.locate(r0 + 0.5 / len(r0))
        err = np.zeros(self.S)
        np.maximum.at(err, seg, d)
        why[err > BK_REFINE_DEFECT * VERIFY_DEFECT_TOL] |= 1
        Pd = self.positions(X, V, self.d_seg, self.d_B)
        margin = self.obstacle_eval(env, Pd)[0].min(axis=1)
        mseg = np.full(self.S, np.inf)
        np.minimum.at(mseg, self.d_seg, margin)
        why[mseg < BK_REFINE_MARGIN_M] |= 2
        long_enough = 2 * self.hs * sigma >= 2 * BK_MIN_SEGMENT_S
        why[~long_enough] &= ~1                             # too short to split further
        why[self.check_dt <= 2 * BK_DENSE_DT_S] &= ~2       # checks already as dense as allowed
        return why > 0, why

    def _split(self, split, why):
        """New segment bounds and check spacing with the flagged segments halved."""
        n_new = self.S + int(np.sum(split & (why & 1 > 0)))
        if n_new > BK_MAX_SEGMENTS:                         # keep the worst ones only
            keep = np.zeros_like(split)
            keep[np.argsort(-why * split)[:max(0, BK_MAX_SEGMENTS - self.S)]] = True
            split = split & (keep | (why & 1 == 0))
            if not split.any():
                return None, None
        tb, dt = [self.tb[0]], []
        for s in range(self.S):
            d = max(self.check_dt[s] / 2, 2 * BK_DENSE_DT_S) if (split[s] and why[s] & 2) else self.check_dt[s]
            if split[s] and why[s] & 1:                     # dynamics: halve the segment
                tb += [0.5 * (self.tb[s] + self.tb[s + 1]), self.tb[s + 1]]
                dt += [d, d]
            else:                                           # obstacles only: same segment, denser checks
                tb.append(self.tb[s + 1])
                dt.append(d)
        if len(tb) - 1 == self.S and np.allclose(dt, self.check_dt):
            return None, None                               # nothing left to change
        return np.array(tb), np.array(dt)

    def _remesh(self, res, new_tb, new_dt):
        """The solution of `res` on a new mesh. Splitting a segment keeps the same polynomials,
        so the warm start satisfies X = X_0 + B V exactly on the finer mesh."""
        self.use_mesh_of(res)
        bk = res.birkhoff
        X, V, U = (np.asarray(bk[k]) for k in ("X", "V", "U"))
        old = (self.tb, self.hs, self.idx)
        self.setup_segments(len(new_tb) - 1, res.flight_time, bounds=new_tb, check_dt=new_dt)
        tb, hs, idx = old
        seg = np.clip(np.searchsorted(tb, self.r, side="right") - 1, 0, len(hs) - 1)
        tl = np.clip((self.r - tb[seg]) / hs[seg] - 1.0, -1.0, 1.0)
        Xn = X[idx[seg, 0]] + hs[seg][:, None] * np.einsum("cj,cjk->ck", self.bb.integral(tl), V[idx[seg]])
        Vn = np.einsum("cj,cjk->ck", self.bb.lagrange(tl), V[idx[seg]])
        Un = np.einsum("cj,cjk->ck", self.bb.lagrange(tl), U[idx[seg]])
        Xn[0] = X[0]
        Xn = self.integrate_segments(Xn[0], Vn)             # remove round-off
        return Xn, Vn, Un, res.flight_time

    # ---- verification and output ------------------------------------------------
    def resample(self, X, V, U, m):
        """The polynomial trajectory at m uniformly spaced times."""
        r = np.linspace(0, 1, m)
        seg, tl = self.locate(r)
        Xo = X[self.idx[seg, 0]] + self.hs[seg][:, None] * np.einsum("cj,cjk->ck", self.bb.integral(tl),
                                                                     V[self.idx[seg]])
        Uo = np.einsum("cj,cjk->ck", self.bb.lagrange(tl), U[self.idx[seg]])
        return Xo, Uo

    def controls_at(self, U, r):
        """Polynomial controls at normalized times r (array)."""
        seg, tl = self.locate(r)
        return np.einsum("cj,cjk->ck", self.bb.lagrange(tl), U[self.idx[seg]])

    def verify_rk4(self, X, V, U, sigma, per_interval=False):
        """Multiple-shooting check of the plan with its true (polynomial) controls:
        from each of VERIFY_INTERVALS uniform points, integrate the real dynamics with
        RK4 to the next point and compare. Returns the max scaled defect, or with
        per_interval=True (interval start times, max defect of each interval)."""
        M, m = max(VERIFY_INTERVALS, 4 * self.S), VERIFY_SUBSTEPS
        Xo, _ = self.resample(X, V, U, M + 1)
        r0 = np.linspace(0, 1, M + 1)[:-1]
        h = 1.0 / (M * m)                                   # normalized substep
        x = Xo[:-1].copy()
        with np.errstate(all="ignore"):
            for j in range(m):
                t = r0 + j * h
                u0, um, u1 = (self.controls_at(U, t), self.controls_at(U, t + h / 2),
                              self.controls_at(U, t + h))
                k1 = self.model.f(x, u0)
                k2 = self.model.f(x + sigma * h / 2 * k1, um)
                k3 = self.model.f(x + sigma * h / 2 * k2, um)
                k4 = self.model.f(x + sigma * h * k3, u1)
                x = x + sigma * h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            d = np.abs((Xo[1:] - x) / self.xs)
        if per_interval:
            return r0, np.where(np.isfinite(d), d, np.inf).max(axis=1)
        return float(d.max()) if np.all(np.isfinite(d)) else float("inf")

    def open_loop(self, X, U, sigma):
        """Simulate the true dynamics from the start with the polynomial controls."""
        K = max(int(np.ceil(sigma / VERIFY_DT)), 10)
        r = np.linspace(0, 1, 2 * K + 1)
        seg, tl = self.locate(r)
        Ug = np.einsum("cj,cjk->ck", self.bb.lagrange(tl), U[self.idx[seg]])
        h = sigma / K
        x = X[0].copy()
        out = [x.copy()]
        with np.errstate(all="ignore"):
            for i in range(K):
                u0, um, u1 = Ug[2 * i], Ug[2 * i + 1], Ug[2 * i + 2]
                k1 = self.model.f(x, u0)
                k2 = self.model.f(x + h / 2 * k1, um)
                k3 = self.model.f(x + h / 2 * k2, um)
                k4 = self.model.f(x + h * k3, u1)
                x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                out.append(x.copy())
        return np.linspace(0, 1, K + 1), np.array(out)

    def _finish(self, env, X, V, U, sigma, defect, viol, converged, status, it, history, elapsed):
        # dense obstacle check on the polynomial path
        Pd = self.positions(X, V, self.d_seg, self.d_B)
        margin = float(self.obstacle_eval(env, Pd)[0].min())
        dense_tol = base.DENSE_TOLERANCE if base.DENSE_TOLERANCE is not None else 0.25 * env.margin
        # RK4 re-check of the plan with its own polynomial controls; then the uniform resampling
        verify_defect = self.verify_rk4(X, V, U, sigma)
        Xo, Uo = self.resample(X, V, U, OUTPUT_NODES)
        # open-loop simulation from the start
        r_ol, x_ol = self.open_loop(X, U, sigma)
        Xr, _ = self.resample(X, V, U, len(r_ol))
        dev = np.linalg.norm(x_ol[:, :3] - Xr[:, :3], axis=1)
        ol_max = float(np.nanmax(dev)) if np.all(np.isfinite(dev)) else float("inf")

        feasible = defect <= base.TOL_DEFECT and viol <= base.TOL_VIOLATION
        if feasible and margin < -dense_tol:
            status = "dense_check_failed"
        elif feasible and verify_defect > VERIFY_DEFECT_TOL:
            status = "verify_failed"
        elif feasible:
            status = "feasible" if converged else "feasible_unconverged"
        elif converged:
            status = "converged_infeasible"
        res = SolveResult(status.startswith("feasible"), status, it, elapsed, sigma,
                          self.cost(X, U, sigma), defect, viol, margin, Xo, Uo, history)
        res.birkhoff = {"segments": self.S, "degree": self.bb.n, "nodes": self.G,
                        "segment_bounds": self.tb.tolist(), "check_dt": self.check_dt.tolist(),
                        "derivative": "dX/dtau, tau in [0, 1] over the whole flight",
                        "node_times_normalized": self.r.tolist(),
                        "X": X.tolist(), "V": V.tolist(), "U": U.tolist(),
                        "verify_defect": verify_defect, "open_loop_max_dev_m": ol_max,
                        "open_loop_end_error_m": float(dev[-1]) if np.isfinite(dev[-1]) else None}
        return res


# --------------------------------------------------------------------------
# CLI (same as scp_aircraft.py; results in localstore)
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Run Birkhoff-discretized fixed-wing SCP.")
    ap.add_argument("--envs", default=ENV_FILE)
    ap.add_argument("--out", default=RESULTS_FILE)
    ap.add_argument("--feasible-out", default=FEASIBLE_FILE)
    ap.add_argument("--max-envs", type=int, default=None)
    ap.add_argument("--show", type=int, default=None, metavar="INDEX",
                    help="plot a SAVED result without solving anything")
    ap.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=base.VERBOSE)
    ap.add_argument("--refine", action=argparse.BooleanOptionalAction, default=BK_REFINE,
                    help="adaptive mesh refinement (start coarse, split hard segments, re-solve)")
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=UserWarning, module="cvxpy")
    globals()["BK_REFINE"] = args.refine

    if args.show is not None:
        env, r, res = load_plan(args.envs, args.out, args.show)
        print(f"Environment {args.show}: {res.status}, flight {res.flight_time:.1f} s, "
              f"closest approach to an obstacle {plan_clearance(env, res):.1f} m")
        plot_result(env, res, AircraftSCP())
        return

    envs = load_environments(args.envs)[: args.max_envs]
    solver = BirkhoffAircraftSCP(verbose=args.verbose)
    results, feasible = [], []
    print(f"Solving {len(envs)} environments with {solver.name} [{base.AIRFRAME_NAME}] "
          f"(degree {BK_DEGREE} per segment, up to {solver.n_starts} starts each"
          + (f", adaptive mesh from {BK_REFINE_START_S:.0f} s segments)" if BK_REFINE else ")"))
    for i, env in enumerate(envs):
        res = solver.solve(env)
        bk = res.birkhoff
        print(f"  env {i:3d}  {bk['segments']:2d} seg / {bk['nodes']:3d} nodes  "
              f"{res.status:22s} best start {res.best_start + 1:2d}/{len(res.attempts)} "
              f"({sum(c is not None for c in res.attempt_costs)} valid)  cost {res.cost:8.3f}  "
              f"iters {res.iterations:3d}  {res.solve_time:6.1f}s  flight {res.flight_time:6.1f}s  "
              f"margin {res.dense_min_margin:+.2f}  RK4 check {bk['verify_defect']:.1e}  "
              f"open-loop drift {bk['open_loop_max_dev_m']:.1f} m")
        results.append({"env_index": i, "env_file": args.envs, "env_fingerprint": env.fingerprint(),
                        "solver": solver.name, **res.to_dict(), "birkhoff": bk})
        if res.success:
            feasible.append(env)

    out_path = localstore_path(args.out)
    with open(out_path, "w") as f:
        json.dump(results, f)
    feas_path = save_environments(feasible, args.feasible_out)
    print(f"\n{len(feasible)}/{len(envs)} feasible.\n  results -> {out_path}\n"
          f"  feasible environments -> {feas_path}")


if __name__ == "__main__":
    main()