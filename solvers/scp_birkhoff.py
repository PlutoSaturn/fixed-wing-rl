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
    python -m solvers.scp_birkhoff --envs c172_envs.json --verbose --show 0
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

    # ---- grid ----------------------------------------------------------------
    def setup_segments(self, S, t_est=200.0):
        n = self.bb.n
        self.S = S
        self.G = S * n + 1                                         # global nodes (shared ends)
        self.idx = np.arange(S)[:, None] * n + np.arange(n + 1)[None, :]     # (S, n+1)
        # normalized time [0, 1] of every global node
        self.r = np.r_[np.concatenate([(s + (self.bb.tau[:-1] + 1) / 2) / S for s in range(S)]), 1.0]
        self.dr = np.diff(self.r)                                  # normalized node spacing
        self.hloc = 2.0 / n                                        # mean local spacing (for scaling)
        # quadrature weights over normalized time [0, 1]
        self.q = np.zeros(self.G)
        for s in range(S):
            self.q[self.idx[s]] += self.bb.w / (2 * S)
        # obstacle check points: uniform in time within each segment, plus the very end
        seg_s = t_est * FLIGHT_TIME_PAD / S
        self.c_seg, self.c_B = self._check_grid(max(8, int(np.ceil(seg_s / BK_CHECK_DT_S))))
        self.d_seg, self.d_B = self._check_grid(max(40, int(np.ceil(seg_s / BK_DENSE_DT_S))))
        self.n_nodes = self.G

    def _check_grid(self, m):
        """m points per segment, uniform in time, plus the final point."""
        t = -1 + 2 * np.arange(m) / m
        seg = np.r_[np.repeat(np.arange(self.S), m), self.S - 1]
        Brows = np.vstack([np.tile(self.bb.integral(t), (self.S, 1)), self.bb.B[-1:]])
        return seg, Brows

    def nodes_for(self, env):
        t_est = np.linalg.norm(env.goal_pos - env.start_pos) / base.V_CRUISE
        S = int(BK_SEGMENTS) if BK_SEGMENTS is not None else max(1, int(np.ceil(t_est / BK_SEGMENT_S)))
        self.setup_segments(S, t_est)
        return self.G

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
            X[i[1:]] = X[i[0]] + self.bb.B[1:] @ V[i]
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
        # V from differentiating each segment's polynomial; shared nodes averaged
        D = self.bb.derivative()
        V = np.zeros_like(X)
        cnt = np.zeros(self.G)
        for s in range(self.S):
            i = self.idx[s]
            V[i] += D @ X[i]
            cnt[i] += 1
        V /= cnt[:, None]
        # re-integrate so X = X_0 + B V holds exactly, then shift V so the end is at the goal
        X = self.integrate_segments(X[0], V)
        V[:, :3] += (p_goal - X[-1, :3]) / (2 * self.S)                 # each segment integrates a constant to 2c
        X = self.integrate_segments(X[0], V)
        # controls that roughly produce the guessed climb and turn rates
        k = sigma / (2 * self.S)
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
        return (base.W_TIME * sigma
                + base.W_SPEED * float(np.sum(self.q * ((X[:, 3] - base.V_CRUISE) / base.V_CRUISE) ** 2))
                + base.W_SMOOTH * float(np.sum(wts[:, None] * (np.diff(U, axis=0) / self.us) ** 2)))

    def cost_expr(self, X, U, sig, N):
        wts = np.sqrt(self._smooth_weights())
        return (base.W_TIME * sig
                + base.W_SPEED * cp.sum(cp.multiply(self.q, cp.square((X[:, 3] - base.V_CRUISE) / base.V_CRUISE)))
                + base.W_SMOOTH * cp.sum_squares(cp.multiply(cp.diff(U, axis=0), wts[:, None] / self.us[None, :])))

    def subproblem_constraints(self, env, X, U, sig, X_bar, U_bar, s_bar, N):
        x0, p_goal = self.boundary(env)
        lo, hi = env.position_bounds()
        tanb = np.tan(np.radians(base.BANK_MAX_DEG))
        gmax = np.radians(base.GAMMA_MAX_DEG)
        cons = [X[0] == x0, X[-1, :3] == p_goal,
                X[:, :3] >= lo, X[:, :3] <= hi,
                X[:, 3] >= base.V_MIN, X[:, 3] <= base.V_MAX,
                cp.abs(X[:, 4]) <= gmax,
                U[:, 0] >= 0,
                U[:, 0] <= base.THRUST_PLAN_FRACTION * (base.T_MAX + base.THRUST_SLOPE * X[:, 3]),
                cp.norm(U[:, 1:], 2, axis=1) <= base.N_MAX,
                cp.abs(U[:, 2]) <= tanb * U[:, 1],
                sig >= base.SIGMA_BOUNDS[0], sig <= base.SIGMA_BOUNDS[1]]
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
            coll = np.abs(V - sigma / (2 * self.S) * f) * self.hloc / self.xs        # scaled collocation defect
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
        for a in range(3):
            eq_r.append(len(eq_k)); eq_c.append(iX(G - 1, a)); eq_v.append(1.0); eq_k.append(-p_goal[a])
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
        groups = []
        for a in range(3):
            groups += [([(iX(nodes, a), -one)], np.full(G, lo[a])),
                       ([(iX(nodes, a), one)], np.full(G, -hi[a]))]
        groups += [([(iX(nodes, 3), -one)], np.full(G, base.V_MIN)),
                   ([(iX(nodes, 3), one)], np.full(G, -base.V_MAX)),
                   ([(iX(nodes, 4), one)], np.full(G, -gmax)),
                   ([(iX(nodes, 4), -one)], np.full(G, -gmax)),
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
    def _solve_from(self, env, guess):
        t0 = time.perf_counter()
        G, S, n = self.G, self.S, self.bb.n
        nx, nu = 6, 3
        nz = 2 * G * nx + G * nu + 1
        oX, oV, oU, isg = 0, G * nx, 2 * G * nx, nz - 1
        xs, us = self.xs, self.us
        vs = xs / self.hloc                                         # V moves ~ X / local spacing

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
        row_scale = np.tile(self.hloc / xs, G)
        # integration rows: X_j - X_0 - B V = 0 per segment / state (exact, linear)
        ir, ic, iv = [], [], []
        row = 0
        for s in range(S):
            i = self.idx[s]
            for j in range(1, n + 1):
                for a in range(nx):
                    ir += [row, row]; ic += [oX + i[j] * nx + a, oX + i[0] * nx + a]; iv += [1.0, -1.0]
                    ir += [row] * (n + 1); ic += list(oV + i * nx + a); iv += list(-self.bb.B[j])
                    row += 1
        Mint = sp.csr_matrix((iv, (ir, ic)), shape=(row, nz))
        int_scale = np.tile(1 / xs, S * n)

        it = 0
        for it in range(1, base.MAX_ITERS + 1):
            k = s_bar / (2 * S)
            f_bar = self.model.f(X_bar, U_bar)
            A, Bu = self.model.jacobians(X_bar, U_bar)
            zbar = np.r_[X_bar.ravel(), V_bar.ravel(), U_bar.ravel(), s_bar]
            zs = np.r_[np.tile(xs, G), np.tile(vs, G), np.tile(us, G), s_bar]

            # linearized collocation: dV - k(A dX + B dU) - f/(2S) dsigma - nu = k f - V
            Mc = sp.csr_matrix((np.r_[np.ones(G * nx), (-k * A).ravel(), (-k * Bu).ravel(),
                                      (-f_bar / (2 * S)).ravel()],
                                (np.r_[np.arange(G * nx), rX.ravel(), rU.ravel(), np.arange(G * nx)],
                                 np.r_[oV + np.arange(G * nx), cX.ravel(), cU.ravel(),
                                       np.full(G * nx, isg)])), shape=(G * nx, nz))
            Mc_s = (sp.diags(row_scale) @ Mc @ sp.diags(zs)).tocsr()
            rhs = ((k * f_bar - V_bar) * self.hloc / xs).ravel()

            # exact Birkhoff integration X = X_0 + B V
            int_res = (X_bar[np.r_[[self.idx[s][1:] for s in range(S)]].ravel()]
                       - np.repeat(X_bar[self.idx[:, 0]], n, axis=0)
                       - np.concatenate([self.bb.B[1:] @ V_bar[self.idx[s]] for s in range(S)])).ravel()
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

    # ---- verification and output ------------------------------------------------
    def resample(self, X, V, U, m):
        """The polynomial trajectory at m uniformly spaced times."""
        r = np.linspace(0, 1, m)
        seg = np.minimum((r * self.S).astype(int), self.S - 1)
        tl = 2 * (r * self.S - seg) - 1
        Xo = X[self.idx[seg, 0]] + np.einsum("cj,cjk->ck", self.bb.integral(tl), V[self.idx[seg]])
        Uo = np.einsum("cj,cjk->ck", self.bb.lagrange(tl), U[self.idx[seg]])
        return Xo, Uo

    def controls_at(self, U, r):
        """Polynomial controls at normalized times r (array)."""
        r = np.clip(np.asarray(r, float), 0.0, 1.0)
        seg = np.minimum((r * self.S).astype(int), self.S - 1)
        tl = 2 * (r * self.S - seg) - 1
        return np.einsum("cj,cjk->ck", self.bb.lagrange(tl), U[self.idx[seg]])

    def verify_rk4(self, X, V, U, sigma):
        """Multiple-shooting check of the plan with its true (polynomial) controls:
        from each of VERIFY_INTERVALS uniform points, integrate the real dynamics with
        RK4 to the next point and compare. Returns the max scaled defect."""
        M, m = VERIFY_INTERVALS, VERIFY_SUBSTEPS
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
        return float(d.max()) if np.all(np.isfinite(d)) else float("inf")

    def open_loop(self, X, U, sigma):
        """Simulate the true dynamics from the start with the polynomial controls."""
        K = max(int(np.ceil(sigma / VERIFY_DT)), 10)
        r = np.linspace(0, 1, 2 * K + 1)
        seg = np.minimum((r * self.S).astype(int), self.S - 1)
        tl = 2 * (r * self.S - seg) - 1
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
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=UserWarning, module="cvxpy")

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
          f"(degree {BK_DEGREE} per segment, up to {solver.n_starts} starts each)")
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