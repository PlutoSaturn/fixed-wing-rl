"""
subproblem.py
==============
Builds and solves the convex subproblem solved at every SCvx iteration
-- the analog of Problem 55 in Malyuta et al., "Convex Optimization
for Trajectory Generation" (arXiv:2106.09125).

Design simplification vs. the full paper treatment: boundary
conditions (x0, xf) are enforced as HARD equality constraints rather
than via virtual control (nu_ic, nu_tc in the paper). This matches how
most trajectory-generation benchmarks pose problems (fixed start and
goal states) and keeps the subproblem simpler. Dynamic feasibility and
nonconvex path/obstacle constraints still use virtual control / slack
exactly as in the paper -- those are what actually cause subproblem
infeasibility in practice. If your benchmark family needs soft/free
boundary conditions, that's a documented extension point (add nu_ic,
nu_tc variables and penalize them the same way nu_s is handled below).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import cvxpy as cp

from .interface import TrajectoryProblem, Obstacle, NonconvexConstraint


@dataclass
class SubproblemResult:
    x: np.ndarray              # (N, n)
    u: np.ndarray              # (N, m)
    p: np.ndarray              # (npar,)
    nu: np.ndarray              # dynamics virtual control, (N-1, n)
    nu_s: Optional[np.ndarray]  # nonconvex-constraint slack, (N, n_nc) or None
    linear_cost: float          # L_lambda(x*, u*, p*, nu*, nu_s*), eq. (52)
    status: str
    solve_time_s: float


def _linearize_obstacle(obs: Obstacle, x_ref_k: np.ndarray) -> Tuple[np.ndarray, float]:
    """Linearize the keep-out constraint ||p - center|| >= radius around
    a reference position as a supporting hyperplane:
        n^T p >= n^T center + radius,   n = (p_ref - center) / ||p_ref - center||
    """
    idx = obs.position_indices
    p_ref = x_ref_k[idx]
    delta = p_ref - obs.center
    dist = np.linalg.norm(delta)
    if dist < 1e-6:
        # Degenerate: reference sits on the obstacle center. Push the
        # linearized constraint in an arbitrary fixed direction so the
        # subproblem stays well-posed; virtual control (nu_s) will pick
        # up the slack until the trajectory moves away from the center.
        n = np.zeros_like(delta)
        n[0] = 1.0
        dist = 1e-6
    else:
        n = delta / dist
    rhs = float(n @ obs.center + obs.radius)
    return n, rhs


def _finite_diff_constraint_jacobian(
    g, x0: np.ndarray, u0: np.ndarray, p0: np.ndarray, eps: float = 1e-6
):
    """Central finite-difference Jacobians of a generic constraint
    g(x, u, p) -> R^dim, evaluated at (x0, u0, p0)."""
    n, m = len(x0), len(u0)
    npar = 0 if p0 is None else len(p0)
    g0 = np.atleast_1d(g(x0, u0, p0))
    d = g0.shape[0]

    Jx = np.zeros((d, n))
    for i in range(n):
        dx = np.zeros(n)
        dx[i] = eps
        Jx[:, i] = (np.atleast_1d(g(x0 + dx, u0, p0)) - np.atleast_1d(g(x0 - dx, u0, p0))) / (2 * eps)

    Ju = np.zeros((d, m))
    for i in range(m):
        du = np.zeros(m)
        du[i] = eps
        Ju[:, i] = (np.atleast_1d(g(x0, u0 + du, p0)) - np.atleast_1d(g(x0, u0 - du, p0))) / (2 * eps)

    Jp = np.zeros((d, npar))
    for i in range(npar):
        dp = np.zeros(npar)
        dp[i] = eps
        Jp[:, i] = (np.atleast_1d(g(x0, u0, p0 + dp)) - np.atleast_1d(g(x0, u0, p0 - dp))) / (2 * eps)

    return g0, Jx, Ju, Jp


def build_and_solve(
    problem: TrajectoryProblem,
    x_ref: np.ndarray,   # (N, n)
    u_ref: np.ndarray,   # (N, m)
    p_ref: np.ndarray,   # (npar,)
    A: np.ndarray,       # (N-1, n, n)   discretized dynamics per interval
    B: np.ndarray,       # (N-1, n, m)
    F: np.ndarray,       # (N-1, n, npar)
    r: np.ndarray,       # (N-1, n)
    eta: float,           # trust region radius
    lam: float,           # virtual-control penalty weight (lambda)
    solver: Optional[str] = None,
    solver_kwargs: Optional[dict] = None,
) -> SubproblemResult:
    dyn = problem.dynamics
    n, m = dyn.state_dim, dyn.control_dim
    N = problem.N
    npar = problem.param_dim()
    dt = (problem.tf - problem.t0) / (N - 1)

    x = cp.Variable((N, n))
    u = cp.Variable((N, m))
    p = cp.Variable(npar) if npar > 0 else None
    nu = cp.Variable((N - 1, n))  # dynamics virtual control (eq. 47a)

    n_nc = len(problem.obstacles) + sum(c.dim for c in problem.extra_constraints)
    nu_s = cp.Variable((N, n_nc), nonneg=True) if n_nc > 0 else None

    constraints = []

    # --- dynamics: linearized, ZOH-discretized, with virtual control ---
    for k in range(N - 1):
        rhs = A[k] @ x[k] + B[k] @ u[k] + r[k] + nu[k]
        if npar > 0:
            rhs = rhs + F[k] @ p
        constraints.append(x[k + 1] == rhs)

    # --- boundary conditions (hard; see module docstring) ---
    constraints.append(x[0] == problem.x0)
    constraints.append(x[N - 1] == problem.xf)

    # --- convex box path constraints ---
    # Use explicit integer indices (via np.where) rather than boolean
    # masks: fancy integer indexing is reliably supported across cvxpy
    # versions, whereas boolean-array indexing support has varied.
    pc = problem.path_constraints
    if pc.x_min is not None:
        idx = np.where(np.isfinite(pc.x_min))[0]
        if idx.size:
            constraints.append(x[:, idx] >= pc.x_min[idx])
    if pc.x_max is not None:
        idx = np.where(np.isfinite(pc.x_max))[0]
        if idx.size:
            constraints.append(x[:, idx] <= pc.x_max[idx])
    if pc.u_min is not None:
        idx = np.where(np.isfinite(pc.u_min))[0]
        if idx.size:
            constraints.append(u[:, idx] >= pc.u_min[idx])
    if pc.u_max is not None:
        idx = np.where(np.isfinite(pc.u_max))[0]
        if idx.size:
            constraints.append(u[:, idx] <= pc.u_max[idx])

    # --- trust region: per-node L2 deviation from the reference ---
    # (eq. 51g in the paper; see Fig. 13's ||z - z_bar||_2 <= eta)
    for k in range(N):
        dev = cp.norm(x[k] - x_ref[k], 2) + cp.norm(u[k] - u_ref[k], 2)
        if p is not None:
            dev = dev + cp.norm(p - p_ref, 2)
        constraints.append(dev <= eta)

    # --- nonconvex constraints: linearized per node, slacked ---
    if n_nc > 0:
        for k in range(N):
            row = 0
            for obs in problem.obstacles:
                a, b = _linearize_obstacle(obs, x_ref[k])
                idx = obs.position_indices
                constraints.append(a @ x[k, idx] >= b - nu_s[k, row])
                row += 1
            for nc in problem.extra_constraints:
                g0, Jx, Ju, Jp = _finite_diff_constraint_jacobian(
                    nc.g, x_ref[k], u_ref[k], p_ref
                )
                lin = g0 + Jx @ (x[k] - x_ref[k]) + Ju @ (u[k] - u_ref[k])
                if p is not None and Jp.shape[1] > 0:
                    lin = lin + Jp @ (p - p_ref)
                constraints.append(lin <= nu_s[k, row : row + nc.dim])
                row += nc.dim

    # --- cost (eq. 49/52): convex part + virtual-control penalty ---
    control_cost = problem.cost.control_effort * dt * cp.sum_squares(u)
    time_cost = (
        problem.cost.time * p[0] if (problem.free_final_time and p is not None) else 0.0
    )
    penalty = lam * cp.sum(cp.abs(nu))
    if nu_s is not None:
        penalty = penalty + lam * cp.sum(nu_s)

    objective = cp.Minimize(control_cost + time_cost + penalty)
    cvx_problem = cp.Problem(objective, constraints)

    kwargs = dict(solver_kwargs or {})
    if solver is not None:
        kwargs["solver"] = solver

    import time as _time

    t0 = _time.perf_counter()
    try:
        cvx_problem.solve(**kwargs)
        status = cvx_problem.status
    except cp.error.SolverError:
        status = "solver_error"
    solve_time = _time.perf_counter() - t0

    if status not in ("optimal", "optimal_inaccurate"):
        return SubproblemResult(
            x=x_ref, u=u_ref, p=p_ref,
            nu=np.zeros((N - 1, n)),
            nu_s=np.zeros((N, n_nc)) if n_nc > 0 else None,
            linear_cost=float("nan"),
            status=status,
            solve_time_s=solve_time,
        )

    return SubproblemResult(
        x=x.value,
        u=u.value,
        p=(p.value if p is not None else np.zeros(0)),
        nu=nu.value,
        nu_s=(nu_s.value if nu_s is not None else None),
        linear_cost=float(cvx_problem.value),
        status=status,
        solve_time_s=solve_time,
    )
