"""
scvx_solver.py
===============
Method A: cold-start Sequential Convex Programming (SCvx).

Implements Malyuta et al., "Convex Optimization for Trajectory
Generation" (arXiv:2106.09125), Part II:
  - Figure 11: the Starting / Iteration / Stopping loop
  - eq. (55): the convex subproblem (see subproblem.py)
  - eq. (60)-(62): nonlinear augmented cost and the accuracy ratio rho
  - Figure 16: the trust-region update rule
  - eq. (66)/(67): the practical stopping criterion
  - "SCvx convergence guarantee" section: lambda continuation (if a
    converged solution still has non-zero virtual control, increase
    lambda by a power of ten and continue -- the paper's own suggested
    practical strategy)

"Cold-start" means no learned warm start anywhere: the initial
reference trajectory is a straight-line interpolation + trim control
guess (initial_guess.py). This is the Tier-A baseline in the RTTG
taxonomy: "guaranteed but slower" -- the trust mechanism that Methods
B-D are compared against.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, List
import time

import numpy as np

from .interface import TrajectoryProblem, Solution, IterationRecord, TrajectorySolver
from .discretization import discretize_zoh, flow_map
from .subproblem import build_and_solve, SubproblemResult
from .initial_guess import straight_line_guess


@dataclass
class SCvxConfig:
    # --- trust-region update rule (Figure 16) ---
    rho0: float = 0.0
    rho1: float = 0.25
    rho2: float = 0.7
    eta0: float = 1e-4      # minimum trust region radius
    eta1: float = 10.0      # maximum trust region radius
    eta_init: float = 1.0
    beta_shrink: float = 2.0
    beta_grow: float = 3.2

    # --- virtual-control penalty weight (lambda), with continuation ---
    lam_init: float = 1e2
    lam_max: float = 1e6
    lam_growth: float = 10.0
    max_lambda_continuations: int = 4

    # --- stopping criterion (eq. 66/67) ---
    max_iterations: int = 30
    state_tol: float = 1e-2        # eq. 66: max per-node state-deviation
    rel_cost_tol: float = 1e-4     # eq. 67: relative nonlinear-cost improvement
    virtual_control_tol: float = 1e-3   # "feasible" means total ||nu|| below this

    # --- numerics ---
    rk4_substeps: int = 5
    finite_diff_eps: float = 1e-6

    solver: Optional[str] = None   # e.g. "CLARABEL", "ECOS", "SCS"; None = cvxpy default
    solver_kwargs: Dict = field(default_factory=dict)

    verbose: bool = False


def _nonlinear_augmented_cost(
    problem: TrajectoryProblem,
    x: np.ndarray,
    u: np.ndarray,
    p: np.ndarray,
    lam: float,
    cfg: SCvxConfig,
) -> Dict[str, float]:
    """eq. (60)/(61): the EXACT cost plus a penalty on ACTUAL constraint
    violations (dynamic defects + nonconvex constraints), evaluated at
    a candidate trajectory with no linearization anywhere. Used both
    for the reference trajectory and the new candidate when computing
    the accuracy ratio rho, and for the final feasibility diagnostics
    reported in `Solution`."""
    N = problem.N
    dt = (problem.tf - problem.t0) / (N - 1)

    control_cost = problem.cost.control_effort * dt * float(np.sum(u ** 2))
    terminal_cost = problem.cost.terminal * float(np.sum((x[-1] - problem.xf) ** 2))
    time_cost = problem.cost.time * (p[0] if problem.free_final_time and len(p) else 0.0)
    convex_cost = control_cost + terminal_cost + time_cost

    max_defect = 0.0
    defect_penalty = 0.0
    for k in range(N - 1):
        x_next_hat = flow_map(problem.dynamics, x[k], u[k], p, dt, cfg.rk4_substeps)
        defect = x[k + 1] - x_next_hat
        defect_penalty += float(np.sum(np.abs(defect)))
        max_defect = max(max_defect, float(np.max(np.abs(defect))))

    constraint_penalty = 0.0
    max_violation = 0.0
    for k in range(N):
        for obs in problem.obstacles:
            idx = obs.position_indices
            dist = np.linalg.norm(x[k, idx] - obs.center)
            viol = max(obs.radius - dist, 0.0)
            constraint_penalty += viol
            max_violation = max(max_violation, viol)
        for nc in problem.extra_constraints:
            g = np.atleast_1d(nc.g(x[k], u[k], p))
            viol = np.maximum(g, 0.0)
            constraint_penalty += float(np.sum(viol))
            if viol.size:
                max_violation = max(max_violation, float(np.max(viol)))

    penalty = lam * (defect_penalty + constraint_penalty)
    return {
        "total": convex_cost + penalty,
        "convex_cost": convex_cost,
        "max_defect": max_defect,
        "max_violation": max_violation,
    }


class SCvxSolver(TrajectorySolver):
    """Method A: cold-start SCP (SCvx). Accepts (and ignores) any
    warm-start hint passed in, by design -- Method A is the "no warm
    start" baseline that owns its own trust mechanism end-to-end."""

    name = "A_coldstart_scp"

    def __init__(self, config: Optional[SCvxConfig] = None):
        self.cfg = config or SCvxConfig()

    def solve(
        self,
        problem: TrajectoryProblem,
        initial_guess: Optional[Dict[str, np.ndarray]] = None,
    ) -> Solution:
        cfg = self.cfg
        t_start = time.perf_counter()

        N = problem.N
        n, m = problem.dynamics.state_dim, problem.dynamics.control_dim
        npar = problem.param_dim()
        dt = (problem.tf - problem.t0) / (N - 1)

        # Cold start: always the naive straight-line guess, regardless
        # of whatever `initial_guess` the harness may have passed in.
        x_ref, u_ref, p_ref = straight_line_guess(problem)

        lam = cfg.lam_init
        history: List[IterationRecord] = []
        it_global = 0
        last_status = "did_not_run"
        last_sub: Optional[SubproblemResult] = None
        nu_norm_at_ref = float("inf")
        converged = False

        for _lam_attempt in range(cfg.max_lambda_continuations + 1):
            eta = cfg.eta_init
            converged = False

            for _it in range(cfg.max_iterations):
                it_global += 1
                x_prev, u_prev, p_prev = x_ref, u_ref, p_ref

                # 1) linearize + exactly ZOH-discretize about the current reference
                A = np.zeros((N - 1, n, n))
                B = np.zeros((N - 1, n, m))
                F = np.zeros((N - 1, n, npar))
                r = np.zeros((N - 1, n))
                for k in range(N - 1):
                    Ac, Bc, Fc = problem.dynamics.jacobians(
                        x_prev[k], u_prev[k], p_prev, eps=cfg.finite_diff_eps
                    )
                    fc = problem.dynamics.f(x_prev[k], u_prev[k], p_prev)
                    rc = fc - Ac @ x_prev[k] - Bc @ u_prev[k]
                    if npar > 0:
                        rc = rc - Fc @ p_prev
                    Ad, Bd, Fd, rd = discretize_zoh(Ac, Bc, Fc, rc, dt)
                    A[k], B[k], r[k] = Ad, Bd, rd
                    if npar > 0:
                        F[k] = Fd

                # 2) solve the convex subproblem (eq. 55)
                sub = build_and_solve(
                    problem, x_prev, u_prev, p_prev, A, B, F, r,
                    eta=eta, lam=lam,
                    solver=cfg.solver, solver_kwargs=cfg.solver_kwargs,
                )
                last_status = sub.status
                last_sub = sub

                if sub.status not in ("optimal", "optimal_inaccurate"):
                    # Subproblem infeasible/unbounded/solver error: treat
                    # like an outright-rejected step (Fig. 16, Case 1) and
                    # shrink the trust region before retrying.
                    eta_new = max(cfg.eta0, eta / cfg.beta_shrink)
                    history.append(IterationRecord(
                        iteration=it_global, trust_region=eta_new, rho=None,
                        cost_linear=float("nan"), cost_nonlinear=float("nan"),
                        virtual_control_norm=float("nan"), max_defect=float("nan"),
                        accepted=False, subproblem_solve_time_s=sub.solve_time_s,
                        subproblem_status=sub.status,
                    ))
                    if eta_new >= eta:  # already at the floor; can't shrink further
                        break
                    eta = eta_new
                    continue

                # 3) accuracy ratio rho (eq. 62), via nonlinear augmented cost
                J_bar = _nonlinear_augmented_cost(problem, x_prev, u_prev, p_prev, lam, cfg)
                J_star = _nonlinear_augmented_cost(problem, sub.x, sub.u, sub.p, lam, cfg)
                L_star = sub.linear_cost
                nu_norm_candidate = float(np.sum(np.abs(sub.nu))) + (
                    float(np.sum(sub.nu_s)) if sub.nu_s is not None else 0.0
                )

                denom = J_bar["total"] - L_star

                if denom <= 1e-9:
                    # Predicted improvement ~ 0: the model already matches
                    # the reference (eq. 63/65 discussion) -> converged.
                    x_ref, u_ref, p_ref = sub.x, sub.u, sub.p
                    nu_norm_at_ref = nu_norm_candidate
                    converged = True
                    history.append(IterationRecord(
                        iteration=it_global, trust_region=eta, rho=None,
                        cost_linear=L_star, cost_nonlinear=J_star["total"],
                        virtual_control_norm=nu_norm_candidate, max_defect=J_star["max_defect"],
                        accepted=True, subproblem_solve_time_s=sub.solve_time_s,
                        subproblem_status=sub.status,
                    ))
                    break

                rho = (J_bar["total"] - J_star["total"]) / denom
                accepted = rho >= cfg.rho0

                if accepted:
                    x_ref, u_ref, p_ref = sub.x, sub.u, sub.p
                    nu_norm_at_ref = nu_norm_candidate
                # else: x_ref/u_ref/p_ref remain x_prev/u_prev/p_prev (unchanged)

                if rho < cfg.rho0:
                    eta = max(cfg.eta0, eta / cfg.beta_shrink)
                elif rho < cfg.rho1:
                    eta = max(cfg.eta0, eta / cfg.beta_shrink)
                elif rho < cfg.rho2:
                    pass  # unchanged
                else:
                    eta = min(cfg.eta1, cfg.beta_grow * eta)

                history.append(IterationRecord(
                    iteration=it_global, trust_region=eta, rho=float(rho),
                    cost_linear=L_star, cost_nonlinear=J_star["total"],
                    virtual_control_norm=nu_norm_candidate, max_defect=J_star["max_defect"],
                    accepted=accepted, subproblem_solve_time_s=sub.solve_time_s,
                    subproblem_status=sub.status,
                ))

                if accepted:
                    traj_change = float(np.max(np.linalg.norm(sub.x - x_prev, axis=1)))
                    if npar > 0:
                        traj_change += float(np.linalg.norm(sub.p - p_prev))
                    rel_cost_improve = abs(J_bar["total"] - J_star["total"]) / max(
                        abs(J_bar["total"]), 1e-9
                    )
                    if traj_change <= cfg.state_tol or rel_cost_improve <= cfg.rel_cost_tol:
                        converged = True
                        break

                if cfg.verbose:
                    print(
                        f"[iter {it_global:3d}] lam={lam:.0e} eta={eta:.3e} "
                        f"rho={rho:+.3f} accepted={accepted} "
                        f"||nu||={nu_norm_candidate:.3e} status={sub.status}"
                    )

            # end inner SCvx loop
            if converged and nu_norm_at_ref <= cfg.virtual_control_tol:
                break  # feasible and converged -- done
            elif converged:
                # Converged to a stationary point that still relies on
                # virtual control -- per the paper, try a larger penalty.
                lam = min(cfg.lam_max, lam * cfg.lam_growth)
                continue
            else:
                # Ran out of iterations, or the subproblem kept failing.
                # More lambda won't fix either of those -- stop here and
                # report the best (infeasible/non-converged) trajectory.
                break

        wall_time = time.perf_counter() - t_start
        final_eval = _nonlinear_augmented_cost(problem, x_ref, u_ref, p_ref, lam, cfg)
        feasible = converged and (nu_norm_at_ref <= cfg.virtual_control_tol)

        return Solution(
            method_name=self.name,
            instance_id=problem.instance_id,
            success=converged and feasible,
            converged=converged,
            feasible=feasible,
            t=np.linspace(problem.t0, problem.tf, N),
            x=x_ref, u=u_ref, p=p_ref,
            cost=final_eval["convex_cost"],
            iterations=it_global,
            wall_time_s=wall_time,
            virtual_control_norm=(
                nu_norm_at_ref if np.isfinite(nu_norm_at_ref) else float("nan")
            ),
            max_constraint_violation=final_eval["max_violation"],
            solver_status=last_status,
            history=history,
            metadata={"final_lambda": lam},
        )
