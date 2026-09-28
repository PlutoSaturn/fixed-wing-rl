"""
sanity_check.py
=================
Run this once you've `pip install -r requirements.txt`'d, BEFORE
trusting the framework for real benchmark runs. It checks a handful of
invariants that are easy to silently violate:

  1. Dynamics + discretization: the ZOH-linearized discrete update,
     evaluated at the linearization point, converges to the true
     nonlinear flow map as dt shrinks (validates discretization.py
     independent of cvxpy).
  2. A small obstacle-free point-to-point problem converges and the
     boundary conditions are met to numerical precision.
  3. A problem with an obstacle directly on the straight-line path
     converges to a feasible trajectory that clears the obstacle.
  4. An intentionally-infeasible problem (goal state violates a hard
     box constraint) is reported as NOT successful, with useful
     diagnostics -- i.e. the solver fails LOUDLY, not silently.

This was written without a working cvxpy install available in the
authoring environment (network-restricted sandbox) -- the algorithm
logic was validated there by substituting a mock convex solve, and
`discretization.py`'s numerics were validated directly. This script
is the missing piece: the actual convex-subproblem math, which needs
a real cvxpy + solver. Please run it and skim the output before
building on top of this.
"""

from __future__ import annotations

import numpy as np

from rttg_method_a import (
    SCvxSolver,
    SCvxConfig,
    TrajectoryProblem,
    Obstacle,
    PathConstraints,
    CostWeights,
    FixedWingPointMass3DOF,
    FixedWingParams,
)
from rttg_method_a.discretization import discretize_zoh, flow_map


def check_discretization():
    print("[1] discretization consistency (no cvxpy needed)")
    dyn = FixedWingPointMass3DOF()
    x0 = np.array([0.0, 0.0, 100.0, 20.0, 0.05, 0.1])
    u0 = np.array([0.5, np.deg2rad(30), 6.0])
    A, B, F = dyn.jacobians(x0, u0, np.zeros(0))

    diffs = []
    for dt in [0.4, 0.2, 0.1, 0.05]:
        rc = dyn.f(x0, u0, np.zeros(0)) - A @ x0 - B @ u0
        Ad, Bd, Fd, rd = discretize_zoh(A, B, F, rc, dt)
        x_affine = Ad @ x0 + Bd @ u0 + rd
        x_nonlinear = flow_map(dyn, x0, u0, np.zeros(0), dt, substeps=50)
        diffs.append(np.max(np.abs(x_affine - x_nonlinear)))
    ok = all(diffs[i] > diffs[i + 1] for i in range(len(diffs) - 1))
    print(f"    max|affine - nonlinear| by dt: {['%.2e' % d for d in diffs]}")
    print(f"    monotonically shrinking as dt -> 0: {ok}  (expected: True)")
    assert ok, "Discretization error should shrink as dt decreases -- see discretization.py docstring."
    print("    OK\n")


def check_basic_convergence():
    print("[2] basic point-to-point convergence + boundary conditions")
    dyn = FixedWingPointMass3DOF(FixedWingParams())
    problem = TrajectoryProblem(
        instance_id="sanity-basic",
        dynamics=dyn,
        N=25, t0=0.0, tf=35.0,
        x0=np.array([0.0, 0.0, 120.0, 20.0, 0.0, 0.0]),
        xf=np.array([500.0, 100.0, 120.0, 20.0, 0.0, 0.0]),
        path_constraints=dyn.default_path_bounds(),
        cost=CostWeights(control_effort=1.0),
    )
    sol = SCvxSolver(SCvxConfig(max_iterations=25)).timed_solve(problem)
    print(f"    success={sol.success} iterations={sol.iterations} "
          f"wall_time_s={sol.wall_time_s:.3f} ||nu||={sol.virtual_control_norm:.2e}")
    if sol.x is not None:
        x0_err = np.max(np.abs(sol.x[0] - problem.x0))
        xf_err = np.max(np.abs(sol.x[-1] - problem.xf))
        print(f"    boundary error: x0={x0_err:.2e}  xf={xf_err:.2e}")
    print("    (inspect the numbers above -- no hard assert, since solver")
    print("     defaults may need tuning for your first real cvxpy run)\n")
    return sol


def check_obstacle_avoidance():
    print("[3] obstacle directly on the straight-line path")
    dyn = FixedWingPointMass3DOF(FixedWingParams())
    x0 = np.array([0.0, 0.0, 120.0, 20.0, 0.0, 0.0])
    xf = np.array([500.0, 0.0, 120.0, 20.0, 0.0, 0.0])
    obstacle = Obstacle(center=np.array([250.0, 0.0, 120.0]), radius=60.0)
    problem = TrajectoryProblem(
        instance_id="sanity-obstacle",
        dynamics=dyn, N=30, t0=0.0, tf=35.0, x0=x0, xf=xf,
        path_constraints=dyn.default_path_bounds(),
        obstacles=[obstacle],
        cost=CostWeights(control_effort=1.0),
    )
    sol = SCvxSolver(SCvxConfig(max_iterations=25)).timed_solve(problem)
    if sol.x is not None:
        min_dist = min(
            np.linalg.norm(xk[:3] - obstacle.center) for xk in sol.x
        )
        print(f"    success={sol.success}  min distance to obstacle center="
              f"{min_dist:.2f} m (must clear radius={obstacle.radius} m)")
    else:
        print(f"    success={sol.success}  (no trajectory returned)")
    print()
    return sol


def check_infeasible_reports_failure():
    print("[4] intentionally infeasible instance -> should NOT report success")
    dyn = FixedWingPointMass3DOF(FixedWingParams())
    x0 = np.array([0.0, 0.0, 120.0, 20.0, 0.0, 0.0])
    # goal airspeed far outside the box constraint -> should be infeasible
    xf = np.array([500.0, 0.0, 120.0, 100.0, 0.0, 0.0])
    problem = TrajectoryProblem(
        instance_id="sanity-infeasible",
        dynamics=dyn, N=20, t0=0.0, tf=25.0, x0=x0, xf=xf,
        path_constraints=dyn.default_path_bounds(),
        cost=CostWeights(control_effort=1.0),
    )
    sol = SCvxSolver(SCvxConfig(max_iterations=15)).timed_solve(problem)
    print(f"    success={sol.success} (expected False)  "
          f"max_constraint_violation={sol.max_constraint_violation:.2e}")
    print()
    return sol


if __name__ == "__main__":
    check_discretization()
    check_basic_convergence()
    check_obstacle_avoidance()
    check_infeasible_reports_failure()
    print("Done. Review the numbers above -- especially [2] and [3] -- "
          "before wiring this into the shared benchmark harness.")
