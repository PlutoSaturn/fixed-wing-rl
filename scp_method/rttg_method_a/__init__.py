"""
scp_method
=============
Method A (cold-start SCP / SCvx) for the RTTG benchmark
(Anduril x UC Berkeley Fung Institute capstone, AY26-27).

This package is split into two layers:

1. The **shared harness contract** (`interface.py`): `TrajectoryProblem`,
   `Solution`, `TrajectorySolver`, and the constraint primitives
   (`Obstacle`, `PathConstraints`, `NonconvexConstraint`, `CostWeights`).
   This is the layer your teammates' Methods B/C/D should also import
   and implement against -- it has no SCP-specific code in it, so it
   costs nothing for them to depend on it.

2. The **Method A implementation** itself (`dynamics.py`,
   `discretization.py`, `subproblem.py`, `initial_guess.py`,
   `scvx_solver.py`), which implements the SCvx algorithm from
   Malyuta et al., "Convex Optimization for Trajectory Generation"
   (arXiv:2106.09125), Part II.

Quick start
-----------
    from scp_method import (
        SCvxSolver, SCvxConfig,
        TrajectoryProblem, PathConstraints, Obstacle, CostWeights,
        FixedWingPointMass3DOF, FixedWingParams,
    )

    dynamics = FixedWingPointMass3DOF(FixedWingParams())
    problem = TrajectoryProblem(
        instance_id="demo-0001",
        dynamics=dynamics,
        N=40, t0=0.0, tf=40.0,
        x0=np.array([...]), xf=np.array([...]),
        path_constraints=PathConstraints(...),
        obstacles=[Obstacle(center=..., radius=...)],
        cost=CostWeights(control_effort=1.0),
    )
    solver = SCvxSolver(SCvxConfig())
    solution = solver.timed_solve(problem)
"""
from .scvx_solver import SCvxSolver, SCvxConfig
from .interface import TrajectoryProblem, Obstacle, PathConstraints, CostWeights
from .dynamics import FixedWingPointMass3DOF, FixedWingParams

from .interface import (
    Obstacle,
    NonconvexConstraint,
    PathConstraints,
    CostWeights,
    TrajectoryProblem,
    IterationRecord,
    Solution,
    TrajectorySolver,
)
from .dynamics import (
    DynamicsModel,
    FixedWingParams,
    FixedWingPointMass3DOF,
)
from .scvx_solver import SCvxSolver, SCvxConfig

__all__ = [
    "Obstacle",
    "NonconvexConstraint",
    "PathConstraints",
    "CostWeights",
    "TrajectoryProblem",
    "IterationRecord",
    "Solution",
    "TrajectorySolver",
    "DynamicsModel",
    "FixedWingParams",
    "FixedWingPointMass3DOF",
    "SCvxSolver",
    "SCvxConfig",
]
