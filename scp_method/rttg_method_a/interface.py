"""
interface.py
=============
The common "harness contract" for the RTTG (Real-Time Trajectory
Generation) benchmark. This is the schema that should be shared across
all four methods:

    A  cold-start SCP           (this package, scvx_solver.py)
    B  Birkhoff-discretized SCP (teammate)
    C  learned warm start       (teammate/you, later)
    D  amortized neural gen.    (teammate/you, later)

Design goals
------------
* Phase-01's benchmark-family generator emits `TrajectoryProblem`
  instances (or something trivially convertible to one).
* Every method implements `TrajectorySolver.solve(problem) -> Solution`.
  The phase-03 comparative study only ever calls that one method, so
  it does not need to know anything about SCP, Birkhoff polynomials,
  or neural nets.
* `Solution` carries everything phase-03 needs directly: feasibility,
  wall-clock time (for the 99.9th-percentile tail statistic),
  constraint-violation magnitude on failure, and a slot for the
  optimality gap (filled in later by phase-03 against a converged
  offline reference).

Keep this file dependency-light (numpy only, plus the dynamics ABC)
so every method's package can import it without pulling in
method-specific dependencies (cvxpy, torch, etc). If your teammates
are in a different repo/package, just copy this one file over --
it's the single point of integration.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Dict, Any
import time

import numpy as np

from .dynamics import DynamicsModel


# ---------------------------------------------------------------------------
# Constraint primitives
# ---------------------------------------------------------------------------

@dataclass
class Obstacle:
    """A spherical (or axis-aligned-ellipsoidal, via `radius` per-axis if
    you pass an array) keep-out zone in position space:

        || (p - center) / radius ||_2 >= 1

    `position_indices` selects which state components are "position"
    (e.g. [0, 1, 2] for x, y, altitude) so this works regardless of the
    full state vector's layout (fixed-wing, quadrotor, whatever).
    """
    center: np.ndarray
    radius: float
    position_indices: List[int] = field(default_factory=lambda: [0, 1, 2])


@dataclass
class NonconvexConstraint:
    """A generic nonconvex inequality constraint g(x, u, p) <= 0
    (vector-valued, want every component <= 0). Linearized internally
    via finite differences at each SCvx iteration. Use this for
    anything beyond simple sphere obstacles: no-fly polygons, keep-in
    cones, sensor-pointing constraints, etc.
    """
    name: str
    g: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
    dim: int


@dataclass
class PathConstraints:
    """Convex box bounds, applied at every temporal node. `None` means
    unconstrained in that direction."""
    x_min: Optional[np.ndarray] = None
    x_max: Optional[np.ndarray] = None
    u_min: Optional[np.ndarray] = None
    u_max: Optional[np.ndarray] = None


@dataclass
class CostWeights:
    """Weights for the (convex, unlinearized) part of the SCvx cost.

        J = terminal * ||x_N - x_f||^2      (usually 0 -- see note below)
          + control_effort * dt * sum_k ||u_k||^2
          + time * tf                        (only used if free_final_time)

    Note: because boundary conditions are enforced as hard equality
    constraints in this baseline (see subproblem.py), the terminal term
    is a no-op unless you relax the final-state constraint. It's kept
    here as an extension point.
    """
    control_effort: float = 1.0
    terminal: float = 0.0
    time: float = 0.0


# ---------------------------------------------------------------------------
# Problem / Solution schema
# ---------------------------------------------------------------------------

@dataclass
class TrajectoryProblem:
    """One benchmark instance. Phase-01's generator should emit these
    (nominal AND out-of-distribution instances alike -- `is_ood` just
    tags them for the phase-03 study)."""

    instance_id: str
    dynamics: DynamicsModel

    N: int                          # number of temporal nodes
    t0: float
    tf: float                       # nominal / initial-guess final time

    x0: np.ndarray                  # fixed initial state
    xf: np.ndarray                  # fixed final state
    free_final_time: bool = False   # extension point; see README

    path_constraints: PathConstraints = field(default_factory=PathConstraints)
    obstacles: List[Obstacle] = field(default_factory=list)
    extra_constraints: List[NonconvexConstraint] = field(default_factory=list)
    cost: CostWeights = field(default_factory=CostWeights)

    is_ood: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def param_dim(self) -> int:
        return 1 if self.free_final_time else 0


@dataclass
class IterationRecord:
    """One SCP iteration's diagnostics. Useful for phase-03 tail
    statistics (e.g. "how many iterations before the tail cases
    converge?") and for debugging non-convergence."""
    iteration: int
    trust_region: float
    rho: Optional[float]
    cost_linear: float
    cost_nonlinear: float
    virtual_control_norm: float
    max_defect: float
    accepted: bool
    subproblem_solve_time_s: float
    subproblem_status: str


@dataclass
class Solution:
    """What every method (A-D) returns for one TrajectoryProblem.

    Phase-03 needs, per the slides: 99.9th-%ile solve time (aggregate
    `wall_time_s` across many `Solution`s), feasibility rate (aggregate
    `success`), OOD degradation (compare `success`/`wall_time_s` split
    by `TrajectoryProblem.is_ood`), constraint-violation magnitude on
    failure (`max_constraint_violation`), and optimality gap
    (`optimality_gap`, filled in by phase-03 against a converged
    offline reference solve).
    """
    method_name: str
    instance_id: str

    success: bool                   # converged AND feasible (||nu|| ~ 0)
    converged: bool                 # stopping criterion triggered at all
    feasible: bool                  # virtual control ~ 0 at the returned solution

    t: Optional[np.ndarray] = None
    x: Optional[np.ndarray] = None
    u: Optional[np.ndarray] = None
    p: Optional[np.ndarray] = None

    cost: float = float("nan")
    iterations: int = 0
    wall_time_s: float = float("nan")

    virtual_control_norm: float = float("nan")
    max_constraint_violation: float = float("nan")
    optimality_gap: Optional[float] = None

    solver_status: str = ""
    history: List[IterationRecord] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Solver interface
# ---------------------------------------------------------------------------

class TrajectorySolver(ABC):
    """Every method (A-D) implements this. The benchmark harness only
    ever calls `.solve(problem)` (or `.timed_solve(problem)`)."""

    name: str = "unnamed-method"

    @abstractmethod
    def solve(
        self,
        problem: TrajectoryProblem,
        initial_guess: Optional[Dict[str, np.ndarray]] = None,
    ) -> Solution:
        """Solve one TrajectoryProblem.

        `initial_guess`, if given, is a dict with optional keys
        'x', 'u', 'p' (each an ndarray) -- e.g. a network's warm start
        for Method C. Methods that don't use a warm start (Method A)
        should still accept the argument (and may ignore it) so the
        harness can call every method identically.
        """
        raise NotImplementedError

    def timed_solve(self, problem: TrajectoryProblem, **kwargs) -> Solution:
        """Convenience wrapper giving a clean wall-clock time even if a
        subclass forgets to set `Solution.wall_time_s` itself. Prefer
        calling this from the benchmark harness."""
        t_start = time.perf_counter()
        solution = self.solve(problem, **kwargs)
        elapsed = time.perf_counter() - t_start
        if solution.wall_time_s is None or not np.isfinite(solution.wall_time_s):
            solution.wall_time_s = elapsed
        return solution
