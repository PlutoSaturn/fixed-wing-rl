# RTTG Benchmark -- Method A: Cold-Start SCP (SCvx)

Anduril x UC Berkeley Fung Institute capstone, AY26-27.
This is the Method-A (cold-start SCP) implementation, plus the shared
`interface.py` contract that Methods B (Birkhoff SCP), C (learned warm
start), and D (amortized generator) should also implement against.

Algorithm reference: Malyuta, Reynolds, Szmuk, Lew, Bonalli, Pavone,
Açıkmeşe, **"Convex Optimization for Trajectory Generation"**,
arXiv:2106.09125 (the article you shared) -- specifically Part II,
"The SCvx Algorithm." Equation/figure numbers in the code comments
refer to this paper.

## Layout

```
rttg_method_a/
  interface.py        Shared harness contract: TrajectoryProblem, Solution,
                       TrajectorySolver ABC, Obstacle, PathConstraints,
                       NonconvexConstraint, CostWeights.
                       <-- teammates working on Methods B/C/D should import
                           this file (or copy it) so every method is a
                           drop-in TrajectorySolver for phase-03.

  dynamics.py          DynamicsModel ABC (pluggable vehicle model) +
                       FixedWingPointMass3DOF, a 3-DOF point-mass fixed-wing
                       model with Skywalker-1900-class placeholder params.

  discretization.py    Van Loan exact ZOH discretization of the linearized
                       dynamics (for the convex subproblem) + RK4 flow-map
                       propagation of the TRUE nonlinear dynamics (for
                       defect/feasibility checking). See the docstring in
                       this file for an important design note on why these
                       are deliberately two different discretizations.

  subproblem.py        Builds and solves the convex subproblem (paper's
                       eq. 55) with cvxpy: linearized dynamics + virtual
                       control, box path constraints, linearized/slacked
                       obstacle & nonconvex constraints, trust region.

  initial_guess.py     The "cold start": straight-line interpolation +
                       trim control. No warm start, on purpose.

  scvx_solver.py        SCvxSolver(TrajectorySolver) -- the main iteration
                       loop (Fig. 11), trust-region update (Fig. 16),
                       accuracy ratio rho (eq. 62), stopping criterion
                       (eq. 66/67), and lambda continuation.

demo.py                Runnable example: fixed-wing point-to-point with a
                       small obstacle field, with an optional matplotlib plot.
sanity_check.py        Run this FIRST after installing cvxpy -- see below.
requirements.txt
```

## Quick start

```bash
pip install -r requirements.txt
python sanity_check.py     # run this first -- see "Testing status" below
python demo.py
```

```python
import numpy as np
from scp_method import (
    SCvxSolver, SCvxConfig,
    TrajectoryProblem, Obstacle, CostWeights,
    FixedWingPointMass3DOF, FixedWingParams,
)

dynamics = FixedWingPointMass3DOF(FixedWingParams())
problem = TrajectoryProblem(
    instance_id="bench-0001",
    dynamics=dynamics,
    N=30, t0=0.0, tf=40.0,
    x0=np.array([0., 0., 120., 20., 0., 0.]),
    xf=np.array([800., 150., 120., 20., 0., 0.]),
    path_constraints=dynamics.default_path_bounds(),
    obstacles=[Obstacle(center=np.array([250., 40., 120.]), radius=60.)],
    cost=CostWeights(control_effort=1.0),
)

solution = SCvxSolver(SCvxConfig()).timed_solve(problem)
print(solution.success, solution.wall_time_s, solution.iterations)
```

## How this maps to the paper

| Paper | Code |
|---|---|
| Fig. 11 (Starting / Iteration / Stopping loop) | `SCvxSolver.solve()` main loop |
| eq. 45-51 (linearize + virtual control + trust region) | `subproblem.py::build_and_solve` |
| eq. 55 (discrete-time convex subproblem) | same, plus `discretization.py::discretize_zoh` for (A_k,B_k,F_k,r_k) |
| eq. 56/59 (defect, flow map) | `discretization.py::flow_map` (RK4) |
| eq. 60-61 (nonlinear augmented cost) | `scvx_solver.py::_nonlinear_augmented_cost` |
| eq. 62-64 (accuracy ratio rho) | inline in `SCvxSolver.solve()` |
| Fig. 16 (trust-region update rule) | inline in `SCvxSolver.solve()`, the `if rho < ...` block |
| eq. 65-67 (stopping criterion) | inline, `traj_change` / `rel_cost_improve` check |
| "if virtual control nonzero, increase lambda" | the outer `_lam_attempt` continuation loop |

## Deliberate simplifications (read before extending)

1. **Boundary conditions are hard equality constraints**, not virtual-
   control-relaxed (paper's `nu_ic`, `nu_tc`). This matches how most
   trajectory benchmarks pose problems (fixed start/goal) and keeps the
   subproblem smaller. If your benchmark family needs soft/free
   boundary states, add `nu_ic`/`nu_tc` variables in `subproblem.py`
   the same way `nu_s` is handled for path constraints.

2. **Frozen-Jacobian discretization, not the paper's Appendix
   state-transition-matrix-along-the-trajectory approach.** We
   linearize once per node and discretize that frozen LTI system
   exactly (Van Loan trick). The paper's Appendix computes a locally
   time-varying linearization by integrating the STM together with the
   reference trajectory across each interval -- more accurate, more
   code. I picked the simpler version deliberately: it makes Method A
   the "simple numerics" baseline, and your teammate's Birkhoff SCP
   (Method B: "same outer loop, better numerics") is a meaningful
   upgrade *because* Method A isn't already doing the fancy thing.
   `discretization.py`'s docstring has the full argument, and
   `sanity_check.py` [1] verifies the resulting discretization error
   shrinks as O(dt^3) as N grows, which is what you'd want to show in
   the writeup regardless of which numerics you pick.

3. **No free-final-time support is wired up end-to-end.** The `p`
   parameter vector and `free_final_time` flag exist throughout the
   plumbing (so the shapes are right), but nothing currently makes use
   of it meaningfully (`FixedWingPointMass3DOF` doesn't scale time by
   `p`). If a benchmark instance needs free final time, extend
   `DynamicsModel.f` to accept a time-scaling parameter and thread it
   through -- the discretization and subproblem code already accept
   nonzero `param_dim()`.

4. **Fixed-wing dynamics parameters are placeholders**, not measured
   Skywalker 1900 specs. Swap `FixedWingParams` fields for values
   derived from your ArduPilot setup (SITL params, or a drag-polar fit
   from flight-log airspeed/throttle data) before treating results as
   physically meaningful for that airframe.

## Testing status (please read)

I don't have network access in the environment I built this in, so I
could not `pip install cvxpy` to run the actual convex solve end-to-end.
What I *did* verify directly:

- `dynamics.py`: trim-guess control produces near-zero state derivative
  (a valid steady-state level-flight trim).
- `discretization.py`: the ZOH-linearized discrete step converges to
  the true nonlinear RK4 flow map at the expected O(dt^3) rate as dt
  shrinks (confirms the Van Loan discretization and the linearization
  are both implemented correctly, and that the "defect" diagnostic is
  measuring something real).
- `subproblem.py`'s pure-numpy helper functions (obstacle linearization,
  finite-difference constraint Jacobian) against hand-computed
  expected values.
- `scvx_solver.py`'s full iteration loop, trust-region update,
  lambda-continuation, and `Solution` construction, by substituting a
  mock convex solve -- this exercises every branch (accept/reject,
  shrink/grow, converged-but-infeasible, ran-out-of-iterations) without
  crashing and with correct output shapes and exact boundary-condition
  satisfaction.

What I could **not** verify: the actual cvxpy problem in
`subproblem.py::build_and_solve` solves to the right answer (i.e., that
I got the cvxpy syntax and constraint formulation exactly right end to
end). Please run `python sanity_check.py` as your first step -- it's
built to catch the likely failure modes (non-convergence, boundary
conditions not met, obstacle not cleared, infeasible instances not
correctly flagged) with printed diagnostics, not just pass/fail asserts,
since solver-tolerance tuning is expected on a first real run.

## Suggested next steps

- Run `sanity_check.py`, then `demo.py`, and eyeball the printed
  diagnostics / plot.
- Swap in your team's actual `TrajectoryProblem` generator from phase 01
  once it exists (it just needs to produce the dataclass in
  `interface.py`, or something you convert to it).
- Share `interface.py` with whoever's building Methods B-D so all four
  methods are interchangeable `TrajectorySolver`s for the phase-03
  comparative study.
- For the 99.9th-percentile tail-time study specifically: run a batch
  of `TrajectoryProblem`s (nominal + OOD) through `SCvxSolver`, collect
  `Solution.wall_time_s` and `Solution.success`, and you have the two
  numbers the "Evaluation" slide asks for. `Solution.history` gives you
  per-iteration diagnostics if you want to characterize *why* the tail
  cases are slow (more SCvx iterations? repeated subproblem rejections?
  lambda continuations?).
