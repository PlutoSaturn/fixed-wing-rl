# `scp_aircraft_birkhoff.py` — Birkhoff Pseudospectral SCP

Shared problem definition (state, dynamics, limits, boundary conditions, obstacles, initialization): see `scp_methodology.md`. This file only covers what is specific to the Birkhoff solver.

---

## 1. What it does

Computes a minimum-time, smooth, collision-free trajectory for the 3-DOF fixed-wing aircraft from start to goal in an `envgen` environment. The problem is nonconvex (dynamics, obstacles, stall), so it is solved with **Sequential Convex Programming (SCP)**:

1. Start from a guess trajectory.
2. Linearize the nonconvex parts around it to get a convex problem (an SOCP).
3. Solve it with CVXPY/Clarabel inside a trust region.
4. Update the guess and repeat until the trajectory stops changing and is feasible.

What differs from `scp_aircraft.py` is the **discretization**: one global polynomial on Legendre-Gauss-Lobatto (LGL) nodes in **Birkhoff** (integral) form, instead of RK4 multiple shooting. No numerical integration happens inside the SCP loop.

The result is always checked afterwards by an **independent RK4 re-simulation**.

---

## 2. Birkhoff vs. shooting

Both solvers share the aircraft model, limits, cost, boundary conditions, virtual control, trust region (PTR/SCvx), merit function, multi-start, final check, and result format. All settings are imported from `scp_aircraft.py`, so both always run on the same airframe. Only the transcription changes.

| | `scp_aircraft.py` (shooting) | `scp_aircraft_birkhoff.py` (Birkhoff) |
|---|---|---|
| Time grid | Uniform nodes, `t = σ·τ`, `τ ∈ [0,1]` | LGL nodes (clustered at ends), `t = σ(τ+1)/2`, `τ ∈ [−1,1]` |
| Controls between nodes | Piecewise linear | One global polynomial |
| Dynamics constraint | `x_{k+1} = RK4(x_k, u_k, u_{k+1}, σ)` | `V_k = (σ/2) f(X_k,U_k)` and `X_k = X_0 + Σ_j B_kj V_j` |
| Extra unknowns | none | node derivatives `V_k = dX/dτ` |
| Linearization | RK4 + sensitivity matrices per interval, every iteration | `f`, `∂f/∂x`, `∂f/∂u` evaluated once per node |
| Obstacle check points | Every RK4 substep | `OBS_SUBDIV` points per node interval (interpolated) |
| Defect used in merit | Shooting defect | Collocation defect |
| `max_defect` in result | In-loop shooting defect | **Final RK4 shooting defect** |
| Failure recovery | Next start | Add nodes (`REFINE`), then next start |

---

## 3. How the Birkhoff method works

**Nodes.** `N` LGL nodes on `[−1, 1]`, including both endpoints. They cluster near the ends, which avoids polynomial oscillation. Time maps as `t = σ(τ+1)/2`.

**Operators** (built once per `N` in `LGLOps`):
- `D` — differentiation matrix: `(D x)_k = x'(τ_k)`. Its norm grows like `N²`.
- `B` — Birkhoff integration matrix: `B_kj = ∫_{−1}^{τ_k} l_j(s) ds`, where `l_j` is the Lagrange basis polynomial. Computed exactly by Gauss quadrature. `B v = x − x₀` when `v = x'`, and the last row of `B` equals the quadrature weights `w`.
- `E_obs` — interpolation matrix from node values to the finer obstacle-check points. Interpolation is linear in the node values, so constraints on interpolated positions stay linear.

**Unknowns:** `X` (N×6), `U` (N×3), `V` (N×6, `dX/dτ`), and flight time `σ`.

**Equations.** By the chain rule `dX/dτ = (σ/2)·f(X,U)`, so:

```
(1)  V_k = (σ/2) f(X_k, U_k)        k = 0..N-1     nonlinear, linearized in SCP
(2)  X_k = X_0 + Σ_j B_kj V_j       k = 1..N-1     linear and exact
```

All nonlinearity is in (1) and is only evaluated at nodes. Equation (2) enters the subproblem exactly, with no linearization.

**Why `B` instead of `D`.** Enforcing dynamics with `D` uses a matrix whose norm grows like `N²`. Integration is smoothing and stays well behaved. Measured on a linear oscillator test system (not the full SCP problem):

| N | Differentiation form (cond.) | Birkhoff form (cond.) |
|---|---|---|
| 8 | 16 | 142 |
| 32 | 387 | 247 |
| 100 | 3,790 | 244 |

The Birkhoff form is flat in `N`; the differentiation form grows like `N²`. The crossover is around `N ≈ 25`, so the advantage only appears at larger node counts. `D` is still built, but only to initialize `V̄ = D @ X̄` and in the self-test.

---

## 4. SCP details specific to this file

**Linearized dynamics.** Around the reference `(X̄, Ū, V̄, σ̄)`, with `half = σ̄/2`:

```
V̄ + δV = half·f̄ + half·A δX + half·B δU + (δσ/2)·f̄   (+ virtual control ν)
```

The right-hand side is the reference's current collocation defect. `ν` is a penalized slack that keeps the subproblem feasible when the linearization is poor.

**Subproblem:**

```
minimize   cost(X,U,σ) + LAMBDA_DYN·‖w_k (vs/xs) ν‖₁ + LAMBDA_OBS·Σ slack / POS_SCALE
           [PTR: + W_TRUST·(‖dXn‖² + ‖dUn‖² + ds²)]
subject to limits, boundary, linearized dynamics, exact integral relation,
           linearized obstacles, trust region on (X, U, σ)
```

The trust region covers `X`, `U`, `σ` only; `V` is fixed by the dynamics. Variables are scaled (`xs`, `us`, `vs`) so the solver sees numbers of order 1.

**Obstacles.**
1. Interpolate reference positions to the fine grid: `P̄ = E_obs @ X̄[:, :3]`.
2. Compute normalized clearance `h̃` and unit normals there.
3. Keep only rows with `h̃ ≤ reach`, where `reach = LEBESGUE_FACTOR·√3·tr·POS_SCALE + PRUNE_BUFFER` (`tr` = current step bound). `LEBESGUE_FACTOR` allows for polynomial overshoot between nodes.
4. Linearize: `n · Δp + h̃ + slack ≥ 0`, with `slack ≥ 0` penalized.

Pruned obstacles still count in the merit, so a step into one is corrected next iteration.

**Merit function** (judges each step):

```
J = cost + LAMBDA_DYN·Σ w_k |V_k − (σ/2) f(X_k,U_k)| / xs + LAMBDA_OBS·Σ violation / POS_SCALE
```

Returns `inf` if the dynamics blow up (e.g. `V → 0`), so the step is rejected.

**Trust region.**
- **PTR** (`ALGORITHM = "ptr"`): hard cap plus quadratic step penalty. Every step is accepted. The cap halves on failure and regrows toward `TRUST_CAP`.
- **SCvx** (`"scvx"`): hard bound `η`, accepted or rejected by the ratio `ρ = actual/predicted decrease`; `η` shrinks or grows accordingly. `η < ETA_MIN` gives `trust_region_collapsed`.

---

## 5. One solve, step by step (`solve(env)`)

1. **Node count** (`nodes_for`): `t_est = ‖goal − start‖ / V_CRUISE`, `N = ceil(0.5π·t_est / MAX_INTERVAL_S) + 1`, clamped to `[N_NODES, MAX_NODES_BK]`. (Or `--nodes`.)
2. **Initial guesses:** straight line first, then polylines through a random via point (fixed seed). Nodes are placed at LGL time fractions at constant speed; heading and climb angle come from the path tangent; controls are trim controls; `σ = path length / V_CRUISE`.
3. **SCP loop** (`_solve_from`), each iteration: evaluate `f̄` and Jacobians at all nodes → build linearized matrices → build CVXPY problem → solve with Clarabel (shrink the trust region on failure) → recover `X, U, V, σ` → evaluate the true merit → accept/reject and update the trust region → check convergence.
4. **Independent verification:** RK4 from every node (`DENSE_STEPS` substeps) using the polynomial controls. Gives the **shooting defect** and the minimum **clearance margin**.
5. **Classify** (see §6). The collocation defect says the polynomial satisfies the dynamics at the nodes; the shooting check says flying those controls lands where the nodes say. A trajectory can pass the first and fail the second if nodes are too sparse.
6. **Refine and restart:** on `shooting_mismatch` or `dense_check_failed`, grow the node count (`max(N+4, 1.5·N)`), interpolate the solution onto the new nodes, and re-solve. Keep the best result so far and stop at the first success. A bad guess just moves to the next start.

---

## 6. Result statuses

| Status | Success? | Meaning |
|---|---|---|
| `feasible` | yes | Converged, collision-free, RK4 agrees |
| `feasible_unconverged` | yes | Feasible and verified, but SCP stopped early (may be suboptimal) |
| `shooting_mismatch` | no (refines) | RK4 disagrees with the nodes by more than `SHOOT_TOL` |
| `dense_check_failed` | no (refines) | RK4 path intrudes on the safety margin between nodes |
| `converged_infeasible` | no | Converged to a point violating defect/violation tolerances |
| `max_iterations` | no | Hit `MAX_ITERS` |
| `trust_region_collapsed` | no | SCvx trust region fell below `ETA_MIN` |
| `solver_error` | no | Subproblem repeatedly unsolvable |
| `nonfinite_step` | no | Steps repeatedly produced NaN/Inf dynamics |

---

## 7. Birkhoff-specific settings

Everything else (airframe, limits, weights, tolerances, trust-region parameters, restarts) is inherited from `scp_aircraft.py`.

| Setting | Default | Meaning |
|---|---|---|
| `MAX_NODES_BK` | 100 | Upper bound on LGL nodes, including refinement |
| `OBS_SUBDIV` | 4 | Obstacle check points per node interval |
| `LEBESGUE_FACTOR` | 2.0 | Extra pruning reach for interpolation overshoot |
| `SHOOT_TOL` | 0.05 | Max scaled RK4 shooting defect for success |
| `REFINE`, `REFINE_FACTOR` | True, 1.5 | Add nodes on quality failures; growth per refinement |
| `RESULTS_FILE_BK`, `FEASIBLE_FILE_BK` | `scp_birkhoff_results.json`, `feasible_envs_birkhoff.json` | Output files |

---

## 8. Usage

The file uses a relative import, so run it as part of its package: `python -m <package>.scp_aircraft_birkhoff ...`

```bash
python scp_aircraft_birkhoff.py --envs c172_envs.json            # solve all
python scp_aircraft_birkhoff.py --envs c172_envs.json --plot 0   # solve all, then plot env 0
python scp_aircraft_birkhoff.py --envs c172_envs.json --show 0   # plot a saved result
python scp_aircraft_birkhoff.py --selftest                       # check B, D, w
```

Other flags: `--out`, `--feasible-out`, `--max-envs`, `--nodes`, `--no-figs`, `--verbose`.

Console line per environment:

```
env  12  nodes  47  fill  5.0%  feasible  start 1  iters  18  4.2s  flight  64.3s  shoot 3.1e-03  margin +1.82
```

Outputs: a results JSON (one record per environment: `env_index`, `env_fingerprint`, `method: "birkhoff"`, all `SolveResult` fields, and `x`, `u` at the nodes), a feasible-environments JSON, and per-environment figures. The fingerprint lets `--show` refuse to pair a plan with a different environment.

---

## 9. Trade-offs

| | Shooting | Birkhoff |
|---|---|---|
| Dynamics accuracy | RK4 per interval | Spectral; few nodes for smooth solutions |
| Linearization | Sensitivity propagation every iteration | Pointwise Jacobians only |
| Constraint structure | Sparse, banded | `X = X₀ + BV` is **dense**; obstacle rows are dense too |
| Problem size | `X, U, σ, ν` | Adds `V`, roughly doubling state-sized variables |
| Sharp features (hard turns) | Handled well | Global polynomial can ring; needs more nodes |
| Verification | Merit already uses integration | Needs a separate RK4 check |
| Conditioning | Good | Flat in `N` (see §3); better above `N ≈ 25` |

Birkhoff tends to need fewer SCP iterations but does more work per iteration, so it is not automatically faster in wall-clock time.

---

## 10. Caveats

- **`max_defect` differs between files.** Birkhoff reports the final RK4 shooting defect; the merit uses the collocation defect. Shooting reports its in-loop defect.
- **Iterations vs. time are counted differently.** `iterations` is from the winning attempt only; `solve_time` covers every start and refinement. Compare total subproblem solves, not these two fields.
- **Polynomial overshoot** between nodes can push the path into obstacles even when node positions look safe. Hence the finer obstacle grid, `LEBESGUE_FACTOR`, and the final RK4 check.
- **Only the two quality failures trigger refinement.** Refinement keeps `σ` and re-derives `V̄` from `D`, so it is a warm start, not an exact continuation.
- **Node cap.** With `MAX_NODES_BK = 100`, very long flights may not meet `MAX_INTERVAL_S` and can fail the shooting check.
- **Rebuilt every iteration.** The CVXPY problem is rebuilt each iteration (no parameter reuse), same as the shooting solver.
- **Memory.** The obstacle matrix is built densely as `(rows × N × nx)` before sparsification, which can be large with many active rows and nodes.

---

## Glossary

| Term | Meaning |
|---|---|
| **LGL nodes** | The `N` Legendre-Gauss-Lobatto points on `[−1, 1]`, including both endpoints |
| **`D`, `B`** | Differentiation and Birkhoff integration matrices on the LGL nodes |
| **`V_k`** | Derivative of the state w.r.t. `τ` at node `k`. Not the airspeed `V` |
| **Collocation defect** | `|V_k − (σ/2) f(X_k,U_k)|`, scaled: how badly the polynomial violates the dynamics at a node |
| **Shooting defect** | `|X_{k+1} − RK4(X_k, …)|`, scaled: how badly the nodes disagree with real integration (the independent check) |
| **Virtual control `ν`** | Penalized slack on the linearized dynamics that prevents artificial infeasibility |
| **PTR / SCvx** | The two trust-region variants (§4) |
| **Merit function `J`** | Nonlinear measure used to judge a step: cost plus dynamics and obstacle penalties |
| **Lebesgue factor** | Allowance for interpolated positions overshooting the node values |
| **`POS_SCALE`** | Minimum turn radius at cruise, `V_cruise² / (g·tan(bank_max))`; the position unit |