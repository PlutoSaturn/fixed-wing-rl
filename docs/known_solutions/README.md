# Known-Solution Tests of the SCP Solver

These write-ups document how the SCP solver (`solvers/scp_aircraft.py`) was checked against problems whose correct answers are known independently of the solver. Each problem has its own write-up; this page explains why the tests exist, how they fit together, and how the solver is shared between them.

| # | Problem | What it checks | Reference answer | Result |
|---|---|---|---|---|
| 1 | [Cart-pole balanced move](01_cartpole_balanced_move.md) | The SCP machinery itself, against an exact closed-form optimum | Exact linear optimal control, $J^* = 10.78111$ N²s | Matches the best possible on the solver's grid to $10^{-7}$; within 0.002 % of the continuous optimum |
| 2 | [Cart-pole swing-up](02_cartpole_swingup.md) | Strongly nonlinear dynamics, from a poor first guess | Kelly (2017), solved independently with the paper's own method: $J = 58.805$ N²s | 58.814 N²s, +0.015 % |
| 3 | [Stadium TFR detour](03_tfr_detour.md) | The full aircraft problem: C172 dynamics, flight envelope, a real-world keep-out zone | Shortest path around a circle (geometry), 26.610 km, a hard lower bound | 26.639 km, +0.11 %, on the correct side, clear of the TFR |

Commands to run all three: [Known_Solutions_Commands.md](../Known_Solutions_Commands.md). Details of the aircraft formulation: [scp_methodology.md](../scp_methodology.md).

## 1. Why known-solution tests

On the benchmark's random obstacle environments, nobody knows the optimal trajectory. The solver can show that its answer is **feasible** (it obeys the dynamics and misses the obstacles), but not that it is **optimal**, or even close. A solver with a subtle bug, such as a sign error in a Jacobian, a constraint that leaks, or a cost that is integrated wrongly, can still return feasible-looking plans.

A known-solution test closes that gap: the solver is given a problem whose answer has been worked out another way, and its result is compared with that answer. The comparison is made only at the end. The solver is never told the answer, and nothing in the solver is tuned to it.

The three problems were chosen to build on each other, each adding one difficulty while keeping an independent answer:

1. **Balanced move (linear cart-pole).** Linear dynamics and a quadratic cost make this a convex problem with an exact closed-form solution. If SCP cannot reproduce it to many digits, the core machinery (discretization, linearization, convex subproblem, convergence tests) is wrong. Nothing else can be blamed.
2. **Swing-up (nonlinear cart-pole).** The pendulum turns through 180°, so the dynamics are strongly nonlinear and the first guess is far from feasible. This exercises what SCP is actually for: repeatedly re-linearizing and recovering from a bad starting point. The reference comes from a published benchmark.
3. **TFR detour (aircraft).** The real problem class: the C172 model, its speed, bank, climb and stall limits, free flight time, obstacle avoidance, and multiple initial guesses. The reference is pure geometry, so it is exact and independent of any aircraft model.

The project manager specifically asked for an inverted-pendulum check, which tests 1 and 2 provide. Test 3 is the "real-world problem with a known solution": it uses the FAA's actual stadium flight restriction.

## 2. One solver, different problems

All three problems are solved by the **same code**: the class `SCPSolver` in `solvers/scp_aircraft.py`. It contains the method and nothing about aircraft. A problem is a subclass that supplies only:

| The engine needs | Aircraft (`AircraftSCP`) | Cart-pole (`CartPoleSCP`, `SwingUpSCP`) |
|---|---|---|
| Dynamics $f(x,u)$ and its Jacobians | 3-DOF C172, 6 states, 3 controls | Cart-pole, 4 states, 1 control |
| Boundary conditions and limits | start state, goal position, flight envelope, obstacles | start and end states, force and track limits |
| Cost | flight time, speed and control-smoothness terms | control effort $\int u^2\,dt$ |
| Initial guesses | 10 (straight line + 9 bent lines) | 1 (straight line, zero force) |

So the cart-pole results are evidence about the aircraft solver itself, not about a separate copy. The solver settings (trust region, penalties, tolerances, iteration limits) are the ones used for the aircraft; none were changed for these tests.

## 3. How the solver works (common to all three)

**Unknowns.** Time is split into $N$ nodes. The unknowns are the state $x_k$ and control $u_k$ at every node, plus the total time $\sigma$ (fixed for the cart-pole, free for the aircraft). With time normalized to $\tau = t/\sigma \in [0,1]$, the dynamics are $dx/d\tau = \sigma f(x,u)$. Controls are linear between nodes (first-order hold).

**Each iteration** starts from the current trajectory $(\bar x, \bar u, \bar\sigma)$ and:

1. **Integrates** the true nonlinear dynamics over every interval separately, starting from that interval's node (multiple shooting, RK4). With $s \in [0,1]$ the position within an interval, $\Delta\tau = 1/(N-1)$ the interval length and $u(s) = (1-s)u_k + s\,u_{k+1}$, it integrates the state together with its sensitivities:

   ```math
   x' = \Delta\tau\,\sigma f,\qquad \Phi' = \Delta\tau\,\sigma A\Phi,\qquad B_-' = \Delta\tau\,\sigma\big(AB_- + (1-s)B\big),\qquad B_+' = \Delta\tau\,\sigma\big(AB_+ + sB\big),\qquad S' = \Delta\tau\,\sigma A S + \Delta\tau f
   ```

   where $A = \partial f/\partial x$ and $B = \partial f/\partial u$ are evaluated along the current trajectory. The sensitivities say how the end of each interval moves when its start state ($\Phi$), its two node controls ($B_-$, $B_+$) or the flight time ($S$) change.

2. **Linearizes** the link between neighboring nodes:

   ```math
   x_{k+1} = \bar x_{k+1}^{\text{prop}} + \Phi_k\,\delta x_k + B_{-,k}\,\delta u_k + B_{+,k}\,\delta u_{k+1} + S_k\,\delta\sigma + \nu_k
   ```

   where $\nu_k$ is a **virtual control**: slack that keeps the subproblem solvable while the trajectory is still far from obeying the dynamics.

3. **Solves one convex problem** over all nodes at once (Clarabel):

   ```math
   \min\; J(x,u,\sigma) + \lambda_{\text{dyn}}\|\nu\|_1 + w_{\text{trust}}\|\delta z\|_2^2 \quad\text{s.t. linearized dynamics, limits, boundary conditions},\ |\delta z| \le \text{cap}
   ```

   The $\lambda_{\text{dyn}} = 1000$ penalty drives the virtual control to zero. The trust-region terms ($w_{\text{trust}} = 0.03$, step cap 0.5 in scaled units) keep each step small enough for the linearization to stay valid.

4. **Updates** the trajectory and repeats.

**Stopping.** A trajectory counts as feasible when the largest dynamics defect (mismatch between an integrated interval and the next node, scaled) is below $10^{-3}$ and no obstacle is violated. Once feasible, the solver stops when the step or the cost improvement becomes negligible, or when the cost has improved by less than 0.01 % over 10 iterations. It gives up on a start if an infeasible trajectory stops improving.

**What "converged" means.** Each convex subproblem is solved to its global optimum, but it is only an approximation around the current trajectory. The sequence therefore converges to a **local** optimum of the original problem. The known-solution tests check whether that local optimum is the true one in cases where the true one is known.

## 4. What the three tests establish together

- **Correct machinery.** Test 1 reproduces an exact optimum to seven significant digits, so the discretization, sensitivities, linearized constraints, cost integration and convergence logic are right.
- **Correct handling of nonlinearity.** Test 2 starts from a guess that violates the dynamics everywhere and still converges to the published optimum, showing that the trust region and virtual control do their jobs.
- **Correct aircraft problem.** Test 3 shows that, on the real model with a real obstacle, the solver finds the true shortest route to within 0.11 %, picks the correct side of the obstacle, and never cuts into the required clearance. The lower-bound check also guarantees that the obstacle constraint does not leak.

What they do **not** establish: global optimality on cluttered random environments, where many obstacle-avoiding routes exist and SCP finds the best one near its initial guesses. Test 3 shows the restarts at work (five of ten starts converged, and the best was kept), but in dense environments that remains evidence, not proof.
