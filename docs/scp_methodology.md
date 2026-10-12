Base Rules

Initialization

## 1. State and Control:

This solver uses a 3-DOF representation of the aircraft, with the state matrix:

$$\mathbf{x} = [p_x,\ p_y,\ p_z,\ V,\ \gamma,\ \chi]^\top$$

This state matrix defines a spherical velocity frame and a cartesian position frame,

with $p_x,\ p_y,\ p_z$ the positions of the aircraft in the inertial coordinate system in which the box is defined,

$\ V$ as the airspeed of the craft

$\ \gamma$ as the flight path angle,

and $\ \chi$ as the heading angle, measured CCW from east.

___

The control matrix is:

$$\qquad \mathbf{u} = [T,\ n_v,\ n_h]^\top$$

where $T$ is the thrust of the aircraft

$\ n_v$ the vertical acceleration,

and $\ n_h$ the horizontal acceleration.

## 2.  Equations of Motion

**Constants**

$$a_1 = \tfrac{1}{2}\rho S C_{D_0}, \qquad a_2 = \frac{2K(mg)^2}{\rho S}, \qquad c_{\text{stall}} = \frac{\rho S C_{L,\max}}{2mg}$$

**Drag**

$$D = a_1 V^2 + a_2\,\frac{n_v^2 + n_h^2}{V^2}$$

**Equations of motion**

$$\dot p_x = V\cos\gamma\cos\chi$$

$$\dot p_y = V\cos\gamma\sin\chi$$

$$\dot p_z = V\sin\gamma$$

$$\dot V = \frac{T - D}{m} - g\sin\gamma$$

$$\dot\gamma = \frac{g\(n_v - \cos\gamma)}{V}$$

$$\dot\chi = \frac{g n_h}{V\cos\gamma}$$

**Approximations:**
 
- **Thrust along the velocity:** the angle of attack is assumed small, so $\cos\alpha \approx 1$ and the small upward thrust component is ignored.
- **Drag polar fitted, not derived:** $C_{D_0}$ and $K$ come from a straight-line fit of JSBSim's trimmed $C_D$ against $C_L^2$ over 60–125 kt. The fitted $K$ implies an Oswald efficiency of about 1.5, which isn't physically possible. The fit absorbs trim drag and other effects the simple polar leaves out. It matches JSBSim's drag well in that speed range, but isn't a physical wing property.
- **Measured at 1 g only:** the polar is fitted in level flight and assumed to hold in turns and pull-ups up to $n = 1.5$.
## 2. Boundary Conditions
 
Start fully specified; end at the goal position:
 
$$
x(0) = \big(p_{\text{start}},\ V_{\text{cruise}},\ 0,\ \chi_0\big), \qquad \chi_0 = \text{heading from start toward goal}
$$
 
$$
p(\sigma) = p_{\text{goal}}
$$
 
The final speed, climb angle, and heading are free. Here $V_{\text{cruise}} = 48.9$ m/s (95 kt).
 
**Simplification:** the initial speed, level flight and heading toward the goal are a convention chosen by the solver, not part of the environment. The start velocity stored in the environment file is ignored. Every problem therefore starts from the same kind of state.
 
## 3. State Limits (Flight Envelope)
 
Stay inside the workspace:
 
$$
p_{\min} + r \le p \le p_{\max} - r
$$
 
Speed band, with $V_{\min} = 33.0$ m/s and $V_{\max} = 59.2$ m/s:
 
$$
V_{\min} \le V \le V_{\max}
$$
 
Climb-angle limit, with $\gamma_{\max} = 6^\circ$:
 
$$
|\gamma| \le \gamma_{\max}
$$
 
**Reasoning:**
 
- **Aircraft as a sphere:** clearance is computed as if the aircraft were a 6 m radius sphere. That covers the 11 m wingspan (5.5 m half-span), plus a separate 30 m safety margin around obstacles.
- **$V_{\min}$:** set at 1.3× stall speed, a standard safety margin, not a physical limit.
- **$V_{\max}$ (115 kt):** a planning choice, well below the C172's never-exceed speed.
- **$\gamma_{\max}$ (6°):** a planning choice. It keeps climbs within what the C172 can sustain for short periods, and keeps $\cos\gamma$ far from zero in the heading equation.
## 4. Control Limits
 
Thrust, with $T_0 = 2218$ N and $k_T = -2.86$ N·s/m:
 
$$
0 \le T \le 0.8\,(T_0 + k_T V)
$$
 
Total load factor, with $n_{\max} = 1.5$:
 
$$
\sqrt{n_v^2 + n_h^2} \le n_{\max}
$$
 
Bank angle, with $\mu_{\max} = 30^\circ$:
 
$$
|n_h| \le \tan(\mu_{\max})\, n_v
$$
 
Stall:
 
$$
\sqrt{n_v^2 + n_h^2} \le c_{\text{stall}}\, V^2, \qquad c_{\text{stall}} = \frac{\rho S C_{L,\max}}{2mg}
$$
 
$$
\sqrt{n_v^2 + n_h^2} \le c_{\text{stall}}\,\big(2\bar V V - \bar V^2\big)
$$
 
**Approximations and reasoning:**
 
- Thrust is a first order fit line, which needs to be changed to second order moving forward.
- Thrust is limited at 80% of max, to leave room for the control system to make maneuvers, this can be optimized for later.
- Control surface lag: Thrust can change instantly
- $C_{L,\max}$ 85% of the highest trimmed lift coefficient measured, a margin below the true stall. It assumes flaps up.
- Stall linearization is conservative - the tangent line lies below the parabola $V^2$, so the linearized limit is stricter than the true one. It can never allow a stall.
## 5. Control Rate Limits
 
Between consecutive nodes $k$ and $k+1$, separated by $\Delta t = \sigma/(N-1)$:
 
$$
|n_{v,k+1} - n_{v,k}| \le \dot n_{v,\max}\,\Delta t, \qquad \dot n_{v,\max} = 0.15\ \mathrm{s^{-1}}
$$
 
$$
|n_{h,k+1} - n_{h,k}| \le \dot n_{h,\max}\,\Delta t, \qquad \dot n_{h,\max} = 0.12\ \mathrm{s^{-1}}
$$
 
These keep the plan smooth enough for an autopilot to follow. Both are linear because $\sigma$ appears linearly.
 
**Approximation: $n_h$ rate as a roll-rate proxy.** A true roll-rate limit, $|\dot\mu| \le p_{\max}$, is nonconvex in the controls. Near level flight $n_h \approx \tan\mu$, so
 
$$
\dot n_h = \sec^2\mu\;\dot\mu
$$
 
With $|\mu| \le 30^\circ$, the $n_h$ limit corresponds to a roll rate of about 6.9°/s wings-level and 5.2°/s at 30° bank, within about 25% of a true roll-rate limit. That is far slower than a C172 can roll; it's chosen for trackability, not to model the airframe.
 
## 6. Flight Time
 
The final time is free and chosen by the optimizer, within loose bounds:
 
$$
1\ \mathrm{s} \le \sigma \le 3600\ \mathrm{s}
$$
 
Time is normalized to $\tau = t/\sigma \in [0, 1]$, so the dynamics become
 
$$
\frac{dx}{d\tau} = \sigma\, f(x, u)
$$
 
This is an exact change of variables, not an approximation. It lets the solver treat flight time as just another variable on a fixed grid.
 
**Note on all constraints:** limits are enforced at the discretization points (nodes, or check points for obstacles), not continuously. The final solution is re-checked on a dense grid, and with an independent RK4 simulation, to confirm nothing is violated between them.
 
## 7. Initialization Process

The solver initializes a guess by choosing a point in the middle third of the box at any point and defining two lines from the start to the midpoint to the end. The points along the path are a discretized version of the path generated by this process, evenly split by distance. The initial path guess has position defined at every node and heading and FPA approximated from a central difference calculation from the two nodes next to it. The control inputs remain zero for the entirely of the trajectory on the first guess. The solver uses multiple shooting to guess and is penalized for the violation of the dynamic constraints, which is tracked in the output variable **defect**.

## 8. Implementation of Obstacles
 
### Clearance
 
The aircraft is treated as a point, and every obstacle is inflated by a clearance distance
 
$$
d = r + \text{margin} = 6 + 30 = 36\ \text{m}
$$
 
where $r$ is the aircraft radius and the margin is extra safety distance. Keeping the point outside every inflated obstacle keeps the whole aircraft at least 30 m from every real surface.
 
### Obstacle functions
 
Each obstacle is a convex shape described by a function $h(p; d)$ of the aircraft position $p$:
 
$$
h(p; d) \ge 0 \quad\Longleftrightarrow\quad \text{a sphere of radius } d \text{ centred at } p \text{ does not touch the obstacle}
$$
 
Notation: $c$ is the obstacle centre, $R$ its rotation matrix (from roll, pitch, yaw), and $q = R^\top (p - c)$ the position in the obstacle's own frame.
 
**Ellipsoid / spheroid** (semi-axes $a_1, a_2, a_3$):
 
$$
h = \big\| diag(1/a)\, q \big\|_2 - \left(1 + \frac{d}{a_{\min}}\right)
$$
 
**Oriented cylinder** (radius $\rho$, half-length $L$, axis $a$):
 
$$
s = a^\top (p - c), \qquad r_\perp = \big\|(p - c) - s\,a\big\|_2, \qquad
h = \max\big(r_\perp - (\rho + d),\ |s| - (L + d)\big)
$$
 
**Upright cylinder / pillar** (radius $\rho$, top height $z_{\text{top}}$):
 
$$
h = \max\big(\|(x, y) - c_{xy}\|_2 - (\rho + d),\ z - (z_{\text{top}} + d)\big)
$$
 
**Box** (half-extents $e$):
 
$$
h = \max_k \big(|q_k| - (e_k + d)\big)
$$
 
**Wall with an opening:** split into up to four boxes around the hole (left, right, below, above). Each box is a separate constraint, and all must hold.
 
**Simplification: inflation is conservative, not exact.** A box is inflated by moving each face out by $d$, which leaves sharp corners where an exact inflation would be rounded. An ellipsoid is inflated by scaling it up by $(1 + d/a_{\min})$, which adds exactly $d$ along its shortest axis but more along its longer axes. Both inflated shapes contain the true inflated shape, so clearance is never less than $d$, but long, thin ellipsoids and box corners take up more space than they need to.
 
### Why the constraint must be linearized
 
Each $h$ is convex, but the requirement to stay *outside*, $h \ge 0$, describes the complement of a convex set, which is not convex. A convex subproblem cannot contain it directly.
 
A convex function always lies above its tangent planes. At a reference position $\bar p$ on the current trajectory, with gradient $g = \nabla h(\bar p)$:
 
$$
h(p) \;\ge\; h(\bar p) + g^\top (p - \bar p)
$$
 
So requiring the right-hand side to be non-negative is **sufficient** for $h(p) \ge 0$:
 
$$
h(\bar p) + g^\top (p - \bar p) \ge 0
$$
 
This is a single half-space: "stay on the far side of this plane." It is linear in $p$, so the convex subproblem can use it.
 
**Approximation: conservative.** The plane forbids everything on the obstacle's side of it, including some free space beside the obstacle, but never allows a collision. As SCP converges, the planes are rebuilt around the final path, so they fit closely where the path actually passes.
 
What the plane looks like for each shape:
 
- **Box:** the gradient is the outward normal of the face the point sticks out from the most, so the plane is that inflated face, extended to infinity.
- **Cylinder:** beside it, the plane is tangent to the side, facing the point radially; beyond an end, it is the inflated end cap.
- **Ellipsoid:** a plane perpendicular to the gradient of the scaled norm (close to, but not exactly, the closest-point direction).
### Normalization
 
Dividing by the gradient's length turns each constraint into an approximate distance in metres:
 
$$
\tilde h = \frac{h(\bar p)}{\|g\|}, \qquad n = \frac{g}{\|g\|}
$$
 
For boxes and cylinders $\|g\| = 1$ already. For ellipsoids this converts the dimensionless scaled norm into metres. All obstacles then carry comparable units, so one penalty weight works for all of them.
 
### Where constraints are applied: check points
 
Obstacles are checked at **check points** along the trajectory, not just at the nodes:
 
- **Shooting solver:** every RK4 substep of every interval, plus the final node. A substep's position change is linear in the decision-variable step through the sensitivity matrices, so collisions between nodes are caught too.
- **Birkhoff solver:** a point every 0.5 s of flight. Positions there are exactly linear in the variables through the integration matrix.
### Which obstacles get a constraint: pruning
 
At each iteration, $\tilde h$ is computed for every pair of check point and obstacle piece. Only nearby pairs become constraints:
 
$$
\tilde h \le \text{reach} = \sqrt{3}\,\Delta_{\max}\, r_{\text{turn}} + \tfrac14 r_{\text{turn}}
$$
 
where $\Delta_{\max}$ is the largest scaled step the trust region allows and $r_{\text{turn}} \approx 420$ m is the turn radius at cruise. This is about 470 m with the default settings. An obstacle farther away than this cannot be reached in one iteration, so leaving it out changes nothing in that subproblem.
 
The Birkhoff solver adds two further reductions:
 
- **Nearest three:** at each check point it keeps only the 3 closest pieces within reach.
- **Adaptive reach:** when the previous step was small, the reach shrinks to match, so fewer obstacles qualify.
**Simplification: pruning is local, not global.** If a step moves toward a previously distant obstacle, it is within reach on the next iteration and gets a plane then. The merit function (used to judge progress) and the final check always use every obstacle.
 
### The constraint in the subproblem
 
For each kept pair (check point $c$, obstacle piece $j$), with slack $s_c \ge 0$:
 
$$
n_{cj}^\top\, \Delta p_c + \tilde h_{cj} + s_c \;\ge\; 0
$$
 
$\Delta p_c$ is the change in check-point position this iteration. The cost adds a heavy penalty on the slacks:
 
$$
\lambda_{\text{obs}} \sum_c s_c, \qquad \lambda_{\text{obs}} = 1000
$$
 
The slack keeps the subproblem solvable while the trajectory still passes through obstacles (as early guesses usually do); the penalty pushes it out quickly. A valid solution has every $s_c = 0$.
 
Each iteration the planes are rebuilt around the new trajectory, so a point next to an obstacle sees a different plane as it moves around it. Together, the planes trace the obstacle's outline along the path.
 
### Final verification
 
After convergence, the trajectory is propagated densely, and the **true** $\tilde h$ (not the linearized version) is evaluated against every obstacle at every dense point. The plan is rejected (`dense_check_failed`) if
 
$$
\min \tilde h < -0.25 \times \text{margin} = -7.5\ \text{m}
$$
 
so a plan may dip at most 7.5 m into the 30 m safety margin between check points, and never into the 6 m aircraft radius.
 
### Consequences
 
- **Local information only:** each plane describes one obstacle near one point. The solver cannot see a gap on the other side of an obstacle, so it only refines the route its initial guess leads it into. This is why multiple initial guesses (Section 7) are needed.
- **Corner chatter:** at a box edge, the most-violated face can switch between iterations, flipping the plane by 90°. This slows convergence near boxes and walls.
- **Conservatism near corners:** a point beside a corner is held to one face's plane, so the path cannot hug the corner as closely as the true geometry allows. Solutions are slightly suboptimal there, but always safe.
 
