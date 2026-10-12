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
 
