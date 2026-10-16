#!/usr/bin/env python3
"""
cartpole_swingup.py - Known-solution test: cart-pole swing-up, solved by the
same SCP engine that plans the aircraft trajectories.

--------------------------------------------------------------------------
Problem (Kelly, "An Introduction to Trajectory Optimization: How to Do Your
Own Direct Collocation", SIAM Review 59(4), 2017, section 6 and Table 3)
--------------------------------------------------------------------------
The pendulum starts hanging straight down under the cart, at rest. In
T = 2 s the cart must swing it up to balance upright, ending d = 1 m along
the track, at rest. Kelly asks for the minimum-force trajectory:

    minimize   J = integral_0^T u(t)^2 dt                        (6.3)
    dynamics   Kelly's full nonlinear equations                  (6.1)-(6.2)
    start      q1 = 0, q2 = 0 (hanging), q1' = q2' = 0          (6.9)
    end        q1 = d, q2 = pi (upright), q1' = q2' = 0
    limits     |q1| <= 2 m, |u| <= 20 N                          (6.7)-(6.8)
    parameters m1 = 1.0 kg, m2 = 0.3 kg, l = 0.5 m, g = 9.81    (Table 3)
    first guess states moving linearly from start to end, u = 0  (6.16)

The pole swings through 180 degrees, so the upright linearization used in
cartpole_scp.py is wrong for almost the whole manoeuvre: this tests SCP's
repeated re-linearization of strongly nonlinear dynamics.

--------------------------------------------------------------------------
How it is solved
--------------------------------------------------------------------------
The SCP engine in solvers/scp_aircraft.py (multiple shooting, RK4,
first-order-hold force, virtual control, PTR trust region) with its current
settings, exactly as for the aircraft and for cartpole_scp.py. This file only
supplies the boundary conditions; the model, limits and cost come from
cartpole_scp.py. Nothing about the known answer is given to the solver.

Afterwards, independently of SCP:
  * the planned force is replayed open loop from the start on the nonlinear
    model with a fine RK4 integrator, to confirm it really swings the pole up;
  * from t = T an LQR controller (designed on the upright linearization from
    cartpole_scp.py) holds the pole upright. The optimizer only plans the
    move; upright is unstable, so staying there needs feedback.

--------------------------------------------------------------------------
The comparison with the paper (made only at the end)
--------------------------------------------------------------------------
The paper shows the optimal trajectory (its Figures 9-10, found with
Hermite-Simpson collocation on 25 segments, 71 FMINCON iterations, 5.91 s)
but does not print the optimal cost. This file therefore rebuilds the paper's
own method, its nonlinear program (6.10)-(6.15) with the guess (6.16),
and solves it with scipy (SLSQP). That reference is computed separately from
SCP, and only the final SCP result is compared with it. The two discretize
the force differently (SCP: straight lines between nodes; Hermite-Simpson:
quadratics), so they agree closely but not to the last digit; both approach
the same continuous optimum as the grid is refined.

--------------------------------------------------------------------------
Outputs (written to the project's localstore/ folder)
--------------------------------------------------------------------------
  --plot  cartpole_swingup_progress.png     selected SCP iterations, each one's
                                            force replayed on the real pendulum:
                                            early ones fail, later ones get closer,
                                            the last swings up and holds
          cartpole_swingup_convergence.png  cost, dynamics error and swing-up miss
                                            at every iteration
          cartpole_swingup_result.png       final trajectory next to the paper's
                                            method (the style of Kelly's Figs 9-10)
  --gif   cartpole_swingup.gif              animation of the same iterations

Usage (from the project root)
    python -m prototypes.cartpole_swingup                  # solve, verify, compare
    python -m prototypes.cartpole_swingup --plot --gif     # ... and make the figures
    python -m prototypes.cartpole_swingup --nodes 81       # finer discretization
    python -m prototypes.cartpole_swingup --reference      # the paper's method only, no SCP
"""
from __future__ import annotations

import argparse
import math
import time
import warnings

import numpy as np
from scipy.linalg import solve_continuous_are
from scipy.optimize import minimize

with warnings.catch_warnings():
    # scp_aircraft warns when the C172 parameter file is missing; the cart-pole doesn't use it
    warnings.filterwarnings("ignore", message="AIRFRAME_FILE")
    import solvers.scp_aircraft as scp
    from prototypes.cartpole_scp import (CartPoleSCP, NonlinearCartPole, linear_matrices,
                                         M_CART, M_POLE, L_POLE, GRAVITY, DISTANCE, DURATION,
                                         U_MAX, X_MAX)

# ---- settings ------------------------------------------------------------------------
N_NODES        = 41        # SCP nodes (40 intervals of 0.05 s)
SIM_DT         = 1e-3      # s, step of the independent open-loop replay (RK4)
HOLD_S         = 1.5       # s of LQR balancing shown after the swing-up
LQR_Q          = (10.0, 50.0, 1.0, 1.0)    # weights on cart position, pole angle, their rates
LQR_R          = 0.1       # weight on force
REF_SEGMENTS   = 25        # Hermite-Simpson segments for the paper's method (the paper uses 25)
PANELS         = 7         # iterations shown in the progression figure / animation
GIF_FPS        = 20

# ---- pass / fail ----------------------------------------------------------------------
TOL_REPLAY      = 1e-2     # open-loop replay must end this close to the goal (m, rad, m/s, rad/s)
UP_DEG          = 2.0      # an attempt counts as "swung up" if the pole ends within this of vertical
TOL_HOLD_DEG    = 1.0      # pole within this of upright at the end of the LQR hold
TOL_VS_PAPER    = 5e-3     # final cost within 0.5% of the paper's method (Hermite-Simpson)

X_START = np.zeros(4)                                   # hanging down, at rest
X_GOAL = np.array([DISTANCE, np.pi, 0.0, 0.0])          # 1 m along, upright, at rest


# --------------------------------------------------------------------------
# the swing-up problem for the SCP engine
# --------------------------------------------------------------------------
class SwingUpSCP(CartPoleSCP):
    """CartPoleSCP (nonlinear model, minimum integral of u^2, |u| <= 20 N,
    |q1| <= 2 m, T = 2 s, Kelly's straight-line guess) with the swing-up's
    boundary conditions. Records every iterate through the engine's
    on_iterate hook; recording does not affect the solve."""
    name = "cartpole_swingup"

    def __init__(self, n_nodes=N_NODES, verbose=False):
        super().__init__(model="nonlinear", n_nodes=n_nodes, verbose=verbose)
        self.x_start, self.x_goal = X_START.copy(), X_GOAL.copy()
        self.iterates = []

    def on_iterate(self, it, X, U, sigma, rec):
        self.iterates.append({"iter": it, "X": X.copy(), "U": U[:, 0].copy(),
                              "cost": self.cost(X, U, sigma), "defect": rec.get("max_defect", np.nan)})


# --------------------------------------------------------------------------
# independent replay: open-loop force, then LQR hold
# --------------------------------------------------------------------------
_MODEL = NonlinearCartPole()


def _f(x, u):
    """Kelly's (6.1)-(6.2) for one state, written out with scalars for speed
    (identical to NonlinearCartPole.f, which the solver uses)."""
    m1, m2, l, g = M_CART, M_POLE, L_POLE, GRAVITY
    _, q2, dq1, dq2 = x
    s, c = math.sin(q2), math.cos(q2)
    den = m1 + m2 * (1 - c * c)
    ddq1 = (l * m2 * s * dq2 * dq2 + u + m2 * g * c * s) / den
    ddq2 = -(l * m2 * c * s * dq2 * dq2 + u * c + (m1 + m2) * g * s) / (l * den)
    return np.array([dq1, dq2, ddq1, ddq2])


def lqr_gain():
    """LQR gain on the upright linearization, state (q1 - d, q2 - pi, q1', q2')."""
    A, B = linear_matrices()
    Q, R = np.diag(LQR_Q), np.array([[LQR_R]])
    P = solve_continuous_are(A, B, Q, R)
    return np.linalg.solve(R, B.T @ P)[0]


def replay(U, T=DURATION, hold=0.0, dt=SIM_DT, x0=X_START):
    """Apply the piecewise-linear force U (node values over [0, T]) open loop
    from x0 with RK4; then, for `hold` seconds, balance with LQR. Returns
    times, states and the force actually applied."""
    tn = np.linspace(0.0, T, len(U))
    K = lqr_gain()
    n_plan, n_hold = int(round(T / dt)), int(round(hold / dt))
    t = np.arange(n_plan + n_hold + 1) * dt
    X = np.zeros((len(t), 4)); F = np.zeros(len(t))
    X[0] = x0
    def u_at(tk, x):
        if tk <= T + 1e-12:
            return float(np.interp(tk, tn, U))
        e = x - X_GOAL
        return float(np.clip(-K @ e, -U_MAX, U_MAX))
    with np.errstate(all="ignore"):
        for k in range(len(t) - 1):
            x, tk = X[k], t[k]
            if tk < T - 1e-12:            # planned force: known at every time, so RK4 uses it exactly
                ua, um, ub = u_at(tk, x), u_at(tk + dt / 2, x), u_at(tk + dt, x)
                k1 = _f(x, ua); k2 = _f(x + dt / 2 * k1, um); k3 = _f(x + dt / 2 * k2, um)
                k4 = _f(x + dt * k3, ub)
            else:                         # feedback: force held over each 1 ms step (zero-order hold)
                ua = u_at(tk + 1e-9, x)
                k1 = _f(x, ua); k2 = _f(x + dt / 2 * k1, ua); k3 = _f(x + dt / 2 * k2, ua)
                k4 = _f(x + dt * k3, ua)
            F[k] = ua
            X[k + 1] = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        F[-1] = F[-2]
    return t, X, F


def attempt_summary(t, X, T=DURATION):
    """How close an open-loop replay gets to the goal at t = T."""
    k = int(np.argmin(np.abs(t - T)))
    ang = np.degrees(X[k, 1])
    miss = abs((ang - 180.0 + 180.0) % 360.0 - 180.0)     # distance from upright; +180 and -180 are both up
    return {"end_angle_deg": float(ang), "angle_miss_deg": float(miss), "cart_end_m": float(X[k, 0]),
            "state_err": float(np.abs(X[k] - X_GOAL).max())}


# --------------------------------------------------------------------------
# the paper's method, rebuilt independently: Hermite-Simpson collocation
# --------------------------------------------------------------------------
def paper_method(n_seg=REF_SEGMENTS, T=DURATION, d=DISTANCE):
    """Kelly's nonlinear program (6.10)-(6.15), separated Hermite-Simpson form,
    uniform grid, guess (6.16), solved with scipy's SLSQP. Returns the optimal
    cost and trajectory at the knot and midpoint times."""
    M, h = 2 * n_seg + 1, T / n_seg
    nX = 4 * M
    tq = np.linspace(0.0, T, M)
    goal = np.array([d, np.pi, 0.0, 0.0])
    z0 = np.r_[np.outer(tq / T, goal).ravel(), np.zeros(M)]           # (6.16)
    w = np.zeros(M)                                                    # Simpson weights (6.10)
    w[0:-1:2] += h / 6; w[2::2] += h / 6; w[1::2] += 4 * h / 6
    k0, km, k1 = np.arange(0, M - 1, 2), np.arange(1, M, 2), np.arange(2, M, 2)

    def split(z):
        return z[:nX].reshape(M, 4), z[nX:]

    def cons(z):
        X, U = split(z)
        F = _MODEL.f(X, U[:, None])
        interp = X[km] - 0.5 * (X[k0] + X[k1]) - h / 8 * (F[k0] - F[k1])     # (6.11)
        coll = X[k1] - X[k0] - h / 6 * (F[k0] + 4 * F[km] + F[k1])           # (6.12)
        return np.r_[interp.ravel(), coll.ravel(), X[0], X[-1] - goal]       # (6.15)

    def jac(z):
        X, U = split(z)
        A, B = _MODEL.jacobians(X, U[:, None])
        J = np.zeros((8 * n_seg + 8, nX + M)); I = np.eye(4)
        def add(r, k, cx, cf):
            J[r:r + 4, 4 * k:4 * k + 4] += cx * I + cf * A[k]
            J[r:r + 4, nX + k] += cf * B[k][:, 0]
        for s in range(n_seg):
            a, m, b = k0[s], km[s], k1[s]
            r = 4 * s
            add(r, m, 1, 0); add(r, a, -0.5, -h / 8); add(r, b, -0.5, h / 8)
            r = 4 * n_seg + 4 * s
            add(r, b, 1, 0); add(r, a, -1, -h / 6); add(r, m, 0, -4 * h / 6); add(r, b, 0, -h / 6)
        J[8 * n_seg:8 * n_seg + 4, 0:4] = I
        J[8 * n_seg + 4:, nX - 4:nX] = I
        return J

    bounds = ([(-X_MAX, X_MAX) if j % 4 == 0 else (None, None) for j in range(nX)]
              + [(-U_MAX, U_MAX)] * M)                                       # (6.13)-(6.14)
    t0 = time.perf_counter()
    r = minimize(lambda z: float(w @ z[nX:] ** 2), z0, jac=lambda z: np.r_[np.zeros(nX), 2 * w * z[nX:]],
                 constraints=[{"type": "eq", "fun": cons, "jac": jac}], bounds=bounds,
                 method="SLSQP", options={"maxiter": 2000, "ftol": 1e-12})
    X, U = split(r.x)
    return {"J": float(r.fun), "ok": bool(r.success), "iterations": int(r.nit),
            "time": time.perf_counter() - t0, "t": tq, "X": X, "U": U,
            "max_violation": float(np.abs(cons(r.x)).max()), "segments": n_seg}


def print_reference(ref):
    print(f"Paper's method (Kelly 2017, Hermite-Simpson collocation, {ref['segments']} segments, guess (6.16)), "
          f"solved independently with scipy SLSQP:")
    print(f"  J = {ref['J']:.4f} N^2 s   ({'converged' if ref['ok'] else 'NOT converged'}, "
          f"{ref['iterations']} iterations, {ref['time']:.1f} s, constraint error {ref['max_violation']:.1e})")
    print(f"  force range {ref['U'].min():+.2f} to {ref['U'].max():+.2f} N, "
          f"cart travels up to {ref['X'][:, 0].max():.3f} m")
    print("  (for scale, the paper reports 71 FMINCON iterations in 5.91 s for this problem)")


# --------------------------------------------------------------------------
# run + report
# --------------------------------------------------------------------------
def select_panels(iterates, n_panels):
    """Iterations to show before the final result (indices into iterates).

    SCP does not improve the replayed swing every single iteration (early on
    it fixes the dynamics everywhere at once, not the end point), so the
    picks are the iterations that set a new best, i.e. got the pole closer to
    upright at t = T than any earlier one, from the first guess up to the
    first that swings up. If there are more of those than panels, they are
    thinned evenly, always keeping the first guess and the first success."""
    best, record = np.inf, []
    for i, it in enumerate(iterates):
        if it["angle_miss_deg"] < best - 1e-9:
            best = it["angle_miss_deg"]
            record.append(i)
            if best <= UP_DEG:
                break
    n = max(n_panels - 1, 2)
    if len(record) <= n:
        return record
    keep = np.unique(np.round(np.linspace(0, len(record) - 1, n)).astype(int))
    return [record[k] for k in keep]


def run(args):
    print("Cart-pole swing-up (Kelly 2017, section 6): from hanging at rest to upright 1 m along, "
          "in 2 s, minimum integral of u^2")
    print(f"\nSolving with the SCP engine from scp_aircraft.py ({scp.ALGORITHM.upper()}, "
          f"W_TRUST={scp.W_TRUST}, MAX_ITERS={scp.MAX_ITERS}), nonlinear model, N={args.nodes}")
    solver = SwingUpSCP(n_nodes=args.nodes, verbose=args.verbose)
    t0 = time.perf_counter()
    res = solver.solve(None)
    wall = time.perf_counter() - t0
    U = res.u[:, 0]
    print(f"  status {res.status}, {res.iterations} iterations, {wall:.2f} s, "
          f"max scaled defect {res.max_defect:.1e}")
    print(f"  J_scp = {res.cost:.4f} N^2 s, force {U.min():+.2f} to {U.max():+.2f} N, "
          f"cart travels up to {res.x[:, 0].max():.3f} m")

    # how each recorded iterate's force performs on the real pendulum
    for it in solver.iterates:
        tt, XX, _ = replay(it["U"], hold=0.0)
        it.update(attempt_summary(tt, XX))
    print("\n  How the solution developed (each iteration's force replayed open loop on the real pendulum;")
    print("  listed: the first guess, every iteration that got the pole closer to upright than before, the last):")
    print("     iter      cost J   dynamics defect   pole at 2 s (from vertical)   cart at 2 s")
    shown = sorted(set(select_panels(solver.iterates, 10 ** 6)) | {len(solver.iterates) - 1})
    for i in shown:
        it = solver.iterates[i]
        print(f"    {it['iter']:5d}  {it['cost']:10.3f}   {it['defect']:15.1e}   "
              f"{it['end_angle_deg']:8.1f} deg ({it['angle_miss_deg']:5.1f} deg off)   {it['cart_end_m']:7.3f} m")
    up_at = next((it["iter"] for it in solver.iterates if it["angle_miss_deg"] <= UP_DEG), None)
    feas_at = next((it["iter"] for it in solver.iterates if it["defect"] <= scp.TOL_DEFECT), None)
    if up_at is not None and feas_at is not None:
        print(f"  first swings up at iteration {up_at}, dynamically feasible from iteration {feas_at}; "
              f"later iterations only lower the cost")

    # independent checks
    t, X, F = replay(U, hold=HOLD_S)
    plan_end = attempt_summary(t, X)
    hold_err = abs(np.degrees(X[-1, 1]) - 180.0)
    verdicts = []
    def judge(ok, text):
        verdicts.append(ok)
        print(f"  {'PASS' if ok else 'FAIL'}  {text}")
    print("\nChecks")
    judge(res.success, f"feasible solution returned (status {res.status})")
    judge(plan_end["state_err"] <= TOL_REPLAY,
          f"replaying the planned force open loop swings the pole up and stops 1 m along "
          f"(state error {plan_end['state_err']:.1e}, limit {TOL_REPLAY:.0e})")
    judge(np.abs(U).max() <= U_MAX + 1e-6 and np.abs(X[:, 0]).max() <= X_MAX + 1e-6,
          f"limits respected (|u| <= {U_MAX:g} N: max {np.abs(U).max():.2f}; "
          f"|q1| <= {X_MAX:g} m: max {np.abs(X[:, 0]).max():.3f})")
    judge(hold_err <= TOL_HOLD_DEG,
          f"LQR holds it upright for {HOLD_S:g} s afterwards (pole {hold_err:.3f} deg from vertical at the end)")

    ref = None
    if args.ref_segments > 0:
        print()
        ref = paper_method(args.ref_segments)
        print_reference(ref)
        gap = (res.cost - ref["J"]) / ref["J"]
        print("\nComparison with the paper's method")
        judge(ref["ok"] and abs(gap) <= TOL_VS_PAPER,
              f"SCP cost {res.cost:.4f} vs Hermite-Simpson {ref['J']:.4f} N^2 s: {100 * gap:+.3f}% "
              f"(limit {100 * TOL_VS_PAPER:.1f}%)")
        Ui = np.interp(ref["t"], np.linspace(0, DURATION, len(U)), U)
        print(f"  force profiles differ by at most {np.abs(Ui - ref['U']).max():.2f} N "
              f"(RMS {np.sqrt(np.mean((Ui - ref['U']) ** 2)):.2f} N) at the reference's {len(ref['t'])} points")
    print(f"\nOVERALL: {'PASS' if all(verdicts) else 'FAIL'}")

    if args.plot or args.gif:
        panels = select_panels(solver.iterates, args.panels)
        attempts = []
        for i in panels:
            it = solver.iterates[i]
            tt, XX, FF = replay(it["U"], hold=0.0)
            attempts.append({"label": f"iteration {it['iter']}" + (" (first guess)" if it["iter"] == 0 else ""),
                             "t": tt, "X": XX, "F": FF, "cost": it["cost"], **attempt_summary(tt, XX)})
        attempts.append({"label": f"final result ({res.iterations} iterations)", "t": t, "X": X, "F": F,
                         "cost": res.cost, "final": True, **plan_end})
        if args.plot:
            plot_progress(attempts)
            plot_convergence(solver.iterates, panels, res)
            plot_result(res, t, X, F, ref)
        if args.gif:
            make_gif(attempts)
        if args.plot and not args.no_show:
            import matplotlib.pyplot as plt
            plt.show()
    return res


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def _out(name):
    from environment.envgen import localstore_path
    return str(localstore_path(name))


def _draw_cartpole(ax, q1, q2, color, alpha=1.0, lw=2.0):
    import matplotlib.patches as mpatches
    cw, ch = 0.22, 0.10
    ax.add_patch(mpatches.Rectangle((q1 - cw / 2, -ch / 2), cw, ch, fc=color, ec="k", lw=0.6, alpha=alpha))
    tip = (q1 + L_POLE * np.sin(q2), -L_POLE * np.cos(q2))
    ax.plot([q1, tip[0]], [0, tip[1]], color=color, lw=lw, alpha=alpha, solid_capstyle="round")
    ax.plot(*tip, "o", color=color, ms=4, alpha=alpha)


def _filmstrip(ax, t, X, t_end, frames=12, xlim=None):
    """Kelly-style stop-action picture: the cart-pole at uniformly spaced times,
    dark (start) to light (end)."""
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("viridis")
    for j, tk in enumerate(np.linspace(0, t_end, frames)):
        k = int(np.argmin(np.abs(t - tk)))
        _draw_cartpole(ax, X[k, 0], X[k, 1], cmap(j / (frames - 1)), alpha=0.9, lw=1.6)
    lo, hi = xlim if xlim else (min(-0.4, X[:, 0].min() - 0.6), max(DISTANCE + 0.4, X[:, 0].max() + 0.6))
    ax.axhline(0, color="0.6", lw=0.8, zorder=0)
    ax.plot([DISTANCE], [L_POLE], marker="*", ms=11, color="crimson", zorder=5)   # the target tip position
    ax.set_xlim(lo, hi); ax.set_ylim(-0.65, 0.65); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])


def plot_progress(attempts):
    import matplotlib.pyplot as plt
    n = len(attempts)
    fig, axs = plt.subplots(n, 3, figsize=(15, 2.25 * n), gridspec_kw={"width_ratios": [2.2, 1.2, 1.2]})
    t_max = max(a["t"][-1] for a in attempts)
    xlim = (min(-0.6, min(a["X"][:, 0].min() for a in attempts) - 0.6),
            max(DISTANCE + 0.6, max(a["X"][:, 0].max() for a in attempts) + 0.6))
    for r, a in enumerate(attempts):
        ax0, ax1, ax2 = axs[r]
        final = a.get("final", False)
        _filmstrip(ax0, a["t"], a["X"], DURATION, xlim=xlim)
        if final:
            verdict = "swings up, then held upright by LQR"
        elif a["angle_miss_deg"] <= UP_DEG:
            verdict = f"first to swing up (pole {a['angle_miss_deg']:.1f} deg from vertical at 2 s)"
        else:
            verdict = f"misses: pole {a['angle_miss_deg']:.0f} deg short of vertical at 2 s"
        ax0.set_title(f"{a['label']}: {verdict}", fontsize=9, loc="left")
        ax1.plot(a["t"], np.degrees(a["X"][:, 1]), "k-", lw=1.3)
        ax1.axhline(180, color="crimson", ls="--", lw=0.9)
        ax1.set_ylabel("pole angle [deg]", fontsize=8)
        ax2.plot(a["t"], a["F"], "-", color="tab:blue", lw=1.3)
        ax2.axhline(U_MAX, color="0.5", ls=":", lw=0.8); ax2.axhline(-U_MAX, color="0.5", ls=":", lw=0.8)
        ax2.set_ylabel("force [N]", fontsize=8)
        ax2.text(0.02, 0.92, f"J = {a['cost']:.1f} N$^2$s", transform=ax2.transAxes, fontsize=8, va="top")
        for ax in (ax1, ax2):
            ax.set_xlim(0, t_max); ax.grid(alpha=0.3); ax.tick_params(labelsize=8)
            if final:
                ax.axvspan(DURATION, t_max, color="tab:green", alpha=0.08)
            if r == n - 1:
                ax.set_xlabel("t [s]", fontsize=8)
    axs[-1, 1].text(DURATION + 0.05, 30, "LQR hold", fontsize=8, color="tab:green")
    fig.suptitle("How SCP found the swing-up: each iteration's force replayed on the real (nonlinear) pendulum\n"
                 "frames dark to light from t = 0 to 2 s; red star = where the pole tip must end", fontsize=11)
    fig.tight_layout()
    path = _out("cartpole_swingup_progress.png")
    fig.savefig(path, dpi=120)
    print(f"Progression figure saved to {path}")


def plot_convergence(iterates, panels, res):
    import matplotlib.pyplot as plt
    it = np.array([d["iter"] for d in iterates])
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
    ax[0].plot(it, [d["cost"] for d in iterates], "k.-", lw=1)
    ax[0].set_ylabel("cost J of the iterate [N$^2$s]")
    ax[1].semilogy(it, np.maximum([d["defect"] for d in iterates], 1e-16), "k.-", lw=1)
    ax[1].axhline(scp.TOL_DEFECT, color="crimson", ls="--", lw=0.9, label="feasibility tolerance")
    ax[1].set_ylabel("max dynamics defect (scaled)"); ax[1].legend(fontsize=8)
    ax[2].semilogy(it, np.maximum([d["angle_miss_deg"] for d in iterates], 1e-6), "k.-", lw=1)
    ax[2].set_ylabel("open-loop replay: pole miss at 2 s [deg]")
    for a in ax:
        for p in panels:
            a.axvline(iterates[p]["iter"], color="tab:blue", lw=0.6, alpha=0.4)
        a.set_xlabel("SCP iteration"); a.grid(alpha=0.3)
    fig.suptitle(f"SCP convergence on the swing-up (blue lines = iterations shown in the progression figure); "
                 f"final J = {res.cost:.4f} N$^2$s", fontsize=11)
    fig.tight_layout()
    path = _out("cartpole_swingup_convergence.png")
    fig.savefig(path, dpi=120)
    print(f"Convergence figure saved to {path}")


def plot_result(res, t, X, F, ref):
    import matplotlib.pyplot as plt
    tn = np.linspace(0, DURATION, len(res.u))
    fig = plt.figure(figsize=(14, 7.5))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.25, 1])
    ax_f = fig.add_subplot(gs[:, 0])
    _filmstrip(ax_f, t, X, DURATION, frames=15)
    ax_f.set_title("SCP swing-up, stop-action (dark = start, light = end)", fontsize=10)
    axs = [fig.add_subplot(gs[i, 1]) for i in range(3)]
    keep = t <= DURATION + 1e-9
    for ax, k, lab, scale in ((axs[0], 0, "position [m]", 1.0), (axs[1], 1, "angle [rad]", 1.0)):
        ax.plot(t[keep], X[keep, k] * scale, "k-", lw=1.4, label="SCP (force replayed)")
        ax.plot(tn, res.x[:, k] * scale, "k.", ms=4, label="SCP nodes")
        if ref is not None:
            ax.plot(ref["t"], ref["X"][:, k] * scale, "-", color="tab:orange", lw=3, alpha=0.45,
                    label=f"paper's method (Hermite-Simpson, {ref['segments']} seg.)")
        ax.set_ylabel(lab)
    axs[2].plot(tn, res.u[:, 0], "k.-", ms=4, lw=1.2, label=f"SCP, J = {res.cost:.4f}")
    if ref is not None:
        axs[2].plot(ref["t"], ref["U"], "-", color="tab:orange", lw=3, alpha=0.45,
                    label=f"paper's method, J = {ref['J']:.4f}")
    axs[2].set_ylabel("force [N]"); axs[2].set_xlabel("time [s]")
    for ax in axs:
        ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="best"); ax.set_xlim(0, DURATION)
    fig.suptitle("Cart-pole swing-up: SCP result next to the paper's method (compare Kelly 2017, Figs. 9-10)",
                 fontsize=11)
    fig.tight_layout()
    path = _out("cartpole_swingup_result.png")
    fig.savefig(path, dpi=120)
    print(f"Result figure saved to {path}")


def make_gif(attempts, fps=GIF_FPS):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    frames = []                                     # (attempt index, time index)
    for a_i, a in enumerate(attempts):
        n = int(round(a["t"][-1] * fps))
        idx = [int(np.argmin(np.abs(a["t"] - j / fps))) for j in range(n + 1)]
        frames += [(a_i, k, False) for k in idx] + [(a_i, idx[-1], True)] * int(fps)   # pause at the end
    lo = min(-0.5, min(a["X"][:, 0].min() for a in attempts) - 0.6)
    hi = max(DISTANCE + 0.5, max(a["X"][:, 0].max() for a in attempts) + 0.6)
    fig, ax = plt.subplots(figsize=(7, 3.4))

    def draw(fr):
        a_i, k, done = fr
        a = attempts[a_i]
        ax.clear()
        ax.axhline(0, color="0.6", lw=0.8)
        ax.plot([DISTANCE], [L_POLE], marker="*", ms=12, color="crimson")
        tip = a["X"][: k + 1, 0] + L_POLE * np.sin(a["X"][: k + 1, 1]), -L_POLE * np.cos(a["X"][: k + 1, 1])
        ax.plot(*tip, "-", color="tab:blue", lw=0.8, alpha=0.5)
        final = a.get("final", False)
        _draw_cartpole(ax, a["X"][k, 0], a["X"][k, 1], "tab:green" if final else "tab:gray")
        tk = a["t"][k]
        phase = "LQR hold" if final and tk > DURATION else "planned force"
        ax.set_title(f"{a['label']}   t = {tk:4.2f} s   ({phase})", fontsize=9)
        if done:
            if final:
                msg, col = "swung up and held upright", "tab:green"
            elif a["angle_miss_deg"] <= UP_DEG:
                msg, col = f"swings up: {a['angle_miss_deg']:.1f} deg from vertical", "tab:green"
            else:
                msg, col = f"misses: {a['angle_miss_deg']:.0f} deg short of vertical", "crimson"
            ax.text(0.5, 0.06, msg, transform=ax.transAxes, ha="center", fontsize=10, color=col,
                    bbox={"fc": "white", "ec": col, "alpha": 0.9})
        ax.set_xlim(lo, hi); ax.set_ylim(-0.7, 0.7); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])

    anim = FuncAnimation(fig, draw, frames=frames, interval=1000 / fps)
    path = _out("cartpole_swingup.gif")
    anim.save(path, writer=PillowWriter(fps=fps), dpi=80)
    plt.close(fig)
    print(f"Animation saved to {path}")


def main():
    ap = argparse.ArgumentParser(description="Cart-pole swing-up (Kelly 2017) solved by the SCP engine.")
    ap.add_argument("--nodes", type=int, default=N_NODES, help="SCP nodes (default %(default)s)")
    ap.add_argument("--ref-segments", type=int, default=REF_SEGMENTS,
                    help="Hermite-Simpson segments for the paper's method (0 = skip; default %(default)s)")
    ap.add_argument("--reference", action="store_true", help="solve the paper's method only, no SCP")
    ap.add_argument("--plot", action="store_true", help="save the progression, convergence and result figures")
    ap.add_argument("--gif", action="store_true", help="save an animation of the progression")
    ap.add_argument("--panels", type=int, default=PANELS, help="iterations to show (default %(default)s)")
    ap.add_argument("--no-show", action="store_true", help="save the figures without opening windows")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    if args.reference:
        print_reference(paper_method(args.ref_segments if args.ref_segments > 0 else REF_SEGMENTS))
        return
    run(args)


if __name__ == "__main__":
    main()
