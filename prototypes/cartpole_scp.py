#!/usr/bin/env python3
"""
cartpole_scp.py - Verify the SCP solver on inverted-pendulum (cart-pole)
problems with known solutions.

The pendulum is solved by the SAME SCP engine that plans the aircraft
trajectories (scp_aircraft.SCPSolver: multiple shooting, first-order-hold
controls, virtual control, PTR/SCvx trust region, the same convergence tests
and the same settings, including any you tuned in scp_aircraft.py). This file
only supplies the pendulum's physics, limits, boundary conditions and cost,
exactly as AircraftSCP supplies the aircraft's.

The baseline it is compared with is computed separately, without SCP.

--------------------------------------------------------------------------
Problem (Kelly, "An Introduction to Trajectory Optimization", SIAM Review
2017, cart-pole model and parameters; Dymos reproduces the same setup)
--------------------------------------------------------------------------
  cart mass m1 = 1.0 kg, pole mass m2 = 0.3 kg, pole length l = 0.5 m, g = 9.81
  state  x = (q1, q2, q1', q2'): cart position [m], pole angle [rad]
         (q2 = 0 hanging down, q2 = pi upright), and their rates
  control u: horizontal force on the cart [N], |u| <= 20 N, |q1| <= 2 m
  minimize  J = integral of u^2 dt  over a fixed T = 2 s

Task "move" (default): start balanced upright at q1 = 0, at rest; end balanced
upright at q1 = 1 m, at rest.

Model "linear" (default; step 1): the equations linearized about upright,
with theta = q2 - pi:
      q1''    = (u + m2 g theta) / m1
      theta'' = (u + (m1 + m2) g theta) / (l m1)
  For linear dynamics the minimum-effort transfer has an exact solution
  (linear optimal control, via the controllability Gramian W):
      u*(t) = B' exp(A'(T - t)) W^-1 (x_T - exp(A T) x_0),    J* = 10.78111 N^2 s
  and the optimal force has the closed form u*(t) = a (t - T/2) + b sinh(w (t - T/2)),
  w = sqrt((m1 + m2) g / (l m1)).
  Model "nonlinear" (step 2): Kelly's full equations (6.1)-(6.2). No closed
  form; the pole leans only ~10 deg, so the answer is close to the linear one.

Two baselines are printed:
  * J*   - the exact continuous-time optimum (what the physics allows);
  * J_N  - the exact optimum among controls your solver can represent
           (piecewise linear between N nodes). Your SCP should match J_N to
           its convergence tolerance; J_N - J* is the discretization error,
           which shrinks as the node count grows.

Usage (from the project folder)
    python cartpole_scp.py                    # step 1: linear model vs exact solution
    python cartpole_scp.py --plot             # ... and plot it (saves cartpole_check.png)
    python cartpole_scp.py --nodes 81         # finer discretization
    python cartpole_scp.py --model nonlinear  # step 2: full nonlinear equations
    python cartpole_scp.py --reference        # print the exact solution only, no solve
"""
from __future__ import annotations

import argparse
import time

import cvxpy as cp
import numpy as np
from scipy.linalg import expm

import solvers.scp_aircraft as scp

# ---- problem settings (Kelly 2017 / Dymos) -------------------------------------
M_CART, M_POLE, L_POLE, GRAVITY = 1.0, 0.3, 0.5, 9.81
DISTANCE   = 1.0          # m, cart travel
DURATION   = 2.0          # s, fixed final time
U_MAX      = 20.0         # N
X_MAX      = 2.0          # m, cart position limit
N_NODES    = 41           # nodes (40 intervals of 0.05 s)

# ---- pass / fail ------------------------------------------------------------------
TOL_VS_DISCRETE  = 1e-4   # |J_scp - J_N| / J_N : solver accuracy (same discretization)
TOL_VS_EXACT     = 2e-3   # |J_scp - J*|  / J*  : includes discretization error
TOL_FINAL_STATE  = 1e-4   # final state error (m, rad, m/s, rad/s)


# --------------------------------------------------------------------------
# models (vectorized over leading dimensions, like FixedWing3DOF)
# --------------------------------------------------------------------------
def linear_matrices(m1=M_CART, m2=M_POLE, l=L_POLE, g=GRAVITY):
    """A, B of the cart-pole linearized about upright, state (q1, theta, q1', theta')."""
    A = np.array([[0, 0, 1, 0], [0, 0, 0, 1],
                  [0, m2 * g / m1, 0, 0], [0, (m1 + m2) * g / (l * m1), 0, 0]], float)
    B = np.array([[0], [0], [1 / m1], [1 / (l * m1)]], float)
    return A, B


class LinearCartPole:
    """Cart-pole linearized about upright. State uses Kelly's angle q2 (pi = upright)."""
    nx, nu = 4, 1

    def __init__(self):
        self.A, self.B = linear_matrices()

    def f(self, x, u):
        dev = x - np.array([0.0, np.pi, 0.0, 0.0])            # theta = q2 - pi
        return dev @ self.A.T + u @ self.B.T

    def jacobians(self, x, u):
        lead = x.shape[:-1]
        return (np.broadcast_to(self.A, lead + (4, 4)).copy(),
                np.broadcast_to(self.B, lead + (4, 1)).copy())


class NonlinearCartPole:
    """Kelly (2017) equations (6.1)-(6.2). Jacobians by complex-step
    differentiation, which is exact to machine precision."""
    nx, nu = 4, 1

    def f(self, x, u):
        m1, m2, l, g = M_CART, M_POLE, L_POLE, GRAVITY
        q2, dq1, dq2, F = x[..., 1], x[..., 2], x[..., 3], u[..., 0]
        s, c = np.sin(q2), np.cos(q2)
        den = m1 + m2 * (1 - c**2)
        ddq1 = (l * m2 * s * dq2**2 + F + m2 * g * c * s) / den
        ddq2 = -(l * m2 * c * s * dq2**2 + F * c + (m1 + m2) * g * s) / (l * den)
        return np.stack([dq1, dq2, ddq1, ddq2], axis=-1)

    def jacobians(self, x, u, h=1e-30):
        lead = x.shape[:-1]
        A = np.empty(lead + (4, 4)); B = np.empty(lead + (4, 1))
        xc, uc = x.astype(complex), u.astype(complex)
        for j in range(4):
            xp = xc.copy(); xp[..., j] += 1j * h
            A[..., :, j] = self.f(xp, uc).imag / h
        up = uc.copy(); up[..., 0] += 1j * h
        B[..., :, 0] = self.f(xc, up).imag / h
        return A, B


# --------------------------------------------------------------------------
# the cart-pole problem for the SCP engine
# --------------------------------------------------------------------------
class CartPoleSCP(scp.SCPSolver):
    name = "cartpole_scp"
    uses_obstacles = False

    def __init__(self, model="linear", n_nodes=N_NODES, verbose=False):
        mdl = LinearCartPole() if model == "linear" else NonlinearCartPole()
        super().__init__(mdl, xs=[1.0, 1.0, 1.0, 1.0], us=[U_MAX], verbose=verbose, n_starts=1)
        self.N, self.T = n_nodes, DURATION
        self.x_start = np.array([0.0, np.pi, 0.0, 0.0])
        self.x_goal = np.array([DISTANCE, np.pi, 0.0, 0.0])

    def nodes_for(self, env):
        return self.N

    def initial_guesses(self, env):
        """Kelly's naive guess: states interpolated linearly between the
        boundary states, zero force."""
        s = np.linspace(0.0, 1.0, self.n_nodes)[:, None]
        X = (1 - s) * self.x_start + s * self.x_goal
        yield X, np.zeros((self.n_nodes, 1)), self.T

    def subproblem_constraints(self, env, X, U, sig, X_bar, U_bar, s_bar, N):
        return [X[0] == self.x_start, X[-1] == self.x_goal,
                cp.abs(U[:, 0]) <= U_MAX, cp.abs(X[:, 0]) <= X_MAX,
                sig == self.T]                                   # fixed final time

    def cost(self, X, U, sigma):
        """Exact integral of u^2 for piecewise-linear u between nodes."""
        u, h = U[:, 0], sigma / (len(U) - 1)
        return float(h / 3 * np.sum(u[:-1]**2 + u[:-1] * u[1:] + u[1:]**2))

    def cost_expr(self, X, U, sig, N):
        # same integral, written as a sum of squares: a^2 + ab + b^2 = ((a+b)^2 + a^2 + b^2) / 2
        u, h = U[:, 0], self.T / (N - 1)
        return h / 6 * (cp.sum_squares(u[:-1] + u[1:]) + cp.sum_squares(u[:-1]) + cp.sum_squares(u[1:]))


# --------------------------------------------------------------------------
# exact baselines (independent of SCP)
# --------------------------------------------------------------------------
def exact_continuous(T=DURATION, d=DISTANCE):
    """Exact minimum-effort transfer for the linearized model. Returns J*, the
    optimal force u*(t) and the optimal states x*(t) as functions of time."""
    A, B = linear_matrices()
    n = 4
    M = np.zeros((2 * n, 2 * n)); M[:n, :n] = -A; M[:n, n:] = B @ B.T; M[n:, n:] = A.T
    E = expm(M * T)                                   # Van Loan: Gramian from one exponential
    eAT = E[n:, n:].T
    W = eAT @ E[:n, n:]
    delta = np.array([d, 0, 0, 0]) - eAT @ np.zeros(n)
    lam = np.linalg.solve(W, delta)
    u = lambda t: (B.T @ expm(A.T * (T - np.atleast_1d(t))[:, None, None]) @ lam)[..., 0]
    # states: with costate p(t) = exp(A'(T - t)) lam and u = B'p, the pair (x, p) obeys
    #   x' = A x + B B' p,  p' = -A' p,   so (x, p)(t) = exp(H t) (x0, p(0)) exactly
    H = np.zeros((2 * n, 2 * n)); H[:n, :n] = A; H[:n, n:] = B @ B.T; H[n:, n:] = -A.T
    xp0 = np.r_[np.zeros(n), expm(A.T * T) @ lam]
    def x(t):
        th = expm(H * np.atleast_1d(t)[:, None, None]) @ xp0
        X = th[:, :n].copy()
        X[:, 1] += np.pi                              # back to Kelly's angle (pi = upright)
        return X
    w = np.sqrt((M_CART + M_POLE) * GRAVITY / (L_POLE * M_CART))
    return {"J": float(delta @ lam), "u": u, "x": x, "omega": w}


def exact_discrete(N=N_NODES, T=DURATION, d=DISTANCE):
    """Exact optimum over piecewise-linear forces with N nodes (what the SCP can
    represent), for the linearized model: exact discretization + KKT solve."""
    A, B = linear_matrices()
    h = T / (N - 1)
    Mx = np.zeros((6, 6)); Mx[:4, :4] = A; Mx[:4, 4:5] = B; Mx[4, 5] = 1.0
    E = expm(Mx * h)
    Ad, G0, G1 = E[:4, :4], E[:4, 4], E[:4, 5] / h      # x+ = Ad x + G0 u_k + G1 (u_k+1 - u_k)
    Gm, Gp = G0 - G1, G1
    # final state is linear in the node forces: x_N = sum_k Ad^(N-2-k) (Gm u_k + Gp u_k+1)
    C = np.zeros((4, N)); P = np.eye(4)
    for k in range(N - 2, -1, -1):
        C[:, k] += P @ Gm; C[:, k + 1] += P @ Gp
        P = P @ Ad
    H = np.zeros((N, N))                                 # cost u'Hu = h/3 sum(u_k^2 + u_k u_k+1 + u_k+1^2)
    for k in range(N - 1):
        H[k, k] += h / 3; H[k + 1, k + 1] += h / 3; H[k, k + 1] += h / 6; H[k + 1, k] += h / 6
    r = np.array([d, 0, 0, 0])
    Hi_Ct = np.linalg.solve(H, C.T)
    u = Hi_Ct @ np.linalg.solve(C @ Hi_Ct, r)
    return {"J": float(u @ H @ u), "u": u, "t": np.linspace(0, T, N)}


# --------------------------------------------------------------------------
# run + report
# --------------------------------------------------------------------------
def print_reference(N):
    ex = exact_continuous()
    dc = exact_discrete(N)
    t = np.linspace(0, DURATION, 9)
    print("Exact solution, linearized cart-pole, move 1 m in 2 s balanced upright (minimum integral of u^2)")
    print(f"  J*  (continuous, exact)                 = {ex['J']:.6f} N^2 s")
    print(f"  J_N (best piecewise-linear force, N={N}) = {dc['J']:.6f} N^2 s   "
          f"(discretization adds {100 * (dc['J'] - ex['J']) / ex['J']:.4f}%)")
    print("  optimal force u*(t) = a (t - 1) + b sinh(w (t - 1)),  w = %.6f 1/s" % ex["omega"])
    print("  " + "  ".join(f"t={x:.2f}:{v:+.3f}N" for x, v in zip(t, ex["u"](t))))
    return ex, dc


def run(args):
    ex, dc = print_reference(args.nodes)
    print(f"\nSolving with the SCP engine from scp_aircraft.py ({scp.ALGORITHM.upper()}, "
          f"W_TRUST={scp.W_TRUST}, MAX_ITERS={scp.MAX_ITERS}), model: {args.model}, N={args.nodes}")
    solver = CartPoleSCP(model=args.model, n_nodes=args.nodes, verbose=args.verbose)
    t0 = time.perf_counter()
    res = solver.solve(None)
    wall = time.perf_counter() - t0
    X, U = res.x, res.u[:, 0]
    t = np.linspace(0, DURATION, len(U))
    final_err = np.abs(solver.propagate(X, res.u, res.flight_time, scp.DENSE_STEPS, sens=False)["x_end"][-1]
                       - solver.x_goal).max()
    print(f"  status {res.status}, {res.iterations} iterations, {wall:.2f} s, max scaled defect {res.max_defect:.1e}")
    print(f"  J_scp = {res.cost:.6f} N^2 s")

    verdicts = []
    def judge(ok, text):
        verdicts.append(ok)
        print(f"  {'PASS' if ok else 'FAIL'}  {text}")
    judge(res.success, f"feasible solution returned (status {res.status})")
    judge(final_err <= TOL_FINAL_STATE, f"reaches the goal state (error {final_err:.1e}, limit {TOL_FINAL_STATE:.0e})")
    if args.model == "linear":
        gd = (res.cost - dc["J"]) / dc["J"]
        ge = (res.cost - ex["J"]) / ex["J"]
        judge(abs(gd) <= TOL_VS_DISCRETE, f"matches the best possible with N={args.nodes} nodes: "
              f"{100 * gd:+.5f}% (limit {100 * TOL_VS_DISCRETE:.3f}%)")
        judge(abs(ge) <= TOL_VS_EXACT, f"matches the exact continuous optimum: {100 * ge:+.4f}% "
              f"(limit {100 * TOL_VS_EXACT:.2f}%)")
        du = np.abs(U - dc["u"]).max()
        Xe = ex["x"](t)
        print(f"  force profile: max |u_scp - u_best| = {du:.2e} N over the {len(U)} nodes; "
              f"max |u_scp - u*(t)| = {np.abs(U - ex['u'](t)).max():.3f} N")
        print(f"  states vs exact x*(t): cart position within {1000 * np.abs(X[:, 0] - Xe[:, 0]).max():.3f} mm, "
              f"pole angle within {np.degrees(np.abs(X[:, 1] - Xe[:, 1]).max()):.4f} deg")
    else:
        ge = (res.cost - ex["J"]) / ex["J"]
        print(f"  nonlinear model: no exact solution; vs the linearized optimum J* = {ex['J']:.4f}: "
              f"{100 * ge:+.2f}% (expected to be close, not equal)")
    print(f"\nOVERALL: {'PASS' if all(verdicts) else 'FAIL'}")
    if args.plot:
        plot(t, X, U, ex, dc, res, args)
    return res


def plot(t, X, U, ex, dc, res, args):
    import matplotlib.pyplot as plt
    tt = np.linspace(0, DURATION, 400)
    Xe = ex["x"](tt)
    ref = "exact" if args.model == "linear" else "linear-model exact (reference)"
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].plot(tt, ex["u"](tt), "g-", lw=3, alpha=0.5, label=f"{ref}, J* = {ex['J']:.4f}")
    ax[0].plot(t, U, "k.-", ms=5, lw=1, label=f"SCP ({args.model}), J = {res.cost:.4f}")
    ax[0].set_ylabel("force u [N]")
    ax[1].plot(tt, Xe[:, 0], "g-", lw=3, alpha=0.5, label=ref)
    ax[1].plot(t, X[:, 0], "k.-", ms=4, lw=1, label=f"SCP ({args.model})")
    ax[1].set_ylabel("cart position q1 [m]")
    ax[2].plot(tt, np.degrees(Xe[:, 1] - np.pi), "g-", lw=3, alpha=0.5, label=ref)
    ax[2].plot(t, np.degrees(X[:, 1] - np.pi), "k.-", ms=4, lw=1, label=f"SCP ({args.model})")
    ax[2].set_ylabel("pole lean from upright [deg]")
    for a in ax:
        a.set_xlabel("t [s]"); a.grid(alpha=0.3); a.legend(fontsize=8)
    fig.suptitle("Cart-pole: move 1 m in 2 s balanced upright. SCP vs. exact optimum")
    fig.tight_layout()
    fig.savefig(args.save, dpi=130)
    print(f"Plot saved to {args.save}")
    plt.show()


def main():
    ap = argparse.ArgumentParser(description="Check the SCP solver against the exact cart-pole solution.")
    ap.add_argument("--model", choices=["linear", "nonlinear"], default="linear")
    ap.add_argument("--nodes", type=int, default=N_NODES)
    ap.add_argument("--reference", action="store_true", help="print the exact solution only")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--save", default="cartpole_check.png")
    args = ap.parse_args()
    if args.reference:
        print_reference(args.nodes)
        return
    run(args)


if __name__ == "__main__":
    main()
