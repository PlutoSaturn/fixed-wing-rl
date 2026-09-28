"""
discretization.py
===================
Two distinct discretizations are used, deliberately, and they are NOT
the same one -- this is standard practice for SCP and is what makes
the "defect" a meaningful diagnostic:

1. `discretize_zoh` -- discretizes the *linearized* dynamics (frozen
   Jacobian at one reference node, zeroth-order-hold on the control
   over the interval) EXACTLY, via the Van Loan block-matrix-exponential
   trick. This produces the affine equality constraint (A_k, B_k, F_k,
   r_k) used inside the convex subproblem (eq. 55b in Malyuta et al.,
   arXiv:2106.09125).

2. `flow_map` -- propagates the TRUE NONLINEAR dynamics forward with
   RK4 (control held constant over the interval, matching the ZOH
   assumption on u). This is used to compute the "defect" (eq. 56): how
   far a candidate solution actually deviates from true dynamic
   feasibility, which drives the trust-region accuracy ratio rho.

Design note (read this before extending): freezing the Jacobian at a
single node and exactly discretizing that frozen-LTI approximation
(#1) is simpler than the paper's Appendix approach, which integrates
the state-transition matrix *along the moving reference trajectory*
within each interval (a locally time-varying linearization). The
frozen-Jacobian approach is a standard, widely-used simplification --
and for THIS benchmark it's actually the right baseline to build,
because Method B (Birkhoff SCP) is specifically "same outer loop,
better numerics." Method A being the simple/coarse discretization is
what makes that A-vs-B comparison meaningful. See README for details.

One consequence: even a fully converged trajectory will show a small
non-zero defect (shrinking as N grows / dt shrinks) rather than being
bit-exact -- that's expected, not a bug.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.linalg import expm


def discretize_zoh(
    A_c: np.ndarray,
    B_c: np.ndarray,
    F_c: np.ndarray,
    r_c: np.ndarray,
    dt: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Exact zero-order-hold discretization of the affine LTI system

        xdot = A_c x + B_c u + F_c p + r_c    (u, p held constant over dt)

    via the Van Loan block-matrix-exponential trick: build an augmented
    matrix whose bottom rows are zero (encoding that u, p, and the
    constant offset all have zero derivative over the interval), then
    a single matrix exponential simultaneously produces the exact
    integrals for A_d, B_d, F_d, and r_d.

    Returns (A_d, B_d, F_d, r_d) such that
        x_{k+1} = A_d x_k + B_d u_k + F_d p + r_d
    is the exact discrete-time equivalent of the continuous affine
    system above.
    """
    n = A_c.shape[0]
    m = B_c.shape[1]
    npar = F_c.shape[1]

    sz = n + m + npar + 1
    M = np.zeros((sz, sz))
    M[:n, :n] = A_c
    M[:n, n : n + m] = B_c
    M[:n, n + m : n + m + npar] = F_c
    M[:n, n + m + npar : n + m + npar + 1] = r_c.reshape(-1, 1)
    M = M * dt

    Phi = expm(M)

    A_d = Phi[:n, :n]
    B_d = Phi[:n, n : n + m]
    F_d = Phi[:n, n + m : n + m + npar]
    r_d = Phi[:n, n + m + npar : n + m + npar + 1].reshape(-1)

    return A_d, B_d, F_d, r_d


def rk4_step(dynamics, x: np.ndarray, u: np.ndarray, p: np.ndarray, h: float) -> np.ndarray:
    """One classical RK4 step of the TRUE nonlinear dynamics."""
    k1 = dynamics.f(x, u, p)
    k2 = dynamics.f(x + 0.5 * h * k1, u, p)
    k3 = dynamics.f(x + 0.5 * h * k2, u, p)
    k4 = dynamics.f(x + h * k3, u, p)
    return x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def flow_map(
    dynamics,
    x: np.ndarray,
    u: np.ndarray,
    p: np.ndarray,
    dt: float,
    substeps: int = 5,
) -> np.ndarray:
    """Propagate the TRUE nonlinear dynamics forward by `dt`, holding u
    (and p) constant (zeroth-order hold), via `substeps` RK4 steps.
    This is psi(t_k, t_{k+1}, x_k, u, p) from eq. (56)/(59) in the
    paper -- used only to compute defects, never inside the convex
    subproblem itself.
    """
    h = dt / substeps
    xk = x.copy()
    for _ in range(substeps):
        xk = rk4_step(dynamics, xk, u, p, h)
    return xk
