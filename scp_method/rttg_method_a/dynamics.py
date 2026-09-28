"""
dynamics.py
============
Vehicle dynamics models for the RTTG benchmark, designed to be
swappable: implement `DynamicsModel` for any vehicle (fixed-wing,
quadrotor, rocket, ...) and `SCvxSolver` in `scvx_solver.py` works
unchanged. All the solver needs is `f(x, u, p)`, `state_dim`,
`control_dim`, and Jacobians (provided by default via finite
differences here; override `jacobians()` with analytic derivatives
later if profiling shows it's a bottleneck).

Included model
--------------
`FixedWingPointMass3DOF` -- the classic 3-DOF point-mass fixed-wing
model (flat-earth, coordinated turn), the standard nonlinear benchmark
dynamics used throughout the trajectory-optimization literature (e.g.
Betts, "Practical Methods for Optimal Control"). It is nonconvex in
exactly the ways SCP is meant to handle: nonlinear kinematics (trig
terms), a nonconvex aerodynamic force model (drag quadratic in lift
coefficient), and coupled state-control terms (bank angle appears
multiplicatively with lift, which itself depends on both airspeed and
the control CL).

State  x = [px, py, h, v, gamma, psi]
    px, py : horizontal position (m)
    h      : altitude, positive up (m)
    v      : airspeed (m/s)
    gamma  : flight path angle (rad)
    psi    : heading / course angle (rad)

Control u = [CL, phi, T]
    CL  : lift coefficient
    phi : bank angle (rad)
    T   : thrust (N)

Default parameters (`FixedWingParams`) are representative placeholder
values for a Skywalker-1900-class flying wing (a common ArduPilot
airframe) -- NOT manufacturer specs. Before treating solver output as
physically meaningful for your ArduPilot setup, replace `mass`,
`wing_area`, and the drag-polar coefficients (`CD0`, `k`) with values
fit to your actual airframe (e.g. from SITL parameters, a static
weigh-in, or a drag-polar estimate from flight-log airspeed/throttle
data).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


class DynamicsModel(ABC):
    """Abstract base for every vehicle model in the benchmark family."""

    state_dim: int
    control_dim: int
    state_names: List[str] = []
    control_names: List[str] = []

    @abstractmethod
    def f(self, x: np.ndarray, u: np.ndarray, p: np.ndarray) -> np.ndarray:
        """Continuous-time dynamics: xdot = f(x, u, p). `p` is a (possibly
        length-0) parameter vector, e.g. for a free-final-time scaling."""
        raise NotImplementedError

    def trim_guess(self) -> np.ndarray:
        """A physically sensible constant control to seed the cold-start
        initial guess with (see initial_guess.py). Default: all zeros --
        override for a better starting point (e.g. level-flight trim)."""
        return np.zeros(self.control_dim)

    def jacobians(
        self,
        x: np.ndarray,
        u: np.ndarray,
        p: np.ndarray,
        eps: float = 1e-6,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Central finite-difference Jacobians (A, B, F) = (df/dx, df/du,
        df/dp) at (x, u, p). Overridable with analytic Jacobians for
        speed/accuracy -- finite differences are simple and robust
        (no extra dependency like autodiff), which is the right
        trade-off for a baseline/reference method like Method A."""
        n, m = self.state_dim, self.control_dim
        npar = 0 if p is None else len(p)

        A = np.zeros((n, n))
        for i in range(n):
            dx = np.zeros(n)
            dx[i] = eps
            A[:, i] = (self.f(x + dx, u, p) - self.f(x - dx, u, p)) / (2 * eps)

        B = np.zeros((n, m))
        for i in range(m):
            du = np.zeros(m)
            du[i] = eps
            B[:, i] = (self.f(x, u + du, p) - self.f(x, u - du, p)) / (2 * eps)

        F = np.zeros((n, npar))
        for i in range(npar):
            dp = np.zeros(npar)
            dp[i] = eps
            F[:, i] = (self.f(x, u, p + dp) - self.f(x, u, p - dp)) / (2 * eps)

        return A, B, F


@dataclass
class FixedWingParams:
    """Representative Skywalker-1900-class fixed-wing parameters.
    THESE ARE APPROXIMATE PLACEHOLDERS. Replace with values measured
    or fit for your actual airframe/payload before trusting solver
    output as physically meaningful."""

    mass: float = 2.7               # kg, typical AUW for this class w/ payload
    wing_area: float = 0.60         # m^2
    g: float = 9.81                 # m/s^2
    rho: float = 1.225               # kg/m^3, sea-level constant-density
                                      # placeholder. If your OOD sweeps vary
                                      # altitude substantially, swap this for
                                      # an ISA lookup keyed on state[2] (h).

    CD0: float = 0.028               # zero-lift drag coefficient
    k: float = 0.045                 # induced-drag factor: CD = CD0 + k*CL^2
    CL_max: float = 1.2
    CL_min: float = -0.3

    T_max: float = 25.0              # N, static thrust ceiling
    T_min: float = 0.0

    bank_max: float = np.deg2rad(45.0)
    v_min: float = 12.0               # m/s, stall margin
    v_max: float = 28.0
    gamma_max: float = np.deg2rad(20.0)


class FixedWingPointMass3DOF(DynamicsModel):
    """3-DOF point-mass fixed-wing model, flat-earth, coordinated turn."""

    state_dim = 6
    control_dim = 3
    state_names = ["px", "py", "h", "v", "gamma", "psi"]
    control_names = ["CL", "phi", "T"]

    def __init__(self, params: Optional[FixedWingParams] = None):
        self.params = params or FixedWingParams()

    def f(self, x: np.ndarray, u: np.ndarray, p: np.ndarray) -> np.ndarray:
        px, py, h, v, gamma, psi = x
        CL, phi, T = u
        prm = self.params

        # Guard against divide-by-zero when the optimizer transiently
        # explores v <= 0 mid-iteration (outside the feasible region,
        # but linearization/line-search can still probe there).
        v_safe = max(v, 1e-3)

        q = 0.5 * prm.rho * v_safe ** 2 * prm.wing_area
        L = q * CL
        CD = prm.CD0 + prm.k * CL ** 2
        D = q * CD

        px_dot = v * np.cos(gamma) * np.cos(psi)
        py_dot = v * np.cos(gamma) * np.sin(psi)
        h_dot = v * np.sin(gamma)
        v_dot = (T - D) / prm.mass - prm.g * np.sin(gamma)
        gamma_dot = (L * np.cos(phi) - prm.mass * prm.g * np.cos(gamma)) / (
            prm.mass * v_safe
        )
        psi_dot = (L * np.sin(phi)) / (prm.mass * v_safe * np.cos(gamma))

        return np.array([px_dot, py_dot, h_dot, v_dot, gamma_dot, psi_dot])

    def trim_guess(self) -> np.ndarray:
        """Approximate level-flight trim at the mid-band airspeed: pick
        CL so that lift ~ weight, phi = 0 (wings level), T ~ drag."""
        prm = self.params
        v = 0.5 * (prm.v_min + prm.v_max)
        q = 0.5 * prm.rho * v ** 2 * prm.wing_area
        CL_trim = np.clip(prm.mass * prm.g / max(q, 1e-6), prm.CL_min, prm.CL_max)
        CD_trim = prm.CD0 + prm.k * CL_trim ** 2
        D_trim = q * CD_trim
        T_trim = np.clip(D_trim, prm.T_min, prm.T_max)
        return np.array([CL_trim, 0.0, T_trim])

    def default_path_bounds(self):
        """Convenience: box bounds matching FixedWingParams, in the
        format PathConstraints expects. Import PathConstraints lazily
        to avoid a circular import (interface.py imports this module)."""
        from .interface import PathConstraints

        prm = self.params
        x_min = np.array([-np.inf, -np.inf, 0.0, prm.v_min, -prm.gamma_max, -np.inf])
        x_max = np.array([np.inf, np.inf, np.inf, prm.v_max, prm.gamma_max, np.inf])
        u_min = np.array([prm.CL_min, -prm.bank_max, prm.T_min])
        u_max = np.array([prm.CL_max, prm.bank_max, prm.T_max])
        return PathConstraints(x_min=x_min, x_max=x_max, u_min=u_min, u_max=u_max)
