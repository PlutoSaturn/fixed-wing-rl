"""
initial_guess.py
==================
The "cold start" in Method A's name: no learned warm start, no
sophisticated shooting method -- just a straight-line interpolation
in state space between x0 and xf, with a constant trim-ish control
guess. SCvx is proven to converge essentially regardless of how bad
this initial guess is (Malyuta et al., Theorem 8 discussion), so this
deliberately-naive guess is the correct baseline: it's what makes
Method A "the baseline trust mechanism" per the method taxonomy.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .interface import TrajectoryProblem


def straight_line_guess(problem: TrajectoryProblem) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    N = problem.N
    x0, xf = problem.x0, problem.xf

    x_ref = np.linspace(x0, xf, N)

    # Calculate the straight-line heading to the target
    dy = xf[1] - x0[1]
    dx = xf[0] - x0[0]
    nominal_heading = np.arctan2(dy, dx)

    # Override the heading state (index 5) for intermediate nodes
    # Leaves the strict x0 and xf boundaries intact
    x_ref[1:-1, 5] = nominal_heading

    u_ref = np.tile(problem.dynamics.trim_guess(), (N, 1))
    npar = problem.param_dim()
    p_ref = np.ones(npar) if npar > 0 else np.zeros(0)

    return x_ref, u_ref, p_ref
