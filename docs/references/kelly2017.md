# Reference: Kelly (2017), An Introduction to Trajectory Optimization

**Citation.** Matthew Kelly, "An Introduction to Trajectory Optimization: How to Do Your Own Direct Collocation," *SIAM Review*, vol. 59, no. 4, pp. 849–904, 2017. DOI: [10.1137/16M1062569](https://epubs.siam.org/doi/10.1137/16M1062569)

```bibtex
@article{kelly2017trajopt,
  author  = {Kelly, Matthew},
  title   = {An Introduction to Trajectory Optimization: How to Do Your Own Direct Collocation},
  journal = {SIAM Review},
  volume  = {59},
  number  = {4},
  pages   = {849--904},
  year    = {2017},
  doi     = {10.1137/16M1062569}
}
```

**Access.** The paper is free to read at the DOI link above. The PDF itself is not stored in this repository: SIAM holds the copyright, and this repository is public. This page records what the project uses from the paper, so that our tests can be checked against it without leaving GitHub.

**Where it is used in this repository**

| File | What it takes from the paper |
|---|---|
| `prototypes/cartpole_scp.py` | Cart-pole model and parameters (Section 6.1, Table 3) |
| `prototypes/cartpole_swingup.py` | The swing-up problem (Section 6), and the paper's Hermite–Simpson method rebuilt as the independent reference (Sections 4 and 6.6) |
| [docs/known_solutions/01_cartpole_balanced_move.md](../known_solutions/01_cartpole_balanced_move.md) | Model and parameters |
| [docs/known_solutions/02_cartpole_swingup.md](../known_solutions/02_cartpole_swingup.md) | The full swing-up problem and its reference solution |

## What we use from the paper

### Section 6: Cart-pole swing-up (pp. 868–873)

The paper's worked example: a cart on a track with a pendulum hanging from it. The cart is driven so that the pendulum swings from hanging at rest to balanced upright at rest, with the minimum-force objective.

**State and control** (Section 6.1): cart position $q_1$, pole angle $q_2$ ($q_2 = 0$ hanging down, $q_2 = \pi$ upright), their rates, and the horizontal force $u$ on the cart:

```math
\mathbf{x} = [q_1,\ q_2,\ \dot q_1,\ \dot q_2]^\top
```

**Dynamics** (eqs. 6.1–6.2):

```math
\ddot q_1 = \frac{\ell m_2 \sin q_2\,\dot q_2^2 + u + m_2 g\cos q_2 \sin q_2}{m_1 + m_2(1-\cos^2 q_2)}, \qquad
\ddot q_2 = -\frac{\ell m_2 \cos q_2 \sin q_2\,\dot q_2^2 + u\cos q_2 + (m_1+m_2) g \sin q_2}{\ell m_1 + \ell m_2(1-\cos^2 q_2)}
```

**Objective** (eq. 6.3): minimize the integral of the force squared, $J = \int_0^T u^2\,dt$.

**Boundary conditions** (Section 6.3): start at rest hanging down, $\mathbf{x}(0) = [0, 0, 0, 0]^\top$; end at rest upright a distance $d$ along the track, $\mathbf{x}(T) = [d, \pi, 0, 0]^\top$.

**Bounds** (Section 6.4): $|q_1| \le d_{\max}$ (finite track) and $|u| \le u_{\max}$ (motor force limit).

**Initial guess** (Section 6.7, eq. 6.16): states moving in a straight line from start to end, $\mathbf{x}(t) = (t/T)\,[d, \pi, 0, 0]^\top$, with zero force.

**Results** (Section 6.8, Figures 9–11): Hermite–Simpson collocation on 25 uniform segments, solved by MATLAB's FMINCON in 71 iterations (5.91 s). Figure 9 shows the swing-up as stop-action frames, Figure 10 the position, angle and force against time, and Figure 11 the collocation error, which peaks mid-swing. **The paper does not print the optimal objective value.**

### Appendix E.1, Table 3: parameters (pp. 896–897)

| Symbol | Value | Meaning |
|---|---|---|
| $m_1$ | 1.0 kg | cart mass |
| $m_2$ | 0.3 kg | pole mass |
| $\ell$ | 0.5 m | pole length |
| $g$ | 9.81 m/s² | gravity |
| $u_{\max}$ | 20 N | maximum actuator force |
| $d_{\max}$ | 2.0 m | track limit, $\lvert q_1\rvert \le d_{\max}$ |
| $d$ | 1.0 m | distance travelled during the swing-up |
| $T$ | 2.0 s | duration of the swing-up |

### Section 4 (from p. 860) and Section 6.6 (pp. 870–871): Hermite–Simpson collocation

The method the paper uses for its result, which we rebuilt as the independent reference for the swing-up test. On a grid of $N$ segments with knots $t_k$, midpoints $t_{k+\frac12}$ and $h_k = t_{k+1} - t_k$, with $f_k = f(\mathbf{x}_k, u_k)$, the nonlinear program (eqs. 6.10–6.15) is:

```math
\min\ \sum_{k=0}^{N-1} \frac{h_k}{6}\left(u_k^2 + 4u_{k+\frac12}^2 + u_{k+1}^2\right)
```

```math
\mathbf{x}_{k+\frac12} = \tfrac12\left(\mathbf{x}_k + \mathbf{x}_{k+1}\right) + \tfrac{h_k}{8}\left(f_k - f_{k+1}\right), \qquad
\mathbf{x}_{k+1} - \mathbf{x}_k = \tfrac{h_k}{6}\left(f_k + 4f_{k+\frac12} + f_{k+1}\right)
```

subject to the bounds and boundary conditions above. The decision variables are the states and forces at every knot and midpoint.

## Our reproduction of the paper's result

Because the paper gives the trajectory but not the optimal cost, we solved its Hermite–Simpson program ourselves (scipy SLSQP; `python -m prototypes.cartpole_swingup --reference`):

| Segments | Optimal $J$ [N²s] |
|---|---|
| 25 (the paper's setup) | 58.8054 |
| 50 | 58.8076 |
| 100 | 58.8077 |

Its force range (−13.8 to +8.9 N) and peak cart travel (1.18 m) match the shape of the paper's Figure 10. Our SCP solver's result, 58.8141 N²s on 41 nodes, is compared with this in [test 2](../known_solutions/02_cartpole_swingup.md).
