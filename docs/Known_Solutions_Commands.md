**Known-solution test: stadium TFR detour**

A C172 flies 24 km past a stadium TFR (3 NM radius, surface to 3,000 ft AGL, FDC NOTAM 4/3621), staying below 600 m, so it has to go around. The shortest possible path is known exactly: 26.610 km on the north side. No collision-free plan can be shorter.

Run these from the project root (the fixed-wing-rl folder). Generated files (tfr_env.json, tfr_results.json, tfr_check.png, the tfr_flight folder) go to the localstore folder.

python -m sim.tfr_benchmark make

Writes tfr_env.json (one full-height cylinder) and prints the analytic optimum.

python -m solvers.scp_aircraft --envs tfr_env.json --out tfr_results.json --feasible-out tfr_feasible.json --verbose

Solves it. The separate --out file keeps your main scp_aircraft_results.json untouched.

python -m sim.tfr_benchmark check --plot

Compares the SCP plan to the optimum. PASS = within 2% and clear of the TFR; WARN = 2–5% or went around the longer side; FAIL = over 5%, entered the TFR, cut into its own clearance, or (impossible for a correct solver) shorter than the optimum. Saves tfr_check.png.

python -m sim.jsbsim_c172 record 0 --envs tfr_env.json --results tfr_results.json --out-dir tfr_flight
python -m sim.tfr_benchmark check --flight tfr_flight/jsbsim_commands.csv --plot

Flies the plan in JSBSim and checks that the flown path stays out of the TFR too.

____

[Demo Video](https://huggingface.co/datasets/PlutoJupiter/fixed-wing-rl/resolve/main/SCPdemo.mp4)

____

**Known-solution tests: cart-pole**

Run these from the project root (the fixed-wing-rl folder). The "-m" form is needed since the files were organized into folders; "python cartpole_scp.py" no longer finds the solver.

**Cart-pole balanced move**

# this solves a "cart-pole" problem, where an inverted pendulum begins upright on a cart and that cart needs to move 1 meter keeping the pendulum upright
# the known solution for the minimum amount of effort required is J* = 10.78111 N^2 * s, which the solver achieves

python -m prototypes.cartpole_scp --plot

Solves the linearized cart-pole problem (move 1 m in 2 s while balanced upright) with the SCP solver, checks the result against the exact optimal solution with PASS/FAIL lines, and saves the comparison plot to cartpole_check.png in the localstore folder.

python -m prototypes.cartpole_scp --model nonlinear

Solves the same task with the full nonlinear cart-pole equations to confirm the solver handles nonlinear dynamics, reporting the cost next to the linear optimum for reference, since this version has no exact solution.

python -m prototypes.cartpole_scp --reference

Prints the exact optimal solution (the optimal cost J*, the best cost achievable with the solver's node count, and the optimal force profile) without running the solver.

____

**Cart-pole swing-up**

# this solves the cart-pole "swing-up" from Kelly (2017), the paper the cart-pole model comes from: the pendulum starts hanging straight down under the cart, at rest, and in 2 s the cart must swing it up to balance upright, ending 1 m along the track, at rest, with as little effort as possible (minimum integral of force squared, force limited to 20 N, cart limited to 2 m either side)
# the solver is not told the answer; at the end its result is compared with the paper's own method (Hermite-Simpson collocation on 25 segments, rebuilt here and solved separately), which gives J = 58.805 N^2 * s. The SCP solver reaches 58.814 N^2 * s with 41 nodes (+0.015%)

python -m prototypes.cartpole_swingup

Solves the swing-up with the SCP solver, replays the planned force on the nonlinear pendulum to confirm it really swings up, holds it upright afterwards with an LQR controller, prints how the solution developed iteration by iteration, and compares the final cost with the paper's method, with PASS/FAIL lines.

python -m prototypes.cartpole_swingup --plot --gif

Does the same and saves four files to the localstore folder: cartpole_swingup_progress.png (selected iterations, each one's force replayed on the pendulum, from failed swings to the final swing-up and hold), cartpole_swingup_convergence.png (cost, dynamics error and swing-up miss at every iteration), cartpole_swingup_result.png (the final trajectory next to the paper's method, in the style of the paper's Figures 9-10) and cartpole_swingup.gif (an animation of the same iterations).

python -m prototypes.cartpole_swingup --nodes 81

Solves on a finer grid (81 nodes instead of 41) to show the cost settling as the discretization is refined.

python -m prototypes.cartpole_swingup --reference

Solves only the paper's method (Hermite-Simpson collocation) and prints its optimal cost and force range, without running the SCP solver. Add --ref-segments 50 for a finer reference (takes about 30 s).
