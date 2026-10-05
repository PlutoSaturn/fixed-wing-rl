**Known-solution test: stadium TFR detour**

A C172 flies 24 km past a stadium TFR (3 NM radius, surface to 3,000 ft AGL, FDC NOTAM 4/3621), staying below 600 m, so it has to go around. The shortest possible path is known exactly: 26.610 km on the north side. No collision-free plan can be shorter.

python tfr_benchmark.py make

Writes tfr_env.json (one full-height cylinder) and prints the analytic optimum.

python scp_aircraft.py --envs tfr_env.json --out tfr_results.json --feasible-out tfr_feasible.json --verbose

Solves it. The separate --out file keeps your main scp_aircraft_results.json untouched.

python tfr_benchmark.py check --plot

Compares the SCP plan to the optimum. PASS = within 2% and clear of the TFR; WARN = 2–5% or went around the longer side; FAIL = over 5%, entered the TFR, cut into its own clearance, or (impossible for a correct solver) shorter than the optimum. Saves tfr_check.png.

python jsbsim_c172.py record 0 --envs tfr_env.json --results tfr_results.json --out-dir tfr_flight
python tfr_benchmark.py check --flight tfr_flight/jsbsim_commands.csv --plot

Flies the plan in JSBSim and checks that the flown path stays out of the TFR too.

____

[Demo Video](https://huggingface.co/datasets/PlutoJupiter/fixed-wing-rl/resolve/main/SCPdemo.mp4)

____

**Known-solution test: cart-pole balanced move**

# this solves a "cart-pole" problem, where an inverted pendulum begins upright on a cart and that cart needs to move 1 meter keeping the pendulum upright
# the known solution for the minimum amount of effort required is J* = 10.78111 N^2 * s, which the solver achieves

python cartpole_scp.py --plot

Solves the linearized cart-pole problem (move 1 m in 2 s while balanced upright) with the SCP solver, checks the result against the exact optimal solution with PASS/FAIL lines, and saves the comparison plot to cartpole_check.png.

python cartpole_scp.py --model nonlinear

Solves the same task with the full nonlinear cart-pole equations to confirm the solver handles nonlinear dynamics, reporting the cost next to the linear optimum for reference, since this version has no exact solution.

python cartpole_scp.py --reference

Prints the exact optimal solution (the optimal cost J*, the best cost achievable with the solver's node count, and the optimal force profile) without running the solver.