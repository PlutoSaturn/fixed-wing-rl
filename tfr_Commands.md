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
