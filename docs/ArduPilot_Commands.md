**ArduPilot SITL with the JSBSim Cessna 172: Tier 1 test flight**

The C172 takes off, flies two laps of a 4 x 2 km rectangle at cruise speed (48.9 m/s) and 300 m, then returns home, all in AUTO. The summary then reports how well ArduPilot holds the C172's airspeed, altitude and course, and the throttle it needs for level flight.

Two terminals are used. "WSL" is the Ubuntu terminal (with your venv-ardupilot active, as when you ran SITL before). "PyCharm" is PyCharm's terminal in the project root (the fixed-wing-rl folder). Files the scripts write go to the localstore folder.

____

**One-time setup**

WSL:

cd /mnt/c/Users/isaia/fixed-wing-rl
bash ardupilot/setup_c172_sitl.sh

Copies JSBSim's c172x model into ArduPilot (~/ardupilot/Tools/autotest/aircraft/c172x), adds a small file that starts the engine and keeps it running, and test-loads the model. It should end with "OK: c172x loads and runs." Needs JSBSim built in WSL (JSBSim --version must work). Run it again only if you rebuild ArduPilot or JSBSim.

If it fails with "$'\r': command not found", the script got Windows line endings; fix it with:  sed -i 's/\r$//' ardupilot/setup_c172_sitl.sh

PyCharm:

python -m ardupilot.test_ardupilot_mission

Checks the mission writer and the log reader without ArduPilot. All tests should pass (the Tier 2 tests are skipped until the TFR plan has been solved).

____

**Each test flight**

PyCharm:

python -m ardupilot.ardupilot_mission box

Writes localstore/box.waypoints and prints the exact commands below with your paths filled in. Options: --heading (runway direction, deg), --alt, --length, --width, --laps, --speed.

WSL:

cd ~/ardupilot/ArduPlane
sim_vehicle.py -v ArduPlane -f jsbsim:c172x --custom-location=37.0000000,-120.0000000,0,0 --add-param-file=/mnt/c/Users/isaia/fixed-wing-rl/ardupilot/c172_sitl.parm --console --map -w

Starts ArduPlane SITL with the JSBSim C172 at the planner's origin, with the C172 parameters. -w starts from clean parameters each time, so edits to c172_sitl.parm always take effect.

MAVProxy console (the window SITL opens), once it reports that the EKF is using GPS (about a minute):

wp load /mnt/c/Users/isaia/fixed-wing-rl/localstore/box.waypoints
mode auto
arm throttle

Loads the mission, switches to AUTO and arms; the C172 takes off by itself. The flight takes about 10 minutes. Watch it on the map: it should climb straight ahead, fly the rectangle twice and then circle home.

WSL (a second Ubuntu terminal, or after closing SITL):

cp ~/ardupilot/ArduPlane/mav.tlog /mnt/c/Users/isaia/fixed-wing-rl/localstore/box.tlog

Copies the telemetry log MAVProxy recorded. If it is not there: find ~/ardupilot -name "*.tlog" -mmin -60

PyCharm:

python -m ardupilot.ardupilot_mission summary --mission box.waypoints --log box.tlog --plot

Prints the airspeed, altitude and course errors and the level-flight throttle, with PASS/WARN lines, and saves localstore/box_summary.png and box_summary.json. If it says to change TRIM_THROTTLE, set the value it gives in ardupilot/c172_sitl.parm (and TRIM_THROTTLE_PCT in ardupilot/ardupilot_mission.py) and fly again.

____

**If something goes wrong**

- SITL says it cannot start JSBSim: the terminal does not have JSBSim on its PATH; run source ~/.bashrc and check JSBSim --version.
- "arm throttle" is refused: MAVProxy prints the reason (PreArm: ...). Most often the EKF is not ready yet; wait and try again.
- The C172 does not move after arming: if the console shows throttle but the airspeed stays at 0, the engine is not running; run the setup script again.
- The C172 noses over or pitches the wrong way on the takeoff run: the pitch direction is reversed. Stop and send the MAVProxy output.

____

**Tier 2: fly the SCP TFR plan**

The C172 takes off upstream of the workspace, climbs in a loiter to the plan's altitude, lines up on the plan's start heading, flies the SCP plan around the stadium TFR as waypoints, then leaves the workspace and loiters. The flight is then compared with the plan, the TFR and the 26.610 km optimum. Needs the solved TFR plan in localstore (tfr_env.json and tfr_results.json; see Known_Solutions_Commands.md).

PyCharm:

python -m ardupilot.ardupilot_mission make --envs tfr_env.json --results tfr_results.json --index 0

Writes localstore/tfr_env0.waypoints and prints the commands below with the right home location. Waypoints follow the plan within 3 m on curves (one every 12 deg of turn or 600 m, never closer than 200 m); ArduPilot flies one airspeed, the plan's mean kept 4 m/s below the C172's maximum (55.2 m/s for the TFR plan).

WSL (close any SITL still running first; note the home location differs from the box test):

cd ~/ardupilot/ArduPlane
sim_vehicle.py -v ArduPlane -f jsbsim:c172x --custom-location=37.0735913,-120.0131200,0,90 --add-param-file=/mnt/c/Users/isaia/fixed-wing-rl/ardupilot/c172_sitl.parm --console --map -w

MAVProxy (the sim_vehicle terminal, at the MANUAL> prompt), once the EKF is using GPS:

wp load /mnt/c/Users/isaia/fixed-wing-rl/localstore/tfr_env0.waypoints
mode auto
arm throttle

The whole flight takes about 25-35 minutes: takeoff and climb to 600 m above home, the lead-in, the 8-minute plan, and the exit. It is done when the C172 circles east of the workspace.

WSL (a second Ubuntu terminal):

cp ~/ardupilot/ArduPlane/mav.tlog /mnt/c/Users/isaia/fixed-wing-rl/localstore/tfr_env0.tlog

PyCharm:

python -m ardupilot.ardupilot_mission analyze --mission tfr_env0.waypoints --log tfr_env0.tlog --plot

Distance of the flight from the planned path (max and rms, horizontal and vertical), closest approach to the TFR, whether it stayed collision-free, and how much the roll, commanded roll, pitch and altitude oscillate (rms and period, leaving out the first 30 s). Saves tfr_env0_summary.json, tfr_env0_flight.csv and tfr_env0_check.png in localstore (--plot also shows the plots).

Every analyzed flight is also archived in localstore/runs/<log name>/: a copy of the log (flight.tlog), the summary, the flown path, the plot, and params.parm, every autopilot parameter the flight actually used (read from the log). localstore/runs/runs.csv lists all runs in the order they were flown, with the main results and the parameters that changed since the previous run. Add --note "what changed" to record why a run was flown. Analyzing the same log again updates its run; if a different flight was saved under a log name already archived (an overwritten log), it is archived as <log name>_2 and analyze says so.

python -m sim.tfr_benchmark check --flight tfr_env0_flight.csv --plot

Compares the flown path with the analytic optimum and the SCP plan: flown length vs 26.610 km, which side, and clearance from the TFR (the flown path may dip into the 36 m planning clearance; it must stay out of the TFR itself).

____

**Tuning the autopilot (TFR mission)**

Change one setting at a time in ardupilot/c172_sitl.parm (with a comment saying why), fly the same mission, and give each flight's log its own name so the comparison stays clean. SITL can run faster than real time: add --speedup 5 to the sim_vehicle.py command (JSBSim may manage only about 2x).

MAVProxy, before arming (check the change was loaded):

param show TECS_TIME_CONST

WSL (a second Ubuntu terminal), with a new name for each run:

cp ~/ardupilot/ArduPlane/mav.tlog /mnt/c/Users/isaia/fixed-wing-rl/localstore/tfr_env0_nav1.tlog

PyCharm:

python -m ardupilot.ardupilot_mission analyze --mission tfr_env0.waypoints --log tfr_env0_nav1.tlog --note "NAVL1_DAMPING 0.75->0.85" --plot

Then open localstore/runs/runs.csv to compare it with the earlier runs.

To test a different waypoint spacing, write the mission under its own name (the plan and home stay the same) and use that name in wp load and analyze; runs.csv records each run's mission, waypoint count and chord tolerance:

python -m ardupilot.ardupilot_mission make --envs tfr_env.json --results tfr_results.json --index 0 --chord-tol 1 --out tfr_env0_chord1.waypoints

To compare settings within one flight, change them with param set in MAVProxy (about 2 minutes of flight each, e.g. in CRUISE after rc 3 1846 and mode CRUISE), then:

python -m ardupilot.ardupilot_mission wobble --log <log name>.tlog

It reports roll, pitch, altitude and throttle oscillation, and how far pitch lags its command, for every stretch of the flight in one flight mode with the same parameters (a new row at every mode change and every param set).

AUTOTUNE without a transmitter: in MAVProxy run output add 127.0.0.1:14552, take off, and once circling at altitude run in a second Ubuntu terminal (ArduPilot Python environment active):

python3 /mnt/c/Users/isaia/fixed-wing-rl/ardupilot/autotune_inputs.py --axis pitch

(or --axis roll). It holds 80% throttle, sends the stick inputs, and ends in LOITER, which keeps the new gains. --level sets AUTOTUNE_LEVEL: 0 (the script's default) keeps the current time constant and rate limit and tunes only the PID gains; 6 (ArduPilot's recommended level) also picks the time constant and rate limit. Run AUTOTUNE at real time (no --speedup): the stick inputs are timed in real seconds. Copy the new values (param show PTCH_RATE* PTCH2SRV* or RLL_RATE*) into ardupilot/c172_sitl.parm; SITL started with -w forgets them otherwise. The roll gains in c172_sitl.parm come from level 0; the pitch gains from level 6, but with the time constant kept at 0.25 s (level 6 chose 0.75 s, which made TECS cycle).

To measure the wobble in straight flight, or try several settings in one flight, without typing param set: take off (e.g. the box mission), run output add 127.0.0.1:14552 in MAVProxy, then in a second Ubuntu terminal:

python3 /mnt/c/Users/isaia/fixed-wing-rl/ardupilot/wobble_sweep.py --baseline-only

It flies CRUISE (straight and level) for 150 simulated seconds per setting and puts every parameter back at the end. Without --baseline-only it steps through the settings listed at the top of wobble_sweep.py (--preset v2: TECS and pitch-damping candidates; --preset l6: the level-6 pitch gains with several time constants). Copy mav.tlog and run wobble on it for one row per setting. SITL may run at --speedup 5 for this; the script times each setting in simulated seconds.

____

**Tier 3: fly SCP plans for generated environments, as a batch**

Generate C172-scale environments, solve them with SCP, turn every solved plan into a mission (checking that it can be flown as waypoints), fly them all in SITL unattended, then score each flight. Everything for one batch lives in localstore/batches/<name>/: manifest.json, missions/, logs/ (one .tlog, .run.json and .sitl.txt per flight), runs/ (per-flight results and plots), results.csv, summary.json and summary.png.

PyCharm, step 1: generate the pilot batch's environments (3 easy, 4 nominal, 3 dense):

python -m environment.envgen --size 12000 2500 600 --vehicle-radius 6 --margin 30 --endpoints ends --occupancy 0.02 0.04 --n 3 --seed 101 --out tier3_easy_envs.json

python -m environment.envgen --size 12000 2500 600 --vehicle-radius 6 --margin 30 --endpoints ends --occupancy 0.05 0.08 --n 4 --seed 102 --out tier3_nominal_envs.json

python -m environment.envgen --size 12000 2500 600 --vehicle-radius 6 --margin 30 --endpoints ends --occupancy 0.09 0.12 --n 3 --seed 103 --out tier3_dense_envs.json

PyCharm, step 2: solve them (about 1-2 minutes per environment):

python -m solvers.scp_aircraft --envs tier3_easy_envs.json --out tier3_easy_results.json --feasible-out tier3_easy_feasible.json

python -m solvers.scp_aircraft --envs tier3_nominal_envs.json --out tier3_nominal_results.json --feasible-out tier3_nominal_feasible.json

python -m solvers.scp_aircraft --envs tier3_dense_envs.json --out tier3_dense_results.json --feasible-out tier3_dense_feasible.json

PyCharm, step 3: build the batch:

python -m ardupilot.ardupilot_mission batch-make --name tier3_v4 --pair tier3_easy_envs.json tier3_easy_results.json --pair tier3_nominal_envs.json tier3_nominal_results.json --pair tier3_dense_envs.json tier3_dense_results.json --note "tracking setup v4"

For each environment it prints flyable, not_flyable (with the reason) or not_solved. A plan is not flyable as waypoints when the expected flown path comes closer to an obstacle than half the planner's margin, or when a corner needs more bank than the autopilot may use.

Missions use tracking setup v4 by default (v3 plus the plan's airspeed on every leg): a waypoint every 30 m along the plan (BATCH_WP_SPACING_M; the plan resampled at small equal steps, so ArduPilot's roll follows the plan's bank instead of jumping at each waypoint), each shifted by the tracking correction (BATCH_TRACK_COMP = "both"): towards the curve's centre by TRACK_COMP_K_M2 / radius, and the altitude ahead of the climb, cancelling the offsets ArduPlane was measured to fly. The flyability check uses the unshifted waypoints, the path the shifts are meant to produce. --wp-spacing N changes the spacing; --wp-spacing 0 goes back to the chord-tolerance rule (--chord-tol, 1 m: tracking setups v1 and v2); --track-comp none or lateral drops all or the altitude part of the correction. The correction's constants were fitted at ~52 m/s with c172_sitl.parm's settings; refit them if NAVL1_PERIOD, TECS_TIME_CONST, TECS_SPDWEIGHT or the pitch gains change. Other mission options: --arcs flies the plan's curves as fractional loiter turns (a steady bank per curve; passed dense_1 but roll spiked between arcs), --max-waypoints N uses exactly N waypoints (an experiment: tracking gets much worse), --fly-all flies missions even when the flyability check rejects them (for experiments). With the 30 m waypoints, batch missions follow the plan's airspeed (BATCH_SPEED_PROFILE = "plan", tracking setup v4): a DO_CHANGE_SPEED with the plan's mean airspeed over each leg, added wherever it changes by 0.5 m/s. On the 7 Tier 3 plans this cut the worst max error from 24.8 to 9.9 m and removed dense_1's near miss, at the cost of a busier throttle (1.3-3x more throttle movement). --speed-profile constant flies one airspeed, the plan's mean (tracking setups v1-v3); with --wp-spacing 0, --arcs, --max-waypoints or --alt-lead the default is constant too, because with long legs the speed commands arrived 5-10 s late and tracking got worse. --speed-lead S commands each speed S seconds early (tried: worse than no lead). make takes the same option. --param NAME=VALUE (repeatable) gives a batch its own autopilot parameters, on top of c172_sitl.parm: run_batch.py sets them before arming, and they are recorded in each flight's log, params.parm and runs.csv. Use it to test a setting on a batch without editing c172_sitl.parm. --param-in-plan NAME=VALUE is set in flight instead, when the aircraft heads for the plan's first waypoint (e.g. a throttle lock, THR_MIN=80 and THR_MAX=80, that would starve the takeoff if set before arming).

WSL, step 4: fly every flyable one, unattended (close any running SITL first; a terminal with your ArduPilot Python environment active, where sim_vehicle.py works):

python3 /mnt/c/Users/isaia/fixed-wing-rl/ardupilot/run_batch.py /mnt/c/Users/isaia/fixed-wing-rl/localstore/batches/tier3_v4/manifest.json

For each instance it starts SITL at that mission's home (no MAVProxy), records the log, uploads the mission, arms in AUTO, waits until the aircraft has circled in the final loiter for 30 s (simulated), and stops SITL. Each flight is about 10 simulated minutes (about 5 real minutes at --speedup 5). Ctrl-C stops it; running the same command again skips the instances already completed (--redo flies them again, --only ID flies just one). If an instance fails to start, logs/<id>.sitl.txt has SITL's output.

PyCharm, step 5: score the batch:

python -m ardupilot.ardupilot_mission batch-analyze --name tier3_v4 --plot

Each flown instance PASSES if it never touched an obstacle or left the workspace, came within 50 m of the goal, stayed within the planner's margin (30 m) of the plan, kept the rms distance to the plan at or under 10 m, and took within 10% of the plan's flight time (PASS_* settings in ardupilot_mission.py). Flown clearance under half the margin is flagged as a near miss. Outcomes: PASS, FAIL, REJECTED (not flyable as waypoints), NOT_FLOWN (e.g. never armed) and NOT_SOLVED.

Tracking setup v5 (GUIDED follower): instead of uploading the plan as waypoints, stream it in GUIDED mode, as a companion computer would. Every solved plan stayed within 8.1 m of its plan and 1.8 m sideways (v4: 9.9 m), at the cost of a busier throttle. Needs ArduPlane 4.3 or newer (cd ~/ardupilot && git describe --tags). Build the batch with --follower guided (lookahead 150 m, altitude lead 30 m and TECS_CLMB_MAX 6.5 are the defaults: GUIDED_* in ardupilot_mission.py):

python -m ardupilot.ardupilot_mission batch-make --name tier3_v5 --follower guided --note "tracking setup v5" --pair tier3_easy_envs.json tier3_easy_results.json --pair tier3_nominal_envs.json tier3_nominal_results.json --pair tier3_dense_envs.json tier3_dense_results.json

Fly it at --speedup 3 (not 5), so guided_follower.py keeps up with its 20 Hz stream:

python3 /mnt/c/Users/isaia/fixed-wing-rl/ardupilot/run_batch.py /mnt/c/Users/isaia/fixed-wing-rl/localstore/batches/tier3_v5/manifest.json --speedup 3

Each flight logs the follower's states (wait, engaging, active, leaving, done) and writes logs/<id>.guided.csv with what it commanded. Score it as usual (batch-analyze --name tier3_v5). Other options: --lookahead, --guided-alt-lead, --guided-alt-comp on/off, --guided-alt-rate, --guided-alt-step, --guided-speed-ramp (tried, worse). In a Windows terminal, don't click inside the window while a batch runs: QuickEdit mode pauses the script until you press Esc.
