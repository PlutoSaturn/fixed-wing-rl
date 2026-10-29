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

Distance of the flight from the planned path (max and rms, horizontal and vertical), closest approach to the TFR, and whether it stayed collision-free. Saves tfr_env0_summary.json, tfr_env0_flight.csv and tfr_env0_check.png in localstore.

python -m sim.tfr_benchmark check --flight tfr_env0_flight.csv --plot

Compares the flown path with the analytic optimum and the SCP plan: flown length vs 26.610 km, which side, and clearance from the TFR (the flown path may dip into the 36 m planning clearance; it must stay out of the TFR itself).

