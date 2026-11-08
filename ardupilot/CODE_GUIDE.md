# ArduPilot code guide

What each file in `ardupilot/` does and how the pieces fit together. For the results see [README.md](README.md); for the exact commands to run see [docs/ArduPilot_Commands.md](../docs/ArduPilot_Commands.md).

## How the pieces fit together

```
SCP plan (solvers/scp_aircraft.py)
   |
   v
ardupilot_mission.py make / batch-make  -> mission file (.waypoints) + flyability check + manifest
   |
   v
run_batch.py (WSL)                       -> starts ArduPlane SITL + JSBSim C172, uploads, flies, records the .tlog
   |  (GUIDED batches, --follower guided: guided_follower.py streams the plan during the flight)
   |
   v
ardupilot_mission.py analyze / batch-analyze -> distance from the plan, clearance, oscillation, PASS/FAIL, plots, archive
```

The autopilot's settings live in `c172_sitl.parm`, loaded every time SITL starts. `autotune_inputs.py` and `wobble_sweep.py` are tuning tools used during a live SITL session. `fix_sweep.py`, `speed_lead_test.py` and `guided_test.py` compare candidate setups as batches. `test_ardupilot_mission.py` checks all of it without ArduPilot.

| File | Runs in | In one line |
| --- | --- | --- |
| `ardupilot_mission.py` | PyCharm | Builds missions from SCP plans, checks they can be flown, scores flights, runs batches |
| `run_batch.py` | WSL | Flies a whole batch in SITL, unattended |
| `autotune_inputs.py` | WSL, during a SITL session | Runs ArduPilot AUTOTUNE without a transmitter |
| `wobble_sweep.py` | WSL, during a SITL session | Flies straight and level and steps through autopilot settings to measure the wobble |
| `fix_sweep.py` | PyCharm | Builds and scores several candidate setups on the same plans |
| `speed_lead_test.py` | PyCharm | Builds and scores dense_1 with the plan's speeds commanded 0, 4 or 8 s early |
| `guided_follower.py` | WSL, inside `run_batch.py` | Tracking setup v5: streams the plan to ArduPlane in GUIDED mode (the "companion computer") |
| `guided_test.py` | PyCharm | Builds and scores GUIDED follower settings on chosen plans |
| `test_ardupilot_mission.py` | PyCharm | Tests (`python -m ardupilot.test_ardupilot_mission`) |

## `ardupilot_mission.py`

The main module (about 2000 lines). Everything is run as `python -m ardupilot.ardupilot_mission <command>` from the project root.

### Commands

| Command | Tier | What it does |
| --- | --- | --- |
| `box` | 1 | Writes a rectangle test mission around home (`localstore/box.waypoints`) |
| `summary` | 1 | Checks a flown box mission: airspeed, altitude and course errors, level-flight throttle |
| `make` | 2 | Writes the mission for one SCP plan (`.waypoints` + `.waypoints.json`) and prints the SITL commands to fly it |
| `analyze` | 2 | Scores one flown plan against its plan and obstacles; saves a summary, the flown path and the 6-panel `check.png`; archives the run |
| `wobble` | tuning | Roll, pitch, altitude and throttle oscillation for every stretch of a flight in one flight mode with the same parameters |
| `batch-make` | 3 | Missions + flyability checks for every solved environment, and the `manifest.json` that `run_batch.py` flies |
| `batch-analyze` | 3 | Scores every flight in a batch: `results.csv`, `summary.json`, `summary.png` |

### Settings (top of the file)

All tunable numbers are named constants near the top, each with a comment saying why it has its value:

- **Waypoints and speed:** `BATCH_WP_SPACING_M` (30 m, tracking setup v3), `BATCH_TRACK_COMP` (`"both"`), `BATCH_SPEED_PROFILE` (`"plan"`, tracking setup v4; `SPEED_PROFILE` is the single-plan `make` default), `CHORD_TOL_M` / `BATCH_CHORD_TOL_M`, `MIN_WP_SPACING_M`, `MAX_WP_SPACING_M`, `MAX_WP_TURN_DEG` (the older chord-tolerance rule).
- **Tracking correction:** `TRACK_COMP_K_M2` (sideways shift = K / radius), `TRACK_COMP_SLOPE_M`, `TRACK_COMP_CURV_M2`, `TRACK_COMP_BIAS_M` (altitude shift), and the caps. Fitted from flight data at ~52 m/s; refit if the autopilot settings change.
- **Arcs (`--arcs`):** `ARC_MAX_RADIUS_M`, `ARC_MIN_TURN_DEG`, `ARC_GAP_M`, `ARC_ENTRY_ACCEPT_M`, and so on.
- **Speed:** `SPEED_PROFILE` (`"constant"`; `"plan"` was tested and was worse).
- **GUIDED follower (`--follower guided`, tracking setup v5):** `GUIDED_LOOKAHEAD_M` (150 m), `GUIDED_ALT_LEAD_M` (30 m), `GUIDED_ALT_RATE_MS` and `GUIDED_ALT_STEP_M` (how the altitude is sent), `GUIDED_PARAM_OVERRIDES` (`TECS_CLMB_MAX` 6.5, GUIDED batches only), `GUIDED_REF_STEP_M`, `GUIDED_EXTEND_BACK_M` / `GUIDED_EXTEND_FWD_M` (the reference trajectory).
- **Flyability:** `FLY_MIN_CLEARANCE_FRAC` (the expected path must keep half the planner's margin), `AP_ROLL_LIMIT_DEG`.
- **Pass/fail:** `PASS_MAX_ERROR_FRAC`, `PASS_RMS_M`, `PASS_GOAL_M`, `PASS_TIME_FRAC`, `NEAR_MISS_FRAC`.
- **Measurement:** `OSC_WINDOW_S`, `OSC_SKIP_S` (how oscillation is measured).

### Functions, by job

**Coordinates and helpers:** `enu_to_geo`, `geo_to_enu` (planner metres <-> latitude/longitude), `compass_deg` (planner course -> compass heading), `wsl_path` (a Windows path as WSL sees it), `load_airframe` (the planner's C172 parameters), `turn_radius`.

**From plan to waypoints:**
- `dense_plan`: re-integrates the SCP plan with the planner's own model (RK4, first-order hold on the controls) into a fine trajectory. All waypoints are picked from this.
- `uniform_indices`: a waypoint every N metres along the plan (v3 default, 30 m).
- `track_compensate`: shifts each waypoint to cancel ArduPilot's measured tracking offsets: towards the curve's centre, and the altitude ahead of the climb.
- `resample`, `chord_deviation`: the older rule (v1, v2): a new waypoint when the straight leg would stray more than the chord tolerance from the plan.
- `select_waypoints`: exactly N waypoints that best keep the plan's shape (the 5-waypoint experiment).
- `lead_altitudes`: waypoint altitudes taken some seconds ahead (tried, not used).
- `plan_arcs`, `fit_circle`, `arc_sequence`, `arc_path`, `loiter_radius_command`: the `--arcs` option: finds the plan's curves, fits a circle to each, writes them as fractional loiter turns, and corrects the radius for ArduPilot's air-density scaling.
- `leg_speeds`: per-leg airspeeds for `--speed-profile plan` (the batch default since v4).
- `lead_speeds`: the plan's airspeed a given number of seconds ahead, for `--speed-lead` (tried, not used).
- `guided_reference`: the trajectory the GUIDED follower flies: the plan every 5 m, with the commanded altitude (plus the climb lead), the airspeed, and straight extensions before the start and after the goal.
- `guided_mission`: turns a waypoint mission into a GUIDED one (takeoff, climb, lead-in, the plan's start, exit, final loiter) and builds its reference.

**Building and checking a mission:**
- `build_mission`: everything for one plan: the waypoints (by any of the rules above), the lead-in, climb loiter, exit route and final loiter, the speed command, and the flyability checks (expected path clearance, corner bank).
- `autopilot_path`, `densify`, `point_to_polyline`, `body_clearance`: the expected flown path and its distance to the plan and to obstacles.
- `exit_route`, `ray_box_exit`: the way out of the workspace after the goal.
- `write_waypoints`, `write_mission`, `sitl_home`, `sitl_commands`: write the QGC `.waypoints` file and its `.json` companion (read later by `analyze`), and the commands to fly it.
- `plot_mission_preview`: a top-view preview of a mission before flying it (saved by `batch-make` as `missions/<id>_preview.png`).
- `make`: the `make` command.

**Tier 1 box test:** `box_corners`, `box`, `legs_position`, `summary`, `plot_box`.

**Reading and analyzing flights:**
- `read_tlog` (with `_crc`): reads a MAVProxy `.tlog` directly (no pymavlink needed): position, airspeed, throttle, attitude, commanded roll and pitch, flight mode and every parameter, on the autopilot's clock, keeping only the latest session in the log.
- `read_bin`, `read_flight`: DataFlash `.BIN` logs (via pymavlink) and a common entry point.
- `oscillation`: size and period of the fast part of a signal (the wobble measure). `follow_lag`: how far a response trails its command.
- `analyze`, `plot`: the `analyze` command and its 6-panel chart.
- `wobble`: the `wobble` command.

**Run archive:** `archive_run`, `write_runs_table`, `param_changes`, `write_params`, `read_params`, `_file_hash`. Every analyzed flight is copied, with the parameters it actually flew, to a `runs/` folder; `runs.csv` lists all runs and what changed between them.

**Batches:** `batch_path`, `batch_make`, `parse_param_overrides` (`--param NAME=VALUE`), `evaluate_flight` (the pass/fail rules), `batch_analyze`, `plot_batch`.

**`main`:** the command-line options for every command.

## `run_batch.py`

Flies every flyable instance in a batch's `manifest.json`, one after another, unattended. Runs in WSL with your ArduPilot Python environment (needs only pymavlink):

```
python3 /mnt/c/Users/isaia/fixed-wing-rl/ardupilot/run_batch.py /mnt/c/Users/isaia/fixed-wing-rl/localstore/batches/<name>/manifest.json
```

For each instance:
1. **`start_sitl`** starts ArduPlane SITL with JSBSim's C172 at the mission's home, with `c172_sitl.parm`, without MAVProxy.
2. **`connect`** opens the MAVLink link (the `Link` class: keeps a heartbeat, records every message to the `.tlog`, tracks mode, arming, altitude and mission item).
3. **`set_params`** applies the batch's `--param` overrides and waits for the aircraft to confirm each.
4. **`fly`** uploads the mission (`Link.upload`), arms in AUTO, follows the flight, sets any `--param-in-plan` parameters once the plan starts, and decides when it is done (30 s circling in the final loiter) or what went wrong.
5. **`stop_sitl`** shuts SITL and JSBSim down and writes `logs/<id>.run.json`.

For GUIDED batches (`--follower guided`), `fly` also starts a `GuidedFollower` (from `guided_follower.py`), asks for position at 20 Hz (simulated), and hands every position and command acknowledgement to it; fly these at `--speedup 3`.

Statuses: `completed`, `timeout`, `crashed`, `sitl_died`, `arm_failed`, `upload_failed`, `param_failed`, `sitl_failed`, `guided_failed`, `error`, `interrupted`. Options: `--only ID ...`, `--redo`, `--speedup N`, `--ardupilot DIR`. Running it again skips the instances already completed.

## `autotune_inputs.py`

Runs ArduPilot AUTOTUNE in SITL without a transmitter: AUTOTUNE needs sharp stick inputs, so the script sends them as RC overrides. It sets the AUTOTUNE options, holds a fixed throttle, checks the aircraft actually takes its stick inputs (it must use MAVLink system id 255, ArduPilot's ground-station id), switches to AUTOTUNE, sends stick doublets, and ends in LOITER, which keeps the new gains. Options: `--axis pitch|roll|both`, `--level` (0 keeps the time constant and rate limit; 6 also tunes them), `--cycles`, `--amplitude`, `--throttle`. Run it at real-time speed: the inputs are timed in real seconds.

## `wobble_sweep.py`

Measures the pitch/roll wobble in straight flight, and tries several autopilot settings in one flight. During a live SITL session it switches to CRUISE (straight and level), holds each setting for a set number of *simulated* seconds (so SITL may run at 5x), puts the original values back before the next one, and ends in LOITER with every parameter restored. `ardupilot_mission wobble` on the log then gives one row per setting.

- `--baseline-only`: just fly straight with the current settings (e.g. right after AUTOTUNE).
- `--preset v2`: the TECS and pitch-damping candidates.
- `--preset l6`: loads the AUTOTUNE level-6 pitch gains in flight and tries several time constants (this found the v3 pitch gains).

## `fix_sweep.py` and `fly_fix_sweep.sh`

Compare several candidate setups on the same three plans (easy_0, nominal_2, dense_1). `python -m ardupilot.fix_sweep make` builds one batch per setup and checks every flight is flyable; `fly_fix_sweep.sh` (WSL) flies them all unattended; `python -m ardupilot.fix_sweep analyze` scores them and prints a side-by-side table against `tier3_v2`. The setups listed in the file are the v3 tuning round (`fixS1`-`fixS4`); edit `CONFIGS` to compare others. Its batches keep the old waypoint rule and one constant airspeed (not the v3/v4 defaults), so they stay comparable.

## `speed_lead_test.py` and `fly_speed_lead.sh`

The v4 experiment: dense_1 on tracking setup v3 with the plan's airspeed on every leg, commanded 0, 4 or 8 s early (batches `spd0`, `spd4`, `spd8`). `python -m ardupilot.speed_lead_test make` builds them, `fly_speed_lead.sh` (WSL) flies them, `python -m ardupilot.speed_lead_test analyze` compares them with `tier3_v3b`. No lead was best (max error 9.5 m, against 15.5 m with 4 s and 14.4 m with 8 s), which became tracking setup v4.

## `guided_follower.py`

Tracking setup v5: the plan flown by streaming setpoints in GUIDED mode instead of uploading waypoints (the way a companion computer drives a flight controller). Pure Python, no numpy, so it runs in ArduPilot's environment. The `GuidedFollower` class:
- **`on_position`** (called for every `GLOBAL_POSITION_INT`): switches to GUIDED near the plan's start; then each update finds the aircraft's place on the plan (`project`) and sends the roll (L1 law aimed `lookahead_m` along the curved plan; `send_roll`, `SET_ATTITUDE_TARGET` with only the roll used), the altitude (`send_alt`, `MAV_CMD_GUIDED_CHANGE_ALTITUDE`) and the airspeed (`send_speed`, `MAV_CMD_GUIDED_CHANGE_SPEED`); at the goal it hands back to AUTO at the exit waypoint.
- **`on_ack`**: fails the flight if ArduPlane keeps rejecting the GUIDED commands (they need ArduPlane 4.3+).
- Writes `logs/<id>.guided.csv`: per update, the distance along the plan, sideways error, aim angle and the roll, airspeed and altitude it commanded.

## `guided_test.py` and `fly_guided_test.sh`

The v5 experiments: GUIDED follower settings flown on chosen plans and compared with `tier3_v4`, including the throttle (travel in %/min and time at 0 or 100%). `python -m ardupilot.guided_test make`, `fly_guided_test.sh` (WSL), `python -m ardupilot.guided_test analyze`. Its docstring records the three rounds (lookahead; speed ramp; altitude rate and lead).

## `test_ardupilot_mission.py`

Checks the module without ArduPilot, using synthetic flights and logs written by the test itself. Run `python -m ardupilot.test_ardupilot_mission`; it prints `ok` per test and ends with the count.

- **Always run:** the box mission, the `.tlog` reader (MAVLink 1 and 2, sessions, attitude on the same clock), the oscillation and wobble measures, geometry helpers, waypoint rules (chord tolerance, fixed count, even spacing), the tracking correction, arcs and the loiter-radius model (against the measured flight), parameter parsing and the pass/fail rules.
- **Tier 2 (when `localstore` has the solved TFR plan):** mission structure, `analyze` recovering known offsets and wobbles, run archiving, flyability rejection, start detection, collision detection, and an end-to-end `batch-make` + `batch-analyze`.

## Other files

| File | What it is |
| --- | --- |
| `c172_sitl.parm` | ArduPlane parameters for the JSBSim C172: ArduPilot's SITL defaults, the airframe's speeds and limits, and the tuned gains (tracking setup v3, used by v3-v5; v5 adds `TECS_CLMB_MAX` 6.5 per batch). Every tuned value has a comment with the test that justified it |
| `setup_c172_sitl.sh` | One-time setup (WSL): copies JSBSim's c172x model into ArduPilot and checks it loads |
| `figures/` | Images used in the README |
| `README.md` | Results and how the tracking setup evolved |
