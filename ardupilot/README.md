# ArduPilot SITL flight validation

This folder flies SCP plans through a real autopilot, ArduPilot (ArduPlane SITL with the JSBSim Cessna 172), and scores each flight against its plan. It answers one question for the benchmark: **can a plan that is feasible on paper actually be flown, and when it can't, is that the plan's fault or the autopilot's?**

Step-by-step commands for every tier are in [docs/ArduPilot_Commands.md](../docs/ArduPilot_Commands.md).

| Tier | What it shows | Status |
| --- | --- | --- |
| 1 | The autopilot flies the C172 (a rectangle test mission) | done |
| 2 | One known-solution plan (the stadium TFR detour) flies collision-free | done: 4.2 m RMS from the plan, 0.26% longer than the analytic optimum |
| 3 | Plans for generated environments fly, as an unattended batch | done: 6 of 7 solved plans pass (results below) |

## Files

| File | Purpose |
| --- | --- |
| `ardupilot_mission.py` | Builds missions from SCP plans (`make`, `batch-make`), checks they can be flown as waypoints, scores flights (`analyze`, `batch-analyze`, `wobble`), archives every run |
| `run_batch.py` | Flies a whole batch in SITL unattended (runs in WSL; needs only pymavlink) |
| `c172_sitl.parm` | Autopilot parameters: **tracking setup v2** (tuning history in its comments) |
| `autotune_inputs.py` | Runs ArduPilot AUTOTUNE in SITL without a transmitter |
| `setup_c172_sitl.sh` | One-time SITL setup for the JSBSim C172 |
| `test_ardupilot_mission.py` | Tests (`python -m ardupilot.test_ardupilot_mission`) |

Run outputs go to `localstore/` (not committed): each batch in `localstore/batches/<name>/` with `results.csv`, `summary.png`, and per flight `runs/<id>/` (log, parameters flown, mission, `check.png`).

## Tier 3 results

**Six of the seven SCP plans that solved were flown by ArduPilot within every pass criterion; the seventh, the most aggressive dense plan, brushed an obstacle.** SCP solved 7 of 10 generated environments; the other 3 stalled.

- **Closed-loop pass rate:** 6/7 solved plans (86%) with tracking setup v2, up from 5/7 with the untuned setup.
- **The one failure is a tracking limit, not a bad plan.** The project's JSBSim tracker (`sim/jsbsim_c172.py track`), which follows the plan in time, flies the same plan within 10 m. ArduPilot's waypoint mode lags it by about 30 m in altitude on a steep descent.
- **Altitude is the dominant remaining error on 6 of 7 plans.** Tuning removed most of the sideways error; the altitude lag needs a different interface (GUIDED-mode streaming), not more tuning.
- **The pipeline runs unattended:** 7 flights in about 25 minutes.

### Setup

1. **Environments:** 10 C172-scale workspaces (12 km x 2.5 km x 600 m, 6 m vehicle radius, 30 m margin) from `environment/envgen.py`: 3 easy (2-4% obstacle occupancy), 4 nominal (5-8%), 3 dense (9-12%).
2. **Plans:** SCP (`solvers/scp_aircraft.py`) on each environment.
3. **Missions:** `batch-make` turns each solved plan into a waypoint mission (legs within 1 m of the plan) and rejects plans that can't be flown as waypoints: the expected autopilot path, including its corner-cutting turns, must keep at least 15 m (half the margin) from obstacles, and no corner may need more than 45 deg of bank.
4. **Flights:** `run_batch.py` starts SITL at each mission's home, uploads, arms in AUTO, records the log, and stops after the final loiter.
5. **Scoring:** `batch-analyze` compares each flight with its plan from the plan's start to its goal.

A flight passes only if all of these hold (the `PASS_*` settings in `ardupilot_mission.py`):

| Check | Pass if |
| --- | --- |
| Collision | never touches an obstacle; stays in the workspace |
| Max distance from the plan | < 30 m (the planner's margin) |
| RMS distance from the plan | <= 10 m |
| Goal | passes within 50 m of the plan's goal |
| Flight time | within +-10% of the plan's |

A flown clearance under 15 m is flagged as a near miss but not failed.

### Tracking setup v2

Each candidate change was tested on the same plans and kept only if it measurably helped. All planners' results go through this identical setup.

| Change | Evidence | Kept |
| --- | --- | --- |
| Pitch and roll rate gains from ArduPilot AUTOTUNE | Defaults (tuned for a small RC plane) followed commands ~1.3 s late; straight-flight pitch weave 1.65 -> 0.73 deg, roll 1.15 -> 0.26 deg, altitude 1.25 -> 0.24 m rms | yes |
| `TECS_TIME_CONST` 5 -> 8 s | Throttle stopped swinging 58-100% every ~5 s | yes |
| Waypoint legs within 1 m of the plan (was 3 m) | Roll pulses on curves roughly halved | yes |
| `NAVL1_PERIOD` 20 -> 15 s | Sideways error 40-45% lower on the curvy plans; nominal_2 went from fail to pass | yes |
| Airspeed following the plan, per waypoint leg (`--speed-profile plan`) | ArduPilot followed the speeds 5-10 s late, throttle swung 0-100%; dense_1 RMS 14.1 -> 18.3 m | no |
| `TECS_CLMB_MAX` 5 -> 6, `TECS_SINK_MAX` 6 -> 7 m/s | No change on two plans; dense_1 slightly worse (altitude max 30.0 -> 33.0 m) | no |

SITL is repeatable: the same plan flown twice with the same setup differed by under 0.5 m in every metric, so single-flight comparisons are meaningful.

### Per-environment results (batch `tier3_v2`)

| Environment | Outcome | RMS from plan (m) | Max sideways (m) | Max altitude (m) | Closest to obstacle (m) | Goal miss (m) |
| --- | --- | --- | --- | --- | --- | --- |
| easy_0 | PASS | 1.7 | 2.4 | 6.0 | 28.5 | 3.4 |
| easy_1 | PASS | 5.7 | 8.0 | 20.7 | 21.4 | 8.3 |
| easy_2 | PASS | 0.7 | 2.2 | 2.1 | 26.5 | 3.0 |
| nominal_0 | NOT SOLVED (SCP stalled) | | | | | |
| nominal_1 | NOT SOLVED (SCP stalled) | | | | | |
| nominal_2 | PASS (failed with v1) | 8.5 | 12.2 | 19.9 | 17.3 | 9.2 |
| nominal_3 | PASS | 6.1 | 2.9 | 12.6 | 18.3 | 13.1 |
| dense_0 | NOT SOLVED (SCP stalled) | | | | | |
| dense_1 | FAIL, near miss | 14.0 | 23.4 | 30.3 | -0.6 | 20.5 |
| dense_2 | PASS, near miss | 7.0 | 6.7 | 26.5 | 13.3 | 27.3 |

By difficulty: easy 3/3 pass, nominal 2/2 of the solved plans, dense 1/2 of the solved plans. Every flight took within 1-3% of the plan's flight time. Compared with the untuned setup (v1), RMS error fell on every plan; the largest gains were dense_1 (19.2 -> 14.0 m) and nominal_2 (11.2 -> 8.5 m).

### Failure analysis: dense_1

The remaining error is altitude lag in ArduPilot's waypoint mode, not an unflyable plan. The JSBSim tracker flies each plan in time, using the plan's own bank and climb as feed-forward, and stays well inside the criteria on dense_1:

| dense_1 | ArduPilot, waypoints (v2) | JSBSim tracker, follows the plan in time |
| --- | --- | --- |
| RMS from plan | 14.0 m | 5.4 m |
| Max from plan | 30.5 m | 9.7 m |
| Closest to obstacle | -0.6 m | 24.2 m |

The JSBSim tracker's error is measured against the plan's position at each moment, a stricter measure than ArduPilot's distance from the path; its throttle never saturated on dense_1.

The failure happens where dense_1 descends at about 6 m/s while slowing from 57 to 44 m/s. Waypoints pass only the path's shape, so ArduPilot's altitude controller (TECS) reacts to the error after it opens instead of anticipating the descent. Raising ArduPilot's climb and sink limits did not help, which rules out a rate cap; passing the plan's speeds as waypoint commands made it worse.

### Next steps

1. **GUIDED-mode streaming:** send ArduPilot the plan's position and velocity in time instead of waypoints, so it anticipates climbs and turns as the JSBSim tracker does.
2. **A larger batch (30-50 environments)** on tracking setup v2, for a pass rate with a usable confidence interval; 7 flights gives only a rough indication.
3. **Planner headroom (team decision):** whether plans should leave margin in bank, climb rate and clearance for tracking error. Several plans already sit 23-25 m from obstacles instead of 30 m.
4. **The 3 unsolved environments** are kept as SCP failure cases for the method comparison.
