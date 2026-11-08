# ArduPilot SITL flight validation

This folder flies SCP plans through ArduPilot (ArduPlane SITL with the JSBSim Cessna 172), and scores each flight against its plan. It answers one question for the benchmark: **can a plan that is feasible on paper actually be flown, and when it can't, is that the plan's fault or the autopilot's?**

Step-by-step commands for every tier are in [docs/ArduPilot_Commands.md](../docs/ArduPilot_Commands.md).

| Tier | What it shows |
| --- | --- | 
| 1 | The autopilot flies the C172 (a rectangle test mission) |
| 2 | One known-solution plan (the stadium TFR detour) flies collision-free |
| 3 | Plans for generated environments fly, as an unattended batch |

## Files

| File | Purpose |
| --- | --- |
| `ardupilot_mission.py` | Builds missions from SCP plans (`make`, `batch-make`), checks they can be flown as waypoints, scores flights (`analyze`, `batch-analyze`, `wobble`), archives every run |
| `run_batch.py` | Flies a whole batch in SITL unattended (runs in WSL; needs only pymavlink); for GUIDED batches it also runs `guided_follower.py` during the flight |
| `c172_sitl.parm` | Autopilot parameters: the tuned gains of tracking setup v3, unchanged in v4 (tuning history in its comments) |
| `autotune_inputs.py` | Runs ArduPilot AUTOTUNE in SITL without a transmitter |
| `wobble_sweep.py` | Tries several autopilot settings in one SITL flight (straight and level in CRUISE) to measure the pitch/roll wobble |
| `fix_sweep.py`, `fly_fix_sweep.sh` | Build, fly and compare several candidate setups on the same plans (the v3 tuning experiments) |
| `speed_lead_test.py`, `fly_speed_lead.sh` | Build, fly and compare dense_1 with the plan's speeds commanded 0, 4 or 8 s early (the v4 experiment) |
| `guided_follower.py` | Tracking setup v5: the "companion computer" that streams the plan to ArduPlane in GUIDED mode (roll, altitude, airspeed), used by `run_batch.py` for `--follower guided` batches |
| `guided_test.py`, `fly_guided_test.sh` | Build, fly and compare GUIDED follower settings on chosen plans (the v5 experiments) |
| `setup_c172_sitl.sh` | One-time SITL setup for the JSBSim C172 |
| `test_ardupilot_mission.py` | Tests (`python -m ardupilot.test_ardupilot_mission`) |

What every Python file does, function by function: [CODE_GUIDE.md](CODE_GUIDE.md).

Run outputs go to `localstore/` (not committed): each batch in `localstore/batches/<name>/` with `results.csv`, `summary.png`, and per flight `runs/<id>/` (log, parameters flown, mission, `check.png`).

## Tier 3 results

A series of environments were generated to be flown in ArduPilot with varying degrees of  density: 3 "easy", 4 "nominal", and 3 "dense" environments.

**All seven SCP plans that solved were flown by ArduPilot within every pass criterion. With the plan streamed in GUIDED mode (tracking setup v5) every one stays within 8.1 m of its plan, and within 1.8 m sideways.** SCP solved 7 of 10 generated environments; the other 3 stalled.

- **Closed-loop pass rate:** 7/7 solved plans (100%) with tracking setups v3, v4 and v5, up from 6/7 with v2 and 5/7 with the untuned setup.
- **Errors:** averaged over the 7 plans, max distance from the plan 17.2 m (v2) -> 9.8 m (v3) -> 6.5 m (v4) -> 3.5 m (v5), and rms 6.3 -> 2.2 -> 1.9 -> 1.05 m. The worst flight's max error is 8.1 m (v4: 9.9 m, v3: 24.8 m).
- **No near misses:** the closest approach to an obstacle on dense_1 went from -0.6 m (a collision, v2) to +5.5 m (v3), 20.5 m (v4) and 23.3 m (v5); the plan itself keeps 23.7 m.
- **Two ways to hand ArduPilot a plan:** as a waypoint mission (v4, the default: a waypoint every 30 m, shifted by a fitted tracking correction, with the plan's airspeed per leg), or streamed in GUIDED mode by a "companion computer" (v5, `--follower guided`: no waypoints, no fitted correction; see [Tracking setup v5](#tracking-setup-v5-guided-follower)).
- **The roll follows the plan's bank** instead of jumping between 0 and 20-40 deg at every waypoint, and the steady pitch wobble is down by about 60% (see [Tracking setup v3](#tracking-setup-v3)).
- **Remaining caveat: the throttle.** Following the plan's airspeed makes the throttle work harder: with v4 it moves 1.3-3x more than with one constant airspeed, with v5 another 73% more on average (though v5 sits at 0% or 100% only half as often as v4).
- **Repeatable and unattended:** the same plan flown twice differs by under 0.5 m in every metric (v2 and v5 both checked); 7 flights take 25-35 minutes.

![Max error, RMS error and closest approach for each plan with tracking setups v1 to v5](figures/tier3_progress.png)

*Each solved plan flown with the untuned setup (v1) and tracking setups v2 to v5, against the pass limits: max distance from the plan (fail above 30 m), RMS distance (fail above 10 m) and closest approach to an obstacle (collision below 0, near miss below 15 m). Failures are marked where the limit was crossed. Batches `tier3_pilot`, `tier3_v2`, `tier3_v3b`, `tier3_v4`, `tier3_v5`.*

### Setup

1. **Environments:** 10 C172-scale workspaces (12 km x 2.5 km x 600 m, 6 m vehicle radius, 30 m margin) from `environment/envgen.py`: 3 easy (2-4% obstacle occupancy), 4 nominal (5-8%), 3 dense (9-12%).
2. **Plans:** SCP (`solvers/scp_aircraft.py`) on each environment.
3. **Missions:** `batch-make` turns each solved plan into a waypoint mission and rejects plans that can't be flown as waypoints: the expected autopilot path must keep at least 15 m (half the margin) from obstacles, and no corner may need more than 45 deg of bank. With v4 (the default) the mission has a waypoint every 30 m along the plan, shifted by a measured tracking correction (see [Tracking setup v3](#tracking-setup-v3)), and a speed command wherever the plan's airspeed changes by 0.5 m/s (see [Tracking setup v4](#tracking-setup-v4)); v1 and v2 used fewer waypoints, with legs within 1 m of the plan, and v1-v3 one constant airspeed.
4. **Flights:** `run_batch.py` starts SITL at each mission's home, uploads, arms in AUTO, records the log, and stops after the final loiter. With `--follower guided` (v5) the mission holds only the takeoff, climb, lead-in, exit and final loiter; near the plan's start `guided_follower.py` switches the aircraft to GUIDED and streams the plan, and at the goal hands it back to AUTO.
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

### Tracking setup v3

v2 left three problems: on curves the roll jumped between 0 and 20-40 deg at every waypoint, a steady ~2.2 s pitch/roll wobble remained even in straight flight, and dense_1 lagged ~30 m in altitude on its steep climb. v3 fixes the first two and most of the third. Each change was tested on the same plans first (batches `wp5`, `fixA`-`fixD`, `fixS1`-`fixS4`, `v3_dense*`, and the in-flight sweeps `wobble_sweep1`, `l6_sweep1`); the full v3 batch is `tier3_v3b`.

| Change | Evidence | Kept |
| --- | --- | --- |
| Pitch gains from AUTOTUNE at level 6 (P 2.00, I 1.50, D 0.19, FF 1.26), keeping the 0.25 s pitch time constant | With the old gains pitch moved ~2x more than TECS commanded. Straight flight: pitch wobble 0.77 -> 0.31 deg rms, roll 0.26 -> 0.13 deg. (AUTOTUNE's own 0.75 s time constant made it worse: TECS cycled at 7.4 s, +-4 deg) | yes |
| `TECS_SPDWEIGHT` 1 -> 0, `TECS_PTCH_DAMP` 0.3 -> 0.1 | Pitch holds height only, so on steep climbs the aircraft trades speed for height like the plan does, instead of pinning the throttle at 100%. dense_1 max altitude error 30.3 -> 19.1 m | yes |
| **Dense waypoints: one every 30 m** along the plan (the RK4 / first-order-hold trajectory resampled at small, equal steps; a teammate's suggestion) | Each waypoint becomes a 1-2 deg step instead of a 10-25 deg corner, so ArduPilot's roll follows the plan's bank: 2.7 deg rms from it on dense_1 (plain waypoints: 10.9 deg), ripple 0.6 deg rms | yes |
| **Tracking correction** (`--track-comp both`): each waypoint shifted towards the curve's centre by 16 950 / radius m, and its altitude ahead of the climb | With dense waypoints ArduPlane flew a steady offset: outside every curve (24 m on 300-700 m radii) and behind the climbs. A two-term fit explained it to 1.3 m (sideways) and 2.3 m (altitude). On dense_1: rms 13.7 -> 5.3 m, goal miss 25 -> 2.6 m | yes |
| Fewer waypoints (5 per mission; or 300 m spacing) | 5 waypoints: the wobble stayed on long straight legs, and tracking got far worse (dense_1 max 268 m). Wider spacing helped altitude but cut corners | no |
| Waypoint altitudes taken 5 s ahead (`--alt-lead`) | Shifted waypoints sat ~30 m off the plan, so the flyability check rejected the nominal/dense missions; easy_0 got worse | no |
| Per-leg airspeed (again, with v2's waypoints) | Speeds arrived ~10 s late; dense_1 vertical error 30 -> 37 m. (With v3's dense waypoints it worked: see v4) | no |
| Curves as loiter arcs (`--arcs`, fractional `NAV_LOITER_TURNS`) | Steady bank in long curves and dense_1 passed (16.8 m max), but roll still spiked between arcs (4.9 deg rms from the plan's bank). Kept as an option | option |
| Whole plan as a chain of short loiter arcs | Two of 53 arcs looped a full circle (ArduPlane's loiter exit rule); every arc added a jolt. Removed | no |
| Other wobble candidates: `PTCH2SRV_TCONST` 0.5, `TECS_PTCH_DAMP` 0 or 0.1 alone, `TECS_INTEG_GAIN`, `TECS_HGT_OMEGA`, `PTCH_RATE_D` alone, throttle locked, real-time (1x) SITL | The wobble's period changed but its size did not (0.7-0.9 deg). Ruled out waypoints, speed commands, the throttle loop and SITL speed as causes | no |

![dense_1 roll against the bank the plan needs, for v2, the arcs experiment and v3](figures/tier3_roll_vs_plan_bank.png)

*dense_1: the roll ArduPilot flew (blue) against the bank the SCP plan needs (dashed), along the plan. With v2's waypoints a few hundred metres apart, the roll jumps at every waypoint (7.4 deg rms from the plan's bank). Flying the curves as loiter arcs gives steady bank inside each curve but spikes between them (4.9 deg). With v3, a waypoint every 30 m plus the tracking correction, the roll follows the plan's bank (2.7 deg).*

The correction's constants were fitted at ~52 m/s with these autopilot settings (`NAVL1_PERIOD` 15, `TECS_TIME_CONST` 8); refit them (`TRACK_COMP_*` in `ardupilot_mission.py`) if those change.

### Tracking setup v4

v3 left one large error: the steep climb on dense_1 (24 m behind the plan, a near miss at 5.5 m). There the plan speeds up to ~58 m/s and then climbs while slowing to 44 m/s, trading speed for height; at v3's one constant airspeed (~52 m/s) the aircraft had no speed to trade, and the throttle hit 100%. v4 adds one change and no parameter changes; the full batch is `tier3_v4`.

| Change | Evidence | Kept |
| --- | --- | --- |
| **The plan's airspeed on every leg** (`BATCH_SPEED_PROFILE = "plan"`: a `DO_CHANGE_SPEED` wherever the plan's speed changes by 0.5 m/s) | With a waypoint every 30 m the commands are nearly continuous and ArduPilot follows them within 1-2 s (with v2's legs, a few hundred metres long, they arrived 5-10 s late and tracking got worse). All 7 plans: worst max error 24.8 -> 9.9 m, mean max 9.8 -> 6.5 m, mean rms 2.2 -> 1.9 m, airspeed 1.6 -> 0.9 m/s rms from the plan. dense_1: max altitude error 23.9 -> 7.4 m, clearance 5.5 -> 20.5 m. Slightly worse on nominal_2 (max sideways 6.6 -> 9.9 m: the tracking correction was fitted at one speed) and nominal_3 (max altitude 1.8 -> 3.7 m) | yes |
| The same, commanded 4 or 8 s early (`--speed-lead`, `speed_lead_test.py`) | dense_1 max error 9.5 m with no lead, 15.5 m with 4 s, 14.4 m with 8 s: the aircraft slowed down before the top of the climb | no |

The cost is a busier throttle: each speed command is a step that TECS answers with a throttle jump. Over each flight the throttle travels 1.3-3x as far as in v3 (about 1000 %/min on nominal_2 and dense_1, against ~330 %/min), and on nominal_2, easy_1 and dense_2 it is at 0% or 100% for 8-11% of the flight (v3: 0-3%). Tracking does not suffer in SITL, but a real engine would; smoothing the speed commands is the next thing to try. `--speed-profile constant` gives v3's single airspeed.

### Per-environment results (batch `tier3_v4`, waypoints)

| Environment | Outcome | RMS from plan (m) | Max sideways (m) | Max altitude (m) | Closest to obstacle (m) | Goal miss (m) |
| --- | --- | --- | --- | --- | --- | --- |
| easy_0 | PASS | 0.6 | 2.4 | 1.4 | 27.5 | 2.2 |
| easy_1 | PASS | 2.0 | 6.7 | 8.8 | 25.1 | 3.5 |
| easy_2 | PASS | 0.4 | 1.4 | 0.8 | 28.4 | 1.5 |
| nominal_0 | NOT SOLVED (SCP stalled) | | | | | |
| nominal_1 | NOT SOLVED (SCP stalled) | | | | | |
| nominal_2 | PASS | 2.4 | 9.9 | 2.8 | 22.8 | 1.9 |
| nominal_3 | PASS | 1.0 | 1.1 | 3.7 | 25.4 | 4.1 |
| dense_0 | NOT SOLVED (SCP stalled) | | | | | |
| dense_1 | PASS (failed with v1 and v2; near miss with v3) | 4.2 | 9.5 | 7.4 | 20.5 | 2.9 |
| dense_2 | PASS | 2.8 | 4.8 | 8.9 | 24.6 | 9.3 |

By difficulty: easy 3/3 pass, nominal 2/2 and dense 2/2 of the solved plans; no near misses. Every flight took within 1-3% of the plan's flight time.

![Max distance from the plan by direction for each flown plan](figures/tier3_max_error_by_direction.png)

*Worst sideways and altitude deviation from the plan for each flown plan, with the 30 m fail limit. Every plan stays within 10 m. Batch `tier3_v4`, 6 Nov 2026.*

**A clean flight, easy_2** (RMS 0.4 m from the plan). Panels: top view, altitude, tracking error, roll, pitch, and throttle with airspeed.

![easy_2 flight against its plan](figures/tier3_easy_2_flight.png)

**The hardest plan, dense_1** (RMS 4.2 m; failed with v1 and v2, a near miss with v3). The roll follows the plan's weaving turns without jumps at the waypoints, the airspeed follows the plan's speed-up before the climb, and the steep climb over the bump at about 6-7 km is now flown within a few metres. The jagged throttle trace after ~80 s is the cost of the speed commands.

![dense_1 flight against its plan](figures/tier3_dense_1_flight.png)

*dense_1 on tracking setup v4. Batch `tier3_v4`, 6 Nov 2026.*

**dense_2** (RMS 2.8 m). The flight stays within about 2 m of the plan for the first 130 s; the largest errors are a brief 9 m on the climb at ~135 s and the last seconds, where the plan ends in a steep climb that the aircraft follows a little late (9.3 m at the goal; 14.9 m with v3).

![dense_2 flight against its plan](figures/tier3_dense_2_flight.png)

*dense_2 on tracking setup v4. Batch `tier3_v4`, 6 Nov 2026.*

### Tracking setup v5 (GUIDED follower)

v4 hands ArduPilot the plan as ~370 waypoints. Two limits remained: ArduPilot's path follower aims along straight legs, so it flies outside every curve and needs a fitted correction (calibrated at one speed: with v4's varying speed, nominal_2's sideways error rose to 9.9 m); and the speed changes arrive as 0.5 m/s steps. v5 hands over the plan differently, the way autonomy is usually layered on a flight controller: a **companion computer** (here `guided_follower.py`, run by `run_batch.py`) streams setpoints while ArduPlane flies in **GUIDED** mode. Takeoff, climb, lead-in, exit and final loiter stay in AUTO.

20 times per simulated second the follower:

1. finds the aircraft's place on the plan (distance along the path, searched in a moving window so it never jumps ahead or back);
2. **roll:** the L1 guidance law aimed at the plan point 150 m further along the *curved* path, `a = 2 V^2 sin(eta) / L`, roll = `atan(a / g)`, sent with `SET_ATTITUDE_TARGET` (roll only; pitch and throttle stay with TECS). On a curve of radius R this asks for exactly `V^2 / R`, the bank the turn needs, so there is no steady outward offset and **no fitted correction**;
3. **altitude:** the plan's altitude 30 m ahead, plus the climb lead of the v3 correction's vertical part, with `MAV_CMD_GUIDED_CHANGE_ALTITUDE`;
4. **airspeed:** the plan's airspeed, with `MAV_CMD_GUIDED_CHANGE_SPEED`, changed at once in small continuous steps.

If the stream stops, ArduPlane drops the forced roll after `GUIDED_TIMEOUT` (3 s) and circles the point where GUIDED began. The settings are in `ardupilot_mission.py` (`GUIDED_*`) and `guided_follower.py`; the experiments are in `guided_test.py` (batches `g100`-`g200`, `h0`-`h2`, `k0`-`k2`, `tier3_guided`, `tier3_guided_clmb`).

| Round / change | Evidence | Kept |
| --- | --- | --- |
| **Round 1: the follower as above, lookahead 100, 150, 200 m** (easy_0, nominal_2, dense_1) | All pass; max sideways error 9.5-9.9 -> 1.0-1.5 m on the curvy plans with no correction; dense_1 rms 4.2 -> 1.7 m. Altitude no better than v4. 150 m kept (100 m similar; 200 m cuts a little more inside curves) | yes |
| Round 2: each speed change ramped at the plan's acceleration | ArduPlane restarts a speed ramp at every command, so with several commands a second the target barely moved: airspeed flat at ~51 m/s, dense_1's climb 17 m behind | no |
| Round 3: altitude sent with a 100 000 m/s climb rate, only when it changes by 0.5 m | ArduPlane moves the altitude target at most "rate x time since the last update" from the current altitude; with its default (1000 m/s) commands arriving a few ms before an update snapped the target back, and TECS answered with a throttle bump about once a second. The bumps halved | yes |
| Round 3: altitude commanded 0, 30 or 60 m ahead | dense_1 max altitude error 8.6 / 5.8 / 7.7 m (60 m sits above the plan on long descents) | 30 m |
| All 7 plans (batch `tier3_guided`) | Better than v4 on 5 of 7; worse on the long climbs that end dense_2 and nominal_2 (13.1 and 9.6 m behind at the goal) | - |
| **`TECS_CLMB_MAX` 5.0 -> 6.5 m/s, GUIDED batches only** (`GUIDED_PARAM_OVERRIDES`) | The plan's 6 deg climbs at 47-57 m/s need 5.1-5.9 m/s, above the 5.0 m/s cap. dense_2 13.1 -> 5.0 m, dense_1 5.9 -> 4.6 m. (In v2, with larger tracking errors, raising the cap changed nothing.) Not in `c172_sitl.parm`, so v1-v4 keep their results | yes |

**Results (batch `tier3_v5`, all 7 plans), against v4:**

| Environment | v4 max (m) | v5 max (m) | v4 RMS | v5 RMS | v4 max sideways | v5 max sideways | v4 max altitude | v5 max altitude | v5 closest to obstacle (plan) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| easy_0 | 2.4 | 2.3 | 0.62 | 0.49 | 2.4 | 0.5 | 1.4 | 2.3 | 26.7 (27.2) |
| easy_1 | 8.9 | 2.2 | 2.03 | 0.76 | 6.7 | 1.0 | 8.8 | 2.2 | 26.3 (27.7) |
| easy_2 | 1.5 | 0.7 | 0.36 | 0.32 | 1.5 | 0.3 | 0.9 | 0.7 | 27.9 (27.2) |
| nominal_2 | 9.9 | 8.1 | 2.35 | 1.62 | 9.9 | 1.0 | 2.8 | 8.1 | 23.4 (23.5) |
| nominal_3 | 3.8 | 2.1 | 1.05 | 1.02 | 1.2 | 0.4 | 3.7 | 2.1 | 26.3 (27.2) |
| dense_1 | 9.5 | 4.5 | 4.23 | 1.80 | 9.5 | 1.8 | 7.4 | 4.4 | 23.3 (23.7) |
| dense_2 | 9.3 | 4.6 | 2.77 | 1.31 | 4.8 | 1.6 | 8.9 | 4.6 | 24.8 (25.6) |
| **Mean** | **6.5** | **3.5** | **1.92** | **1.05** | **5.1** | **0.9** | **4.8** | **3.5** | |

All 7 pass, no near misses; mean goal miss 3.6 -> 2.3 m; every flight within 1-3% of the plan's time. The airspeed follows the plan a little less closely than in v4 (1.3 against 0.9 m/s rms). The same plan flown twice (dense_1, batch `tier3_v5_repeat`) agreed to under 0.5 m in every metric (max 4.45 / 4.92 m, rms 1.80 / 1.85 m, clearance 23.25 / 22.91 m).

![Max distance from the plan by direction for each plan, tracking setup v5](figures/tier3_v5_max_error_by_direction.png)

*Worst sideways and altitude deviation from the plan for each flown plan with tracking setup v5. Sideways error stays under 1.8 m everywhere; what remains is altitude on the steepest climbs. Batch `tier3_v5`, 7 Nov 2026.*

**dense_1 with v5** (max 4.5 m, rms 1.8 m; v4: 9.5 and 4.2 m). The roll follows the follower's command closely, the climb over the bump at about 6-7 km is flown within a few metres, and the closest approach (23.3 m) is within 0.4 m of the plan's own.

![dense_1 flight against its plan, tracking setup v5](figures/tier3_v5_dense_1_flight.png)

**nominal_2 with v5**: the largest remaining error. The flight stays within 3.7 m of the plan until its last ~8 s, then falls 8.1 m behind at the goal: there the plan climbs steeply *and* speeds up to 57 m/s at once, the throttle is at 100% and the airspeed drops to ~50 m/s instead. The C172 does not have the power for both, so this is a limit of the plan, not of the follower (an energy check in the flyability check would flag it).

![nominal_2 flight against its plan, tracking setup v5](figures/tier3_v5_nominal_2_flight.png)

**The throttle is v5's main cost.** Throttle travel averages 1195 %/min over the 7 plans against 690 with v4 (and ~330 with one constant airspeed), though v5 sits at 0% or 100% only 2.9% of the time against 5.5% with v4. Some of it is real (the plan's speed changes, harder climbs with the higher climb cap); smaller bumps about once a second in steady flight remain unexplained. Not yet fixed; v4 stays the default (`batch-make` without `--follower`) until it is.

### Per-environment results (batch `tier3_v3b`, for comparison)

| Environment | Outcome | RMS from plan (m) | Max sideways (m) | Max altitude (m) | Closest to obstacle (m) | Goal miss (m) |
| --- | --- | --- | --- | --- | --- | --- |
| easy_0 | PASS | 0.6 | 2.6 | 1.2 | 27.0 | 0.6 |
| easy_1 | PASS | 3.2 | 6.6 | 17.4 | 24.3 | 1.4 |
| easy_2 | PASS | 0.3 | 0.4 | 0.7 | 27.4 | 2.4 |
| nominal_0 | NOT SOLVED (SCP stalled) | | | | | |
| nominal_1 | NOT SOLVED (SCP stalled) | | | | | |
| nominal_2 | PASS | 2.1 | 6.6 | 2.5 | 22.6 | 2.7 |
| nominal_3 | PASS | 1.0 | 1.1 | 1.8 | 25.7 | 0.9 |
| dense_0 | NOT SOLVED (SCP stalled) | | | | | |
| dense_1 | PASS, near miss (failed with v1 and v2) | 5.7 | 8.0 | 23.9 | 5.5 | 3.1 |
| dense_2 | PASS | 2.4 | 4.4 | 14.8 | 24.3 | 14.9 |

By difficulty: easy 3/3 pass, nominal 2/2 and dense 2/2 of the solved plans. Every flight took within 1-3% of the plan's flight time. RMS error fell on every plan compared with v2, most on dense_1 (14.0 -> 5.7 m), nominal_3 (6.1 -> 1.0 m) and nominal_2 (8.5 -> 2.1 m). (The first v3 batch, `tier3_v3`, also passed 7/7; its correction still had a constant +2.4 m altitude term, which made the aircraft fly ~2 m high on the straight plans. Removing it gave `tier3_v3b`.)

### Per-environment results (batch `tier3_v2`, for comparison)

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

### Failure analysis: dense_1 (tracking setup v2)

*With v3, dense_1 passes (rms 5.7 m, max 24.8 m), and with v4 it is no longer a near miss (rms 4.2 m, max 9.5 m; see its flight above). This section explains the v2 failure.*

The remaining error is altitude lag in ArduPilot's waypoint mode, not an unflyable plan. The JSBSim tracker flies each plan in time, using the plan's own bank and climb as feed-forward, and stays well inside the criteria on dense_1:

| dense_1 | ArduPilot, waypoints (v2) | JSBSim tracker, follows the plan in time |
| --- | --- | --- |
| RMS from plan | 14.0 m | 5.4 m |
| Max from plan | 30.5 m | 9.7 m |
| Closest to obstacle | -0.6 m | 24.2 m |

The JSBSim tracker's error is measured against the plan's position at each moment, a stricter measure than ArduPilot's distance from the path; its throttle never saturated on dense_1.

The failure happens where dense_1 descends at about 6 m/s while slowing from 57 to 44 m/s. Waypoints pass only the path's shape, so ArduPilot's altitude controller (TECS) reacts to the error after it opens instead of anticipating the descent. Raising ArduPilot's climb and sink limits did not help, which rules out a rate cap; passing the plan's speeds as waypoint commands made it worse with v2's long legs (with v3's dense waypoints it fixed the climb: tracking setup v4).

On tracking setup v2 the dense_1 flight stayed above the plan through the descent at about 6-8 km, and the tracking error peaked at about 30 m near the end, where the aircraft brushed an obstacle (batch `tier3_v2`, 30 Oct 2026; its flight chart is `localstore/batches/tier3_v2/runs/tier3_dense_envs_1/check.png`).

