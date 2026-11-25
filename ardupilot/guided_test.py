#!/usr/bin/env python3
"""
guided_test.py - Tests of the GUIDED follower (guided_follower.py): the plan streamed from a
"companion computer" in GUIDED mode instead of uploaded as waypoints, compared with tracking setup v4
(batch tier3_v4, waypoints).

Round 1 (2026-11-07; batches g100, g150, g200 = lookahead 100/150/200 m, on easy_0, nominal_2, dense_1):
all pass; max sideways error 9.5-9.9 -> 1.0-1.5 m on the curvy plans with no tracking correction,
dense_1 rms 4.2 -> 1.7 m; altitude no better (it now sets the max error); throttle travel +20-40%,
single spikes at each speed command (sent as "change at once"). Lookahead 150 m kept.

Round 2 (batches h0, h1, h2; dense_1): speed changes ramped, altitude lead 0/30/60 m. Worse: ArduPlane
restarts a speed ramp at every command, so the target barely moved (airspeed flat at ~51 m/s, the climb
17 m behind without a lead); throttle bumps about once a second came from the ALTITUDE commands
(clipped to "rate x ms since the last update" around the current altitude), not from the speed.

Round 3 (batches k0, k1, k2; dense_1): speed changed at once again; altitude sent with a 100000 m/s
rate (no clipping) and only when it changes by 0.5 m; altitude lead none / 30 / 60 m. 30 m kept; with
TECS_CLMB_MAX 6.5 this became tracking setup v5 (batches tier3_v5, tier3_v5_repeat).

Round 4 (batches r4a, r4b, r4s, r4c; dense_1): where do v5's throttle bumps and pitch wobble come
from? Each run changes ONE thing from v5:
  r4a  altitude sent every update (step 0 instead of 0.5 m)
  r4b  airspeed held at the plan's mean (no speed changes sent)   - diagnostic only
  r4s  airspeed sent every update (step 0 instead of 0.05 m/s)
  r4c  TECS_PTCH_DAMP 0.1 -> 0.2
Before flying (tier3_v5 logs): throttle jitter 4.6% (v3b, one constant speed: 1.2%; v4: 11.7%), so the
bumps came with following the plan's speed; pitch-command jitter ~0.2 deg in v3b, v4 and v5 alike.
Result (v5 / v5 repeat: travel 1598 / 1730 %/min, jitter 4.55 / 4.63 %, pitch jitter 0.22 / 0.21 deg):
  r4a  no change (1625, 4.67, 0.23): the altitude steps are not the cause
  r4b  1.11% jitter, 269 %/min - the speed commands ARE the cause - but max error 24.5 m, clearance 5.4 m
  r4s  1108 %/min (-31%), 4.03%, pitch 0.18; tracking as good or better (airspeed 2.08 m/s rms): KEPT
       (SPEED_STEP_MS 0 in guided_follower.py)
  r4c  no change (1241, 4.46, 0.21): dropped
  Pitch jitter ~0.2 deg in every run, even r4a/r4b: not from our commands but TECS's own height loop.

Round 5 (batches r5a-r5d; dense_1): v5 with airspeed sent every update (r4s), plus ONE ArduPilot setting:
  r5a  TECS_THR_DAMP 0.5 -> 0.25     throttle reacts less to the rate of speed change
  r5b  TECS_SPD_OMEGA 2 -> 1         smoother airspeed-rate estimate
  r5c  THR_SLEWRATE 100 -> 30 %/s    caps how fast the throttle moves (treats the symptom)
  r5d  TECS_HGT_OMEGA 3 -> 1.5       smoother height-rate estimate (aimed at the pitch wobble)
Result (travel %/min, throttle jitter %, pitch jitter deg; r4s 1108, 4.03, 0.18):
  r5a  907, 3.84, 0.18 - tracking slightly worse (max 4.79 m)      r5b  891, 4.02, 0.17
  r5c  691, 2.88, 0.19 - tracking unchanged (max 4.45 m, clearance 23.1 m): KEPT (GUIDED_PARAM_OVERRIDES)
  r5d  877, 3.94, 0.17 - pitch unchanged
  r5a/b/d's ~20% are within run-to-run noise (r4s was likely a high sample). Pitch jitter 0.17-0.19 deg
  in every run: TECS's own height loop; accepted. v5 + speed every update + THR_SLEWRATE 30 = tracking
  setup v6 (batches tier3_v6, tier3_v6_repeat).

Round 6 (dense_1; robustness, design-review feedback): tracking setup v6 with a delay, in simulated seconds.
  r6c010 ... r6c200  every roll/altitude/airspeed setpoint reaches ArduPlane 0.1, 0.25, 0.5, 1, 2 s late
                     (link or companion-computer lag; mode changes are not delayed)
  r6p050, r6p100     the follower steers from a position 0.5 or 1 s old (sensor/telemetry lag)
  Question: at what delay does tracking (max error, clearance) start to break down?
  Result (max error / max sideways / max vertical, m; v6 = 4.64/1.65/4.61, repeat 4.21/1.67/4.18):
    0.1 s 3.77/1.86/3.75   0.25 s 2.65/2.27/2.63   0.5 s 3.24/2.79/3.20   pos 0.5 s 3.14/2.81/3.04  all PASS
    1 s   48.1 (FAIL; ~9 s lateral limit cycle, roll at the 40 deg limit 24% of the time), pos 1 s 48.0 (FAIL)
    2 s   257 (FAIL; diverged, collision)
  Tolerates 0.5 s of extra loop delay; command and position delay act the same (total loop delay matters).
  A robustness check only; no compensation added. Side finding: 0.25-0.5 s of delay LOWERED the vertical
  error - at ~54 m/s that cancels 13-27 m of the 30 m altitude lead, so the lead looks too long -> round 7.

Round 7 (dense_1): v6 with a shorter altitude lead (guided_alt_lead; v6 = 30 m): r7a10, r7a15, r7a20.
  Result (max error / max vertical / rms, m): 30 m (v6, repeat) 4.64/4.61/1.96, 4.21/4.18/1.83;
    20 m 3.18/3.15/1.32; 15 m 2.86/2.83/1.13; 10 m 2.61/2.33/1.00. Sideways error, clearance, throttle
    (travel ~700 %/min, 1.7% at limits, jitter 2.8%) unchanged; pitch jitter 0.18 -> 0.20 deg (small).
    10 m KEPT (GUIDED_ALT_LEAD_M) = tracking setup v7 (batch tier3_v7, all 7 plans).

Round 8 (dense_1; design-review feedback "closed-loop velocity"): tracking setup v7 in a steady wind
  (SITL's SIM_WIND_SPD / SIM_WIND_DIR, set before arming, so the whole flight is in it). dense_1 runs
  roughly east (start -> goal bearing 081 deg); SIM_WIND_DIR is the direction the wind comes FROM:
    r8h05, r8h10  headwind  (from 080) 5, 10 m/s
    r8t05, r8t10  tailwind  (from 260) 5, 10 m/s
    r8c05, r8c10  crosswind (from 170) 5, 10 m/s
  The follower sends the plan's AIRSPEED open-loop, so a headwind should leave the aircraft behind the
  plan's schedule and a tailwind ahead of it, while the path itself (max error, which is distance to the
  nearest plan point) should hold. The extra table measures that: flight time against the plan, and the
  schedule lag (time flown minus the plan's time to the same point; > 0 = behind).
  CORRECTION after flying: with the JSBSim C172, SIM_WIND_DIR is where the wind blows TOWARD (the r8h
  flights' groundspeed went UP, 53.8 -> 58.7 / 63.5 m/s). So r8h05/r8h10 were TAILwinds and r8t05/r8t10
  HEADwinds; the batch names are kept. Result (max error / max vertical m, clearance m, flight time vs plan,
  schedule lag at the goal s; no wind: 2.62/2.34, 23.3, -3.5%, -7.5):
    tail 5   11.35/11.28, 18.7, -12% FAIL (time)       tail 10  25.16/25.06, 9.4 (near miss), -18% FAIL (time)
    head 5    3.65/3.60, 22.7,  +6%                     head 10   5.47/5.43, 22.6, +18% FAIL (time)
    cross 5   2.37/2.26, 22.6, -3%                      cross 10  2.87/2.84, 22.5, -2%
  Sideways error stayed under 2.5 m in every wind (L1 steers by ground track). Open-loop airspeed puts the
  aircraft off the plan's schedule by about wind/airspeed (+-39 s over 214 s at 10 m/s). [Corrected in round 10:
  the tailwind's large error is ABOVE the plan in the steep descent, not on climbs.] In a tailwind the
  aircraft also falls BELOW the plan on climbs: the plan's ground-referenced climbs need a steeper climb
  through the air at the higher groundspeed (6 deg at 54 m/s -> ~7 deg at 63.5), past what TECS/the C172 give
  (throttle at its limits 5.9% of the time vs 1.7%). Closed-loop speed (fly the plan's GROUNDspeed) would fix
  both, within the C172's airspeed range.

Round 9 (dense_1): v7 + the speed loop (guided_follower.py speed_ctrl: PI on the schedule error added to
  the plan's airspeed, kp 0.1, ki 0.005, clamped to 38-59 m/s). Winds named by where they blow, and
  compared with the round 8 flight in the same wind:
    r9n      no wind                      (vs tier3_v7)
    r9t05    tailwind 5 m/s  (DIR 080)    (vs r8h05)       r9t10  tailwind 10 m/s (vs r8h10)
    r9h10    headwind 10 m/s (DIR 260)    (vs r8t10)       r9c10  crosswind 10 m/s, DIR 170 (vs r8c10)
  Expect: flight time within ~1% of the plan everywhere except the 10 m/s headwind (needs ~64 m/s of
  airspeed, above the 59 m/s clamp), and the tailwind climbs back on the plan.
  Result (max error / max vertical m, clearance m, flight time vs plan; without the loop in brackets):
    none     22.65/22.53, 14.5 (near miss), 0.0%      [2.62/2.34, 23.3, -3.5%]
    tail 5   41.42/41.33, -0.3 COLLISION, +0.2%       [11.35/11.28, 18.7, -12%]
    tail 10  30.43/30.28, 7.1 (near miss), +0.1%      [25.16/25.06, 9.4, -18%]
    head 10   3.64/3.62, 22.8, +4.0% PASS             [5.47/5.43, 22.6, +18%]
    cross 10 22.83/22.69, 18.7, +0.4%                 [2.87/2.84, 22.5, -2%]
  Timing fixed everywhere, but the path got much WORSE: sideways stays < 2.5 m, the vertical error is all
  in dense_1's steep descent (s ~6300-8400 m), where the aircraft ends up to 50 m ABOVE the plan. There the
  loop asked for 38 m/s (its floor) while descending ~5.5 m/s: at low airspeed that sink rate is a steeper
  path through the air than the C172 can glide at idle while also slowing down (no airbrakes), so TECS
  stays high. In no wind it was set off by the plan's own deceleration at the top of the descent (the
  aircraft lagged it, got 111 m ahead, and the loop piled more slowing on top); in a tailwind the loop
  wants low airspeed all the time. The PI loop as built is NOT kept; speed_ctrl stays off by default.

Round 10 (dense_1): the two safer speed loops (guided_follower.py docstring), same 5 winds as round 9:
  "ff" (wind feedforward from groundspeed - airspeed + a proportional trim of at most 3 m/s) and
  "hybrid" (the same + an integral on the schedule error, inside the same 3 m/s trim), both with altitude
  first and a 42 m/s floor. Batches r10ff_<wind>, r10hy_<wind>; wind = none, t05, t10 (tail), h10 (head),
  c10 (cross). In a 1-D check (dense_1's speeds, 3 s airspeed lag) both keep the schedule within ~1 s
  except where clamped (10 m/s tail: 42 m/s floor, ~7 s early; 10 m/s head: 59 m/s cap, ~5 s late), and
  with a 5% airspeed-sensor error only "hybrid" still did (2.4 s vs 0.2 s off at the goal).
  Result (max error / max vertical m, clearance m, flight time vs plan; ff | hybrid; no loop in brackets):
    none     17.1/17.0, 18.5, -0.4% | 19.6/19.5, 17.4, -0.1%    [2.6/2.3, 23.3, -3.5%]
    tail 5   24.9/24.8,  9.9, -2.1% | 24.7/24.6, 10.2, -1.6%    [11.4/11.3, 18.7, -12%]
    tail 10  20.5/20.4, 11.8, -6.7% | 19.5/19.4, 12.4, -6.7%    [25.2/25.1, 9.4, -18%]
    head 10   3.6/3.5, 22.7, +4.0%  |  3.4/3.3, 23.4, +4.0%     [5.5/5.4, 22.6, +18%]
    cross 10  2.8/2.8, 22.5, -0.2%  |  3.3/3.3, 22.8, +0.1%     [2.9/2.8, 22.5, -2%]
  All PASS (flight time now within 10% everywhere), but tail 5/10 are near misses. The wind estimate is
  right (none +1.4 = the indicated/true airspeed offset; tail 5 +6.2, tail 10 +11.0, head 10 -8.3,
  cross +0.5 m/s), and ff and hybrid fly almost the same (SITL's airspeed sensor has no bias to trim).
  CORRECTION to round 8: its tailwind error was mostly ABOVE the plan in the steep descent
  (s ~8000-8200 m, plan grade -10.5%: +25.3 m at 10 m/s), not below it on climbs (-8.7 m). Every large
  vertical error in rounds 8-10 is in that descent. With a tailwind its ground grade needs 6.6 m/s of
  sink at 63.5 m/s groundspeed, past TECS_SINK_MAX (6 m/s) and steeper through the air than an idle C172
  glides; slowing down (the loops, even by 2-4 m/s in no wind) steepens it further. In v7 (no wind,
  no loop) the aircraft never slows to the plan's 44 m/s there (it stays ~53 m/s) and holds height.

Round 11 (dense_1): is the descent fixable? Two changes, apart and together:
  descent guard (guided_follower.py descent_guard): the hybrid loop never commands below the plan's
    airspeed while the plan within 500 m ahead descends steeper than 5%;
  TECS_SINK_MAX 6 -> 8 m/s: if the demanded sink rate was capped, this lifts the cap (if the C172's idle
    glide is the real limit, it will not help).
    r11g_none, r11g_t05, r11g_t10     hybrid + descent guard: no wind, tail 5, tail 10
    r11gs_t05, r11gs_t10              hybrid + descent guard + TECS_SINK_MAX 8: tail 5, tail 10
    r11s_t10                          no speed loop + TECS_SINK_MAX 8: tail 10 (vs r8h10)
  Result (max error m, highest / lowest vs the plan's altitude m, clearance m, flight time vs plan):
    no wind  hybrid 19.6, +19.8/-3.2, 17.4, -0.1%  ->  + guard 12.8, +1.0/-12.8, 16.8, -0.9%
    tail 5   no loop 11.4 (-12% FAIL); hybrid 24.7, +24.9/-18.4, 10.2, -1.6%
             + guard 13.0, +10.3/-13.0, 16.8, -6.6%;  + guard + sink 8: 12.6, +1.1/-12.6, 17.3, -6.3%
    tail 10  no loop 25.2, +25.3/-8.7, 9.4, -18% FAIL; hybrid 19.5, +19.6/-11.2, 12.4, -6.7%
             + guard 23.7 (-14% FAIL); + guard + sink 8: 11.5, +4.4/-11.5, 18.4, -13% FAIL
             no loop + sink 8: 10.3, +2.8/-10.3, 19.6, -18% FAIL
  TECS_SINK_MAX was the binding limit: with 8 m/s the steep descent is flown within ~3-4 m even in a
  10 m/s tailwind with no loop (+25.3 -> +2.8 m). The descent guard helps with sink 6 but costs the
  timing in a tailwind (dense_1 descends on half its length, where the guard forbids slowing).
  What is left in tailwinds is 10-13 m BELOW the plan on the short steep climb at s ~6400 m (~9% grade:
  at 63.5 m/s groundspeed that needs ~6 m/s of climb, at the C172's limit). Next: the hybrid with
  TECS_SINK_MAX 8 and NO guard (round 12).

Round 12 (dense_1): candidate v8 = v7 + the hybrid speed loop + TECS_SINK_MAX 8, no descent guard, in the
  same 5 winds as round 10 (r12_none, r12_t05, r12_t10, r12_h10, r12_c10; compared with round 10's hybrid,
  sink 6). Expect round 10's timing (within ~7%) with the descent fixed as in round 11. Pass for v8: every
  flight PASS, clearance > 15 m (no near miss), and no wind close to v7.
  Result (max error m, highest above the plan m, clearance m, flight time vs plan; round 10 hybrid with sink 6):
    none     13.4, +13.5, 22.5, +0.1%   [19.6, +19.8, 17.4, -0.1%]    (v7: 2.6, +2.4, 23.3, -3.4%)
    tail 5   19.0, +19.1, 10.9, -1.3%   [24.7, +24.9, 10.2, -1.6%]    near miss
    tail 10  21.4, +21.5, 15.3, -6.0%   [19.5, +19.6, 12.4, -6.7%]
    head 10   3.5,  +3.6, 22.9, +4.0%   [3.4]       cross 10  3.3, +1.7, 22.6, +0.1%   [3.3]
  All PASS, but NOT v8: the descent is still 13-21 m high. TECS_SINK_MAX 8 fixed it only WITHOUT the loop
  (r11s_t10 +2.8 m). The logs show why: every jump above the plan comes right after the "altitude first"
  guard fires (> 3 m high) and steps the airspeed demand UP by 4-11 m/s within ~2 s in the middle of the
  steep descent (r12_t10 at s ~7800: 42 -> 53 m/s, then +12 -> +21 m). Asking for more speed while
  descending at idle sends TECS above the path, so the guard feeds the error it is meant to stop. (v7
  asks for LESS than it flies there, 44 vs 53 m/s, and holds height.)

Round 13 (dense_1): hybrid + TECS_SINK_MAX 8 with the altitude guard OFF and the loop's correction
  rate-limited to 0.5 m/s per s (the plan's own speed profile is not limited), in the 3 winds that failed
  round 12's target: r13_none, r13_t05, r13_t10. v8 if the descent comes back near v7 (a few m) with
  round 12's timing.
  Result (max error m, highest / lowest vs the plan's altitude m, clearance m, flight time vs plan; round 12):
    none     9.9, +10.0/-3.4, 22.7, +0.3%     [13.4, +13.5/-1.6, 22.5, +0.1%]   (v7 2.6, +2.4/-2.2, -3.4%)
    tail 5  20.7, +20.8/-18.7, 10.8, 0.0%     [19.0, +19.1/-18.6, 10.9, -1.3%]  near miss
    tail 10 11.3, +11.4/-11.1, 19.0, -4.1%    [21.4, +21.5/-11.1, 15.3, -6.0%]  (no loop + sink 8: 10.3, -18%)
  Better (tail 10 halved, no wind -26%) but still NOT v8: in still air and tailwinds every speed loop tried
  (rounds 9-13) leaves dense_1 10-21 m off in altitude, against v7's 2.6 m. Slowing to hold the schedule
  in a tailwind costs altitude on both the steep climb (s ~6400, -11 to -19 m) and the steep descent; the
  plan's 6-deg grades with large speed changes leave no energy margin for TECS. Where the loop only has
  to SPEED UP (headwind: 3.5 m, +4% vs 5.5 m, +18% FAIL) it works.

Round 14 (dense_1): round 13's loop made speed-up-only (never below the plan's airspeed): r14_none, r14_t10,
  r14_h10. Expect still air = v7 (the loop never adds speed there), tail 10 = no loop + sink 8 (~10 m,
  ~18% early), head 10 = the loop's fix. If so, v8 = v7 + TECS_SINK_MAX 8 + this loop.
  Result (max error m, highest / lowest vs plan altitude m, clearance m, flight time vs plan): as expected.
    none     2.18, +1.6/-2.2, 23.2, -3.3%   (v7 2.62, -3.4%; the loop added nothing)
    tail 10  9.84, +2.8/-9.8, 20.0, -18% FAIL on time only   (= no loop + sink 8: 10.31, -18%)
    head 10  3.58, +3.6/-3.5, 22.6, +4.1%   (no loop: 5.47, +18% FAIL; loop added up to +11.6 m/s)
  Never worse than v7, fixes headwind lateness. Candidate v8 = v7 + TECS_SINK_MAX 8 + this loop (next:
  all 7 plans, no wind). A tailwind still arrives early: holding the schedule there needs a plan with
  energy margin (or the wind), not the follower.
  -> tracking setup v8 = these defaults in ardupilot_mission.py (GUIDED_SPEED_*, TECS_SINK_MAX 8 in
  GUIDED_PARAM_OVERRIDES); batch tier3_v8, all 7 plans. NOTE: since v8, GUIDED_PARAM_OVERRIDES holds
  TECS_SINK_MAX 8, so rebuilding a round before 11 here also needs "TECS_SINK_MAX=6" in its param list.
  tier3_v8 (7 plans, no wind): all PASS, mean max 1.57 m (v7 1.59), worst 2.52 (2.62), rms 0.56 both,
  throttle 525 %/min (582); the loop added 0.00 m/s on every plan. v8 KEPT as the default.

Usage (PyCharm terminal, project root):
  python -m ardupilot.guided_test make       # writes localstore/batches/r14_none, r14_t10, r14_h10
  (WSL)  bash /mnt/c/Users/isaia/fixed-wing-rl/ardupilot/fly_guided_test.sh
  python -m ardupilot.guided_test analyze    # compares with the same winds without the speed loop
"""
import csv
import json
import sys
from types import SimpleNamespace

import numpy as np

from ardupilot import ardupilot_mission as M

PAIRS = [["tier3_easy_envs.json", "tier3_easy_results.json"],
         ["tier3_nominal_envs.json", "tier3_nominal_results.json"],
         ["tier3_dense_envs.json", "tier3_dense_results.json"]]
FLY = ["tier3_dense_envs_1"]                     # round 1: also tier3_easy_envs_0, tier3_nominal_envs_2
V5 = dict(lookahead=150.0, alt_lead=30.0, alt_step=0.5, speed_step=None, speed_hold="off", param=[],
          cmd_delay=None, pos_delay=None, speed_ctrl="off", descent_guard="off", alt_guard="on", speed_rate=None,
          up_only="off")
# round 4: r4a alt_step 0; r4b speed_hold "on"; r4s speed_step 0; r4c param TECS_PTCH_DAMP=0.2
R4S = {**V5, "speed_step": 0.0}
R5 = {"r5a": {**R4S, "param": ["TECS_THR_DAMP=0.25"], "what": "airspeed every update, TECS_THR_DAMP 0.25"},
           "r5b": {**R4S, "param": ["TECS_SPD_OMEGA=1"], "what": "airspeed every update, TECS_SPD_OMEGA 1"},
           "r5c": {**R4S, "param": ["THR_SLEWRATE=30"], "what": "airspeed every update, THR_SLEWRATE 30"},
           "r5d": {**R4S, "param": ["TECS_HGT_OMEGA=1.5"], "what": "airspeed every update, TECS_HGT_OMEGA 1.5"}}
# v6 = R4S; THR_SLEWRATE 30 comes from ardupilot_mission.GUIDED_PARAM_OVERRIDES
V6 = {**R4S, "param": []}
R6 = {**{f"r6c{round(d * 100):03d}": {**V6, "cmd_delay": d, "what": f"setpoints delayed {d:g} s"}
              for d in (0.1, 0.25, 0.5, 1.0, 2.0)},
           **{f"r6p{round(d * 100):03d}": {**V6, "pos_delay": d, "what": f"position {d:g} s old"}
              for d in (0.5, 1.0)}}
R7 = {f"r7a{L}": {**V6, "alt_lead": float(L), "what": f"altitude lead {L} m"} for L in (10, 15, 20)}
V7 = {**V6, "alt_lead": 10.0}
WIND_FROM = {"h": 80, "t": 260, "c": 170}   # SIM_WIND_DIR; meant as head/tail/cross for dense_1 (bearing 081),
                                            # but JSBSim's wind blows TOWARD it: "h" = tailwind, "t" = headwind
R8 = {f"r8{k}{v:02d}": {**V7, "param": [f"SIM_WIND_SPD={v}", f"SIM_WIND_DIR={d}"],
                             "what": f"{v} m/s wind from {d:03d} deg"}
           for k, d in WIND_FROM.items() for v in (5, 10)}
wind = lambda v, d: [f"SIM_WIND_SPD={v}", f"SIM_WIND_DIR={d}"]    # JSBSim: the wind blows TOWARD d
V7S = {**V7, "speed_ctrl": "on"}
R9 = {"r9n": {**V7S, "what": "speed loop, no wind"},
           "r9t05": {**V7S, "param": wind(5, 80), "what": "speed loop, 5 m/s tailwind"},
           "r9t10": {**V7S, "param": wind(10, 80), "what": "speed loop, 10 m/s tailwind"},
           "r9h10": {**V7S, "param": wind(10, 260), "what": "speed loop, 10 m/s headwind"},
           "r9c10": {**V7S, "param": wind(10, 170), "what": "speed loop, 10 m/s crosswind"}}
WINDS = {"none": [], "t05": wind(5, 80), "t10": wind(10, 80), "h10": wind(10, 260), "c10": wind(10, 170)}
R10 = {f"r10{m}_{w}": {**V7, "speed_ctrl": mode, "param": p, "what": f"speed loop {mode}, wind {w}"}
           for m, mode in (("ff", "ff"), ("hy", "hybrid")) for w, p in WINDS.items()}
HG = {**V7, "speed_ctrl": "hybrid", "descent_guard": "on"}
SINK8 = ["TECS_SINK_MAX=8"]
R11 = {"r11g_none": {**HG, "what": "hybrid + descent guard, no wind"},
           "r11g_t05": {**HG, "param": WINDS["t05"], "what": "hybrid + descent guard, 5 m/s tailwind"},
           "r11g_t10": {**HG, "param": WINDS["t10"], "what": "hybrid + descent guard, 10 m/s tailwind"},
           "r11gs_t05": {**HG, "param": WINDS["t05"] + SINK8, "what": "hybrid + descent guard + TECS_SINK_MAX 8, 5 m/s tailwind"},
           "r11gs_t10": {**HG, "param": WINDS["t10"] + SINK8, "what": "hybrid + descent guard + TECS_SINK_MAX 8, 10 m/s tailwind"},
           "r11s_t10": {**V7, "param": WINDS["t10"] + SINK8, "what": "no speed loop, TECS_SINK_MAX 8, 10 m/s tailwind"}}
HS8 = {**V7, "speed_ctrl": "hybrid"}
R12 = {f"r12_{w}": {**HS8, "param": p + SINK8, "what": f"hybrid speed loop + TECS_SINK_MAX 8, wind {w}"}
           for w, p in WINDS.items()}
H13 = {**HS8, "alt_guard": "off", "speed_rate": 0.5}
R13 = {f"r13_{w}": {**H13, "param": WINDS[w] + SINK8,
                        "what": f"hybrid + TECS_SINK_MAX 8, no altitude guard, correction <= 0.5 m/s/s, wind {w}"}
           for w in ("none", "t05", "t10")}
H14 = {**H13, "up_only": "on"}
CONFIGS = {f"r14_{w}": {**H14, "param": WINDS[w] + SINK8, "what": f"round 13 loop, speed-up only, wind {w}"}
           for w in ("none", "t10", "h10")}
BASELINE = "tier3_v7"
ROUND1 = ["r8h10", "r11s_t10", "r13_t10", "r8t10", "r12_h10", "r13_none"]   # no loop, sink 8, earlier loops                       # earlier batches, shown in the comparison too
COLS = [("outcome", "result"), ("max_error_m", "max err"), ("rms_error_m", "rms"),
        ("max_horizontal_error_m", "max side"), ("max_vertical_error_m", "max vert"),
        ("min_clearance_m", "clear"), ("airspeed_vs_plan_rms_ms", "spd rms")]


def make():
    for name, c in CONFIGS.items():
        L = c["lookahead"]
        note = f"{name}: GUIDED follower (tracking setup v7) with {c['what']}"
        print(f"\n##### {note}")
        M.batch_make(SimpleNamespace(name=name, pair=PAIRS, chord_tol=M.BATCH_CHORD_TOL_M, speed_profile=None,
                                     speed_lead=0.0, speedup=M.GUIDED_SPEEDUP, note=note, param=c["param"], overwrite=True,
                                     max_waypoints=None, fly_all=False, alt_lead=0.0, min_spacing=None,
                                     wp_spacing=None, track_comp=None, follower="guided", lookahead=L,
                                     guided_alt_comp="on", guided_alt_lead=c["alt_lead"], guided_speed_ramp="off",
                                     guided_alt_rate=100000.0, guided_alt_step=c["alt_step"],
                                     guided_speed_step=c["speed_step"], guided_speed_hold=c["speed_hold"],
                                     guided_cmd_delay=c["cmd_delay"], guided_pos_delay=c["pos_delay"],
                                     guided_speed_ctrl=c["speed_ctrl"], guided_descent_guard=c["descent_guard"],
                                     guided_alt_guard=c["alt_guard"], guided_speed_rate=c["speed_rate"],
                                     guided_speed_up_only=c["up_only"]))
    bad, n = [], 0
    for name in CONFIGS:
        for inst in json.load(open(M.batch_path(name) / "manifest.json"))["instances"]:
            if inst["id"] in FLY:
                n += 1
                if inst["status"] != "flyable":
                    bad.append(f"  {name} {inst['id']}: {inst['status']} {'; '.join(inst.get('rejects', []))}")
    print("\n" + "=" * 70)
    if bad:
        print(f"NOT READY: {len(bad)} of {n} flights are not flyable. Do not fly yet; send this:")
        print("\n".join(bad))
    else:
        print(f"READY: all {n} flights are flyable. Now fly them (WSL):")
        print("  bash " + M.wsl_path(M.project_root() / "ardupilot" / "fly_guided_test.sh"))


def read(name):
    f = M.batch_path(name) / "results.csv"
    return {r["id"]: r for r in csv.DictReader(open(f))} if f.exists() else {}


def throttle_activity(name, iid):
    """Over the plan segment: throttle travel (% per minute), share of time at 0 or 100%, and the
    jitter (rms of the part faster than 5 s) of the throttle (%) and of TECS's pitch command (deg)."""
    run = M.batch_path(name, "runs", iid)
    log, flight = run / "flight.tlog", run / "flight.csv"
    if not (log.exists() and flight.exists()):
        return None
    f = M.read_tlog(str(log))
    c = list(csv.DictReader(open(flight)))
    if "throttle" not in f or not c:
        return None
    ll = np.c_[f["lat"], f["lon"]]
    g = lambda r: np.array([float(r["lat_deg"]), float(r["lon_deg"])])
    i0, i1 = (int(np.argmin(np.linalg.norm(ll - g(r), axis=1))) for r in (c[0], c[-1]))
    th, t = np.asarray(f["throttle"][i0:i1], float), np.asarray(f["t"][i0:i1], float)
    if len(th) < 10:
        return None
    travel = np.abs(np.diff(th)).sum() / ((t[-1] - t[0]) / 60.0)
    thr_jit = M.oscillation(t, th, window=5.0)[0]
    pit_jit = None
    if "nav" in f:
        nv = f["nav"][(f["nav"][:, 0] >= t[0]) & (f["nav"][:, 0] <= t[-1])]
        pit_jit = M.oscillation(nv[:, 0], nv[:, 2], window=5.0)[0]
    return travel, 100.0 * np.mean((th <= 1) | (th >= 99)), thr_jit, pit_jit


def timing(name, iid):
    """From the follower's log: flight time over the plan, the plan's time for the same stretch (the plan's
    airspeed, flown in still air), the schedule lag at the end and its largest value (s; > 0 = behind),
    and the mean groundspeed."""
    f = M.batch_path(name, "logs", f"{iid}.guided.csv")
    if not f.exists():
        return None
    r = [x for x in csv.DictReader(open(f)) if x["state"] == "active"]
    if len(r) < 10:
        return None
    r = [x for x in r if float(x["s_m"]) >= 0.0]          # from the plan's start
    if len(r) < 10:
        return None
    vk = "v_plan_ms" if "v_plan_ms" in r[0] else "v_cmd_ms"  # the plan's airspeed (logged since round 9)
    t, s, v = (np.array([float(x[k]) for x in r]) for k in ("t_boot_s", "s_m", vk))
    t_plan = np.concatenate([[0.0], np.cumsum(np.diff(s) / np.maximum(v[1:], 1.0))])
    lag = (t - t[0]) - t_plan
    gs = np.mean([float(x["groundspeed_ms"]) for x in r])
    return t[-1] - t[0], t_plan[-1], lag[-1], lag[np.argmax(np.abs(lag))], gs


def analyze():
    for name in CONFIGS:
        M.batch_analyze(SimpleNamespace(name=name, plot=False))
    names = [BASELINE] + ROUND1 + list(CONFIGS)
    res = {n: read(n) for n in names}
    print(f"\n\n========== GUIDED follower round 14 ({BASELINE} = v7, no wind; r8h10 / r8t10 = no loop, tail / head 10; r11s_t10 = no loop + sink 8, tail 10) ==========")
    for iid in FLY:
        print(f"\n{iid}")
        print(f"  {'':16}" + "".join(f"{lab:>10}" for _, lab in COLS)
              + f"{'thr %/min':>11}{'thr 0/100':>10}{'thr jit %':>10}{'pitch jit':>10}")
        for n in names:
            r = res[n].get(iid, {})
            th = throttle_activity(n, iid)
            fmt = lambda x, d: f"{x:>10.{d}f}" if x is not None else f"{'-':>10}"
            print(f"  {n:16}" + "".join(f"{r.get(k, '-') or '-':>10}" for k, _ in COLS)
                  + (f"{th[0]:>11.0f}{th[1]:>9.0f}%" + fmt(th[2], 2) + fmt(th[3], 3) if th
                     else f"{'-':>11}{'-':>10}{'-':>10}{'-':>10}"))
        print(f"  {'schedule':16}{'time s':>10}{'plan s':>10}{'lag end s':>10}{'lag max s':>10}{'gs m/s':>10}")
        for n in names:
            tm = timing(n, iid)
            print(f"  {n:16}" + ("".join(f"{x:>10.1f}" for x in tm) if tm else f"{'-':>10}"))
    print("\nthr jit / pitch jit: rms of the throttle (%) and of TECS's pitch command (deg) faster than 5 s.")
    print("\nPer-flight charts: localstore/batches/<batch>/runs/<id>/check.png; "
          "the follower's own log: localstore/batches/<batch>/logs/<id>.guided.csv")


if __name__ == "__main__":
    {"make": make, "analyze": analyze}.get(sys.argv[1] if len(sys.argv) > 1 else "", lambda: print(__doc__))()
