# Pole-climbing experiment rig

Written for: whoever runs or analyses these experiments next.

A snake robot climbing a vertical pole by **adaptive helical rolling** —
the method of Takemori, Tanaka & Matsuno, *IEEE Transactions on Robotics*
39(1):437–451, 2023 — set up so you can vary the things that matter, press
Start, and get a labelled CSV of everything that happened.

---

## Running it

One-time setup (Python 3.10+; tkinter ships with the python.org installer):

```
python -m venv .venv
.venv\Scripts\activate            Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

Then:

```
python experiment_gui.py          control panel: friction slider, Start / Stop
python climb_rig.py               one run, headless, defaults
python climb_rig.py --viewer      one run, watch it (annotated, tracking camera)
python climb_rig.py --viewer --immersive false    the plain MuJoCo viewer
python climb_rig.py --mu 0.3 --alpha 0.30 --label plastic_pipe
python climb_rig.py --start_wrapped true    skip the approach; test the controller alone
python sweep_example.py           one parameter at a time, the README table
```

Every `Config` field is a `--flag`; `--mu`, `--mass`, `--dt`, `--fps` and
`--speed` are short aliases for the ones you reach for most.

It is also importable, which is how you sweep:

```python
from climb_rig import Config, run
for mu in (0.3, 0.5, 0.8, 1.2):
    print(mu, run(Config(mu_robot_pole=mu, label=f"mu{mu}"))['rolling_rate_cm_s'])
```

Each run creates `results/<timestamp>_<label>/` holding `run.csv`,
`joints.csv`, `stages.csv`, `summary.csv` and `config.json`. The
`20260923_225741_example` folder is from an earlier version (friction 2.0,
held setpoints) and no longer matches the defaults.

The defaults model a **non-sticky pipe, friction 0.5** (rubber-ish on a clean
pipe; plastic is about 0.3). The robot reaches the pole at about t = 10 s, has
hold of it by t = 17 s and reaches the top of the 3 m pole at about t = 93 s,
climbing at 3.7 cm/s. It climbs down to a friction of about 0.25.

The catch is verified, not assumed: the climb only starts once the coil is
round the pole (see *Catching the pole* below), so a run can no longer set off
"climbing" a coil that is lying on the floor beside it.

**Start placed on the pole** (`--start_wrapped true`, or the checkbox) drops
the robot onto the pole already wrapped, as all four of the paper's own
experiments do, and skips straight to the control loop. Use it whenever the
question is about the controller: a parameter that happens to break the floor
gait otherwise shows up as a climbing failure, which it is not.

---

## The panel

One control up front: a **friction slider**, plus **Start**, **Stop** and
**Rerun**.

The slider sets the *starting* friction before you press Start. Once the 3-D
viewer opens with **Immersive** ticked, a matching slider is drawn right in
that window (see below) and becomes the live control — this panel's slider
goes read-only for the run and just mirrors wherever the window's slider is
dragged to, since a control you can't act on is more confusing left active
than disabled. With the viewer off, or Immersive unticked, this slider stays
the one that's live. Either way, whatever value is in force is written to the
`mu_robot_pole` column of every row of `run.csv` — a run with a moving slider
is still completely described by its own log.

The slider runs 0.1–1.5: realistic, non-sticky pipes. Measured, held constant
for a whole run: mu 0.25 → 1.47 cm/s, 0.3 → 2.27, 0.5 → 3.69, 0.8 → 4.30,
1.2 → 4.53. Dragged down below about 0.25 mid-climb, the climb stalls and
`mu_required_p95` pins to whatever is available — that is the robot sliding.

**Rerun** repeats the exact configuration of the last Start — same friction,
same everything — with one click. It enables after any outcome: a completed
run, a Stop, a solver divergence, or an outright exception. Nothing in this
panel can crash it into an unusable state; an exception during a run is
caught, shown in a dialog, logged in the run log, and leaves Start and Rerun
both live so you can immediately try again rather than restarting the app.

Everything else keeps its default under **Show all parameters**, collapsed, or
from the command line. The panel is 797 × 571 collapsed.

---

## What is in this folder

| file | what it is |
| --- | --- |
| `experiment_gui.py` | the control panel — parameters, Start/Stop, live readout |
| `climb_rig.py` | the experiment itself: parameters, stage machine, CSV logging |
| `immersive.py` | the annotated, tracking 3-D view — camera, heads-up panel, 2-D cross-section plot, native friction slider |
| `helical_rolling.py` | the control method, step by step, cross-referenced to the paper's equations |
| `snake_backbone.py` | link-edge points and link lengths, read out of the MuJoCo model |
| `calibrate_helical.py` | solves the sign convention eq. (32) takes on this robot |
| `climb_from_wrapped.py` | the control loop alone, from a pre-wrapped start — used to check the mechanism |
| `robot_pipe_climb.py` | the original single-purpose script this rig was built from; kept for reference |
| `9motor_sidewinder_pipe.xml`, `*.STL` | the model |

`climb_rig.py` supersedes `robot_pipe_climb.py`. They do the same run; the rig
parameterises it and logs it.

---

## Watching it: the immersive view

`--viewer` (or the **Show 3-D viewer** tick) opens an annotated MuJoCo window
that follows the run and lets you steer friction from inside it. Untick
**Immersive**, or pass `--immersive false`, for the plain viewer.

The problem it solves: the stock view is a fixed camera on a 3 m pole, so
twenty seconds in the robot has climbed out of frame, and nothing the
controller is *deciding* is visible at all — only where the bodies ended up.

**The camera follows the robot.** It tracks the centre of mass with heavy
smoothing, because a rolling helix wobbles a couple of centimetres every
revolution and a camera that copies that wobble is unwatchable. `B` cycles
follow → orbit (which also rotates slowly around the pole) → free.

**A friction slider is drawn and dragged right in this window** — the bar
running along the bottom of the screen. This is the point of it: rather than
alt-tabbing to a Tkinter panel, you watch the robot and drag the same window
it's climbing in. It is the authoritative control whenever this viewer is
open — the Tkinter panel's own slider goes read-only for the duration of the
run and just mirrors wherever this one is, since a control you can't see
doing anything is worse than no control. Drag it left mid-climb and watch the
robot's grip fail in real time; drag it right and watch the climb speed up.
Whatever value is in force is written to the `mu_robot_pole` column of every
row of `run.csv`, so a run with a moving slider is still fully described by
its own log.

**A 2-D plot, bottom-left, own dedicated space.** The cross-sectional shape
the controller is working with — measured vs. commanded, the paper's Fig. 11
— is a proper flat plot here, not something drawn as rings overlaid on top of
the 3-D robot. Cyan is the measured shape (the robot's form projected onto
the plane perpendicular to the estimated axis, eqs. 10–16); orange is the
cross-section the compliance term just asked for (eq. 23). **The gap between
the two curves is the grip.**

**The 3-D scene itself carries only the height ticks**, every 0.5 m up the
pole, labelled.

**Nothing is drawn on or sticking out of the body.** Contact-force arrows, a
translucent ghost of the commanded form, and cross-section rings round the
robot were all tried and all made the wrap harder to read, not easier. The
contact arrows are still one keypress away (`C`) when they're the question.
The floor's mirror reflection is off for the same reason — it drew a second,
see-through snake under the real one.

The plot sits beside the viewer's own FPS / solver readout in the bottom-left
corner and the slider to its right, all sized from the font, so nothing
overlaps at any window size.

**A heads-up panel** (top right) carries the same numbers that are going into
`run.csv` that tick — stage, height, climb rate, coil radius, grip squeeze,
axis tilt, pole and self contacts, friction demanded against friction
available, and how many of the 18 motors are saturated. `Z` hides it.

Keys, on top of the viewer's own:

| key | |
| --- | --- |
| `B` | camera: follow → orbit → free |
| `N` | annotations (height ticks and the 2-D plot): on → off |
| `G` | also toggles the 2-D plot — shared with the base viewer's own "graph" key, which otherwise has nothing left to hide |
| `Z` | heads-up panel |
| `SPACE` | pause (`→` steps one frame) |
| `C` `J` `M` `I` `O` `V` | contact arrows (off by default), joints, centre of mass, inertia, shadows, convex hulls |
| `H` | menus |

The slider isn't gated by any of these — it's a control, not decoration, so
it stays live regardless of what `N`/`Z`/`G` are set to.

### On the scene itself

`9motor_sidewinder_pipe.xml` gained a skybox, three lights instead of one,
shadow and near-plane settings that suit a tall thin scene, and haze so the
floor fades into the sky rather than ending on a hard line. The floor's
reflectance is 0, so no mirror image. All of it is inside `<visual>`,
`<asset>` and `<light>`.

**None of it changes the physics**, and that is checked rather than asserted:
every physics field of the compiled model (masses, inertias, geom sizes,
frictions, solver parameters, joint and actuator ranges) compares equal to the
previous XML, and 8 s of identical control input produces trajectories that
differ by 0.0.

---

## The six stages

A climbing run is not one behaviour, it is six, and averaging across them
hides everything interesting — the approach barely changes height, the
catching stage gains a few centimetres by curling, and only one stage is the
experiment. **Every logged row carries the stage it came from**, and
`stages.csv` gives one row per stage with its duration and height change.

| stage | what is happening | what to look at |
| --- | --- | --- |
| `approach` | sidewinding across the floor to the pole. Open loop, and nothing to do with the paper — its experiments all start with the robot already on the pipe | did it arrive at all (`pairs_wrapped` starts rising) |
| `catching` | the robot straightens against the pole, checks where the pole sits along its body, then curls round it pair by pair outward from the pair nearest it. A miss sends it back to `approach` (or, if the coil is round the pole but crooked, loosens and re-closes it in place) — see *Catching the pole* | the run's messages say which, and why |
| `settling` | all nine pairs wrapped, shape held still so contacts settle, then the grip is checked before anything climbs | height should be roughly flat; a drop means it is sliding |
| `pitch_ramp` | the adaptive loop is running and the pitch angle opens from the catching value to the climbing one. Already holding on, already climbing a little | `alpha_rad` ramping, `wrap_turns` settling |
| `rolling` | the experiment proper: adaptive helical rolling at full pitch angle | **quote climb rates from this stage** |
| `holding` | top reached, `psi_roll` frozen, robot just gripping | whether height holds or creeps down |

Only `rolling` is a steady state, so `summary.csv` reports
`result.rolling_rate_cm_s` separately from the whole-run numbers.

---

## `run.csv` — one row per sample, whole robot

Sampled at `log_hz` (50 Hz by default). Columns, grouped by what they tell you:

**When and where**

| column | unit | meaning |
| --- | --- | --- |
| `t_s`, `step` | s, – | simulated time and physics step |
| `stage`, `stage_t_s` | –, s | stage name and time since that stage began |
| `mu_robot_pole` | – | the friction coefficient **in force on this row**. Constant unless the slider was dragged |
| `com_x_m`, `com_y_m`, `com_z_m` | m | centre of mass. `com_z_m` is the headline result |
| `climb_rate_cm_s` | cm/s | finite difference of `com_z_m` between samples — noisy, so average it |

**What the controller is commanding**

| column | unit | meaning |
| --- | --- | --- |
| `psi_roll_rad` | rad | `psi_roll` of eq. (33). The **only** term that produces climbing: it rolls the body around an otherwise unchanged target form |
| `alpha_rad` | rad | pitch angle currently commanded. Constant during `rolling`; that constancy is what confines compliance to the direction perpendicular to the pole |
| `axis_x/y/z`, `axis_tilt_deg` | –, deg | the helical axis the controller **estimated from the joint angles alone** (eqs. 5–9), and its tilt from vertical. Blank before the loop starts. A healthy climb holds this under ~5° |
| `coil_r_meas_m` | m | how far the measured cross-section sits from its own centre |
| `coil_r_cmd_m` | m | the radius the compliance term just asked for |
| `squeeze_mm` | mm | `coil_r_meas − coil_r_cmd`. **This gap is the grip** — the deflection the position servos are pressing into the pole. Zero means the robot is not gripping; much larger than `K × radius` means the wrap has come off |
| `wrap_turns` | turns | how far the cross-section winds around. Below 1.0 the coil cannot hold |

**What the pole is doing back**

| column | unit | meaning |
| --- | --- | --- |
| `n_contact_pole` | – | contacts against the pole. The paper's central claim is *locality* — many independent contacts, each adapted to the pole in front of it |
| `n_contact_self` | – | body-on-body contacts. Should be near zero: if the robot is climbing on these, it is shoving itself up rather than rolling |
| `n_contact_floor` | – | mostly an approach-stage number |
| `pole_normal_force_N` | N | total normal force on the pole. Around 50 N for a 0.74 kg (7.2 N) robot — the grip is far heavier than the weight it carries |
| `mu_required_p95`, `mu_required_max` | – | at each pole contact, tangential force ÷ normal force: the friction coefficient that contact is *demanding*. **Compare with `mu_robot_pole`.** Baseline runs average 0.45 against the 0.5 available: the gait runs right at the slip boundary |

**Whether the hardware could do this**

| column | unit | meaning |
| --- | --- | --- |
| `joint_torque_rms_Nm`, `joint_torque_max_Nm` | N·m | *delivered* joint torque, after MuJoCo clamps to the model's ±0.8 N·m |
| `joints_saturated` | – | how many of the 18 joints are asking for more than ±0.8 N·m. Typically 2–4 during the climb — these motors are working at their limit, which is a real constraint and not a simulation artefact |
| `mech_power_W` | W | Σ\|torque × velocity\| across the joints |

**Whether the run is trustworthy**

| column | unit | meaning |
| --- | --- | --- |
| `max_abs_qvel` | – | largest generalised velocity. The rig aborts above 120; a healthy climb sits near 5 |
| `body_roll_rev` | rev | cumulative rotation of a mid-body link about **its own long axis**, integrated from ω·x̂ |
| `orbit_rev` | rev | cumulative rotation of that link **around the pole** |
| `pairs_wrapped` | – | how many of the nine pairs have been triggered into the wrap |

`body_roll_rev` and `orbit_rev` together are the proof that the mechanism is
what it claims to be. A rolling helix screws itself up the pole *without*
orbiting it: baseline runs show roll tracking 95% of commanded `psi_roll`
while orbiting only ~1.9 revolutions in 16.8.

---

## `joints.csv` — one row per sample **per joint**

18 rows per sample, so `t_s` repeats. Long format, which is what you want for
`groupby(['stage','joint'])` or a small-multiples plot.

| column | unit | meaning |
| --- | --- | --- |
| `t_s`, `step`, `stage` | | as above |
| `joint`, `joint_name` | – | 1–18, `J1`–`J18`. **`J1` is at the tail**, `J18` at the head |
| `pair` | – | 1–9. Joints come in orthogonal pairs; a pair is the unit that bends in 3-D |
| `axis_kind` | – | `pitch` or `yaw`, classified from the model's own joint axes rather than assumed |
| `cmd_rad` | rad | commanded angle — the target form, held between control ticks |
| `pos_rad` | rad | actual angle |
| `err_rad` | rad | `cmd − pos`. **This is not a defect, it is the measurement**: the pole blocks the commanded squeeze, and the resulting tracking error is what the servo converts into grip force |
| `vel_rad_s` | rad/s | joint velocity |
| `torque_Nm` | N·m | torque actually delivered, clamped to ±0.8 |
| `torque_demand_Nm` | N·m | what the servo law asked for, *unclamped* — often 10× the limit during transients |
| `torque_frac` | – | `torque_Nm` ÷ 0.8 |
| `saturated` | 0/1 | whether demand exceeded the limit |
| `range_frac` | – | position within the joint's ±1.4 rad range, 0…1. Near 0 or 1 means the joint is against its stop and the shape cannot be achieved |

Size: 18 rows × `log_hz` × duration. A full 140 s run at 50 Hz is about
126 000 rows, ~12 MB. Drop `log_hz` to 10–20 if that is inconvenient; the
controller only runs at 10 Hz, so little is lost.

---

## `stages.csv`, `summary.csv`, `config.json`

`stages.csv` — one row per stage: `stage`, `t_start_s`, `t_end_s`,
`duration_s`, `com_z_start_m`, `com_z_end_m`, `climb_m`, `mean_rate_cm_s`.
This is the "categorised by stage" view in its smallest form.

`summary.csv` — a flat `key,value` table with **every** config field prefixed
`config.` and every outcome prefixed `result.`, so a sweep is just
concatenating these files. Results include `peak_com_z_m`, `reached_top`,
`time_to_top_s`, `rolling_climb_m`, `rolling_rate_cm_s`, `pairs_wrapped`,
`body_roll_rev`, `orbit_rev`, `diverged`, `stopped_early`, `wall_time_s`.

`config.json` — the same config, machine-readable, for re-running.

---

## The parameters

The panel is grouped the same way `Config` is.

**Environment** — `pole_radius`, `mu_robot_pole`, `mu_floor`, `mass_scale`,
`gravity`. Friction and mass are applied to the compiled model, not the XML,
so sweeps never touch a file. `mass_scale` multiplies mass *and* inertia, so
it is a clean "heavier robot, same robot" knob.

**Solver and timing** — `timestep`, `control_period`, `setpoint_period`,
`max_duration`, `target_speedup`, `target_fps`. `control_period` is how often
the controller re-estimates the form from the joints, and 0.1 s is the paper's
own hardware rate. `setpoint_period` is how often joint targets are streamed
between those ticks (see *Why it climbs on a slippery pipe*).
`target_speedup`/`target_fps` are display only and change nothing physical.

**Friction** — `mu_robot_pole` applies to robot–pole contacts and `mu_floor`
to robot–floor contacts, each on its own. (MuJoCo normally resolves a contact
pair by taking the larger of the two coefficients, which used to let robot–floor
contacts silently use the pole's value. The floor and pole now take contact
priority, so each surface's own number is the one in force.)

**Controller** — `alpha` (pitch angle of the climbing helix; −1, the
default, chooses it from the pole radius — see *Pitch angle from the pole
radius*), `psi_dot` and
`spin` (rolling speed and direction), `k_mid`/`k_end` (compliance gain,
eq. 23), `lead_sign` (handedness). `spin` and `lead_sign` must match: `+1/+1`
and `−1/−1` climb, the mixed pairs slide off.

Note there is **no coil radius parameter for the climb**, and that is the
point of the method: the radius is whatever the pole leaves the robot at, and
the compliance term only ever asks for "a bit tighter than that". The same
numbers work on a pole of a different size.

**Start** — either `start_wrapped` with `start_radius` and `start_height`
(placed on the pole, like the paper), or the catching parameters below.
Catching is not in the paper, whose experiments all start with the robot
already on the pipe. A catching shape has different
requirements from a climbing shape: nearly flat (a steeply pitched helix
screws off the pole instead of closing around it) and tight. `wrap_psi` sets
**which plane** the ring closes in and **on which side of the body** — at 0
the ring stands up in a vertical plane and misses the pole; at +π/2 it lies
flat but closes on the side away from the pole; −π/2 (the default) closes it
round the pole.

### Catching the pole

The original catch curled each pair as it happened to pass near the pole and
then started climbing regardless. It worked about 60% of the time, and which
60% depended on millimetres of start position and on the MuJoCo version (the
same code caught the pole on 3.13 and missed it on 3.14). A miss left a coil
on the floor that the controller then "climbed" with zero pole contacts.

It now works like this, and caught and climbed in 57 of 60 test runs at pole
friction 0.3, 0.5 and 0.8, including randomised starts:

1. **Aim.** The start-placement probe aims a point `aim_u` along the body at
   the pole, not the tail, so the pole arrives mid-catch-window.
2. **Straighten against the pole** when any pair comes within
   `prox_threshold`, and come to rest (`straighten_time`, `straighten_hold`).
   The catch window was mapped from rest; curling straight out of the moving,
   wavy gait gives a coil tens of degrees off vertical.
3. **Check the window.** Curl only if the pole is `catch_u_min`–`catch_u_max`
   along the body from pair 1 and within `catch_v_max` of it; otherwise it is
   a miss. The body's roll about its own axis is measured and subtracted from
   `wrap_psi`, so the ring always closes toward the pole.
4. **Curl as a wave** outward from the pair nearest the pole, one pair every
   `wrap_wave_delay`, so the body winds round the pole like a rope round a
   post.
5. **Verify the grip** at the end of `settling`: at least `min_grip_contacts`
   pole contacts, the backbone winding at least `min_wrap_turns` round the
   pole, the COM within `grip_com_radius` of it, and the coil's own axis
   within `max_catch_tilt` of vertical. Only then does the climb start.
6. **Retry.** A coil round the pole but crooked is loosened to `reopen_frac`
   and re-closed in place from the same pair, **with the next catching shape**
   in `CATCH_ALTERNATES` (pitch, radius, wave spacing). How a coil folds on
   the floor is repeatable for one shape, so re-closing with the same shape
   tends to repeat the same crooked coil; each alternate catches starts the
   others miss. A
   coil that closed off the pole releases back into the approach gait. A
   catch that passes the check but peels off in the first moments of the
   climb (`climb_abort_*`) is re-closed too. After `max_catch_attempts` the
   run stops and says so — `result.caught_pole` and `result.catch_misses`
   in `summary.csv` record what happened.

---

## Why it climbs on a slippery pipe

The paper's controller runs at 0.1 s: each tick it re-reads the joints,
re-estimates the helical form (its steps 1–6) and computes the joint targets
for the current roll phase `psi_roll` (step 7, eqs. 32–33). The rig used to
**hold** those targets for the whole tick, so every 0.1 s all 18 setpoints
jumped by `psi_dot × control_period` = 9° of roll. The servos slammed toward
each jump, saturated at ±0.8 N·m, and the contacts slipped. With grip pads
(mu 2.0) that cost half the climb speed; on a real pipe (mu < 1) the robot slid
down.

The paper's own structure separates the two: estimating the form needs a
fresh read of the joints, but rolling the body round that form (eq. 33) does
not. So the form is still estimated every `control_period` (0.1 s, as in the
paper), while step 7 is re-evaluated with the advancing `psi_roll` every
`setpoint_period` (0.01 s) and streamed to the servos
(`HelicalRollingController.retarget`). Streaming goal positions at 100 Hz is
routine for bus servos. Measured from a pre-wrapped start:

| mu | 0.2 | 0.25 | 0.3 | 0.4 | 0.6 | 0.8 | 1.0 | 2.0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| setpoints held (old), cm/s | – | – | – | −0.86 | – | −0.02 | 1.02 | 2.43 |
| setpoints streamed, cm/s | −0.02 | 1.47 | 2.27 | 3.17 | 4.02 | 4.33 | 4.46 | 4.57 |

Below about mu 0.25 it stops climbing whatever the gains: 0.8 N·m servos can
only squeeze a 0.74 kg robot so hard. Raising K, or rolling faster or slower,
moves that limit by a few hundredths at most.

---

## Measured parameter sensitivity

Twenty-one runs, 60 s of simulated time each, **started placed on the pole**
so that what is being measured is the controller and not the floor gait.
Baseline is the defaults (mu 0.5). One parameter changed at a time.
`mu needed` and the contact counts are averages over the `rolling` stage.
`python sweep_example.py` reproduces it.

| case | rolling cm/s | climb m | pole contacts | mu needed (p95) | motors saturated |
| --- | --- | --- | --- | --- | --- |
| **baseline (mu 0.5)** | **3.69** | 2.10 | 11.2 | 0.45 | 2.7 |
| mu 0.25 | 1.47 | 0.84 | 12.5 | 0.24 | 2.6 |
| mu 0.3 | 2.27 | 1.29 | 12.2 | 0.28 | 2.7 |
| mu 0.8 | 4.30 | 2.43 | 10.7 | 0.68 | 2.9 |
| mu 1.2 | 4.53 | 2.42 | 11.3 | 0.92 | 2.9 |
| setpoints held (old behaviour) | **−0.41** | −0.24 | 9.7 | 0.43 | 3.8 |
| mass x0.5 | 2.11 | 1.20 | 11.8 | 0.44 | 2.7 |
| mass x2 | 4.07 | 2.14 | 10.4 | 0.47 | 2.8 |
| mass x4 | **−0.09** | −0.05 | 3.6 | 0.31 | 4.8 |
| dt 2 ms | 3.62 | 2.07 | 11.3 | 0.45 | 2.7 |
| dt 0.5 ms | 3.71 | 2.11 | 11.2 | 0.45 | 2.8 |
| control period 0.2 s | 3.56 | 2.03 | 10.8 | 0.45 | 2.6 |
| control period 0.02 s | 3.86 | 2.20 | 10.7 | 0.45 | 3.1 |
| alpha 0.15 | **0.22** | 0.12 | 9.2 | 0.45 | 3.8 |
| alpha 0.35 | 3.38 | 1.92 | 11.8 | 0.45 | 2.6 |
| psi_dot x0.5 | 1.81 | 1.03 | 12.1 | 0.43 | 2.9 |
| psi_dot x2 | **6.96** | 2.45 | 10.1 | 0.46 | 2.8 |
| K 0.02 | **0.09** | 0.05 | 5.5 | 0.45 | 0.0 |
| K 0.15 | 3.70 | 2.11 | 12.2 | 0.44 | 7.0 |
| pole radius 0.025 | **−0.02** | −0.01 | 3.5 | 0.38 | 4.2 |
| ~~pole radius 0.060~~ | ~~−0.15~~ | ~~−0.09~~ | ~~6.1~~ | ~~0.40~~ | ~~3.5~~ |

The struck-out row was simulated against a partly non-colliding pole (see
*Collision testbench*) and is invalid. Re-run with the fix, a 6 cm pole at
mu 0.5 climbs at **2.08 cm/s** and reaches the top. The 2.5 cm row is
unaffected by that bug and stands.

What this says:

**Streaming the setpoints is the whole difference.** Holding them for the
tick, as before, slides down at the default friction (−0.41 cm/s).

**The robot runs at the slip boundary at every friction.** The friction the
gait demands sits just under whatever is available (0.24 of 0.25, 0.45 of
0.5, 0.92 of 1.2). It does not keep a margin; spare grip becomes speed.

**The feedback rate no longer matters much.** With streamed setpoints, 0.2 s,
0.1 s and 0.02 s feedback give 3.56, 3.69 and 3.86 cm/s. The step-by-step
roll was the problem, not how often the form is re-estimated, so the paper's
10 Hz is enough.

**Roll speed is now the throttle.** Doubling `psi_dot` nearly doubles the
climb (6.96 cm/s); it used to collapse it, because it doubled the jump per
tick.

**Heavier is fine up to a cliff.** x2 mass climbs faster (4.07), x4 fails with
the motors saturated: the grip stops scaling before the weight does.

**Compliance needs a minimum.** K = 0.02 barely grips (5.5 contacts); K = 0.15
grips but saturates 7 of 18 motors for no gain. 0.08 is near the middle.

**Pole radius has hard limits on both sides, and both are the robot's
size.** At mu 0.5 a 2.5 cm pole gives too few contacts to carry the robot:
the links collide before the coil is tight enough. A 6 cm pole climbs, but
only between mu 0.4 and 0.5; from 7 cm the body winds barely once round the
pole and the coil tips off. See *Operating envelope* below. (An earlier
version of this paragraph said 6 cm does not climb; that came from the
pole-collision bug.)

**A flat helix does not climb on a slippery pipe.** alpha 0.15 stalls
(0.22 cm/s); 0.25–0.35 all work.

---

## Operating envelope

Some runs cannot succeed whatever the controller does, because the pole or
the friction is outside what this robot can physically do. Every run now
checks for that before it starts (`climb_rig.envelope`), prints the reason
in the run log, and writes `result.envelope` (`ok` / `marginal` /
`outside`), `result.envelope_turns` and `result.envelope_issues` to
`summary.csv`. The GUI asks before starting a run that is not `ok`. Nothing
is refused: a run outside the envelope is still a valid experiment, it just
fails for a reason that is not the method.

| limit | rule | measured |
| --- | --- | --- |
| wrap | predicted turns = (body length − end half-links) × cos α ÷ 2π(r + 3.4 cm); ok ≥ 1.15, outside < 1.05 | 5.5 cm (1.15) climbs at mu 0.4–1.0; 6 cm (1.09) only at 0.4–0.5; 7 cm (0.98) and above never. The formula matches the measured wrap within 0.02–0.09 turns wherever the coil grips |
| tightest coil | links collide before the backbone coils tighter than 6.4 cm radius, i.e. poles under ~3.0 cm; outside < 2.5 cm, marginal 2.5–3.5 cm | 2 cm never climbs; 2.5 cm only at mu ≥ 1.0; 3 cm at ≥ 0.5 |
| grip | outside mu < 0.25, marginal < 0.30 | below 0.25 the robot slides down even with rolling switched off |

Scored against all 195 placed-on-the-pole runs of the radius × friction
sweep: of runs it calls `ok`, 65 of 70 reach the top; of runs it calls
`outside`, 2 of 96 do (2.5 cm at mu 1.2 and 1.5). The five `ok` runs that
do not: 5 cm at mu 0.3 and 5.5 cm at mu 0.30–0.35 climb 2.0–2.2 m but not
to the top in 150 s, and 5.5 cm at mu 1.2–1.5 slide off. The last is the
same unexplained high-friction failure seen at 6 cm (coil tilt grows with
friction there); it is not modelled by the envelope.

For comparison, the robot in Takemori et al. (28 joints, 89.5 mm links,
a 300 mm tail link, 115 mm thick with its sponge rubber) is about 2.8 m
long and was only ever run on 150–250 mm pipes. Estimated the same way,
with the paper's own 55 mm link radius in place of this robot's offset,
that is roughly 2.4–3.3 turns. The paper states no size or friction limit and never gets
near one. This robot has 1.38 turns on its default 4 cm pole.

The constants (`BACKBONE_OFFSET`, `MIN_COIL_RADIUS`, the turn and friction
thresholds) were measured for this model. A different robot needs them
measured again.

## Holding on at low friction

Below mu 0.25 the robot does not fail to climb — it fails to hold on: with
rolling switched off it still slides down. That is odd at first sight: the
grip is ~78 N on a 7.2 N robot, so carrying the weight needs only ~0.09,
yet the contacts demand all 0.20 that is available.

One suspect was the compliance loop itself: re-squeezing every 0.1 s could
drag the links sideways along the pole and spend the friction the weight
needs. `python study_hold.py` tests that with the `freeze_form_at` switch,
which holds the joint targets from a given time. **It is not the cause.**
At mu 0.20 (the one friction where the robot slides without reaching the
floor inside the window) frozen slides 69 mm in 28 s against 76 mm
adaptive. Re-squeezing does double the friction the contacts demand at
mu 0.5 (0.36 vs 0.18), but that is not what lets the robot slide. Even
frozen, the contacts demand ~0.18, about twice what the weight needs, so
roughly half the tangential load is internal to the grip. Where it comes
from is still open. `freeze_form_at` is off by default and exists only for
this study.

## Pitch angle from the pole radius

The paper treats the pitch angle α as a design parameter and fixes it per
experiment (0.20–0.30). Measured here, the best α falls steadily as the
pole gets thicker, so **`alpha` now defaults to −1, meaning "choose it from
the pole radius"** (`climb_rig.alpha_for_radius`). Any value ≥ 0 — from the
panel, `--alpha 0.30`, or `Config(alpha=...)` — is used exactly as given, as
before. The value used is printed at the start of every run and written to
`config.alpha`, `result.alpha_used` and `result.alpha_source`; `config.json`
holds the resolved value, so it re-runs identically.

The rule is piecewise-linear through three points and held flat outside
them: 3 cm → 0.30, 4 cm → 0.25, 5.5 cm and above → 0.15. On the default 4 cm
pole it gives 0.25, so default runs are unchanged (checked: identical to
the digit).

It comes from `python sweep_pitch_radius.py` (α 0.10–0.30 × pole 3–8 cm ×
mu 0.3/0.5/0.8, placed on the pole, 120 runs, all passing the penetration
check; figure in `results/pitch_sweep_*/figures/`). At a fixed 0.25, 6 cm at
mu 0.8 and 6.5 cm at mu 0.5 slide off; at 0.15 both reach the top.

`python validate_pitch_rule.py` then ran the rule against fixed 0.25 at
3.5–7 cm, placed and from the floor (84 runs, all valid):

| | better | same | worse |
| --- | --- | --- | --- |
| placed on the pole | 9 | 12 | 0 |
| from the floor | 8 | 13 | 0 |

Read this honestly: **3.5, 4.2 and 4.5 cm were held out of the fit, and
there the rule is safe but gains little** (+0.0 to +0.2 cm/s). The large
gains are at 5–7 cm, which the rule was fitted on — though the from-the-floor
runs there are new (the sweep was placed-only) and show the same gains:
5 cm at mu 0.5 goes from stuck (0.01 cm/s) to the top, 6 cm at mu 0.8 from
sliding off to the top, 6.5 cm at mu 0.5 likewise.

What it does not do: it does not create more wrap (cos 0.15 / cos 0.25 is
only 2% more turns), so it does not move the size limits much — 7 cm still
does not reach the top, and 8 cm does not climb at any pitch. Why a flatter
helix holds a thick pole better is not established. A prediction that the
limit is where successive turns start to sit on each other was tested and
failed: α 0.10 climbs well below that line on 5–6.5 cm poles.

Sweeps made before this change (`results/sweep_20260929_002829`) used a
fixed 0.25; re-running `sweep_radius_friction.py` now uses the rule.

---

## Collision testbench

```
python test_collisions.py            full, ~5 min
python test_collisions.py --quick    ~1 min
```

Run it after any change to the model, `build_model`, or anything that edits
geometry. Exit code 0 means nothing in the simulation passes through
anything else. It checks four things, and each check is also run against a
planted fault it must catch, so a check that cannot fail cannot pass:

| check | what it guarantees | planted fault it must catch |
| --- | --- | --- |
| `what_you_see_collides` | every drawn geom also collides | a drawn pole with collision off |
| `pole_geometry` | the compiled pole's bounding volumes match its radius, at 2–10 cm | the old resize-after-compile build |
| `joints_do_not_blend` | every joint, swept over its full range, overlaps its neighbour no more than the CAD assembly does straight (real meshes, 0.3 mm voxels) | a link shifted 5 mm along its joint axis, every joint, both ways |
| `no_penetration_in_motion` | short runs at 2–10 cm, placed and from the floor: no link held more than 3 mm inside the pole, floor or another link for 50 ms, and no momentary spike over 10 mm | the old build on an 8 cm pole (36 mm held) |

Why these are needed: MuJoCo collides links by their convex hulls and never
collides a link with its own neighbour, and a size written into an
already-compiled model does not update its collision bounds. Until
2026-09-29 `build_model` did exactly that to the pole, so every pole larger
than 4 cm was partly non-colliding (robot links 17 mm inside a 6 cm pole,
38 mm inside an 8 cm one). Any result from before that date with
`pole_radius` above 0.04 is invalid — including the "pole radius 0.060" row
of the sensitivity table above. Against that old code this testbench fails
two of its four checks.

Touching is not a failure: a tight coil presses links against each other
and against the pole, as the real robot would. What is checked is parts
sitting *inside* each other.

`collision_check.py` holds the real-mesh measurement the testbench uses.

---

## Gotchas

- **The approach is re-measured for every config.** The start position comes
  from running the sidewinding gait on a throwaway copy of the model and
  mirroring the displacement, so friction and mass changes are accounted for.
  It costs about 0.7 s of wall clock per run and is the "measuring the start
  position" message.
- **`mu_floor` only affects the approach and the catch**, but it affects
  *where the robot ends up*, so it can change whether the pole is caught at
  all. The catch was tuned at the default 0.9.
- **A failed catch is not a failed method.** If `result.caught_pole` is
  False, the robot never got a verified grip — that is the catching stage,
  not the controller. The run's messages give the reason for every miss
  (outside the window, too few turns, axis tilted). Check `wrap_psi` first.
- **Blank cells in `run.csv`** before the loop starts are deliberate: there is
  no estimated axis or coil radius until the controller is running, and a
  placeholder would plot as real data.
- The rig aborts a run if `max_abs_qvel` exceeds 120, and records
  `result.diverged`. Everything after a solver blow-up is numerical garbage.
