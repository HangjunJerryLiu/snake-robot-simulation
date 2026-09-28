# Pole-climbing experiment rig

Written for: whoever runs or analyses these experiments next.

A snake robot climbing a vertical pole by **adaptive helical rolling** —
the method of Takemori, Tanaka & Matsuno, *IEEE Transactions on Robotics*
39(1):437–451, 2023 — set up so you can vary the things that matter, press
Start, and get a labelled CSV of everything that happened.

---

## Running it

```
python experiment_gui.py          control panel: friction slider, Start / Stop
python climb_rig.py               one run, headless, defaults
python climb_rig.py --viewer      one run, watch it (annotated, tracking camera)
python climb_rig.py --viewer --immersive false    the plain MuJoCo viewer
python climb_rig.py --mu 1.5 --alpha 0.30 --label soft_pads
python climb_rig.py --start_wrapped true    skip the approach; test the controller alone
python sweep_example.py           one parameter at a time, the README table
```

Every `Config` field is a `--flag`; `--mu`, `--mass`, `--dt`, `--fps` and
`--speed` are short aliases for the ones you reach for most.

It is also importable, which is how you sweep:

```python
from climb_rig import Config, run
for mu in (1.0, 1.5, 2.0, 2.5):
    print(mu, run(Config(mu_robot_pole=mu, label=f"mu{mu}"))['rolling_rate_cm_s'])
```

Each run creates `results/<timestamp>_<label>/` holding `run.csv`,
`joints.csv`, `stages.csv`, `summary.csv` and `config.json`. One example run
with the defaults is already there.

With the defaults the robot catches the pole at about t = 10 s and reaches the
top of the 3 m pole at about t = 132 s:

```
stage,      t_start, t_end,  duration, z_start, z_end,  climb,   rate cm/s
approach,     0.000,   9.917,   9.917,  0.0496, 0.0528, +0.0032,     0.033
catching,     9.917,  17.796,   7.879,  0.0528, 0.0972, +0.0443,     0.563
settling,    17.796,  20.797,   3.001,  0.0972, 0.0878, -0.0094,    -0.312
pitch_ramp,  20.797,  23.797,   3.000,  0.0878, 0.1569, +0.0691,     2.303
rolling,     23.797, 132.091, 108.294,  0.1569, 2.8002, +2.6434,     2.441
holding,    132.091, 137.092,   5.001,  2.8002, 2.7923, -0.0079,    -0.158
```

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

Measured, held constant for a whole run: mu 1.0 → 0.99 cm/s, 1.5 → 1.95,
2.0 → 2.40, 3.0 → 3.11. Dragged from 3.0 down to 0.9 mid-climb, the climb
rate collapses and `mu_required_p95` pins to whatever is available — that is
the robot sliding.

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

**The 3-D scene itself carries only what has to be in 3-D:**

| what you see | what it is |
| --- | --- |
| **faint orange body** | a ghost of the commanded form: forward kinematics of the target joint angles on the robot's actual base pose. The gap between ghost and robot is the tracking error the servos are turning into grip force |
| **height ticks** | every 0.5 m up the pole, labelled |

**Nothing is drawn sticking out of the body.** Contact-force arrows were
tried and made the wrap harder to read, not easier — a dozen arrows radiating
off a coiled body hide the coil. They're still one keypress away (`C`) when
they're the question. The cross-section used to be drawn as two rings
wrapped around the robot for the same reason it no longer is — it's the 2-D
plot above instead.

**A heads-up panel** (top right) carries the same numbers that are going into
`run.csv` that tick — stage, height, climb rate, coil radius, grip squeeze,
axis tilt, pole and self contacts, friction demanded against friction
available, and how many of the 18 motors are saturated. `Z` hides it.

Keys, on top of the viewer's own:

| key | |
| --- | --- |
| `B` | camera: follow → orbit → free |
| `N` | annotations: full → plot only → off (also gates the 2-D plot) |
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
floor fades into the sky rather than ending on a hard line. All of it is
inside `<visual>`, `<asset>` and `<light>`.

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
| `catching` | each joint pair ramps from the ground gait into the catching helix as it reaches the pole, so the body closes in the order it arrives | `n_contact_pole` climbing to ~15–20 |
| `settling` | all nine pairs wrapped, shape held still so contacts settle | height should be roughly flat; a drop means it is sliding |
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
| `mu_required_p95`, `mu_required_max` | – | at each pole contact, tangential force ÷ normal force: the friction coefficient that contact is *demanding*. **Compare with `mu_robot_pole`.** Baseline runs average 1.46 against the 2.0 available, so about a 1.4× margin before it slides |

**Whether the hardware could do this**

| column | unit | meaning |
| --- | --- | --- |
| `joint_torque_rms_Nm`, `joint_torque_max_Nm` | N·m | *delivered* joint torque, after MuJoCo clamps to the model's ±0.8 N·m |
| `joints_saturated` | – | how many of the 18 joints are asking for more than ±0.8 N·m. Typically 4–6 during the climb — these motors are working at their limit, which is a real constraint and not a simulation artefact |
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

**Solver and timing** — `timestep`, `control_period`, `max_duration`,
`target_speedup`, `target_fps`. `control_period` is the controller's rate, and
0.1 s is the paper's own hardware rate; `target_speedup`/`target_fps` are
display only and change nothing physical.

**Controller** — `alpha` (pitch angle of the climbing helix), `psi_dot` and
`spin` (rolling speed and direction), `k_mid`/`k_end` (compliance gain,
eq. 23), `lead_sign` (handedness). `spin` and `lead_sign` must match: `+1/+1`
and `−1/−1` climb, the mixed pairs slide off.

Note there is **no coil radius parameter for the climb**, and that is the
point of the method: the radius is whatever the pole leaves the robot at, and
the compliance term only ever asks for "a bit tighter than that". The same
numbers work on a pole of a different size.

**Start** — either `start_wrapped` with `start_radius` and `start_height`
(placed on the pole, like the paper), or the catching parameters
`wrap_alpha`, `wrap_radius`, `wrap_psi`, `alpha_ramp` and `stop_height`.
Catching is not in the paper, whose experiments all start with the robot
already on the pipe. A catching shape has different
requirements from a climbing shape: nearly flat (a steeply pitched helix
screws off the pole instead of closing around it) and tight. `wrap_psi` sets
**which plane** the ring closes in and is the single most sensitive number
here — at 0 the robot closes a ring standing up in a vertical plane and
misses the pole entirely.

---

## Measured parameter sensitivity

Nineteen runs, 60 s of simulated time each, **started placed on the pole** so
that what is being measured is the controller and not the floor gait. One
parameter changed at a time. `mu needed` and the contact counts are averages
over the `rolling` stage.

| case | rolling cm/s | climb m | pole contacts | mu needed (p95) | motors saturated |
| --- | --- | --- | --- | --- | --- |
| **baseline** | **2.40** | 1.37 | 11.5 | 1.46 | 4.2 |
| mu 1.0 | 0.99 | 0.57 | 9.6 | 0.78 | 3.9 |
| mu 1.5 | 1.95 | 1.11 | 10.3 | 1.11 | 4.1 |
| mu 3.0 | 3.11 | 1.77 | 12.9 | 2.03 | 4.1 |
| mass x0.5 | 1.47 | 0.84 | 12.7 | 1.33 | 3.5 |
| mass x2 | 2.89 | 1.65 | 9.8 | 1.48 | 4.6 |
| mass x4 | **-1.66** | -0.95 | 7.2 | 1.42 | 6.1 |
| dt 2 ms | 2.61 | 1.49 | 11.3 | 1.45 | 4.2 |
| dt 0.5 ms | 2.30 | 1.31 | 11.4 | 1.46 | 4.1 |
| control period 0.2 s | **0.03** | 0.02 | 10.1 | 1.33 | 4.4 |
| control period 0.02 s | **4.58** | 2.39 | 12.9 | 1.35 | 3.6 |
| alpha 0.15 | 2.61 | 1.49 | 9.7 | 1.38 | 4.6 |
| alpha 0.35 | 2.26 | 1.29 | 10.7 | 1.41 | 3.9 |
| psi_dot x0.5 | 1.85 | 1.05 | 13.1 | 1.21 | 3.4 |
| psi_dot x2 | **0.27** | 0.15 | 7.9 | 1.58 | 6.0 |
| K 0.02 | **0.06** | 0.03 | 4.8 | 1.27 | 1.2 |
| K 0.15 | **-0.56** | -0.32 | 1.6 | 0.22 | 7.8 |
| pole radius 0.025 | 2.82 | 1.61 | 5.8 | 1.32 | 5.3 |
| pole radius 0.060 | **-0.44** | -0.25 | 0.9 | 0.19 | 4.9 |

What this says:

**Friction sets the climb rate almost linearly** — 0.99, 1.95, 2.40, 3.11 cm/s
for mu 1.0, 1.5, 2.0, 3.0 — and the friction the gait *demands* tracks it at
roughly 0.7x whatever is available (0.78, 1.11, 1.46, 2.03). The controller
does not back off when it has grip to spare; it climbs faster and keeps
running near the slip boundary. There is no "safe" value of mu that gives a
big margin, only a faster or slower climb at about the same margin.

**The real timing parameter is roll per control tick**, `psi_dot x
control_period`, not either one alone. The baseline is 9 deg per tick.
Doubling `psi_dot` and doubling `control_period` both give 18 deg per tick and
both collapse the climb (0.27 and 0.03 cm/s). Cutting the control period to
0.02 s gives 1.8 deg per tick and the fastest climb measured, 4.58 cm/s — but
that is 50 Hz control, five times the rate of the paper's hardware, so it is a
simulation result rather than something the robot could do.

**The physics timestep barely matters** — 2.30 to 2.61 cm/s across a 4x range
of `dt`. Worth knowing: the climb is not a numerical artefact.

**Heavier is better, up to a cliff.** x2 mass climbs *faster* than nominal
(2.89 vs 2.40), because more weight means more normal force and more traction.
At x4 it fails outright, with 6 of 18 motors saturated against the +-0.8 N.m
limit: the grip the robot can generate stops scaling before its weight does.

**Compliance has a narrow window.** At K = 0.02 the robot barely grips (4.8
pole contacts against the baseline's 11.5) and hangs there; at K = 0.15 it
squeezes itself off the pole entirely (1.6 contacts, mu demand 0.22, i.e.
almost nothing left in contact) and slides down. 0.08 is near the middle.

**Pole radius is not a controller parameter, and it shows.** A 2.5 cm pole
works with the same numbers, at 2.82 cm/s. A 6 cm pole fails — but not because
the method cannot handle it: at that radius 72 cm of backbone makes only about
1.05 wraps, and a coil that barely closes once cannot hold. That is a limit of
this robot's length, not of the control law.

**alpha is the gentlest knob here**, 2.26 to 2.61 cm/s over 0.15 to 0.35 rad.
Which is the expected result: the pitch angle trades wraps against lead, and
those largely cancel.

---

## Gotchas

- **The approach is re-measured for every config.** The start position comes
  from running the sidewinding gait on a throwaway copy of the model and
  mirroring the displacement, so friction and mass changes are accounted for.
  It costs about 0.7 s of wall clock per run and is the "measuring the start
  position" message.
- **`mu_floor` only affects the approach**, but it affects *where the robot
  ends up*, so it can change whether the pole is caught at all.
- **A failed catch is not a failed method.** If `pairs_wrapped` reaches 9 but
  `n_contact_pole` stays near 0, the robot coiled up beside the pole — that is
  the catching stage, not the controller. Tune `wrap_psi` first.
- **Blank cells in `run.csv`** before the loop starts are deliberate: there is
  no estimated axis or coil radius until the controller is running, and a
  placeholder would plot as real data.
- The rig aborts a run if `max_abs_qvel` exceeds 120, and records
  `result.diverged`. Everything after a solver blow-up is numerical garbage.
