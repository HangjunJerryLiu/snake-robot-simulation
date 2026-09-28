"""
Ground-to-pipe climbing simulation, using the adaptive helical rolling method
of Takemori, Tanaka and Matsuno (IEEE T-RO 39(1):437-451, 2023).

The robot sidewinds across the floor to a vertical pole, wraps it as a normal
helix, and then climbs by HELICAL ROLLING: the commanded form stays a single
continuous helix at a fixed pitch angle, and the body rolls around that form
while the cross-sectional shape is continuously re-measured and re-squeezed to
whatever the pole is actually doing. The algorithm itself lives in
helical_rolling.py, which carries the equation-by-equation commentary; this
file is the ground gait, the hand-off into the helix, and the run loop.

WHAT CHANGED, AND WHY
---------------------------------------------------------------------------
The previous version of this file climbed with an open-loop "breathing coil":
all nine joint pairs were driven in unison through a phase cycle whose coil
radius grew and shrank, and the climb came from that clamp/release ratchet
plus the body rolling lumpily against the pole. It worked, but it was not the
paper's method and it had two real problems -- the shape was a fixed family of
circles that could not adapt to anything the pole did, and the ratchet leaned
on parts of the body pressing against each other, so self-collision was doing
some of the lifting.

The method here has neither property:

  * the target form is a helix at ALL times, never a squeezing circle, so the
    only thing touching the pole is the wrap itself;
  * the cross-sectional shape is measured from the joint angles every control
    tick and deformed to fit whatever it found, so the level of locality is
    the number of joints (18) rather than 1 -- each joint adapts to the piece
    of pole in front of it, which is the paper's central claim (its Fig. 2);
  * compliance acts ONLY perpendicular to the pole. The pitch angle alpha is a
    fixed design constant and never adapts, so tightening the grip cannot also
    change the lead of the helix and let the robot slide.

Climbing comes from psi_roll in eq. (33) alone: advancing it rolls the whole
body around an unchanged target form, and the wrap screws itself up the pole.

MEASURED, not assumed  (climb_from_wrapped.py mechanism, 67 s of climbing)
---------------------------------------------------------------------------
  robot-on-pole contacts   11.5 per sample on average, peak 24
  robot-on-robot contacts   0.5 per sample, and 61% of samples have none
  body roll about its own   95% of the commanded psi_roll -- it really rolls,
    axis                    rather than ratcheting
  orbit around the pole     1.9 revolutions against 16.8 of roll -- it screws
                            upward while staying put around the pole
  climb per psi_roll turn   9.65 cm against an ideal screw lead of 11.93 cm,
                            so 81% of the ideal and 19% slip

Full run, ground to the top: 2.71 m of climb at 2.44 cm/s.

FILES
---------------------------------------------------------------------------
  helical_rolling.py     the method, step by step, with equation references
  snake_backbone.py      link-edge points and link lengths out of the model
  calibrate_helical.py   solves the sign convention eq. (32) takes here
  climb_from_wrapped.py  the loop on its own, from a wrapped start
"""

import sys
import time

import mujoco
import numpy as np

import helical_rolling as hr
from snake_backbone import Backbone

HEADLESS = '--headless' in sys.argv


def deg2rad(d):
    return d * np.pi / 180


# ─────────────────────────────────────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────────────────────────────────────
model = mujoco.MjModel.from_xml_path('9motor_sidewinder_pipe.xml')
data = mujoco.MjData(model)

N = 9      # number of yaw/pitch joint pairs along the body
NJ = 18    # joints

backbone = Backbone(model)

# ─────────────────────────────────────────────────────────────────────────────
# PIPE GEOMETRY  (must match the XML values)
# ─────────────────────────────────────────────────────────────────────────────
PIPE_RADIUS = 0.04    # m  – cylinder radius in XML: size="0.04 1.5"
PIPE_X = 0.0          # m  – pipe axis at world origin
PIPE_Y = 0.0

# Where the pole actually holds the robot, measured over a climb: the coil
# settles at 7.4 cm of radius and stays there. Nothing in the controller sets
# that number -- the compliance term only ever asks for "tighter than now",
# and this is simply where the pole stops it. It is quoted so the run log has
# something to compare against.
#
# It is smaller than the pole radius (4.0 cm) plus the fattest part of the
# body (4.3 cm, the L links' convex hull, which is what MuJoCo collides), and
# that is not a contradiction: the coil radius is measured to the JOINT
# CENTRES, which sit inside the links, while contact happens out at the hull.
SETTLED_RADIUS = 0.074

# One representative body per joint pair (n = 1..9), used only to notice when
# a pair has arrived at the pole during the approach.
PAIR_BODY_IDS = [model.body(f'M{n}').id for n in range(1, N + 1)]


def radial_distance(xy):
    """Distance from an (x, y) point to the pipe surface."""
    return np.linalg.norm(np.asarray(xy) - np.array([PIPE_X, PIPE_Y])) - PIPE_RADIUS


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 GAIT – GROUND APPROACH (flat-ground sidewinding)
# ─────────────────────────────────────────────────────────────────────────────
#
# Unchanged from before, and unrelated to the paper: the same zero-mean
# traveling-wave serpenoid gait validated in robot_command.py (0.9 m of travel
# over 9.5 s). No DC bias, so it sidewinds across the floor and does not wrap
# around anything by itself. Its only job is to carry the snake from its
# starting spot on the ground to the pole.
AY_APPROACH = deg2rad(45)
AP_APPROACH = deg2rad(30)
OMEGA_APPROACH = 2.0
NY_APPROACH = NPITCH_APPROACH = 1.5
DELTA_Y_APPROACH = DELTA_P_APPROACH = -4.202
DELTA_D_APPROACH = 0.5 * np.pi


def approach_cmd(t):
    joints = np.zeros(NJ)
    for joint_idx in range(NJ):
        joint_num = joint_idx + 1
        if joint_num % 2 == 1:          # yaw
            n = (joint_num + 1) // 2
            joints[joint_idx] = AY_APPROACH * np.sin(
                OMEGA_APPROACH * t + 2 * np.pi * n * (NY_APPROACH / N)
                + DELTA_Y_APPROACH + DELTA_D_APPROACH)
        else:                            # pitch
            n = joint_num // 2
            joints[joint_idx] = AP_APPROACH * np.sin(
                OMEGA_APPROACH * t + 2 * np.pi * n * (NPITCH_APPROACH / N)
                + DELTA_P_APPROACH)
    return joints


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2/3 – ADAPTIVE HELICAL ROLLING  (the paper's method)
# ─────────────────────────────────────────────────────────────────────────────
#
# THE FOUR NUMBERS THAT MATTER, and what each one does
# ───────────────────────────────────────────────────────────────────────────
# ALPHA       The pitch angle of the target helix -- the angle between the
#             body and the plane perpendicular to the pole. It is a design
#             parameter, held constant, and holding it constant is the whole
#             trick: compliance is then confined to the cross-sectional shape,
#             i.e. to the direction perpendicular to the pole, so squeezing
#             harder can never change the lead of the wrap. Bigger alpha means
#             a steeper wrap, fewer turns around the pole and less of the body
#             in contact; smaller alpha means more wraps but a flatter screw.
#             The paper used 0.20-0.30 rad across its four experiments.
#
#             Nothing in the loop ever sets the coil RADIUS: the radius is
#             whatever the pole leaves the robot at, and the compliance term
#             below only ever asks for "a bit tighter than that". So there is
#             no radius parameter here at all, which is the point -- it is what
#             lets the same numbers work on a pole of any size.
#
# K           Compliance gain, eq. (23). Each tick the controller commands a
#             cross-section whose radius is (1-K) times the one it just
#             measured -- "tighten a little more than you currently are". The
#             pole stops the robot from getting there, and the shortfall is
#             what becomes grip force through the position servos. K is per
#             joint, and following the paper it is halved over the last few
#             joints at each end: those ends are cantilevered off the pole, so
#             a full-strength squeeze there just curls them into the air.
#
# PSI_DOT     Rolling speed, the rate psi_roll advances in eq. (33). This is
#             the ONLY thing that produces climbing. The target form does not
#             change as psi_roll advances; the body rolls around it, and the
#             wrap screws itself along the pole. The paper used pi/2 rad/s.
#             SPIN picks which way.
ALPHA = 0.25                 # rad – pitch angle of the target helix
GAMMA_MIN = 0.01 * np.pi     # rad – case-D threshold, eq. (22); paper's value
PSI_DOT = 0.5 * np.pi        # rad/s – rolling velocity, eq. (33)
SPIN = +1                    # +1 climbs, -1 descends

# Compliance gain per joint, in the robot's own J1..J18 order. The paper used
# 0.04 over the outermost four joints of 28 and 0.08 in between; the same
# fraction of 18 joints is three at each end.
K_GAIN = np.full(NJ, 0.08)
K_GAIN[:3] = 0.04
K_GAIN[-3:] = 0.04

# Handedness of the wrap. Either sign wraps the pole perfectly well; it decides
# which way the body spirals, and together with SPIN, which way rolling drives
# it. Measured from a pre-wrapped start, 38 s of rolling each:
#     SPIN +1, LEAD +1 : +0.95 m      SPIN +1, LEAD -1 : -0.38 m
#     SPIN -1, LEAD -1 : +0.93 m      SPIN -1, LEAD +1 : -0.34 m
# i.e. the two matching pairs climb and the two mixed pairs slide off. The
# matching pairs are mirror images of each other, so either will do.
LEAD_SIGN = +1.0

# The paper's snake robot runs its control loop at 10 Hz (0.1 s sampling), and
# so does this. Everything in between is held, exactly as on hardware.
CONTROL_PERIOD = 0.1         # s

# ── Catching the pole in the first place ────────────────────────────────────
#
# The paper does not address this: in all four of its experiments the robot
# STARTS wrapped around the pipe as a normal helix, placed there by hand, and
# the method begins from that. Here the robot has to get there off the floor,
# so the shape it closes with is its own parameter set -- still a normal helix
# from the same eq. (32) machinery, just a different helix from the one it
# climbs with.
#
# What a catching shape needs is different from what a climbing shape needs.
# It has to be a nearly FLAT ring (a steeply pitched helix screws off the pole
# instead of closing around it) and TIGHT (looser than the pole plus the body
# and the robot hangs off rather than gripping). Swept over pitch angle, radius
# and roll phase, counting how far the closed body actually sweeps around the
# pole axis and how many pole contacts it ends up with:
#
#     pitch  radius  psi     sweep   contacts   axis vertical?
#      0.10   0.065  pi/2   -1.37 turns   19      yes  <- used here
#      0.10   0.075  pi/2   -1.42 turns   12      yes
#      0.02   0.065  pi/2   -1.16 turns   12      yes
#      0.25   0.065  0      -0.07 turns    0      no   (screws off)
#      0.10   0.065  0      -0.08 turns    0      no   (closes the wrong way)
#
# PSI_WRAP matters as much as the shape: it sets WHICH PLANE the ring closes
# in. At psi = 0 this robot closes a ring standing up in a vertical plane,
# which simply misses the pole; at pi/2 it closes a ring lying in a horizontal
# plane, which catches it.
WRAP_ALPHA = 0.10            # rad – nearly flat, so the ring closes around the pole
WRAP_R0 = 0.065              # m   – tighter than the pole plus the body, so it grips
PSI_WRAP = 0.5 * np.pi       # rad – roll phase, i.e. which plane the ring closes in

# Once caught, the pitch angle is opened from WRAP_ALPHA up to the climbing
# ALPHA over this long, rather than jumped, so the body stretches into its
# working helix while it is already holding on.
ALPHA_RAMP = 3.0             # s

controller = hr.HelicalRollingController(
    backbone.link_lengths, alpha=WRAP_ALPHA, K=K_GAIN,
    mode='outside',          # the robot wraps the OUTSIDE of the pole
    gamma_min=GAMMA_MIN,
)

# The form the robot first closes around the pole, built through the same
# eq. (32) approximation the adaptive loop uses, so handing over to the loop
# changes nothing about how the target is expressed.
WRAP_TARGET = controller.normal_helix_target(WRAP_R0, PSI_WRAP, lead_sign=LEAD_SIGN)


# ─────────────────────────────────────────────────────────────────────────────
# STARTUP DIAGNOSTIC – what shape is actually being commanded
# ─────────────────────────────────────────────────────────────────────────────
#
# Unlike the old breathing-coil gait, there is no "does it breathe enough"
# question to answer here: the target form is a helix and stays one. What is
# worth checking before a run is that the helix eq. (32) produces is the helix
# that was asked for -- with 18 joints over roughly one turn the approximation
# is a polygon, and how good a polygon matters. So measure it: put the robot in
# the commanded pose, run forward kinematics, and fit a helix to the result.
def _report_wrap_shape():
    probe = mujoco.MjData(model)
    mujoco.mj_resetData(model, probe)
    probe.qpos[3] = 1.0
    probe.qpos[7:7 + NJ] = WRAP_TARGET
    mujoco.mj_forward(model, probe)
    return hr.fit_helix(backbone.nodes(model, probe))


_fit = _report_wrap_shape()
print("=" * 70)
print("TARGET FORM  (adaptive helical rolling, Takemori et al. 2023)")
print("=" * 70)
print(f"  robot         : {backbone.n_link} links, {backbone.n_joint} joints, "
      f"{backbone.link_lengths.sum()*100:.1f} cm of backbone")
print(f"                  link lengths alternate "
      f"{backbone.link_lengths[1]*1000:.0f}/{backbone.link_lengths[2]*1000:.0f} mm, "
      f"so the paper's single l is a per-link array here")
print(f"  pole          : radius {PIPE_RADIUS*100:.1f} cm; a climbing run settles at "
      f"about {SETTLED_RADIUS*100:.1f} cm of coil radius")
print(f"  catching form : normal helix, r = {WRAP_R0*100:5.2f} cm at pitch angle "
      f"{WRAP_ALPHA:.2f} rad, roll phase {PSI_WRAP:.2f} rad")
print(f"  what eq. (32) : r = {_fit['radius']*100:5.2f} cm, pitch angle "
      f"{_fit['pitch_angle']:.2f} rad, {abs(_fit['turns']):.2f} turns")
print(f"    actually got  fit residual {_fit['residual']*1000:.1f} mm, from approximating a "
      f"helix with only {backbone.n_joint/abs(_fit['turns']):.0f} joints per turn.")
print( "                  The catching form is the coarsest shape in the run -- a tight")
print( "                  coil -- so its pitch angle comes out well off what was asked,")
print( "                  which does not matter: all it has to do is close on the pole.")
print(f"  climbing form : pitch angle opened to {ALPHA:.2f} rad over "
      f"{ALPHA_RAMP:.0f} s once the pole is caught")
print(f"  compliance    : K = {K_GAIN.min():.2f} at the ends, {K_GAIN.max():.2f} "
      f"in the middle, so each 0.1 s tick")
print(f"                  asks for a cross-section {K_GAIN.max()*100:.0f}% tighter "
      f"than the one measured")
print(f"  rolling       : psi_roll advances {SPIN*PSI_DOT:+.2f} rad/s -- the only "
      f"term that climbs")
if abs(_fit['turns']) < 1.0:
    print("  WARNING: the catching form is less than one full wrap, so it cannot")
    print("           close around the pole. Lower WRAP_R0 or WRAP_ALPHA.")


# ─────────────────────────────────────────────────────────────────────────────
# INITIAL POSE  (flat on the ground, near the pipe -- NOT wrapped)
# ─────────────────────────────────────────────────────────────────────────────
#
#  data.qpos layout
#  ──────────────────────────────────────────────────────────────────────────
#  [0..2]   tail x, y, z (world)      [3..6]  free-joint quaternion
#  [7..24]  J1..J18 joint angles
GROUND_Z = 0.05   # m -- lying flat, matches the flat-ground gait's own height
theta = deg2rad(90)
Q_LYING_FLAT = np.array([np.cos(theta / 2), np.sin(theta / 2), 0.0, 0.0])

# The sidewinding gait's travel path is a curved track, not a straight line
# along the body axis, so the start position is measured rather than derived:
# run the approach gait on a throwaway MjData seeded far from the pipe (so
# nothing is contaminated by contact with it), see where the tail ends up
# after T_PROBE seconds, and start the real robot at the mirror image of that
# displacement. The body then sweeps into the pipe partway through.
_PROBE_START = np.array([3.0, 3.0])   # far from the pipe; floor physics is translation-invariant
T_PROBE = 10.5                        # s

_placement_probe = mujoco.MjData(model)
mujoco.mj_resetData(model, _placement_probe)
_placement_probe.qpos[0:2] = _PROBE_START
_placement_probe.qpos[2] = GROUND_Z
_placement_probe.qpos[3:7] = Q_LYING_FLAT
mujoco.mj_forward(model, _placement_probe)
_tail0 = _placement_probe.qpos[0:2].copy()
_probe_dt = model.opt.timestep
for _step in range(int(T_PROBE / _probe_dt)):
    _placement_probe.ctrl[:NJ] = approach_cmd(_step * _probe_dt)
    mujoco.mj_step(model, _placement_probe)
_tail_displacement = _placement_probe.qpos[0:2].copy() - _tail0

mujoco.mj_resetData(model, data)
data.qpos[0:2] = -_tail_displacement
data.qpos[2] = GROUND_Z
data.qpos[3:7] = Q_LYING_FLAT
mujoco.mj_forward(model, data)

# ─────────────────────────────────────────────────────────────────────────────
# VIEWER
# ─────────────────────────────────────────────────────────────────────────────
dt = model.opt.timestep

viewer = None
if not HEADLESS:
    import mujoco_viewer
    viewer = mujoco_viewer.MujocoViewer(model, data)

# Rendering, not physics, decides how long this takes to watch. Measured here:
# physics costs 0.06 s of wall clock per simulated second (16x faster than
# real time), while one viewer frame costs ~48 ms -- and that cost does not
# change with window size, so it is vsync/driver bound, not fill-rate bound.
# Pace the frames rather than rendering every Nth step blindly: one frame every
# RENDER_EVERY steps gives the playback speed below.
TARGET_SPEEDUP = 3.0    # simulated seconds to play per wall-clock second
TARGET_FPS = 15.0       # frames per wall-clock second
RENDER_EVERY = max(1, int(round(TARGET_SPEEDUP / (TARGET_FPS * dt))))

print()
print("=" * 70)
print("SNAKE ROBOT - GROUND-TO-PIPE CLIMBING SIMULATION")
print("=" * 70)
print(f"Pipe          : radius {PIPE_RADIUS*100:.0f} cm, 3.0 m tall, standing on the floor")
print(f"Robot start   : x={data.qpos[0]:.3f} m, y={data.qpos[1]:.3f} m, "
      f"z={data.qpos[2]:.3f} m (lying flat on the ground)")
print("\nPhases: 1) sidewind across the ground to the pole    (to t~8 s)")
print("        2) close onto it as a normal helix            (to t~33 s)")
print("        3) adaptive helical rolling -- measure, squeeze, roll, repeat")
if not HEADLESS:
    print("Close the viewer window to stop.")

# ─────────────────────────────────────────────────────────────────────────────
# STATE MACHINE:  approach -> wrapping -> climb
# ─────────────────────────────────────────────────────────────────────────────
PROX_PAIR_THRESHOLD = 0.06   # m – a pair starts curling once its body is this close to the pipe
RAMP_DURATION_PAIR = 1.5     # s – how long one pair takes to go from wave to wrap
FORCE_WRAP_AFTER = 20.0      # s – a pair that never got close enough curls anyway, so the
                             #     coil closes instead of leaving a slack gap in the loop
SETTLE_HOLD = 3.0            # s – hold the closed helix still, letting contacts settle and
                             #     letting the compliance take up the slack, before rolling
APPROACH_TIMEOUT = 40.0      # s – safety cutoff if the pipe is never reached
CLIMB_DURATION = 180.0       # s – how long to keep rolling. Measured climb rate
                             #     is about 2.6 cm/s, so 3 m of pole needs ~115 s.

# Stop rolling near the top of the pole and just hold on, instead of driving on
# and sliding off the end. The pole is 3.0 m tall and the wrapped body spans
# roughly 0.15 m of it. Measured from a pre-wrapped start: the robot was still
# gripping with the centre of mass at 2.94 m and had come off the top by 3.01 m,
# so freeze a little below that.
STOP_HEIGHT = 2.80           # m

phase = 'approach'
psi_roll = PSI_WRAP
psi_frozen = False
top_reached_time = None
pair_trigger_time = [None] * N
wrapping_start_time = None
all_locked_time = None
climb_start_time = None
cmd = np.zeros(NJ)
last_control_t = -1e9
last_info = None

time_history = []
com_z_history = []

_total_sim = 33.0 + CLIMB_DURATION
print()
if not HEADLESS:
    print(f"Playback : ~{TARGET_SPEEDUP:.0f}x real time, one frame every {RENDER_EVERY} steps,")
    print(f"           {_total_sim:.0f} simulated seconds in roughly "
          f"{_total_sim/TARGET_SPEEDUP/60:.1f} minutes of watching.")
print()

max_steps = int((APPROACH_TIMEOUT + FORCE_WRAP_AFTER + CLIMB_DURATION + 40.0) / dt)
_wall_start = time.perf_counter()
_sim_end = 0.0
viewer_ok = viewer is not None
diverged = False

# Divergence threshold, set from measured behaviour rather than guessed. The
# approach transient legitimately reaches ~85; a solver blow-up goes past 150.
QVEL_LIMIT = 120.0

try:
    for step in range(max_steps):
        t = step * dt
        a_cmd = approach_cmd(t)

        if phase == 'approach':
            cmd = a_cmd
            for n in range(1, N + 1):
                if pair_trigger_time[n - 1] is None and \
                        radial_distance(data.xpos[PAIR_BODY_IDS[n - 1]][:2]) < PROX_PAIR_THRESHOLD:
                    pair_trigger_time[n - 1] = t
            if any(tt is not None for tt in pair_trigger_time):
                phase = 'wrapping'
                wrapping_start_time = t
                print(f"PHASE 2: reached the pole at t={t:.2f}s -- closing into a helix...")
            elif t > APPROACH_TIMEOUT:
                print(f"\nWARNING: never reached the pole in {APPROACH_TIMEOUT:.0f}s "
                      f"-- check T_PROBE / the placement measurement.")
                phase = 'wrapping'
                wrapping_start_time = t

        elif phase == 'wrapping':
            # Ramp each joint pair from the ground gait into the normal helix
            # as it arrives at the pole, so the body closes onto the pole in
            # the order it reaches it rather than snapping shut all at once.
            cmd = np.zeros(NJ)
            for n in range(1, N + 1):
                if pair_trigger_time[n - 1] is None:
                    if radial_distance(data.xpos[PAIR_BODY_IDS[n - 1]][:2]) < PROX_PAIR_THRESHOLD:
                        pair_trigger_time[n - 1] = t
                    elif (t - wrapping_start_time) > FORCE_WRAP_AFTER:
                        pair_trigger_time[n - 1] = t
                if pair_trigger_time[n - 1] is None:
                    alpha_ramp = 0.0
                else:
                    alpha_ramp = min(1.0, (t - pair_trigger_time[n - 1]) / RAMP_DURATION_PAIR)
                yi, pidx = 2 * n - 2, 2 * n - 1
                cmd[yi] = (1 - alpha_ramp) * a_cmd[yi] + alpha_ramp * WRAP_TARGET[yi]
                cmd[pidx] = (1 - alpha_ramp) * a_cmd[pidx] + alpha_ramp * WRAP_TARGET[pidx]

            fully_locked = all(tt is not None for tt in pair_trigger_time) and all(
                (t - tt) >= RAMP_DURATION_PAIR for tt in pair_trigger_time)
            if fully_locked and all_locked_time is None:
                all_locked_time = t
                print(f"  all {N} joint pairs wrapped at t={t:.2f}s -- settling...")
            if all_locked_time is not None and (t - all_locked_time) > SETTLE_HOLD:
                phase = 'climb'
                climb_start_time = t
                last_control_t = -1e9
                print(f"\nPHASE 3: helical rolling starts at t={t:.2f}s\n")

        else:
            # ── The paper's control loop, run at its own 10 Hz sampling rate ──
            #
            #   1) read the form           backbone.nodes() -- eq. (1)-(4)
            #   2-7) everything else       controller.target()
            #
            # psi_roll advances continuously between ticks; the target form it
            # rolls around is whatever the last tick measured and squeezed.
            # Open the pitch angle from the catching helix to the climbing
            # one. Everything else about the loop is unchanged while this
            # happens -- it is still measuring and squeezing every tick, so the
            # robot stretches out while already holding on.
            controller.alpha = WRAP_ALPHA + (ALPHA - WRAP_ALPHA) * min(
                1.0, (t - climb_start_time) / ALPHA_RAMP)

            if not psi_frozen:
                if data.subtree_com[0, 2] < STOP_HEIGHT:
                    # psi continues from where the catching form left it, so
                    # the body does not lurch at the hand-off.
                    psi_roll = PSI_WRAP + SPIN * PSI_DOT * (t - climb_start_time)
                else:
                    psi_frozen = True
                    top_reached_time = t
                    print(f"\nReached the top of the pole at t={t:.1f}s "
                          f"(height {data.subtree_com[0, 2]:.2f} m) -- holding on.")

            if (t - last_control_t) >= CONTROL_PERIOD:
                last_control_t = t
                nodes = backbone.nodes(model, data)
                cmd, last_info = controller.target(nodes, psi_roll)
                cmd = np.clip(cmd, model.jnt_range[1:1 + NJ, 0],
                              model.jnt_range[1:1 + NJ, 1])

            if top_reached_time is not None and (t - top_reached_time) > 5.0:
                print("Done -- stopping at the top.")
                break
            if (t - climb_start_time) > CLIMB_DURATION:
                print("\nClimb duration complete.")
                break

        data.ctrl[:NJ] = cmd
        mujoco.mj_step(model, data)
        _sim_end = t

        if not np.all(np.isfinite(data.qpos)) or np.max(np.abs(data.qvel)) > QVEL_LIMIT:
            print(f"\nSTOPPED at t={t:.2f}s: the physics diverged "
                  f"(max |qvel| = {np.max(np.abs(data.qvel)):.0f}, limit {QVEL_LIMIT:.0f}).")
            print("Nothing after this point would be meaningful, so the run ends here.")
            diverged = True
            break

        if viewer_ok and viewer.is_alive and step % RENDER_EVERY == 0:
            try:
                viewer.render()
            except Exception as exc:                     # window died mid-frame
                print(f"\nViewer stopped rendering ({type(exc).__name__}). "
                      f"Finishing the run headless.")
                viewer_ok = False

        if phase == 'climb':
            time_history.append(t - climb_start_time)
            com_z_history.append(float(data.subtree_com[0, 2]))

        if step % 2000 == 0:
            if phase == 'climb' and last_info is not None:
                # r_meas is the coil radius the pole is actually holding the
                # robot at; r_cmd is what the compliance term just asked for.
                # The gap between them IS the grip: it is the deflection the
                # position servos are pressing into the pole.
                print(f"t={t:6.1f}s | climb    | height={data.subtree_com[0, 2]:.3f} m "
                      f"| r={last_info['radius_measured']*100:5.2f}cm "
                      f"-> {last_info['radius_target']*100:5.2f}cm "
                      f"| {last_info['turns']:.2f} turns "
                      f"| axis tilt {np.degrees(np.arccos(np.clip(abs(last_info['axis'][2]), 0, 1))):4.1f} deg")
            else:
                n_wrapped = sum(1 for tt in pair_trigger_time if tt is not None)
                print(f"t={t:6.1f}s | {phase:8s} | height={data.subtree_com[0, 2]:.3f} m "
                      f"| wrapped={n_wrapped}/{N}")

        if viewer_ok and not viewer.is_alive:
            print("\nViewer closed by user.")
            break

except KeyboardInterrupt:
    print("\nInterrupted by user.")

if viewer is not None:
    try:
        viewer.close()
    except Exception:
        pass

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
time_history = np.array(time_history)
com_z_history = np.array(com_z_history)

print("\n" + "=" * 70)
print("SIMULATION COMPLETE")
print("=" * 70)
if len(com_z_history) > 10:
    # Longest sustained ascent (the robot may climb, slip back, and climb again,
    # so a single start-to-end difference understates what it actually did).
    best_span, best_rate, lo = 0.0, 0.0, 0
    for i in range(1, len(com_z_history)):
        if com_z_history[i] < com_z_history[lo]:
            lo = i
        if com_z_history[i] - com_z_history[lo] > best_span:
            best_span = com_z_history[i] - com_z_history[lo]
            span_t = time_history[i] - time_history[lo]
            best_rate = best_span / span_t if span_t > 0 else 0.0
    print(f"Climb-phase duration  : {time_history[-1]:.1f} s")
    print(f"Highest point reached : {com_z_history.max():.3f} m")
    print(f"Longest single climb  : {best_span:.3f} m at {best_rate*100:.2f} cm/s")
    print(f"Height at end         : {com_z_history[-1]:.3f} m")
else:
    print("Climb phase never started -- the robot did not wrap the pole.")
_wall = time.perf_counter() - _wall_start
if _wall > 0 and _sim_end > 0:
    print(f"Playback speed        : {_sim_end/_wall:.1f}x real time "
          f"({_sim_end:.0f} simulated s in {_wall/60:.1f} min of wall clock)")
print("=" * 70)
print()
print("TUNING GUIDE  (all four knobs are in the PHASE 2/3 block above)")
print("  - Wraps but will not lift  -> flip SPIN, or flip LEAD_SIGN. Those two")
print("    together set which way the screw turns; one combination climbs.")
print("  - Climbs then slides back  -> the grip is marginal. Raise K, or lower")
print("    ALPHA so more of the body lies against the pole per turn.")
print("  - Squeezes itself off the pole, or the ends flail -> lower the end")
print("    values in K_GAIN; cantilevered ends curl into the air if squeezed.")
print("  - 'r=... -> ...' in the log shows the measured coil radius and the one")
print("    the compliance asked for. If they are equal the robot is not gripping")
print("    at all; if the gap is much larger than K the wrap has come off.")
print("  - Fewer than 1.0 turns in the log -> the coil cannot hold. Lower ALPHA.")
print("  - Never reaches the pole   -> adjust T_PROBE.")
print("  - Reaches the pole but does not catch it -> that is WRAP_ALPHA/WRAP_R0/")
print("    PSI_WRAP, not the climbing parameters. PSI_WRAP decides which plane")
print("    the ring closes in and is the one that most often needs changing.")
