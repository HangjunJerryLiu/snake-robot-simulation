import time

import mujoco
import mujoco_viewer
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────
def deg2rad(d):
    return d * np.pi / 180


# ─────────────────────────────────────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────────────────────────────────────
model = mujoco.MjModel.from_xml_path('9motor_sidewinder_pipe.xml')
data  = mujoco.MjData(model)

N = 9   # number of motor (yaw/pitch) pairs along the body

# ─────────────────────────────────────────────────────────────────────────────
# PIPE GEOMETRY  (must match the XML values)
# ─────────────────────────────────────────────────────────────────────────────
PIPE_RADIUS = 0.04    # m  – cylinder radius in XML: size="0.04 1.5"
PIPE_X      = 0.0     # m  – pipe axis at world origin
PIPE_Y      = 0.0

BODY_NAMES = ['tail']
for i in range(1, 9):
    BODY_NAMES += [f'M{i}', f'L{i}']
BODY_NAMES += ['M9', 'head']
BODY_IDS = [model.body(n).id for n in BODY_NAMES]

# One representative body per yaw/pitch joint pair (n = 1..9). M_n carries the
# yaw joint of pair n, so it is the leading body of that pair as the snake
# advances toward the pipe.
PAIR_BODY_IDS = [model.body(f'M{n}').id for n in range(1, N + 1)]


def radial_distance(xy):
    """Distance from an (x, y) point to the pipe surface."""
    return np.linalg.norm(np.asarray(xy) - np.array([PIPE_X, PIPE_Y])) - PIPE_RADIUS


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 GAIT – GROUND APPROACH (flat-ground sidewinding)
# ─────────────────────────────────────────────────────────────────────────────
#
# The same zero-mean traveling-wave serpenoid gait validated in
# robot_command.py (0.9 m of travel over 9.5 s). No DC bias, so it sidewinds
# across the floor and does not wrap around anything by itself. Its only job
# is to carry the snake from its starting spot on the ground to the pipe.
AY_APPROACH = deg2rad(45)
AP_APPROACH = deg2rad(30)
OMEGA_APPROACH = 2.0
NY_APPROACH = NPITCH_APPROACH = 1.5
DELTA_Y_APPROACH = DELTA_P_APPROACH = -4.202
DELTA_D_APPROACH = 0.5 * np.pi


def approach_cmd(t):
    joints = np.zeros(18)
    for joint_idx in range(18):
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
# PHASE 2/3 GAIT – WRAP THE POLE, THEN CLIMB BY CLAMPING AND EXTENDING
# ─────────────────────────────────────────────────────────────────────────────
#
#  WHAT THIS GAIT ACTUALLY DOES  (measured, not assumed)
#  ───────────────────────────────────────────────────────────────────────
#  Every joint pair is driven in unison (XI = 0): all yaw joints follow
#  A_YAW*sin(phase), all pitch joints A_PITCH*sin(phase + DELTA). Because the
#  two amplitudes differ and the offset is not 90 deg, the body does not hold
#  one fixed coil -- it cycles through a family of coils whose radius breathes
#  over one phase cycle.
#
#  Touching the pole at all takes roughly 8.3 cm of coil radius: 4.0 cm of
#  pole plus about 4.3 cm of body. (That 4.3 cm is the L link's CONVEX HULL,
#  not its drawn shape -- MuJoCo collides the hull of a mesh, and these
#  clevis-shaped links have hulls 6.4x their drawn volume. It is why only
#  about 3 of the 20 bodies actually reach the pole at any moment, and why
#  the snake can look like it is not touching.) The breathing cycle varies
#  how hard those few contacts are pressed -- a clamp / extend / re-clamp
#  squeeze cycle on top of the body roll described next.
#
#  DOES IT ROLL?  YES -- measured two ways, which disagree until you are
#  careful about which rotation you mean:
#
#    about its OWN long axis : -0.86 revolutions per gait cycle. The body
#                              really is rolling, at close to the commanded
#                              one-revolution-per-cycle.
#    about the POLE axis     : 0.003-0.012 revolutions per gait cycle, i.e.
#                              essentially nothing. It does not orbit.
#
#  Both are correct and both are expected: a helix rolling against a pole
#  screws itself upward while staying at the same angular position around
#  the pole. (An earlier version of this comment claimed the robot "climbs
#  almost without rotating" -- that was measuring the pole-axis number and
#  was wrong about the mechanism.)
#
#  What this robot does NOT do is roll as a rigid helix. Fitting a rigid
#  screw to the phase advance leaves a 5-16 mm residual for every shape in
#  this family, because 9 joint pairs at 8 cm spacing holding >=1.2 wraps
#  gives at most ~7 pairs per turn -- a 7-sided polygon, which clunks rather
#  than rolls. So the body rolls, but lumpily, and the climb comes from the
#  combination of that roll and the squeeze cycle below.
#
#  THIS GAIT DEPENDS ON THE GRIP PADS IN THE XML
#  ───────────────────────────────────────────────────────────────────────
#  With bare-plastic friction (mu = 1.0) no gait in this family climbs at
#  all -- the robot grips the pole happily but makes no net height. mu = 2.0,
#  which assumes soft high-grip elastomer pads on the modules, is what makes
#  the climb possible. Measured during the climb, the gait demands mu = 1.20
#  at the 95th percentile against the 2.0 available, so roughly a 1.7x
#  margin. The robot model itself is untouched -- only the pad material.
#
#  WHERE THE ROTATION FORMULA LIVES  (two-wave serpenoid template)
#  ───────────────────────────────────────────────────────────────────────
#  Both gaits in this file are the standard two-wave template used in the
#  limbless-robot literature (Chong et al.; the same template Hirose's
#  serpenoid curve generalises to):
#
#      alpha_y(t,i) = A_y * sin(omega*t + 2*pi*i*n_y/N + Delta_d)     yaw
#      alpha_p(t,i) = A_p * sin(omega*t + 2*pi*i*n_p/N)               pitch
#
#  with A the amplitudes, n the number of spatial waves along the body,
#  omega the temporal frequency, N = 9 joint pairs, and Delta_d the phase
#  offset between the two waves. Delta_d is the ROTATION knob: a full body
#  revolution about the body axis happens only at Delta_d = +-pi/2, and
#  rolling is the special case A_y = A_p, n_y = n_p = 0, Delta_d = -pi/2,
#  which collapses to alpha_y = A sin(omega t), alpha_p = A cos(omega t).
#
#  In this file the template appears twice:
#    approach_cmd() above  -- the ground gait. A_y=45 deg, A_p=30 deg,
#                             n_y=n_p=1.5, omega=2, Delta_d=+pi/2
#                             (DELTA_D_APPROACH). That is sidewinding.
#    coil_pose() below     -- the climbing gait, XI*n playing the role of
#                             2*pi*i*n/N, and `phase` the role of omega*t.
#
#  NOTE ON SIGN: coil_pose() puts the offset on the PITCH term while the
#  template puts it on the yaw, so DELTA here equals -Delta_d. The shipped
#  DELTA = 98 deg therefore means Delta_d = -98 deg, 8 degrees off the
#  -90 deg of the canonical rolling case.
#
#  THE CANONICAL ROLLING CASE IS ALSO AVAILABLE, AND IT IS FASTER
#  ───────────────────────────────────────────────────────────────────────
#  Setting A_YAW = A_PITCH = 65 deg, XI = 0, DELTA = 90 deg (Delta_d =
#  -90 deg) and PHASE0 = -90 deg is exactly the template's rolling gait.
#  Measured: it reaches the top in 50 s at 8.45 cm/s, against 102 s and
#  3.67 cm/s for the shipped gait -- about 2.3x faster.
#
#  It is not the default only because it is rougher at the finish: holding
#  at the top it trips the QVEL_LIMIT divergence guard after ~3 s, where the
#  shipped gait ends cleanly. Both roll about equally (-0.87 vs -0.86
#  revolutions per cycle), so the default trades speed for a tidy ending.
#  Switch to it by changing those four constants.
#
#  PHASE0 matters as much as DELTA: the same rolling gait started at
#  PHASE0 = 0 instead of -90 deg climbs 0.07 m rather than 2.5 m, because
#  PHASE0 sets the shape the snake first closes around the pole with.
#
#  TUNING GUIDE   (watch the startup diagnostic after any change)
#  ───────────────────────────────────────────────────────────────────────
#  A_YAW, A_PITCH   How hard the clamp squeezes. The window is NARROW:
#                   82/70 climbs the full pole, while 80/68 and 84/72 both
#                   fail to lift off at all. Change these in 1 deg steps and
#                   re-check, or not at all.
#  DELTA            Offset between the yaw and pitch waves, and the knob that
#                   creates the breathing. At 90 deg the radius barely moves
#                   (a fixed ring, no ratchet, no climb). 95-98 deg works;
#                   101 deg already fails.
#  XI               Spatial phase gradient per pair. 0 here, so every pair
#                   clamps together. Non-zero values were swept repeatedly
#                   and always did worse -- above about 20 deg the snake
#                   leaves the pole entirely.
#  OMEGA_CLIMB      Clamp/release cycles per second (rad/s of phase). 1.5 and
#                   2.0 both climb the full pole; 0.75 does not lift off.
#  SPIN             Direction of the cycle. +1 climbs, -1 does not.
A_YAW    = deg2rad(82)
A_PITCH  = deg2rad(70)
XI       = deg2rad(0)
DELTA    = deg2rad(98)
OMEGA_CLIMB = 2.0    # rad/s of phase advance (clamp/release rate)
SPIN        = +1     # +1 climbs, -1 does not

# Where in the grip cycle the body is when it first coils onto the pole. This
# matters more than it looks: the wrap target is coil_pose(PHASE0), so PHASE0
# decides which shape the snake closes around the pole with, and a different
# starting shape grabs the pole differently. Measured: the same gait run from
# two different PHASE0 values climbed 3.0 m and 0.07 m respectively.
PHASE0 = deg2rad(0)

# Stop cycling near the top of the pole and just hold on, instead of driving
# on and sliding off the end. The pole is 3.0 m tall and the wrapped body
# spans roughly 0.25 m of it, so the coil starts running out of pole once the
# centre of mass passes about 2.6 m. Measured: with this set to 2.75 the robot
# overran the top and fell the whole way down.
STOP_HEIGHT = 2.55   # m -- freeze the grip cycle above this height


def coil_pose(phase):
    """Joint targets for the wrapped coil at a given point in the grip cycle."""
    joints = np.zeros(18)
    for n in range(1, N + 1):
        joints[2 * n - 2] = A_YAW * np.sin(XI * n + phase)
        joints[2 * n - 1] = A_PITCH * np.sin(XI * n + phase + DELTA)
    return joints


# ─────────────────────────────────────────────────────────────────────────────
# STARTUP DIAGNOSTIC – measure the wrapped shape before running anything
# ─────────────────────────────────────────────────────────────────────────────
#
# What decides whether this gait can climb is how far the commanded coil
# radius BREATHES across a phase cycle. Both ends of the cycle are usually
# tighter than the 7.0 cm at which the body would just touch the pole
# (4.0 cm pole + 3.0 cm body), so the body does not actually let go of the
# pole at any point -- the pole holds it open and what varies is how hard it
# is squeezed. The ratchet runs on that varying squeeze pressure.
#
# So the number that matters is the SWING, not whether the coil opens past
# the pole. A shape whose radius barely moves cannot ratchet at all: it just
# hangs on. Measured on the old contact model:
#     6.94 -> 6.96 cm (0.02 cm swing)  holds on, never climbs  [the old gait]
#
# Deliberately NOT reported: a predicted climb speed. The obvious estimate
# (treat the motion as a rigid screw and read off its lead) was tried and is
# not trustworthy here -- the body deforms as it cycles (5-16 mm off rigid),
# and that estimate rates the jamming shape as the best climber of all.
# Climb rate is quoted from physics only.
def _shape_geometry(n_phases=16):
    probe = mujoco.MjData(model)

    def backbone(ph):
        mujoco.mj_resetData(model, probe)
        probe.qpos[0:3] = 0.0
        probe.qpos[3] = 1.0
        probe.qpos[4:7] = 0.0
        probe.qpos[7:25] = coil_pose(ph)
        mujoco.mj_forward(model, probe)
        return np.array([probe.xpos[b].copy() for b in BODY_IDS])

    def fit_for_axis(P, a):
        a = a / np.linalg.norm(a)
        ref = np.array([1.0, 0.0, 0.0])
        if abs(a @ ref) > 0.9:
            ref = np.array([0.0, 1.0, 0.0])
        u = np.cross(a, ref); u /= np.linalg.norm(u)
        v = np.cross(a, u)
        x, y, h = P @ u, P @ v, P @ a
        sol, *_ = np.linalg.lstsq(np.column_stack([x, y, np.ones_like(x)]),
                                  x ** 2 + y ** 2, rcond=None)
        cx, cy = sol[0] / 2, sol[1] / 2
        R = np.sqrt(max(sol[2] + cx ** 2 + cy ** 2, 1e-12))
        rr = np.hypot(x - cx, y - cy)
        th = np.unwrap(np.arctan2(y - cy, x - cx))
        sol2, *_ = np.linalg.lstsq(np.column_stack([th, np.ones_like(th)]), h, rcond=None)
        cost = (np.sqrt(np.mean((rr - R) ** 2))
                + np.sqrt(np.mean((h - (sol2[0] * th + sol2[1])) ** 2)))
        # cost FIRST: these tuples get compared directly to pick the best axis
        return cost, R, abs(2 * np.pi * sol2[0]), abs(th[-1] - th[0]) / (2 * np.pi)

    # coarse axis search (Fibonacci hemisphere), then local refinement
    i = np.arange(600) + 0.5
    z = 1.0 - i / 600
    rad = np.sqrt(np.clip(1 - z * z, 0, 1))
    ang = np.pi * (1 + 5 ** 0.5) * i
    axes = np.stack([rad * np.cos(ang), rad * np.sin(ang), z], axis=1)

    out = []
    for ph in [2 * np.pi * k / n_phases for k in range(n_phases)]:
        P = backbone(ph)
        best = min((fit_for_axis(P, a), tuple(a)) for a in axes)
        a0 = np.array(best[1]); step = 0.05
        for _ in range(4):
            ref = np.array([1.0, 0.0, 0.0])
            if abs(a0 @ ref) > 0.9:
                ref = np.array([0.0, 1.0, 0.0])
            u = np.cross(a0, ref); u /= np.linalg.norm(u)
            v = np.cross(a0, u)
            for du in (-step, 0.0, step):
                for dv in (-step, 0.0, step):
                    cand = a0 + du * u + dv * v
                    f = fit_for_axis(P, cand)
                    if f < best[0]:
                        best = (f, tuple(cand / np.linalg.norm(cand)))
            a0 = np.array(best[1]); step *= 0.4
        out.append(best[0])                            # (cost, radius, pitch, turns)
    radii = [o[1] for o in out]
    turns = [o[3] for o in out]
    return min(radii), max(radii), min(turns), max(turns)


_r_min, _r_max, _turn_min, _turn_max = _shape_geometry()
_BODY_RADIUS = 0.043          # L-link convex hull, the fattest part of the body
_needed = PIPE_RADIUS + _BODY_RADIUS
print("=" * 70)
print("GRIP CYCLE (measured by forward kinematics over one full phase cycle)")
print("=" * 70)
print(f"  pole contact  : needs {_needed*100:5.2f} cm of coil radius "
      f"(pole {PIPE_RADIUS*100:.1f} + body {_BODY_RADIUS*100:.1f})")
# These are the COMMANDED coil radii in free space. The pole physically holds
# the body open, so the interference below is not how far anything sinks in --
# measured penetration during the climb is about 3.5 mm. It is a measure of
# how hard the motors are asked to squeeze.
print(f"  hard squeeze  : {_r_min*100:5.2f} cm  -> commands "
      f"{(_needed - _r_min)*1000:5.1f} mm of interference")
print(f"  light squeeze : {_r_max*100:5.2f} cm  -> commands "
      f"{(_needed - _r_max)*1000:5.1f} mm of interference")
print(f"  breathing     : {(_r_max - _r_min)*100:5.2f} cm of swing "
      f"-- this is what drives the ratchet")
print(f"  wraps of grip : {_turn_min:.2f} - {_turn_max:.2f} turns around the pole")
if (_r_max - _r_min) < 0.005:
    print("  WARNING: the coil barely breathes, so there is no clamp/release cycle.")
    print("           It will hold onto the pole without climbing. Adjust DELTA.")

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
_placement_probe.qpos[2]   = GROUND_Z
_placement_probe.qpos[3:7] = Q_LYING_FLAT
mujoco.mj_forward(model, _placement_probe)
_tail0 = _placement_probe.qpos[0:2].copy()
_probe_dt = model.opt.timestep
for _step in range(int(T_PROBE / _probe_dt)):
    _placement_probe.ctrl[:18] = approach_cmd(_step * _probe_dt)
    mujoco.mj_step(model, _placement_probe)
_tail_displacement = _placement_probe.qpos[0:2].copy() - _tail0

mujoco.mj_resetData(model, data)
data.qpos[0:2] = -_tail_displacement
data.qpos[2]   = GROUND_Z
data.qpos[3:7] = Q_LYING_FLAT
mujoco.mj_forward(model, data)

# ─────────────────────────────────────────────────────────────────────────────
# VIEWER
# ─────────────────────────────────────────────────────────────────────────────
viewer = mujoco_viewer.MujocoViewer(model, data)

dt = model.opt.timestep

# Rendering, not physics, decides how long this takes to watch. Measured here:
# physics costs 0.06 s of wall clock per simulated second (16x faster than
# real time), while one viewer frame costs ~48 ms -- and that cost does not
# change with window size, so it is vsync/driver bound, not fill-rate bound.
#
# Redrawing every 10th step (the old setting, inherited from robot_command.py)
# meant 100 frames per simulated second = ~4.8 s of rendering per simulated
# second. The run played at 1/4 speed, so the climb -- which happens between
# roughly t=60 s and t=200 s of simulated time -- needed ~16 minutes of
# watching to get anywhere, and the first thing on screen was 2 minutes of
# crawling and coiling. Pace the frames instead: one frame every
# RENDER_EVERY steps gives the playback speed below.
TARGET_SPEEDUP = 3.0    # simulated seconds to play per wall-clock second
TARGET_FPS     = 15.0   # frames per wall-clock second
RENDER_EVERY   = max(1, int(round(TARGET_SPEEDUP / (TARGET_FPS * dt))))

print()
print("=" * 70)
print("SNAKE ROBOT - GROUND-TO-PIPE CLIMBING SIMULATION")
print("=" * 70)
print(f"Pipe          : radius {PIPE_RADIUS*100:.0f} cm, 3.0 m tall, standing on the floor")
print(f"Robot start   : x={data.qpos[0]:.3f} m, y={data.qpos[1]:.3f} m, "
      f"z={data.qpos[2]:.3f} m (lying flat on the ground)")
print("\nPhases: 1) sidewind across the ground to the pipe   (to t~8 s)")
print("        2) coil around it, one joint pair at a time    (to t~33 s)")
print("        3) cycle the coil open and shut -- it ratchets up the pipe")
print("Close the viewer window to stop.")

# ─────────────────────────────────────────────────────────────────────────────
# STATE MACHINE:  approach -> wrapping -> climb
# ─────────────────────────────────────────────────────────────────────────────
PROX_PAIR_THRESHOLD = 0.06   # m – a pair starts curling once its body is this close to the pipe
RAMP_DURATION_PAIR  = 1.5    # s – how long one pair takes to go from wave to wrap
FORCE_WRAP_AFTER    = 20.0   # s – a pair that never got close enough curls anyway, so the
                             #     coil closes instead of leaving a slack gap in the loop
SETTLE_HOLD         = 3.0    # s – hold the closed coil still, letting contacts settle,
                             #     before starting the grip cycle
APPROACH_TIMEOUT    = 40.0   # s – safety cutoff if the pipe is never reached
CLIMB_DURATION      = 110.0  # s – how long to keep climbing (it tops out around 76 s)

phase = 'approach'
held_phase = 0.0          # grip-cycle phase, frozen once the top is reached
top_reached_time = None
pair_trigger_time = [None] * N
wrapping_start_time = None
all_locked_time = None
climb_start_time = None

time_history = []
com_z_history = []

_total_sim = 33.0 + CLIMB_DURATION   # setup is about 33 simulated seconds
print()
print(f"Playback : ~{TARGET_SPEEDUP:.0f}x real time, one frame every {RENDER_EVERY} steps,")
print(f"           {_total_sim:.0f} simulated seconds in roughly "
      f"{_total_sim/TARGET_SPEEDUP/60:.1f} minutes of watching.")
print("Expect   : a steady climb once wrapped -- about 1 m by t~60 s, and the top")
print("           of the 3 m pole around t~110 s, where it stops and holds on.")
print()

max_steps = int((APPROACH_TIMEOUT + FORCE_WRAP_AFTER + CLIMB_DURATION + 40.0) / dt)
_wall_start = time.perf_counter()
_sim_end = 0.0
viewer_ok = True      # False once the viewer window has gone away
diverged = False      # True if the solver blew up and we stopped early

# Divergence threshold, set from measured behaviour rather than guessed:
#   approach (legitimate transient)   max |qvel|  85
#   climbing, healthy                             34
#   holding at the top                            65
#   the blow-up that flings the robot off        154
# 120 sits clear above everything legitimate and below the blow-up.
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
                print(f"PHASE 2: reached the pipe at t={t:.2f}s -- coiling around it...")
            elif t > APPROACH_TIMEOUT:
                print(f"\nWARNING: never reached the pipe in {APPROACH_TIMEOUT:.0f}s "
                      f"-- check T_PROBE / the placement measurement.")
                phase = 'wrapping'
                wrapping_start_time = t

        elif phase == 'wrapping':
            wrap_target = coil_pose(PHASE0)
            cmd = np.zeros(18)
            for n in range(1, N + 1):
                if pair_trigger_time[n - 1] is None:
                    if radial_distance(data.xpos[PAIR_BODY_IDS[n - 1]][:2]) < PROX_PAIR_THRESHOLD:
                        pair_trigger_time[n - 1] = t
                    elif (t - wrapping_start_time) > FORCE_WRAP_AFTER:
                        pair_trigger_time[n - 1] = t
                if pair_trigger_time[n - 1] is None:
                    alpha = 0.0
                else:
                    alpha = min(1.0, (t - pair_trigger_time[n - 1]) / RAMP_DURATION_PAIR)
                yi, pidx = 2 * n - 2, 2 * n - 1
                cmd[yi]   = (1 - alpha) * a_cmd[yi]   + alpha * wrap_target[yi]
                cmd[pidx] = (1 - alpha) * a_cmd[pidx] + alpha * wrap_target[pidx]

            fully_locked = all(tt is not None for tt in pair_trigger_time) and all(
                (t - tt) >= RAMP_DURATION_PAIR for tt in pair_trigger_time)
            if fully_locked and all_locked_time is None:
                all_locked_time = t
                print(f"  all {N} joint pairs wrapped at t={t:.2f}s -- settling...")
            if all_locked_time is not None and (t - all_locked_time) > SETTLE_HOLD:
                phase = 'climb'
                climb_start_time = t
                print(f"\nPHASE 3: grip cycle starts at t={t:.2f}s -- climbing...\n")

        else:  # climb -- cycle the coil: clamp, extend, re-clamp, ratcheting up
            if data.subtree_com[0, 2] < STOP_HEIGHT:
                held_phase = PHASE0 + SPIN * OMEGA_CLIMB * (t - climb_start_time)
            elif top_reached_time is None:
                top_reached_time = t
                print(f"\nReached the top of the pole at t={t:.1f}s "
                      f"(height {data.subtree_com[0, 2]:.2f} m) -- holding on.")
            cmd = coil_pose(held_phase)   # freeze the cycle once at the top
            # A frozen grip is not a permanent one: held still, the coil creeps
            # down a few cm every 10 s and eventually lets go. Finish the run
            # at the top rather than wait for that.
            if top_reached_time is not None and (t - top_reached_time) > 5.0:
                print("Done -- stopping at the top.")
                break
            if (t - climb_start_time) > CLIMB_DURATION:
                print("\nClimb duration complete.")
                break

        data.ctrl[:18] = cmd
        mujoco.mj_step(model, data)
        _sim_end = t

        # Bail out if the solver has diverged, instead of letting the robot be
        # flung through the pole. Deep contact interference (this gait commands
        # 17-29 mm of it) can occasionally produce an enormous impulse, and
        # once qvel blows up the remaining "motion" is numerical garbage.
        if not np.all(np.isfinite(data.qpos)) or np.max(np.abs(data.qvel)) > QVEL_LIMIT:
            print(f"\nSTOPPED at t={t:.2f}s: the physics diverged "
                  f"(max |qvel| = {np.max(np.abs(data.qvel)):.0f}, limit {QVEL_LIMIT:.0f}).")
            print("Nothing after this point would be meaningful, so the run ends here.")
            diverged = True
            break

        # Render only while the window is really open. mujoco_viewer.render()
        # raises if the window has been closed, and closing it is exactly how
        # this script tells you to stop -- so check first and guard the call.
        if viewer.is_alive and step % RENDER_EVERY == 0:
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
            n_wrapped = sum(1 for tt in pair_trigger_time if tt is not None)
            print(f"t={t:6.1f}s | {phase:8s} | height={data.subtree_com[0, 2]:.3f} m "
                  f"| wrapped={n_wrapped}/{N}")

        if viewer_ok and not viewer.is_alive:
            print("\nViewer closed by user.")
            break

except KeyboardInterrupt:
    print("\nInterrupted by user.")

try:
    viewer.close()
except Exception:
    pass

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
time_history  = np.array(time_history)
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
    print("Climb phase never started -- the robot did not wrap the pipe.")
_wall = time.perf_counter() - _wall_start
if _wall > 0 and _sim_end > 0:
    print(f"Playback speed        : {_sim_end/_wall:.1f}x real time "
          f"({_sim_end:.0f} simulated s in {_wall/60:.1f} min of wall clock)")
    print(f"                        raise TARGET_SPEEDUP for a faster, choppier replay")
print("=" * 70)
print()
print("TUNING TIPS")
print("  - No breathing in the startup diagnostic -> the coil never releases, so")
print("    it will hold onto the pipe without ever climbing. Adjust DELTA.")
print("  - Climbs then slides back down -> grip is marginal; nudge A_YAW and")
print("    A_PITCH up together by 1-2 deg, or lower OMEGA_CLIMB.")
print("  - Never leaves the floor -> the operating window is narrow. 82/70 with")
print("    DELTA=98 climbs; 80/68, 84/72 and DELTA=101 all fail to lift off.")
print("  - Never reaches the pipe -> adjust T_PROBE.")
print("  - Wraps but will not lift -> check SPIN (+1 climbs) and PHASE0, which")
print("    decides the shape it first grabs the pole with and matters a lot.")
print("  - 'physics diverged' -> the run was stopped on purpose rather than let")
print("    the robot be flung through the pole. Usually happens only at the top.")
