"""
Experiment rig for adaptive helical rolling on a straight pole.

This is robot_pipe_climb.py turned into something you can run experiments
with: every number that changes the outcome is a parameter instead of a
constant, the run is divided into named stages, and everything worth looking
at afterwards is written to CSV.

The control method itself is unchanged and still lives in helical_rolling.py
(Takemori, Tanaka & Matsuno, IEEE T-RO 39(1):437-451, 2023).

    python climb_rig.py                      one run with the defaults, headless
    python climb_rig.py --mu 1.5 --alpha 0.3 override any Config field
    python climb_rig.py --viewer             watch it
    python experiment_gui.py                 the same thing with a control panel

See README.md for the CSV schema and what each stage is.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as _dt
import json
import os
import time

import mujoco
import numpy as np

import helical_rolling as hr
from snake_backbone import Backbone

MODEL_XML = '9motor_sidewinder_pipe.xml'
NJ = 18            # joints
N_PAIR = 9         # yaw/pitch joint pairs


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------
# The run is not one behaviour, it is six, and mixing them in one plot hides
# everything interesting. Every logged row carries the stage it came from.
#
#   approach    Sidewinding across the floor toward the pole. Open loop, and
#               nothing to do with the paper -- the paper's experiments all
#               start with the robot already on the pipe.
#   catching    Each joint pair ramps from the ground gait into the catching
#               helix as it arrives at the pole, so the body closes on the
#               pole in the order it reaches it.
#   settling    All pairs are wrapped; the shape is held still so the contacts
#               can settle before anything starts rolling.
#   pitch_ramp  The adaptive loop is running and the pitch angle is opening
#               from the catching value to the climbing one. The robot is
#               already holding on and already climbing a little.
#   rolling     The experiment proper: adaptive helical rolling at the full
#               pitch angle. This is the stage to quote climb rates from.
#   holding     The top of the pole was reached, psi_roll is frozen, and the
#               robot is just gripping. Shows whether the grip holds statically.
STAGES = ('approach', 'catching', 'settling', 'pitch_ramp', 'rolling', 'holding')

# Catching shapes tried on successive attempts after the Config's own
# (wrap_alpha, wrap_radius, wrap_wave_delay). How a coil folds round the pole
# on the floor is repeatable for a given shape, so a retry with the same shape
# tends to repeat the same crooked coil; these were picked because each
# catches starts that the others miss.
CATCH_ALTERNATES = (
    (0.05, 0.060, 0.10),
    (0.10, 0.050, 0.25),
    (0.10, 0.060, 0.10),
)


@dataclasses.dataclass
class Config:
    """Everything that changes the outcome of a run."""

    # -- environment ------------------------------------------------------
    pole_radius: float = 0.04      # m,   cylinder radius
    mu_robot_pole: float = 0.5     # -,   sliding friction between robot and pole.
                                   #      ~0.3 plastic, ~0.5-0.8 rubber on a clean
                                   #      pipe. Climbs down to about 0.25.
    mu_floor: float = 0.9          # -,   sliding friction of the floor; only the
                                   #      approach stage cares
    mass_scale: float = 1.0        # x,   multiplies every link's mass AND its
                                   #      inertia, so the robot stays physical.
                                   #      Nominal total is 738 g.
    gravity: float = 9.81          # m/s2

    # -- solver and timing -------------------------------------------------
    timestep: float = 0.001        # s,   physics step
    control_period: float = 0.1    # s,   how often the controller re-estimates the
                                   #      form from the joints. 0.1 s is the
                                   #      paper's hardware rate.
    setpoint_period: float = 0.01  # s,   how often joint targets are streamed
                                   #      between those ticks (eq. 33 with the
                                   #      current psi_roll). >= control_period
                                   #      holds each target for a whole tick,
                                   #      which slips below mu ~1
    max_duration: float = 260.0    # s,   hard stop on simulated time

    # -- controller (adaptive helical rolling) -----------------------------
    alpha: float = 0.25            # rad, pitch angle of the climbing helix
    k_mid: float = 0.08            # -,   compliance gain, middle joints, eq. (23)
    k_end: float = 0.04            # -,   compliance gain, end joints
    k_end_count: int = 3           # how many joints at each end get k_end
    psi_dot: float = 0.5 * np.pi   # rad/s, rolling speed, eq. (33)
    spin: int = +1                 # +1 climbs, -1 descends
    lead_sign: float = +1.0        # handedness of the wrap

    # -- catching the pole (not part of the paper) -------------------------
    wrap_alpha: float = 0.07       # rad, nearly flat so the ring closes on the pole
    wrap_radius: float = 0.060     # m,   tighter than the pole so it grips
                                   #      (retries cycle CATCH_ALTERNATES below)
    wrap_psi: float = -0.5 * np.pi # rad, which PLANE the ring closes in, and on
                                   #      which SIDE of the body. -pi/2 closes it
                                   #      on the side the sidewinding gait
                                   #      travels toward, i.e. round the pole;
                                   #      +pi/2 closes it on the far side
    straighten_time: float = 1.0   # s,   on reaching the pole, straighten against it...
    straighten_hold: float = 0.5   # s,   ...and come to rest before measuring
    catch_u_min: float = 0.02      # m,   then curl only if the pole is this far
    catch_u_max: float = 0.27      #      along the body from pair 1...
    catch_v_max: float = 0.15      # m,   ...and this close beside it, on the side
                                   #      the ring closes toward. Mapped from rest:
                                   #      inside this window the catch encircles
                                   #      the pole and climbs every time
    retry_cooldown: float = 3.0    # s,   after a miss, sidewind this long before
                                   #      trying again
    reopen_frac: float = 0.4       # -,   a crooked coil is loosened to this fraction
    reopen_time: float = 1.0       # s,   of the catching shape, then closed again
    aim_u: float = 0.12            # m,   the approach is aimed to bring the pole to
                                   #      this point along the body, mid-window
    wrap_wave_delay: float = 0.12  # s,   pairs curl one after another outward from
                                   #      the one nearest the pole, so the body
                                   #      winds round it like a rope round a post
    alpha_ramp: float = 6.0        # s,   opening the pitch angle after catching
    stop_height: float = 2.80      # m,   freeze psi_roll above this COM height
    hold_time: float = 5.0         # s,   how long to hold at the top before stopping

    # -- starting already on the pole --------------------------------------
    # All four of the paper's experiments begin with the robot placed on the
    # pipe by hand as "a normal helix with a radius of R and a pitch angle of
    # alpha". Setting start_wrapped does the same here: it skips the approach
    # and catching stages entirely and drops the robot onto the pole in that
    # helix. Use it whenever the question is about the CONTROLLER -- otherwise
    # a parameter that happens to break the floor gait looks like a climbing
    # failure, which it is not.
    start_wrapped: bool = False
    start_radius: float = 0.085    # m, radius of the placed helix
    start_height: float = 0.35     # m, height of its lowest point

    # -- stage machine timings ---------------------------------------------
    prox_threshold: float = 0.06   # m,   the approach stops and straightens when
                                   #      any pair is this close to the pole
    ramp_per_pair: float = 1.5     # s,   wave -> wrap blend for one pair
    settle_hold: float = 3.0       # s
    min_grip_contacts: float = 6.0 # -,   pole contacts (mean, last 1 s of settling)
                                   #      needed before climbing is allowed to start
    grip_com_radius: float = 0.10  # m,   and the COM must be this close to the pole axis
    min_wrap_turns: float = 0.9    # -,   and the backbone must wind this far round
                                   #      it. A coil squeezing the pole from one side
                                   #      has contacts but cannot screw itself up
    max_catch_tilt: float = 34.0   # deg, and the coil's own helix axis must be this
                                   #      close to vertical: the rolling controller
                                   #      corrects ~30 deg, but from ~38 it peels off
    climb_abort_below: float = 0.30  # m, while the COM is still this low...
    climb_abort_tilt: float = 55.0   # deg, ...an axis this far off vertical, or no
    climb_abort_after: float = 2.0   # s,   pole contact at all, for this long, means
                                     #      the catch is peeling off: re-close it
    max_catch_attempts: int = 8    # misses before the run gives up
    approach_timeout: float = 40.0 # s
    probe_time: float = 10.5       # s,   the start-placement probe (see below)

    # -- logging and display ----------------------------------------------
    log_hz: float = 50.0           # Hz,  CSV sampling rate
    label: str = ''                # goes into the run folder name
    show_viewer: bool = False
    immersive: bool = True         # annotated, tracking view (immersive.py).
                                   # False gives the plain MuJoCo viewer.
    target_speedup: float = 3.0    # simulated seconds per wall-clock second
    target_fps: float = 15.0       # viewer frames per wall-clock second
    out_dir: str = 'results'

    def k_gain(self):
        """Per-joint compliance gain, in the robot's own J1..J18 order."""
        k = np.full(NJ, self.k_mid, dtype=float)
        n = max(0, int(self.k_end_count))
        if n:
            k[:n] = self.k_end
            k[-n:] = self.k_end
        return k


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------
def build_model(cfg: Config):
    """Load the XML and apply the experiment's parameters to it.

    Everything here is done on the compiled model rather than by editing the
    XML, so a sweep can vary friction or mass without touching a file. The
    robot's geometry, joint layout and actuators are untouched.
    """
    # Pole radius. Set on the spec BEFORE compiling, not on the compiled model:
    # the compiler derives the pole's bounding box from its size, and writing
    # geom_size afterwards leaves that box at the XML's 0.04 m. Collision then
    # misses much of any larger pole and the robot sinks into it (37.8 mm on
    # an 0.08 m pole). The half-height and position are left alone, so the
    # pole still runs from the floor to z = 3.0 m.
    spec = mujoco.MjSpec.from_file(MODEL_XML)
    spec.geom('pipe').size[0] = cfg.pole_radius
    model = spec.compile()

    model.opt.timestep = cfg.timestep
    model.opt.gravity[:] = (0.0, 0.0, -abs(cfg.gravity))

    floor_gid = model.geom('floor').id
    pipe_gid = model.geom('pipe').id

    # Sliding friction. MuJoCo combines a contact pair by taking the
    # elementwise maximum, which let robot-floor contacts silently use the
    # pole's coefficient whenever it was the larger one. Giving the floor and
    # the pole contact priority makes each surface's own value win against
    # the robot. The floor's condim and torsional friction are matched to
    # what the combined pair used before, so only mu_floor itself changes.
    model.geom_priority[floor_gid] = 1
    model.geom_priority[pipe_gid] = 1
    model.geom_condim[floor_gid] = 4
    model.geom_friction[floor_gid, 1] = model.geom_friction[pipe_gid, 1]
    model.geom_friction[floor_gid, 0] = cfg.mu_floor
    set_friction(model, cfg.mu_robot_pole, floor_gid)

    # Mass. Scaling inertia by the same factor keeps the mass distribution --
    # and therefore every natural frequency ratio -- unchanged, so this is a
    # clean "heavier robot, same robot" knob.
    if cfg.mass_scale != 1.0:
        model.body_mass[:] *= cfg.mass_scale
        model.body_inertia[:] *= cfg.mass_scale

    return model


# ---------------------------------------------------------------------------
# The flat-ground approach gait (unchanged, and unrelated to the paper)
# ---------------------------------------------------------------------------
_AY = np.deg2rad(45)
_AP = np.deg2rad(30)
_OMEGA = 2.0
_NWAVE = 1.5
_DELTA = -4.202
_DELTA_D = 0.5 * np.pi


def approach_cmd(t):
    """Zero-mean traveling-wave serpenoid gait: sidewinds, does not wrap."""
    j = np.zeros(NJ)
    for i in range(NJ):
        n = i + 1
        if n % 2 == 1:
            j[i] = _AY * np.sin(_OMEGA * t + 2 * np.pi * ((n + 1) // 2) * (_NWAVE / N_PAIR)
                                + _DELTA + _DELTA_D)
        else:
            j[i] = _AP * np.sin(_OMEGA * t + 2 * np.pi * (n // 2) * (_NWAVE / N_PAIR) + _DELTA)
    return j


def measure_start_offset(model, cfg: Config):
    """Where to put the robot so the approach gait delivers it to the pole.

    The sidewinding gait's travel path is a curved track, not a straight line
    along the body, so this is measured rather than derived: run the gait on a
    throwaway MjData seeded far from the pole (floor physics is
    translation-invariant, so nothing is contaminated by contact with it), see
    where the tail ends up, and start the real robot at the mirror image of
    that displacement.

    It has to be redone for every Config, because friction and mass change how
    far the gait travels.
    """
    probe = mujoco.MjData(model)
    mujoco.mj_resetData(model, probe)
    probe.qpos[0:2] = (3.0, 3.0)
    probe.qpos[2] = 0.05
    probe.qpos[3:7] = _Q_LYING_FLAT
    mujoco.mj_forward(model, probe)
    start = probe.qpos[0:2].copy()
    dt = model.opt.timestep
    for step in range(int(cfg.probe_time / dt)):
        probe.ctrl[:NJ] = approach_cmd(step * dt)
        mujoco.mj_step(model, probe)
    # Aim a point aim_u along the body at the pole, not the tail: the catch
    # only encircles the pole when it arrives inside the catch window.
    p1 = probe.xpos[model.body('M1').id][:2]
    ax = probe.xpos[model.body(f'M{N_PAIR}').id][:2] - p1
    return p1 + cfg.aim_u * ax / np.linalg.norm(ax) - start


_theta = np.deg2rad(90)
_Q_LYING_FLAT = np.array([np.cos(_theta / 2), np.sin(_theta / 2), 0.0, 0.0])


# ---------------------------------------------------------------------------
# Placing the robot already wrapped around the pole
# ---------------------------------------------------------------------------
def _circle_fit(xy):
    """Centre of the circle that best fits a set of 2-D points."""
    A = np.column_stack([xy[:, 0], xy[:, 1], np.ones(len(xy))])
    sol, *_ = np.linalg.lstsq(A, xy[:, 0] ** 2 + xy[:, 1] ** 2, rcond=None)
    return np.array([sol[0] / 2, sol[1] / 2])


def _rot_to_z(a):
    """Rotation taking the unit vector `a` onto +z."""
    a = a / np.linalg.norm(a)
    z = np.array([0.0, 0.0, 1.0])
    v, c = np.cross(a, z), float(a @ z)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    V = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + V + V @ V * (1.0 / (1.0 + c))


def place_wrapped(model, data, backbone, joints, base_z=0.35):
    """Put the robot on the pole in the commanded helix, axis vertical.

    Sets the joints with the free joint at the origin, works out where that
    shape's own helical axis points, then uses the free joint to rotate that
    axis onto the pole and slide the whole thing up to `base_z`.
    """
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = 0.0
    data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
    data.qpos[7:7 + NJ] = joints
    mujoco.mj_forward(model, data)

    R = _rot_to_z(hr.fit_helix(backbone.nodes(model, data))['axis'])
    rotated = backbone.nodes(model, data) @ R.T
    centre = _circle_fit(rotated[:, :2])

    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, R.reshape(9))
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = (-centre[0], -centre[1], base_z - rotated[:, 2].min())
    data.qpos[3:7] = quat
    data.qpos[7:7 + NJ] = joints
    mujoco.mj_forward(model, data)
    return data


# ---------------------------------------------------------------------------
# Per-run measurements
# ---------------------------------------------------------------------------
def classify_joint_axes(model):
    """Label each joint 'pitch' or 'yaw' from the model, not by assumption.

    Every joint in this model rotates about its own local z and the link
    frames alternate +-90 degrees about x, which makes consecutive joints
    orthogonal. At the reference pose (body laid out along world x) one family
    ends up bending about world z and the other about world y.
    """
    probe = mujoco.MjData(model)
    mujoco.mj_resetData(model, probe)
    probe.qpos[3] = 1.0
    mujoco.mj_forward(model, probe)
    out = []
    for j in range(1, NJ + 1):
        axis = probe.xaxis[model.joint(f'J{j}').id]
        out.append('yaw' if abs(axis[2]) > 0.9 else 'pitch')
    return out


def contact_report(model, data, pipe_gid, floor_gid):
    """Contacts split by what they are against, plus how hard the pole is held.

    `mu_required` is the killer number for a climbing robot: at each contact it
    is |tangential force| / |normal force|, i.e. the coefficient of friction
    that contact is DEMANDING. Compare it with mu_robot_pole to see how much
    margin against slipping the gait is running on. A 95th percentile close to
    the available mu means the robot is about to slide.
    """
    n_pole = n_self = n_floor = 0
    normal_sum = 0.0
    mus = []
    f = np.zeros(6)
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = c.geom1, c.geom2
        if g1 == pipe_gid or g2 == pipe_gid:
            n_pole += 1
            mujoco.mj_contactForce(model, data, i, f)
            fn = abs(f[0])
            normal_sum += fn
            if fn > 1e-9:
                mus.append(float(np.hypot(f[1], f[2]) / fn))
        elif g1 == floor_gid or g2 == floor_gid:
            n_floor += 1
        else:
            n_self += 1
    mu_p95 = float(np.percentile(mus, 95)) if mus else 0.0
    mu_max = float(np.max(mus)) if mus else 0.0
    return n_pole, n_self, n_floor, normal_sum, mu_p95, mu_max


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------
def set_friction(model, mu, floor_gid):
    """Set the robot/pole sliding friction on an already-compiled model.

    Safe to call mid-run: MuJoCo reads geom_friction when it builds each
    contact, so the new value takes effect on the next step. Torsional and
    rolling friction, and the floor, are left alone.
    """
    for g in range(model.ngeom):
        if g != floor_gid:
            model.geom_friction[g, 0] = mu


def run(cfg: Config, on_progress=None, should_stop=None, on_message=None,
        live_mu=None):
    """Run one experiment. Returns a summary dict; writes CSVs as it goes.

    on_progress(dict)  called at the logging rate with the current state
    should_stop()      return True to end the run early
    on_message(str)    human-readable progress lines
    live_mu()          optional: returns the friction coefficient to use right
                       now, polled at the logging rate, so it can be dragged
                       on a slider while the robot is climbing. Only consulted
                       when the immersive viewer is NOT showing its own native
                       slider -- when it is, viewer.mu is authoritative, since
                       that is the control the operator is actually looking
                       at. Either way, whatever value is in force is written
                       to every row of run.csv, so a run with a moving slider
                       is still fully described by its own log.
    """
    say = on_message or (lambda s: None)

    run_dir = _make_run_dir(cfg)
    say(f"writing to {run_dir}")

    model = build_model(cfg)
    data = mujoco.MjData(model)
    backbone = Backbone(model)
    dt = model.opt.timestep
    pipe_gid = model.geom('pipe').id
    floor_gid = model.geom('floor').id
    axis_kind = classify_joint_axes(model)
    jnt_frc_lim = np.abs(model.jnt_actfrcrange[1:1 + NJ, 1])
    jnt_lo = model.jnt_range[1:1 + NJ, 0]
    jnt_hi = model.jnt_range[1:1 + NJ, 1]
    pair_body_ids = [model.body(f'M{n}').id for n in range(1, N_PAIR + 1)]
    mid_body = model.body('L4').id
    roll_ref = _roll_reference(model)

    controller = hr.HelicalRollingController(
        backbone.link_lengths, alpha=cfg.wrap_alpha, K=cfg.k_gain(),
        mode='outside', gamma_min=0.01 * np.pi)
    # psi_roll has to start somewhere. Coming off the floor it continues from
    # the catching form's roll phase so the hand-off does not lurch; placed on
    # the pole it just starts at zero.
    psi0 = 0.0 if cfg.start_wrapped else cfg.wrap_psi
    placed_pose = None

    catch_shapes = [(cfg.wrap_alpha, cfg.wrap_radius, cfg.wrap_wave_delay)] + \
        list(CATCH_ALTERNATES)

    def catch_shape(attempt):
        """(pitch, wave spacing, joint targets) of the catch for this attempt."""
        a, r, wave = catch_shapes[attempt % len(catch_shapes)]
        controller.alpha = a
        return a, wave, controller.normal_helix_target(r, psi0, lead_sign=cfg.lead_sign)

    catch_alpha, wave_delay, wrap_target = catch_shape(0)

    if cfg.start_wrapped:
        controller.alpha = cfg.alpha
        placed_pose = controller.normal_helix_target(
            cfg.start_radius, psi0, lead_sign=cfg.lead_sign)
        place_wrapped(model, data, backbone, placed_pose, cfg.start_height)
        say(f"placed on the pole as a normal helix, r={cfg.start_radius:.3f} m, "
            f"alpha={cfg.alpha:.2f} rad")
    else:
        say("measuring the start position (running the approach gait on a probe)...")
        offset = measure_start_offset(model, cfg)
        mujoco.mj_resetData(model, data)
        data.qpos[0:2] = -offset
        data.qpos[2] = 0.05
        data.qpos[3:7] = _Q_LYING_FLAT
        mujoco.mj_forward(model, data)

    viewer = None
    if cfg.show_viewer:
        _patch_viewer_camera()
    if cfg.show_viewer and cfg.immersive:
        from immersive import ImmersiveViewer
        viewer = ImmersiveViewer(model, data, pole_radius=cfg.pole_radius,
                                 pole_height=2 * model.geom_size[pipe_gid, 1],
                                 mu=cfg.mu_robot_pole)
        say("viewer keys: SPACE pause | B camera | N annotations | Z panel | "
            "C contact arrows | H menus")
        say("drag the friction slider drawn in the window to change grip live")
    elif cfg.show_viewer:
        import mujoco_viewer
        viewer = mujoco_viewer.MujocoViewer(model, data)
    render_every = max(1, int(round(cfg.target_speedup / (cfg.target_fps * dt))))
    log_every = max(1, int(round(1.0 / (cfg.log_hz * dt))))

    # -- CSV files --------------------------------------------------------
    jf = open(os.path.join(run_dir, 'joints.csv'), 'w', newline='')
    rf = open(os.path.join(run_dir, 'run.csv'), 'w', newline='')
    jw, rw = csv.writer(jf), csv.writer(rf)
    # torque_demand_Nm is what the position servo law asks for; torque_Nm is
    # what the joint actually delivers after MuJoCo clamps it to the model's
    # actuatorfrcrange (+-0.8 N.m). They differ constantly -- these motors
    # saturate -- and the gap is one of the more interesting things in the run,
    # so both are recorded along with a saturation flag.
    jw.writerow(['t_s', 'step', 'stage', 'joint', 'joint_name', 'pair', 'axis_kind',
                 'cmd_rad', 'pos_rad', 'err_rad', 'vel_rad_s',
                 'torque_Nm', 'torque_demand_Nm', 'torque_frac', 'saturated',
                 'range_frac'])
    rw.writerow(['t_s', 'step', 'stage', 'stage_t_s', 'mu_robot_pole',
                 'com_x_m', 'com_y_m', 'com_z_m', 'climb_rate_cm_s',
                 'psi_roll_rad', 'alpha_rad',
                 'axis_x', 'axis_y', 'axis_z', 'axis_tilt_deg',
                 'coil_r_meas_m', 'coil_r_cmd_m', 'squeeze_mm', 'wrap_turns',
                 'n_contact_pole', 'n_contact_self', 'n_contact_floor',
                 'pole_normal_force_N', 'mu_required_p95', 'mu_required_max',
                 'body_roll_rev', 'orbit_rev', 'max_abs_qvel',
                 'joint_torque_rms_Nm', 'joint_torque_max_Nm', 'mech_power_W',
                 'joints_saturated', 'pairs_wrapped'])

    # -- state ------------------------------------------------------------
    stage = 'settling' if cfg.start_wrapped else 'approach'
    stage_start = 0.0
    stage_log = []              # (stage, t_start, t_end, z_start, z_end)
    stage_z0 = float(data.subtree_com[0, 2])

    pair_trigger = [0.0] * N_PAIR if cfg.start_wrapped else [None] * N_PAIR
    catching_start = None
    catch_attempts = 0
    straight_from = None
    reopen_start = reopen_from = None
    curl_from = 0.0
    catch_pair = 0
    bad_since = None
    grip_samples = []
    release_t = release_cmd = None
    all_locked = 0.0 if cfg.start_wrapped else None
    # loop_start: when the adaptive loop took over. roll_start: when psi_roll
    # starts advancing. Placed on the pole they differ -- the loop grips for
    # settle_hold seconds before any rolling -- and off the floor they coincide.
    loop_start = 0.0 if cfg.start_wrapped else None
    roll_start = cfg.settle_hold if cfg.start_wrapped else None
    top_time = None
    psi_roll = psi0
    psi_frozen = False
    cmd = placed_pose.copy() if cfg.start_wrapped else approach_cmd(0.0)
    last_control = last_setpoint = -1e9
    info = None

    roll_own = 0.0              # revolutions about the body's own long axis
    orbit = 0.0                 # revolutions about the pole
    prev_ang = None
    vel6 = np.zeros(6)

    mu_now = cfg.mu_robot_pole

    prev_log_t = None
    prev_log_z = None
    diverged = False
    stopped_early = False
    qvel_limit = 120.0
    peak_z = float(data.subtree_com[0, 2])
    t = 0.0

    def switch(new_stage, tt):
        nonlocal stage, stage_start, stage_z0
        stage_log.append((stage, stage_start, tt, stage_z0, float(data.subtree_com[0, 2])))
        stage, stage_start = new_stage, tt
        stage_z0 = float(data.subtree_com[0, 2])

    wall0 = time.perf_counter()
    max_steps = int(cfg.max_duration / dt)
    try:
        for step in range(max_steps):
            t = step * dt
            a_cmd = approach_cmd(t)

            # ---------------- stage machine ----------------
            if stage == 'approach':
                # After a missed catch, fade the curl back out into the gait
                # rather than snapping the joints straight.
                fade = 0.0
                cmd = a_cmd
                if release_t is not None:
                    fade = max(0.0, 1.0 - (t - release_t) / cfg.ramp_per_pair)
                    cmd = fade * release_cmd + (1.0 - fade) * a_cmd
                cooled = release_t is None or t - release_t > cfg.retry_cooldown
                if cooled and min(_radial(data.xpos[b], cfg.pole_radius)
                                  for b in pair_body_ids) < cfg.prox_threshold:
                    switch('catching', t)
                    catching_start = t
                    straight_from = cmd.copy()
                    say(f"t={t:6.2f}s  reached the pole (attempt {catch_attempts + 1}); "
                        f"straightening against it")
                elif t - stage_start > cfg.approach_timeout:
                    say(f"t={t:6.2f}s  STOPPED: never reached the pole; check probe_time")
                    break

            elif stage in ('catching', 'settling') and not cfg.start_wrapped:
                # Catching is done from a straight body at rest against the
                # pole, because that is the state the catch window was mapped
                # in. Curling straight out of the moving, wavy gait closes a
                # coil whose axis is tens of degrees off vertical.
                missed = None
                if reopen_start is not None:
                    b = min(1.0, (t - reopen_start) / cfg.reopen_time)
                    cmd = (1.0 - b) * reopen_from + b * cfg.reopen_frac * wrap_target
                    if t - reopen_start >= cfg.reopen_time + cfg.straighten_hold:
                        # From the pair that made the first catch: once coiled,
                        # whichever pair is nearest the pole may be anywhere
                        # along the body, and closing from the head end forms
                        # the coil beside the pole.
                        pair_trigger = [t + wave_delay * abs(n - catch_pair)
                                        for n in range(N_PAIR)]
                        curl_from = cfg.reopen_frac
                        reopen_start = None
                        say(f"t={t:6.2f}s  closing again from pair {catch_pair + 1} "
                            f"(catch shape {catch_attempts % len(catch_shapes) + 1} "
                            f"of {len(catch_shapes)})")
                elif all(x is None for x in pair_trigger):
                    b = min(1.0, (t - catching_start) / cfg.straighten_time)
                    cmd = (1.0 - b) * straight_from
                    if t - catching_start >= cfg.straighten_time + cfg.straighten_hold:
                        u, v = _pole_in_body_frame(data, pair_body_ids)
                        dist = [_radial(data.xpos[b_], cfg.pole_radius) for b_ in pair_body_ids]
                        n0 = int(np.argmin(dist))
                        if cfg.catch_u_min <= u <= cfg.catch_u_max and 0.0 < v <= cfg.catch_v_max:
                            # The gait leaves the body rolled about its own
                            # axis; the ring's phase is taken relative to the
                            # body, so subtract that roll to close it on the
                            # pole side and screwing upward.
                            roll = _body_roll(data, roll_ref)
                            psi0 = cfg.wrap_psi - roll
                            catch_alpha, wave_delay, wrap_target = catch_shape(catch_attempts)
                            pair_trigger = [t + wave_delay * abs(n - n0)
                                            for n in range(N_PAIR)]
                            catch_pair = n0
                            curl_from = 0.0
                            say(f"t={t:6.2f}s  pole at u={u:.3f} m along the body, "
                                f"v={v:.3f} m beside it, body rolled "
                                f"{np.degrees(roll):.0f} deg; curling from pair {n0 + 1}")
                        else:
                            missed = (f"pole at u={u:.3f} m, v={v:.3f} m, outside the "
                                      f"catch window")
                else:
                    cmd = np.zeros(NJ)
                    for n in range(N_PAIR):
                        blend = min(1.0, max(0.0, (t - pair_trigger[n]) / cfg.ramp_per_pair))
                        blend = curl_from + (1.0 - curl_from) * blend
                        cmd[2 * n:2 * n + 2] = blend * wrap_target[2 * n:2 * n + 2]

                locked = all(x is not None and t - x >= cfg.ramp_per_pair for x in pair_trigger)
                if locked and all_locked is None:
                    all_locked = t
                    grip_samples = []
                    switch('settling', t)
                    say(f"t={t:6.2f}s  all {N_PAIR} pairs wrapped; settling")

                # Never hand a coil that is not properly round the pole to the
                # climbing controller -- let go and make another pass instead.
                if all_locked is not None:
                    if step % 10 == 0 and t - all_locked > cfg.settle_hold - 1.0:
                        grip_samples.append(_pole_contacts(data, pipe_gid))
                    if t - all_locked > cfg.settle_hold:
                        grip = float(np.mean(grip_samples)) if grip_samples else 0.0
                        com_r = float(np.hypot(*data.subtree_com[0, :2]))
                        nodes = backbone.nodes(model, data)
                        turns = _turns_round_pole(nodes)
                        tilt = float(np.degrees(np.arccos(min(1.0, abs(
                            hr.fit_helix(nodes)['axis'][2])))))
                        if grip >= cfg.min_grip_contacts and com_r < cfg.grip_com_radius \
                                and turns >= cfg.min_wrap_turns and tilt <= cfg.max_catch_tilt:
                            switch('pitch_ramp', t)
                            loop_start = roll_start = t
                            last_control = -1e9
                            say(f"t={t:6.2f}s  gripping ({grip:.1f} pole contacts, "
                                f"{turns:.2f} turns round it, axis {tilt:.0f} deg off "
                                f"vertical); adaptive helical rolling starts")
                        else:
                            missed = (f"coil not usable ({grip:.1f} pole contacts, "
                                      f"{turns:.2f} turns round it, axis {tilt:.0f} deg off "
                                      f"vertical, centre {com_r:.2f} m away)")
                if missed:
                    catch_attempts += 1
                    if catch_attempts >= cfg.max_catch_attempts:
                        say(f"t={t:6.2f}s  STOPPED: missed the pole {catch_attempts} times "
                            f"({missed})")
                        break
                    at_pole = all_locked is not None and \
                        np.hypot(*data.subtree_com[0, :2]) < cfg.grip_com_radius
                    pair_trigger = [None] * N_PAIR
                    all_locked = None
                    if at_pole:
                        # The coil is round the pole but came out crooked. How
                        # it folds is chaotic, so loosen it part way and close
                        # it again. Straightening fully would slide the body
                        # along its length, out of the catch window.
                        say(f"t={t:6.2f}s  missed: {missed}; loosening to try again")
                        reopen_start, reopen_from = t, cmd.copy()
                        catch_alpha, wave_delay, wrap_target = catch_shape(catch_attempts)
                        switch('catching', t)
                    else:
                        say(f"t={t:6.2f}s  missed: {missed}; letting go and trying again")
                        release_t, release_cmd = t, cmd.copy()
                        switch('approach', t)

            else:   # the paper's loop: pitch_ramp / rolling / holding, and
                    # 'settling' too when the robot was placed on the pole
                ramp = 0.0 if cfg.start_wrapped else cfg.alpha_ramp
                frac = 1.0 if ramp <= 0 else min(1.0, (t - loop_start) / ramp)
                controller.alpha = catch_alpha + (cfg.alpha - catch_alpha) * frac
                if stage in ('settling', 'pitch_ramp') and frac >= 1.0 and t >= roll_start:
                    switch('rolling', t)
                    if cfg.start_wrapped:
                        say(f"t={t:6.2f}s  rolling starts")

                if not psi_frozen:
                    if data.subtree_com[0, 2] < cfg.stop_height:
                        psi_roll = psi0 + cfg.spin * cfg.psi_dot * max(0.0, t - roll_start)
                    else:
                        psi_frozen = True
                        top_time = t
                        switch('holding', t)
                        say(f"t={t:6.2f}s  reached the top "
                            f"({data.subtree_com[0, 2]:.2f} m); holding on")

                if (t - last_control) >= cfg.control_period:
                    last_control = last_setpoint = t
                    cmd, info = controller.target(backbone.nodes(model, data), psi_roll)
                    cmd = np.clip(cmd, jnt_lo, jnt_hi)
                elif (t - last_setpoint) >= cfg.setpoint_period:
                    # Between feedback ticks, keep rolling the body round the
                    # last estimated form instead of holding the setpoint and
                    # jumping it by psi_dot * control_period at the next tick.
                    last_setpoint = t
                    cmd = np.clip(controller.retarget(psi_roll), jnt_lo, jnt_hi)

                    # Watchdog for a catch that passed the grip check but is
                    # peeling off in the first moments of the climb: the
                    # estimated axis keels over or the pole is let go of,
                    # while still near the floor. Go back and re-close.
                    if not cfg.start_wrapped and stage in ('pitch_ramp', 'rolling') \
                            and data.subtree_com[0, 2] < cfg.climb_abort_below:
                        tilt_now = float(np.degrees(np.arccos(min(1.0, abs(info['axis'][2])))))
                        bad = tilt_now > cfg.climb_abort_tilt or \
                            _pole_contacts(data, pipe_gid) == 0
                        bad_since = (bad_since or t) if bad else None
                        if bad_since is not None and t - bad_since > cfg.climb_abort_after:
                            catch_attempts += 1
                            bad_since = None
                            why = (f"coil peeling off at the start of the climb (axis "
                                   f"{tilt_now:.0f} deg off vertical)")
                            if catch_attempts >= cfg.max_catch_attempts:
                                say(f"t={t:6.2f}s  STOPPED: missed the pole "
                                    f"{catch_attempts} times ({why})")
                                break
                            say(f"t={t:6.2f}s  missed: {why}; loosening to try again")
                            reopen_start, reopen_from = t, cmd.copy()
                            catch_alpha, wave_delay, wrap_target = catch_shape(catch_attempts)
                            pair_trigger = [None] * N_PAIR
                            all_locked = None
                            info = None
                            switch('catching', t)

                if top_time is not None and (t - top_time) > cfg.hold_time:
                    say("finished at the top")
                    break

            # ---------------- step ----------------
            data.ctrl[:NJ] = cmd
            mujoco.mj_step(model, data)

            # rotation bookkeeping, integrated every step
            R = data.xmat[mid_body].reshape(3, 3)
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY,
                                     mid_body, vel6, 0)
            roll_own += float(vel6[0:3] @ R[:, 0]) * dt
            ang = float(np.arctan2(data.xpos[mid_body][1], data.xpos[mid_body][0]))
            if prev_ang is not None:
                orbit += (ang - prev_ang + np.pi) % (2 * np.pi) - np.pi
            prev_ang = ang

            peak_z = max(peak_z, float(data.subtree_com[0, 2]))

            if not np.all(np.isfinite(data.qpos)) or np.max(np.abs(data.qvel)) > qvel_limit:
                say(f"t={t:6.2f}s  STOPPED: the solver diverged "
                    f"(max |qvel| = {np.max(np.abs(data.qvel)):.0f})")
                diverged = True
                break

            # ---------------- logging ----------------
            if step % log_every == 0:
                # The immersive viewer's own slider is authoritative when it
                # is showing -- it's the control the operator can actually
                # see and drag. Otherwise fall back to the Tkinter slider
                # (or whatever else `live_mu` reads from).
                requested = None
                if viewer is not None and hasattr(viewer, 'mu'):
                    requested = float(viewer.mu)
                elif live_mu is not None:
                    requested = float(live_mu())
                if requested is not None and abs(requested - mu_now) > 1e-9:
                    mu_now = requested
                    set_friction(model, mu_now, floor_gid)

                z = float(data.subtree_com[0, 2])
                rate = 0.0
                if prev_log_t is not None and t > prev_log_t:
                    rate = (z - prev_log_z) / (t - prev_log_t) * 100.0
                prev_log_t, prev_log_z = t, z

                n_pole, n_self, n_floor, fn_sum, mu95, mumax = contact_report(
                    model, data, pipe_gid, floor_gid)

                tq = data.qfrc_actuator[6:6 + NJ]          # delivered, clamped
                tq_cmd = data.actuator_force[:NJ]             # demanded, unclamped
                sat = np.abs(tq_cmd) > jnt_frc_lim * 1.001
                qv = data.qvel[6:6 + NJ]
                if info is not None:
                    axis = info['axis']
                    tilt = float(np.degrees(np.arccos(np.clip(abs(axis[2]), 0.0, 1.0))))
                    ax_cols = [f"{axis[0]:.5f}", f"{axis[1]:.5f}", f"{axis[2]:.5f}",
                               f"{tilt:.3f}"]
                else:
                    # The controller is not running yet, so there is no
                    # estimated axis. Leave the cells empty rather than
                    # writing a placeholder that would plot as real data.
                    tilt = float('nan')
                    ax_cols = ['', '', '', '']

                rw.writerow([
                    f"{t:.4f}", step, stage, f"{t - stage_start:.4f}", f"{mu_now:.4f}",
                    f"{data.subtree_com[0,0]:.5f}", f"{data.subtree_com[0,1]:.5f}", f"{z:.5f}",
                    f"{rate:.3f}", f"{psi_roll:.5f}", f"{controller.alpha:.5f}",
                    *ax_cols,
                    f"{info['radius_measured']:.5f}" if info else '',
                    f"{info['radius_target']:.5f}" if info else '',
                    f"{(info['radius_measured']-info['radius_target'])*1000:.3f}" if info else '',
                    f"{info['turns']:.4f}" if info else '',
                    n_pole, n_self, n_floor, f"{fn_sum:.4f}", f"{mu95:.4f}", f"{mumax:.4f}",
                    f"{roll_own/(2*np.pi):.4f}", f"{orbit/(2*np.pi):.4f}",
                    f"{np.max(np.abs(data.qvel)):.3f}",
                    f"{np.sqrt(np.mean(tq**2)):.5f}", f"{np.max(np.abs(tq)):.5f}",
                    f"{float(np.sum(np.abs(tq*qv))):.5f}", int(sat.sum()),
                    sum(1 for x in pair_trigger if x is not None),
                ])

                pos = data.qpos[7:7 + NJ]
                for i in range(NJ):
                    span = jnt_hi[i] - jnt_lo[i]
                    jw.writerow([
                        f"{t:.4f}", step, stage, i + 1, f"J{i+1}", i // 2 + 1, axis_kind[i],
                        f"{cmd[i]:.6f}", f"{pos[i]:.6f}", f"{cmd[i]-pos[i]:.6f}",
                        f"{qv[i]:.5f}", f"{tq[i]:.6f}", f"{tq_cmd[i]:.6f}",
                        f"{tq[i]/jnt_frc_lim[i]:.4f}" if jnt_frc_lim[i] else '',
                        int(sat[i]),
                        f"{(pos[i]-jnt_lo[i])/span:.4f}" if span else '',
                    ])

                live = {
                    'stage': stage, 't': t, 'z': z, 'rate_cm_s': rate,
                    'mu_robot_pole': mu_now,
                    'n_pole': n_pole, 'n_self': n_self, 'mu_p95': mu95,
                    'coil_r': info['radius_measured'] if info else float('nan'),
                    'squeeze_mm': ((info['radius_measured'] - info['radius_target']) * 1000
                                   if info else float('nan')),
                    'axis_tilt': tilt, 'peak_z': peak_z,
                    'saturated': int(sat.sum()),
                    'wall': time.perf_counter() - wall0,
                }
                if on_progress:
                    on_progress(live)
                if viewer is not None and hasattr(viewer, 'update_experiment'):
                    viewer.update_experiment(live, info)

                if should_stop and should_stop():
                    say("stopped by the operator")
                    stopped_early = True
                    break

            # ---------------- viewer ----------------
            if viewer is not None and viewer.is_alive and step % render_every == 0:
                try:
                    viewer.render()
                except Exception as exc:
                    # Say what went wrong rather than quietly finishing the run
                    # headless: a broken annotation used to look exactly like a
                    # closed window.
                    say(f"viewer stopped rendering ({type(exc).__name__}: {exc}); "
                        f"continuing headless")
                    viewer = None
            if viewer is not None and not viewer.is_alive:
                say("viewer closed")
                stopped_early = True
                break

    except KeyboardInterrupt:
        say("interrupted")
        stopped_early = True
    finally:
        stage_log.append((stage, stage_start, t, stage_z0, float(data.subtree_com[0, 2])))
        jf.close()
        rf.close()
        if viewer is not None:
            try:
                viewer.close()
            except Exception:
                pass

    # -- stage and summary CSVs -------------------------------------------
    with open(os.path.join(run_dir, 'stages.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['stage', 't_start_s', 't_end_s', 'duration_s',
                    'com_z_start_m', 'com_z_end_m', 'climb_m', 'mean_rate_cm_s'])
        for name, t0, t1, z0, z1 in stage_log:
            dur = t1 - t0
            w.writerow([name, f"{t0:.3f}", f"{t1:.3f}", f"{dur:.3f}",
                        f"{z0:.4f}", f"{z1:.4f}", f"{z1-z0:+.4f}",
                        f"{(z1-z0)/dur*100:.3f}" if dur > 1e-9 else ''])

    climb = {name: z1 - z0 for name, t0, t1, z0, z1 in stage_log}
    dur = {name: t1 - t0 for name, t0, t1, z0, z1 in stage_log}
    summary = dict(
        run_dir=run_dir,
        sim_time_s=round(t, 3),
        wall_time_s=round(time.perf_counter() - wall0, 2),
        peak_com_z_m=round(peak_z, 4),
        final_com_z_m=round(float(data.subtree_com[0, 2]), 4),
        reached_top=top_time is not None,
        time_to_top_s=round(top_time, 3) if top_time else '',
        rolling_climb_m=round(climb.get('rolling', 0.0), 4),
        rolling_duration_s=round(dur.get('rolling', 0.0), 3),
        rolling_rate_cm_s=round(climb.get('rolling', 0.0) / dur['rolling'] * 100, 3)
        if dur.get('rolling', 0) > 1e-9 else '',
        pairs_wrapped=sum(1 for x in pair_trigger if x is not None),
        caught_pole=loop_start is not None,
        catch_misses=catch_attempts,
        mu_at_end=round(mu_now, 4),
        body_roll_rev=round(roll_own / (2 * np.pi), 3),
        orbit_rev=round(orbit / (2 * np.pi), 3),
        diverged=diverged,
        stopped_early=stopped_early,
    )
    with open(os.path.join(run_dir, 'summary.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['key', 'value'])
        for k, v in dataclasses.asdict(cfg).items():
            w.writerow([f"config.{k}", v])
        for k, v in summary.items():
            w.writerow([f"result.{k}", v])
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(dataclasses.asdict(cfg), f, indent=2)

    say(f"peak height {peak_z:.3f} m"
        + (f", rolling stage {summary['rolling_rate_cm_s']} cm/s"
           if summary['rolling_rate_cm_s'] != '' else ''))
    return summary


def _patch_viewer_camera():
    """mujoco-python-viewer 0.1.4 still calls mjv_moveCamera with a scene
    argument that current MuJoCo no longer takes, so every mouse drag or
    scroll in the viewer raised TypeError. Accept both forms."""
    move = mujoco.mjv_moveCamera
    if getattr(move, '_accepts_scene', False):
        return

    def move_camera(m, action, reldx, reldy, *scn_cam):
        return move(m, action, reldx, reldy, scn_cam[-1])

    move_camera._accepts_scene = True
    mujoco.mjv_moveCamera = move_camera


def _radial(pos, pole_radius):
    return float(np.hypot(pos[0], pos[1]) - pole_radius)


def _pole_in_body_frame(data, pair_body_ids):
    """The pole axis in the body's own frame: u along the body from pair 1
    toward pair 9, v to the body's left (+z up)."""
    p0 = data.xpos[pair_body_ids[0]][:2]
    ax = data.xpos[pair_body_ids[-1]][:2] - p0
    ax = ax / np.linalg.norm(ax)
    rel = -p0
    return float(rel @ ax), float(ax[0] * rel[1] - ax[1] * rel[0])


def _turns_round_pole(nodes):
    """Winding number of the backbone about the pole axis (x = y = 0)."""
    ang = np.unwrap(np.arctan2(nodes[:, 1], nodes[:, 0]))
    return float(abs(ang[-1] - ang[0]) / (2 * np.pi))


def _roll_reference(model):
    """Pair 1's long axis and 'up', in its own frame, with the body lying flat."""
    probe = mujoco.MjData(model)
    probe.qpos[3:7] = _Q_LYING_FLAT
    mujoco.mj_forward(model, probe)
    b1, b9 = model.body('M1').id, model.body(f'M{N_PAIR}').id
    R = probe.xmat[b1].reshape(3, 3)
    axis = probe.xpos[b9] - probe.xpos[b1]
    return b1, b9, R.T @ (axis / np.linalg.norm(axis)), R.T @ np.array([0.0, 0.0, 1.0])


def _body_roll(data, ref):
    """Roll of a straight body about its own long axis, relative to lying flat."""
    b1, b9, _, up_local = ref
    a = data.xpos[b9] - data.xpos[b1]
    a /= np.linalg.norm(a)
    up = data.xmat[b1].reshape(3, 3) @ up_local
    z = np.array([0.0, 0.0, 1.0])
    z_perp = z - (z @ a) * a
    return float(np.arctan2(np.cross(z_perp, up) @ a, z_perp @ up))


def _pole_contacts(data, pipe_gid):
    return sum(1 for i in range(data.ncon)
               if data.contact[i].geom1 == pipe_gid or data.contact[i].geom2 == pipe_gid)


def _make_run_dir(cfg: Config):
    stamp = _dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    name = f"{stamp}_{cfg.label}" if cfg.label else stamp
    path = os.path.join(cfg.out_dir, name)
    os.makedirs(path, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
_ALIASES = {'mu': 'mu_robot_pole', 'mass': 'mass_scale', 'dt': 'timestep',
            'fps': 'target_fps', 'speed': 'target_speedup'}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # `from __future__ import annotations` leaves dataclass field types as
    # strings, so map them back to callables by name.
    casts = {'float': float, 'int': int, 'str': str,
             'bool': lambda s: s.lower() in ('1', 'true', 'yes', 'on')}
    fields = {f.name: f for f in dataclasses.fields(Config)}
    for name, f in fields.items():
        p.add_argument(f'--{name}', type=casts[f.type], default=None)
    for short, full in _ALIASES.items():
        p.add_argument(f'--{short}', type=casts[fields[full].type], default=None,
                       dest=full, help=f'alias for --{full}')
    p.add_argument('--viewer', action='store_true', help='alias for --show_viewer true')
    args = p.parse_args()

    cfg = Config()
    for name in fields:
        v = getattr(args, name, None)
        if v is not None:
            setattr(cfg, name, v)
    if args.viewer:
        cfg.show_viewer = True

    run(cfg, on_message=print)


if __name__ == '__main__':
    main()
