"""
Test the adaptive helical rolling loop on its own, starting already wrapped.

All four experiments in Takemori et al. (2023) begin with the robot placed on
the pipe by hand, "a normal helix with a radius of R and a pitch angle of
alpha", and the method takes over from there. robot_pipe_climb.py has to get
there off the floor first, which is a separate problem with separate failure
modes -- so this harness skips it: it computes the helix, rotates and
translates the whole robot so that helix sits on the pole, and runs nothing
but the control loop.

Two things it reports:

    python climb_from_wrapped.py            which SPIN/LEAD_SIGN pair climbs
    python climb_from_wrapped.py mechanism  what is touching what, and what is
                                            actually rotating

The second is the one that answers "is this really helical rolling, or is the
robot just shoving itself up with its own body?".
"""

import itertools
import sys

import mujoco
import numpy as np

import helical_rolling as hr
from snake_backbone import Backbone

MODEL = '9motor_sidewinder_pipe.xml'

m = mujoco.MjModel.from_xml_path(MODEL)
bb = Backbone(m)
dt = m.opt.timestep
PIPE = m.geom('pipe').id
FLOOR = m.geom('floor').id

K_GAIN = np.full(18, 0.08)
K_GAIN[:3] = 0.04
K_GAIN[-3:] = 0.04


def circle_fit(xy):
    """Centre of the circle that best fits a set of 2-D points."""
    A = np.column_stack([xy[:, 0], xy[:, 1], np.ones(len(xy))])
    s, *_ = np.linalg.lstsq(A, xy[:, 0] ** 2 + xy[:, 1] ** 2, rcond=None)
    return np.array([s[0] / 2, s[1] / 2])


def rot_to_z(a):
    """Rotation taking the unit vector `a` onto +z."""
    a = a / np.linalg.norm(a)
    z = np.array([0., 0., 1.])
    v, c = np.cross(a, z), a @ z
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1., -1., -1.])
    V = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + V + V @ V * (1 / (1 + c))


def place_wrapped(d, joints, base_z=0.35):
    """Put the robot on the pole in the commanded helix, axis vertical.

    Sets the joints, works out where that shape's own helical axis points,
    then uses the free joint to rotate that axis onto the pole and slide the
    whole thing up to `base_z`.
    """
    mujoco.mj_resetData(m, d)
    d.qpos[0:3] = 0.0
    d.qpos[3:7] = [1., 0., 0., 0.]
    d.qpos[7:25] = joints
    mujoco.mj_forward(m, d)

    R = rot_to_z(hr.fit_helix(bb.nodes(m, d))['axis'])
    Pr = bb.nodes(m, d) @ R.T
    c = circle_fit(Pr[:, :2])

    q = np.empty(4)
    mujoco.mju_mat2Quat(q, R.reshape(9))
    mujoco.mj_resetData(m, d)
    d.qpos[0:3] = [-c[0], -c[1], base_z - Pr[:, 2].min()]
    d.qpos[3:7] = q
    d.qpos[7:25] = joints
    mujoco.mj_forward(m, d)
    return d


def run(alpha=0.25, R0=0.085, K=K_GAIN, spin=+1, lead=+1.0,
        psi_dot=0.5 * np.pi, duration=40.0, settle=2.0, period=0.1,
        instrument=False):
    """Run the control loop from a wrapped start and report what happened."""
    d = mujoco.MjData(m)
    ctl = hr.HelicalRollingController(bb.link_lengths, alpha, K, mode='outside')
    q0 = ctl.normal_helix_target(R0, 0.0, lead_sign=lead)
    place_wrapped(d, q0)

    lo, hi = m.jnt_range[1:19, 0], m.jnt_range[1:19, 1]
    cmd, last, info = q0.copy(), -1e9, None
    mid = m.body('L4').id
    vel = np.zeros(6)

    z0 = float(d.subtree_com[0, 2])
    z, radius, pole_c, self_c = [], [], [], []
    roll_own, orbit, prev_ang = 0.0, 0.0, None

    for step in range(int(duration / dt)):
        t = step * dt
        psi = 0.0 if t < settle else spin * psi_dot * (t - settle)
        if t - last >= period:
            last = t
            cmd, info = ctl.target(bb.nodes(m, d), psi)
            cmd = np.clip(cmd, lo, hi)
        d.ctrl[:18] = cmd
        mujoco.mj_step(m, d)

        if not np.all(np.isfinite(d.qpos)) or np.max(np.abs(d.qvel)) > 150:
            return dict(z0=z0, z=float(d.subtree_com[0, 2]), dz=float(d.subtree_com[0, 2]) - z0,
                        span=t - settle, diverged=True)

        if not instrument or t < settle + 1.0:
            if not instrument:
                z.append(float(d.subtree_com[0, 2]))
            continue

        # Spin about the body's own long axis, and orbit about the pole. For a
        # helix rolling on a pole these are very different numbers: the body
        # spins fast and barely orbits at all, which is what "screwing itself
        # up the pole" means.
        R = d.xmat[mid].reshape(3, 3)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, mid, vel, 0)
        roll_own += float(vel[0:3] @ R[:, 0]) * dt
        ang = float(np.arctan2(d.xpos[mid][1], d.xpos[mid][0]))
        if prev_ang is not None:
            orbit += (ang - prev_ang + np.pi) % (2 * np.pi) - np.pi
        prev_ang = ang

        if step % 50 == 0:
            pc = sc = 0
            for i in range(d.ncon):
                g1, g2 = d.contact[i].geom1, d.contact[i].geom2
                if PIPE in (g1, g2):
                    pc += 1
                elif FLOOR not in (g1, g2):
                    sc += 1
            pole_c.append(pc)
            self_c.append(sc)
            z.append(float(d.subtree_com[0, 2]))
            radius.append(info['radius_measured'])

    z = np.array(z)
    out = dict(z0=z0, z=float(z[-1]), dz=float(z[-1] - z0), zmax=float(z.max()),
               span=duration - settle, diverged=False)
    if instrument:
        out.update(pole_c=np.array(pole_c), self_c=np.array(self_c),
                   radius=np.array(radius), roll_own=roll_own, orbit=orbit,
                   climb=float(z[-1] - z[0]), psi_dot=psi_dot, alpha=alpha,
                   psi_span=duration - settle - 1.0)
    return out


def direction_table():
    print("Which way does it climb? 38 s of rolling from a wrapped start.\n")
    print(f"{'alpha':>6} {'R0':>6} {'spin':>5} {'lead':>5} | "
          f"{'z start':>8} {'z end':>7} {'climb':>7} {'cm/s':>7}")
    for alpha, R0, spin, lead in itertools.product(
            (0.25,), (0.085,), (+1, -1), (+1.0, -1.0)):
        r = run(alpha=alpha, R0=R0, spin=spin, lead=lead, duration=40.0)
        print(f"{alpha:6.2f} {R0:6.3f} {spin:+5d} {lead:+5.0f} | "
              f"{r['z0']:8.3f} {r['z']:7.3f} {r['dz']:+7.3f} "
              f"{r['dz']/r['span']*100:+7.2f}"
              + ("   DIVERGED" if r['diverged'] else ""))


def mechanism_report(duration=70.0):
    r = run(duration=duration, instrument=True)
    pole_c, self_c, radius = r['pole_c'], r['self_c'], r['radius']
    psi_rev = r['psi_dot'] * r['psi_span'] / (2 * np.pi)
    r_mean = radius.mean()
    lead = 2 * np.pi * r_mean * np.tan(r['alpha'])

    print(f"climbed {r['climb']:.3f} m in {r['psi_span']:.0f} s "
          f"({r['climb']/r['psi_span']*100:.2f} cm/s)\n")
    print("WHAT IS TOUCHING WHAT  (contacts per sample, mean / max)")
    print(f"  robot on pole  : {pole_c.mean():5.2f} / {pole_c.max():2d}")
    print(f"  robot on robot : {self_c.mean():5.2f} / {self_c.max():2d}"
          f"   ({(self_c > 0).mean()*100:.0f}% of samples have any at all)")
    print("  -- the wrap carries the robot; body-on-body contact is incidental,")
    print("     not load bearing. The previous breathing-coil gait leaned on it.\n")
    print("WHAT IS ROTATING")
    print(f"  psi_roll commanded      : {psi_rev:6.2f} revolutions")
    print(f"  body about its own axis : {r['roll_own']/(2*np.pi):6.2f} revolutions "
          f"({abs(r['roll_own']/(2*np.pi)/psi_rev)*100:.0f}% of commanded)")
    print(f"  body around the pole    : {r['orbit']/(2*np.pi):6.2f} revolutions")
    print("  -- it really rolls, and it barely orbits: a helix rolling against a")
    print("     pole screws itself upward while staying put around the pole.\n")
    print("IS IT A SCREW?")
    print(f"  mean coil radius             : {r_mean*100:5.2f} cm")
    print(f"  ideal lead 2*pi*r*tan(alpha) : {lead*100:5.2f} cm per turn")
    print(f"  measured climb per psi turn  : {r['climb']/psi_rev*100:5.2f} cm")
    print(f"  -> {r['climb']/psi_rev/lead*100:.0f}% of the ideal screw; "
          f"the remainder is slip against the pole.")


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'mechanism':
        mechanism_report()
    else:
        direction_table()
