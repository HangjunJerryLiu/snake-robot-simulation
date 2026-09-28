"""
Solve for the sign convention that eq. (32) takes on THIS robot.

Eq. (32) of Takemori et al. (2023) reads

    theta_i = INT -kappa(s) sin psi(s) ds   (i odd)
              INT +kappa(s) cos psi(s) ds   (i even)

for a robot whose odd joints are yaw joints and whose even joints are pitch
joints, in a particular link frame. Our model numbers its joints from the
TAIL, defines every joint about its own local z, and alternates the link
frames by +-90 degrees about x. Whether our J1 plays the paper's "odd" role,
and with which sign, therefore follows from the model's layout rather than
from the method -- and getting it wrong yields a mirrored or rotated shape
that looks plausible until the robot is asked to grip something.

So rather than derive it, solve it: command a NORMAL HELIX of known radius
and pitch angle through (32), run forward kinematics, fit a helix to the
result, and keep the convention whose output matches what was asked for.

Run:  python calibrate_helical.py
Then paste the printed Orientation(...) into _ORIENT in helical_rolling.py.
"""

import itertools

import mujoco
import numpy as np

import helical_rolling as hr
from snake_backbone import Backbone

MODEL = '9motor_sidewinder_pipe.xml'

# Test shapes: a couple of radii and pitch angles spanning what the climb
# actually uses, at several roll phases and both handednesses. A convention
# that is right for one of these and wrong for another is not the one we want.
RADII = (0.085, 0.110)
ALPHAS = (0.20, 0.30)
PSI_ROLLS = (0.0, 0.7, 1.9, 3.4, 5.0)
LEADS = (+1.0, -1.0)


def _backbone_for(model, data, bb, joints):
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = 0.0
    data.qpos[3] = 1.0
    data.qpos[4:7] = 0.0
    data.qpos[7:7 + bb.n_joint] = joints
    mujoco.mj_forward(model, data)
    return bb.nodes(model, data)


def score(model, data, bb, orient, verbose=False):
    """Total mismatch between the commanded helix and the one FK produces."""
    total = 0.0
    rows = []
    for radius, alpha, lead, psi in itertools.product(RADII, ALPHAS, LEADS, PSI_ROLLS):
        ctl = hr.HelicalRollingController(bb.link_lengths, alpha, 0.0)
        q = ctl.normal_helix_target(radius, psi, lead_sign=lead, orient=orient)
        if np.max(np.abs(q)) > 1.4:            # outside the joint limits
            total += 10.0
            continue
        fit = hr.fit_helix(_backbone_for(model, data, bb, q))
        e_r = abs(fit['radius'] - radius) / radius
        e_a = abs(fit['pitch_angle'] - alpha) / alpha
        e_h = 0.0 if np.sign(fit['lead']) == lead else 2.0
        total += e_r + e_a + e_h + fit['residual'] / radius
        rows.append((radius, alpha, lead, psi, fit['radius'], fit['pitch_angle'],
                     fit['lead'], fit['turns'], fit['residual']))
    if verbose:
        print(f"{'R_cmd':>7} {'a_cmd':>6} {'lead':>5} {'psi':>5} | "
              f"{'R_fit':>7} {'a_fit':>6} {'lead_fit':>9} {'turns':>6} {'resid':>7}")
        for r in rows:
            print(f"{r[0]:7.3f} {r[1]:6.2f} {r[2]:5.0f} {r[3]:5.2f} | "
                  f"{r[4]:7.3f} {r[5]:6.2f} {r[6]:+9.4f} {r[7]:6.2f} {r[8]:7.4f}")
    return total


def main():
    model = mujoco.MjModel.from_xml_path(MODEL)
    data = mujoco.MjData(model)
    bb = Backbone(model)
    print(bb)
    print()

    results = []
    for parity, sign, psi_sign, off in itertools.product(
            (False, True), (1.0, -1.0), (1.0, -1.0),
            (0.0, np.pi / 2, np.pi, 3 * np.pi / 2)):
        o = hr.Orientation(parity, sign, psi_sign, off)
        results.append((score(model, data, bb, o), o))
    results.sort(key=lambda r: r[0])

    print("Best conventions (lower is better):")
    for s, o in results[:6]:
        print(f"  {s:9.3f}  {o}")
    print()
    print("Detail for the winner:")
    score(model, data, bb, results[0][1], verbose=True)
    print()
    print("Paste into helical_rolling.py:")
    print(f"  _ORIENT = hr.{results[0][1]}".replace("hr.", ""))


if __name__ == '__main__':
    main()
