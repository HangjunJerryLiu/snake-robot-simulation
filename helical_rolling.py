"""
Adaptive helical rolling of a snake robot to a straight pipe.

Implementation of the control method described in

    T. Takemori, M. Tanaka and F. Matsuno, "Adaptive Helical Rolling of a
    Snake Robot to a Straight Pipe With Irregular Cross-Sectional Shape",
    IEEE Transactions on Robotics, vol. 39, no. 1, pp. 437-451, Feb. 2023.

Equation and section numbers in the comments below refer to that paper.

WHAT THE METHOD DOES, IN ONE PARAGRAPH
--------------------------------------------------------------------------
The robot is held in a *helical form*, which the paper decomposes into three
independent components (Fig. 4): a straight AXIS, a CROSS-SECTIONAL SHAPE
(the form projected onto the plane perpendicular to that axis), and a PITCH
ANGLE (how steeply the body rises out of that plane). Every control tick the
controller reads the joint angles, reconstructs the form, estimates the axis,
projects the form down to a 2-D cross-sectional curve, squeezes that curve a
little tighter than it currently is, re-inflates it into a 3-D helix at a
FIXED pitch angle, and solves for the joint angles that approximate it. The
pitch angle never gets a compliance term, so the robot is compliant only
perpendicular to the pipe -- it can change how tightly it grips without the
grip slipping along the pipe. Climbing then comes from one extra scalar:
psi_roll in (33), which rolls the whole body around the target form. That is
the "helical rolling" motion, and it screws the robot along the pipe.

WHAT IT IS NOT
--------------------------------------------------------------------------
There is no clamp/release cycle, no ratchet, and no part of the body pushing
against another part. The commanded form is a single continuous helix at all
times; only its cross-sectional shape breathes, and only in response to what
the pipe is doing to the robot.

THE SEVEN STEPS (Section III-B, Figs. 5 and 6), and where each one lives
--------------------------------------------------------------------------
  1) form of the robot from the joint angles      caller -> backbone_nodes()
  2) estimate the helical axis                    estimate_axis()
  3) project onto the plane perpendicular to it   _cross_section()
  4) approximate the projection with lines+arcs   _fit_cross_section()
  5) deform it for compliance                     _comply()       (23), (24)
  6) helicalize: give it the pitch angle          _helicalize()   (25)-(31)
  7) approximate with the discrete robot          _approximate()  (32), (33)
"""

import numpy as np


# ---------------------------------------------------------------------------
# Small vector helpers
# ---------------------------------------------------------------------------
def _unit(v):
    n = np.linalg.norm(v)
    return np.asarray(v, float) / n if n > 1e-12 else np.array([0.0, 0.0, 1.0])


def _frame(axis):
    """A right-handed frame (u, v, a) with `a` along `axis`, so u x v = a.

    The paper says the orientation of the x- and y-axes of Sigma_H does not
    matter (Section III-B2), and it does not -- but the frame must be
    right-handed, otherwise the sign of every turn angle gamma_i flips and
    with it the handedness of the helix handed back.
    """
    a = _unit(axis)
    ref = np.array([1.0, 0.0, 0.0])
    if abs(a @ ref) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    u = _unit(np.cross(ref, a))
    return u, np.cross(a, u), a


def _signed_turn(a, b):
    """Signed angle from 2-D vector a to 2-D vector b, positive = CCW.

    This is (14)-(16) written in one line. The paper's atan2(x, y) ordering
    measures clockwise from +y; the sign convention it implies is recovered
    by the winding factor w in _fit_cross_section(), which is read off the
    shape instead of being assumed.
    """
    return np.arctan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1])


def _fibonacci_hemisphere(n):
    i = np.arange(n) + 0.5
    z = 1.0 - i / n
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
    ang = np.pi * (1.0 + 5.0 ** 0.5) * i
    return np.stack([r * np.cos(ang), r * np.sin(ang), z], axis=1)


_AXIS_SEEDS = _fibonacci_hemisphere(500)


# ---------------------------------------------------------------------------
# Step 2 - estimate the axis of the helical form  (Section III-B2, (5)-(9))
# ---------------------------------------------------------------------------
#
# The paper's criterion: assume a candidate unit vector v_temp is the helical
# axis, measure the local pitch angle alpha_hat_i of every part of the body
# against it (7), and pick the v_temp that makes those pitch angles as uniform
# as possible -- i.e. minimises their variance V(k, v_temp). That works
# precisely because the target form this controller commands has a CONSTANT
# pitch angle, so the true axis is the one about which the body looks uniform.
#
# (7)-(9) simplify. With vpxy = vp - (vp.a)a, the angle between vp and vpxy is
#
#       alpha_hat = arccos( |vpxy| / |vp| ) = arcsin( |vp.a| / |vp| )
#
# which is just "the angle vp makes with the plane perpendicular to a" -- the
# definition of the pitch angle. Using arcsin turns the whole candidate sweep
# into one matrix product.
def _pitch_angles(vp, axes):
    """alpha_hat for every (body part, candidate axis) pair.  (7)-(9)"""
    nvp = np.linalg.norm(vp, axis=1)
    return np.arcsin(np.clip(np.abs(vp @ axes.T) / nvp[:, None], 0.0, 1.0))


def estimate_axis(nodes, seed=None):
    """Axis of the helical form, estimated from the robot's form alone.

    `nodes` are the link-edge points 1p_i of (3), ordered head -> tail. The
    returned axis points toward the head side, as the paper requires.
    """
    P = np.asarray(nodes, float)
    P = P - P.mean(axis=0)                     # (5), (6): work about 1p_H

    # (8): skip one joint, because pitch-axis and yaw-axis joints alternate
    # and the form zigzags about the smooth helix. Differencing i-1 against
    # i+1 takes the two joints of a pair as a set, which the paper notes
    # measures the pitch angle far more accurately.
    vp = P[:-2] - P[2:]

    cands = _AXIS_SEEDS if seed is None else np.vstack([_unit(seed), _AXIS_SEEDS])
    var = _pitch_angles(vp, cands).var(axis=0)
    a = cands[int(np.argmin(var))]
    best = float(var.min())

    # Local pattern refinement in the tangent plane of the best candidate.
    step = 0.08
    for _ in range(5):
        u, v, a = _frame(a)
        offs = np.array([[du, dv] for du in (-step, 0.0, step)
                         for dv in (-step, 0.0, step)])
        cand = a + offs[:, 0:1] * u + offs[:, 1:2] * v
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        var = _pitch_angles(vp, cand).var(axis=0)
        j = int(np.argmin(var))
        if var[j] < best:
            best, a = float(var[j]), cand[j]
        step *= 0.4

    # "the head side of the snake is positive" (Section III-B2)
    return a if a @ (P[0] - P[-1]) >= 0 else -a


# ---------------------------------------------------------------------------
# Orientation convention for (32)
# ---------------------------------------------------------------------------
class Orientation(object):
    """Which joints count as 'odd' in (32), and the signs that go with them.

    Eq. (32) alternates -kappa*sin(psi) and +kappa*cos(psi) between odd and
    even joints. Which of THIS robot's joints plays the paper's "odd" role,
    and the overall sign, follow from how the model's joint axes are laid out
    -- not from the method. calibrate_helical.py solves for them, by commanding
    a helix of known radius and pitch and checking, through forward kinematics,
    which convention reproduces it.
    """

    __slots__ = ('parity_flip', 'sign', 'psi_sign', 'psi_offset')

    def __init__(self, parity_flip=False, sign=1.0, psi_sign=1.0, psi_offset=0.0):
        self.parity_flip = bool(parity_flip)
        self.sign = float(sign)
        self.psi_sign = float(psi_sign)
        self.psi_offset = float(psi_offset)

    def __repr__(self):
        return (f"Orientation(parity_flip={self.parity_flip}, sign={self.sign:+.0f}, "
                f"psi_sign={self.psi_sign:+.0f}, psi_offset={self.psi_offset:.4f})")


# Solved by calibrate_helical.py against 9motor_sidewinder_pipe.xml. Re-run
# that script if the model's joint axes or body quaternions ever change.
#
# The sweep splits cleanly: 16 of the 32 conventions score ~8, the other 16
# score ~88, the gap being entirely the handedness penalty. The 16 winners are
# all the same convention written differently (cos(psi + pi/2) = -sin(psi), and
# negating every joint angle is the same as psi -> psi + pi), so the single
# thing the calibration actually pins down is psi_sign = -1: this model's link
# frames twist the opposite way to the paper's, so psi accumulates -INT(tau)ds
# rather than +INT(tau)ds. Everything else is a free relabelling, and
# psi_offset is absorbed by psi_roll in any case.
_ORIENT = Orientation(parity_flip=False, sign=1.0, psi_sign=-1.0, psi_offset=0.0)


def set_orientation(orient):
    """Install the convention (32) is evaluated with, globally."""
    global _ORIENT
    _ORIENT = orient


def get_orientation():
    return _ORIENT


# ---------------------------------------------------------------------------
# The controller
# ---------------------------------------------------------------------------
class HelicalRollingController:
    """Turns the robot's measured form into the next target form.

    Parameters
    ----------
    link_lengths : (n_link,) float
        Length of every link, ordered HEAD -> TAIL. The paper assumes a single
        link length l; this robot alternates 24 mm motor links with 56 mm
        spacer links, so l appears here as a per-link array and every place
        the paper writes l or 2l uses the local value instead. Nothing else
        about the method changes.
    alpha : float
        Pitch angle of the target helix, radians. A design parameter chosen by
        the motion designer (Section III-B6); the paper used 0.20-0.30 rad.
        It is deliberately NOT adapted -- holding it constant is exactly what
        confines the compliance to the direction perpendicular to the pipe.
    K : float or (n_joint,) float
        Compliance gain per joint, (23). Larger = squeezes harder.
    mode : 'outside' | 'inside'
        Whether the robot is wrapped around a pipe or threaded through one.
    gamma_min : float
        Threshold separating cases A-C from case D, (22). Small enough that
        r_pi stays finite; the paper used 0.01*pi.
    ds : float
        Quadrature step for (32), in metres of arc length.
    """

    def __init__(self, link_lengths, alpha, K, mode='outside',
                 gamma_min=0.01 * np.pi, ds=1.0e-3):
        self.len = np.asarray(link_lengths, float)      # head -> tail
        self.n_link = len(self.len)
        self.n_joint = self.n_link - 1
        self.alpha = float(alpha)
        self.K = np.broadcast_to(np.asarray(K, float), (self.n_joint,)).astype(float).copy()
        if mode not in ('outside', 'inside'):
            raise ValueError("mode must be 'outside' or 'inside'")
        self.mode = mode
        self.gamma_min = float(gamma_min)
        self.ds = float(ds)

        # Arc-length coordinate of every node on the target curve. Node i sits
        # at the sum of the link lengths ahead of it, which is what makes the
        # integration limits of (32) come out to "one link either side".
        self.s_node = np.concatenate([[0.0], np.cumsum(self.len)])
        self.total_length = float(self.s_node[-1])

        self._axis = None          # warm start for the axis search

    # -- Step 3: project the form onto the plane perpendicular to the axis --
    # (10): v_i is 1p_i expressed in Sigma_H and dropped onto its xy-plane.
    @staticmethod
    def _cross_section(nodes, axis):
        P = np.asarray(nodes, float)
        P = P - P.mean(axis=0)
        u, v, a = _frame(axis)
        return np.column_stack([P @ u, P @ v]), P @ a

    # -- Which way does the current form spiral? ---------------------------
    #
    # The paper fixes the handedness implicitly, by fixing the axis direction
    # ("head side positive") and traversing the cross-sectional shape in one
    # direction. Deriving it instead is safer: a target form with the wrong
    # handedness is the mirror image of the shape the robot is actually in,
    # and commanding it would tear the robot off the pipe. So read the lead --
    # the rise along the axis per radian of turn about it -- off the measured
    # form, and give the new target the same sign.
    @staticmethod
    def _lead_sign(v2d, h):
        c = v2d - v2d.mean(axis=0)
        th = np.unwrap(np.arctan2(c[:, 1], c[:, 0]))
        if abs(th[-1] - th[0]) < 1e-9:
            return 1.0
        slope = np.polyfit(th, h, 1)[0]
        return 1.0 if slope >= 0 else -1.0

    # -- Step 4: approximate the projected form with lines and arcs --------
    #  (11)-(22) and Fig. 7. Returns, per interior node, the arc C_i and the
    #  line S_i that interpolate between the projected midpoints v_m(i-1) and
    #  v_mi, plus which of the two comes first along the body.
    def _fit_cross_section(self, v2d):
        vm = 0.5 * (v2d[:-1] + v2d[1:])                    # (11)

        a_vec = v2d[1:-1] - vm[:-1]                        # (12)  a_i
        b_vec = vm[1:] - v2d[1:-1]                         # (13)  b_i
        gamma = np.array([_signed_turn(a, b) for a, b in zip(a_vec, b_vec)])

        # Winding of the cross-sectional shape. The sign of gamma depends on
        # which way the loop is traversed; the paper's case split assumes
        # positive means "bending toward the inside of the helix", so measure
        # the winding once and fold it in rather than assume it.
        w = 1.0 if gamma.sum() >= 0 else -1.0
        g = w * gamma

        na = np.linalg.norm(a_vec, axis=1)
        nb = np.linalg.norm(b_vec, axis=1)

        sharp = g > self.gamma_min                         # cases A, B, C
        g_eff = np.where(sharp, g, self.gamma_min)
        r_p = np.where(sharp,
                       np.minimum(na, nb) / np.tan(g_eff / 2.0),   # (19)
                       (na + nb) / self.gamma_min)                 # (22)
        phi_p = g_eff                                      # (20) / (22)
        l_p = np.where(sharp, np.abs(na - nb), 0.0)        # (21); case D: none

        # Case A puts the straight part first (nearer v_m(i-1)), case B puts
        # it last, case C has none. Case D is arc-only and curves INWARD by
        # gamma_min whatever the links were doing -- that is the term that
        # makes the robot actively wrap rather than drape.
        line_first = na > nb

        end_lines = (0.5 * self.len[0], 0.5 * self.len[-1])   # (17), (18), (27)
        return r_p, phi_p, l_p, line_first, end_lines, w

    # -- Step 5: compliance  (23), (24) ------------------------------------
    #
    #  (r_hi, phi_hi) = (Khat_i * r_pi, phi_pi / Khat_i)
    #
    #  The arc keeps its length (r*phi is invariant) but changes how sharply
    #  it turns, so the cross-sectional curve keeps the robot's length while
    #  enclosing a different area. Khat < 1 sharpens every corner, the loop
    #  closes tighter, and the robot squeezes the pipe; Khat > 1 flattens the
    #  corners, the loop opens, and the robot presses outward.
    #
    #  NOTE ON (24). The paper prints Khat = 1-K for "inside" and 1+K for
    #  "outside". Taken literally that tightens the robot when it is inside a
    #  pipe and loosens it when wrapped around one -- the opposite of Fig. 8,
    #  which is unambiguous about the intent ("pushes the robot body outward
    #  when the robot snake is inside the pipe and tightens the robot body
    #  inward when the robot snake outside the pipe"). This follows Fig. 8,
    #  so the two labels are swapped with respect to (24).
    def _comply(self, r_p, phi_p):
        khat = (1.0 - self.K) if self.mode == 'outside' else (1.0 + self.K)
        khat = np.clip(khat, 0.05, None)
        return khat * r_p, phi_p / khat

    # -- Step 6: helicalize  (25)-(31) -------------------------------------
    #
    # Give the 2-D cross-sectional shape the pitch angle alpha and it becomes
    # a 3-D curve: straight segments stretch (25), arcs become helices whose
    # slope is b = r tan(alpha) (26). Then normalise each piece so the target
    # curve is exactly as long as the robot (27)-(30), and read off curvature
    # and torsion (31).
    def _helicalize(self, r_h, phi_h, l_p, line_first, end_lines, lead_sign):
        ca, ta = np.cos(self.alpha), np.tan(self.alpha)
        l_h = l_p / ca                                     # (25)
        b_h = r_h * ta                                     # (26)
        helix_len = phi_h * np.hypot(r_h, b_h)
        l_f = l_h + helix_len                              # (28)

        # (29), (30): the pair (S_hi, H_i) spans the robot from the midpoint
        # of link i-1 to the midpoint of link i, so its length must equal half
        # of each. With uniform links that is the paper's single l.
        want = 0.5 * (self.len[:-1] + self.len[1:])
        scale = want / np.maximum(l_f, 1e-12)
        r_b, b_b = scale * r_h, scale * b_h
        l_b = scale * l_h
        helix_b = scale * helix_len

        denom = np.maximum(r_b ** 2 + b_b ** 2, 1e-16)     # (31)
        kappa = r_b / denom
        tau = lead_sign * b_b / denom

        pieces = [(end_lines[0], 0.0, 0.0)]                # S_h1, straight
        for i in range(self.n_joint):
            line = (l_b[i], 0.0, 0.0)
            helix = (helix_b[i], kappa[i], tau[i])
            pieces += [line, helix] if line_first[i] else [helix, line]
        pieces.append((end_lines[1], 0.0, 0.0))            # S_h(nlink+1)

        geom = {'r': r_b, 'phi': phi_h, 'line': l_b,
                'line_first': line_first, 'ends': end_lines}
        return pieces, geom

    def _target_cross_section(self, geom, winding):
        """The commanded target form, projected back down onto its own plane.

        Undoes step 6 for reporting only: walk the normalised line and arc
        pieces in 2-D, with each 3-D length shortened by cos(alpha) to give
        its footprint in the cross-sectional plane. What comes out is the
        shape drawn along the top of Figs. 11 and 14-16 of the paper, one
        control tick ahead of the measured one.
        """
        ca = np.cos(self.alpha)
        pos = np.zeros(2)
        head = 0.0                 # heading in the plane, radians
        pts = [pos.copy()]

        def straight(length):
            nonlocal pos
            pos = pos + length * np.array([np.cos(head), np.sin(head)])
            pts.append(pos.copy())

        def arc(r, phi):
            nonlocal pos, head
            turn = winding * phi
            n = max(2, int(abs(turn) / 0.15) + 1)
            for _ in range(n):
                dh = turn / n
                # chord of a circular arc of radius r subtending dh
                c = 2.0 * r * np.sin(abs(dh) / 2.0)
                mid = head + dh / 2.0
                pos = pos + c * np.array([np.cos(mid), np.sin(mid)])
                head += dh
                pts.append(pos.copy())

        straight(geom['ends'][0] * ca)
        for i in range(self.n_joint):
            if geom['line_first'][i]:
                straight(geom['line'][i] * ca)
                arc(geom['r'][i], geom['phi'][i])
            else:
                arc(geom['r'][i], geom['phi'][i])
                straight(geom['line'][i] * ca)
        straight(geom['ends'][1] * ca)
        return np.array(pts)

    # -- Step 7: fit the discrete robot to the target curve  (32), (33) ----
    #
    #   theta_i = INT over [s_(i-1), s_(i+1)] of  -kappa(s) sin psi(s) ds  (i odd)
    #                                             +kappa(s) cos psi(s) ds  (i even)
    #   psi(s)  = INT over [0, s] of tau(s') ds' + psi_roll                 (33)
    #
    # Yamada's approximation: each joint bends in one fixed plane, so it takes
    # responsibility for the component of the target curve's bending that lies
    # in that plane over the two links meeting at it, and its neighbour --
    # whose axis is orthogonal -- takes the other component of the same span.
    # psi tracks how far the curve's normal has rotated about the tangent, and
    # psi_roll offsets it: advancing psi_roll rolls the whole body around an
    # unchanged target form. That is the rolling motion which climbs the pipe.
    def _approximate(self, pieces, psi_roll, orient):
        lens = np.array([p[0] for p in pieces])
        kap = np.array([p[1] for p in pieces])
        tor = np.array([p[2] for p in pieces])

        nsub = np.maximum(1, np.ceil(lens / self.ds).astype(int))
        ds = np.repeat(lens / nsub, nsub)
        k = np.repeat(kap, nsub)
        t = np.repeat(tor, nsub)

        s_end = np.cumsum(ds)
        s_start = s_end - ds
        # psi at the midpoint of each quadrature cell
        psi = orient.psi_sign * (np.cumsum(t * ds) - 0.5 * t * ds) \
            + psi_roll + orient.psi_offset

        f_sin = -k * np.sin(psi)
        f_cos = k * np.cos(psi)

        # The built curve is as long as the robot by construction; rescale
        # anyway so rounding cannot push the limits of (32) off the end.
        scale = self.total_length / s_end[-1]
        s_end, s_start = s_end * scale, s_start * scale

        theta = np.empty(self.n_joint)
        for j in range(self.n_joint):                      # paper index j+1
            lo, hi = self.s_node[j], self.s_node[j + 2]
            wgt = np.clip(np.minimum(s_end, hi) - np.maximum(s_start, lo), 0.0, None)
            odd = (((j + 1) % 2) == 1) != orient.parity_flip
            theta[j] = orient.sign * ((f_sin if odd else f_cos) @ wgt)
        return theta

    # -- The whole control tick --------------------------------------------
    def target(self, nodes, psi_roll, orient=None):
        """Target joint angles for the next step, in the robot's own order.

        `nodes` are the link-edge points, head -> tail, in world coordinates.
        Returns (joint_targets, info) where info carries the measured shape.
        """
        orient = orient or _ORIENT
        P = np.asarray(nodes, float)

        axis = estimate_axis(P, seed=self._axis)                                  # 1-2
        self._axis = axis
        v2d, h = self._cross_section(P, axis)                                     # 3
        lead = self._lead_sign(v2d, h)

        r_p, phi_p, l_p, line_first, end_lines, w = self._fit_cross_section(v2d)  # 4
        r_h, phi_h = self._comply(r_p, phi_p)                                     # 5
        pieces, geom = self._helicalize(r_h, phi_h, l_p, line_first,             # 6
                                        end_lines, lead)
        theta = self._approximate(pieces, psi_roll, orient)                       # 7
        self._pieces = pieces

        tgt2d = self._target_cross_section(geom, w)
        c = v2d - v2d.mean(axis=0)
        ct = tgt2d - tgt2d.mean(axis=0)
        info = {
            'axis': axis,
            'winding': w,
            'lead_sign': lead,
            'cross_section': v2d,
            'cross_section_target': tgt2d,
            # Mean distance from the cross-section to its own centre, now and
            # as just commanded. The gap between them is the squeeze from (23):
            # for mode='outside' the target should sit about K below the
            # measurement, and the pipe is what stops the robot getting there.
            'radius_measured': float(np.mean(np.linalg.norm(c, axis=1))),
            'radius_target': float(np.mean(np.linalg.norm(ct, axis=1))),
            'turns': float(abs(np.sum(phi_p)) / (2 * np.pi)),
            'pitch_angle': self.alpha,
        }
        return self.to_robot_order(theta), info

    def retarget(self, psi_roll, orient=None):
        """Step 7 alone: the last target form, re-evaluated at a new psi_roll.

        Estimating the form (steps 1-6) needs a fresh read of the joints and
        runs at the control rate; rolling the body round that form, eq. (33),
        needs neither and can be streamed to the servos between ticks.
        """
        return self.to_robot_order(self._approximate(self._pieces, psi_roll,
                                                     orient or _ORIENT))

    def normal_helix_target(self, radius, psi_roll, lead_sign=1.0, orient=None):
        """Joint angles for an ordinary helix of the given radius.

        The paper's experiments all start from "a normal helix with a radius
        of R and a pitch angle of alpha" before the adaptive loop takes over.
        Building that initial form with the same (32) machinery makes the
        hand-off into the adaptive loop seamless.
        """
        orient = orient or _ORIENT
        b = radius * np.tan(self.alpha)
        denom = radius ** 2 + b ** 2
        pieces = [(self.total_length, radius / denom, lead_sign * b / denom)]
        return self.to_robot_order(self._approximate(pieces, psi_roll, orient))

    def to_robot_order(self, theta_paper):
        """Paper order (joint 1 at the head) -> this robot's J1..J18 order.

        J1 is the joint nearest the TAIL, so the two orders are reversed.
        """
        return np.asarray(theta_paper)[::-1].copy()


# ---------------------------------------------------------------------------
# Fitting a helix to a measured backbone -- used for calibration and reporting
# ---------------------------------------------------------------------------
def fit_helix(nodes):
    """Best-fit helix through a backbone.

    Returns a dict with the axis (head-positive), radius, `lead` (rise along
    the axis per radian of turn -- its sign is the handedness), pitch angle,
    signed number of turns, and the rms residual of the fit.
    """
    P = np.asarray(nodes, float)
    P = P - P.mean(axis=0)

    def cost_for(a):
        u, v, a = _frame(a)
        x, y, h = P @ u, P @ v, P @ a
        A = np.column_stack([x, y, np.ones_like(x)])
        sol, *_ = np.linalg.lstsq(A, x ** 2 + y ** 2, rcond=None)
        cx, cy = sol[0] / 2, sol[1] / 2
        R = np.sqrt(max(sol[2] + cx ** 2 + cy ** 2, 1e-12))
        rr = np.hypot(x - cx, y - cy)
        th = np.unwrap(np.arctan2(y - cy, x - cx))
        s2, *_ = np.linalg.lstsq(np.column_stack([th, np.ones_like(th)]), h, rcond=None)
        res = float(np.sqrt(np.mean((rr - R) ** 2))
                    + np.sqrt(np.mean((h - (s2[0] * th + s2[1])) ** 2)))
        return res, float(R), float(s2[0]), float((th[-1] - th[0]) / (2 * np.pi)), a

    # Seed from the pitch-angle axis estimator, which is vectorised and is
    # exact for a uniform helix, then refine against the least-squares
    # residual. Sweeping all 500 seeds through cost_for() instead would be
    # two orders of magnitude slower for the same answer.
    best = cost_for(estimate_axis(nodes))
    step = 0.08
    for _ in range(6):
        u, v, a0 = _frame(best[4])
        for du in (-step, 0.0, step):
            for dv in (-step, 0.0, step):
                if du == 0.0 and dv == 0.0:
                    continue
                c = cost_for(_unit(a0 + du * u + dv * v))
                if c[0] < best[0]:
                    best = c
        step *= 0.4

    res, R, lead, turns, axis = best
    if axis @ (P[0] - P[-1]) < 0:            # head-positive, like the estimator
        axis, lead, turns = -axis, -lead, -turns
    return {
        'axis': axis,
        'radius': R,
        'lead': lead,
        'pitch_angle': float(np.arctan2(abs(lead), R)),
        'turns': turns,
        'residual': res,
    }
