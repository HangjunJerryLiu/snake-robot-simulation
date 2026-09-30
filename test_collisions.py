"""
Collision testbench: nothing in the simulation may pass through anything else.

    python test_collisions.py            full run (~5 min)
    python test_collisions.py --quick    fewer radii and joints (~1.5 min)

Exit code 0 = every check passed. Also collectable by pytest if pytest is
installed in the venv (each check is a test_* function).

WHAT IS CHECKED, AND WHY EACH ONE CAN BE TRUSTED
------------------------------------------------
Every check is run twice: on the real model, where it must PASS, and on a
deliberately broken case (a "planted fault"), where it must FAIL. A check
that cannot fail proves nothing, so a planted fault that is not caught is a
testbench failure in its own right.

1. what_you_see_collides
   Every geom that is drawn also collides, with the same shape. Otherwise
   the viewer could show a part blending into something that, physically,
   is not there -- or hide one that is.
   Planted fault: a drawn geom with collisions switched off.

2. pole_geometry
   For every pole radius, the compiled pole's bounding box and bounding
   sphere match its size. This is exactly the pole-radius bug of 2026-09-28:
   resizing the pole after compiling left its bounding box at 0.04 m and
   collisions with larger poles were missed.
   Planted fault: the old build (resize after compile).

3. no_penetration_in_motion  (pole, floor, link-link)
   Short runs at several pole radii, placed on the pole and from the floor.
   Every 10 steps, MuJoCo's exact signed distance (mj_geomDistance) is taken
   between each link and the pole, the floor and every non-adjacent link.
   Link collision shapes are convex hulls, and every real part lies inside
   its own hull, so a hull that is not inside something guarantees the real
   part is not either. Two limits, because MuJoCo's contact is soft:
     HELD   penetration sustained for HOLD_S may not exceed PEN_LIMIT. This
            is "blending": the pole bug held links 38 mm inside the pole.
     SPIKE  a momentary impact may not exceed SPIKE_LIMIT. Measured: the
            robot dropping 0.3 m onto the floor at 2.3 m/s dips 3.2 mm for
            ~10 ms and then rests 1 mm clear; that is the contact absorbing
            an impact, not blending, but anything near SPIKE_LIMIT would be
            a part tunnelling through.
   Planted fault: the old build on a 0.08 m pole.

4. joints_do_not_blend
   Neighbouring links are never collided by MuJoCo (parent/child pairs are
   filtered), so this is checked on the REAL meshes (collision_check.py,
   0.3 mm voxels): each joint is swept through its whole range, and the
   overlap with its neighbour must never exceed what the CAD assembly has
   in the straight pose (the servo horn is modelled pressed ~0.3 mm into
   its bracket) by more than BLEND_VOL or BLEND_REACH. The voxel grid
   aliases as a part rotates: measured on J18, extra reach scatters between
   -0.6 and +0.6 mm at 0.3 mm voxels and shrinks to +0.15 mm at 0.15 mm
   voxels, with extra volume under 5 mm^3 -- so BLEND_REACH sits just above
   that noise. Planted fault: a bracket shifted 5 mm into its motor, which
   adds over 1000 mm^3.
"""
import contextlib
import os
import shutil
import sys
import tempfile

import mujoco
import numpy as np

import climb_rig
from climb_rig import Config, build_model
from collision_check import Checker

PEN_LIMIT = 3.0e-3        # m, deepest penetration HELD for HOLD_S
HOLD_S = 0.05             # s
SPIKE_LIMIT = 10.0e-3     # m, deepest momentary penetration
SAMPLE_EVERY = 10         # physics steps between distance samples (10 ms)
BLEND_VOL = 30.0          # mm^3 above the as-built joint overlap
BLEND_REACH = 1.0e-3      # m  above the as-built joint overlap (see docstring)
QUICK = '--quick' in sys.argv


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def old_buggy_build():
    """The pre-fix build_model: pole resized AFTER compiling."""
    real = climb_rig.build_model

    def buggy(cfg):
        model = real(Config(**{**cfg.__dict__, 'pole_radius': 0.04}))
        model.geom_size[model.geom('pipe').id, 0] = cfg.pole_radius
        return model
    climb_rig.build_model = buggy
    try:
        yield buggy
    finally:
        climb_rig.build_model = real


CLASSES = ('pole', 'floor', 'link-link')


def _penetrations(m, d):
    """Deepest hull penetration (m, positive = inside) per class, right now,
    plus which bodies it is between."""
    pipe, floor = m.geom('pipe').id, m.geom('floor').id
    robot = [g for g in range(m.ngeom) if m.geom_bodyid[g] != 0]
    out = {c: (0.0, None) for c in CLASSES}

    def note(c, dist, a, b):
        if -dist > out[c][0]:
            out[c] = (-dist, (m.body(m.geom_bodyid[a]).name, m.body(m.geom_bodyid[b]).name))
    for g in robot:
        note('pole', mujoco.mj_geomDistance(m, d, pipe, g, 0.05, None), pipe, g)
        note('floor', mujoco.mj_geomDistance(m, d, floor, g, 0.05, None), floor, g)
    for i, a in enumerate(robot):
        for b in robot[i + 1:]:
            ba, bb = m.geom_bodyid[a], m.geom_bodyid[b]
            if m.body_parentid[ba] == bb or m.body_parentid[bb] == ba:
                continue                      # neighbours: see joints_do_not_blend
            note('link-link', mujoco.mj_geomDistance(m, d, a, b, 0.05, None), a, b)
    return out


def run_and_watch(cfg):
    """Run climb_rig.run(cfg); return per class the peak penetration and the
    deepest penetration held for HOLD_S, each with when and between what."""
    series = {c: [] for c in CLASSES}
    n = [0]
    step = mujoco.mj_step

    def watched(m, d):
        step(m, d)
        n[0] += 1
        if n[0] % SAMPLE_EVERY == 0:
            for c, (pen, who) in _penetrations(m, d).items():
                series[c].append((d.time, pen, who))
    out = tempfile.mkdtemp(prefix='collision_tb_')
    mujoco.mj_step = watched
    try:
        cfg.out_dir, cfg.log_hz = out, 5.0
        climb_rig.run(cfg)
    finally:
        mujoco.mj_step = step
        shutil.rmtree(out, ignore_errors=True)

    win = max(1, int(round(HOLD_S / (SAMPLE_EVERY * cfg.timestep))))
    res = {}
    for c, s in series.items():
        pen = np.array([p for _, p, _ in s])
        k = int(pen.argmax())
        held = np.array([pen[i:i + win].min() for i in range(max(1, len(pen) - win + 1))])
        h = int(held.argmax())
        res[c] = dict(peak=float(pen[k]), peak_t=s[k][0], peak_who=s[k][2],
                      held=float(held[h]), held_t=s[h][0], held_who=s[h][2])
    return res


# ---------------------------------------------------------------------------
# 1. what you see is what collides
# ---------------------------------------------------------------------------
def check_what_you_see_collides(model):
    bad = []
    for g in range(model.ngeom):
        drawn = model.geom_rgba[g, 3] > 0 and model.geom_group[g] < 3
        collides = model.geom_contype[g] or model.geom_conaffinity[g]
        if drawn and not collides:
            bad.append(model.geom(g).name or f'geom {g}')
    return (not bad), (f"drawn but not colliding: {bad}" if bad else
                       f"all {model.ngeom} drawn geoms collide")


def test_what_you_see_collides():
    ok, msg = check_what_you_see_collides(build_model(Config()))
    assert ok, msg
    planted = build_model(Config())
    g = planted.geom('pipe').id
    planted.geom_contype[g] = planted.geom_conaffinity[g] = 0
    ok, _ = check_what_you_see_collides(planted)
    assert not ok, "planted fault not caught: a non-colliding drawn pole passed"
    return msg


# ---------------------------------------------------------------------------
# 2. pole geometry is consistent with its size
# ---------------------------------------------------------------------------
RADII = [0.02, 0.04, 0.06, 0.08, 0.10] if not QUICK else [0.02, 0.06, 0.10]


def check_pole_geometry(builder):
    bad = []
    for r in RADII:
        m = builder(Config(pole_radius=r))
        g = m.geom('pipe').id
        size = m.geom_size[g]
        half = m.geom_aabb[g, 3:]
        want_half = np.array([size[0], size[0], size[1]])
        want_rb = np.hypot(size[0], size[1])
        if not (np.allclose(half, want_half, atol=1e-9) and abs(m.geom_rbound[g] - want_rb) < 1e-9):
            bad.append(f"r={r}: aabb half {np.round(half, 4)} vs {np.round(want_half, 4)}, "
                       f"rbound {m.geom_rbound[g]:.4f} vs {want_rb:.4f}")
    return (not bad), ("; ".join(bad) if bad else f"bounding volumes match at r = {RADII}")


def test_pole_geometry():
    ok, msg = check_pole_geometry(build_model)
    assert ok, msg
    with old_buggy_build() as buggy:
        ok, _ = check_pole_geometry(buggy)
    assert not ok, "planted fault not caught: the old resize-after-compile build passed"
    return msg


# ---------------------------------------------------------------------------
# 3. no penetration in motion: pole, floor, non-adjacent links
# ---------------------------------------------------------------------------
def _motion_cases():
    cases = [Config(pole_radius=r, start_wrapped=True, start_radius=r + 0.045,
                    max_duration=6.0) for r in RADII]
    cases.append(Config(pole_radius=0.04, max_duration=25.0 if not QUICK else 20.0))   # floor start, catches
    cases.append(Config(pole_radius=0.10, max_duration=25.0 if not QUICK else 20.0))   # floor start, the 10 cm case
    return cases


def check_motion(cfg):
    res = run_and_watch(cfg)
    start = 'placed' if cfg.start_wrapped else 'floor'
    parts, bad = [], []
    for c in CLASSES:
        r = res[c]
        parts.append(f"{c} held {r['held']*1000:.1f} / peak {r['peak']*1000:.1f} mm")
        if r['held'] > PEN_LIMIT:
            bad.append(f"{c}: {r['held']*1000:.1f} mm held for {HOLD_S*1000:.0f} ms from "
                       f"t={r['held_t']:.2f} s, {r['held_who']}")
        if r['peak'] > SPIKE_LIMIT:
            bad.append(f"{c}: {r['peak']*1000:.1f} mm spike at t={r['peak_t']:.2f} s, {r['peak_who']}")
    msg = f"r={cfg.pole_radius:.2f} {start}: " + ", ".join(parts)
    if bad:
        msg += "  <-- " + "; ".join(bad)
    return (not bad), msg


def test_no_penetration_in_motion():
    msgs = []
    for cfg in _motion_cases():
        ok, msg = check_motion(cfg)
        msgs.append(msg)
        assert ok, msg
    with old_buggy_build():
        ok, msg = check_motion(Config(pole_radius=0.08, start_wrapped=True,
                                      start_radius=0.125, max_duration=6.0))
    assert not ok, f"planted fault not caught: old build on a 0.08 m pole passed ({msg})"
    msgs.append(f"planted fault caught -> {msg}")
    return "\n      ".join(msgs)


# ---------------------------------------------------------------------------
# 4. joints do not let neighbouring links blend
# ---------------------------------------------------------------------------
def _joint_overlap(C, d, gp, gc):
    A, B = C.volume(gc), C.volume(gp)
    Ra, Rb = d.geom_xmat[gc].reshape(3, 3), d.geom_xmat[gp].reshape(3, 3)
    in_b = (A.solid_pts @ Ra.T + d.geom_xpos[gc] - d.geom_xpos[gp]) @ Rb
    vol = float((B.depth_at(in_b) > 0).sum() * A.h ** 3 * 1e9)       # mm^3
    return vol, C.pair_penetration(d, gp, gc)


def _pose(m, d, j=None, angle=0.0):
    d.qpos[:] = 0
    d.qpos[2], d.qpos[3] = 0.3, 1.0
    if j is not None:
        d.qpos[m.jnt_qposadr[j]] = angle
    mujoco.mj_kinematics(m, d)


def check_joints(m, pairs, planted_shift=0.0, C=None):
    """Sweep each joint over its range; flag overlap beyond the as-built pose.
    planted_shift (m) slides the child link along its own joint axis -- the
    direction that drives a bracket's side plate into the motor it holds."""
    C = C or Checker(m)
    d = mujoco.MjData(m)
    bad, worst = [], (0.0, 0.0, None)
    for j, gp, gc in pairs:
        _pose(m, d)
        vol0, reach0 = _joint_overlap(C, d, gp, gc)
        lo, hi = m.jnt_range[j]
        for a in np.linspace(lo, hi, 15 if not QUICK else 7):
            _pose(m, d, j, a)
            if planted_shift:
                d.geom_xpos[gc] = d.geom_xpos[gc] + planted_shift * d.xaxis[j]
            vol, reach = _joint_overlap(C, d, gp, gc)
            ev, er = vol - vol0, reach - reach0
            if ev > worst[0]:
                worst = (ev, er, (m.joint(j).name, round(float(a), 2)))
            if ev > BLEND_VOL or er > BLEND_REACH:
                bad.append(f"{m.joint(j).name} at {a:+.2f} rad: +{ev:.0f} mm^3, "
                           f"+{er*1000:.1f} mm beyond as-built")
                break
    name = worst[2]
    return (not bad), ("; ".join(bad[:4]) if bad else
                       f"{len(pairs)} joints swept over their full range; worst extra overlap "
                       f"{worst[0]:.0f} mm^3 / {worst[1]*1000:.1f} mm"
                       + (f" ({name[0]} at {name[1]:+.2f} rad)" if name else ''))


def test_joints_do_not_blend():
    m = build_model(Config())
    pairs = Checker(m).adjacent_pairs()
    if QUICK:
        pairs = pairs[:2] + pairs[-2:]
    C = Checker(m)
    ok, msg = check_joints(m, pairs, C=C)
    assert ok, msg
    missed = []
    for p in pairs:                          # every joint checked must catch it
        for sign in (+1, -1):
            caught, _ = check_joints(m, [p], planted_shift=sign * 0.005, C=C)
            if caught:                       # check_joints returns ok=True when clean
                missed.append(f"{m.joint(p[0]).name} {'+' if sign > 0 else '-'}5 mm")
    assert not missed, f"planted fault not caught: link shifted along its joint axis passed at {missed}"
    return msg + f"; planted 5 mm shift caught on all {len(pairs)} joints, both directions"


# ---------------------------------------------------------------------------
TESTS = [test_what_you_see_collides, test_pole_geometry,
         test_joints_do_not_blend, test_no_penetration_in_motion]


def main():
    failed = 0
    for t in TESTS:
        name = t.__name__[5:]
        try:
            msg = t()
            print(f"PASS  {name}\n      {msg}", flush=True)
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {name}\n      {e}", flush=True)
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} checks passed")
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
