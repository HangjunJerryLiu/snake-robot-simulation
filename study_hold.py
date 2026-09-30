"""
Holding-on study: does the compliance loop itself use up friction?

At friction 0.20 on a 4 cm pole the robot slides down even with rolling
switched off, although its ~78 N of grip on a 7.2 N robot needs only ~0.09
to carry the weight -- and the contacts report demanding all 0.20 that is
available. Hypothesis: re-squeezing every control tick (eq. 23 recomputed
from the measured form) drags the links sideways along the pole, and that
sliding spends the friction the weight needed.

Test: same pre-wrapped start, rolling off (psi_dot = 0) in both arms.
    adaptive   the controller re-estimates and re-squeezes every 0.1 s
    frozen     the same, until FREEZE_AT; then the joint targets are held
If frozen holds where adaptive slides, the hypothesis stands.

RESULT (2026-09-30, results/study_hold_*): REFUTED. At mu 0.20, the one
friction where the robot slides without reaching the floor, frozen slides
69 mm in 28 s against 76 mm adaptive -- about 10% less, not a hold.
Re-squeezing does raise the friction the contacts demand (at mu 0.5:
0.36 adaptive vs 0.18 frozen), but that is not what lets the robot slide.
Even frozen, the contacts demand ~0.18, about twice the ~0.09 the weight
alone needs, so roughly half the tangential load is internal to the grip.
At mu 0.10-0.15 the robot is on the floor before or during the window.

    python study_hold.py
"""
import csv
import datetime as _dt
import os
from multiprocessing import Pool

import numpy as np

MUS = [0.10, 0.15, 0.20, 0.25, 0.50]
FREEZE_AT = 2.0      # s, after the placed helix has settled onto the pole
DURATION = 30.0      # s


def one(job):
    i, arm, mu, out = job
    import sweep_radius_friction as sw
    from climb_rig import Config, run
    stats = dict(n=0, pen=0.0)
    sw._watch_penetration(stats)
    cfg = Config(pole_radius=0.04, start_radius=0.085, start_wrapped=True, mu_robot_pole=mu,
                 psi_dot=0.0, max_duration=DURATION, log_hz=20.0, out_dir=out,
                 label=f"{i:02d}_{arm}_mu{mu:.2f}".replace('.', 'p'),
                 freeze_form_at=FREEZE_AT if arm == 'frozen' else -1.0)
    s = run(cfg)
    rows = list(csv.DictReader(open(os.path.join(s['run_dir'], 'run.csv'))))
    t = np.array([float(x['t_s']) for x in rows])
    z = np.array([float(x['com_z_m']) for x in rows])
    after = [x for x in rows if float(x['t_s']) >= FREEZE_AT + 1.0]
    mean = lambda k: float(np.mean([float(x[k]) for x in after if x[k] != '']))
    z0 = float(np.interp(FREEZE_AT, t, z))
    # A robot already resting on the floor has nothing left to slide: its
    # row says nothing about holding on, so flag it rather than report ~0.
    floor = max(int(x['n_contact_floor']) for x in rows if float(x['t_s']) >= FREEZE_AT)
    return dict(arm=arm, mu=mu, touched_floor=floor > 0,
                slide_m=round(float(z[-1] - z0), 4),
                slide_rate_mm_s=round(float(z[-1] - z0) / (t[-1] - FREEZE_AT) * 1000, 3),
                normal_N=round(mean('pole_normal_force_N'), 1),
                contacts=round(mean('n_contact_pole'), 1),
                mu_demanded_p95=round(mean('mu_required_p95'), 3),
                weight_share_mu=round(0.7378 * 9.81 / max(mean('pole_normal_force_N'), 1e-9), 3),
                motors_saturated=round(mean('joints_saturated'), 1),
                max_penetration_mm=round(stats['pen'] * 1000, 2),
                run_dir=s['run_dir'])


if __name__ == '__main__':
    out = os.path.join('results', 'study_hold_' + _dt.datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(out, exist_ok=True)
    jobs = [(i, arm, mu, out) for i, (arm, mu) in
            enumerate((a, m) for m in MUS for a in ('adaptive', 'frozen'))]
    with Pool(min(7, len(jobs)), maxtasksperchild=1) as p:
        res = p.map(one, jobs)
    with open(os.path.join(out, 'study_hold.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(res[0]))
        w.writeheader()
        w.writerows(res)
    print(f"{'arm':9s} {'mu':>5s} {'slide (m)':>10s} {'mm/s':>7s} {'normal N':>9s} {'contacts':>9s} "
          f"{'mu demanded':>12s} {'weight/N':>9s} {'sat':>5s} {'pen mm':>7s}")
    for r in res:
        print(f"{r['arm']:9s} {r['mu']:5.2f} {r['slide_m']:+10.3f} {r['slide_rate_mm_s']:+7.2f} "
              f"{r['normal_N']:9.1f} {r['contacts']:9.1f} {r['mu_demanded_p95']:12.3f} "
              f"{r['weight_share_mu']:9.3f} {r['motors_saturated']:5.1f} {r['max_penetration_mm']:7.2f}"
              + ("   reached the floor: slide not measurable" if r['touched_floor'] else ""))
    print('wrote', out)
