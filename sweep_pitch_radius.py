"""
Pitch angle x pole radius x friction, placed on the pole.

The pitch angle alpha is the paper's one free design parameter for the
climbing helix (Section III-B6; its experiments use 0.20-0.30 rad, fixed).
A flatter helix winds more turns out of the same body, and one run at 7 cm
went from sliding off (alpha 0.25) to climbing 0.78 m in 60 s (alpha 0.15),
while alpha 0.15 is known to stall on a 4 cm pole. This sweep measures that
trade-off so the pitch can be chosen from the pole size (climb_rig.alpha_for).

    python sweep_pitch_radius.py
    python plot_pitch_sweep.py results/pitch_sweep_<stamp>
"""
import csv
import datetime as _dt
import os
from multiprocessing import Pool

from sweep_radius_friction import FIELDS, START_CLEARANCE, _watch_penetration

ALPHAS = [0.10, 0.15, 0.20, 0.25, 0.30]
RADII = [0.030, 0.040, 0.050, 0.055, 0.060, 0.065, 0.070, 0.080]
MUS = [0.3, 0.5, 0.8]
DURATION = 150.0


def one(job):
    alpha, r, mu, out = job
    from climb_rig import Config, run
    stats = dict(n=0, pen=0.0)
    _watch_penetration(stats)
    label = f"a{alpha:.2f}_r{r*1000:.0f}mm_mu{mu:.2f}".replace('.', 'p')
    cfg = Config(pole_radius=r, mu_robot_pole=mu, alpha=alpha, max_duration=DURATION,
                 log_hz=10.0, out_dir=out, label=label, start_wrapped=True,
                 start_radius=r + START_CLEARANCE)
    row = dict(alpha=alpha, pole_radius=r, mu=mu)
    try:
        s = run(cfg)
        row.update({k: s.get(k, '') for k in FIELDS if k in s})
        row['max_penetration_mm'] = round(stats['pen'] * 1000, 2)
    except Exception as e:
        row['error'] = repr(e)
    return row


def main():
    out = os.path.join('results', 'pitch_sweep_' + _dt.datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(out, exist_ok=True)
    jobs = [(a, r, mu, out) for a in ALPHAS for r in RADII for mu in MUS]
    print(f"{len(jobs)} runs, writing to {out}", flush=True)
    fields = ['alpha', 'pole_radius', 'mu'] + [f for f in FIELDS if f not in ('mode', 'pole_radius', 'mu')]
    with open(os.path.join(out, 'index.csv'), 'w', newline='') as f, \
            Pool(max(1, os.cpu_count() - 1), maxtasksperchild=1) as pool:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for i, row in enumerate(pool.imap_unordered(one, jobs), 1):
            w.writerow(row)
            f.flush()
            print(f"[{i}/{len(jobs)}] alpha={row['alpha']:.2f} r={row['pole_radius']:.3f} "
                  f"mu={row['mu']:.2f} top={row.get('reached_top', '')} "
                  f"rate={row.get('rolling_rate_cm_s', '')} pen={row.get('max_penetration_mm', '')} "
                  f"{row.get('error', '')}", flush=True)
    print('done:', out)


if __name__ == '__main__':
    main()
