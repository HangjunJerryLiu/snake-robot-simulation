"""
Does choosing the pitch from the pole radius (climb_rig.alpha_for_radius)
beat the old fixed 0.25?

Runs the rule (alpha = -1) at every radius x friction below, placed on the
pole and from the floor, and compares with the fixed-0.25 runs of the
radius x friction sweep (results/sweep_20260929_002829), which used the
same settings and are deterministic, so they are not re-run. 6.5 cm was
not in that sweep, so its fixed-0.25 runs are made here.

3.5, 4.2 and 4.5 cm were NOT in the pitch sweep the rule was fitted to:
those rows are the out-of-sample test.

    python validate_pitch_rule.py
"""
import csv
import datetime as _dt
import os
from multiprocessing import Pool

from sweep_radius_friction import FIELDS, START_CLEARANCE, _watch_penetration

RADII = [0.035, 0.042, 0.045, 0.050, 0.060, 0.065, 0.070]
HELD_OUT = {0.035, 0.042, 0.045}
MUS = [0.3, 0.5, 0.8]
BASELINE = os.path.join('results', 'sweep_20260929_002829', 'index.csv')


def one(job):
    mode, r, mu, alpha, out = job
    from climb_rig import Config, run, alpha_for_radius
    stats = dict(n=0, pen=0.0)
    _watch_penetration(stats)
    tag = 'auto' if alpha < 0 else f"a{alpha:.2f}"
    cfg = Config(pole_radius=r, mu_robot_pole=mu, alpha=alpha, max_duration=150.0, log_hz=10.0,
                 out_dir=out, label=f"{mode}_{tag}_r{r*1000:.0f}mm_mu{mu:.2f}".replace('.', 'p'),
                 start_wrapped=(mode == 'wrapped'), start_radius=r + START_CLEARANCE)
    s = run(cfg)
    return dict(mode=mode, pole_radius=r, mu=mu, rule='auto' if alpha < 0 else 'fixed',
                alpha=round(alpha_for_radius(r) if alpha < 0 else alpha, 3),
                reached_top=s['reached_top'], rate=s['rolling_rate_cm_s'],
                peak=s['peak_com_z_m'], caught=s['caught_pole'],
                pen_mm=round(stats['pen'] * 1000, 2), diverged=s['diverged'], run_dir=s['run_dir'])


def main():
    out = os.path.join('results', 'pitch_rule_check_' + _dt.datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(out, exist_ok=True)
    jobs = [(m, r, mu, -1.0, out) for m in ('wrapped', 'floor') for r in RADII for mu in MUS]
    jobs += [('floor', 0.065, mu, 0.25, out) for mu in MUS]          # not in the baseline sweep
    jobs += [('wrapped', 0.065, mu, 0.25, out) for mu in MUS]
    with Pool(max(1, os.cpu_count() - 1), maxtasksperchild=1) as p:
        res = p.map(one, jobs)
    for x in csv.DictReader(open(BASELINE)):                       # fixed 0.25, already run
        r, mu = float(x['pole_radius']), float(x['mu'])
        if r in RADII and mu in MUS and r != 0.065:
            res.append(dict(mode=x['mode'], pole_radius=r, mu=mu, rule='fixed', alpha=0.25,
                            reached_top=x['reached_top'] == 'True', rate=x['rolling_rate_cm_s'],
                            peak=float(x['peak_com_z_m']), caught=x['caught_pole'] == 'True',
                            pen_mm=float(x['max_penetration_mm']), diverged=x['diverged'] == 'True',
                            run_dir=x['run_dir']))
    with open(os.path.join(out, 'pitch_rule_check.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(res[0]))
        w.writeheader()
        w.writerows(res)

    cell = {(x['mode'], x['pole_radius'], x['mu'], x['rule']): x for x in res}
    bad = [x for x in res if x['pen_mm'] > 5.0 or x['diverged']]
    for x in bad:
        print('INVALID', x)
    rate = lambda x: float(x['rate']) if x['rate'] not in ('', None) else float('nan')
    for mode in ('wrapped', 'floor'):
        print(f"\n{'placed on the pole' if mode == 'wrapped' else 'from the floor'}"
              f"   (rate cm/s, * = reached the top; held-out radii marked)")
        print(f"{'r (cm)':>7s} {'mu':>4s}  {'fixed 0.25':>11s}  {'rule':>16s}")
        tally = {'better': 0, 'same': 0, 'worse': 0}
        for r in RADII:
            for mu in MUS:
                a, b = cell[(mode, r, mu, 'fixed')], cell[(mode, r, mu, 'auto')]
                fa = f"{rate(a):6.2f}{'*' if a['reached_top'] else ' '}"
                fb = f"{rate(b):6.2f}{'*' if b['reached_top'] else ' '} (a={b['alpha']:.3f})"
                if b['reached_top'] != a['reached_top']:
                    verdict = 'better' if b['reached_top'] else 'worse'
                elif abs(rate(b) - rate(a)) < 0.15 or (rate(a) != rate(a) and rate(b) != rate(b)):
                    verdict = 'same'
                else:
                    verdict = 'better' if rate(b) > rate(a) else 'worse'
                tally[verdict] += 1
                print(f"{r*100:7.1f} {mu:4.1f}  {fa:>11s}  {fb:>16s}  {verdict}"
                      + ("   <- held out" if r in HELD_OUT else ""))
        print('  ', tally)
    print('\nwrote', out)


if __name__ == '__main__':
    main()
