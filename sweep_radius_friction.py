"""
Pole radius x robot/pole friction grid, run in parallel.

Each (radius, mu) cell is run twice:

    wrapped   placed on the pole as a normal helix (start_wrapped=True), so
              only the CONTROLLER is being tested
    floor     the full run -- approach, catch, climb -- as the GUI does it

Running both separates "the catch failed" from "the climb failed", which a
floor start alone cannot. Every run leaves its usual folder under
results/sweep_<stamp>/, and index.csv there has one row per run.
plot_sweep.py turns that into the figures.

The pitch angle is left at the Config default, which since 2026-09-30 is
chosen from the pole radius (climb_rig.alpha_for_radius). The sweep of
2026-09-29 (results/sweep_20260929_002829) predates that and used a fixed
0.25; pass alpha=0.25 in one() to reproduce it.

    python sweep_radius_friction.py                  full grid
    python sweep_radius_friction.py --quick          a few cells, to check it works
"""
import argparse
import csv
import datetime as _dt
import os
from multiprocessing import Pool

RADII = [0.020, 0.025, 0.030, 0.035, 0.038, 0.040, 0.042, 0.045, 0.050, 0.055, 0.060,
         0.070, 0.080, 0.090, 0.100]
MUS = [0.10, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.80, 1.00, 1.20, 1.50]
MODES = ('wrapped', 'floor')

# The placed helix sits this far outside the pole surface. 0.045 is the
# default (start_radius 0.085 on a 0.04 pole), kept constant so every radius
# starts with the same clearance.
START_CLEARANCE = 0.045

FIELDS = ['mode', 'pole_radius', 'mu', 'run_dir', 'caught_pole', 'catch_misses',
          'reached_top', 'time_to_top_s', 'peak_com_z_m', 'final_com_z_m',
          'rolling_rate_cm_s', 'rolling_climb_m', 'diverged', 'stopped_early',
          'sim_time_s', 'wall_time_s', 'max_penetration_mm', 'error']


def _watch_penetration(stats, every=20):
    """Wrap mj_step so every `every` steps it records how far the deepest
    robot link sits inside the pole, from MuJoCo's exact signed distance.
    Soft contact allows a few mm; much more means the pole is not colliding
    properly and the run is not physics. This is the check that caught the
    pole-radius bug, kept in so it cannot come back unnoticed."""
    import mujoco
    step = mujoco.mj_step

    def watched(m, d):
        step(m, d)
        stats['n'] += 1
        if stats['n'] % every:
            return
        pipe = m.geom('pipe').id
        for g in range(m.ngeom):
            if m.geom_bodyid[g] != 0:
                dist = mujoco.mj_geomDistance(m, d, pipe, g, 0.05, None)
                stats['pen'] = max(stats['pen'], -dist)
    mujoco.mj_step = watched


def one(job):
    mode, r, mu, out_dir, duration = job
    from climb_rig import Config, run
    stats = dict(n=0, pen=0.0)
    _watch_penetration(stats)
    label = f"{mode}_r{r*1000:.0f}mm_mu{mu:.2f}".replace('.', 'p')
    cfg = Config(pole_radius=r, mu_robot_pole=mu, max_duration=duration,
                 log_hz=10.0, out_dir=out_dir, label=label,
                 start_wrapped=(mode == 'wrapped'),
                 start_radius=r + START_CLEARANCE)
    row = dict(mode=mode, pole_radius=r, mu=mu)
    try:
        s = run(cfg)
        row.update({k: s.get(k, '') for k in FIELDS if k in s})
        row['max_penetration_mm'] = round(stats['pen'] * 1000, 2)
    except Exception as e:                      # keep the sweep going
        row['error'] = repr(e)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--duration', type=float, default=150.0)
    ap.add_argument('--workers', type=int, default=max(1, os.cpu_count() - 1))
    ap.add_argument('--modes', default=','.join(MODES))
    a = ap.parse_args()

    radii, mus = (RADII, MUS) if not a.quick else ([0.03, 0.04], [0.3, 0.5])
    out_dir = os.path.join('results', 'sweep_' + _dt.datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(out_dir, exist_ok=True)
    jobs = [(m, r, mu, out_dir, a.duration)
            for m in a.modes.split(',') for r in radii for mu in mus]
    print(f"{len(jobs)} runs, {a.workers} workers, writing to {out_dir}", flush=True)

    index = os.path.join(out_dir, 'index.csv')
    # A fresh process per run: _watch_penetration patches mj_step, and a
    # reused worker would stack one patch per run.
    with open(index, 'w', newline='') as f, Pool(a.workers, maxtasksperchild=1) as pool:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for i, row in enumerate(pool.imap_unordered(one, jobs), 1):
            w.writerow(row)
            f.flush()
            print(f"[{i}/{len(jobs)}] {row['mode']:7s} r={row['pole_radius']:.3f} "
                  f"mu={row['mu']:.2f} peak={row.get('peak_com_z_m', '')} "
                  f"rate={row.get('rolling_rate_cm_s', '')} {row.get('error', '')}",
                  flush=True)
    print("done:", index)


if __name__ == '__main__':
    main()
