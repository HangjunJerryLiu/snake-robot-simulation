"""
Figure and table for a sweep_pitch_radius.py run.

    python plot_pitch_sweep.py results/pitch_sweep_<stamp>

Writes <sweep>/figures/pitch_vs_radius.png/.pdf and <sweep>/best_pitch.csv.
Refuses to plot if any run failed the penetration check (as plot_sweep.py).

Cells: climb rate over the rolling stage (cm/s), from placed-on-the-pole
runs. A run whose solver blew up the moment it was placed is NOT a climbing
failure -- the placed helix was self-intersecting -- and is drawn hatched.
The dashed line is a PREDICTION THAT FAILED, kept on the figure so the
failure stays visible: where the helix's rise per turn equals the body's
thickness, i.e. where successive turns were expected to sit on each other.
Pitch 0.10 climbs on 5-6.5 cm poles, well below that line; the only run
that blew up on placement was 0.10 on a 3 cm pole. So it is not a limit.
"""
import csv
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

BODY_THICKNESS = 0.058     # m, link cross-section (geom half-sizes ~0.029 m)
BACKBONE_OFFSET = 0.034    # m, as climb_rig.BACKBONE_OFFSET
MAX_PEN_MM = 5.0

plt.rcParams.update({'font.size': 10, 'axes.titleweight': 'bold', 'axes.titlesize': 11,
                     'figure.facecolor': 'white', 'savefig.bbox': 'tight', 'pdf.fonttype': 42})


def load(d):
    rows = list(csv.DictReader(open(os.path.join(d, 'index.csv'))))
    for r in rows:
        for k in ('alpha', 'pole_radius', 'mu'):
            r[k] = float(r[k])
        r['placement_blowup'] = r['diverged'] == 'True' and float(r['sim_time_s'] or 0) < 1.0
        r['rate'] = float(r['rolling_rate_cm_s']) if r['rolling_rate_cm_s'] else np.nan
        r['top'] = r['reached_top'] == 'True'
    bad = [r for r in rows if r.get('error') or
           (not r['placement_blowup'] and float(r['max_penetration_mm'] or 0) > MAX_PEN_MM)]
    if bad:
        for r in bad:
            print('INVALID', r['alpha'], r['pole_radius'], r['mu'], r['max_penetration_mm'], r.get('error'))
        sys.exit(f"{len(bad)} runs failed the validity check; not plotting")
    late = [r for r in rows if r['diverged'] == 'True' and not r['placement_blowup']]
    if late:
        sys.exit(f"{len(late)} runs diverged after placement -- investigate before plotting")
    return rows


def clearance_alpha(r):
    """Flattest pitch at which successive turns clear the body thickness."""
    return np.arctan(BODY_THICKNESS / (2 * np.pi * (r + BACKBONE_OFFSET)))


def main():
    d = sys.argv[1]
    rows = load(d)
    A = sorted({r['alpha'] for r in rows})
    R = sorted({r['pole_radius'] for r in rows})
    M = sorted({r['mu'] for r in rows})
    cell = {(r['alpha'], r['pole_radius'], r['mu']): r for r in rows}

    norm = colors.Normalize(0, 5)
    cmap = matplotlib.colormaps['viridis']
    fig, axes = plt.subplots(1, len(M), figsize=(5.2 * len(M), 4.6), sharey=True,
                             constrained_layout=True)
    for ax, mu in zip(axes, M):
        for i, a in enumerate(A):
            for j, r in enumerate(R):
                c = cell[(a, r, mu)]
                if c['placement_blowup']:
                    ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fc='#eeeeee', ec='#bbbbbb',
                                               hatch='///', lw=0.5))
                    continue
                v = c['rate'] if np.isfinite(c['rate']) else 0.0
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fc=cmap(norm(max(v, 0))),
                                           ec='white', lw=1))
                ax.text(j, i, f"{v:.1f}" + ("↑" if c['top'] else ""), ha='center',
                        va='center', fontsize=8, color='white' if v < 3 else 'black')
        # turn-clearance line, in cell coordinates
        rr = np.linspace(R[0], R[-1], 100)
        ax.plot(np.interp(rr, R, range(len(R))), np.interp(clearance_alpha(rr), A, range(len(A))),
                ls='--', color='#d1495b', lw=1.6)
        ax.set_xlim(-.5, len(R) - .5)
        ax.set_ylim(-.5, len(A) - .5)
        ax.set_xticks(range(len(R)), [f"{r*100:.1f}" for r in R])
        ax.set_yticks(range(len(A)), [f"{a:.2f}" for a in A])
        ax.set_xlabel('Pole radius (cm)')
        ax.set_title(f"friction μ = {mu}", loc='left')
    axes[0].set_ylabel('Pitch angle α (rad)')
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm, cmap), ax=axes, shrink=0.85)
    cb.set_label('Rolling-stage climb rate (cm/s)')
    fig.suptitle('Pitch angle vs pole radius (placed on the pole).  ↑ = reached the top in '
                 '150 s;  hatched = placed helix self-intersecting (solver blew up at t = 0);  '
                 'red dashes = predicted turn-overlap limit (disproved: runs below it climb)', x=0.01, ha='left',
                 fontsize=9.5)
    out = os.path.join(d, 'figures')
    os.makedirs(out, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(out, f'pitch_vs_radius.{ext}'), dpi=200)

    # best pitch per (radius, friction): reached the top, then fastest
    with open(os.path.join(d, 'best_pitch.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['pole_radius', 'mu', 'best_alpha', 'rate_cm_s', 'reached_top',
                    'alpha_that_reach_top', 'rate_at_0p25'])
        print(f"{'r(cm)':>6s} {'mu':>4s}  best alpha  rate   top   alphas reaching top      rate@0.25")
        for r in R:
            for mu in M:
                ok = [cell[(a, r, mu)] for a in A if not cell[(a, r, mu)]['placement_blowup']]
                ok.sort(key=lambda c: (c['top'], np.nan_to_num(c['rate'], nan=-9)), reverse=True)
                b = ok[0] if ok else None
                tops = [c['alpha'] for c in ok if c['top']]
                base = cell[(0.25, r, mu)] if (0.25, r, mu) in cell else None
                w.writerow([r, mu, b['alpha'] if b else '', b['rate'] if b else '',
                            b['top'] if b else '', ' '.join(f"{a:.2f}" for a in sorted(tops)),
                            base['rate'] if base else ''])
                print(f"{r*100:6.1f} {mu:4.1f}  {b['alpha'] if b else '':>10}  "
                      f"{(b['rate'] if b else float('nan')):5.2f}  {str(b['top'] if b else ''):5s} "
                      f"{' '.join(f'{a:.2f}' for a in sorted(tops)):24s} "
                      f"{(base['rate'] if base else float('nan')):5.2f}")
    n_blow = sum(r['placement_blowup'] for r in rows)
    print(f"\n{len(rows)} runs; {n_blow} blew up on placement (self-intersecting start, "
          f"not a climbing result); figure in {out}")


if __name__ == '__main__':
    main()
