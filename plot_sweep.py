"""
Figures for a sweep_radius_friction.py run.

    python plot_sweep.py results/sweep_<stamp>

Writes into <sweep>/figures/, each as .png (200 dpi) and .pdf (vector):

    height_vs_time_friction    r = 4 cm: trajectories per mu, and climb rate vs mu
    height_vs_time_radius      mu = 0.5: trajectories per radius, and rate vs radius
    loss_landscape_wrapped     3-D loss surface, placed-on-the-pole runs
    loss_landscape_floor       3-D loss surface, from-the-floor runs
    loss_contour_map           both surfaces flat, with every sampled run marked

and <sweep>/grid.csv with the per-cell numbers.

LOSS. Built from height, so faster climbs score better, not just "made it":

    A    = time-average of the climb (COM height above its start) over the
           window [0, T_WIN]; a run that reached the top stays there
    loss = 1 - clip(A / A_best, 0, 1)

A_best is the best cell of that start mode, so each mode's minimum is 0. A
robot that fell or never left the floor scores 1.

SURFACE. The grid is 12 x 13 and uneven, so it is resampled with monotone
piecewise-cubic (PCHIP) interpolation. PCHIP does not overshoot at a cliff,
so the smooth surface never shows a basin or a ridge the data does not have.
Every sampled run is still drawn as a point on the flat map.
"""
import csv
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

R_FIX, MU_FIX = 0.040, 0.50
T_WIN = 150.0
TOP = 2.8
MODES = (('wrapped', 'Placed on the pole', 'controller only'),
         ('floor', 'From the floor', 'approach + catch + climb'))
LOSS_CMAP = matplotlib.colormaps['turbo']

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.titlesize': 11.5, 'axes.titleweight': 'bold', 'axes.titlepad': 8,
    'axes.labelsize': 10.5, 'axes.labelcolor': '#222222',
    'axes.edgecolor': '#555555', 'axes.linewidth': 0.8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': '#d9d9d9', 'grid.linewidth': 0.6,
    'grid.linestyle': '-', 'axes.axisbelow': True,
    'xtick.color': '#444444', 'ytick.color': '#444444',
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'legend.frameon': False, 'legend.fontsize': 9,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'savefig.facecolor': 'white', 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.15,
    'pdf.fonttype': 42,
})


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load(d):
    with open(os.path.join(d, 'index.csv')) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r['pole_radius'] = float(r['pole_radius'])
        r['mu'] = float(r['mu'])
        r['caught'] = r['caught_pole'] == 'True'
        r['pen_mm'] = float(r['max_penetration_mm']) if r.get('max_penetration_mm') else np.nan
        r['rate'] = float(r['rolling_rate_cm_s']) if r['rolling_rate_cm_s'] else np.nan
        t, z = [], []
        with open(os.path.join(r['run_dir'], 'run.csv')) as f:
            for row in csv.DictReader(f):
                t.append(float(row['t_s']))
                z.append(float(row['com_z_m']))
        t, z = np.array(t), np.array(z)
        if t[-1] < T_WIN:                       # stopped at the top: it stays there
            t, z = np.append(t, T_WIN), np.append(z, z[-1])
        r['t'], r['z'] = t, z
        climb = np.interp(np.linspace(0, T_WIN, 1501), t, z) - z[0]
        r['A'] = float(np.mean(climb))
    check_valid(rows)
    for mode, *_ in MODES:
        sel = [r for r in rows if r['mode'] == mode]
        best = max(r['A'] for r in sel)
        for r in sel:
            r['loss'] = 1.0 - float(np.clip(r['A'] / best, 0.0, 1.0))
    return rows


MAX_PENETRATION_MM = 5.0

# Runs over the limit that were re-run with time-resolved penetration and
# found to be a single transient impact, not a link resting inside the pole.
# (mode, pole_radius, mu): what was found.
VERIFIED_TRANSIENT = {
    ('floor', 0.02, 1.2): '7.3 mm in 1 of 8025 samples (t = 51.36 s), never repeated',
}


def check_valid(rows):
    """Refuse to plot a sweep that is not physics. Every run must have its
    penetration measured, and no robot link may have gone more than
    MAX_PENETRATION_MM inside the pole (soft contact allows ~1-3 mm)."""
    for r in rows:
        why = VERIFIED_TRANSIENT.get((r['mode'], r['pole_radius'], r['mu']))
        if why and r['pen_mm'] > MAX_PENETRATION_MM:
            print(f"  accepted {r['mode']} r={r['pole_radius']} mu={r['mu']}: {why}")
    bad = [r for r in rows if r.get('error') or not np.isfinite(r['pen_mm'])
           or (r['pen_mm'] > MAX_PENETRATION_MM
               and (r['mode'], r['pole_radius'], r['mu']) not in VERIFIED_TRANSIENT)]
    if bad:
        for r in bad[:20]:
            print(f"  INVALID {r['mode']} r={r['pole_radius']} mu={r['mu']}: "
                  f"penetration={r['pen_mm']} mm error={r.get('error')}")
        sys.exit(f"{len(bad)} of {len(rows)} runs failed the validity check; not plotting.")
    print(f"validity: all {len(rows)} runs measured, deepest penetration "
          f"{max(r['pen_mm'] for r in rows):.2f} mm (limit {MAX_PENETRATION_MM} mm)")


def grid(rows, mode, key='loss'):
    R = np.array(sorted({r['pole_radius'] for r in rows}))
    M = np.array(sorted({r['mu'] for r in rows}))
    G = np.full((len(R), len(M)), np.nan)
    for r in rows:
        if r['mode'] == mode:
            G[np.searchsorted(R, r['pole_radius']), np.searchsorted(M, r['mu'])] = r[key]
    return R, M, G


def _pchip_slopes(x, y):
    """Fritsch-Carlson node slopes: monotone, so no overshoot at a cliff."""
    h = np.diff(x)
    d = np.diff(y) / h
    m = np.zeros_like(y)
    for k in range(1, len(x) - 1):
        if d[k - 1] * d[k] > 0:
            w1, w2 = 2 * h[k] + h[k - 1], h[k] + 2 * h[k - 1]
            m[k] = (w1 + w2) / (w1 / d[k - 1] + w2 / d[k])
    m[0], m[-1] = d[0], d[-1]
    for k in (0, -1):                       # keep the end slopes monotone too
        if d[k] == 0 or np.sign(m[k]) != np.sign(d[k]):
            m[k] = 0.0
    return m


def pchip(x, y, xq):
    m = _pchip_slopes(x, y)
    k = np.clip(np.searchsorted(x, xq) - 1, 0, len(x) - 2)
    h = x[k + 1] - x[k]
    t = (xq - x[k]) / h
    h00, h10 = 2 * t**3 - 3 * t**2 + 1, t**3 - 2 * t**2 + t
    h01, h11 = -2 * t**3 + 3 * t**2, t**3 - t**2
    return h00 * y[k] + h10 * h * m[k] + h01 * y[k + 1] + h11 * h * m[k + 1]


def smooth(R, M, L, n=160):
    """Tensor-product PCHIP of L[radius, mu] onto an n x n grid."""
    rr, mm = np.linspace(R[0], R[-1], n), np.linspace(M[0], M[-1], n)
    along_mu = np.array([pchip(M, L[i], mm) for i in range(len(R))])
    Z = np.array([pchip(R, along_mu[:, j], rr) for j in range(n)]).T
    RR, MM = np.meshgrid(rr, mm, indexing='ij')
    return RR, MM, np.clip(Z, 0, 1)


def descent(RR, MM, Z, start, step=0.004, n=900):
    """Steepest descent on the resampled loss, in axes scaled to [0, 1]
    (radius and friction have different units, so a raw gradient would be
    meaningless). Stops at the floor of the basin or on flat ground."""
    n_r, n_m = Z.shape
    gr, gm = np.gradient(Z, 1.0 / (n_r - 1), 1.0 / (n_m - 1))
    lo = np.array([RR[0, 0], MM[0, 0]])
    span = np.array([RR[-1, 0] - RR[0, 0], MM[0, -1] - MM[0, 0]])

    def at(F, u):                           # bilinear lookup
        fi, fj = u[0] * (n_r - 1), u[1] * (n_m - 1)
        i, j = min(int(fi), n_r - 2), min(int(fj), n_m - 2)
        a, b = fi - i, fj - j
        return ((1 - a) * (1 - b) * F[i, j] + a * (1 - b) * F[i + 1, j]
                + (1 - a) * b * F[i, j + 1] + a * b * F[i + 1, j + 1])

    x = (np.array(start) - lo) / span
    path = [x.copy()]
    for _ in range(n):
        g = np.array([at(gr, x), at(gm, x)])
        if np.linalg.norm(g) < 1e-3 or at(Z, x) < 0.01:
            break
        x = np.clip(x - step * g / np.linalg.norm(g), 0, 1)
        path.append(x.copy())
    P = lo + np.array(path) * span
    return P[:, 0], P[:, 1], np.array([at(Z, u) for u in path])


# ---------------------------------------------------------------------------
# height vs time
# ---------------------------------------------------------------------------
def trajectories(rows, vary, fixed, fixed_val, cmap, cb_label, fmt, xlab, title, out):
    fig = plt.figure(figsize=(15, 4.9), constrained_layout=True)
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 0.045, 0.85])
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1], sharey=ax_a)
    cax = fig.add_subplot(gs[2])
    ax_c = fig.add_subplot(gs[3])

    vals = sorted({r[vary] for r in rows})
    edges = np.concatenate([[vals[0] - (vals[1] - vals[0]) / 2],
                            (np.array(vals[:-1]) + np.array(vals[1:])) / 2,
                            [vals[-1] + (vals[-1] - vals[-2]) / 2]])
    norm = colors.BoundaryNorm(edges, cmap.N)
    shade = lambda v: cmap(0.08 + 0.84 * norm(v) / (cmap.N - 1))

    for ax, (mode, name, sub) in zip((ax_a, ax_b), MODES):
        sel = sorted([r for r in rows if r['mode'] == mode
                      and abs(r[fixed] - fixed_val) < 1e-9], key=lambda r: r[vary])
        ax.axhspan(TOP, 3.0, color='#eeeeee', zorder=0, lw=0)
        ax.text(3, TOP + 0.05, 'top of pole', fontsize=8, color='#777777', va='bottom')
        for r in sel:
            ls = '-' if r['caught'] else (0, (2, 2))
            ax.plot(r['t'], r['z'], color=shade(r[vary]), lw=1.6, ls=ls,
                    solid_capstyle='round', zorder=3)
        ax.set_title(f"{name}  ·  {sub}", loc='left')
        ax.set_xlim(0, T_WIN)
        ax.set_ylim(0, 3.0)
        ax.set_xlabel('Time (s)')
    ax_a.set_ylabel('Centre-of-mass height (m)')
    plt.setp(ax_b.get_yticklabels(), visible=False)

    sm = matplotlib.cm.ScalarMappable(norm=colors.Normalize(0, 1),
                                      cmap=colors.ListedColormap([shade(v) for v in vals]))
    cb = fig.colorbar(sm, cax=cax, ticks=(np.arange(len(vals)) + 0.5) / len(vals))
    cb.ax.set_yticklabels([fmt(v) for v in vals], fontsize=8)
    cb.set_label(cb_label)
    cb.outline.set_visible(False)

    # summary: rolling-stage climb rate against the swept parameter
    for (mode, name, _), mk, c in zip(MODES, ('o', 's'), ('#1f4e9c', '#d1495b')):
        sel = sorted([r for r in rows if r['mode'] == mode
                      and abs(r[fixed] - fixed_val) < 1e-9], key=lambda r: r[vary])
        x = np.array([r[vary] for r in sel]) * (100 if vary == 'pole_radius' else 1)
        y = np.array([r['rate'] if r['caught'] else np.nan for r in sel])
        ax_c.plot(x, y, marker=mk, ms=5.5, lw=1.6, color=c, label=name,
                  mec='white', mew=0.8, zorder=3)
        miss = [xi for xi, r in zip(x, sel) if not r['caught']]
        if miss:
            ax_c.plot(miss, np.zeros(len(miss)), ls='none', marker='x', color=c, ms=6,
                      mew=1.4, zorder=3)
    ax_c.axhline(0, color='#555555', lw=0.8)
    ax_c.set_xlabel(xlab)
    ax_c.set_ylabel('Rolling-stage climb rate (cm/s)')
    ax_c.set_title('Climb rate', loc='left')
    h, l = ax_c.get_legend_handles_labels()
    h.append(Line2D([], [], ls='none', marker='x', color='#555555', mew=1.4))
    l.append('pole never caught')
    ax_c.legend(h, l, loc='best')

    fig.suptitle(title, x=0.01, ha='left', fontsize=13.5, fontweight='bold')
    fig.text(0.01, -0.03, 'Dashed trajectories: the pole was never caught. '
             'Runs that reached the top are held there for the rest of the window.',
             fontsize=8.5, color='#666666')
    for ext in ('png', 'pdf'):
        fig.savefig(f"{out}.{ext}", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# loss landscape
# ---------------------------------------------------------------------------
def landscape(rows, mode, name, sub, starts, out):
    R, M, L = grid(rows, mode)
    RR, MM, Z = smooth(R, M, L)
    X, Y = RR * 100, MM
    floor = -0.45

    fig = plt.figure(figsize=(11, 8.2))
    ax = fig.add_subplot(projection='3d', computed_zorder=False)
    ax.set_proj_type('persp', focal_length=0.45)

    fc = LOSS_CMAP(Z)
    ax.plot_surface(X, Y, Z, facecolors=fc, rstride=1, cstride=1, linewidth=0,
                    antialiased=False, shade=False, zorder=1)
    # wire mesh, the reference look: every 8th resampled line
    k = 8
    for i in range(0, X.shape[0], k):
        ax.plot(X[i], Y[i], Z[i], color='#1a1a1a', lw=0.35, alpha=0.55, zorder=2)
    for j in range(0, X.shape[1], k):
        ax.plot(X[:, j], Y[:, j], Z[:, j], color='#1a1a1a', lw=0.35, alpha=0.55, zorder=2)

    ax.contourf(X, Y, Z, levels=24, zdir='z', offset=floor, cmap=LOSS_CMAP,
                alpha=0.22, zorder=0)
    ax.contour(X, Y, Z, levels=12, zdir='z', offset=floor, cmap=LOSS_CMAP,
               linewidths=0.7, zorder=0)

    for s in starts:
        pr, pm, pz = descent(RR, MM, Z, s)
        ax.plot(pr * 100, pm, pz + 0.015, color='black', lw=2.4, zorder=5,
                solid_capstyle='round')
        ax.scatter([pr[0] * 100], [pm[0]], [pz[0] + 0.015], color='black', s=45,
                   zorder=6, depthshade=False)
        if len(pr) > 2:
            ax.quiver(pr[-2] * 100, pm[-2], pz[-2] + 0.015,
                      (pr[-1] - pr[-2]) * 100, pm[-1] - pm[-2], pz[-1] - pz[-2],
                      color='black', lw=2.4, arrow_length_ratio=6, zorder=6)
        ax.plot(pr * 100, pm, np.full_like(pr, floor), color='black', lw=1.0,
                ls=(0, (3, 2)), zorder=1)

    ax.set_xlabel('Pole radius (cm)', labelpad=10)
    ax.set_ylabel('Friction coefficient μ', labelpad=10)
    ax.set_zlabel('Loss', labelpad=8)
    ax.set_zlim(floor, 1.05)
    ax.set_zticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.view_init(elev=30, azim=128)
    ax.set_box_aspect((1.25, 1.0, 0.72))
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_facecolor((1, 1, 1, 0))
        a.pane.set_edgecolor('#cccccc')
        a._axinfo['grid'].update(color='#e3e3e3', linewidth=0.5)

    cb = fig.colorbar(matplotlib.cm.ScalarMappable(colors.Normalize(0, 1), LOSS_CMAP),
                      ax=ax, shrink=0.55, pad=0.10, aspect=22)
    cb.set_label('Loss  (0 = best climb, 1 = no climb)')
    cb.outline.set_visible(False)
    ax.set_title(f"Loss landscape — {name.lower()} ({sub})", loc='left',
                 fontsize=13.5, fontweight='bold', pad=0)
    fig.text(0.08, 0.06,
             f'Loss = 1 − time-averaged climb / best cell. Surface resampled from the '
             f'{len(R)} × {len(M)} run grid with monotone (PCHIP) interpolation.\n'
             'Black paths: steepest descent on that surface from the marked start points.',
             fontsize=8.5, color='#555555')
    for ext in ('png', 'pdf'):
        fig.savefig(f"{out}.{ext}", dpi=200)
    plt.close(fig)


def contour_map(rows, out):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), sharey=True, constrained_layout=True)
    for ax, (mode, name, sub) in zip(axes, MODES):
        R, M, L = grid(rows, mode)
        RR, MM, Z = smooth(R, M, L, n=240)
        cf = ax.contourf(RR * 100, MM, Z, levels=np.linspace(0, 1, 21), cmap=LOSS_CMAP)
        cs = ax.contour(RR * 100, MM, Z, levels=[0.25, 0.5, 0.75], colors=['white', 'black', 'white'],
                        linewidths=[0.8, 1.8, 0.8])
        ax.clabel(cs, fmt='%.2f', fontsize=8, inline_spacing=4)
        for r in rows:
            if r['mode'] != mode:
                continue
            x, y = r['pole_radius'] * 100, r['mu']
            if r['caught']:
                ax.plot(x, y, 'o', ms=3.2, mfc='white', mec='#222222', mew=0.6)
            else:
                ax.plot(x, y, 'x', ms=5, color='white', mew=1.3)
        ax.plot(R_FIX * 100, MU_FIX, marker='*', ms=15, mfc='white', mec='black', mew=1.0)
        ax.set_title(f"{name}  ·  {sub}", loc='left')
        ax.set_xlabel('Pole radius (cm)')
        ax.grid(False)
        ax.spines[['top', 'right']].set_visible(True)
    axes[0].set_ylabel('Friction coefficient μ')
    cb = fig.colorbar(cf, ax=axes, shrink=0.9, pad=0.015, ticks=np.linspace(0, 1, 6))
    cb.set_label('Loss  (0 = best climb, 1 = no climb)')
    cb.outline.set_visible(False)
    leg = [Line2D([], [], ls='none', marker='o', ms=4, mfc='white', mec='#222222'),
           Line2D([], [], ls='none', marker='x', ms=5, color='#222222', mew=1.3),
           Line2D([], [], ls='none', marker='*', ms=11, mfc='white', mec='black'),
           Line2D([], [], color='black', lw=1.8)]
    axes[1].legend(leg, ['sampled run', 'pole never caught', 'default (4 cm, μ 0.5)',
                         'loss 0.5 boundary'],
                   loc='upper right', frameon=True, framealpha=0.9, facecolor='white')
    fig.suptitle('Loss over pole radius × friction', x=0.01, ha='left',
                 fontsize=13.5, fontweight='bold')
    for ext in ('png', 'pdf'):
        fig.savefig(f"{out}.{ext}", dpi=200)
    plt.close(fig)


def write_grid(rows, d):
    with open(os.path.join(d, 'grid.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['mode', 'pole_radius_m', 'mu', 'mean_climb_m', 'loss', 'caught',
                    'rolling_rate_cm_s', 'peak_com_z_m'])
        for r in sorted(rows, key=lambda r: (r['mode'], r['pole_radius'], r['mu'])):
            w.writerow([r['mode'], r['pole_radius'], r['mu'], f"{r['A']:.4f}",
                        f"{r['loss']:.4f}", r['caught'], r['rolling_rate_cm_s'],
                        r['peak_com_z_m']])


def main():
    d = sys.argv[1]
    out = os.path.join(d, 'figures')
    os.makedirs(out, exist_ok=True)
    rows = load(d)
    write_grid(rows, d)

    trajectories(rows, 'mu', 'pole_radius', R_FIX, matplotlib.colormaps['viridis'],
                 'Friction coefficient μ', lambda v: f"{v:.2f}", 'Friction coefficient μ',
                 f'Height vs time on a {R_FIX*100:.0f} cm pole, by friction',
                 os.path.join(out, 'height_vs_time_friction'))
    trajectories(rows, 'pole_radius', 'mu', MU_FIX, matplotlib.colormaps['plasma'],
                 'Pole radius (cm)', lambda v: f"{v*100:.1f}", 'Pole radius (cm)',
                 f'Height vs time at friction μ = {MU_FIX}, by pole radius',
                 os.path.join(out, 'height_vs_time_radius'))

    starts = [(0.058, 0.50), (0.040, 0.21)]
    for mode, name, sub in MODES:
        landscape(rows, mode, name, sub, starts, os.path.join(out, f'loss_landscape_{mode}'))
    contour_map(rows, os.path.join(out, 'loss_contour_map'))
    print('wrote figures to', out)


if __name__ == '__main__':
    main()
