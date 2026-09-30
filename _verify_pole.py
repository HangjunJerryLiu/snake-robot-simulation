"""Temporary verification script: is a runtime-resized pole a real collider?"""
import sys, os, mujoco, numpy as np
import climb_rig
from climb_rig import Config, run

r, mode, dur = float(sys.argv[1]), sys.argv[2], float(sys.argv[3])
if mode == 'xml':
    xml = open('9motor_sidewinder_pipe.xml').read()
    s = xml.replace('size="0.04 1.5"', f'size="{r} 1.5"', 1)
    open('_verify_tmp.xml', 'w').write(s)
    climb_rig.MODEL_XML = '_verify_tmp.xml'

stats = dict(n=0, worst=np.inf, deep=0, samples=0, times=[])
_step = mujoco.mj_step
def step(m, d):
    _step(m, d)
    stats['n'] += 1
    if stats['n'] % 20:
        return
    pipe = m.geom('pipe').id
    worst = np.inf
    for g in range(m.ngeom):
        if m.geom_bodyid[g] == 0:
            continue
        dist = mujoco.mj_geomDistance(m, d, pipe, g, 0.05, None)
        worst = min(worst, dist)
    stats['worst'] = min(stats['worst'], worst)
    stats['samples'] += 1
    stats["deep"] += worst < -0.005
    if worst < -0.003: stats["times"].append((round(d.time, 2), round(worst * 1000, 1)))
mujoco.mj_step = step

cfg = Config(pole_radius=r, mu_robot_pole=float(sys.argv[5]), max_duration=dur, start_wrapped=(sys.argv[4] == 'wrapped'),
             start_radius=r + 0.045, out_dir=os.path.join('results', '_verify'),
             label=f"{mode}_{sys.argv[4]}_r{r}", log_hz=10)
s = run(cfg)
print(f"r={r} build={mode} start={sys.argv[4]}: peak={s['peak_com_z_m']} final={s['final_com_z_m']} "
      f"caught={s['caught_pole']} | deepest penetration={stats['worst']*1000:.1f} mm, "
      f"samples with >5 mm penetration: {stats['deep']}/{stats['samples']}")
print("times with >3 mm:", stats["times"][:40], "count", len(stats["times"]))
