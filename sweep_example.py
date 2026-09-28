"""
One parameter at a time, from a pre-wrapped start: the sweep that produced the
sensitivity table in README.md.

Also a worked example of scripting the rig -- `Config(...)` plus `run(...)` is
the whole API, and every run leaves its own folder of CSVs behind, so a sweep
is just a loop and a concatenation of the summary.csv files afterwards.

start_wrapped=True matters here. Varying friction or mass also changes how the
floor gait travels, so a run started from the floor can fail to CATCH the pole
and look like a climbing failure when the controller was never given a chance.
Placing the robot on the pole -- which is how all four of the paper's own
experiments begin -- isolates the question being asked.

    python sweep_example.py
"""
import csv
import os

from climb_rig import Config, run

CASES = [
    ("baseline",        dict()),
    ("mu 1.0",          dict(mu_robot_pole=1.0)),
    ("mu 1.5",          dict(mu_robot_pole=1.5)),
    ("mu 3.0",          dict(mu_robot_pole=3.0)),
    ("mass x0.5",       dict(mass_scale=0.5)),
    ("mass x2",         dict(mass_scale=2.0)),
    ("mass x4",         dict(mass_scale=4.0)),
    ("dt 2 ms",         dict(timestep=0.002)),
    ("dt 0.5 ms",       dict(timestep=0.0005)),
    ("ctrl 0.2 s",      dict(control_period=0.2)),
    ("ctrl 0.02 s",     dict(control_period=0.02)),
    ("alpha 0.15",      dict(alpha=0.15)),
    ("alpha 0.35",      dict(alpha=0.35)),
    ("psi_dot x0.5",    dict(psi_dot=0.7854)),
    ("psi_dot x2",      dict(psi_dot=3.14159)),
    ("K 0.02",          dict(k_mid=0.02, k_end=0.01)),
    ("K 0.15",          dict(k_mid=0.15, k_end=0.075)),
    ("pole r 0.025",    dict(pole_radius=0.025, start_radius=0.070)),
    ("pole r 0.06",     dict(pole_radius=0.060, start_radius=0.105)),
]


def rolling_means(run_dir):
    pole, mu, sat = [], [], []
    with open(os.path.join(run_dir, 'run.csv')) as f:
        for row in csv.DictReader(f):
            if row['stage'] != 'rolling':
                continue
            pole.append(float(row['n_contact_pole']))
            mu.append(float(row['mu_required_p95']))
            sat.append(float(row['joints_saturated']))
    n = max(1, len(pole))
    return sum(pole) / n, sum(mu) / n, sum(sat) / n


print(f"| case | rolling cm/s | climb m | pole contacts | mu needed (p95) | motors saturated |")
print(f"| --- | --- | --- | --- | --- | --- |")
for name, over in CASES:
    cfg = Config(label=name.replace(' ', '_').replace('.', 'p'),
                 start_wrapped=True, max_duration=60.0, **over)
    s = run(cfg)
    pole, mu, sat = rolling_means(s['run_dir'])
    print(f"| {name} | {s['rolling_rate_cm_s']} | {s['rolling_climb_m']} | "
          f"{pole:.1f} | {mu:.2f} | {sat:.1f} |"
          + ("  DIVERGED" if s['diverged'] else ""))
