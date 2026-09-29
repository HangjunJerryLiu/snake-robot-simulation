"""
Control panel for the pole-climbing experiment.

One control that matters, up front: a friction slider. It sets the STARTING
friction before you press Start. Once the 3-D viewer opens (with Immersive
ticked), a matching slider is drawn right in that window, and THAT is the one
you drag while the robot climbs -- this panel's slider goes read-only and
just mirrors it, since dragging a control in a window you aren't looking at
is not useful. With the viewer off, or Immersive unticked, this slider stays
the live control. Either way, whatever value is in force is written to every
row of run.csv.

Everything else keeps its default and is one button away under "Show all
parameters", or available from the command line (`python climb_rig.py
--help`). The simulation itself is climb_rig.py; this file only collects
numbers, starts and stops it, and shows what it is doing.

    python experiment_gui.py

Why the panel drives the simulation directly rather than using a worker
thread: the MuJoCo viewer is an OpenGL window, and OpenGL contexts belong to
the thread that made them. So the run loop stays on the main thread and the
progress callback pumps the Tk event loop instead -- which is also what makes
the Stop button, and the slider, responsive mid-run.
"""

import dataclasses
import os
import subprocess
import sys
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np

import climb_rig
from climb_rig import Config
# Same bounds the native slider in the 3-D window uses (immersive.py) -- one
# definition, imported here, so the two controls can never drift apart.
from immersive import MU_MIN, MU_MAX, MU_STEP

# Fields under "Show all parameters". Friction is deliberately NOT here: the
# slider owns it, and two controls for one number is how they end up
# disagreeing.
GROUPS = [
    ("Environment", [
        ('pole_radius', 'Pole radius', 'm', '0.02 - 0.10'),
        ('mu_floor', 'Friction, floor', 'mu', 'approach stage only'),
        ('mass_scale', 'Mass scale', 'x', 'nominal robot is 738 g'),
        ('gravity', 'Gravity', 'm/s2', ''),
    ]),
    ("Solver and timing", [
        ('timestep', 'Physics timestep', 's', 'smaller = stabler, slower'),
        ('control_period', 'Control period', 's', "0.1 = the paper's 10 Hz rate"),
        ('max_duration', 'Max simulated time', 's', ''),
        ('target_speedup', 'Playback speed', 'x', 'simulated s per wall-clock s'),
        ('target_fps', 'Viewer frame rate', 'fps', ''),
    ]),
    ("Controller (helical rolling)", [
        ('alpha', 'Pitch angle alpha', 'rad', '0.20 - 0.30 in the paper'),
        ('psi_dot', 'Roll rate psi_dot', 'rad/s', 'the only term that climbs'),
        ('spin', 'Spin direction', '+1/-1', '+1 climbs with lead +1'),
        ('k_mid', 'Compliance K, middle', '-', 'squeeze per tick, eq. (23)'),
        ('k_end', 'Compliance K, ends', '-', 'lower: the ends are cantilevered'),
        ('lead_sign', 'Wrap handedness', '+1/-1', 'must match Spin to climb'),
    ]),
    ("Start: catching the pole", [
        ('wrap_alpha', 'Catching pitch angle', 'rad', 'near flat, or it screws off'),
        ('wrap_radius', 'Catching radius', 'm', 'tighter than the pole'),
        ('wrap_psi', 'Catching roll phase', 'rad', 'which PLANE the ring closes in'),
        ('alpha_ramp', 'Pitch open time', 's', 'catching -> climbing'),
        ('stop_height', 'Stop height', 'm', 'freeze the roll above this'),
    ]),
    ("Start: placed on the pole", [
        ('start_radius', 'Placed helix radius', 'm', 'only used when placed'),
        ('start_height', 'Placed height', 'm', 'lowest point of the wrap'),
    ]),
    ("Logging", [
        ('log_hz', 'CSV sample rate', 'Hz', '50 Hz -> ~12 MB for a full run'),
        ('label', 'Run label', '', 'goes into the results folder name'),
    ]),
]

LIVE_FIELDS = [
    ('stage', 'Stage'), ('t', 'Sim time (s)'), ('z', 'Height (m)'),
    ('peak_z', 'Peak height (m)'), ('rate_cm_s', 'Climb rate (cm/s)'),
    ('coil_r', 'Coil radius (m)'), ('squeeze_mm', 'Grip squeeze (mm)'),
    ('axis_tilt', 'Axis tilt (deg)'), ('n_pole', 'Pole contacts'),
    ('n_self', 'Self contacts'), ('mu_p95', 'Friction needed (p95)'),
    ('saturated', 'Motors saturated'), ('wall', 'Wall clock (s)'),
]

_CASTS = {'float': float, 'int': int, 'str': str, 'bool': bool}


class App:
    def __init__(self, root):
        self.root = root
        root.title("Adaptive helical rolling - pole climbing experiment")
        self.vars = {}
        self.running = False
        self.stop_requested = False
        self.last_gui_update = 0.0
        self.last_run_dir = None
        self.last_cfg = None          # set on every Start; Rerun reuses it

        outer = ttk.Frame(root, padding=12)
        outer.grid(sticky='nsew')

        ttk.Label(outer, text="Snake robot climbing a pole by adaptive helical rolling",
                  font=('Segoe UI', 12, 'bold')).grid(row=0, column=0, sticky='w')
        ttk.Label(outer, text="Takemori, Tanaka & Matsuno, IEEE T-RO 39(1):437-451, 2023."
                              "   Each run writes CSVs to results/.",
                  foreground='#555').grid(row=1, column=0, sticky='w', pady=(0, 10))

        self._build_slider(outer, row=2)
        self._build_controls(outer, row=3)
        self._build_advanced(outer, row=4)
        self._build_live(outer, row=5)
        self._build_log(outer, row=6)

        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(6, weight=1)
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.say("Ready. Press Start. With the 3-D viewer open, drag the friction "
                 "slider drawn IN that window --")
        self.say("this panel's slider then just mirrors the value; without the viewer "
                 "(or with Immersive off), drag it here instead.")
        self.say("Measured, held constant for a whole run: mu 0.25 -> 1.47 cm/s, "
                 "0.3 -> 2.27, 0.5 -> 3.69, 0.8 -> 4.30. Below ~0.25 it slides.")
        self.say("In the 3-D window: SPACE pause | B camera | N annotations | Z panel | "
                 "C contact arrows | H menus")

    # -- the one control that matters --------------------------------------
    def _build_slider(self, parent, row):
        frame = ttk.LabelFrame(parent, text="Friction coefficient, robot <-> pole",
                               padding=(12, 8))
        frame.grid(row=row, column=0, sticky='ew', pady=(0, 8))
        frame.columnconfigure(0, weight=1)

        self.mu_var = tk.DoubleVar(value=Config().mu_robot_pole)
        self.scale = ttk.Scale(frame, from_=MU_MIN, to=MU_MAX, orient='horizontal',
                               variable=self.mu_var, command=self._on_slider)
        self.scale.grid(row=0, column=0, sticky='ew', padx=(0, 12))

        self.mu_label = ttk.Label(frame, text=f"{Config().mu_robot_pole:.2f}",
                                  font=('Consolas', 20), width=6, anchor='e')
        self.mu_label.grid(row=0, column=1, rowspan=2, sticky='e')

        ticks = ttk.Frame(frame)
        ticks.grid(row=1, column=0, sticky='ew', padx=(0, 12))
        for text, anchor in (("0.1", 'w'), ("0.3  plastic", 'w'),
                             ("0.7  rubber", 'center'),
                             ("1.5  grip pads", 'e')):
            ttk.Label(ticks, text=text, foreground='#777', font=('Segoe UI', 8),
                      anchor=anchor).pack(side='left', expand=True, fill='x')

        ttk.Label(frame, text="Drag during a run: friction is applied to the live model, "
                              "and the value in force is logged on every row.",
                  foreground='#555', font=('Segoe UI', 8)).grid(
            row=2, column=0, columnspan=2, sticky='w', pady=(6, 0))

    def _on_slider(self, _value=None):
        self.mu_label.configure(text=f"{self.mu_value():.2f}")

    def mu_value(self):
        """The slider, snapped to MU_STEP so runs are reproducible."""
        return round(round(float(self.mu_var.get()) / MU_STEP) * MU_STEP, 4)

    # -- buttons and toggles ------------------------------------------------
    def _build_controls(self, parent, row):
        bar = ttk.Frame(parent)
        bar.grid(row=row, column=0, sticky='w', pady=(0, 8))

        self.viewer_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text="Show 3-D viewer",
                        variable=self.viewer_var).pack(side='left', padx=(0, 6))
        self.immersive_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text="Immersive",
                        variable=self.immersive_var).pack(side='left', padx=(0, 6))
        # Starting already wrapped is what all four of the paper's experiments
        # do. Use it when the question is about the controller: otherwise a
        # parameter that happens to break the floor gait looks like a climbing
        # failure, which it is not.
        self.wrapped_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(bar, text="Start placed on the pole",
                        variable=self.wrapped_var).pack(side='left', padx=(0, 16))

        self.start_btn = ttk.Button(bar, text="Start", command=self.start, width=12)
        self.start_btn.pack(side='left', padx=2)
        self.stop_btn = ttk.Button(bar, text="Stop", command=self.stop,
                                   width=12, state='disabled')
        self.stop_btn.pack(side='left', padx=2)
        # Enabled once a run has been attempted, whatever the outcome --
        # success, Stop, or a crash. One click repeats the exact same
        # configuration rather than needing everything re-entered.
        self.rerun_btn = ttk.Button(bar, text="Rerun", command=self.rerun,
                                    width=12, state='disabled')
        self.rerun_btn.pack(side='left', padx=2)
        ttk.Button(bar, text="Open results folder",
                   command=self.open_results).pack(side='left', padx=2)
        self.adv_btn = ttk.Button(bar, text="Show all parameters",
                                  command=self.toggle_advanced, width=18)
        self.adv_btn.pack(side='left', padx=2)

    # -- everything else, collapsed ----------------------------------------
    def _build_advanced(self, parent, row):
        self.advanced = ttk.Frame(parent)
        self.advanced.grid(row=row, column=0, sticky='ew')
        self.advanced.grid_remove()
        self.advanced_shown = False

        defaults = Config()
        for i, (group, fields) in enumerate(GROUPS):
            frame = ttk.LabelFrame(self.advanced, text=group, padding=8)
            frame.grid(row=i // 3, column=i % 3, sticky='nsew', padx=4, pady=4)
            for j, (attr, label, unit, hint) in enumerate(fields):
                ttk.Label(frame, text=label).grid(row=j, column=0, sticky='w', pady=1)
                var = tk.StringVar(value=self._fmt(getattr(defaults, attr)))
                self.vars[attr] = var
                ttk.Entry(frame, textvariable=var, width=9).grid(row=j, column=1,
                                                                 sticky='w', padx=4)
                ttk.Label(frame, text=unit, foreground='#555',
                          width=5).grid(row=j, column=2, sticky='w')
                if hint:
                    ttk.Label(frame, text=hint, foreground='#888',
                              font=('Segoe UI', 8)).grid(row=j, column=3, sticky='w')
        self.advanced.columnconfigure((0, 1, 2), weight=1)

    def toggle_advanced(self):
        self.advanced_shown = not self.advanced_shown
        if self.advanced_shown:
            self.advanced.grid()
            self.adv_btn.configure(text="Hide parameters")
        else:
            self.advanced.grid_remove()
            self.adv_btn.configure(text="Show all parameters")

    # -- live readout (numbers, no plots -- the plots are run.csv) ----------
    def _build_live(self, parent, row):
        live = ttk.LabelFrame(parent, text="Live", padding=8)
        live.grid(row=row, column=0, sticky='ew', pady=(8, 4))
        self.live = {}
        for i, (key, label) in enumerate(LIVE_FIELDS):
            r, c = divmod(i, 5)
            cell = ttk.Frame(live)
            cell.grid(row=r, column=c, sticky='w', padx=10, pady=2)
            ttk.Label(cell, text=label, foreground='#555',
                      font=('Segoe UI', 8)).grid(row=0, column=0, sticky='w')
            v = ttk.Label(cell, text='-', font=('Consolas', 11), width=12)
            v.grid(row=1, column=0, sticky='w')
            self.live[key] = v

    def _build_log(self, parent, row):
        frame = ttk.LabelFrame(parent, text="Run log", padding=4)
        frame.grid(row=row, column=0, sticky='nsew', pady=(4, 0))
        self.log = tk.Text(frame, height=10, width=104, font=('Consolas', 9), wrap='none')
        self.log.grid(row=0, column=0, sticky='nsew')
        sb = ttk.Scrollbar(frame, orient='vertical', command=self.log.yview)
        sb.grid(row=0, column=1, sticky='ns')
        self.log.configure(yscrollcommand=sb.set, state='disabled')
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def _fmt(v):
        return f"{v:.6g}" if isinstance(v, float) else str(v)

    def say(self, msg):
        self.log.configure(state='normal')
        self.log.insert('end', msg + '\n')
        self.log.see('end')
        self.log.configure(state='disabled')

    def open_results(self):
        path = os.path.abspath(self.last_run_dir or Config().out_dir)
        os.makedirs(path, exist_ok=True)
        if sys.platform == 'win32':
            os.startfile(path)
        elif sys.platform == 'darwin':
            subprocess.Popen(['open', path])
        else:
            subprocess.Popen(['xdg-open', path])

    def collect(self):
        """Read the controls into a Config, or raise ValueError naming the field."""
        cfg = Config()
        cfg.mu_robot_pole = self.mu_value()
        types = {f.name: f.type for f in dataclasses.fields(Config)}
        for attr, var in self.vars.items():
            text = var.get().strip()
            cast = _CASTS[types[attr]]
            try:
                setattr(cfg, attr, text if cast is str else cast(text))
            except ValueError:
                raise ValueError(f"{attr!r}: {text!r} is not a valid "
                                 f"{types[attr]}") from None
        cfg.show_viewer = bool(self.viewer_var.get())
        cfg.immersive = bool(self.immersive_var.get())
        cfg.start_wrapped = bool(self.wrapped_var.get())

        # Cheap sanity checks, so a typo fails here rather than 30 s into a run.
        if not MU_MIN <= cfg.mu_robot_pole <= MU_MAX:
            raise ValueError(f"friction must be between {MU_MIN} and {MU_MAX}")
        if cfg.timestep <= 0 or cfg.timestep > 0.01:
            raise ValueError("timestep must be in (0, 0.01] s; 0.001 is the default")
        if cfg.control_period < cfg.timestep:
            raise ValueError("control period cannot be shorter than the timestep")
        if cfg.pole_radius <= 0:
            raise ValueError("pole radius must be positive")
        if cfg.mass_scale <= 0:
            raise ValueError("mass scale must be positive")
        if cfg.log_hz <= 0:
            raise ValueError("CSV sample rate must be positive")
        if abs(cfg.spin) != 1:
            raise ValueError("spin must be +1 or -1")
        if not 0.0 <= cfg.k_mid < 1.0 or not 0.0 <= cfg.k_end < 1.0:
            raise ValueError("compliance gains must be in [0, 1)")
        return cfg

    # -- run control --------------------------------------------------------
    def start(self):
        if self.running:
            return
        try:
            cfg = self.collect()
        except ValueError as exc:
            messagebox.showerror("Check the parameters", str(exc))
            return
        self._launch(cfg)

    def rerun(self):
        # Re-run the exact configuration of the last Start, unmodified. This
        # is the recovery path: instead of a failed or stopped run leaving
        # you to re-enter every field, one click repeats precisely what was
        # tried -- including whatever friction it STARTED at, not wherever a
        # mid-run drag left it.
        if self.running or self.last_cfg is None:
            return
        self.say("")
        self.say("=== rerunning the last configuration ===")
        self._launch(dataclasses.replace(self.last_cfg))

    def _launch(self, cfg):
        self.last_cfg = cfg
        self.running = True
        self.stop_requested = False
        self.start_btn.configure(state='disabled')
        self.stop_btn.configure(state='normal')
        self.rerun_btn.configure(state='disabled')
        # The native slider in the 3-D window becomes the live control once it
        # opens; dragging this one during that run would silently do nothing,
        # which is worse than it looking inert. Re-enabled in `finally` below.
        if cfg.show_viewer and cfg.immersive:
            self.scale.configure(state='disabled')
        for v in self.live.values():
            v.configure(text='-')
        self.say("")
        self.say(f"=== starting {'placed on the pole' if cfg.start_wrapped else 'from the floor'}"
                 f", friction {cfg.mu_robot_pole:.2f} ===")
        self.root.update()

        try:
            summary = climb_rig.run(cfg,
                                    on_progress=self.on_progress,
                                    should_stop=lambda: self.stop_requested,
                                    on_message=self.on_message,
                                    live_mu=self.mu_value)
            self.last_run_dir = summary['run_dir']
            self.say("--- finished ---")
            for key in ('peak_com_z_m', 'reached_top', 'time_to_top_s',
                        'rolling_climb_m', 'rolling_rate_cm_s', 'mu_at_end',
                        'body_roll_rev', 'orbit_rev', 'diverged', 'wall_time_s'):
                self.say(f"    {key:18s} {summary[key]}")
            self.say(f"    CSVs in {summary['run_dir']}")
        except Exception as exc:                      # keep the panel alive
            self.say(f"!!! run failed: {type(exc).__name__}: {exc}")
            self.say("    click Rerun to try the exact same configuration again")
            messagebox.showerror("Run failed - click Rerun to try again",
                                 f"{type(exc).__name__}: {exc}")
        finally:
            self.running = False
            self.start_btn.configure(state='normal')
            self.stop_btn.configure(state='disabled')
            self.rerun_btn.configure(state='normal')
            self.scale.configure(state='normal')

    def stop(self):
        if self.running:
            self.stop_requested = True
            self.say("stop requested...")

    def on_message(self, msg):
        self.say(msg)
        self.root.update()

    def on_progress(self, p):
        # The rig calls this at the CSV rate (50 Hz by default). Redrawing Tk
        # that often would cost more than the physics, so throttle to ~10 Hz;
        # the update() call is also what lets the Stop button and the slider
        # be noticed mid-run.
        if p['wall'] - self.last_gui_update < 0.1:
            return
        self.last_gui_update = p['wall']
        fmt = {
            'stage': str,
            't': lambda v: f"{v:8.2f}", 'z': lambda v: f"{v:8.3f}",
            'peak_z': lambda v: f"{v:8.3f}", 'rate_cm_s': lambda v: f"{v:8.2f}",
            'coil_r': lambda v: f"{v:8.4f}" if np.isfinite(v) else '-',
            'squeeze_mm': lambda v: f"{v:8.2f}" if np.isfinite(v) else '-',
            'axis_tilt': lambda v: f"{v:8.1f}" if np.isfinite(v) else '-',
            'n_pole': lambda v: f"{v:8d}", 'n_self': lambda v: f"{v:8d}",
            'mu_p95': lambda v: f"{v:8.2f}", 'saturated': lambda v: f"{v:8d}",
            'wall': lambda v: f"{v:8.1f}",
        }
        for key, _ in LIVE_FIELDS:
            if key in p:
                self.live[key].configure(text=fmt[key](p[key]))
        if 'mu_robot_pole' in p:
            # Mirrors whichever slider is actually authoritative right now --
            # the native one in the 3-D window, most of the time.
            self.mu_var.set(p['mu_robot_pole'])
            self.mu_label.configure(text=f"{p['mu_robot_pole']:.2f}")
        self.root.update()

    def on_close(self):
        if self.running:
            self.stop_requested = True
            self.say("stopping before close...")
            return
        self.root.destroy()


def main():
    root = tk.Tk()
    try:
        ttk.Style().theme_use('vista')
    except tk.TclError:
        pass
    App(root)
    root.mainloop()


if __name__ == '__main__':
    main()
