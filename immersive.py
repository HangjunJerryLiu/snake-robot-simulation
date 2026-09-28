"""
Immersive MuJoCo view of the climbing experiment.

The plain viewer shows a robot on a pole from a camera that does not move, so
within twenty seconds the robot has climbed out of frame and everything the
controller is doing is invisible. This wraps it so that the run is something
you can actually watch, read, and steer:

  * the camera FOLLOWS the robot up the pole, and can orbit while it does;
  * a heads-up panel carries the live experiment state -- stage, height, climb
    rate, grip, friction demand, motor saturation;
  * a 2-D PLOT, bottom-left, own dedicated space, shows the cross-sectional
    shape the controller is working with -- measured vs. commanded -- the way
    the paper's Fig. 11 draws it, not overlaid on the robot;
  * a FRICTION SLIDER, drawn and dragged right in this window, sets the
    robot/pole sliding friction live while the sim runs. This is the
    authoritative control when this viewer is open -- drag it and watch the
    climb speed up, stall or slip in real time;
  * a height ruler runs up the pole so progress is legible at a glance;
  * a faint ghost of the commanded joint targets shows the tracking error the
    servos are turning into grip force.

Deliberately NOT drawn: anything that sticks out of the body. Contact-force
arrows and a 3-D axis rod were both tried and both made the wrap harder to
read, not easier -- a dozen arrows radiating off a coiled body hide the coil.
The contact arrows are still one keypress away (C) when a specific question
needs them.

Everything here is decoration plus the one thing that isn't: dragging the
slider calls back into the running experiment through `self.mu`, which
climb_rig.py polls once per log tick. Nothing else here writes to `data`, so a
run looks identical with the viewer off.

Keys added on top of the viewer's own (SPACE pause, C contacts, H menus, ...):

    B   camera: follow -> orbit -> free
    N   annotations: full -> plot only -> off
    Z   hide/show the heads-up panel
    G   also toggles the cross-section plot (shared with the base viewer's
        own "hide graph" key, which otherwise has nothing left to hide)
"""

import time

import glfw
import mujoco
import mujoco_viewer
import numpy as np

import helical_rolling as hr

CAM_MODES = ('follow', 'orbit', 'free')
ANNOTATION_LEVELS = ('full', 'plot', 'off')

# The friction slider's range. 1.0 is bare plastic on steel, 2.0 assumes soft
# high-grip elastomer pads, and much above 3 is not a real material pair --
# it is there so you can see what the gait would do with grip it cannot
# actually have. Matches experiment_gui.py, which imports these rather than
# repeating them, so the two controls can never drift apart.
MU_MIN, MU_MAX, MU_STEP = 0.5, 4.0, 0.05

# Colours, kept consistent between the plot and the ghost body.
C_MEASURED = (0.20, 0.85, 1.00)      # cyan   - what the robot IS
C_TARGET = (1.00, 0.45, 0.10)        # orange - what it is ASKED for
C_GHOST = (1.00, 0.45, 0.10, 0.35)   # orange, faint - the target form, in 3-D
C_RULER = (0.85, 0.85, 0.90, 0.75)
C_SLIDER_FILL = (0.20, 0.85, 1.00, 0.95)
C_SLIDER_TRACK = (0.12, 0.13, 0.16, 0.92)
C_SLIDER_BORDER = (0.04, 0.04, 0.05, 0.95)

# Layout, in pixels. MARGIN is the gap from every screen edge; the block
# heights stack bottom-up as: hint text, then the track itself, then a header
# line above it. RIGHT_RESERVE keeps the slider clear of the heads-up panel's
# legend text in the bottom-right corner.
MARGIN = 20
SLIDER_HINT_H = 18
SLIDER_TRACK_H = 50
SLIDER_HEADER_GAP = 8
RIGHT_RESERVE = 300
SLIDER_MIN_W = 200

PLOT_MIN, PLOT_MAX = 220, 340   # the 2-D panel is square; side length clamps here
PLOT_FRACTION = 0.24            # ...as this fraction of the shorter screen edge
PROJ_RANGE = 0.13                # m, fixed half-width so the shape keeps true aspect


def _z_to(direction):
    """Rotation matrix whose z axis points along `direction`."""
    z = np.asarray(direction, float)
    n = np.linalg.norm(z)
    if n < 1e-12:
        return np.eye(3)
    z = z / n
    ref = np.array([1.0, 0.0, 0.0])
    if abs(z @ ref) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    x = np.cross(ref, z)
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


class ImmersiveViewer(mujoco_viewer.MujocoViewer):
    """MujocoViewer that knows what experiment it is showing, and can steer it."""

    def __init__(self, model, data, pole_radius=0.04, pole_height=3.0,
                 mu=2.0, title="Adaptive helical rolling"):
        super().__init__(model, data, title=title)

        self.pole_radius = float(pole_radius)
        self.pole_height = float(pole_height)

        # THE authoritative friction value once this viewer is open. climb_rig
        # reads it once per log tick and applies it to the model; dragging the
        # slider below is the only thing that changes it.
        self.mu = float(mu)
        self._dragging_slider = False
        self._slider_track = None      # (left, bottom, w, h) px, last drawn frame
        self._plot_rect = None         # (left, bottom, side, side) px, or None if hidden

        self.cam_mode = 'follow'
        self.annotations = 'plot'
        self.show_hud = True

        self._status = {}
        self._info = None
        self._cmd = None
        self._backbone = None
        self._ghost = mujoco.MjData(model)
        self._tracked = None          # smoothed camera target

        # A close, slightly low three-quarter view: near enough that the wrap
        # is readable, not so near that the pole fills the frame.
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.distance = 1.1
        self.cam.azimuth = 135.0
        self.cam.elevation = -10.0
        self.cam.lookat[:] = (0.0, 0.0, 0.3)

        # Contact arrows off: they point out of the body at every contact and
        # a coiled robot has a dozen at once, which buries the shape they are
        # supposed to explain. `C` turns them on when they are the question.
        self._contacts = False
        self.vopt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 0
        self.vopt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 0
        self.scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1
        self.scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 1
        self.scn.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 1

        self._setup_projection_figure()

    # -- the 2-D cross-section panel: its own figure, drawn at a rect we pick,
    #    not one of the base viewer's fixed right-edge slots --------------
    def _setup_projection_figure(self):
        fig = mujoco.MjvFigure()
        mujoco.mjv_defaultFigure(fig)
        fig.flg_extend = 0          # fixed range -- see _draw_projection_panel
        fig.flg_legend = 1
        fig.flg_ticklabel[0] = 1
        fig.flg_ticklabel[1] = 1
        fig.gridsize[:] = (3, 3)
        fig.title = "Cross-section (Fig. 11): measured vs. commanded"
        fig.xlabel = "m"
        fig.figurergba[:] = (0.05, 0.06, 0.08, 0.78)
        fig.panergba[:] = (0.0, 0.0, 0.0, 0.0)
        fig.gridrgb[:] = (0.30, 0.30, 0.33)
        fig.textrgb[:] = (0.85, 0.85, 0.90)
        fig.legendrgba[:] = (0.05, 0.06, 0.08, 0.55)
        fig.linergb[0][:] = C_MEASURED
        fig.linergb[1][:] = C_TARGET
        fig.linename[0] = "measured"
        fig.linename[1] = "commanded"
        self._proj_fig = fig

    @staticmethod
    def _set_fig_line(fig, idx, xy):
        """Write a closed polygon into figure line `idx`, or clear it."""
        if xy is None or len(xy) == 0:
            fig.linepnt[idx] = 0
            return
        pts = np.asarray(xy, float)
        pts = np.vstack([pts, pts[0]])           # repeat the first point to close it
        n = min(len(pts), mujoco.mjMAXLINEPNT)
        line = fig.linedata[idx]
        line[0:2 * n:2] = pts[:n, 0]
        line[1:2 * n:2] = pts[:n, 1]
        fig.linepnt[idx] = n

    # -- keys on top of the viewer's own -----------------------------------
    def _key_callback(self, window, key, scancode, action, mods):
        if action == glfw.RELEASE:
            if key == glfw.KEY_B:
                i = CAM_MODES.index(self.cam_mode)
                self.cam_mode = CAM_MODES[(i + 1) % len(CAM_MODES)]
                return
            if key == glfw.KEY_N:
                i = ANNOTATION_LEVELS.index(self.annotations)
                self.annotations = ANNOTATION_LEVELS[(i + 1) % len(ANNOTATION_LEVELS)]
                return
            if key == glfw.KEY_Z:
                self.show_hud = not self.show_hud
                return
        super()._key_callback(window, key, scancode, action, mods)

    # -- mouse: the slider intercepts clicks/drags inside its own rectangle,
    #    everything else falls through to the base viewer's camera controls --
    def _cursor_px(self, window):
        """Cursor position in framebuffer pixels, bottom-up (matches MjrRect)."""
        x, y = glfw.get_cursor_pos(window)
        fb_h = glfw.get_framebuffer_size(window)[1]
        return self._scale * x, fb_h - self._scale * y

    def _inside(self, rect, x_px, y_px):
        if rect is None:
            return False
        left, bottom, w, h = rect
        return left <= x_px <= left + w and bottom <= y_px <= bottom + h

    def _set_mu_from_x(self, x_px):
        left, _bottom, w, _h = self._slider_track
        frac = 0.0 if w <= 0 else (x_px - left) / w
        frac = min(1.0, max(0.0, frac))
        raw = MU_MIN + frac * (MU_MAX - MU_MIN)
        self.mu = round(round(raw / MU_STEP) * MU_STEP, 4)

    def _mouse_button_callback(self, window, button, act, mods):
        if button == glfw.MOUSE_BUTTON_LEFT:
            x_px, y_px = self._cursor_px(window)
            if act == glfw.PRESS and self._inside(self._slider_track, x_px, y_px):
                self._dragging_slider = True
                self._set_mu_from_x(x_px)
                return
            if act == glfw.RELEASE and self._dragging_slider:
                self._dragging_slider = False
                return
        super()._mouse_button_callback(window, button, act, mods)

    def _cursor_pos_callback(self, window, xpos, ypos):
        if self._dragging_slider:
            self._set_mu_from_x(self._scale * xpos)
            return
        super()._cursor_pos_callback(window, xpos, ypos)

    # -- what the rig feeds in ---------------------------------------------
    def update_experiment(self, status, info=None, cmd=None, backbone=None):
        """Hand the viewer the latest experiment state.

        `status` is the rig's progress dict, `info` the controller's own
        report for this tick, `cmd` the commanded joint angles and `backbone`
        the node extractor. All optional -- whatever is missing is simply not
        drawn.
        """
        self._status = status or {}
        if info is not None:
            self._info = info
        if cmd is not None:
            self._cmd = np.asarray(cmd, float)
        if backbone is not None:
            self._backbone = backbone

        self._track_camera()

    def _track_camera(self):
        if self.cam_mode == 'free':
            return
        target = np.array(self.data.subtree_com[0], float)
        if self._tracked is None:
            self._tracked = target
        else:
            # Heavy smoothing: the centre of mass of a rolling helix wobbles
            # by a couple of centimetres every revolution, and a camera that
            # copies that wobble is unwatchable.
            self._tracked += 0.06 * (target - self._tracked)
        self.cam.lookat[:] = self._tracked
        if self.cam_mode == 'orbit':
            self.cam.azimuth = (self.cam.azimuth + 0.25) % 360.0

    # -- 3-D scene annotation: ghost body + height ruler only now; the
    #    cross-section rings moved to their own 2-D plot, see below --------
    def _segment(self, p0, p1, width, rgba, label=""):
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        d = p1 - p0
        length = np.linalg.norm(d)
        if length < 1e-9:
            return
        self.add_marker(type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                        pos=(p0 + p1) / 2, mat=_z_to(d),
                        size=(width, width, length / 2), rgba=rgba, label=label)

    def decorate(self):
        """Re-add every marker. Called once per rendered frame.

        The viewer clears its marker list after each render, so nothing here
        persists and everything is rebuilt from the current state.
        """
        if self.annotations == 'off':
            return
        self._height_ruler()
        if self.annotations == 'full':
            self._ghost_target()

    def _height_ruler(self):
        """Marks up the pole every 0.5 m, so progress is readable at a glance."""
        off = self.pole_radius + 0.018
        for k in range(1, int(self.pole_height * 2) + 1):
            z = 0.5 * k
            if z > self.pole_height:
                break
            self.add_marker(type=mujoco.mjtGeom.mjGEOM_BOX,
                            pos=(off, 0.0, z), mat=np.eye(3),
                            size=(0.012, 0.002, 0.002),
                            rgba=C_RULER, label=f"{z:.1f} m")

    def _ghost_target(self):
        """The shape the joints are being commanded into, drawn in place.

        Forward kinematics of the COMMANDED joint angles on the robot's actual
        base pose: where the body would be if the joints tracked perfectly.
        The gap between the ghost and the robot is the tracking error that the
        position servos are converting into grip force.
        """
        if self._cmd is None or self._backbone is None:
            return
        g = self._ghost
        mujoco.mj_resetData(self.model, g)
        g.qpos[0:7] = self.data.qpos[0:7]
        g.qpos[7:7 + len(self._cmd)] = self._cmd
        mujoco.mj_forward(self.model, g)
        pts = self._backbone.nodes(self.model, g)
        for p0, p1 in zip(pts, pts[1:]):
            self._segment(p0, p1, 0.006, C_GHOST)

    # -- the 2-D cross-section panel, drawn as raw MuJoCo figure ------------
    def _draw_projection_panel(self, width, height):
        if self.annotations == 'off' or self._hide_graph:
            self._plot_rect = None
            return

        side = int(np.clip(min(width, height) * PLOT_FRACTION, PLOT_MIN, PLOT_MAX))
        self._plot_rect = (MARGIN, MARGIN, side, side)

        fig = self._proj_fig
        fig.range[0] = [-PROJ_RANGE, PROJ_RANGE]
        fig.range[1] = [-PROJ_RANGE, PROJ_RANGE]
        info = self._info
        self._set_fig_line(fig, 0, info.get('cross_section') if info else None)
        self._set_fig_line(fig, 1, info.get('cross_section_target') if info else None)

        mujoco.mjr_figure(mujoco.MjrRect(*self._plot_rect), fig, self.ctx)

    # -- the friction slider, drawn and hit-tested in raw pixel space -------
    def _draw_friction_slider(self, width, height):
        plot_w = self._plot_rect[2] if self._plot_rect else 0
        left = MARGIN + (plot_w + 48 if self._plot_rect else 0)
        track_w = max(SLIDER_MIN_W, width - left - RIGHT_RESERVE)
        bottom = MARGIN + SLIDER_HINT_H
        self._slider_track = (left, bottom, track_w, SLIDER_TRACK_H)

        border = mujoco.MjrRect(left - 2, bottom - 2, track_w + 4, SLIDER_TRACK_H + 4)
        track = mujoco.MjrRect(left, bottom, track_w, SLIDER_TRACK_H)
        frac = min(1.0, max(0.0, (self.mu - MU_MIN) / (MU_MAX - MU_MIN)))
        fill = mujoco.MjrRect(left, bottom, max(1, int(track_w * frac)), SLIDER_TRACK_H)

        mujoco.mjr_rectangle(border, *C_SLIDER_BORDER)
        mujoco.mjr_rectangle(track, *C_SLIDER_TRACK)
        mujoco.mjr_rectangle(fill, *C_SLIDER_FILL)
        mujoco.mjr_label(track, mujoco.mjtFont.mjFONT_BIG,
                         f"friction (mu)   {self.mu:4.2f}",
                         0, 0, 0, 0.0, 1, 1, 1, self.ctx)

        # A hint line below, and a header above -- both in the coordinate
        # system mjr_text actually uses: normalized [0,1] of the FULL window,
        # not of the slider rect.
        mujoco.mjr_text(mujoco.mjtFont.mjFONT_NORMAL,
                        "drag to change friction while it climbs "
                        f"({MU_MIN:.1f} bare plastic .. {MU_MAX:.1f} not a real material)",
                        self.ctx, left / width, MARGIN / height, 0.6, 0.6, 0.65)
        mujoco.mjr_text(mujoco.mjtFont.mjFONT_NORMAL,
                        "Friction coefficient, robot <-> pole",
                        self.ctx, left / width,
                        (bottom + SLIDER_TRACK_H + SLIDER_HEADER_GAP) / height,
                        0.85, 0.85, 0.90)

    # -- heads-up panel ----------------------------------------------------
    def _create_overlay(self):
        super()._create_overlay()
        if not self.show_hud:
            return

        topright = mujoco.mjtGridPos.mjGRID_TOPRIGHT
        bottomright = mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT

        def put(gridpos, left, right):
            if gridpos not in self._overlay:
                self._overlay[gridpos] = ["", ""]
            self._overlay[gridpos][0] += left + "\n"
            self._overlay[gridpos][1] += str(right) + "\n"

        s = self._status

        def num(key, fmt):
            v = s.get(key)
            if v is None or (isinstance(v, float) and not np.isfinite(v)):
                return "-"
            if fmt.endswith('d'):
                v = int(v)
            return format(v, fmt)

        put(topright, "STAGE", s.get('stage', '-'))
        put(topright, "sim time", num('t', '7.2f') + " s")
        put(topright, "height", num('z', '7.3f') + " m")
        put(topright, "peak", num('peak_z', '7.3f') + " m")
        put(topright, "climb rate", num('rate_cm_s', '7.2f') + " cm/s")
        put(topright, "", "")
        put(topright, "coil radius", num('coil_r', '7.4f') + " m")
        put(topright, "grip squeeze", num('squeeze_mm', '7.2f') + " mm")
        put(topright, "axis tilt", num('axis_tilt', '7.1f') + " deg")
        put(topright, "", "")
        put(topright, "pole contacts", num('n_pole', '7d'))
        put(topright, "self contacts", num('n_self', '7d'))
        mu = s.get('mu_p95')
        put(topright, "friction needed",
            (f"{mu:7.2f} of {self.mu:.2f}" if mu is not None else '-'))
        put(topright, "motors saturated",
            (f"{int(s['saturated']):7d} of 18" if 'saturated' in s else '-'))

        put(bottomright, "camera [B]", self.cam_mode)
        put(bottomright, "annotations [N]", self.annotations)
        put(bottomright, "panel [Z]", "on")
        put(bottomright, "cyan / orange", "measured / commanded cross-section")
        put(bottomright, "faint body", "the form the joints are commanded into")

    # -- render: a full custom pipeline ------------------------------------
    #
    # The base MujocoViewer.render() draws the 3-D scene, the text overlays,
    # and (optionally) its own figures, then swaps buffers -- all inside one
    # method with no hook to inject extra drawing before the swap. Since the
    # slider and the cross-section panel are raw mjr_* calls that MUST land
    # before that swap, this reimplements the same sequence (pause/step/speed
    # handling included, copied from the base) and adds two calls in the
    # middle: _draw_projection_panel and _draw_friction_slider.
    def render(self):
        if self.render_mode == 'offscreen':
            raise NotImplementedError("Use 'read_pixels()' for 'offscreen' mode.")
        if not self.is_alive:
            raise Exception("GLFW window does not exist but you tried to render.")
        if glfw.window_should_close(self.window):
            self.close()
            return

        self.decorate()

        def update():
            self._create_overlay()
            render_start = time.time()

            width, height = glfw.get_framebuffer_size(self.window)
            self.viewport.width, self.viewport.height = width, height

            with self._gui_lock:
                mujoco.mjv_updateScene(
                    self.model, self.data, self.vopt, self.pert, self.cam,
                    mujoco.mjtCatBit.mjCAT_ALL.value, self.scn)
                for marker in self._markers:
                    self._add_marker_to_scene(marker)
                mujoco.mjr_render(self.viewport, self.scn, self.ctx)

                for gridpos, [t1, t2] in self._overlay.items():
                    menu_positions = [mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                      mujoco.mjtGridPos.mjGRID_BOTTOMLEFT]
                    if gridpos in menu_positions and self._hide_menus:
                        continue
                    mujoco.mjr_overlay(mujoco.mjtFontScale.mjFONTSCALE_150, gridpos,
                                       self.viewport, t1, t2, self.ctx)

                self._draw_projection_panel(width, height)
                self._draw_friction_slider(width, height)

                glfw.swap_buffers(self.window)
            glfw.poll_events()
            self._time_per_render = (0.9 * self._time_per_render
                                     + 0.1 * (time.time() - render_start))
            self._overlay.clear()

        if self._paused:
            while self._paused:
                update()
                if glfw.window_should_close(self.window):
                    self.close()
                    break
                if self._advance_by_one_step:
                    self._advance_by_one_step = False
                    break
        else:
            self._loop_count += self.model.opt.timestep / (
                self._time_per_render * self._run_speed)
            if self._render_every_frame:
                self._loop_count = 1
            while self._loop_count > 0:
                update()
                self._loop_count -= 1

        self._markers[:] = []
        self.apply_perturbations()
