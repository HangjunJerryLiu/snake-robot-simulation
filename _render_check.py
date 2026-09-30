"""Temporary: render snapshots of a run, to check by eye what the numbers say."""
import os
import struct
import sys
import zlib

import mujoco

from climb_rig import Config, run

r, start = float(sys.argv[1]), sys.argv[2]
times = [float(x) for x in sys.argv[3].split(',')]
out = sys.argv[4]
os.makedirs(out, exist_ok=True)


def png(path, img):
    h, w, _ = img.shape
    raw = b''.join(b'\x00' + img[y].tobytes() for y in range(h))

    def chunk(t, d):
        return (struct.pack('>I', len(d)) + t + d
                + struct.pack('>I', zlib.crc32(t + d) & 0xffffffff))
    with open(path, 'wb') as f:
        f.write(b'\x89PNG\r\n\x1a\n'
                + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0))
                + chunk(b'IDAT', zlib.compress(raw, 6)) + chunk(b'IEND', b''))


state = dict(rend=None, want=list(times))
_step = mujoco.mj_step


def step(m, d):
    _step(m, d)
    if state['want'] and d.time >= state['want'][0]:
        t = state['want'].pop(0)
        if state['rend'] is None:
            state['rend'] = mujoco.Renderer(m, 480, 640)
        for az in (45, 135):
            cam = mujoco.MjvCamera()
            c = d.subtree_com[0]; cam.lookat[:] = (c[0] / 2, c[1] / 2, max(0.15, c[2]))
            cam.distance, cam.azimuth, cam.elevation = 1.6, az, -35
            state['rend'].update_scene(d, cam)
            png(os.path.join(out, f"r{int(round(r*1000))}mm_{start}_t{int(t):03d}_az{az}.png"),
                state['rend'].render())


mujoco.mj_step = step
cfg = Config(pole_radius=r, start_wrapped=(start == 'wrapped'), start_radius=r + 0.045,
             max_duration=max(times) + 0.1, out_dir=os.path.join(out, 'runs'), log_hz=10)
s = run(cfg, on_message=lambda m: print(' ', m))
print(r, start, 'peak', s['peak_com_z_m'], 'caught', s['caught_pole'])
