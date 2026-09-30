"""
Collision sanity checks for the snake-robot model.

MuJoCo collides the robot's links using each mesh's CONVEX HULL, and it
never collides a body with its own parent (the "filterparent" rule). Both
are sensible for simulation, and both can let parts pass visibly through
each other without any contact being reported:

  * a parent/child pair is never tested, so if a joint could rotate further
    than the physical parts allow, the two links would simply blend;
  * a geometry change applied after compiling (the old pole-radius bug)
    leaves stale bounding volumes, so real overlaps go undetected.

This module measures overlap directly from the REAL triangle meshes,
independently of MuJoCo's contact pipeline, so it can catch both.

How: each mesh is voxelised once, in its own frame, into a solid grid
(VOXEL = 0.3 mm) by counting surface crossings along each grid column, and
every solid voxel stores how deep it lies (cube erosion, so the stored depth
is a guaranteed lower bound on the true Euclidean depth). Two parts
interpenetrate where a solid voxel of one maps inside the other; the
overlap is reported as how deep the deepest such voxel sits inside the
other part, taking the worse of the two directions. The pole is an exact cylinder, so robot-pole penetration is
computed analytically from each part's solid voxels.

test_collisions.py is the testbench built on this.
"""
import mujoco
import numpy as np

VOXEL = 0.3e-3                 # m
_JITTER = np.array([0.1234567, 0.3456789])   # column-centre offset, avoids edge hits


class MeshVolume:
    """Solid voxel model of one mesh, in the mesh's own frame."""

    def __init__(self, verts, faces, h=VOXEL):
        self.h = h
        lo = verts.min(axis=0) - 2 * h
        hi = verts.max(axis=0) + 2 * h
        self.lo = lo
        n = np.ceil((hi - lo) / h).astype(int) + 1
        self.shape = tuple(n)
        occ = self._fill(verts, faces, n)
        self.depth = self._erode_depth(occ)          # uint8, 0 = outside
        idx = np.argwhere(self.depth > 0)
        self.solid_pts = lo + (idx + 0.5) * h         # voxel centres, mesh frame
        self.solid_depth = self.depth[tuple(idx.T)].astype(np.float32) * h

    def _fill(self, V, F, n):
        """Parity fill: a voxel is solid if the column ray below it crosses
        the surface an odd number of times."""
        h, lo = self.h, self.lo
        T = V[F]                                      # (nf, 3, 3)
        x, y, z = T[..., 0], T[..., 1], T[..., 2]
        # column range covered by each triangle's xy bounding box
        i0 = np.ceil((x.min(1) - lo[0]) / h - _JITTER[0]).astype(int)
        i1 = np.floor((x.max(1) - lo[0]) / h - _JITTER[0]).astype(int)
        j0 = np.ceil((y.min(1) - lo[1]) / h - _JITTER[1]).astype(int)
        j1 = np.floor((y.max(1) - lo[1]) / h - _JITTER[1]).astype(int)
        ni, nj = np.maximum(i1 - i0 + 1, 0), np.maximum(j1 - j0 + 1, 0)
        cnt = ni * nj
        keep = cnt > 0
        tri = np.repeat(np.nonzero(keep)[0], cnt[keep])
        start = np.repeat(np.cumsum(cnt[keep]) - cnt[keep], cnt[keep])
        local = np.arange(len(tri)) - start
        ci = i0[tri] + local // nj[tri]
        cj = j0[tri] + local % nj[tri]
        px = lo[0] + (ci + _JITTER[0]) * h
        py = lo[1] + (cj + _JITTER[1]) * h
        # barycentric coordinates of the column centre in the projected triangle
        ax, ay = x[tri, 0], y[tri, 0]
        bx, by = x[tri, 1], y[tri, 1]
        cx, cy = x[tri, 2], y[tri, 2]
        den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        good = np.abs(den) > 1e-20
        den = np.where(good, den, 1.0)
        l1 = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / den
        l2 = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / den
        l3 = 1 - l1 - l2
        hit = good & (l1 >= 0) & (l2 >= 0) & (l3 >= 0)
        zc = l1 * z[tri, 0] + l2 * z[tri, 1] + l3 * z[tri, 2]
        ci, cj, zc = ci[hit], cj[hit], zc[hit]
        k = np.clip(np.ceil((zc - lo[2]) / h - 0.5).astype(int), 0, n[2])
        toggles = np.zeros((n[0], n[1], n[2] + 1), np.int32)
        np.add.at(toggles, (ci, cj, k), 1)
        return (np.cumsum(toggles, axis=2)[..., :n[2]] % 2).astype(bool)

    @staticmethod
    def _erode_depth(occ):
        """Voxel depth by repeated 3x3x3 erosion: a voxel surviving k erosions
        has a cube of half-width k fully inside, so its Euclidean distance to
        the surface is at least k voxels."""
        depth = occ.astype(np.uint8)
        cur = occ.copy()
        k = 1
        while cur.any() and k < 255:
            e = cur.copy()
            for ax in range(3):
                m = e.copy()
                sl_lo = [slice(None)] * 3
                sl_hi = [slice(None)] * 3
                sl_lo[ax], sl_hi[ax] = slice(1, None), slice(None, -1)
                m[tuple(sl_lo)] &= e[tuple(sl_hi)]
                m[tuple(sl_hi)] &= e[tuple(sl_lo)]
                edge = [slice(None)] * 3
                edge[ax] = 0
                m[tuple(edge)] = False
                edge[ax] = -1
                m[tuple(edge)] = False
                e = m
            cur = e
            k += 1
            depth[cur] = k
        return depth

    def depth_at(self, pts):
        """Depth (m) of mesh-frame points inside this volume; 0 outside."""
        idx = np.floor((pts - self.lo) / self.h).astype(int)
        ok = np.all((idx >= 0) & (idx < np.array(self.shape)), axis=1)
        out = np.zeros(len(pts), np.float32)
        out[ok] = self.depth[tuple(idx[ok].T)].astype(np.float32) * self.h
        return out


class Checker:
    """Real-mesh overlap measurements on one compiled model."""

    def __init__(self, model):
        self.m = model
        self._vol = {}
        self.pipe = model.geom('pipe').id
        self.floor = model.geom('floor').id
        self.robot = [g for g in range(model.ngeom)
                      if model.geom_bodyid[g] != 0 and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]

    def volume(self, g):
        mid = int(self.m.geom_dataid[g])
        if mid not in self._vol:
            va, nv = self.m.mesh_vertadr[mid], self.m.mesh_vertnum[mid]
            fa, nf = self.m.mesh_faceadr[mid], self.m.mesh_facenum[mid]
            self._vol[mid] = MeshVolume(self.m.mesh_vert[va:va + nv].astype(float),
                                        self.m.mesh_face[fa:fa + nf])
        return self._vol[mid]

    @staticmethod
    def _pose(data, g):
        return data.geom_xpos[g].copy(), data.geom_xmat[g].reshape(3, 3).copy()

    def _reach(self, data, g_from, g_into):
        """How deep (m) any solid part of g_from sits inside g_into."""
        A, B = self.volume(g_from), self.volume(g_into)
        pa, Ra = self._pose(data, g_from)
        pb, Rb = self._pose(data, g_into)
        in_b = (A.solid_pts @ Ra.T + pa - pb) @ Rb    # into B's mesh frame
        return float(B.depth_at(in_b).max(initial=0.0))

    def pair_penetration(self, data, g1, g2):
        """How far the real meshes of g1 and g2 pass into each other (m): the
        deepest point of either one inside the other. A plate stabbed 10 mm
        into a block reads 10 mm, however thin the plate."""
        return max(self._reach(data, g1, g2), self._reach(data, g2, g1))

    def pole_penetration(self, data, g):
        """How far (m) the real mesh of robot geom g reaches inside the pole."""
        A = self.volume(g)
        p, R = self._pose(data, g)
        w = A.solid_pts @ R.T + p
        r = self.m.geom_size[self.pipe, 0]
        half = self.m.geom_size[self.pipe, 1]
        c = data.geom_xpos[self.pipe]
        within = np.abs(w[:, 2] - c[2]) <= half
        if not within.any():
            return 0.0
        rad = np.hypot(w[within, 0] - c[0], w[within, 1] - c[1])
        return float(max(0.0, r - rad.min()))

    def floor_penetration(self, data, g):
        A = self.volume(g)
        p, R = self._pose(data, g)
        z = (A.solid_pts @ R.T + p)[:, 2]
        return float(max(0.0, -z.min()))

    def adjacent_pairs(self):
        """(joint id, parent geom, child geom) for every hinge between links."""
        m, out = self.m, []
        for j in range(m.njnt):
            if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
                continue
            child = m.jnt_bodyid[j]
            parent = m.body_parentid[child]
            gc = [g for g in self.robot if m.geom_bodyid[g] == child]
            gp = [g for g in self.robot if m.geom_bodyid[g] == parent]
            if gc and gp:
                out.append((j, gp[0], gc[0]))
        return out

    def report(self, data):
        """Worst real-mesh penetration (m) for each pair class, right now."""
        m = self.m
        res = {'robot-pole': 0.0, 'robot-floor': 0.0,
               'robot-robot adjacent': 0.0, 'robot-robot other': 0.0}
        for g in self.robot:
            res['robot-pole'] = max(res['robot-pole'], self.pole_penetration(data, g))
            res['robot-floor'] = max(res['robot-floor'], self.floor_penetration(data, g))
        for i, a in enumerate(self.robot):
            for b in self.robot[i + 1:]:
                ba, bb = m.geom_bodyid[a], m.geom_bodyid[b]
                adj = m.body_parentid[ba] == bb or m.body_parentid[bb] == ba
                key = 'robot-robot adjacent' if adj else 'robot-robot other'
                # links more than two bodies apart and far apart in space: skip
                if not adj and np.linalg.norm(data.geom_xpos[a] - data.geom_xpos[b]) > 0.12:
                    continue
                res[key] = max(res[key], self.pair_penetration(data, a, b))
        return res
