"""
Backbone extraction for the 9-motor snake in MuJoCo.

The adaptive-helical-rolling controller in helical_rolling.py works on the
link-edge points 1p_i of eq. (3) of Takemori et al. (2023): n_link+1 points
running head -> tail, with the joints in between and the two outer ends of
the head and tail links at the extremities. This module produces exactly that
list from an MjModel/MjData pair, and measures the link lengths that go with
it straight off the model instead of hard-coding them.

This robot is NOT the paper's uniform-link robot. Its joints alternate at
24 mm and 56 mm spacing (motor link, spacer link, motor link, ...), so the
paper's single link length l becomes a per-link array. helical_rolling.py
takes that array and uses the local value everywhere the paper writes l.
"""

import numpy as np


def _body_mesh_x_extent(model, body_id):
    """(min, max) x of a body's mesh geometry, in the body's own frame."""
    lo, hi = np.inf, -np.inf
    for g in range(model.ngeom):
        if model.geom_bodyid[g] != body_id or model.geom_dataid[g] < 0:
            continue
        mid = model.geom_dataid[g]
        v0, nv = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        V = np.array(model.mesh_vert[v0:v0 + nv], dtype=float)
        q = model.geom_quat[g]
        R = np.empty(9)
        import mujoco
        mujoco.mju_quat2Mat(R, q)
        V = V @ R.reshape(3, 3).T + model.geom_pos[g]
        lo, hi = min(lo, V[:, 0].min()), max(hi, V[:, 0].max())
    return float(lo), float(hi)


class Backbone:
    """Node specification plus the measured link lengths, head -> tail."""

    def __init__(self, model, head='head', tail='tail', n_joint=18):
        import mujoco

        self.head_id = model.body(head).id
        self.tail_id = model.body(tail).id

        # Joints, head -> tail: J18 is the head-most joint, J1 the tail-most.
        self.joint_ids = [model.joint(f'J{j}').id for j in range(n_joint, 0, -1)]

        # The two outer nodes: the far end of the head link and of the tail
        # link, taken from the actual mesh extents along each body's x-axis.
        self.head_offset = np.array([_body_mesh_x_extent(model, self.head_id)[1], 0.0, 0.0])
        self.tail_offset = np.array([_body_mesh_x_extent(model, self.tail_id)[0], 0.0, 0.0])

        # Link lengths, measured at the model's reference pose.
        probe = mujoco.MjData(model)
        mujoco.mj_resetData(model, probe)
        probe.qpos[3] = 1.0
        mujoco.mj_forward(model, probe)
        P = self.nodes(model, probe)
        self.link_lengths = np.linalg.norm(np.diff(P, axis=0), axis=1)
        self.n_link = len(self.link_lengths)
        self.n_joint = n_joint

    def nodes(self, model, data):
        """Link-edge points 1p_i, head -> tail, in world coordinates."""
        out = np.empty((len(self.joint_ids) + 2, 3))
        R = data.xmat[self.head_id].reshape(3, 3)
        out[0] = data.xpos[self.head_id] + R @ self.head_offset
        for i, jid in enumerate(self.joint_ids):
            out[i + 1] = data.xanchor[jid]
        R = data.xmat[self.tail_id].reshape(3, 3)
        out[-1] = data.xpos[self.tail_id] + R @ self.tail_offset
        return out

    def __repr__(self):
        ll = ", ".join(f"{x*1000:.0f}" for x in self.link_lengths)
        return (f"Backbone(n_link={self.n_link}, n_joint={self.n_joint}, "
                f"total={self.link_lengths.sum():.3f} m, links[mm]=[{ll}])")
