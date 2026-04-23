import numpy as np
from simplearm.geom import Spheres, SpheresInWorld
def __get_frame_positions(q: np.ndarray, linklenghts: np.ndarray) -> np.ndarray:
    """Compute the forward kinematics of a 2D arm with n joints."""

    q_ = np.atleast_2d(q)
    q_cum = np.cumsum(q_, axis=-1)
    x = np.cos(q_cum) * linklenghts
    y = np.sin(q_cum) * linklenghts
    x = np.concatenate((np.zeros((x.shape[0], 1)), np.cumsum(x, axis=-1)), axis=-1)
    y = np.concatenate((np.zeros((y.shape[0], 1)), np.cumsum(y, axis=-1)), axis=-1)
    return np.stack((x, y), axis=-1)


def forward_kinematic(q: np.ndarray, link_lenghts: np.ndarray) -> np.ndarray:
    """Return homogeneous transforms for each link frame and end effector."""
    q_ = np.atleast_2d(q)
    t = __get_frame_positions(q_, link_lenghts)
    n_dof = q_.shape[-1]
    n_frames = q_.shape[-2]

    frames = np.repeat(np.eye(3)[None, None, :, :], n_frames, axis=0)
    frames = np.repeat(frames, n_dof + 1, axis=1)
    q_cum = np.cumsum(q_, axis=-1)

    frames[:, :-1, 0, :2] = np.stack([np.cos(q_cum), -np.sin(q_cum)], axis=-1)
    frames[:, :-1, 1, :2] = np.stack([np.sin(q_cum), np.cos(q_cum)], axis=-1)
    frames[:, :, :2, 2] = t

    frames[:, -1, 0, :2] = np.stack([np.cos(q_cum[:, -1]), -np.sin(q_cum[:, -1])], axis=-1)
    frames[:, -1, 1, :2] = np.stack([np.sin(q_cum[:, -1]), np.cos(q_cum[:, -1])], axis=-1)

    return frames.squeeze() if q.ndim == 1 else frames

def world_spheres_from_frames(frames: np.ndarray, spheres: Spheres) -> SpheresInWorld:
    """Transform local-link spheres into world coordinates using frame transforms."""
    frames_ = frames[..., spheres.frame_idx, :, :]
    xy_spheres = np.stack([spheres.x, spheres.y], axis=1)
    rotations = frames_[..., :, :2, :2]
    translations = frames_[..., :, :2, 2]
    xy_spheres = np.einsum("...sij,sj->...si", rotations, xy_spheres) + translations
    r = spheres.r
    return SpheresInWorld(
        x=xy_spheres[..., :, 0], y=xy_spheres[..., :, 1], r=r, frame_idx=spheres.frame_idx
    )