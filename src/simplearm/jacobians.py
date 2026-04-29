from copy import deepcopy

import numpy as np

from simplearm.geom import SpheresInWorld


def joint_jacobians(frames: np.ndarray) -> np.ndarray:
    """Compute planar spatial Jacobians for each link frame origin."""
    n_frames = (
        frames.shape[-3] - 1
    )  # last frame is end effector, which doesn't have a link
    positions = frames[..., :-1, :2, 2]
    offsets = positions[..., :, None, :] - positions[..., None, :, :]
    mask = np.tri(n_frames, n_frames, k=0, dtype=float)

    batch_dims = frames.shape[:-3]
    J = np.zeros((*batch_dims, n_frames, 3, n_frames), dtype=float)
    J[..., 0, :] = -offsets[..., :, :, 1] * mask[None]
    J[..., 1, :] = offsets[..., :, :, 0] * mask[None]
    J[..., 2, :] = mask
    return J


def sphere_jacobians_from_joint_jacobians(
    frames: np.ndarray, joint_jacs: np.ndarray, spheres_in_world: SpheresInWorld
) -> np.ndarray:
    """Project joint Jacobians to sphere centers via frame-to-sphere offsets."""
    f_idx = spheres_in_world.frame_idx
    spheres_xy_global = spheres_in_world.xy
    bases = frames[..., f_idx, :, :][..., :2, 2]
    offsets = spheres_xy_global - bases

    J = deepcopy(joint_jacs[..., f_idx, :, :])
    J[..., 0, :] = J[..., 0, :] - offsets[..., 1][..., None] * J[..., 2, :]
    J[..., 1, :] = J[..., 1, :] + offsets[..., 0][..., None] * J[..., 2, :]
    return J


def com_jacobians_from_joint_jacobians(
    frames: np.ndarray, joint_jacs: np.ndarray, linklengths: np.ndarray
) -> np.ndarray:
    """Project joint Jacobians from frame origins to link centers of mass."""
    offsets_local = np.stack([linklengths / 2.0, np.zeros_like(linklengths)], axis=-1)
    offsets = np.einsum("...fij,fj->...fi", frames[..., :-1, :2, :2], offsets_local)

    J = deepcopy(joint_jacs)
    J[..., :, 0, :] = J[..., :, 0, :] - offsets[..., :, 1][..., None] * J[..., :, 2, :]
    J[..., :, 1, :] = J[..., :, 1, :] + offsets[..., :, 0][..., None] * J[..., :, 2, :]
    return J
