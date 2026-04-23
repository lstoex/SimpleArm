import numpy as np
from simplearm.robot import Inertias
def mass_matrix_from_com_jacobians(
    com_jacobians: np.ndarray, inertias: Inertias
) -> np.ndarray:

    n_batch_dims = com_jacobians.ndim - 3
    augmented_jacobians = np.concatenate(
        [
            com_jacobians[..., :, :2, :] * inertias.mass[(np.newaxis,) * n_batch_dims + (slice(None),) + (None,)*2],
            com_jacobians[..., :, 2:, :] * inertias.inertia[(np.newaxis,) * n_batch_dims + (slice(None),) + (None,)*2],
        ],
        axis=-2,
    )  # shape: (..., n_frames, 3, n_dof)
    return np.einsum(
        "...ijm,...ijn->...mn", augmented_jacobians, augmented_jacobians
    )