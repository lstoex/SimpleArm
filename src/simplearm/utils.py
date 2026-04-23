import numpy as np
from simplearm.geom import Spheres


def __find_bin_and_normalize(
    position: float, linklengths: np.ndarray
) -> tuple[int, float]:
    """
    Find the bin a position is in and normalize it relative to the start of the respective bin.

    Args:
        position (float): The position along the line.
        linklengths (list of float): The lengths of the bins along the line.

    Returns:
        bin_index (int): The index of the bin the position is in.
        normalized_position (float): The position normalized relative to the start of the respective bin.
    """
    cumulative_lengths = np.concatenate(([0.0], np.cumsum(linklengths)))
    cumulative_lengths[-1] += 1e-6
    bin_index = np.searchsorted(cumulative_lengths, position, side="right")
    bin_start = cumulative_lengths[bin_index - 1]
    normalized_position = position - bin_start
    return int(bin_index) - 1, normalized_position


def make_spheres(
    n_dof: int, linklengths: np.ndarray, sphere_rad: float, overlap: float = 0.75
) -> Spheres:
    """Create collision spheres along all links with fixed overlap."""
    spacing = 2 * sphere_rad * np.sqrt(1 - overlap**2)
    assert len(linklengths) == n_dof, "Link lengths must be of length n_dof"
    total_len = np.sum(linklengths)
    n_elems = int(np.ceil(total_len / spacing))
    sphere_x_total = np.linspace(0.0, total_len, n_elems)

    cumulative_lengths = np.concatenate(([0.0], np.cumsum(linklengths)))
    cumulative_lengths[-1] += 1e-6
    f_idx = np.searchsorted(cumulative_lengths, sphere_x_total, side="right") - 1
    sphere_x = sphere_x_total - cumulative_lengths[f_idx]

    sphere_y = np.zeros_like(sphere_x)
    spheres_r = np.ones_like(sphere_x) * sphere_rad
    return Spheres(frame_idx=f_idx, x=sphere_x, y=sphere_y, r=spheres_r)
