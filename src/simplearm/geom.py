from dataclasses import dataclass
import numpy as np


@dataclass
class Spheres:
    frame_idx: np.ndarray
    x: np.ndarray
    y: np.ndarray
    r: np.ndarray

    def __repr__(self):
        batch_dims = self.x.shape[:-1]
        n_spheres = self.x.shape[-1]
        if batch_dims:
            return f"A set of {n_spheres} spheres with batch dimensions {batch_dims} that live in their respective local link frames."
        else:
            return f"A set of {n_spheres} spheres that live in their respective local link frames."


@dataclass
class SpheresInWorld(Spheres):
    def __repr__(self):
        batch_dims = self.x.shape[:-1]
        n_spheres = self.x.shape[-1]
        if batch_dims:
            return f"A set of {n_spheres} spheres with batch dimensions {batch_dims} that live in world coordinates."
        else:
            return f"A set of {n_spheres} spheres that live in world coordinates."


@dataclass
class Obstacles:
    x: np.ndarray
    y: np.ndarray
    r: np.ndarray

    def __repr__(self):
        num_obstacles = len(self.x)
        return f"A set of {num_obstacles} circular obstacles in the world."


def pairwise_sphere_dist(
    spheres: SpheresInWorld, ignore_pairs: set
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Compute signed pairwise sphere distances while skipping ignored index pairs. Also returns the corresponding index pairs."""
    # We only care about the unordered pairs.
    # Ignore all pairs on the same link and adjacent spheres.
    n_spheres = spheres.x.shape[-1]
    i_idx, j_idx = np.triu_indices(n_spheres, k=1)

    forbidden = np.zeros((n_spheres, n_spheres), dtype=bool)
    if ignore_pairs:
        ij = np.array(list(ignore_pairs), dtype=np.int32)
        forbidden[ij[:, 0], ij[:, 1]] = True
        forbidden[ij[:, 1], ij[:, 0]] = True

    mask = ~forbidden[i_idx, j_idx]
    i_idx = i_idx[mask]
    j_idx = j_idx[mask]

    active_spheres = np.stack([spheres.x, spheres.y], axis=-1)
    active_radii = spheres.r
    p1 = active_spheres[..., i_idx, :]
    p2 = active_spheres[..., j_idx, :]
    r1 = active_radii[i_idx]
    r2 = active_radii[j_idx]

    center_dist = np.linalg.norm(p1 - p2, axis=-1)
    return center_dist - (r1 + r2), (i_idx, j_idx)


def get_min_signed_distance(p: np.ndarray, obstacles: Obstacles) -> np.ndarray:
    """Compute the minimum signed distance from points to a set of circular obstacles.

    Supports arbitrary batch dimensions in `p`, as long as the last dimension is 2.
    """
    xy = np.stack([obstacles.x, obstacles.y], axis=-1)  # (M, 2)
    center_dist = np.linalg.norm(p[..., None, :] - xy, axis=-1)  # (..., M)
    signed = center_dist - obstacles.r  # (..., M)
    return np.min(signed, axis=-1)
