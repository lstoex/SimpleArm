from dataclasses import dataclass
import numpy as np
from scipy.ndimage import map_coordinates
from .perlin import perlin_noise_2d

from viser.transforms import SE2


@dataclass
class SquareGrid:
    """2D square grids including smart construction and indexing logic. We enforce a visual coordinate system that sits at the outer edge of the first voxel. This then entails a grid coordinate system that sits in the center of the first voxel, which is what the data is indexed by and most visualizer will use."""

    data: np.ndarray  # (nx, ny) array of values at the grid points
    length: float
    origin: SE2  # origin of the edge of the first grid cell in the world frame

    def __post_init__(self):
        # assert self.data.ndim == 2, "Grid data must be 2D"
        assert self.length > 0, "Grid length must be positive"
        assert self.origin is not None, "Grid origin must be provided"
        assert self.data.shape[0] == self.data.shape[1], "Grid data must be square"

    @classmethod
    def from_random_perlin(
        cls, length: float, number_of_vox: int, res: int, layers: int = 1
    ) -> "SquareGrid":
        """Create a Grid with random perlin noise data."""
        shape = (number_of_vox, number_of_vox)
        res_ = (res, res)
        data = [perlin_noise_2d(shape=shape, res=res_) for _ in range(layers)]
        data = np.stack(data, axis=0)  # (layers, nx, ny)
        data = np.sum(data, axis=0)  # (nx, ny)

        data = ~np.logical_or(data < -0.5, data < 0.5) #TODO: make the threshold a parameter
        return cls.from_zero_centered(limits=(-length / 2, length / 2), data=data)
    
    def derive_sdf_from_voxels(self) -> "SquareGrid":
        """Derive a signed distance field from the grid data, treating nonzero values as obstacles."""
        sdf_data = voxel2sdf(voxels=self.data.astype(bool), voxel_size=self.voxel_size)
        return SquareGrid(data=sdf_data, length=self.length, origin=self.origin) 

    def gradient(self) -> tuple["SquareGrid", "SquareGrid"]:
        """Compute the gradient of the grid data and return it as a 2 new SquareGrids, one for each component of the gradient."""
        g_x, g_y = np.gradient(self.data, self.voxel_size)
        g = np.stack([g_x, g_y], axis=-1)  # (nx, ny, 2)
        #normalize the gradient to have unit length, and set zero gradients to zero to avoid NaNs
        g_norm = np.linalg.norm(g, axis=-1, keepdims=True)
        g_norm[g_norm == 0] = 1.0
        g = g / g_norm
        return (SquareGrid(data=g[..., 0], length=self.length, origin=self.origin), SquareGrid(data=g[..., 1], length=self.length, origin=self.origin))

    @classmethod
    def from_zero_centered(
        cls, limits: tuple[float, float], data: np.ndarray
    ) -> "SquareGrid":
        """Create a SquareGrid from limits and data, assuming the grid is zero-centered. The origin is set so that the grid is symmetric around the world origin."""
        length = limits[1] - limits[0]
        origin = SE2.from_translation(np.array([-length / 2, -length / 2]))
        return cls(data=data, length=length, origin=origin)

    @property
    def T_g_v(self) -> SE2:
        """Transformation from grid coordinates to edge of visual block coordinates. This is a pure translation by half a voxel size in each direction, since the grid coordinates refer to the center of the first voxel, while the visual block coordinates refer to the edge of the first voxel."""
        vs = self.voxel_size
        return SE2.from_translation(-np.array([vs / 2, vs / 2]))

    @property
    def voxel_size(self) -> float:
        """Size of a single voxel in meters."""
        vz = self.length / np.array(self.data.shape)[0]
        return vz

    @property
    def gridlimits(self) -> np.ndarray:
        """Local limits of the grid in grid(!) coordinates."""
        vs = self.voxel_size
        d = self.length
        return np.array(
            [
                [-vs / 2, -vs / 2],
                [
                    d - vs / 2,
                    d - vs / 2,
                ],
            ]
        )
    
    @property
    def limits(self) -> np.ndarray:
        """Real physical limits of the grid in world coordinates."""
        lower = self.origin.apply(np.array([0, 0]))
        upper = self.origin.apply(np.array([self.length, self.length]))
        return np.array([lower, upper])

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the grid in number of voxels along each axis."""
        return self.data.shape

    @property
    def T_g_w(self) -> SE2:
        """Transformation from world coordinates to grid coordinates."""
        return self.T_g_v @ self.origin.inverse()

    @property
    def T_v(self) -> SE2:
        """Transformation from visual block coordinates to world coordinates. Just an alias for the origin"""
        return self.origin

    def __getitem__(
        self, p: np.ndarray, p_isin_world: bool = True, order: int = 1
    ) -> np.ndarray:
        """
        Index into the grid at given WORLD positions continuously. Out of bounds positions will be clamped to the nearest valid value.

        :param positions: Points at which to sample the SDF. Shape [..., 2].
        """
        # ensure that positions is converted to a [batch, 2] array, even when a single point is given
        # or multiple batch dimensions are given. This allows for fast lookups. Batch dimensions are preserved in the output.
        p = np.atleast_2d(p).reshape(-1, 2)
        # world -> voxel -> grid, or voxel -> grid
        p_grid = self.T_g_w.apply(p) if p_isin_world else self.T_g_v.apply(p)
        grid_idx = coords_to_indices(
            coords=p_grid,
            limits=self.gridlimits,
            lengths=np.asarray(self.length),
            shape=self.shape,
        )
        dists = map_coordinates(
            input=self.data, coordinates=grid_idx.T, order=order, mode="nearest"
        )  # type: ignore
        return dists.reshape(p.shape[:-1])  # reshape to original batch shape   

@dataclass
class Spheres:
    frame_idx: np.ndarray
    x: np.ndarray
    y: np.ndarray
    r: np.ndarray

    @property
    def xy(self) -> np.ndarray:
        """Return the sphere centers as an (S, 2) array."""
        return np.stack([self.x, self.y], axis=-1)

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

    def __getitem__(self, p: np.ndarray) -> np.ndarray:
        """Allow the Obstacles object to be called as a function to compute the signed distance from points to the obstacles."""
        return get_min_signed_distance(p, self)

    @property
    def xy(self) -> np.ndarray:
        """Return the obstacle centers as an (N, 2) array."""
        return np.stack([self.x, self.y], axis=-1)


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

    active_spheres = spheres.xy
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
    center_dist = np.linalg.norm(p[..., None, :] - obstacles.xy, axis=-1)  # (..., M)
    signed = center_dist - obstacles.r  # (..., M)
    return np.min(signed, axis=-1)

def index_with_interpolation(
    p: np.ndarray,
    arr: np.ndarray,
    limits: np.ndarray,
    shape: tuple[int, int],
    lengths: np.ndarray,
    order: int = 1,
) -> np.ndarray:
    """Index into a 2D array with interpolation. The coordinates `p` are continuous coordinates between the outer edges of the grid, and are converted to grid coordinates before indexing."""
    p_grid = np.atleast_2d(p).reshape(-1, 2)
    grid_idx = coords_to_indices(
        coords=p_grid, limits=limits, lengths=lengths, shape=shape
    )
    dists = map_coordinates(
        input=arr, coordinates=grid_idx.T, order=order, mode="nearest"
    )  # type: ignore
    return dists.reshape(p.shape[:-1])  # reshape to original batch shape


def coords_to_indices(
    coords: np.ndarray,
    limits: np.ndarray,
    lengths: np.ndarray,
    shape: tuple[int, ...],
) -> np.ndarray:
    """Convert continuous coordinates (between outer edges of the grid) to grid coordinates (between -0.5 and (N-1)+0.5)."""
    return (coords - limits[0, :]) * (np.asarray(shape)) / (lengths) + -0.5

def voxel2sdf(
    voxels, voxel_size: float, add_boundary=True
) -> np.ndarray:
    """
    Calculate the signed distance field from an 2D/3D image of the world.
    Obstacles are 1/True, free space is 0/False.
    The distance image is of the same shape as the input image and has positive values outside objects and negative
    values inside objects see 'CHOMP - signed distance field' (10.1177/0278364913488805)
    The voxel_size is used to scale the distance field correctly (the shape of a single pixel / voxel)
    Args:
        voxels: binary image of the world
        world_info: WorldInfo object
        add_boundary: if True, the boundary is filled with obstacles
    """
    from scipy.ndimage import distance_transform_edt

    n_voxels = np.array(voxels.shape)

    if not add_boundary:
        # Main function
        #                                         # EDT wants objects as 0, rest as 1
        dist_img = distance_transform_edt(-voxels.astype(int) + 1, sampling=voxel_size)
        dist_img_complement = distance_transform_edt(
            voxels.astype(int), sampling=voxel_size
        )
        dist_img[voxels] = -dist_img_complement[voxels]  # Add interior information

    else:
        # Additional branch, to include boundary filled with obstacles
        obstacle_img_wb = np.ones(n_voxels + 2, dtype=bool)
        inner_image_idx = tuple(
            map(slice, np.ones(voxels.ndim, dtype=int), (n_voxels + 1))
        )
        obstacle_img_wb[inner_image_idx] = voxels

        dist_img = voxel2sdf(
            voxels=np.array(obstacle_img_wb), voxel_size=voxel_size, add_boundary=False
        )
        dist_img = dist_img[inner_image_idx]
    return dist_img
