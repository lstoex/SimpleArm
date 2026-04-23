"""Basic Robot Definition"""

from simplearm.geom import Spheres
from dataclasses import dataclass, field
import numpy as np
from numpy.typing import ArrayLike
from simplearm.utils import make_spheres

@dataclass
class Inertias:
    mass: np.ndarray
    inertia: np.ndarray

    def __post_init__(self):
        """Validate shape and dtype consistency for inertia arrays."""
        assert self.mass.shape == self.inertia.shape, "Mass and inertia must have the same shape."
        assert self.mass.dtype == self.inertia.dtype == float, "Mass and inertia must be of float type."

@dataclass
class RobotInfo:
    linklengths: np.ndarray
    spheres: Spheres
    inertias: Inertias
    ignore_pairs: set[tuple[int, int]] = field(default_factory=set)

    @property
    def n_dof(self) -> int:
        """Return the number of degrees of freedom of the robot."""
        return len(self.linklengths)

    def __repr__(self):
        str = "A Robot with the following properties:\n"
        str += f"  Link Lengths: {self.linklengths}\n"
        str += f"  Number of Spheres: {self.spheres.x.shape[0]}\n"
        str += f"  Mass per Link: {self.inertias.mass}\n"
        str += f"  Inertia per Link: {self.inertias.inertia}\n"
        str += f"  Number of ignore sphere pairs: {len(self.ignore_pairs)}"
        return str

    @classmethod
    def from_linklengths(cls, linklengths: ArrayLike, sphere_rad=0.05, sphere_overlap=0.75, mass=1.0, inertia=0.1) -> "RobotInfo":
        """Construct robot metadata from link lengths and default physical parameters."""
        linklengths = np.asarray(linklengths, dtype=float)
        n_dof = len(linklengths)
        spheres = make_spheres(n_dof, linklengths, sphere_rad, sphere_overlap)
        inertias = Inertias(mass=np.full(n_dof, mass), inertia=np.full(n_dof, inertia))
        # By default, all spheres are active for external collision.
        n_spheres = spheres.x.shape[0]
        ignore_pairs: list[tuple[int, int]] = []

        f_idx = spheres.frame_idx
        for idx in np.unique(f_idx):
            same_link = np.where(f_idx == idx)[0]
            for i in same_link:
                for j in same_link:
                    if i < j:
                        ignore_pairs.append((int(i), int(j)))

        # Add pairs of spheres that follow each other in the order.
        for i in range(n_spheres - 1):
            ignore_pairs.append((i, i + 1))

        # Add pairs where the first sphere is two spheres ahead.
        for i in range(n_spheres - 2):
            ignore_pairs.append((i, i + 2))

        return cls(
            linklengths=linklengths,
            spheres=spheres,
            inertias=inertias,
            ignore_pairs=set(ignore_pairs),
        )