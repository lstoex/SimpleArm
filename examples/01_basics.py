# %%
import numpy as np

from simplearm.dynamics import mass_matrix_from_com_jacobians
from simplearm.geom import Obstacles, get_min_signed_distance, pairwise_sphere_dist
from simplearm.jacobians import (
    com_jacobians_from_joint_jacobians,
    joint_jacobians,
    sphere_jacobians_from_joint_jacobians,
)
from simplearm.kinematics import forward_kinematic, world_spheres_from_frames
from simplearm.robot import RobotInfo

linklengths = [0.5, 0.5, 0.25, 0.25]  # this fully describes the shape of the robot.
robot = RobotInfo.from_linklengths(
    linklengths
)  # This computes the robot metadata, including spheres and inertias, from the link lengths.
print(robot)
# %%
q = np.random.uniform(
    low=-np.pi, high=np.pi, size=(robot.n_dof,)
)  # A random configuration of the robot.

# Compute the joint-centered link frames + the TCP frames (end of last link)
frames = forward_kinematic(q, robot.linklengths)
print(
    "Frames are shaped", frames.shape
)  # There are n_dof + 1 frames, each with a 3x3 homogeneous transform. The last frame is the end effector frame.
# %%
# Compute the joint jacobians for each link frame.
jacobians = joint_jacobians(frames)
print(
    "Jacobians are shaped", jacobians.shape
)  # There are n_dof link frames, each with a 3 x n_dof spatial jacobian.
# %%
# compute the sphere positions in world coordinates.
spheres_world = world_spheres_from_frames(frames, robot.spheres)
print(spheres_world)

# %%
sphere_jacobians = sphere_jacobians_from_joint_jacobians(
    frames, jacobians, spheres_world
)
print(
    "Sphere Jacobians are shaped", sphere_jacobians.shape
)  # There are S spheres, each with a 3 x n_dof jacobian.
# %%
com_jacobians = com_jacobians_from_joint_jacobians(frames, jacobians, robot.linklengths)
print(
    "COM Jacobians are shaped", com_jacobians.shape
)  # There are n_dof link COMs, each with a 3 x n_dof jacobian.
# %%
pairwise_dists, (pair_i, pair_j) = pairwise_sphere_dist(
    spheres_world, robot.ignore_pairs
)
print(
    "Pairwise distances are shaped", pairwise_dists.shape
)  # Of course we only compute pairs (i,j) where i < j, so there are (S-1)*S/2 - len(ignore_pairs) pairs.
# %%
# We can also define obstacles in the world and compute distances to them.

obstacles_xy = np.array([[0.5, 0.5], [0.75, 0.25]])
obstacles_r = np.array([0.1, 0.1])
obstacles = Obstacles(x=obstacles_xy[:, 0], y=obstacles_xy[:, 1], r=obstacles_r)
point = np.array([0.6, 0.6])
signed_dist_to_obstacles = get_min_signed_distance(point, obstacles)
print("Signed distance from point to obstacles is", signed_dist_to_obstacles)
# or for the robot spheres:
spheres_signed_dist_to_obstacles = get_min_signed_distance(spheres_world.xy, obstacles)
print(
    "Min Signed distance from spheres to obstacles is",
    spheres_signed_dist_to_obstacles.min(),
)
# The obstacle class implement the __getitem__ method so we can do the shorthand
spheres_signed_dist_to_obstacles_shorthand = obstacles[spheres_world.xy]
assert np.allclose(
    spheres_signed_dist_to_obstacles, spheres_signed_dist_to_obstacles_shorthand
)
# %%
# We can use trajectories as well, since the code is fully vectorized for a batch dimension on the configuration, frames, jacobians, and spheres. Since the robot stays the same, we dont vectorize over radii etc.
q_traj = np.random.uniform(
    low=-np.pi, high=np.pi, size=(10, robot.n_dof)
)  # A random trajectory of 10 configurations.
frames_traj = forward_kinematic(q_traj, robot.linklengths)
spheres_traj = world_spheres_from_frames(frames_traj, robot.spheres)
jacobians_traj = joint_jacobians(frames_traj)
sphere_jacobians_traj = sphere_jacobians_from_joint_jacobians(
    frames_traj, jacobians_traj, spheres_traj
)
com_jacobians_traj = com_jacobians_from_joint_jacobians(
    frames_traj, jacobians_traj, robot.linklengths
)
pairwise_dists_traj, (pair_i_traj, pair_j_traj) = pairwise_sphere_dist(
    spheres_traj, robot.ignore_pairs
)
# %%
# We can also compute the mass matrix from the COM jacobians and the link inertias.
M = mass_matrix_from_com_jacobians(com_jacobians, robot.inertias)
print("Mass matrix is shaped", M.shape)
# Of course, it is also vectorized for trajectories:
M_traj = mass_matrix_from_com_jacobians(com_jacobians_traj, robot.inertias)
print("Mass matrix for trajectory is shaped", M_traj.shape)
