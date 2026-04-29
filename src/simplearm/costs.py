import numpy as np

from .geom import Obstacles, SquareGrid
from .kinematics import forward_kinematic, world_spheres_from_frames


def chomp_obstacle_cost_and_grad(dist, eps=0.1):
    """Computes the CHOMP obstacle cost and its gradient w.r.t. the signed distance."""
    # CHOMP paper obstacle cost c(d)
    cost = np.where(
        dist < 0,
        -dist + 0.5 * eps,
        np.where(dist <= eps, 0.5 * (dist - eps) ** 2 / eps, 0.0),
    )
    # analytical derivative dc/dd
    dcost_ddist = np.where(
        dist < 0,
        -1.0,
        np.where(dist <= eps, (dist - eps) / eps, 0.0),
    )
    return cost, dcost_ddist


def chomp_smoothness_cost_and_grad(q_traj: np.ndarray, dt: float):
    """Computes the CHOMP smoothness cost and its gradient for a given trajectory."""
    # cost = 0.5 * sum_t (q_t+1 - q_t)^2 / dt^2
    q_dot = np.diff(q_traj, axis=0) / dt
    cost = 0.5 * np.sum(q_dot**2)
    # when the trajectory has T+1 timesteps, so the index t runs from 0 to T, we differentiate only w.r.t. the inner T-1 configurations, since the first and last are fixed endpoints.
    grad = np.zeros_like(q_traj)
    # grad for inner points is (-q_t+1 + 2*q_t - q_t-1) / dt^2
    grad[1:-1] = -(q_traj[2:] - 2 * q_traj[1:-1] + q_traj[:-2]) / dt**2
    return cost, grad


def is_feasible(
    q_traj: np.ndarray, sdf_or_obstacles: Obstacles | SquareGrid, robot_info
):
    """Returns a boolean indicating whether the trajectory is collision free, and an array of the indices of the trajectory that are in collision if not."""
    T = forward_kinematic(q_traj, robot_info.linklengths)
    spheres_w = world_spheres_from_frames(T, robot_info.spheres)
    dists = sdf_or_obstacles[spheres_w.xy] - spheres_w.r
    return np.all(dists > 0), np.argwhere(dists <= 0)
