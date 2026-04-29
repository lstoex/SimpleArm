# %%
import numpy as np
import scipy.interpolate as interp

from simplearm.geom import Obstacles
from simplearm.robot import RobotInfo
from simplearm.viz import RobotViewer

linklengths = [0.5, 0.5]
robot = RobotInfo.from_linklengths(linklengths)
print(robot)
# %%
# Plot a single configuration of the robot with the spheres.
obstacles_xy = np.array([[0.5, 0.5], [0.75, 0.25]])
obstacles_r = np.array([0.1, 0.1])
obstacles = Obstacles(x=obstacles_xy[:, 0], y=obstacles_xy[:, 1], r=obstacles_r)
q = np.random.uniform(
    low=-np.pi, high=np.pi, size=(robot.n_dof,)
)  # A random configuration of the robot.
viz = RobotViewer(q, robot, obstacles=obstacles)
viz.plot()
# %%
# Plot an animation of a trajectory of the robot with the spheres.
# lets sample a smooth trajectory by sampling random waypoints and interpolating them with a cubic spline.
n_waypoints = 3
waypoints = np.random.uniform(low=-np.pi, high=np.pi, size=(n_waypoints, robot.n_dof))
t = np.linspace(0, 1, n_waypoints)
cs = interp.CubicSpline(t, waypoints, axis=0)
t_traj = np.linspace(0, 1, 100)
q_traj = cs(t_traj)
viz = RobotViewer(q_traj, robot, obstacles=obstacles)
viz.plot()
# %%
