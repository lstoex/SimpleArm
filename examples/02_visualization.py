#%%
from simplearm.viz import RobotViewer
from simplearm.robot import RobotInfo
import numpy as np
import scipy.interpolate as interp
linklengths = [0.5, 0.5, 0.25, 0.25]
robot = RobotInfo.from_linklengths(linklengths)
print(robot)
#%%
#Plot a single configuration of the robot with the spheres.
q = np.random.uniform(
    low=-np.pi, high=np.pi, size=(robot.n_dof,)
)  # A random configuration of the robot.
viz = RobotViewer(q,robot)
viz.plot()
#%%
#Plot an animation of a trajectory of the robot with the spheres.
#lets sample a smooth trajectory by sampling random waypoints and interpolating them with a cubic spline.
n_waypoints = 3
waypoints = np.random.uniform(low=-np.pi, high=np.pi, size=(n_waypoints, robot.n_dof))
t = np.linspace(0, 1, n_waypoints)
cs = interp.CubicSpline(t, waypoints, axis=0)
t_traj = np.linspace(0, 1, 100)
q_traj = cs(t_traj)
viz = RobotViewer(q_traj, robot)
viz.plot()
# %%
