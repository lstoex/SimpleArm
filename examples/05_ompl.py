"""Lets use OMPL to solve a motion planning problem using sampling. Is is evident that the Python binding of OMPL are beyond terrible. It should be straightforward to build a custom RRT"""
# %%
import numpy as np
import plotly.express as px
np.random.seed(0)
from simplearm.geom import SquareGrid, pairwise_sphere_dist
from simplearm.robot import RobotInfo

from simplearm.viz import RobotViewer
from simplearm import kinematics as kin

from ompl import base as ob
from ompl import geometric as og

# %%
# Build a simple voxel world and its signed distance field.
g = SquareGrid.from_random_perlin(length=3.0, number_of_vox=64, res=4, layers=2)
sdf = g.derive_sdf_from_voxels()


# Robot model and planning problem in joint space.
robot = RobotInfo.from_linklengths([0.5, 0.5])

def get_dists(q):
    T = kin.forward_kinematic(q, robot.linklengths)
    spheres_w = kin.world_spheres_from_frames(T, robot.spheres)
    dists = sdf[spheres_w.xy] - spheres_w.r - 0.05  # add a small safety buffer around the robot
    pairwise_dists, _ = pairwise_sphere_dist(spheres_w, robot.ignore_pairs)
    return dists, pairwise_dists

def state_is_valid(state):
    q = np.array([state[0], state[1]], dtype=float)
    dists, pairwise_dists = get_dists(q)
    return bool(np.all(dists > 0)) and bool(np.all(pairwise_dists > 0))

#sample random start and goal states in the joint space that are feasible
def sample_start_and_goal(n_samples=10000):
    q_start = np.random.uniform(low=-np.pi, high=np.pi, size=(n_samples, robot.n_dof))
    q_goal = np.random.uniform(low=-np.pi, high=np.pi, size=(n_samples, robot.n_dof))
    #find all feasible start and goal states
    dists, pairwise_dists = get_dists(q_start)
    feas = (dists > 0).all(axis=1) & (pairwise_dists > 0).all(axis=1)
    q_start_feas = q_start[feas]
    dists, pairwise_dists = get_dists(q_goal)
    feas = (dists > 0).all(axis=1) & (pairwise_dists > 0).all(axis=1)
    q_goal_feas = q_goal[feas]
    if len(q_start_feas) == 0 or len(q_goal_feas) == 0:
        raise ValueError("No feasible start or goal states found. Try increasing the number of samples.")
    #take some pair
    i,j = np.random.randint(len(q_start_feas)), np.random.randint(len(q_goal_feas))
    return q_start_feas[i], q_goal_feas[j]

q_start, q_goal = sample_start_and_goal()
print("Sampled successfully a start and goal state")
RobotViewer(np.array([q_start, q_goal]), robot, voxels=g).plot()
# %%
space = ob.RealVectorStateSpace(2)
bounds = ob.RealVectorBounds(2)
bounds.setLow(0, -np.pi)
bounds.setHigh(0, np.pi)
bounds.setLow(1, -np.pi)
bounds.setHigh(1, np.pi)
space.setBounds(bounds)

ss = og.SimpleSetup(space)
ss.setStateValidityChecker(state_is_valid)


start = ss.getStateSpace().allocState()  # ty:ignore[unresolved-attribute]
start[0] = float(q_start[0])
start[1] = float(q_start[1])

goal = ss.getStateSpace().allocState()  # ty:ignore[unresolved-attribute]
goal[0] = float(q_goal[0])
goal[1] = float(q_goal[1])

print(f"Start is valid: {state_is_valid(start)}, Goal is valid: {state_is_valid(goal)}")

ss.setStartAndGoalStates(start, goal)

# planner = og.RRTConnect(ss.getSpaceInformation())
# planner = og.RRTstar(ss.getSpaceInformation())
planner = og.AORRTC(ss.getSpaceInformation())
ss.setPlanner(planner)

solved = ss.solve(5.0)
if solved:
    path = ss.getSolutionPath()
    path.interpolate(16)
    q_path = np.array([[s[0], s[1]] for s in path.getStates()], dtype=float)  # ty:ignore[not-subscriptable]
    RobotViewer(q_path, robot, voxels=g).plot()

else:
    print("No solution found.")

# %%
