# %%
from copy import deepcopy
from simplearm.viz import RobotViewer
from simplearm.robot import RobotInfo
import numpy as np
import scipy.interpolate as interp
from simplearm.geom import Obstacles
from simplearm.costs import chomp_smoothness_cost_and_grad

np.random.seed(42)
linklengths = [0.5, 0.5]
robot = RobotInfo.from_linklengths(linklengths)

# %%
# Sample a random trajectory and smoothen it with the CHOMP smoothness cost to see how it behaves.
n_waypoints = 6
waypoints = np.random.uniform(low=-np.pi, high=np.pi, size=(n_waypoints, robot.n_dof))
t = np.linspace(0, 1, n_waypoints)
cs = interp.CubicSpline(t, waypoints, axis=0)
t_traj = np.linspace(0, 1, 32)
q_traj = cs(t_traj)
RobotViewer(q_traj, robot).plot()
# %%
import plotly.express as px

# plot joint values q_traj (32, 2) vs time t_traj (32,) to see how the trajectory looks in joint space
import pandas as pd

df = pd.DataFrame(q_traj, columns=[f"Joint {i}" for i in range(robot.n_dof)])
px.line(df, title="Joint Trajectory")
# %%
cost, grad = chomp_smoothness_cost_and_grad(q_traj, dt=1.0)
print(f"Initial smoothness cost: {cost}")
# %%
q_traj = cs(t_traj)  # reset to original trajectory before optimization
q_traj_opt = deepcopy(q_traj)
for i in range(1000):
    cost, grad = chomp_smoothness_cost_and_grad(q_traj_opt, dt=1.0)
    q_traj_opt -= 0.1 * grad  # gradient descent step
    if i % 10 == 0:
        print(f"Iteration {i}, Smoothness Cost: {cost}")

# RobotViewer(q_traj, robot).plot()

# Ideal trajectory is a straight line in joint space between the start and end configurations, so we can compare our optimized trajectory to that as well.
q_ideal = np.linspace(q_traj[0], q_traj[-1], num=q_traj.shape[0])
fig = px.line(title="Trajectory Before and After Optimization")
colors = px.colors.qualitative.Plotly
for i in range(robot.n_dof):
    fig.add_scatter(
        x=t_traj,
        y=q_traj_opt[:, i],
        mode="lines",
        name=f"Joint {i} After",
        line=dict(dash="dash", color=colors[i]),
    )
    fig.add_scatter(
        x=t_traj,
        y=q_traj[:, i],
        mode="lines",
        name=f"Joint {i} Before",
        line=dict(color=colors[i]),
    )
    fig.add_scatter(
        x=t_traj,
        y=q_ideal[:, i],
        mode="lines",
        name=f"Joint {i} Ideal",
        line=dict(color=colors[i], width=4, dash="dot"),
    )
fig.show()

# Again, the cost is a sum over squared terms.
# %%
# we can work out the math on paper and see that the jacbian is a bidiagonal matrix. Lets ignore dt for now to keep it simple.
from scipy import sparse

T = q_traj.shape[0] - 1  # now we dont refer to the frames but rather time...
D = q_traj.shape[1]
J = np.zeros((T, D, T - 1, D))


def assemble_inner_jacobian(T, D):
    # 1. Create the base difference matrix for T terms and T+1 points
    # Shape: (T, T+1)
    diagonals = [-np.ones(T), np.ones(T)]
    # Offset 0 is -1, Offset 1 is +1
    base_diff = sparse.diags(diagonals, [0, 1], shape=(T, T + 1))

    # 2. Slice the matrix to remove columns corresponding to q_0 and q_T
    # We remove the first column (index 0) and the last column (index T)
    # New Shape: (T, T-1)
    inner_diff = base_diff.tocsr()[
        :, 1:-1
    ]  # Keep columns from index 1 to T-1 (inclusive)

    # 3. Expand to D dimensions using Kronecker product
    # Final Shape: (T*D, (T-1)*D)
    jac_sparse = sparse.kron(inner_diff, sparse.eye(D))
    return jac_sparse.tocsr()


J_sparse = assemble_inner_jacobian(T, D)
cost, grad = chomp_smoothness_cost_and_grad(q_traj, dt=1.0)
g = grad[1:-1].flatten()  # Flatten the inner gradients to match the shape of J_sparse
update = sparse.linalg.spsolve(J_sparse.T @ J_sparse, g).reshape(T - 1, D)
q_traj_opt_gn = deepcopy(q_traj)
q_traj_opt_gn[1:-1] -= update

fig = px.line(title="Trajectory Before and After Gauss-Newton Optimization")
colors = px.colors.qualitative.Plotly
for i in range(robot.n_dof):
    fig.add_scatter(
        x=t_traj,
        y=q_traj_opt_gn[:, i],
        mode="lines",
        name=f"Joint {i} After GN",
        line=dict(dash="dash", color=colors[i]),
    )
    fig.add_scatter(
        x=t_traj,
        y=q_traj[:, i],
        mode="lines",
        name=f"Joint {i} Before",
        line=dict(color=colors[i]),
    )
    fig.add_scatter(
        x=t_traj,
        y=q_ideal[:, i],
        mode="lines",
        name=f"Joint {i} Ideal",
        line=dict(color=colors[i], width=4, dash="dot"),
    )
fig.show()

# Convergence in one step! This is due to the fact that the cost is exactly quadratic in the inner points, so the Gauss-Newton step finds the optimal update in one iteration. CHOMP finds a middle ground by using the natural gradient...
# %%
