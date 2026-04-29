# %%
# lets build the gradient that should push the robot out of collision.
from simplearm.geom import SquareGrid
import numpy as np
import plotly.express as px
from simplearm import kinematics as kin
from simplearm.jacobians import joint_jacobians, sphere_jacobians_from_joint_jacobians
from copy import deepcopy
from simplearm.viz import RobotViewer
from simplearm.robot import RobotInfo

np.random.seed(0)
world = SquareGrid.from_random_perlin(length=3.0, number_of_vox=64, res=4, layers=2)
sdf = world.derive_sdf_from_voxels()
sdf_grad_x, sdf_grad_y = sdf.gradient()
robot = RobotInfo.from_linklengths([0.5, 0.5])


# Taken from costs.py
def chomp_obstacle_cost_and_grad(dist, eps=0.1):
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


# Plot the cost function vs signed distance to see how it behaves. Note the nonzero cost for slightly positive distances, which creates a "buffer" around obstacles.
px.line(
    x=np.arange(-0.5, 0.5, 0.01),
    y=chomp_obstacle_cost_and_grad(np.arange(-0.5, 0.5, 0.01), eps=0.1)[0],
    title="CHOMP Obstacle Cost vs Signed Distance",
    labels={"x": "Signed Distance", "y": "Cost"},
)
# %%


q_t = np.array([np.pi, np.pi / 2])

q_history = [q_t]
q_i = deepcopy(q_t)
for i in range(50):
    T = kin.forward_kinematic(q_i, robot.linklengths)
    spheres_w = kin.world_spheres_from_frames(T, robot.spheres)
    dist_to_world = sdf[spheres_w.xy] - spheres_w.r

    # Get jacobians for the sphere centers
    J_joint = joint_jacobians(T)
    J_spheres = sphere_jacobians_from_joint_jacobians(
        T, J_joint, spheres_w
    )  # (num_spheres, 3, num_joints)

    # Get the sdf gradient at the sphere centers
    sdf_grad_x_spheres = sdf_grad_x[spheres_w.xy]
    sdf_grad_y_spheres = sdf_grad_y[spheres_w.xy]
    sdf_grad_spheres = np.stack(
        [sdf_grad_x_spheres, sdf_grad_y_spheres], axis=-1
    )  # (num_spheres, 2)

    # Smooth collision cost with eps, and its analytical derivative w.r.t. signed distance.
    _, dcost_ddist = chomp_obstacle_cost_and_grad(dist_to_world, eps=0.1)

    # Chain rule: (dc/ddist) * (ddist/dq)
    J_spheres_positiononly = J_spheres[:, :2, :]  # (num_spheres, 2, num_joints)
    joint_space_gradient = np.einsum(
        "s,sd,sdn->n", dcost_ddist, sdf_grad_spheres, J_spheres_positiononly
    )

    if i % 10 == 0:
        max_cost = np.max(chomp_obstacle_cost_and_grad(dist_to_world, eps=0.1)[0])
        print(f"Max collision cost at iteration {i}: {max_cost:.4f}")

    # Gradient descent step on the smooth collision cost
    q_i -= 0.1 * joint_space_gradient
    q_history.append(deepcopy(q_i))

q_diff = np.array(q_history)
RobotViewer(q_diff, robot, voxels=world).plot()


# The CHOMP model assumes the cost is a sum over the spheres of the robot. Sound like Gauss-Newton could work...
