# %%
from simplearm.geom import SquareGrid
import numpy as np
import plotly.express as px

# %%
# Generate a random perlin noise grid and derive an sdf from it
g = SquareGrid.from_random_perlin(
    length=3.0, number_of_vox=64, res=4, layers=2
)  # resolution controls how "wild" the noise is, higher res means more variation in the noise. Layers control how many independent perlin noise layers are generated and combined.
sdf = g.derive_sdf_from_voxels()
# we have to be careful to align visualization and data indexing logic, so we transpose the data for visualization and set origin to lower left
fig = px.imshow(
    np.stack([g.data.T, sdf.data.T], axis=0),
    animation_frame=0,
    title="Perlin Noise (0) and Derived SDF (1)",
    origin="lower",
)
fig.show()
# %%
# make a quiver plot of the sdf gradient to see how the collision repulsion field looks.
sdf_grad_x, sdf_grad_y = sdf.gradient()
import plotly.figure_factory as ff

x0 = sdf_grad_x.limits[0][0]
x1 = sdf_grad_x.limits[1][0]
y0 = sdf_grad_x.limits[0][1]
y1 = sdf_grad_x.limits[1][1]
x, y = np.meshgrid(
    np.arange(x0, x1, sdf_grad_x.voxel_size),
    np.arange(y0, y1, sdf_grad_x.voxel_size),
    indexing="ij",
)
data_x = sdf_grad_x.data
data_y = (
    sdf_grad_y.data
)  # I know, this is cumbersome.... Care to improve it? Feel free :)
data = np.stack([data_x, data_y], axis=-1)
# zero out gradients where sdf is >0 to focus on the collisions
data = np.where(sdf.data[..., None] > 0, 0, data)
sdf_grad_x.data = data[..., 0]
sdf_grad_y.data = data[..., 1]
u = data[..., 0]
v = data[..., 1]

figq = ff.create_quiver(x, y, u, v, scale=0.03)
fig = px.imshow(
    sdf.data.T,
    origin="lower",
    x=np.arange(x0, x1, sdf_grad_x.voxel_size),
    y=np.arange(y0, y1, sdf_grad_y.voxel_size),
)
fig.add_trace(figq.data[0])
fig.update_layout(title="SDF Gradient Field (quiver) over SDF (heatmap)")
fig.show()
# %%
# Generate a sequence of worlds with increasing resolution perlin noise
worlds = []
res = 1
for i in range(100):
    if i % 20 == 0:
        res *= 2 if res <= 32 else 32
    print(f"Generating world with resolution {res}x{res}...")
    worlds.append(
        SquareGrid.from_random_perlin(
            length=1.0, number_of_vox=64, res=res, layers=4
        ).data.T
    )
worlds = np.stack(worlds, axis=0)
px.imshow(
    worlds,
    animation_frame=0,
    title="Perlin Noise with Increasing Resolution",
    origin="lower",
)

# %%
# Visualize a moving robot in a perlin noise world
from simplearm.viz import RobotViewer
from simplearm.robot import RobotInfo

robot = RobotInfo.from_linklengths([0.5, 0.5])
q = np.linspace(
    np.array([-np.pi / 2, -np.pi / 2]), np.array([np.pi, np.pi / 2]), num=64
)
viewer = RobotViewer(q, robot, voxels=g)
viewer.plot()
fig = viewer.fig
