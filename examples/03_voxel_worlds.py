# %%
from simplearm.geom import SquareGrid
import numpy as np
import plotly.express as px
# %%
# Generate a random perlin noise grid and derive an sdf from it
g = SquareGrid.from_random_perlin(length=3., number_of_vox=128, res=4, layers=2) #resolution controls how "wild" the noise is, higher res means more variation in the noise. Layers control how many independent perlin noise layers are generated and combined.
sdf = g.derive_sdf_from_voxels()
#we have to be careful to align visualization and data indexing logic, so we transpose the data for visualization and set origin to lower left
px.imshow(np.stack([g.data.T, sdf.data.T], axis=0), animation_frame=0, title="Perlin Noise (0) and Derived SDF (1)", origin="lower")
# %%
# Generate a sequence of worlds with increasing resolution perlin noise
worlds = []
res = 1
for i in range(100):
    if i % 20 == 0:
        res *= 2 if res <= 32 else 32
    print(f"Generating world with resolution {res}x{res}...")
    worlds.append(SquareGrid.from_random_perlin(length=1., number_of_vox=64, res=res, layers=4).data.T)
worlds = np.stack(worlds, axis=0)
px.imshow(worlds, animation_frame=0, title="Perlin Noise with Increasing Resolution", origin="lower")

# %%
# Visualize a moving robot in a perlin noise world
from simplearm.viz import RobotViewer
from simplearm.robot import RobotInfo
robot = RobotInfo.from_linklengths([0.5, 0.5])
q = np.linspace(np.array([-np.pi/2, -np.pi/2]), np.array([np.pi, np.pi/2]), num=64)
viewer = RobotViewer(q, robot, voxels=g)
viewer.plot()
fig = viewer.fig
# %%
