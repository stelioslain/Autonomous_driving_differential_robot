# Testing the maze's spawn and walls

import pybullet as p
import pybullet_data
import time
from maze_builder import build_maze

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)

p.loadURDF("plane.urdf")
build_maze()

while True:
    p.stepSimulation()
    time.sleep(1/240)
