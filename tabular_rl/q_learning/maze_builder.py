import pybullet as p
import numpy as np
import math

WALL_THICKNESS = 0.15
WALL_HEIGHT = 0.5

def create_wall(start, end):
    x1, y1 = start
    x2, y2 = end

    length = math.hypot(x2 - x1, y2 - y1)
    angle = math.atan2(y2 - y1, x2 - x1)

    collision = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[length / 2, WALL_THICKNESS / 2, WALL_HEIGHT / 2]
    )

    visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[length / 2, WALL_THICKNESS / 2, WALL_HEIGHT / 2],
        rgbaColor=[0, 0, 0, 1]
    )

    p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=[(x1 + x2) / 2, (y1 + y2) / 2, WALL_HEIGHT / 2],
        baseOrientation=p.getQuaternionFromEuler([0, 0, angle])
    )

def build_maze():
    walls = [
        # Outer boundaries
        # The spaces in which the robot can move are all 2x2
        ((0.0, 0.0), (12.0, 0.0)),
        ((12.0, 0.0), (12.0, 20.0)),
        ((12.0, 20.0), (10.0, 20.0)),
        ((8.0, 20.0), (0.0, 20.0)),
        ((0.0, 20.0), (0.0, 8.0)),
        ((0.0, 8.0), (0.0, 0.0)),

        # Internal walls (from image)
        # First wall chunk (below entrance)
        ((0.0, 2.0), (4.0, 2.0)),
        ((4.0, 2.0), (4.0, 18.0)),
        ((4.0, 18.0), (6.0, 18.0)),
        ((4.0, 12.0), (2.0, 12.0)),
        ((4.0, 10.0), (8.0, 10.0)),
        ((8.0, 10.0), (8.0, 12.0)),
        ((8.0, 10.0), (8.0, 6.0)),
        ((8.0, 6.0), (6.0, 6.0)),
        ((6.0, 6.0), (6.0, 8.0)),
        
        # Second wall chunk (above entrance and top left corner)
        ((0.0, 8.0), (2.0, 8.0)),
        ((2.0, 8.0), (2.0, 10.0)),
        ((2.0, 8.0), (2.0, 4.0)),
        ((0.0, 14.0), (2.0, 14.0)),
        ((2.0, 14.0), (2.0, 18.0)),
        
        # Third wall chunk (left of the exit and bottom right corner)
        ((8.0, 20.0), (8.0, 16.0)),
        ((8.0, 16.0), (6.0, 16.0)),
        ((6.0, 16.0), (6.0, 12.0)),
        ((6.0, 14.0), (10.0, 14.0)),
        ((10.0, 18.0), (10.0, 2.0)),
        ((10.0, 4.0), (6.0, 4.0)),
        ((6.0, 4.0), (6.0, 2.0)),
        ((8.0, 0.0), (8.0, 2.0))
    ]
    
    for wall in walls:
        create_wall(*wall)
        
def maze_formation():
        array = [[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, -1],
                [-1, -1, -1, -1, -1, 0, -1, 0, -1, 0, -1, 0, -1],
                [-1, 0, 0, 0, -1, 0, -1, 0, 0, 0, -1, 0, -1],
                [-1, 0, -1, 0, -1, 0, -1, -1, -1, -1, -1, 0, -1],
                [-1, 0, -1, 0, -1, 0, 0, 0, 0, 0, -1, 0, -1],
                [-1, 0, -1, 0, -1, 0, -1, -1, -1, 0, -1, 0, -1],
                [-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1],
                [-1, -1, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1],
                [-1, 0, -1, 0, -1, 0, 0, 0, -1, 0, -1, 0, -1],
                [-1, 0, -1, 0, -1, -1, -1, -1, -1, 0, -1, 0, -1],
                [-1, 0, 0, 0, -1, 0, 0, 0, -1, 0, -1, 0, -1],
                [-1, 0, -1, -1, -1, 0, -1, 0, -1, 0, -1, 0, -1],
                [-1, 0, 0, 0, -1, 0, -1, 0, 0, 0, -1, 0, -1],
                [-1, -1, -1, 0, -1, 0, -1, -1, -1, -1, -1, 0, -1],
                [-1, 0, -1, 0, -1, 0, -1, 0, 0, 0, -1, 0, -1],
                [-1, 0, -1, 0, -1, 0, -1, -1, -1, 0, -1, 0, -1],
                [-1, 0, -1, 0, -1, 0, 0, 0, -1, 0, -1, 0, -1],
                [-1, 0, -1, 0, -1, -1, -1, 0, -1, 0, -1, 0, -1],
                [-1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1]]
        return np.array(array, dtype=np.int8)