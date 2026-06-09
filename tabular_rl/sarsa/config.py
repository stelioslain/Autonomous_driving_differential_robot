# config.py
import numpy as np
from collections import deque

# ==================== CONFIGURATION ====================
# Discrete movement actions
ACTIONS = {
    0: (1, 0, 0),   # move forward 1 cell
    1: (0, 1, 0),   # turn left 90 degrees
    2: (0, 0, 1),   # turn right 90 degrees
    3: (-1, 0, 0),  # move backward 1 cell
}
N_ACTIONS = len(ACTIONS)

# Learning parameters
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.5
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.995

# Training parameters
N_EPISODES = 500
MAX_STEPS_PER_EPISODE = 200

# Maze parameters
MAZE_WIDTH = 13     # 13 columns (grid cells)
MAZE_HEIGHT = 21    # 21 rows (grid cells)
CELL_SIZE = 1.0

# Start position (grid coordinates)
START_X = 1
START_Y = 7
START_POS = [START_X, START_Y, 0.1]  # Center of cell
START_YAW = -np.pi/2  # Facing right/east

# Goal position (grid coordinates)
GOAL_X = 9
GOAL_Y = 20
GOAL_POS = [GOAL_X, GOAL_Y, 0.0]

# Direction vectors
DIRECTIONS = {
    0: (1, 0),   # East (right)
    1: (0, 1),   # North (up)
    2: (-1, 0),  # West (left)
    3: (0, -1),  # South (down)
}

# BFS distance parameters
MAX_BFS_DISTANCE = 100  # Maximum distance for normalization