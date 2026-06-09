# environment.py
import numpy as np
import math
import pybullet as p
import pybullet_data
from config import MAZE_WIDTH, MAZE_HEIGHT, START_POS, START_YAW, GOAL_POS, CELL_SIZE, DIRECTIONS
from maze_builder import build_maze, maze_formation

class PyBulletMazeEnvironment:
    """PyBullet environment using your maze definition with discrete movement"""
    def __init__(self, gui=True, render=True):
        # Initialize PyBullet
        self.render = render
        if gui:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        
        # Load ground
        self.ground_id = p.loadURDF("plane.urdf")
        
        # Load maze using your builder
        build_maze()
        
        # Maze configuration
        self.maze_array = maze_formation()
        self.cell_size = CELL_SIZE
        self.maze_width = MAZE_WIDTH
        self.maze_height = MAZE_HEIGHT
        
        # Start and goal positions
        self.start_pos = START_POS
        self.start_yaw = START_YAW
        self.goal_pos = GOAL_POS
        
        # Discrete robot state
        self.robot_x = START_POS[0]
        self.robot_y = START_POS[1]
        self.direction = 0  # 0: East, 1: North, 2: West, 3: South
        
        # Load robot
        self.robot_id = self.load_simple_robot()
        
        # Episode tracking
        self.episode_steps = 0
        self.collision = False
        self.path = []
        
        # Reset to initial state
        self.reset()
    
    def load_simple_robot(self):
        """Load a simple robot for discrete movement"""
        ROBOT_HALF_SIZE = 0.45
        ROBOT_HEIGHT = 0.2

        body_col = p.createCollisionShape(
            p.GEOM_BOX, 
            halfExtents=[ROBOT_HALF_SIZE, ROBOT_HALF_SIZE, ROBOT_HEIGHT/2]
        )

        body_vis = p.createVisualShape(
            p.GEOM_BOX, 
            halfExtents=[ROBOT_HALF_SIZE, ROBOT_HALF_SIZE, ROBOT_HEIGHT/2], 
            rgbaColor=[0, 0, 1, 1]
        )

        robot_id = p.createMultiBody(
            baseMass=0,  # Massless for teleportation
            baseCollisionShapeIndex=body_col,
            baseVisualShapeIndex=body_vis,
            basePosition=[self.robot_x, self.robot_y, ROBOT_HEIGHT/2],
            baseOrientation=p.getQuaternionFromEuler([0, 0, self.start_yaw])
        )
        
        return robot_id
    
    def get_robot_grid_position(self):
        """Get current robot grid position"""
        pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        grid_x = int(pos[0] // self.cell_size)
        grid_y = int(pos[1] // self.cell_size)
        return grid_x, grid_y, self.direction
    
    def teleport_robot(self, dx, dy, dyaw):
        """Teleport robot to new position (discrete movement)"""
        # Update direction if turning
        if dyaw != 0:
            if dyaw > 0:  # Turn left
                self.direction = (self.direction + 1) % 4
            else:  # Turn right
                self.direction = (self.direction - 1) % 4
        
        # Update position if moving
        if dx != 0 or dy != 0:
            # Calculate new position based on current direction
            if self.direction == 0:  # East
                new_x = self.robot_x + dx * self.cell_size
                new_y = self.robot_y
            elif self.direction == 1:  # North
                new_x = self.robot_x
                new_y = self.robot_y + dx * self.cell_size
            elif self.direction == 2:  # West
                new_x = self.robot_x - dx * self.cell_size
                new_y = self.robot_y
            else:  # South
                new_x = self.robot_x
                new_y = self.robot_y - dx * self.cell_size
        else:
            new_x = self.robot_x
            new_y = self.robot_y
        
        # Check if new position is valid (not inside walls)
        grid_x = int(new_x // self.cell_size)
        grid_y = int(new_y // self.cell_size)
        
        if (0 <= grid_x < self.maze_array.shape[1] and 
            0 <= grid_y < self.maze_array.shape[0] and
            self.maze_array[grid_y, grid_x] == 0):
            
            # Update position
            self.robot_x = new_x
            self.robot_y = new_y
            
            # Calculate yaw from direction
            direction_yaw = {
                0: -np.pi/2,  # East
                1: 0,         # North
                2: np.pi/2,   # West
                3: np.pi      # South
            }
            
            # Teleport robot
            p.resetBasePositionAndOrientation(
                self.robot_id,
                [self.robot_x, self.robot_y, 0.2],
                p.getQuaternionFromEuler([0, 0, direction_yaw[self.direction]])
            )
            
            self.collision = False
            return True
        else:
            # Collision with wall
            self.collision = True
            return False
    
    def _bfs_distance(self, start, goal):
        """
        Calculate shortest path distance using BFS on the grid maze.
        
        Args:
            start: (x, y) grid coordinates
            goal: (x, y) grid coordinates
        
        Returns:
            Distance in number of cells, or inf if no path exists
        """
        from collections import deque
        
        sx, sy = start
        gx, gy = goal
        
        # Convert to integer grid coordinates
        sx, sy = int(sx), int(sy)
        gx, gy = int(gx), int(gy)
        
        # Check bounds
        if not (0 <= sx < self.maze_array.shape[1] and 0 <= sy < self.maze_array.shape[0]):
            return float('inf')
        if not (0 <= gx < self.maze_array.shape[1] and 0 <= gy < self.maze_array.shape[0]):
            return float('inf')
        
        # Check if start or goal is a wall
        if self.maze_array[sy, sx] == -1 or self.maze_array[gy, gx] == -1:
            return float('inf')
        
        # Already at goal
        if (sx, sy) == (gx, gy):
            return 0
        
        # BFS
        queue = deque([(sx, sy, 0)])
        visited = set([(sx, sy)])
        
        while queue:
            x, y, dist = queue.popleft()
            
            # Check all 4 directions
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = x + dx, y + dy
                
                # Check bounds
                if not (0 <= nx < self.maze_array.shape[1] and 0 <= ny < self.maze_array.shape[0]):
                    continue
                
                # Check if visited
                if (nx, ny) in visited:
                    continue
                
                # Check if wall
                if self.maze_array[ny, nx] == -1:
                    continue
                
                # Check if goal
                if (nx, ny) == (gx, gy):
                    return dist + 1
                
                # Add to queue
                visited.add((nx, ny))
                queue.append((nx, ny, dist + 1))
        
        return float('inf')

    def get_bfs_distance_to_goal(self):
        """Get BFS distance from current position to goal"""
        grid_x, grid_y, _ = self.get_robot_grid_position()
        goal_grid_x = int(self.goal_pos[0] // self.cell_size)
        goal_grid_y = int(self.goal_pos[1] // self.cell_size)
        
        return self._bfs_distance((grid_x, grid_y), (goal_grid_x, goal_grid_y))
    
    def get_lidar(self):
        """Simulate 4-direction lidar using ray casting"""
        pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        x, y, z = pos
        
        # 4 directions: front, left, right, back relative to current direction
        angles_relative = [0, np.pi/2, -np.pi/2, np.pi]
        distances = []
        
        for rel_angle in angles_relative:
            # Convert to global angle
            direction_angles = {
                0: -np.pi/2,   # East
                1: 0,          # North
                2: np.pi/2,    # West
                3: np.pi       # South
            }
            base_angle = direction_angles[self.direction]
            ray_angle = base_angle + rel_angle
            
            ray_from = [x, y, z + 0.1]
            ray_to = [
                x + 2.0 * math.cos(ray_angle),
                y + 2.0 * math.sin(ray_angle),
                z + 0.1
            ]
            
            ray_result = p.rayTest(ray_from, ray_to)
            
            if ray_result[0][0] != -1:
                hit_fraction = ray_result[0][2]
                distance = hit_fraction * 2.0
            else:
                distance = 2.0
            
            distances.append(distance)
        
        return distances
    
    def get_goal_angle(self):
        """Calculate angle to goal relative to robot orientation"""
        # Get goal in grid coordinates
        goal_grid_x = int(self.goal_pos[0] // self.cell_size)
        goal_grid_y = int(self.goal_pos[1] // self.cell_size)
        
        # Get robot grid position
        robot_grid_x = int(self.robot_x // self.cell_size)
        robot_grid_y = int(self.robot_y // self.cell_size)
        
        # Calculate vector to goal
        dx = goal_grid_x - robot_grid_x
        dy = goal_grid_y - robot_grid_y
        
        # Calculate global angle to goal
        goal_global = math.atan2(dy, dx)
        
        # Convert to robot's local frame
        direction_angles = {
            0: -np.pi/2,   # East
            1: 0,          # North
            2: np.pi/2,    # West
            3: np.pi       # South
        }
        robot_yaw = direction_angles[self.direction]
        
        relative = goal_global - robot_yaw
        relative = math.atan2(math.sin(relative), math.cos(relative))
        
        return relative
    
    def detect_goal(self):
        """Check if robot reached the goal"""
        robot_grid_x = int(self.robot_x // self.cell_size)
        robot_grid_y = int(self.robot_y // self.cell_size)
        
        goal_grid_x = int(self.goal_pos[0] // self.cell_size)
        goal_grid_y = int(self.goal_pos[1] // self.cell_size)
        
        return (robot_grid_x == goal_grid_x and 
                robot_grid_y == goal_grid_y)
    
    def reset(self):
        """Reset environment for new episode"""
        self.episode_steps = 0
        self.collision = False
        self.path = []
        
        # Reset to start position
        self.robot_x = self.start_pos[0]
        self.robot_y = self.start_pos[1]
        self.direction = 0  # Facing East
        
        # Teleport robot to start
        p.resetBasePositionAndOrientation(
            self.robot_id,
            [self.robot_x, self.robot_y, 0.2],
            p.getQuaternionFromEuler([0, 0, self.start_yaw])
        )
        
        self.path.append((self.robot_x, self.robot_y))
        
        if self.render: 
            p.resetDebugVisualizerCamera(
                cameraDistance=25,
                cameraYaw=0,
                cameraPitch=-80,
                cameraTargetPosition=[6, 10, 0]
            )
        
        for _ in range(10):
            p.stepSimulation()
        
        return self.get_lidar(), self.get_goal_angle()
    
    def step(self, action_type, dx=0, dy=0, dyaw=0):
        """Execute one discrete action"""
        self.episode_steps += 1
        
        # Perform the discrete action
        if action_type == 0:  # Move forward
            success = self.teleport_robot(1, 0, 0)
        elif action_type == 1:  # Turn left
            success = self.teleport_robot(0, 0, 1)
        elif action_type == 2:  # Turn right
            success = self.teleport_robot(0, 0, -1)
        elif action_type == 3:  # Move backward
            success = self.teleport_robot(-1, 0, 0)
        
        # Record new position
        self.path.append((self.robot_x, self.robot_y))
        
        # Get new observations
        lidar = self.get_lidar()
        goal_angle = self.get_goal_angle()
        
        return lidar, goal_angle, success
    
    def close(self):
        """Close PyBullet connection"""
        p.disconnect()