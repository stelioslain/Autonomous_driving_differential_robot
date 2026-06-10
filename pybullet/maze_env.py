import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from collections import deque
from maze_builder import build_maze, maze_formation
from lidar import lidar_scan

class MazeEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render=False):
        super().__init__()

        self.render = render
        self.client = p.connect(p.GUI if render else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        
        # Enriched observation space
        self.observation_space = gym.spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(23,), # 16 lidar + 7 features
            dtype=np.float32
        )
        
        # Forward + steering action space
        self.action_space = gym.spaces.Box(
            low=np.array([-1, -1]), 
            high=np.array([1, 1]), 
            dtype=np.float32
        )
        
        self.start = np.array([1.0, 7.0])
        self.goal = np.array([9.0, 20.0])
        
        self.max_steps = 8000
        self.steps = 0
        self.goal_ever_reached = False
        self.reference_path = None
        
        self.reset()

    def reset(self, seed=None, options=None):
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)

        p.loadURDF("plane.urdf")
        build_maze()
        self.maze = maze_formation()
        self.maze_h, self.maze_w = self.maze.shape
        
        self.steps = 0
        
        # Occupancy grid
        # @ = unknown
        # _ = free
        # ! = wall
        self.occupancy = np.full((self.maze_h, self.maze_w),'@' , dtype='U1')

        # Visitation counter (episode-local)
        self.visit_count = np.zeros((self.maze_h, self.maze_w), dtype=np.int32)
        
        # Robot body new dimensions (1x1 robot)
        ROBOT_HALF_SIZE = 0.45  # Half-extent for 1x1 robot (0.45 * 2 = 0.9)
        ROBOT_HEIGHT = 0.2  # Height of robot body

        body_col = p.createCollisionShape(
            p.GEOM_BOX, 
            halfExtents=[ROBOT_HALF_SIZE, ROBOT_HALF_SIZE, ROBOT_HEIGHT/2]  # 0.9 x 1.0 x 0.25
        )

        body_vis = p.createVisualShape(
            p.GEOM_BOX, 
            halfExtents=[ROBOT_HALF_SIZE, ROBOT_HALF_SIZE, ROBOT_HEIGHT/2], 
            rgbaColor=[0, 0, 1, 1]
        )

        # Also update wheel position and size
        WHEEL_RADIUS = 0.12  # Larger wheels
        WHEEL_WIDTH = 0.04

        wheel_col = p.createCollisionShape(
            p.GEOM_CYLINDER, 
            radius=WHEEL_RADIUS, 
            height=WHEEL_WIDTH
        )

        wheel_vis = p.createVisualShape(
            p.GEOM_CYLINDER, 
            radius=WHEEL_RADIUS, 
            length=WHEEL_WIDTH, 
            rgbaColor=[0, 0, 0, 1]
        )
        
        # Create robot with 2 wheels
        self.robot = p.createMultiBody(
            baseMass=3.0,
            baseCollisionShapeIndex=body_col,
            baseVisualShapeIndex=body_vis,
            basePosition=[1, 7, ROBOT_HEIGHT/2],
            baseOrientation=p.getQuaternionFromEuler([0, 0, -np.pi/2]),

            linkMasses=[0.3, 0.3],
            linkCollisionShapeIndices=[wheel_col, wheel_col],
            linkVisualShapeIndices=[wheel_vis, wheel_vis],
            linkPositions=[
                [0,  ROBOT_HALF_SIZE + 0.03, -ROBOT_HEIGHT/4],   # left wheel
                [0, -ROBOT_HALF_SIZE - 0.03, -ROBOT_HEIGHT/4]    # right wheel
            ],
            linkOrientations=[
                p.getQuaternionFromEuler([1.57, 0, 0]),
                p.getQuaternionFromEuler([1.57, 0, 0])
            ],
            linkParentIndices=[0, 0],
            linkJointTypes=[p.JOINT_REVOLUTE, p.JOINT_REVOLUTE],
            linkJointAxis=[[0, 0, 1], [0, 0, 1]],
            linkInertialFramePositions=[
                [0, 0, 0],
                [0, 0, 0]
            ],
            linkInertialFrameOrientations=[
                [0, 0, 0, 1],
                [0, 0, 0, 1]
            ]
        )
        
        # Enable wheel control
        for j in range(2):
            p.setJointMotorControl2(
                self.robot,
                j,
                p.VELOCITY_CONTROL,
                force=0
            )
            
        p.changeDynamics(self.robot, -1, 
                        lateralFriction=1.0, 
                        mass=3.0,
                        linearDamping=0.08,
                        angularDamping=0.08)

        for j in range(2):
            p.changeDynamics(self.robot, j, 
                            lateralFriction=2.0,
                            rollingFriction=0.001)
            
        # Initialize BFS distance tracking
        pos, _ = p.getBasePositionAndOrientation(self.robot)
        start = (int(pos[0]), int(pos[1]))
        goal = (int(self.goal[0]), int(self.goal[1]))
        
        self.prev_grid_dist = self._bfs_distance(start, goal)
        
        if self.render: 
            p.resetDebugVisualizerCamera(
                cameraDistance=25,
                cameraYaw=0,
                cameraPitch=-80,
                cameraTargetPosition=[6, 10, 0]
            )
        
        # Initialize visited cells tracking
        self.visited_cells = set()
        
        # Record starting cell
        pos, _ = p.getBasePositionAndOrientation(self.robot)
        start_cell = (int(pos[0]), int(pos[1]))
        self.visited_cells.add(start_cell)

        obs = self._get_obs()
        
        # Bootstrap occupancy with robot cell
        pos, _ = p.getBasePositionAndOrientation(self.robot)
        cx = int(pos[0])
        cy = int(pos[1])
        if 0 <= cx < (self.maze_w) and 0 <= cy < (self.maze_h):
            self.occupancy[cy, cx] = '_' # Starting cell is always free
        return obs, {}
    
    def _bfs_distance(self, start, goal):
        """
        Wall-aware Manhattan distance on lattice grid.
        Cells are at odd indices.
        Walls are at even indices.
        """

        sx, sy = start
        sx = int(np.floor(sx))
        sy = int(np.floor(sy))
        gx, gy = goal
        gx = int(np.floor(gx))
        gy = int(np.floor(gy))
        
        if not (0 <= sx < self.maze_w and 0 <= sy < self.maze_h):
            return np.inf
        if not (0 <= gx < self.maze_w and 0 <= gy < self.maze_h):
            return np.inf

        if start == goal:
            return 0
        
        queue = deque([(sx, sy, 0)])
        visited = set([(sx, sy)])

        while queue:
            x, y, d = queue.popleft()

            for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]: # Looks to the neighbouring cells
                nx, ny = x + dx, y + dy

                if not (0 <= nx < self.maze_w and 0 <= ny < self.maze_h):
                    continue
                if (nx, ny) in visited:
                    continue
                if self.maze[ny, nx] == -1:
                    continue

                if (nx, ny) == (gx, gy):
                    return d + 1

                visited.add((nx, ny))
                queue.append((nx, ny, d + 1))

        return np.inf

    def _update_occupancy_from_lidar(self):
        pos, ori = p.getBasePositionAndOrientation(self.robot)
        yaw = p.getEulerFromQuaternion(ori)[2]

        lidar, hits = lidar_scan(self.robot)
        angles = np.linspace(0, 2*np.pi, len(lidar), endpoint=False)

        for dist, hit, a in zip(lidar, hits, angles):
            angle = yaw + a

            # Mark wall cell at the end of ray
            if hit == True:
                wx = pos[0] + dist * np.cos(angle)
                wy = pos[1] + dist * np.sin(angle)

                gx = int(np.floor(wx))
                gy = int(np.floor(wy))

                if 0 <= gx < self.maze_w and 0 <= gy < self.maze_h:
                    if self.occupancy[gy, gx] == '_':
                        continue
                    else:
                        self.occupancy[gy, gx] = '!'

        # Always mark robot cell as free
        cx = int(np.floor(pos[0]))
        cy = int(np.floor(pos[1]))

        if 0 <= cx < self.maze_w and 0 <= cy < self.maze_h:
            self.occupancy[cy, cx] = '_'

    def _update_visit_count(self):
        pos, _ = p.getBasePositionAndOrientation(self.robot)
        cx = int(pos[0])
        cy = int(pos[1])

        if 0 <= cx < (self.maze_w) and 0 <= cy < (self.maze_h):
            self.visit_count[cy, cx] += 1
            
            return cx, cy
        return None, None
    
    def _get_obs(self):
        pos, ori = p.getBasePositionAndOrientation(self.robot)
        yaw = p.getEulerFromQuaternion(ori)[2]
        
        # Get goal direction relative to robot
        goal_rel_x = self.goal[0] - pos[0]
        goal_rel_y = self.goal[1] - pos[1]
        
        # Normalize to robot's perspective
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        goal_forward = goal_rel_x * cos_yaw + goal_rel_y * sin_yaw
        goal_right = -goal_rel_x * sin_yaw + goal_rel_y * cos_yaw
        
        # LIDAR scan
        lidar, hits = lidar_scan(self.robot)
        
        # Velocity
        lin_vel, ang_vel = p.getBaseVelocity(self.robot)
        
        # Combine all observations
        obs = np.concatenate([
            lidar / 3.0,  # Normalize by max LIDAR range (3.0)
            [goal_forward / 21.0, goal_right / 21.0],  # Normalize by maze size
            [np.cos(yaw), np.sin(yaw)],  # Orientation
            lin_vel[:2],  # Linear velocity in x,y
            [ang_vel[2]]  # Angular velocity (yaw rate)
        ])
        
        return obs.astype(np.float32)

    def _compute_reference_path(self):
        """
        Compute shortest path from start to goal using BFS.
        Returns a set of (x, y) grid cells.
        """
        start = (int(self.start[0]), int(self.start[1]))
        goal = (int(self.goal[0]), int(self.goal[1]))

        queue = deque([start])
        came_from = {start: None}
        visited = set([start])

        while queue:
            x, y = queue.popleft()

            if (x, y) == goal:
                break

            for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                nx, ny = x + dx, y + dy

                if not (0 <= nx < self.maze_w and 0 <= ny < self.maze_h):
                    continue
                if self.maze[ny, nx] == -1:
                    continue
                if (nx, ny) in visited:
                    continue

                visited.add((nx, ny))
                came_from[(nx, ny)] = (x, y)
                queue.append((nx, ny))

        # Reconstruct path
        path = set()
        node = goal
        while node is not None and node in came_from:
            path.add(node)
            node = came_from[node]

        return path
    
    def _validate_reference_path(self, path):
        start = (int(self.start[0]), int(self.start[1]))
        goal = (int(self.goal[0]), int(self.goal[1]))

        print("\n=== REFERENCE PATH VALIDATION ===")

        # 1. Empty check
        if not path or len(path) == 0:
            print("Path is empty")
            return False

        print(f"✔ Path length: {len(path)}")

        # 2. Start / goal presence
        if start not in path:
            print("Start NOT in path:", start)
            return False
        print("✔ Start in path")

        if goal not in path:
            print("Goal NOT in path:", goal)
            return False
        print("✔ Goal in path")

        # 3. Wall check
        for (x, y) in path:
            if self.maze[y, x] == -1:
                print("Path goes through wall at:", (x, y))
                return False
        print("✔ No wall cells in path")

        # 4. Connectivity check
        for (x, y) in path:
            neighbors = [
                (x+1, y), (x-1, y),
                (x, y+1), (x, y-1)
            ]
            if (x, y) not in [start, goal]:
                if not any(n in path for n in neighbors):
                    print("Disconnected cell in path:", (x, y))
                    return False
        print("Path is 4-connected")

        # 5. Optimal length check
        bfs_dist = self._bfs_distance(start, goal)
        if bfs_dist == np.inf:
            print("BFS distance is infinite")
            return False

        if len(path) != bfs_dist + 1:
            print(f"Path length mismatch: BFS={bfs_dist}, path={len(path)}")
        else:
            print("Path length matches BFS shortest distance")

        print("Reference path VALID")
        return True
    
    def _distance_to_reference_path(self, cx, cy):
        if self.reference_path is None:
            return None

        return min(abs(cx - px) + abs(cy - py) for px, py in self.reference_path)
    
    def _draw_reference_path(self):
        if self.reference_path is None:
            return

        for (x, y) in self.reference_path:
            p.addUserDebugLine(
                [x - 0.4, y - 0.4, 0.05],
                [x + 0.4, y + 0.4, 0.05],
                [0, 0, 1],  # Blue
                lineWidth=2,
                lifeTime=10
            )

    def step(self, action):
        forward, steering = action
        
        # Convert to differential drive
        left = forward - steering
        right = forward + steering
        
        # Clip to valid range
        left = np.clip(left, -1, 1)
        right = np.clip(right, -1, 1)
        
        # Apply motor control
        speed = 12
        p.setJointMotorControlArray(
            self.robot,
            jointIndices=[0, 1],
            controlMode=p.VELOCITY_CONTROL,
            targetVelocities=[left * speed, right * speed],
            forces=[4, 4]
        )
        
        # Step simulation
        self.steps += 1
        for _ in range(10):
            p.stepSimulation()
        
        # Update internal map
        self._update_occupancy_from_lidar()

        # Update visitation
        cx, cy = self._update_visit_count()
        
        goal_cell = (
            int(self.goal[0]),
            int(self.goal[1])
        )

        if cx is None or cy is None:
            grid_dist = self.prev_grid_dist
        else:
            grid_dist = self._bfs_distance((cx, cy), goal_cell)
        
        # To not "pass" through/jump between walls
        if abs(grid_dist - self.prev_grid_dist) >= 8:
            grid_dist = self.prev_grid_dist
            
        self.grid_dist = grid_dist

        # Get observation
        obs = self._get_obs()
        
        # Check termination conditions
        reached_goal = self._distance_to_goal() <= 1.0
        out = self._out_of_bounds()
        
        terminated = reached_goal or out
        truncated = self.steps >= self.max_steps
        
        # Adding weights to the path-to-goal after first finding of the goal
        if reached_goal and not self.goal_ever_reached:
            self.goal_ever_reached = True
            self.reference_path = self._compute_reference_path()
            # self._validate_reference_path(self.reference_path)
            # self._draw_reference_path()
        
        # Compute reward
        reward = self._compute_reward(reached_goal, out)
        
        # Debugging output
        if self.render and self.steps % 50 == 0:
            print(
                f"step={self.steps} "
                f"grid_dist={self.grid_dist:.2f} "
                f"prev_grid_dist={self.prev_grid_dist:.2f}"
                f"reward={reward:.3f}"
            )
        
        return obs, reward, terminated, truncated, {}
    
    def _out_of_bounds(self):
        pos, _ = p.getBasePositionAndOrientation(self.robot)
        x, y = pos[0], pos[1]
        return x < 0 or x > 12 or y < 0 or y > 20

    def _distance_to_goal(self):
        pos, _ = p.getBasePositionAndOrientation(self.robot)
        return np.linalg.norm(np.array(pos[:2]) - self.goal)

    def _compute_reward(self, reached_goal, out_of_bounds):
        pos, _ = p.getBasePositionAndOrientation(self.robot)
        
        # ---------- 1. BFS grid progress ----------
        grid_progress_reward = 0.0

        if hasattr(self, "prev_grid_dist"):
            if np.isfinite(self.prev_grid_dist) and np.isfinite(self.grid_dist):
                grid_progress_reward = np.clip((self.prev_grid_dist - self.grid_dist), -2, 2) * 0.85

        self.prev_grid_dist = self.grid_dist

        # ---------- 2. Goal reward ----------
        goal_reward = 50.0 if reached_goal else 0.0

        # ---------- 3. Time penalty ----------
        time_penalty = -0.001

        # ---------- 4. Exploration / novelty ----------
        cx = int(pos[0])
        cy = int(pos[1])
        novelty_bonus = 0.0

        if 0 <= cx < self.maze_w and 0 <= cy < self.maze_h:
            visits = self.visit_count[cy, cx]
            novelty_bonus = 2.0 / (1.0 + visits) if self.steps < 1500 else 0.3 / (1.0 + visits)

        # ---------- 5. Wall proximity ----------
        lidar, hits = lidar_scan(self.robot)
        min_dist = np.min(lidar)
        proximity_penalty = -0.001 if min_dist < 0.15 else 0.0

        # ---------- 6. Collision ----------
        collision_penalty = -0.005 * len(p.getContactPoints(self.robot))

        # ---------- 7. Out of bounds ----------
        bounds_penalty = -20.0 if out_of_bounds else 0.0
        
        # ---------- 8. Reference path shaping (AFTER goal found) ----------
        path_reward = 0.0

        if self.goal_ever_reached and self.reference_path is not None:
            cx = int(pos[0])
            cy = int(pos[1])

            if (cx, cy) in self.reference_path:
                path_reward = +0.0          # On optimal path
            else:
                d = self._distance_to_reference_path(cx, cy)
                if d == 1:
                    path_reward = -0.1     # Very close
                elif d == 2:
                    path_reward = -0.3
                elif d >= 4:
                    path_reward = -0.8      # Far from optimal

        # ---------- 9. Getting closer to goal ----------
        closing_in_on_goal_reward = 0.0
        
        if self.goal_ever_reached:
            starting_closing_in = (int(pos[0]), int(pos[1]))
            goal_closing_in = (int(self.goal[0]), int(self.goal[1]))
            closing_in_on_goal_reward = 8.0 / (self._bfs_distance(starting_closing_in, goal_closing_in))
            

        reward = (
            grid_progress_reward
            + goal_reward
            + novelty_bonus
            + time_penalty
            + proximity_penalty
            + collision_penalty
            + bounds_penalty
            + path_reward
            + closing_in_on_goal_reward
        )

        return float(np.clip(reward, -10.0, 70.0))

    def close(self):
        p.disconnect(self.client)
