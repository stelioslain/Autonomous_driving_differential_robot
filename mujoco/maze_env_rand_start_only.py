import gymnasium as gym
import numpy as np
import mujoco
import mujoco.viewer
from collections import deque

class PID:
    def __init__(self, kp, ki, kd, output_limits=(-1.0, 1.0)):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.min_out, self.max_out = output_limits
        self.integral = 0.0
        self.prev_error = 0.0

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0

    def step(self, error, dt):
        self.integral += error * dt
        self.integral = np.clip(self.integral, -1.0, 1.0)
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        self.prev_error = error

        output = (
            self.kp * error
            + self.ki * self.integral
            + self.kd * derivative
        )

        return np.clip(output, self.min_out, self.max_out)


class MazeEnv(gym.Env):

    def __init__(self, render=False):
        super().__init__()

        self.render = render

        self.model = mujoco.MjModel.from_xml_path("maze.xml")
        self.data = mujoco.MjData(self.model)

        self.viewer = None
        if render:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        # action = forward, steering
        self.action_space = gym.spaces.Box(
            low=np.array([-1, -1]),
            high=np.array([1, 1]),
            dtype=np.float32
        )

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(23,), dtype=np.float32
        )

        # Initializing maze array
        self.maze = np.array([[-1, -1, -1, -1, -1, -1, -1],
            [-1, 0, -1, 0, 0, 0, -1],
            [-1, 0, -1, -1, -1, 0, -1],
            [-1, 0, 0, 0, 0, 0, -1],
            [-1, -1, -1, 0, -1, 0, -1],
            [-1, 0, -1, 0, -1, 0, -1],
            [-1, 0, -1, -1, -1, 0, -1],
            [-1, 0, 0, 0, 0, 0, -1],
            [-1, 0, -1, 0, -1, -1, -1],
            [-1, 0, -1, 0, 0, 0, -1],
            [-1, 0, -1, -1, -1, -1, -1],
            [-1, 0, 0, 0, 0, 0, -1],
            [-1, -1, -1, -1, -1, -1, -1]], dtype=np.int8)

        self.maze_h, self.maze_w = self.maze.shape

        self.start = np.array([1.0, 1.0])
        self.goal = np.array([5.0, 11.0])

        self.prev_grid_dist = None
        self.grid_dist = None

        self.max_linear_speed = 9.6
        self.max_angular_speed = 18

        self.max_steps = 2000
        self.steps = 0

        self.robot_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot")

        # Initializing the collision id of the robot
        self.robot_geom_ids = []

        for i in range(self.model.ngeom):
            body_id = self.model.geom_bodyid[i]
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)

            if body_name == "robot":
                self.robot_geom_ids.append(i)

        # PID controller for staying on path
        self.orientation_pid = PID(
            kp = 2.0,
            ki = 0.0,
            kd = 0.08,
            output_limits=(-1.0, 1.0)
        )

        self.reset()

    # -------------------------------------------------

    def reset(self, seed=None, options=None):
        mujoco.mj_resetData(self.model, self.data)

        # ---------- FOR RANDOM STARTING POINT ----------
        # Find all free cells (value 0) except the goal
        free_cells = np.argwhere(self.maze == 0)
        goal_cell = np.array([5, 11])  # (y, x) in your maze indexing
        free_cells = np.array([cell for cell in free_cells if not np.array_equal(cell[::-1], goal_cell)])

        # Random orientation
        random_yaw = np.random.uniform(-np.pi, np.pi)
        # random_yaw = - np.pi / 2  # Fixed orientation for testing
        qw = np.cos(random_yaw / 2)
        qx = 0.0
        qy = 0.0
        qz = np.sin(random_yaw / 2)
        self.data.qpos[3:7] = np.array([qw, qx, qy, qz])

        mujoco.mj_forward(self.model, self.data)

        # Choose a random start from free cells
        rand_idx = np.random.choice(len(free_cells))
        start_cell = free_cells[rand_idx]
        start_x, start_y = start_cell[::-1]  # reverse because maze[y,x]
        # start_x, start_y = 1, 1 # Fixed start for testing

        self.start = np.array([start_x, start_y])

        # Set robot position in MuJoCo
        self.data.qpos[0] = self.start[0]
        self.data.qpos[1] = self.start[1]

        # Reset PID state on episode reset
        self.orientation_pid.reset()

        # Get observations
        obs = self._get_obs()

        self.steps = 0
        self.prev_grid_dist = self._bfs_distance(self.start, self.goal)

        return obs, {}

    # -------------------------------------------------

    def step(self, action):
        forward, steering = action

        qw, qx, qy, qz = self.data.qpos[3:7]
        yaw = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))

        if steering < -0.5:
            target_yaw = np.pi # West
        elif steering < 0.0:
            target_yaw = -np.pi/2 # South
        elif steering < 0.5:
            target_yaw = 0.0 # East
        else:
            target_yaw = np.pi/2 # North

        # Yaw error and normalization to range [-np.pi, np.pi]
        yaw_error = target_yaw - yaw
        yaw_error = (yaw_error + np.pi) % (2*np.pi) - np.pi

        # PID control
        dt = self.model.opt.timestep * 10 # 0.1s
        angular_correction = self.orientation_pid.step(yaw_error, dt)

        # Steering result
        steering = steering + angular_correction
        steering = np.clip(steering, -1.0, 1.0)

        # Convert to differential drive
        left = forward - steering
        right = forward + steering

        left = np.clip(left, -1, 1)
        right = np.clip(right, -1, 1)

        # Applying motor control
        max_wheel_speed = 60

        self.data.ctrl[0] = left * max_wheel_speed
        self.data.ctrl[1] = right * max_wheel_speed

        # Simulating steps
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        if self.viewer:
            self.viewer.sync()

        self.steps += 1

        # Calculating the Manhattan distance from the robot to the exit taking into account the walls
        self.grid_dist = self._bfs_distance((self.data.qpos[0], self.data.qpos[1]), self.goal)

        obs = self._get_obs()

        reached_goal = self._distance_to_goal() < 1.0
        terminated = reached_goal
        truncated = self.steps >= self.max_steps

        reward = self._compute_reward(reached_goal)

        return obs, reward, terminated, truncated, {}

    # -------------------------------------------------

    def _get_obs(self):
        x = self.data.qpos[0]
        y = self.data.qpos[1]

        qw, qx, qy, qz = self.data.qpos[3:7]
        yaw = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))

        # Get goal direction relative to robot
        goal_rel_x = self.goal[0] - x
        goal_rel_y = self.goal[1] - y

        # Normalize to robot's perspective
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        goal_forward = goal_rel_x * cos_yaw + goal_rel_y * sin_yaw
        goal_right = -goal_rel_x * sin_yaw + goal_rel_y * cos_yaw

        # LIDAR scan
        self.lidar = self._lidar_scan(x, y, yaw)

        # Velocities in world
        vx, vy, vz = self.data.qvel[0:3]
        wx, wy, wz = self.data.qvel[3:6]

        v_forward = vx * cos_yaw + vy * sin_yaw
        v_right = -vx * sin_yaw + vy * cos_yaw

        # Normalization factors for the MLP (Multi Layer Perceptron)
        max_distance = np.linalg.norm(self.goal - self.start)

        obs = np.concatenate([
            self.lidar / 3.0, # Normalize with LIDAR max range
            [goal_forward / max_distance, goal_right / max_distance], # Normalize with max distance
            [np.cos(yaw), np.sin(yaw)], # Orientation already normalized
            [v_forward / self.max_linear_speed, # Normalize linear speed
            v_right / self.max_linear_speed],
            [wz / self.max_angular_speed] # Normalize angular speed
        ])

        return obs.astype(np.float32)

    # -------------------------------------------------

    def _lidar_scan(self, x, y, yaw, num_rays=16, max_dist=3.0):
        angles = np.linspace(0, 2*np.pi, num_rays, endpoint=False)
        dists = []

        for a in angles:
            dx = np.cos(yaw + a)
            dy = np.sin(yaw + a)

            origin = np.array([x, y, 0.2])
            direction = np.array([dx, dy, 0])

            geomid = np.array([-1], dtype=np.int32)

            dist = mujoco.mj_ray(
                self.model,
                self.data,
                origin,
                direction,
                None, # geomgroup
                1, # hit static geoms
                self.robot_body_id,
                geomid
            )

            if dist < 0:
                dist = max_dist

            dists.append(min(dist, max_dist))

        return np.array(dists)

    # -------------------------------------------------

    def _bfs_distance(self, start, goal):
        # Wall-aware Manhattan distance

        sx, sy = start
        sx = int(np.floor(sx))
        sy = int(np.floor(sy))
        gx, gy = goal
        gx = int(np.floor(gx))
        gy = int(np.floor(gy))
        
        if not (0 <= sx < self.maze_w and 0 <= sy < self.maze_h):
            print("out of x-axis")
            return np.inf
        if not (0 <= gx < self.maze_w and 0 <= gy < self.maze_h):
            print("out of y-axis")
            return np.inf

        if np.array_equal(start, goal):
            return 0
        
        queue = deque([(sx, sy, 0)])
        visited = set([(sx, sy)])

        while queue:
            x, y, d = queue.popleft()

            for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
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

    # -------------------------------------------------

    def _distance_to_goal(self):
        x = self.data.qpos[0]
        y = self.data.qpos[1]
        return np.linalg.norm(np.array([x, y]) - self.goal)

    # -------------------------------------------------

    def _compute_reward(self, reached):
        reward = 0.0
        robot_contacts = 0

        # Progress toward goal
        if np.isfinite(self.prev_grid_dist) and np.isfinite(self.grid_dist):
            distance_change = np.clip((self.prev_grid_dist - self.grid_dist), -1.0, 1.0)
            reward += distance_change
            if distance_change < 0:
                reward -= 0.1

        # Collision penalty
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if c.geom1 in self.robot_geom_ids or c.geom2 in self.robot_geom_ids:
                robot_contacts += 1
        
        reward -= 0.05 * robot_contacts

        # Reached goal
        if reached:
            reward += 50

        # Time penalty
        reward -= 0.005

        # Setting the grid distance
        self.prev_grid_dist = self.grid_dist

        return float(reward)

    # -------------------------------------------------

    def close(self):
        if self.viewer:
            self.viewer.close()
