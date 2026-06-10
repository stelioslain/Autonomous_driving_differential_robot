#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

import numpy as np
import serial
import time
import math

from stable_baselines3 import PPO


# ============================================================
# REAL → SIMULATION MAZE SCALE
# ============================================================

# Real maze:
# one physical node-to-node movement = 0.38 m
#
# Simulation/matrix:
# one node-to-node movement is:
# (1,1) -> (3,1)
# or
# (1,1) -> (1,3)
#
# Therefore:
# 0.38 m real = 2 simulation coordinate units

REAL_NODE_DISTANCE_M = 0.38
SIM_NODE_DISTANCE = 2.0
SIM_UNITS_PER_METER = SIM_NODE_DISTANCE / REAL_NODE_DISTANCE_M


# ============================================================
# RF2O → MAZE POSITION SIGNS
# ============================================================

MAZE_FORWARD_SIGN = 1.0
MAZE_RIGHT_SIGN = -1.0


# ============================================================
# IMPORTANT SIMULATION BODY ORIENTATION
# ============================================================

# In simulation:
# yaw = 0      -> body/front points +X/east
# yaw = +90°   -> body/front points +Y/north
# yaw = -90°   -> body/front points -Y/south
#
# Since the trained model effectively moves with the rear,
# and you start the real robot backwards so its rear travels up the maze,
# the BODY FRONT is pointing down the maze.
#
# Therefore initial simulation body yaw should be -90 degrees.

INITIAL_SIM_YAW = - math.pi / 2


# ============================================================
# LIDAR SETTINGS
# ============================================================

MIN_VALID_LIDAR = 0.05


# ============================================================
# PID CLASS
# ============================================================

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

        self.integral = np.clip(
            self.integral,
            -1.0,
            1.0
        )

        derivative = 0.0

        if dt > 0.0:

            derivative = (
                error -
                self.prev_error
            ) / dt

        self.prev_error = error

        output = (
            self.kp * error
            + self.ki * self.integral
            + self.kd * derivative
        )

        return np.clip(
            output,
            self.min_out,
            self.max_out
        )


# ============================================================
# PPO DEPLOYMENT NODE
# ============================================================

class PPOMazeDeploy(Node):

    def __init__(self):

        super().__init__('ppo_maze_deploy')

        # ====================================================
        # SIMULATION PARAMETERS
        # ====================================================

        self.start_pos = np.array(
            [1.0, 1.0],
            dtype=np.float32
        )

        self.goal = np.array(
            [5.0, 11.0],
            dtype=np.float32
        )

        self.max_lidar = 3.0
        self.num_lidar_rays = 16

        self.max_linear_speed = 9.6
        self.max_angular_speed = 18.0

        self.max_distance = np.linalg.norm(
            self.goal -
            self.start_pos
        )

        # ====================================================
        # REAL ROBOT MOTOR PARAMETERS
        # ====================================================

        self.MIN_PWM = 100
        self.MAX_PWM = 220
        self.DEADZONE = 0.02
        
        self.MOTOR_ENABLE = True
        
        # ====================================================
        # PER-MOTOR CALIBRATION
        # ====================================================

        # The real motors are not perfectly symmetric.
        # The test showed that left reverse needs more PWM
        # than right reverse to produce a mirrored curve.
        #
        # Tune only these values after motor tests.

        self.LEFT_GAIN_FORWARD = 1.00
        self.LEFT_GAIN_BACKWARD = 1.08

        self.RIGHT_GAIN_FORWARD = 1.00
        self.RIGHT_GAIN_BACKWARD = 1.00

        self.LEFT_MIN_FORWARD = 110
        self.LEFT_MIN_BACKWARD = 100

        self.RIGHT_MIN_FORWARD = 110
        self.RIGHT_MIN_BACKWARD = 100

        self.PWM_LIMIT = 230

        # Conservative real-world scaling.
        self.FORWARD_SCALE = 0.55
        self.STEERING_SCALE = 0.32

        # ====================================================
        # MOTOR FRAME
        # ====================================================

        # If raw_act[0] > 0 and the robot moves the wrong way,
        # test changing this to -1.0.
        self.FORWARD_SIGN = 1.0

        # If robot turns opposite from PPO steering, change this to -1.0.
        self.STEERING_SIGN = 1.0

        # If left/right motors are physically swapped, set this True.
        self.SWAP_MOTORS = False

        # ====================================================
        # LIDAR ORIENTATION
        # ====================================================

        # Even if the model moves rear-first, the SIMULATION OBSERVATION
        # still uses the body/front direction as lidar ray 0.
        #
        # Therefore:
        # lidar_data[0] must represent real physical FRONT
        #
        # The robot's physical front corresponds to lidar 180 degrees.
        self.LIDAR_YAW_OFFSET = math.pi

        # If lidar_L and lidar_R are swapped, change this to -1.0.
        self.LIDAR_ANGLE_SIGN = 1.0

        # ====================================================
        # OBSERVATION / CONTROL OPTIONS
        # ====================================================

        # RF2O twist may not match MuJoCo qvel frame.
        # Zeroing it is safer for first successful deployment.
        self.USE_ODOM_TWIST = False

        # Use simulation-like PID.
        self.USE_ORIENTATION_PID = False

        # ====================================================
        # POSITION FILTERING
        # ====================================================

        # Smooth RF2O position before giving x,y to PPO.
        self.x_filtered = None
        self.y_filtered = None
        self.POSITION_ALPHA = 0.25

        # ====================================================
        # LATCHED TARGET YAW
        # ====================================================

        self.current_target_yaw = None
        self.last_target_change_time = time.time()

        # Prevent rapid target switching that causes:
        # turn left -> reverse -> turn right -> reverse.
        self.TARGET_HOLD_TIME = 0.3

        # Do not accept new target unless robot is not extremely misaligned.
        self.TARGET_CHANGE_ALLOWED_ERROR = math.radians(35)

        # ====================================================
        # ACTION SMOOTHING / NO-PIVOT PROTECTION
        # ====================================================

        self.ACTION_ALPHA = 0.75
        self.prev_forward_cmd = 0.0
        self.prev_steering_cmd = 0.0

        # Prevent very sharp real-world turns caused by PWM deadzone.
        self.MAX_STEERING_WHEN_MOVING = 0.32

        # ====================================================
        # SERIAL
        # ====================================================

        self.SERIAL_PORT = '/dev/ttyUSB1'
        self.BAUDRATE = 115200

        self.ser = serial.Serial(
            self.SERIAL_PORT,
            self.BAUDRATE,
            timeout=0.01
        )

        time.sleep(2)

        self.get_logger().info(
            "Serial connected to Arduino."
        )

        # ====================================================
        # LOAD PPO MODEL
        # ====================================================
        
        # Change this path according to where the trained model is stored on the Raspberry Pi, or place the model's zip file in the same directory as this script.
        # Example:
        # "/home/username/Autonomous_driving_differential_robot/models/ppo_mujoco_fine_tuned.zip"
        # Where "username" is your Raspberry Pi username and "ppo_mujoco_fine_tuned.zip" is the name of the model file you want to test.
        
        self.model = PPO.load(
            "ppo_mujoco_fine_tuned.zip"
        )

        self.get_logger().info(
            "PPO model loaded."
        )

        # ====================================================
        # ORIENTATION PID
        # ====================================================

        # Much weaker than simulation.
        # Simulation used kp=2.0, kd=0.08.
        # Real robot needs lower correction due to deadzone/friction.
        self.orientation_pid = PID(
            kp=0.35,
            ki=0.0,
            kd=0.01,
            output_limits=(-0.20, 0.20)
        )

        self.last_time = time.time()

        # ====================================================
        # STATE VARIABLES
        # ====================================================

        self.scan_msg = None
        self.lidar_data = None
        self.odom_data = None

        self.initial_odom_x = None
        self.initial_odom_y = None
        self.initial_odom_yaw = None

        # ====================================================
        # ROS SUBSCRIBERS
        # ====================================================

        self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        self.create_subscription(
            Odometry,
            '/odom_rf2o',
            self.odom_callback,
            10
        )

        # Same control period as simulation step group: 10 * 0.01 = 0.1 s.
        self.create_timer(
            0.1,
            self.control_loop
        )

        self.get_logger().info(
            "PPO Deployment Node Started."
        )

    # ========================================================
    # CALLBACKS
    # ========================================================

    def scan_callback(self, msg):

        self.scan_msg = msg

        self.lidar_data = self.make_lidar_observation(
            msg
        )

    def odom_callback(self, msg):

        self.odom_data = msg

    # ========================================================
    # BASIC HELPERS
    # ========================================================

    def normalize_angle(self, angle):

        return math.atan2(
            math.sin(angle),
            math.cos(angle)
        )

    def quaternion_to_yaw(self, q):

        siny_cosp = 2.0 * (
            q.w * q.z +
            q.x * q.y
        )

        cosy_cosp = 1.0 - 2.0 * (
            q.y * q.y +
            q.z * q.z
        )

        return math.atan2(
            siny_cosp,
            cosy_cosp
        )

    def nearest_odd_cell(self, value):

        return int(
            2 * round(
                (value - 1.0) / 2.0
            ) + 1
        )

    # ========================================================
    # LIDAR HELPERS
    # ========================================================

    def angle_to_index(self, msg, angle_rad):

        angle_rad = self.normalize_angle(
            angle_rad
        )

        index = int(
            (
                angle_rad -
                msg.angle_min
            )
            / msg.angle_increment
        )

        index %= len(msg.ranges)

        return index

    def get_sector_value(self, msg, center_angle_rad, width_deg=8):
        # Returns the distances from the nearest wall/obstacle

        ranges = np.array(
            msg.ranges,
            dtype=np.float32
        )

        distances = []

        width_rad = math.radians(
            width_deg
        )

        samples = np.linspace(
            center_angle_rad - width_rad,
            center_angle_rad + width_rad,
            9
        )

        for angle in samples:

            idx = self.angle_to_index(
                msg,
                angle
            )

            d = ranges[idx]

            if not np.isfinite(d):
                continue

            if d < MIN_VALID_LIDAR:
                continue
            
            d = d * SIM_UNITS_PER_METER
            
            d = np.clip(
                d,
                0.0,
                self.max_lidar
            )

            distances.append(d)

        if len(distances) == 0:

            return self.max_lidar

        return float(np.median(distances))

    def make_lidar_observation(self, msg):

        """
        Creates the 16 lidar rays expected by the PPO model.

        Simulation order:
        ray 0  = simulation body/front
        ray 4  = simulation body/left
        ray 8  = simulation body/back
        ray 12 = simulation body/right

        Even if the policy drives rear-first, ray 0 is still body/front.
        """

        lidar_16 = []

        for i in range(self.num_lidar_rays):

            robot_angle = (
                2.0 *
                math.pi *
                i /
                self.num_lidar_rays
            )

            lidar_angle = self.normalize_angle(
                self.LIDAR_ANGLE_SIGN *
                robot_angle
                +
                self.LIDAR_YAW_OFFSET
            )

            d = self.get_sector_value(
                msg,
                lidar_angle,
                width_deg=3
            )

            lidar_16.append(d)

        lidar_16 = np.array(
            lidar_16,
            dtype=np.float32
        )

        return lidar_16 / self.max_lidar

    # ========================================================
    # RF2O ODOMETRY → SIMULATION MAZE POSITION
    # ========================================================

    def get_maze_position_from_odom(self, pose):

        real_x = pose.position.x
        real_y = pose.position.y

        raw_yaw = self.quaternion_to_yaw(
            pose.orientation
        )

        raw_yaw = self.normalize_angle(
            raw_yaw
        )

        # Save starting RF2O pose once.
        #
        # This physical pose corresponds to simulation position (1,1).
        # The simulation BODY yaw at this pose is INITIAL_SIM_YAW.
        if self.initial_odom_x is None:

            self.initial_odom_x = real_x
            self.initial_odom_y = real_y
            self.initial_odom_yaw = raw_yaw

            self.get_logger().info(
                f"Initial odom pose saved: "
                f"x={real_x:.3f}, "
                f"y={real_y:.3f}, "
                f"raw_yaw={math.degrees(raw_yaw):.1f}, "
                f"initial_sim_yaw={math.degrees(INITIAL_SIM_YAW):.1f}"
            )

        dx_odom = (
            real_x -
            self.initial_odom_x
        )

        dy_odom = (
            real_y -
            self.initial_odom_y
        )

        # Convert RF2O odom displacement to the starting odom frame.
        cos0 = np.cos(
            self.initial_odom_yaw
        )

        sin0 = np.sin(
            self.initial_odom_yaw
        )

        forward_m = (
            dx_odom * cos0
            +
            dy_odom * sin0
        )

        right_m = (
            -dx_odom * sin0
            +
            dy_odom * cos0
        )

        # Convert meters to simulation units.
        #
        # 0.38 m real movement = 2 simulation coordinate units.
        x_maze = (
            self.start_pos[0]
            +
            MAZE_RIGHT_SIGN *
            right_m *
            SIM_UNITS_PER_METER
        )

        y_maze = (
            self.start_pos[1]
            +
            MAZE_FORWARD_SIGN *
            forward_m *
            SIM_UNITS_PER_METER
        )

        # Convert real RF2O yaw difference to simulation BODY yaw.
        sim_yaw = self.normalize_angle(
            raw_yaw
            -
            self.initial_odom_yaw
            +
            INITIAL_SIM_YAW
        )

        return (
            x_maze,
            y_maze,
            sim_yaw,
            forward_m,
            right_m,
            raw_yaw
        )

    # ========================================================
    # TARGET YAW HELPER
    # ========================================================

    def steering_to_cardinal_yaw(self, steering):

        if steering < -0.5:

            return math.pi          # West

        elif steering < 0.0:

            return -math.pi / 2.0   # South

        elif steering < 0.5:

            return 0.0              # East

        else:

            return math.pi / 2.0    # North

    # ========================================================
    # PWM MAPPING
    # ========================================================

    def apply_motor_deadzone(self, value):
        """
        Convert one normalized wheel command [-1, 1] to PWM,
        while respecting motor deadzone.

        This function is used after left/right have already been jointly scaled.
        """

        if abs(value) < 1e-6:
            return 0

        value = np.clip(value, -1.0, 1.0)

        pwm = int(
            self.MIN_PWM
            + (self.MAX_PWM - self.MIN_PWM) * abs(value)
        )

        if value < 0:
            pwm = -pwm

        return pwm


    def wheels_to_pwm(self, left, right):
        """
        Convert normalized left/right wheel commands to PWM.

        Important:
        This preserves the left/right ratio as much as possible,
        while ensuring that if the robot is supposed to move,
        both motors receive enough PWM to overcome deadzone.
        """

        left = float(np.clip(left, -1.0, 1.0))
        right = float(np.clip(right, -1.0, 1.0))

        max_abs = max(abs(left), abs(right))

        # If both are tiny, stop.
        if max_abs < self.DEADZONE:
            return 0, 0

        # Normalize so the stronger wheel becomes 1.0.
        # This preserves the curve ratio.
        left_scaled = left / max_abs
        right_scaled = right / max_abs

        # Now apply a speed envelope based on how strong the original command was.
        # This prevents every movement from jumping to MAX_PWM.
        envelope = np.clip(max_abs, 0.0, 1.0)

        pwm_peak = int(
            self.MIN_PWM
            + (self.MAX_PWM - self.MIN_PWM) * envelope
        )

        def convert_scaled(v):
            if abs(v) < 0.15:
                return 0

            pwm = int(pwm_peak * abs(v))

            # If it should move, force it above minimum usable PWM.
            if 0 < pwm < self.MIN_PWM:
                pwm = self.MIN_PWM

            if v < 0:
                pwm = -pwm

            return int(np.clip(pwm, -255, 255))

        left_pwm = convert_scaled(left_scaled)
        right_pwm = convert_scaled(right_scaled)

        return left_pwm, right_pwm
    
    def calibrate_motor_pwm(self, left_pwm, right_pwm):
        """
        Apply separate calibration for left/right motors and
        forward/backward directions.

        This compensates for the fact that the two motors do not
        produce the same real torque for the same PWM.
        """

        # -------------------------
        # LEFT MOTOR
        # -------------------------

        if left_pwm > 0:

            left_pwm = int(
                left_pwm *
                self.LEFT_GAIN_FORWARD
            )

            if 0 < abs(left_pwm) < self.LEFT_MIN_FORWARD:

                left_pwm = self.LEFT_MIN_FORWARD

        elif left_pwm < 0:

            left_pwm = int(
                left_pwm *
                self.LEFT_GAIN_BACKWARD
            )

            if 0 < abs(left_pwm) < self.LEFT_MIN_BACKWARD:

                left_pwm = -self.LEFT_MIN_BACKWARD

        # -------------------------
        # RIGHT MOTOR
        # -------------------------

        if right_pwm > 0:

            right_pwm = int(
                right_pwm *
                self.RIGHT_GAIN_FORWARD
            )

            if 0 < abs(right_pwm) < self.RIGHT_MIN_FORWARD:

                right_pwm = self.RIGHT_MIN_FORWARD

        elif right_pwm < 0:

            right_pwm = int(
                right_pwm *
                self.RIGHT_GAIN_BACKWARD
            )

            if 0 < abs(right_pwm) < self.RIGHT_MIN_BACKWARD:

                right_pwm = -self.RIGHT_MIN_BACKWARD

        left_pwm = int(
            np.clip(
                left_pwm,
                -self.PWM_LIMIT,
                self.PWM_LIMIT
            )
        )

        right_pwm = int(
            np.clip(
                right_pwm,
                -self.PWM_LIMIT,
                self.PWM_LIMIT
            )
        )

        return left_pwm, right_pwm
    
    # ========================================================
    # CONTROL LOOP
    # ========================================================

    def control_loop(self):

        if self.lidar_data is None or self.odom_data is None:

            self.get_logger().info(
                "Waiting for lidar/odom..."
            )

            return

        now = time.time()

        dt = now - self.last_time

        self.last_time = now

        if dt <= 0.0:

            dt = 0.1

        pose = self.odom_data.pose.pose
        twist = self.odom_data.twist.twist

        (
            x,
            y,
            yaw,
            forward_m,
            right_m,
            raw_yaw
        ) = self.get_maze_position_from_odom(
            pose
        )

        # ====================================================
        # FILTER POSITION BEFORE PPO OBSERVATION
        # ====================================================

        x_raw = x
        y_raw = y

        # For PPO observation, use the current RF2O-derived position directly.
        # Filtering causes delay around turns and can make the robot continue curves too long.
        x = x_raw
        y = y_raw

        # ====================================================
        # GOAL VECTOR IN ROBOT FRAME
        # Same as simulation.
        # ====================================================

        goal_rel_x = (
            self.goal[0] -
            x
        )

        goal_rel_y = (
            self.goal[1] -
            y
        )

        cos_yaw = np.cos(
            yaw
        )

        sin_yaw = np.sin(
            yaw
        )

        goal_forward = (
            goal_rel_x *
            cos_yaw
            +
            goal_rel_y *
            sin_yaw
        )

        goal_right = (
            -goal_rel_x *
            sin_yaw
            +
            goal_rel_y *
            cos_yaw
        )

        goal_forward /= self.max_distance
        goal_right /= self.max_distance

        goal_forward = np.clip(
            goal_forward,
            -1.0,
            1.0
        )

        goal_right = np.clip(
            goal_right,
            -1.0,
            1.0
        )

        # ====================================================
        # VELOCITY OBSERVATION
        # ====================================================

        if self.USE_ODOM_TWIST:

            vx = twist.linear.x
            vy = twist.linear.y
            wz = twist.angular.z

            v_forward = (
                vx *
                cos_yaw
                +
                vy *
                sin_yaw
            )

            v_right = (
                -vx *
                sin_yaw
                +
                vy *
                cos_yaw
            )

            v_forward /= self.max_linear_speed
            v_right /= self.max_linear_speed
            wz /= self.max_angular_speed

            v_forward = np.clip(
                v_forward,
                -1.0,
                1.0
            )

            v_right = np.clip(
                v_right,
                -1.0,
                1.0
            )

            wz = np.clip(
                wz,
                -1.0,
                1.0
            )

        else:

            v_forward = 0.0
            v_right = 0.0
            wz = 0.0

        # ====================================================
        # OBSERVATION: 23D
        # ====================================================

        obs = np.concatenate([
            self.lidar_data,
            [goal_forward, goal_right],
            [np.cos(yaw), np.sin(yaw)],
            [v_forward, v_right],
            [wz]
        ]).astype(
            np.float32
        )

        # ====================================================
        # PPO ACTION
        # ====================================================

        action, _ = self.model.predict(
            obs,
            deterministic=True
        )

        forward = float(
            action[0]
        )

        steering = float(
            action[1]
        )

        # ====================================================
        # LATCHED SIMULATION-LIKE ORIENTATION PID
        # ====================================================

        if self.USE_ORIENTATION_PID:

            proposed_target_yaw = self.steering_to_cardinal_yaw(
                steering
            )

            now_time = time.time()

            if self.current_target_yaw is None:

                self.current_target_yaw = proposed_target_yaw
                self.last_target_change_time = now_time
                self.orientation_pid.reset()

            target_error_now = abs(
                self.normalize_angle(
                    self.current_target_yaw -
                    yaw
                )
            )

            can_change_target = (
                (now_time - self.last_target_change_time)
                >
                self.TARGET_HOLD_TIME
                and
                target_error_now
                <
                self.TARGET_CHANGE_ALLOWED_ERROR
            )

            if can_change_target:

                new_target_difference = abs(
                    self.normalize_angle(
                        proposed_target_yaw -
                        self.current_target_yaw
                    )
                )

                if new_target_difference > math.radians(45):

                    self.current_target_yaw = proposed_target_yaw
                    self.last_target_change_time = now_time
                    self.orientation_pid.reset()

            target_yaw = self.current_target_yaw

            yaw_error = self.normalize_angle(
                target_yaw -
                yaw
            )

            angular_correction = self.orientation_pid.step(
                yaw_error,
                dt
            )

            steering = (
                steering +
                angular_correction
            )

            steering = np.clip(
                steering,
                -1.0,
                1.0
            )

        else:

            target_yaw = yaw
            angular_correction = 0.0

        # ====================================================
        # REAL ROBOT SAFETY SCALING
        # ====================================================

        forward *= self.FORWARD_SCALE
        steering *= self.STEERING_SCALE

        # Limit steering globally.
        steering = np.clip(
            steering,
            -self.MAX_STEERING_WHEN_MOVING,
            self.MAX_STEERING_WHEN_MOVING
        )

        # Smooth commands.
        forward = (
            self.ACTION_ALPHA * forward
            +
            (1.0 - self.ACTION_ALPHA) * self.prev_forward_cmd
        )

        steering = (
            self.ACTION_ALPHA * steering
            +
            (1.0 - self.ACTION_ALPHA) * self.prev_steering_cmd
        )

        self.prev_forward_cmd = forward
        self.prev_steering_cmd = steering

        forward *= self.FORWARD_SIGN
        steering *= self.STEERING_SIGN
        
        # ====================================================
        # REVERSE TURN BOOST
        # ====================================================

        # The trained model often moves rear-first.
        # In real life, reverse curves may be weaker than in MuJoCo,
        # so we slightly increase steering only while reversing.
        if forward < 0.0:
            steering *= 1.15

        # ====================================================
        # NO-PIVOT PROTECTION
        # ====================================================

        max_allowed_steering = max(
            0.05,
            abs(forward) * 1.05
        )

        steering = np.clip(
            steering,
            -max_allowed_steering,
            max_allowed_steering
        )

        # ====================================================
        # DIFFERENTIAL DRIVE
        # Same as simulation: left = forward - steering
        # ====================================================

        left = (
            forward -
            steering
        )

        right = (
            forward +
            steering
        )

        left = np.clip(
            left,
            -1.0,
            1.0
        )

        right = np.clip(
            right,
            -1.0,
            1.0
        )

        left_pwm_raw, right_pwm_raw = self.wheels_to_pwm(
            left,
            right
        )

        left_pwm, right_pwm = self.calibrate_motor_pwm(
            left_pwm_raw,
            right_pwm_raw
        )

        # ====================================================
        # DEBUG
        # ====================================================

        cell_x = self.nearest_odd_cell(
            x
        )

        cell_y = self.nearest_odd_cell(
            y
        )

        direction_text = (
            "PPO_FORWARD"
            if action[0] >= 0.0
            else
            "PPO_REVERSE"
        )

        self.get_logger().info(
            f"initial_sim_yaw={math.degrees(INITIAL_SIM_YAW):.1f} "
            f"maze_pos=({x:.2f},{y:.2f}) "
            f"raw_pos=({x_raw:.2f},{y_raw:.2f}) "
            f"cell=({cell_x},{cell_y}) "
            f"forward_m={forward_m:.2f} "
            f"right_m={right_m:.2f} "
            f"raw_yaw={math.degrees(raw_yaw):.1f} "
            f"sim_yaw={math.degrees(yaw):.1f} "
            f"target_yaw={math.degrees(target_yaw):.1f} "
            f"goal_f={goal_forward:.2f} "
            f"goal_r={goal_right:.2f} "
            f"lidar_F={self.lidar_data[0]:.2f} "
            f"lidar_L={self.lidar_data[4]:.2f} "
            f"lidar_B={self.lidar_data[8]:.2f} "
            f"lidar_R={self.lidar_data[12]:.2f} "
            f"raw_act=({action[0]:.2f},{action[1]:.2f}) "
            f"{direction_text} "
            f"pid={angular_correction:.2f} "
            f"raw_cmd=({left_pwm_raw},{right_pwm_raw}) "
            f"cmd=({left_pwm},{right_pwm})"
        )
        
        if not self.MOTOR_ENABLE:
            self.send_motor(0, 0)
            return

        if self.SWAP_MOTORS:

            self.send_motor(
                right_pwm,
                left_pwm
            )

        else:

            self.send_motor(
                left_pwm,
                right_pwm
            )

    # ========================================================
    # SERIAL MOTOR COMMAND
    # ========================================================

    def send_motor(self, left, right):

        self.ser.write(
            f"l{left}\n".encode()
        )

        self.ser.write(
            f"r{right}\n".encode()
        )


# ============================================================
# MAIN
# ============================================================

def main():

    rclpy.init()

    node = PPOMazeDeploy()

    try:

        rclpy.spin(
            node
        )

    except KeyboardInterrupt:

        pass

    finally:

        node.send_motor(
            0,
            0
        )

        node.destroy_node()

        rclpy.shutdown()


if __name__ == '__main__':

    main()
