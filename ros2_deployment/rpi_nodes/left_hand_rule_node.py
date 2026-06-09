#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

import numpy as np
import serial
import math
import time


# ============================================================
# PARAMETERS
# ============================================================

FORWARD_SPEED = 100
TURN_SPEED = 100

FRONT_THRESHOLD = 0.35
LEFT_OPEN_THRESHOLD = 0.35

WALL_FOLLOW_DIST = 0.25

STABILIZE_TIME = 0.35
TURN_ON_TIME = 0.02
TURN_OFF_TIME = 0.03

TURN_TOLERANCE = math.radians(6)

SERIAL_PORT = '/dev/ttyUSB1'
BAUDRATE = 115200


# ============================================================
# NODE
# ============================================================

class MazeSolver(Node):

    def __init__(self):

        super().__init__('maze_solver')

        # ====================================================
        # SUBSCRIBERS
        # ====================================================

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom_rf2o',
            self.odom_callback,
            10
        )

        # ====================================================
        # TIMER
        # ====================================================

        self.timer = self.create_timer(
            0.02,
            self.control_loop
        )

        # ====================================================
        # VARIABLES
        # ====================================================
        
        self.turn_pulse_start = time.time()
        self.turn_pulse_on = True

        self.scan_data = None

        self.yaw = 0.0

        self.state = "FORWARD"

        self.target_yaw = 0.0
        self.forward_target_yaw = 0.0
        
        self.just_turned_left = False

        self.left_turn_lock_time = 1.0
        
        self.stabilize_start_time = 0.0

        self.left_turn_time = 0.0

        # Maze reference frame
        self.maze_zero_yaw = None

        # Cardinal direction index
        # 0 = NORTH
        # 1 = EAST
        # 2 = SOUTH
        # 3 = WEST
        self.direction_index = 0

        # Heading controller
        self.kp_heading = 3.0

        # Wall following controller
        self.kp_wall = 25.0

        # ====================================================
        # SERIAL
        # ====================================================

        self.ser = serial.Serial(
            SERIAL_PORT,
            BAUDRATE,
            timeout=1
        )

        time.sleep(2)

        self.get_logger().info("Maze Solver Started")

    # ========================================================
    # CALLBACKS
    # ========================================================

    def scan_callback(self, msg):

        self.scan_data = msg

    def odom_callback(self, msg):

        q = msg.pose.pose.orientation

        # Quaternion -> yaw

        siny_cosp = 2.0 * (
            q.w * q.z +
            q.x * q.y
        )

        cosy_cosp = 1.0 - 2.0 * (
            q.y * q.y +
            q.z * q.z
        )

        self.yaw = math.atan2(
            siny_cosp,
            cosy_cosp
        )

        # ================================================
        # INITIAL MAZE ORIENTATION
        # ================================================

        if self.maze_zero_yaw is None:

            self.maze_zero_yaw = self.yaw

            self.forward_target_yaw = self.maze_zero_yaw

            self.get_logger().info(
                f"Maze zero yaw = "
                f"{math.degrees(self.maze_zero_yaw):.2f}"
            )

    # ========================================================
    # MOTOR CONTROL
    # ========================================================

    def send_motor(self, left, right):

        left = int(np.clip(left, -255, 255))
        right = int(np.clip(right, -255, 255))

        self.ser.write(f"l{left}\n".encode())
        self.ser.write(f"r{right}\n".encode())

    def stop(self):

        self.send_motor(0, 0)

    def pulse_turn_left_motor(self):

        now = time.time()

        if self.turn_pulse_on:

            self.send_motor(-TURN_SPEED, TURN_SPEED)

            if now - self.turn_pulse_start > TURN_ON_TIME:
                self.turn_pulse_on = False
                self.turn_pulse_start = now

        else:

            self.stop()

            if now - self.turn_pulse_start > TURN_OFF_TIME:
                self.turn_pulse_on = True
                self.turn_pulse_start = now


    def pulse_turn_right_motor(self):

        now = time.time()

        if self.turn_pulse_on:

            self.send_motor(TURN_SPEED, -TURN_SPEED)

            if now - self.turn_pulse_start > TURN_ON_TIME:
                self.turn_pulse_on = False
                self.turn_pulse_start = now

        else:

            self.stop()

            if now - self.turn_pulse_start > TURN_OFF_TIME:
                self.turn_pulse_on = True
                self.turn_pulse_start = now

    # ========================================================
    # ANGLE HELPERS
    # ========================================================

    def normalize_angle(self, angle):

        return math.atan2(
            math.sin(angle),
            math.cos(angle)
        )

    # ========================================================
    # MAZE CARDINAL SYSTEM
    # ========================================================

    def get_cardinal_yaw(self):

        base = self.maze_zero_yaw

        cardinals = [

            self.normalize_angle(base),                 # NORTH

            self.normalize_angle(base - math.pi/2),    # EAST

            self.normalize_angle(base - math.pi),      # SOUTH

            self.normalize_angle(base + math.pi/2)     # WEST
        ]

        return cardinals[self.direction_index]

    # ========================================================
    # LIDAR PROCESSING
    # ========================================================

    def angle_to_index(self, angle_deg):

        angle_rad = math.radians(angle_deg)

        angle_rad = self.normalize_angle(angle_rad)

        index = int(
            (angle_rad - self.scan_data.angle_min)
            / self.scan_data.angle_increment
        )

        index %= len(self.scan_data.ranges)

        return index

    def get_sector_distance(self, center_deg, width_deg=10):

        if self.scan_data is None:
            return float('inf')

        distances = []

        start = center_deg - width_deg
        end = center_deg + width_deg

        for angle in range(start, end + 1):

            idx = self.angle_to_index(angle)

            d = self.scan_data.ranges[idx]

            if np.isfinite(d):
                distances.append(d)

        if len(distances) == 0:
            return float('inf')

        return np.mean(distances)

    # ========================================================
    # LIDAR DIRECTIONS
    # ========================================================

    # RPLidar mounted reversed:
    #
    # Robot Front = Lidar Back
    # Robot Left  = Lidar Right
    # Robot Right = Lidar Left

    def get_front_distance(self):

        return self.get_sector_distance(
            180,
            15
        )

    def get_left_distance(self):

        return self.get_sector_distance(
            -90,
            15
        )

    def get_right_distance(self):

        return self.get_sector_distance(
            90,
            15
        )

    # ========================================================
    # TURNING
    # ========================================================

    def start_turn_left(self):

        # Update maze direction

        self.direction_index = (
            self.direction_index - 1
        ) % 4

        # Target yaw from maze cardinal system

        self.target_yaw = self.get_cardinal_yaw()

        self.state = "TURN_LEFT"
        
        self.turn_pulse_start = time.time()
        self.turn_pulse_on = True

        self.get_logger().info(
            f"TURN LEFT -> "
            f"{math.degrees(self.target_yaw):.1f}"
        )

    def start_turn_right(self):

        # Update maze direction

        self.direction_index = (
            self.direction_index + 1
        ) % 4

        # Target yaw from maze cardinal system

        self.target_yaw = self.get_cardinal_yaw()

        self.state = "TURN_RIGHT"
        
        self.turn_pulse_start = time.time()
        self.turn_pulse_on = True

        self.get_logger().info(
            f"TURN RIGHT -> "
            f"{math.degrees(self.target_yaw):.1f}"
        )

    def turning_finished(self):

        error = self.normalize_angle(
            self.target_yaw - self.yaw
        )

        return abs(error) < TURN_TOLERANCE

    # ========================================================
    # CONTROL LOOP
    # ========================================================

    def control_loop(self):

        if self.scan_data is None:
            return

        if self.maze_zero_yaw is None:
            return

        # ====================================================
        # SENSOR DATA
        # ====================================================

        front = self.get_front_distance()

        left = self.get_left_distance()

        right = self.get_right_distance()

        # ====================================================
        # DEBUG
        # ====================================================

        self.get_logger().info(

            f"State={self.state} | "

            f"Yaw={math.degrees(self.yaw):.1f} | "

            f"Target={math.degrees(self.forward_target_yaw):.1f} | "

            f"Front={front:.2f} | "

            f"Left={left:.2f} | "

            f"Right={right:.2f}"
        )

        # ====================================================
        # FORWARD
        # ====================================================

        if self.state == "FORWARD":

            # -----------------------------------------------
            # LEFT OPEN -> TURN LEFT
            # -----------------------------------------------
            left_turn_allowed = True

            if self.just_turned_left:

                elapsed = time.time() - self.left_turn_time

                if elapsed < self.left_turn_lock_time:

                    left_turn_allowed = False

                else:

                    self.just_turned_left = False

            if left_turn_allowed and left > LEFT_OPEN_THRESHOLD:

                self.stop()

                time.sleep(0.15)

                self.start_turn_left()

                return

            # -----------------------------------------------
            # FRONT BLOCKED -> TURN RIGHT
            # -----------------------------------------------

            if front < FRONT_THRESHOLD:

                self.stop()

                time.sleep(0.15)

                self.start_turn_right()

                return
            
            # =================================================
            # HEADING CONTROL
            # =================================================

            heading_error = self.normalize_angle(
                self.forward_target_yaw - self.yaw
            )

            heading_correction = (
                self.kp_heading *
                heading_error
            )

            # =================================================
            # WALL FOLLOWING
            # =================================================

            wall_correction = 0.0

            # Only follow wall if wall actually exists
            if left < 0.25:

                wall_error = WALL_FOLLOW_DIST - left

                wall_correction = (
                    self.kp_wall *
                    wall_error
                )

            # =================================================
            # COMBINE
            # =================================================

            correction = (
                heading_correction
                # + wall_correction
            )

            left_motor = FORWARD_SPEED - correction
            right_motor = FORWARD_SPEED + correction

            # ------------------------------------------------
            # Ensure motors never go below minimum usable speed
            # ------------------------------------------------

            MIN_SPEED = 100

            if 0 < left_motor < MIN_SPEED:
                left_motor = MIN_SPEED

            if 0 < right_motor < MIN_SPEED:
                right_motor = MIN_SPEED

            # Optional upper clamp
            left_motor = np.clip(left_motor, 100, 150)
            right_motor = np.clip(right_motor, 100, 150)
                        
            self.send_motor(
                left_motor,
                right_motor
            )

        # ====================================================
        # TURN LEFT
        # ====================================================

        elif self.state == "TURN_LEFT":

            if not self.turning_finished():
            
                self.pulse_turn_left_motor()

            else:
                
                self.send_motor(90, -90)
                time.sleep(0.015)
                
                self.stop()

                time.sleep(0.2)

                # Snap forward heading
                self.forward_target_yaw = (
                    self.target_yaw
                )
                
                self.just_turned_left = True

                self.left_turn_time = time.time()
                
                # self.stabilize_start_time = time.time()

                # self.state = "STABILIZE"
                
                self.state = "FORWARD"

                self.get_logger().info(
                    "LEFT TURN COMPLETE"
                )

        # ====================================================
        # TURN RIGHT
        # ====================================================

        elif self.state == "TURN_RIGHT":

            if not self.turning_finished():

                self.pulse_turn_right_motor()

            else:
            
                self.send_motor(-90, 90)
                time.sleep(0.015)
                
                self.stop()

                time.sleep(0.2)

                # Snap forward heading
                self.forward_target_yaw = (
                    self.target_yaw
                )
                
                # self.stabilize_start_time = time.time()

                # self.state = "STABILIZE"
                
                self.state = "FORWARD"

                self.get_logger().info(
                    "RIGHT TURN COMPLETE"
                )
        
        # ====================================================
        # STABILIZE
        # ====================================================

        elif self.state == "STABILIZE":

            elapsed = (
                time.time() -
                self.stabilize_start_time
            )

            # -----------------------------------------------
            # FINISHED
            # -----------------------------------------------

            if elapsed > STABILIZE_TIME:

                self.state = "FORWARD"

                return

            # -----------------------------------------------
            # STRONG HEADING PID
            # -----------------------------------------------

            heading_error = self.normalize_angle(
                self.forward_target_yaw - self.yaw
            )

            correction = 320.0 * heading_error

            correction = np.clip(
                correction,
                -60,
                60
            )

            left_motor = FORWARD_SPEED - correction
            right_motor = FORWARD_SPEED + correction

            MIN_SPEED = 100

            if 0 < left_motor < MIN_SPEED:
                left_motor = MIN_SPEED

            if 0 < right_motor < MIN_SPEED:
                right_motor = MIN_SPEED

            left_motor = np.clip(left_motor, -255, 255)
            right_motor = np.clip(right_motor, -255, 255)

            self.send_motor(
                left_motor,
                right_motor
            )


# ============================================================
# MAIN
# ============================================================

def main(args=None):

    rclpy.init(args=args)

    node = MazeSolver()

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:

        pass

    finally:

        node.stop()

        node.destroy_node()

        rclpy.shutdown()


if __name__ == '__main__':

    main()
