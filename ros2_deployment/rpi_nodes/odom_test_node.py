#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import serial
import time
import numpy as np
import transformations

##################### TO TEST MOTORS CHANGE THE LINE 119 TO WHATEVER VALUES YOU WANT #####################

class OdomTest(Node):

    def __init__(self):
        super().__init__('odom_test')

        # =========================================
        # SERIAL
        # =========================================

        self.ser = serial.Serial('/dev/ttyUSB1', 115200, timeout=0.01)
        time.sleep(2)

        self.get_logger().info("Serial connected.")

        # =========================================
        # ODOM SUBSCRIBER
        # =========================================

        self.odom_data = None

        self.create_subscription(
            Odometry,
            '/odom_rf2o',
            self.odom_callback,
            10
        )

        # =========================================
        # TEST STATE
        # =========================================

        self.test_started = False
        self.start_time = None
        self.initial_pose = None

        # =========================================
        # TIMER
        # =========================================

        self.create_timer(0.1, self.control_loop)

        self.get_logger().info("Odom test node started.")

    # ==================================================
    # CALLBACK
    # ==================================================

    def odom_callback(self, msg):
        self.odom_data = msg

    # ==================================================
    # CONTROL LOOP
    # ==================================================

    def control_loop(self):

        if self.odom_data is None:
            print("Waiting for odometry...")
            return

        pose = self.odom_data.pose.pose

        x = pose.position.x
        y = pose.position.y

        q = pose.orientation

        quat = [q.x, q.y, q.z, q.w]
        yaw = 2 * np.arctan2(q.z, q.w)

        # =========================================
        # START TEST
        # =========================================

        if not self.test_started:

            self.initial_pose = (x, y, yaw)

            self.start_time = time.time()

            self.test_started = True

            print("\n==============================")
            print("START POSE")
            print("==============================")

            print(f"x    = {x:.3f}")
            print(f"y    = {y:.3f}")
            print(f"yaw = {yaw:.3f}")

            print("\nDriving forward...\n")

        # =========================================
        # DRIVE FORWARD
        # =========================================

        elapsed = time.time() - self.start_time

        if elapsed < 3.0:

            # forward motion

            self.send_motor(-210, -110)

            print(
                f"RUNNING | "
                f"x={x:.3f} | "
                f"y={y:.3f} | "
                f"yaw={yaw:.3f}"
            )

        # =========================================
        # STOP + RESULTS
        # =========================================

        else:

            self.send_motor('s', 's')

            x0, y0, yaw0 = self.initial_pose

            dx = x - x0
            dy = y - y0
            dyaw = yaw - yaw0

            dist = np.sqrt(dx**2 + dy**2)

            print("\n==============================")
            print("FINAL POSE")
            print("==============================")

            print(f"x    = {x:.3f}")
            print(f"y    = {y:.3f}")
            print(f"yaw = {yaw:.3f}")

            print("\n==============================")
            print("DISPLACEMENT")
            print("==============================")

            print(f"dx       = {dx:.3f}")
            print(f"dy       = {dy:.3f}")
            print(f"distance = {dist:.3f}")
            print(f"dyaw     = {dyaw:.3f}")

            print("\nTEST COMPLETE")

            rclpy.shutdown()

    # ==================================================
    # MOTOR COMMAND
    # ==================================================

    def send_motor(self, left, right):

        self.ser.write(f"l{left}\n".encode())
        self.ser.write(f"r{right}\n".encode())


# ======================================================
# MAIN
# ======================================================

def main():

    rclpy.init()

    node = OdomTest()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()
