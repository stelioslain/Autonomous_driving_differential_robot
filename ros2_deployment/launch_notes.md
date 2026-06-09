# ROS 2 Launch Notes

This document summarizes the ROS 2 launch configuration used during the real-world deployment of the robot.

The real robot used two main ROS 2 processes:

1. The RPLidar C1 node for LiDAR scan acquisition.
2. The RF2O laser odometry node for odometry and orientation estimation from LiDAR scans.

The launch files used during the experiments are stored in:

```text
ros2_deployment/launch/
```

## Workspace Setup

Before launching the ROS 2 nodes, the ROS 2 environment and the local workspace must be sourced:

```bash
source /opt/ros/jazzy/setup.bash
source ~/lidar_ws/install/setup.bash
```

If these commands have been added to `.bashrc`, they are executed automatically when opening a new terminal.

## RPLidar C1

The RPLidar C1 was launched using:

```bash
ros2 launch rplidar_ros rplidar_c1_launch.py
```

The launch file used in this thesis was configured with the following main parameters:

```text
channel_type: serial
serial_port: /dev/ttyUSB0
serial_baudrate: 460800
frame_id: laser
inverted: false
angle_compensate: true
scan_mode: Standard
```

The main output topic of the RPLidar node was:

```text
/scan
```

This topic provides the laser scan measurements used by the navigation algorithms and by the RF2O laser odometry node.

The serial port may change depending on the USB connection order. In the final setup, the LiDAR was connected through `/dev/ttyUSB0`, but on another boot it may appear as `/dev/ttyUSB1`.

## RViz2 Visualization

RViz2 was used to verify that the LiDAR data were being published correctly:

```bash
ros2 run rviz2 rviz2
```

Inside RViz2, the `/scan` topic can be visualized as laser scan points on a 2D plane.

## RF2O Laser Odometry

The RF2O laser odometry node was launched using:

```bash
ros2 launch rf2o_laser_odometry rf2o_laser_odometry.launch.py
```

The launch file used in this thesis was configured with the following main parameters:

```text
laser_scan_topic: /scan
odom_topic: /odom_rf2o
publish_tf: true
base_frame_id: base_link
odom_frame_id: odom
freq: 10.0
```

The RF2O node subscribes to:

```text
/scan
```

and publishes odometry information on:

```text
/odom_rf2o
```

The `/odom_rf2o` topic was used by the navigation algorithms to obtain orientation feedback for the real robot.

## Topic Inspection

The available ROS 2 topics can be inspected using:

```bash
ros2 topic list
```

The most important topics during the real-world experiments were:

```text
/scan
/odom_rf2o
```

To inspect the LiDAR data:

```bash
ros2 topic echo /scan
```

To inspect the RF2O odometry data:

```bash
ros2 topic echo /odom_rf2o
```

## Final Topic Flow

The real-world sensing and odometry pipeline can be summarized as:

```text
RPLidar C1 → /scan → RF2O Laser Odometry → /odom_rf2o → Navigation Algorithm
```

The navigation algorithm used `/scan` for wall detection and `/odom_rf2o` for orientation feedback. Motor commands were then sent from the Raspberry Pi to the Arduino Nano through serial communication.
