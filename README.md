# Autonomous Driving for a Differential Mobile Robot using Reinforcement Learning and ROS 2

This repository contains the source code developed for the diploma thesis:

**Autonomous Driving for a Differential Mobile Robot using Reinforcement Learning and ROS 2 Jazzy on a Raspberry Pi - Arduino Platform**

Author: Stylianos Lainas  
Department of Electrical and Computer Engineering  
University of Thessaly  

## Overview

The project focuses on autonomous navigation for a differential-drive mobile robot. The robot uses a Raspberry Pi 4 as the main computational unit and an Arduino Nano as the low-level motor controller.

The system combines:

- ROS 2 Jazzy Jalisco
- Raspberry Pi 4
- Arduino Nano
- RPLidar C1
- RF2O laser odometry
- MuJoCo simulation
- PyBullet simulation experiments
- Proximal Policy Optimization (PPO)
- Q-learning and SARSA exploratory implementations
- Left-hand-rule classical navigation baseline

The reinforcement learning policy was trained in simulation and later tested on the real robot inside a cardboard maze.

## Repository Structure

```text
arduino/
    Arduino Nano motor control code.

mujoco/
    Final MuJoCo reinforcement learning environment and PPO training scripts.

pybullet/
    Initial PyBullet simulation environment and experiments.

ros2_deployment/
    ROS 2 launch files and deployment notes for the real robot.

tabular_rl/
    Q-learning and SARSA implementations.

models/
    Trained PPO model files and model documentation.

results/
    Simulation and real-world evaluation results.
```

## Hardware Platform

The physical robot platform consisted of:

- Raspberry Pi 4
- Arduino Nano
- RPLidar C1
- L298N motor driver
- Two DC motors
- Differential-drive chassis
- Power bank for the Raspberry Pi
- Separate battery for the motor driver and motors

The Raspberry Pi handled high-level processing, ROS 2 communication, LiDAR data acquisition, odometry estimation, and navigation logic. The Arduino Nano handled low-level motor control through serial commands sent from the Raspberry Pi.

## Simulation

The final reinforcement learning environment was implemented in MuJoCo.

To train the PPO model:

```bash
cd mujoco
python train.py
```

To fine-tune the trained model:

```bash
python fine_tune.py
```

To test the fine-tuned model:

```bash
python test.py
```

The final fine-tuned PPO policy achieved the following simulation results:

```text
Testing Results (100 episodes)
Success Rate: 96/100 (96.0%)
Average Steps: 170.2
Average Reward: 53.88
Minimum Reward: 49.38
Maximum Reward: 70.38
```

## ROS 2 Real-World Deployment

The real robot used ROS 2 Jazzy Jalisco on Ubuntu 24.04.

The RPLidar C1 was launched using:

```bash
ros2 launch rplidar_ros rplidar_c1_launch.py
```

RViz2 was used to visualize LiDAR scan data:

```bash
ros2 run rviz2 rviz2
```

RF2O laser odometry was launched using:

```bash
ros2 launch rf2o_laser_odometry rf2o_laser_odometry.launch.py
```

The most important ROS 2 topics were:

```text
/scan
/odom_rf2o
```

The `/scan` topic provided LiDAR measurements, while `/odom_rf2o` provided odometry and orientation feedback.

Additional ROS 2 launch details are provided in:

```text
ros2_deployment/launch_notes.md
```

## Arduino Motor Control

The Arduino Nano received text-based serial commands from the Raspberry Pi.

Examples:

```text
l120     set left motor speed to 120
r120     set right motor speed to 120
l-120    set left motor speed to -120
r-120    set right motor speed to -120
ls       stop left motor
rs       stop right motor
```

Motor values were in the range:

```text
-255 to 255
```

Positive values corresponded to forward rotation and negative values corresponded to reverse rotation.

## Reinforcement Learning

The main reinforcement learning algorithm used in this project was Proximal Policy Optimization (PPO), implemented with Stable-Baselines3.

The final observation space contained:

- 16 LiDAR distance measurements
- Relative goal position in the robot frame
- Robot orientation
- Linear velocity
- Angular velocity

The action space consisted of:

```text
[forward_command, steering_command]
```

These actions were converted into differential-drive wheel commands.

## Real-World Evaluation

The robot was tested in a physical cardboard maze.

Two navigation methods were tested:

1. Left-hand-rule algorithm
2. Fine-tuned PPO policy

Both methods successfully completed the real maze in assisted trials. The main real-world limitations were:

- Wheel friction
- Actuator dead-zone
- Turning overshoot
- Battery-dependent motor response
- Occasional contact with cardboard walls

The PPO policy produced smoother motion than the left-hand-rule algorithm, while the left-hand-rule algorithm served as a useful deterministic baseline.

## Dependencies

The main Python dependencies are listed in:

```text
requirements.txt
```

A Conda environment file is also provided:

```text
environment.yml
```

To create the Conda environment:

```bash
conda env create -f environment.yml
conda activate autonomous_robot_rl
```

## Code Availability

This repository is provided as accompanying material for the diploma thesis. It includes the simulation environments, PPO training and testing scripts, Arduino firmware, ROS 2 deployment files, tabular reinforcement learning experiments, trained model information, and evaluation results.