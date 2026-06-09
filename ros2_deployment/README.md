# ROS 2 Real-World Deployment

This folder contains the files used to deploy the navigation algorithms on the physical Raspberry Pi--Arduino robot.

The Raspberry Pi was responsible for:

- receiving LiDAR data from the RPLidar C1,
- receiving odometry and orientation feedback from RF2O laser odometry,
- executing the navigation algorithm,
- sending motor commands to the Arduino Nano through serial communication.

## Folder Structure

```text
launch/
    Launch files used for the RPLidar C1 and RF2O laser odometry.

rpi_nodes/
    Python ROS 2 nodes executed on the Raspberry Pi.

launch_notes.md
    Notes about ROS 2 launch commands, topics, and configuration.
```

## Main ROS 2 Topics

```text
/scan
/odom_rf2o
```

The `/scan` topic provided LiDAR measurements from the RPLidar C1.

The `/odom_rf2o` topic provided odometry and orientation feedback from RF2O laser odometry.

## Raspberry Pi Python Nodes

The Python files in `rpi_nodes/` contain the real-world navigation logic.

```text
left_hand_rule_node.py
    Classical left-hand-rule maze-solving node.

ppo_deploy_node.py
    Real-world PPO deployment node.

odom_test_node.py
    Test node used to inspect RF2O odometry and motor behaviour.
```

## Serial Communication

The Raspberry Pi communicated with the Arduino Nano through serial communication.

In the current deployment scripts, the Arduino serial port is configured as:

```text
/dev/ttyUSB1
```

The baud rate is:

```text
115200
```

The serial port may change depending on the USB connection order. If the Arduino appears on a different port, the serial port variable inside the Python node must be updated.

## ROS 2 Dependencies

The real-world deployment requires ROS 2 Jazzy Jalisco.

The following Python modules are provided by ROS 2 and are not installed through `requirements.txt`:

```text
rclpy
sensor_msgs
nav_msgs
geometry_msgs
std_msgs
```

Additional Python dependencies such as `numpy`, `pyserial`, `torch`, and `stable-baselines3` are listed in the root `requirements.txt`.

## PPO Model Path for Real-World Deployment

The PPO deployment node currently loads the model using:

```python
self.model = PPO.load("best_model.zip")
```

This means that the path inside `ppo_deploy_node.py` must be changed to `ppo_mujoco.zip` or to `ppo_mujoco_fine_tuned.zip` and the corresponding zip file has to be placed in the same working directory from which the node is executed.

For example, if the fine-tuned model is stored in the root `models/` folder, the path should be changed according to the execution directory.

A safer option is to use an absolute path on the Raspberry Pi, for example:

```python
self.model = PPO.load("/home/username/Autonomous_driving_differential_robot/models/ppo_mujoco_fine_tuned.zip")
```

Replace `username` with the actual Raspberry Pi username.