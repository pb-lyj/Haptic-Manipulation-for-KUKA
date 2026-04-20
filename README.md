# 🖐️ haptic (ROS 2)

ROS 2 package for **KUKA LBR + Tac3D tactile sensors**, built for:
- Human teaching and tactile data collection
- Tactile/force data broadcasting and synchronized logging
- Policy‑driven Cartesian pose control with safety thresholds

---

## ✨ Key Features

- 🤖 **KUKA LBR integration** (based on lbr_fri_ros2_stack)
- 🧠 **Tac3D dual‑sensor publishers** (left/right)
  - 3D positions / displacements / forces (sensor_msgs/Image, 32FC3)
  - resultant force / resultant moment (geometry_msgs/Vector3)
- 🗂️ **Data logging**
  - dataset_recorder: multi‑topic raw text logging with timestamps
  - dataset_recorder_h5: fixed‑rate sampling into HDF5
- 🎯 **Cartesian controller** (cartesian_controller)
  - accepts policy pose targets
  - interpolation rate limiting + workspace clipping + force/moment limits
- 🔁 **One‑shot launch flow** (launch/teach_prepare.py)
  - starts admittance control, Tac3D SDK, sensor ROS nodes, optional reset

---

## 🧱 Requirements

- Ubuntu 24.04
- ROS 2 Jazzy
- Python 3.12
- KUKA FRI + LBR ROS 2 driver
  - https://github.com/lbr-stack/lbr_fri_ros2_stack

Optional simulation dependency (Gazebo):

```bash
sudo apt update
sudo apt install ros-jazzy-gz-gazebo
sudo apt install ros-jazzy-ros-gz-sim
sudo apt install ros-jazzy-ros-gz
```

---

## 🚀 Quick Start

> ⚠️ **Safety Notice**
> 
> For real robot motion, make sure E‑Stop is available, the workspace is clear, and the initial pose is safe.

### 1) Build Workspace

Run at the ROS 2 workspace root:

```bash
colcon build --packages-select haptic
source install/setup.bash
```

### 2) Tac3D Sensor Setup

**2.1 Check device IDs**

```bash
v4l2-ctl --list-devices
```

> 💡 Device IDs may change with USB order; verify after each power cycle.

**2.2 Start Tac3D SDK**

```bash
./Tac3D -c config/<sensor_serial> -d <device_id> -i 127.0.0.1 -p <port>
```

Parameters:
- <sensor_serial>: e.g. A1-0001R
- <device_id>: e.g. 0, 1
- <port>: left 9988, right 9989

**2.3 Start ROS publishers**

```bash
ros2 run haptic tac3d_l
ros2 run haptic tac3d_r
```

### 3) Connect KUKA (Two Common Modes)

#### A. Human Teach / Recording (Admittance)

SmartPAD:

JOINT_IMPEDANCE_CONTROL -> POSITION (choose impedance mode as needed)

```bash
ros2 launch lbr_bringup hardware.launch.py \
  ctrl:=admittance_controller \
  model:=iiwa14
```

#### B. Program Control (Joint Position)

SmartPAD:

POSITION_CONTROL -> POSITION

```bash
ros2 launch lbr_bringup hardware.launch.py \
  ctrl:=lbr_joint_position_command_controller \
  model:=iiwa14
```

If Cartesian pose interface is needed, run in another terminal:

```bash
ros2 run lbr_demos_advanced_cpp pose_control --ros-args -r __ns:=/lbr
```

### 4) Data Recording

**4.1 Text logging (all topics)**

```bash
ros2 run haptic dataset_recorder
```

Output directory (auto‑created unique subfolder):

`training_data/dataset_recorder/<timestamp_pid_uuid>/`

**4.2 HDF5 logging (fixed‑rate sampling)**

```bash
ros2 run haptic dataset_recorder_h5
```

Output file:

`training_data/h5_datasets/dataset_<timestamp_pid_uuid>.h5`

---

## 🕹️ Robot Control and Debug

### Reset to initial joint pose

```bash
ros2 run haptic reset
```

### Start Cartesian controller (policy bridge)

```bash
ros2 run haptic cartesian_controller
```

The controller subscribes to /ab_action and publishes to /lbr/command/pose.

---

### ⚡ One‑shot teaching prep flow

*Currently limited due to blocking bringup.

This package provides an integrated launch file:

```bash
ros2 launch haptic teach_prepare.py
```

Optional parameter (whether to run reset):

```bash
ros2 launch haptic teach_prepare.py reset:=true
ros2 launch haptic teach_prepare.py reset:=false
```

This flow starts, in order:
- Admittance control process
- Tac3D SDK services (left/right)
- tac3d_l / tac3d_r nodes
- Optional reset

---

## 📁 Key Files

- haptic/tac3d_l.py / haptic/tac3d_r.py: Tac3D data publishers
- haptic/dataset_recorder.py: text data logger
- haptic/dataset_recorder_h5.py: HDF5 logger
- haptic/cartesian_controller.py: Cartesian interpolation & safety
- haptic/reset.py: reset to initial pose
- launch/teach_prepare.py: one‑shot teaching prep

---

## 📚 References

- LBR documentation (advanced demos):
  https://lbr-stack.readthedocs.io/en/latest/lbr_fri_ros2_stack/lbr_demos/lbr_demos_advanced_py/doc/lbr_demos_advanced_py.html

---

If this project helps you, feel free to ⭐️ Star.
