# 🖐️ haptic (ROS 2)

用于 **KUKA LBR + Tac3D 触觉传感器** 的 ROS 2 实验包，面向以下场景：
- 人工示教与触觉数据采集
- 触觉/力觉信息广播与同步记录
- 基于策略输出的笛卡尔位姿控制（含安全阈值）

---

## ✨ 核心功能

- 🤖 **KUKA 机械臂接入**（基于 lbr_fri_ros2_stack）
- 🧠 **Tac3D 双传感器数据发布**（左/右传感器）
  - 3D positions / displacements / forces（sensor_msgs/Image, 32FC3）
  - resultant force / resultant moment（geometry_msgs/Vector3）
- 🗂️ **数据记录**
  - dataset_recorder：多话题原始文本记录（带时间戳）
  - dataset_recorder_h5：固定频率采样并写入 HDF5
- 🎯 **笛卡尔控制器**（cartesian_controller）
  - 接收策略输出位姿目标
  - 插值限速 + 工作空间裁剪 + 力/力矩阈值保护
- 🔁 **一键流程编排**（launch/teach_prepare.py）
  - 启动导纳控制、Tac3D SDK、传感器 ROS 节点，可选自动 reset

---

## 🧱 环境依赖

- Ubuntu 24.04
- ROS 2 Jazzy
- Python 3.12
- KUKA FRI + LBR ROS 2 驱动
  - https://github.com/lbr-stack/lbr_fri_ros2_stack

可选仿真依赖（Gazebo）：

```bash
sudo apt update
sudo apt install ros-jazzy-gz-gazebo
sudo apt install ros-jazzy-ros-gz-sim
sudo apt install ros-jazzy-ros-gz
```

---

## 🚀 快速开始

> ⚠️ **安全提示**
> 
> 涉及真实机械臂运动，请确保急停可用、工作空间清空、初始姿态安全。

### 1) 编译工作区

在 ROS 2 工作区根目录执行：

```bash
colcon build --packages-select haptic
source install/setup.bash
```

### 2) Tac3D 传感器使用

**2.1 检查设备编号**

```bash
v4l2-ctl --list-devices
```

> 💡 设备编号可能随 USB 插拔顺序变化，建议每次上电后确认。

**2.2 启动 Tac3D SDK**

```bash
./Tac3D -c config/<sensor_serial> -d <device_id> -i 127.0.0.1 -p <port>
```

参数示例：
- <sensor_serial>：如 A1-0001R
- <device_id>：如 0、1
- <port>：建议左 9988，右 9989

**2.3 启动 ROS 发布节点**

```bash
ros2 run haptic tac3d_l
ros2 run haptic tac3d_r
```

### 3) 连接 KUKA（两种常用模式）

#### A. 人工示教 / 采集模式（导纳）

SmartPAD 参考：

JOINT_IMPEDANCE_CONTROL -> POSITION（或按实验需求选择对应阻抗模式）

```bash
ros2 launch lbr_bringup hardware.launch.py \
  ctrl:=admittance_controller \
  model:=iiwa14
```

#### B. 程序控制模式（关节位置）

SmartPAD 参考：

POSITION_CONTROL -> POSITION

```bash
ros2 launch lbr_bringup hardware.launch.py \
  ctrl:=lbr_joint_position_command_controller \
  model:=iiwa14
```

如需笛卡尔位姿接口，再开一个终端：

```bash
ros2 run lbr_demos_advanced_cpp pose_control --ros-args -r __ns:=/lbr
```

### 4) 数据记录

**4.1 文本记录（全量话题）**

```bash
ros2 run haptic dataset_recorder
```

输出目录（自动创建唯一子目录）：

training_data/dataset_recorder/<timestamp_pid_uuid>/

**4.2 HDF5 记录（固定频率采样）**

```bash
ros2 run haptic dataset_recorder_h5
```

输出文件：

training_data/h5_datasets/dataset_<timestamp_pid_uuid>.h5

---

## 🕹️ 机械臂控制与调试

### 回到初始关节位姿

```bash
ros2 run haptic reset
```

### 启动笛卡尔控制器（策略桥接）

```bash
ros2 run haptic cartesian_controller
```

控制器订阅策略位姿（/ab_action），并发布到：

/lbr/command/pose

---

### ⚡ 一键启动示教准备流程

*受限于 bringup 的阻塞暂未实现

本包提供了整合启动文件：

```bash
ros2 launch haptic teach_prepare.py
```

可选参数（是否执行 reset）：

```bash
ros2 launch haptic teach_prepare.py reset:=true
ros2 launch haptic teach_prepare.py reset:=false
```

该流程会依次启动：
- 导纳控制进程
- Tac3D SDK 左/右服务
- tac3d_l / tac3d_r 节点
- （可选）reset

---

## 📁 关键文件

- haptic/tac3d_l.py / haptic/tac3d_r.py：Tac3D 数据发布
- haptic/dataset_recorder.py：文本数据记录
- haptic/dataset_recorder_h5.py：HDF5 数据记录
- haptic/cartesian_controller.py：笛卡尔插值控制与安全保护
- haptic/reset.py：回零/初始位姿恢复
- launch/teach_prepare.py：示教准备一键启动

---

## 📚 参考

- LBR 官方文档（advanced demos）：
  https://lbr-stack.readthedocs.io/en/latest/lbr_fri_ros2_stack/lbr_demos/lbr_demos_advanced_py/doc/lbr_demos_advanced_py.html

---

如果这个项目对你有帮助，欢迎 ⭐️ Star。
