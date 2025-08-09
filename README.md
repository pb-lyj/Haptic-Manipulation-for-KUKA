# haptic  ros2 package

This ROS package is designed for tactile robotics experiments on a KUKA robot. 
# Dependence
Its main dependencies include:
- ubuntu 24.04
- ros2-jazzy
- python 3.12

- FRI
- Gazebo
	```
	sudo apt update
	sudo apt install ros-jazzy-gz-gazebo
	sudo apt install ros-jazzy-ros-gz-sim
	sudo apt install ros-jazzy-ros-gz
	```
- LBR

	https://github.com/lbr-stack/lbr_fri_ros2_stack

# Function
## Robotic arm movement
- Connect KUKA
	- https://lbr-stack.readthedocs.io/en/latest/lbr_fri_ros2_stack/lbr_demos/lbr_demos_advanced_py/doc/lbr_demos_advanced_py.html
```
	ros2 launch lbr_bringup hardware.launch.py \
    ctrl:=lbr_joint_position_command_controller \
    model:=iiwa14 # [iiwa7, iiwa14, med7, med14]
```
```
ros2 run lbr_demos_advanced_py admittance_control --ros-args   -r __ns:=/lbr   --params-file $(ros2 pkg prefix lbr_demos_advanced_py)/share/lbr_demos_advanced_py/config/admittance_control.yaml
```
:red_circle: Executing this command may cause the robotic arm to enter impedance mode
This will start the robotic arm information broadcast at the same time.

- Moving the robotic arm to its original position

```
	ros2 run haptic reset
```

## Tac3D Sensor

- Confirm the Serial number of Sensor

	:black_circle: The camera serial number may be related to the time and location of the USB device plugged in, and we recommend that you check it in after it is plugged in
	```
	v4l2-ctl --list-devices
	```
- Start the SDK
	```
	./Tac3D -c config/sensor_serial_number -d serial_number -i 127.0.0.1 -p port
	```
	- Sensor serial number : eg. A1-0001R
	- Serial_number : eg. 0 1 
	- Port : Set one , default = 9988(left), 9989(right)

- Sensor information broadcasting Node
	```
	ros2 run haptic tac3d_l
	```

	```
	ros2 run haptic tac3d_r
	```

## Human Teach and Data record

After the robotic arm is connected and the tactile sensor is activated, you can run this file to record all the data they broadcast.

```
ros2 run haptic dataset_recorder
```

## *Launch all of the above procedures

1. connect the robot

```
ros2 launch haptic teach_record.py
```
With the hand controller, this command will connect the robotic arm in wrench mode
- please wait for the haptic sensor to be calibrated
- Assist the robotic arm to cope with the remaining dead weight compensation error

- Using Java native impedance here, we set a high three-axis rotational impedance in the experiment in July 2025, with a certain z-axis impedance and an XY impedance of almost 0 to facilitate our on-plane jack task

2. start data recording

```
ros2 run haptic dataset_recorder
```

That will start data recording immediately.
- The specific type and format of the recorded data can be viewed in `data_recorder_improved.py`.
- The txt file contains the raw data and the timestamp when the data arrived at the host computer