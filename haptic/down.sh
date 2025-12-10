#!/bin/bash
# 获取当前位姿并修改XYZ值后发布
# chmod +x down.sh

# 从命令行参数获取XYZ变化量，如果没有提供则使用默认值
DELTA_X=${1:-0.0}
DELTA_Y=${2:-0.0}
DELTA_Z=${3:--0.05}

# 获取当前位姿
echo "获取当前位姿..."
POSE=$(ros2 topic echo /lbr/state/pose --once)

# 提取XYZ坐标（修复grep正则）
X=$(echo "$POSE" | grep "^\s*x:" | head -1 | awk '{print $2}')
Y=$(echo "$POSE" | grep "^\s*y:" | head -1 | awk '{print $2}')
Z=$(echo "$POSE" | grep "^\s*z:" | head -1 | awk '{print $2}')

# 强制使用固定的四元数 (x=0, y=1, z=0, w=0)
QX=0.0
QY=1.0
QZ=0.0
QW=0.0

echo "当前位置: X=$X, Y=$Y, Z=$Z"

# 检查是否成功获取位姿
if [ -z "$X" ] || [ -z "$Y" ] || [ -z "$Z" ]; then
    echo "错误：无法获取当前位姿！"
    exit 1
fi

# 计算新的XYZ坐标
TARGET_X=$(echo "$X + $DELTA_X" | bc)
TARGET_Y=$(echo "$Y + $DELTA_Y" | bc)
TARGET_Z=$(echo "$Z + $DELTA_Z" | bc)
echo "XYZ变化量: [$DELTA_X, $DELTA_Y, $DELTA_Z]"
echo "目标位置: X=$TARGET_X, Y=$TARGET_Y, Z=$TARGET_Z"

# 发布到/ab_action，由cartesian_controller插值后发送
ros2 topic pub --once /ab_action geometry_msgs/msg/Pose \
"{
  position: {x: $TARGET_X, y: $TARGET_Y, z: $TARGET_Z},
  orientation: {x: $QX, y: $QY, z: $QZ, w: $QW}
}"

echo "位姿命令已发布到/ab_action (将由cartesian_controller插值)！"