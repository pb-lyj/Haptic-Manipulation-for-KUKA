"""
笛卡尔控制器节点 - 订阅ab_action并插值发布到pose_control
接收来自LSTM策略的位姿命令，进行线性插值和安全限制后发布
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
import numpy as np


class CartesianControllerNode(Node):
    """笛卡尔控制器 - 插值和安全限制"""
    
    def __init__(self):
        super().__init__('cartesian_controller_node')
        
        # 当前状态
        self.current_pose = None
        self.target_pose = None
        self.is_init = False
        self.has_target = False  # 跟踪是否收到过目标命令
        
        # 插值参数（参考pose_planning）
        self.declare_parameter('max_step_size', 0.005)  # 每步最大移动距离 (m)
        self.declare_parameter('control_rate', 100.0)   # 控制频率 (Hz)
        
        # 工作空间限制（KUKA iiwa14实际工作空间）
        self.declare_parameter('workspace_x_min', -0.85)
        self.declare_parameter('workspace_x_max', 0.85)
        self.declare_parameter('workspace_y_min', -0.85)
        self.declare_parameter('workspace_y_max', 0.85)
        self.declare_parameter('workspace_z_min', 0.15)
        self.declare_parameter('workspace_z_max', 1.30)
        
        self.max_step_size = self.get_parameter('max_step_size').value
        self.control_rate = self.get_parameter('control_rate').value
        self.workspace_limits = {
            'x': [self.get_parameter('workspace_x_min').value, 
                  self.get_parameter('workspace_x_max').value],
            'y': [self.get_parameter('workspace_y_min').value, 
                  self.get_parameter('workspace_y_max').value],
            'z': [self.get_parameter('workspace_z_min').value, 
                  self.get_parameter('workspace_z_max').value],
        }
        
        # 订阅ab_action（来自lstm_policy）
        self.action_sub = self.create_subscription(
            Pose, '/ab_action', self.action_callback, 10)
        
        # 订阅当前位姿（用于插值）
        self.pose_sub = self.create_subscription(
            Pose, '/lbr/state/pose', self.pose_callback, 10)
        
        # 发布位姿命令
        self.pose_pub = self.create_publisher(Pose, '/lbr/command/pose', 10)
        
        # 创建控制定时器
        self.timer = self.create_timer(1.0 / self.control_rate, self.control_callback)
        
        self.get_logger().info("笛卡尔控制器节点启动")
        self.get_logger().info(f"控制频率: {self.control_rate}Hz, 最大步长: {self.max_step_size}m")
        self.get_logger().info(f"插值密度: {1.0/self.max_step_size:.0f}点/米, 时间步: {1000.0/self.control_rate:.1f}ms")
        self.get_logger().info("⚠️  等待 /ab_action 命令才会开始控制")
    
    def pose_callback(self, msg):
        """当前位姿回调 - 用于初始化和插值计算"""
        if not self.is_init:
            self.current_pose = msg
            self.is_init = True
            self.get_logger().info(
                f"初始位姿: [{msg.position.x:.3f}, {msg.position.y:.3f}, {msg.position.z:.3f}]")
        else:
            self.current_pose = msg
    
    def action_callback(self, msg):
        """接收新的目标位姿（来自lstm_policy的ab_action）"""
        if not self.is_init:
            return
        
        # 工作空间安全检查
        target_clipped = self._clip_to_workspace(msg)
        self.target_pose = target_clipped
        
        # ✅ 标记已收到目标
        if not self.has_target:
            self.has_target = True
            self.get_logger().info("✅ 收到第一个目标命令，开始控制")
    
    def control_callback(self):
        """控制循环 - 线性插值朝目标移动（位置+姿态）"""
        # ✅ 关键检查：没有收到目标就不发布任何命令
        if not self.is_init or not self.has_target:
            return
        
        if self.current_pose is None or self.target_pose is None:
            return
        
        # 提取当前位姿和目标位姿的7维向量 [x, y, z, qx, qy, qz, qw]
        current_vec = np.array([
            self.current_pose.position.x,
            self.current_pose.position.y,
            self.current_pose.position.z,
            self.current_pose.orientation.x,
            self.current_pose.orientation.y,
            self.current_pose.orientation.z,
            self.current_pose.orientation.w
        ])
        target_vec = np.array([
            self.target_pose.position.x,
            self.target_pose.position.y,
            self.target_pose.position.z,
            self.target_pose.orientation.x,
            self.target_pose.orientation.y,
            self.target_pose.orientation.z,
            self.target_pose.orientation.w
        ])
        
        # 计算差值和距离（仅考虑位置部分的距离）
        delta = target_vec - current_vec
        position_distance = np.linalg.norm(delta[:3])
        
        # 如果已经到达目标，不发布命令
        if position_distance < 1e-5:
            return
        
        # 线性插值：每步最多移动max_step_size
        if position_distance > self.max_step_size:
            # 按比例缩放所有7个分量
            interpolation_ratio = self.max_step_size / position_distance
            step = delta * interpolation_ratio
        else:
            step = delta
        
        # 计算新的命令位姿
        new_vec = current_vec + step
        
        # 创建命令消息
        command_pose = Pose()
        command_pose.position.x = float(new_vec[0])
        command_pose.position.y = float(new_vec[1])
        command_pose.position.z = float(new_vec[2])
        command_pose.orientation.x = float(new_vec[3])
        command_pose.orientation.y = float(new_vec[4])
        command_pose.orientation.z = float(new_vec[5])
        command_pose.orientation.w = float(new_vec[6])
        
        # 发布命令
        self.pose_pub.publish(command_pose)
    
    def _clip_to_workspace(self, pose):
        """将位姿裁剪到工作空间内"""
        clipped_pose = Pose()
        clipped_pose.position.x = np.clip(
            pose.position.x, 
            self.workspace_limits['x'][0], 
            self.workspace_limits['x'][1])
        clipped_pose.position.y = np.clip(
            pose.position.y,
            self.workspace_limits['y'][0], 
            self.workspace_limits['y'][1])
        clipped_pose.position.z = np.clip(
            pose.position.z,
            self.workspace_limits['z'][0], 
            self.workspace_limits['z'][1])
        clipped_pose.orientation = pose.orientation
        
        # 检查是否发生裁剪
        if (abs(clipped_pose.position.x - pose.position.x) > 1e-6 or
            abs(clipped_pose.position.y - pose.position.y) > 1e-6 or
            abs(clipped_pose.position.z - pose.position.z) > 1e-6):
            self.get_logger().warn(
                f"位姿被裁剪: [{pose.position.x:.3f}, {pose.position.y:.3f}, {pose.position.z:.3f}] "
                f"-> [{clipped_pose.position.x:.3f}, {clipped_pose.position.y:.3f}, {clipped_pose.position.z:.3f}]")
        
        return clipped_pose
    
    def destroy_node(self):
        """销毁节点"""
        self.get_logger().info("正在停止笛卡尔控制器节点...")
        super().destroy_node()


def main():
    """主函数"""
    print("=== 笛卡尔控制器节点 ===")
    print("订阅: /ab_action (Pose) - LSTM策略输出")
    print("订阅: /lbr/state/pose (Pose) - 当前位姿")
    print("发布: /lbr/command/pose (Pose) - 插值后的位姿命令")
    print("⚠️  只有在收到 /ab_action 后才会发布控制命令")
    print()
    
    rclpy.init()
    
    node = None
    try:
        node = CartesianControllerNode()
        print("控制器节点已启动，等待位姿初始化...")
        print("按 Ctrl+C 停止")
        print()
        
        rclpy.spin(node)
        
    except KeyboardInterrupt:
        print("\n收到停止信号...")
    except Exception as e:
        print(f"\n运行时发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if node:
            try:
                node.destroy_node()
            except Exception as e:
                print(f"清理节点时发生错误: {e}")
        
        try:
            rclpy.shutdown()
        except:
            pass
        
        print("程序已退出")


if __name__ == '__main__':
    main()