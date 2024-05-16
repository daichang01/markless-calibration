import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import struct
import numpy as np
import open3d as o3d

def parse_rgb_float(rgb_float):
    # 将float32编码的rgb值转换为整数
    rgb_int = struct.unpack('I', struct.pack('f', rgb_float))[0]
    # 按位提取rgb值
    red = (rgb_int >> 16) & 0x0000ff
    green = (rgb_int >> 8) & 0x0000ff
    blue = (rgb_int) & 0x0000ff
    return (red, green, blue)

class PointCloud2Subscriber(Node):
    def __init__(self):
        super().__init__('pcd_sub')
        self.subscription = self.create_subscription(
            PointCloud2,
            '/camera/camera/depth/color/points',
            self.listener_callback,
            10)
        self.subscription  # 防止未使用变量警告
        self.points = []
        self.colors = []
        self.save_index = 80  # 初始化文件索引计数器
        
        # 使用可重入回调组，以便同时处理定时器和订阅者回调
        self.callback_group = ReentrantCallbackGroup()
        # 设置一个计时器，每3秒调用一次保存点云的函数
        self.timer = self.create_timer(3.0, self.timer_callback, callback_group=self.callback_group)

    def listener_callback(self, msg):
        # 解析点云数据
        point_list = list(pc2.read_points_list(msg, field_names=("x", "y", "z", "rgb"), skip_nans=True))

        temp_points = []
        temp_colors = []

        for point in point_list:
            x, y, z, rgb = point
            r, g, b = parse_rgb_float(rgb)
            temp_points.append([x, y, z])
            # 颜色值需要从[0, 255]范围转换到[0, 1]范围
            temp_colors.append([r / 255.0, g / 255.0, b / 255.0])
        
        self.points = temp_points
        self.colors = temp_colors

    def timer_callback(self):
        # 定时保存点云数据
        if self.points:
            filename = f"src/markless-calibration/pcd/01origin_data/ori{self.save_index}.txt"
            self.save_point_cloud_to_txt(filename)
            self.get_logger().info(f'已保存 {len(self.points)} 个点到文件 {filename}')
            self.save_index += 1  # 更新文件序号
            # 清空点列表和颜色列表以准备下一个点云
            self.points = []
            self.colors = []
    
    def save_point_cloud_to_txt(self, filename):
        # 保存点云到TXT文件
        with open(filename, 'w') as file:
            for point, color in zip(self.points, self.colors):
                r, g, b = [int(c * 255) for c in color]
                file.write(f"{point[0]} {point[1]} {point[2]} {r} {g} {b}\n")
        self.get_logger().info(f"点云已保存至 {filename}")

def main(args=None):
    rclpy.init(args=args)
    pointcloud_subscriber = PointCloud2Subscriber()
    rclpy.spin(pointcloud_subscriber)
    pointcloud_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
