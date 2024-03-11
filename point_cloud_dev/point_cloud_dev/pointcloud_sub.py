import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as pc2
import struct
import numpy as np
import open3d as o3d


pcdPath = "src/markless-calibration/pcd/output_point_cloud.ply"

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
        super().__init__('pointcloud_sub')
        self.subscription = self.create_subscription(
            PointCloud2,
            '/camera/camera/depth/color/points',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.points = []
        self.colors = []
        

    def listener_callback(self, msg):

        # 解析点云数据
        point_list = list(pc2.read_points_list(msg, field_names=("x", "y", "z", "rgb"),skip_nans= True))

        temp_points = []
        temp_colors = []

        for point in point_list:
            x, y, z, rgb = point
            r, g, b = parse_rgb_float(rgb)
            temp_points.append([x, y, z])
            # 颜色值需要从[0, 255]范围转换到[0, 1]范围
            temp_colors.append([r / 255.0, g / 255.0 , b / 255.0])
        
        self.points = temp_points
        self.colors = temp_colors

        if len(self.points) > 10000:
            self.save_point_cloud_to_ply(pcdPath)
            self.get_logger().info(f'Received {len(self.points)} points to output_point_cloud.ply')
            self.points = []
            self.colors = []



        # self.get_logger().info(f"Fields: {msg.fields}")
        
    
    def save_point_cloud_to_ply(self, filename):
        # 创建一个空的点云对象
        point_cloud = o3d.geometry.PointCloud()
        # 设置点云和颜色
        point_cloud.points = o3d.utility.Vector3dVector(self.points)
        point_cloud.colors = o3d.utility.Vector3dVector(self.colors)
        # 保存点云到PLY文件
        o3d.io.write_point_cloud(filename, point_cloud)
        self.get_logger().info(f"Received {len(self.points)} points saved to {filename}")


def main(args=None):
    rclpy.init(args=args)
    pointcloud_subscriber = PointCloud2Subscriber()
    rclpy.spin(pointcloud_subscriber)
    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    pointcloud_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
