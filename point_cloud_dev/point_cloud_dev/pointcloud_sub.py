import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as pc2
import struct
import numpy as np
import open3d as o3d


pcdPlyPath = "src/markless-calibration/pcd/out_pcd.ply"
pcdTxtPath = "src/markless-calibration/pcd/out_pcd.txt"


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
            self.save_point_cloud_to_ply(pcdPlyPath)
            self.save_point_cloud_to_txt(pcdTxtPath)
            self.get_logger().info(f'Received {len(self.points)} points to file')
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
    
    def save_point_cloud_to_txt(self, filename):
        # 保存点云到TXT文件
        with open(filename, 'w') as file:
            for point, color in zip(self.points, self.colors):
                # 将颜色值从[0, 1]范围转换回[0, 255]范围，并转换为整数
                r, g, b = [int(c * 255) for c in color]
                # 格式化字符串以包含点的坐标和颜色
                file.write(f"{point[0]} {point[1]} {point[2]} {r} {g} {b}\n")
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
