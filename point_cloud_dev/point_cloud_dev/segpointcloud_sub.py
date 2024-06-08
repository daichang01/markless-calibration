import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import struct
import numpy as np
import open3d as o3d
from datetime import datetime
import time




class PointCloud2Subscriber(Node):
    def __init__(self):
        super().__init__('pcd_sub')
        # self.upsubscription = self.create_subscription(PointCloud2,  'upteeth_point_cloud', self.uplistener_callback, 10)
        self.lowsubscription = self.create_subscription(PointCloud2,  'lowfront_point_cloud', self.lowlistener_callback, 10)
        self.upsubscription = self.create_subscription(PointCloud2,  'roi_point_cloud', self.uplistener_callback, 10)
        # self.subscription  # 防止未使用变量警告
        self.uppoints = []
        self.upcolors = []
        self.lowpoints = []
        self.lowcolors = []
        
        # 使用可重入回调组，以便同时处理定时器和订阅者回调
        self.callback_group = ReentrantCallbackGroup()
        # 设置一个计时器，每3秒调用一次保存点云的函数
        self.timerup = self.create_timer(3.0, self.uptimer_callback, callback_group=self.callback_group)
        self.timerlow = self.create_timer(3.0, self.lowtimer_callback, callback_group=self.callback_group)

    def uplistener_callback(self, msg):
        # 解析点云数据
        point_list = list(pc2.read_points_list(msg, field_names=("x", "y", "z", "rgb"), skip_nans=True))

        temp_points = []
        temp_colors = []

        for point in point_list:
            x, y, z, rgb = point
            # r, g, b = self.parse_rgb_float(rgb)
            rgb_int = np.array([rgb], dtype=np.uint32)
            r = (rgb_int >> 16) & 0x0000ff
            g = (rgb_int >> 8) & 0x0000ff
            b = rgb_int & 0x0000ff
            # print(f"Extracted RGB values: R={r}, G={g}, B={b}")
            temp_points.append([x, y, z])
            # 颜色值需要从[0, 255]范围转换到[0, 1]范围
            temp_colors.append([r / 255.0, g / 255.0, b / 255.0])
        
        self.uppoints = temp_points
        self.upcolors = temp_colors

    def lowlistener_callback(self, msg):
        # 解析点云数据
        point_list = list(pc2.read_points_list(msg, field_names=("x", "y", "z", "rgb"), skip_nans=True))

        temp_points = []
        temp_colors = []

        for point in point_list:
            x, y, z, rgb = point
            # r, g, b = self.parse_rgb_float(rgb)
            rgb_int = np.array([rgb], dtype=np.uint32)
            r = (rgb_int >> 16) & 0x0000ff
            g = (rgb_int >> 8) & 0x0000ff
            b = rgb_int & 0x0000ff
            # print(f"Extracted RGB values: R={r}, G={g}, B={b}")
            temp_points.append([x, y, z])
            # 颜色值需要从[0, 255]范围转换到[0, 1]范围
            temp_colors.append([r / 255.0, g / 255.0, b / 255.0])
        
        self.lowpoints = temp_points
        self.lowcolors = temp_colors

        

    def uptimer_callback(self):
        # 定时保存点云数据
        if self.uppoints:
            current_time = datetime.now().strftime("%m%d_%H%M%S")
            # filename = f"src/markless-calibration/pcd/auto-seg/upteeth/upseg{current_time}.txt"
            filename = f"src/markless-calibration/pcd/auto-seg/valteeth/roi{current_time}.txt"
            self.save_uppoint_cloud_to_txt(filename)
            # 清空点列表和颜色列表以准备下一个点云
            self.uppoints = []
            self.upcolors = []
    
    def lowtimer_callback(self):
        # 定时保存点云数据
        if self.lowpoints:
            current_time = datetime.now().strftime("%m%d_%H%M%S")
            filename = f"src/markless-calibration/pcd/auto-seg/lowteeth/lowseg{current_time}.txt"
            self.save_lowpoint_cloud_to_txt(filename)
            # 清空点列表和颜色列表以准备下一个点云
            self.lowpoints = []
            self.lowcolors = []
    
    def save_uppoint_cloud_to_txt(self, filename):
        point_count = 0 
        # 保存点云到TXT文件
        try:
            with open(filename, 'w') as file:
                for point, color in zip(self.uppoints, self.upcolors):
                    r, g, b = [int(c * 255) for c in color]
                    # if max(r, g, b) - min(r, g, b) < 80:
                    file.write(f"{point[0]} {point[1]} {point[2]} {r} {g} {b}\n")
                    point_count += 1 
            self.get_logger().info(f"{point_count} 个点已保存至 {filename}")
            self.process_and_save_outliers(filename, filename.replace("roi", "roifilter"))
        except Exception as e:
            self.get_logger().error(f"Failed to save point cloud to {filename}: {str(e)}")

    def save_lowpoint_cloud_to_txt(self, filename):
        start_time = time.time()  # 记录函数开始执行的时间
        point_count = 0 
        # 保存点云到TXT文件
        try:
            with open(filename, 'w') as file:
                for point, color in zip(self.lowpoints, self.lowcolors):  
                    r, g, b = [int(c * 255) for c in color]
                    # if max(r, g, b) - min(r, g, b) < 80:
                    file.write(f"{point[0]} {point[1]} {point[2]} {r} {g} {b}\n") 
                    point_count += 1
            self.get_logger().info(f"{point_count} 个点已保存至 {filename}")
            self.process_and_save_outliers(filename, filename.replace("lowseg", "lowsegfilter"))
        except Exception as e:
            self.get_logger().error(f"Failed to save point cloud to {filename}: {str(e)}")
        finally:
            end_time = time.time()  # 记录函数执行完毕的时间
            elapsed_time = (end_time - start_time) * 1000  # 计算总耗时
            self.get_logger().info(f"Total execution time: {elapsed_time:.2f} milliseconds")  # 输出总耗时

    def process_and_save_outliers(self, input_filename, output_filename, nb_neighbors=20, std_ratio=1.5):
        try:
            # 读取点云数据
            points = []
            colors = []
            with open(input_filename, 'r') as file:
                for line in file:
                    parts = line.strip().split()
                    points.append([float(parts[0]), float(parts[1]), float(parts[2])])
                    colors.append([int(parts[3])/255.0, int(parts[4])/255.0, int(parts[5])/255.0])

            # 创建 Open3D 点云对象
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(np.array(points))
            pc.colors = o3d.utility.Vector3dVector(np.array(colors))

            # 去除离群值
            # 指定用于计算每个点的局部密度的邻居的数量。具体来说，对于每个点，算法将计算它与最近的 nb_neighbors 个点的平均距离。
            # 这个参数用于设定识别离群值的敏感度。具体操作是，首先计算每个点与其 nb_neighbors 的平均距离，然后计算所有点的这些距离的平均值和标准偏差。点的平均距离超过 平均距离 + std_ratio × 标准偏差 的将被认为是离群值。
            filtered_pc, ind = pc.remove_statistical_outlier(nb_neighbors, std_ratio)

            # 保存处理后的点云数据
            filtered_points = np.asarray(filtered_pc.points)
            filtered_colors = np.asarray(filtered_pc.colors)
            point_count = 0
            with open(output_filename, 'w') as file:
                for point, color in zip(filtered_points, filtered_colors):
                    r, g, b = [int(c * 255) for c in color]
                    file.write(f"{point[0]} {point[1]} {point[2]} {r} {g} {b}\n")
                    point_count += 1 
            
            self.get_logger().info(f"处理后 {output_filename},Total points saved: {point_count}")
        except Exception as e:
            self.get_logger().info(f"Failed to process and save point cloud: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    pointcloud_subscriber = PointCloud2Subscriber()
    rclpy.spin(pointcloud_subscriber)
    pointcloud_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
