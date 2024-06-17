import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np

class PointCloudPublisher(Node):
    def __init__(self):
        super().__init__('pointcloud_publisher')
        self.publisher = self.create_publisher(PointCloud2, 'scanpointcloud', 10)
        self.timer = self.create_timer(1.0, self.timer_callback)
        self.pcd = self.load_pointcloud_from_txt('/home/daichang/Desktop/ros2_ws/src/markless-calibration/pcd/wait-to-reg/newteeth_m_uniform_down.txt')

    def load_pointcloud_from_txt(self, file_path):
        points = np.loadtxt(file_path, delimiter=' ')
        return points

    def timer_callback(self):
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'camera_infra1_optical_frame'  # Specify your frame ID here

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        # Convert the point cloud to a list of tuples
        points = [tuple(point) for point in self.pcd]

        # Create the PointCloud2 message
        pointcloud_msg = pc2.create_cloud(header, fields, points)
        self.publisher.publish(pointcloud_msg)
        self.get_logger().info('Publishing point cloud')

def main(args=None):
    rclpy.init(args=args)
    pointcloud_publisher = PointCloudPublisher()
    rclpy.spin(pointcloud_publisher)
    pointcloud_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
