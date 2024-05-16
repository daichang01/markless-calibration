import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import message_filters
import numpy as np
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from std_msgs.msg import Header
from sensor_msgs.msg import PointField
import struct

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        self.bridge = CvBridge()
        self.color_sub = message_filters.Subscriber(self, Image, '/camera/camera/color/image_rect_raw')
        self.depth_sub = message_filters.Subscriber(self, Image, '/camera/camera/aligned_depth_to_color/image_raw')
        self.publisher = self.create_publisher(PointCloud2, 'seg_point_cloud', 10)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.color_sub, self.depth_sub], 10, 0.1)
        self.ts.registerCallback(self.callback)

    def callback(self, color_msg, depth_msg):
        # print(f"Received color image of shape: {color_msg.height}x{color_msg.width}")
        # print(f"Received depth image of shape: {depth_msg.height}x{depth_msg.width}")
        cv_color_image = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
        cv_depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")

        height, width, channels = cv_color_image.shape
        center_x, center_y = width // 2, height // 2
        half_width, half_height = 150, 150  # 因为我们要裁剪300x300区域
        start_x, end_x = center_x - half_width, center_x + half_width
        start_y, end_y = center_y - half_height, center_y + half_height


        ######测试demo，裁减图像中间300*300区域生成点云

        points = []
        # height, width, channels = cv_color_image.shape
        for v in range(start_y, end_y):
            for u in range(start_x, end_x):
                depth = cv_depth_image[v, u]
                if depth > 0:  # Simple depth filter to remove zero depth values
                    # 这里的内参需要根据实际相机调整
                    z = depth * 0.001  # scale depth to meters
                    x = (u - 425.98785400390625) * z / 425.2796325683594
                    y = (v - 241.7391357421875) * z / 425.2796325683594
                    b, g, r = cv_color_image[v, u].astype(np.uint8)
                    # points.append([x, y, z, r, g, b])
                    rgb = struct.pack('BBBB', r, g, b, 255)  # 封装BGR到一个uint32中
                    rgb = struct.unpack('I', rgb)[0]
                    points.append([x, y, z, rgb])

        # Create PointCloud2 message
        header = Header(frame_id='camera_link', stamp=self.get_clock().now().to_msg())
        fields = [PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1)]
        point_cloud_msg = pc2.create_cloud(header, fields, points)
        self.publisher.publish(point_cloud_msg)
        print("Published Point Cloud")

        # 归一化深度图像以增强显示效果
        cv_image_normalized = cv2.normalize(cv_depth_image, None, 0, 255, cv2.NORM_MINMAX)
        cv_image_normalized = np.uint8(cv_image_normalized)  # 转换为8位图像
        # 显示处理后的深度图像
        cv2.imshow("Color Image", cv_color_image)
        cv2.imshow("Depth Image", cv_image_normalized)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    image_subscriber = ImageSubscriber()
    rclpy.spin(image_subscriber)
    image_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
