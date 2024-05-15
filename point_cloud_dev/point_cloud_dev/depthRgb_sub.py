import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import message_filters
import numpy as np

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        self.bridge = CvBridge()
        self.color_sub = message_filters.Subscriber(self, Image, '/camera/camera/color/image_rect_raw')
        self.depth_sub = message_filters.Subscriber(self, Image, '/camera/camera/aligned_depth_to_color/image_raw')
        self.ts = message_filters.ApproximateTimeSynchronizer([self.color_sub, self.depth_sub], 10, 0.1)
        self.ts.registerCallback(self.callback)

    def callback(self, color_msg, depth_msg):
        # print(f"Received color image of shape: {color_msg.height}x{color_msg.width}")
        # print(f"Received depth image of shape: {depth_msg.height}x{depth_msg.width}")
        cv_color_image = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
        cv_depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")

        # 归一化深度图像以增强显示效果
        cv_image_normalized = cv2.normalize(cv_depth_image, None, 0, 255, cv2.NORM_MINMAX)
        cv_image_normalized = np.uint8(cv_image_normalized)  # 转换为8位图像
        # 显示处理后的深度图像
        cv2.imshow("Color Image", cv_color_image)
        cv2.imshow("Depth Image", cv_image_normalized)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    image_subscriber = ImageSubscriber()
    rclpy.spin(image_subscriber)
    image_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
