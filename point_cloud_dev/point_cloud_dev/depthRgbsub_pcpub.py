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
import os
import torch
from datetime import datetime

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        self.bridge = CvBridge()
        self.color_sub = message_filters.Subscriber(self, Image, '/camera/camera/color/image_rect_raw')
        self.depth_sub = message_filters.Subscriber(self, Image, '/camera/camera/aligned_depth_to_color/image_raw')
        self.publisher = self.create_publisher(PointCloud2, 'seg_point_cloud', 10)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.color_sub, self.depth_sub], 10, 0.1)
        self.ts.registerCallback(self.callback)

        self.latest_color_image = None
        self.latest_depth_image = None
        self.image_index = 0
        self.image_folder = "src/markless-calibration/image"  # 路径需要根据你的文件系统进行修改

##################  采集RGB和深度图并保存,用于yolo训练  ####################################################################
        self.timer = self.create_timer(2.0, self.save_images)

##################   yolo集成，用于加载模型 ########################################################################################
        # self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        # self.model.eval()
        # self.class_names = self.model.names  # 从模型自动获取类别名称


    def save_images(self):
        if self.latest_color_image is not None and self.latest_depth_image is not None:
            current_time = datetime.now().strftime("%m%d_%H%M%S")
            color_filename = os.path.join(self.image_folder, f"color_{current_time}.png")
            depth_filename = os.path.join(self.image_folder, f"depth_{current_time}.png")

            # 保存彩色图像
            cv2.imwrite(color_filename, self.latest_color_image)
            # 保存深度图像，先进行归一化处理
            # depth_normalized = cv2.normalize(self.latest_depth_image, None, 0, 255, cv2.NORM_MINMAX)
            # depth_normalized = np.uint8(depth_normalized)
            # cv2.imwrite(depth_filename, depth_normalized)

            self.get_logger().info(f'rgb Images saved: {color_filename}')
            # self.get_logger().info(f'depth Images saved: {depth_filename}')


    def draw_boxes(self, results, image):
        # 处理模型的检测结果
        for det in results.xyxy[0]:  # detections
            xmin, ymin, xmax, ymax, conf, cls_id = int(det[0]), int(det[1]), int(det[2]), int(det[3]), det[4], int(det[5])
            label = self.class_names[cls_id]
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, 0.5, 2)
            label_width = label_size[0][0]
            label_height = label_size[0][1]
            cv2.rectangle(image, (xmin, ymin), (xmin + label_width, ymin - label_height - 10), (255, 0, 0), cv2.FILLED)
            cv2.putText(image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
        return image

    def callback(self, color_msg, depth_msg):
        # print(f"Received color image of shape: {color_msg.height}x{color_msg.width}")
        # print(f"Received depth image of shape: {depth_msg.height}x{depth_msg.width}")
        cv_color_image = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
        cv_depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")
        self.latest_color_image = cv_color_image
        self.latest_depth_image = cv_depth_image
######### 显示接收到的rgb和depth图像 ##################################
        # 归一化深度图像以增强显示效果
        cv_depth_normalized = cv2.normalize(cv_depth_image, None, 0, 255, cv2.NORM_MINMAX)
        cv_depth_normalized = np.uint8(cv_depth_normalized)  # 转换为8位图像
        # 显示处理后的深度图像
        cv2.imshow("Color Image", cv_color_image)
        cv2.imshow("Depth Image", cv_depth_normalized)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
######  yolo检测demo #########################################
        # yolo_results = self.model(cv_color_image)
        # print(yolo_results)
        # #绘制检测结果
        # cv_image = self.draw_boxes(yolo_results, cv_color_image)
        # cv2.imshow("Yolo detection", cv_image)
        # cv2.waitKey(1)
######  裁减测试demo，裁减图像中间300*300区域生成点云  ########################
        # height, width, channels = cv_color_image.shape
        # center_x, center_y = width // 2, height // 2
        # half_width, half_height = 150, 150  # 因为我们要裁剪300x300区域
        # start_x, end_x = center_x - half_width, center_x + half_width
        # start_y, end_y = center_y - half_height, center_y + half_height       

        # points = []
        # # height, width, channels = cv_color_image.shape
        # for v in range(start_y, end_y):
        #     for u in range(start_x, end_x):
        #         depth = cv_depth_image[v, u]
        #         if depth > 0:  # Simple depth filter to remove zero depth values
        #             # 这里的内参需要根据实际相机调整
        #             z = depth * 0.001  # scale depth to meters
        #             x = (u - 425.98785400390625) * z / 425.2796325683594
        #             y = (v - 241.7391357421875) * z / 425.2796325683594
        #             b, g, r = cv_color_image[v, u].astype(np.uint8)
        #             # points.append([x, y, z, r, g, b])
        #             rgb = struct.pack('BBBB', r, g, b, 255)  # 封装BGR到一个uint32中
        #             rgb = struct.unpack('I', rgb)[0]
        #             points.append([x, y, z, rgb])

        # # Create PointCloud2 message
        # header = Header(frame_id='camera_link', stamp=self.get_clock().now().to_msg())
        # fields = [PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        #         PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        #         PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        #         PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1)]
        # point_cloud_msg = pc2.create_cloud(header, fields, points)
        # self.publisher.publish(point_cloud_msg)
        # print("Published Point Cloud")


        

def main(args=None):
    rclpy.init(args=args)
    image_subscriber = ImageSubscriber()
    rclpy.spin(image_subscriber)
    cv2.destroyAllWindows()
    image_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
