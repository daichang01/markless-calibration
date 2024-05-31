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
from ultralytics import YOLO
from pathlib import Path


class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        self.bridge = CvBridge()
        self.color_sub = message_filters.Subscriber(self, Image, '/camera/camera/color/image_rect_raw')
        self.depth_sub = message_filters.Subscriber(self, Image, '/camera/camera/aligned_depth_to_color/image_raw')
        self.upteeth_publisher = self.create_publisher(PointCloud2, 'upteeth_point_cloud', 10)
        self.lowteeth_publisher = self.create_publisher(PointCloud2, 'lowteeth_point_cloud', 10)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.color_sub, self.depth_sub], 10, 0.1)
        self.ts.registerCallback(self.callback)

        self.latest_color_image = None
        self.latest_depth_image = None
        self.image_index = 0
        self.image_folder = "src/markless-calibration/image"  # 路径需要根据你的文件系统进行修改

##################  采集RGB和深度图并保存,用于yolo训练  ####################################################################
        # self.timer = self.create_timer(2.0, self.save_images)

##################   yolo集成，用于加载模型 ########################################################################################
        # self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        # self.model.eval()
        # self.class_names = self.model.names  # 从模型自动获取类别名称
        self.model = YOLO("best.pt") #yolov8在本地训练的实例分割模型
        


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
        # 显示处理后的RGB图像
        cv2.imshow("Color Image", cv_color_image)
        # 显示处理后的深度图像
        # cv2.imshow("Depth Image", cv_depth_normalized)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
######  yolo检测demo #########################################
        yolo_results = self.model(cv_color_image)
        yolo_results = self.model.predict(cv_color_image)
        if yolo_results:
            for r in yolo_results:
                img = np.copy(r.orig_img)
                img_name = Path(r.path).stem
                # 遍历每个结果中的对象，这些对象可能代表不同的检测到的实体
                for ci, c in enumerate(r):
                    # 获取检测到的对象的标签名称
                    # label = c.names[c.boxes.cls.tolist().pop()]
                    cls_idx = int(c.boxes.cls[0])  # 获取类别索引
                    label = c.names[cls_idx]  # 使用索引获取标签名称

                    # 创建一个与原图大小相同的黑色掩码
                    b_mask = np.zeros(img.shape[:2], np.uint8)
                    # 从检测对象中提取轮廓并转换为整数坐标
                    contour = c.masks.xy.pop()
                    contour = contour.astype(np.int32)
                    contour = contour.reshape(-1, 1, 2) #符合 OpenCV cv2.drawContours 函数的要求
                    _ = cv2.drawContours(b_mask, [contour], -1, (255, 255, 255), cv2.FILLED)
                    # 将单通道的黑白掩码转换为三通道格式
                    mask3ch = cv2.cvtColor(b_mask, cv2.COLOR_GRAY2BGR)
                    # 使用掩码与原图进行按位与操作，仅保留掩码区域的像素
                    isolated = cv2.bitwise_and(mask3ch, img)
                    #  Bounding box coordinates
                    x1, y1, x2, y2 = c.boxes.xyxy.cpu().numpy().squeeze().astype(np.int32)
                    print(f"{cls_idx}_{label}: {x1, y1, x2, y2}")
                    # Crop image to object region
                    iso_crop = isolated[y1:y2, x1:x2]

                    # 将处理后的图像保存到文件
                    # cv2.imwrite(f"{img_name}_{label}.png", isolated)
                    cv2.namedWindow(f"{cls_idx}_{label}", cv2.WINDOW_NORMAL)
                    cv2.imshow(f"{cls_idx}_{label}", iso_crop)
                    cv2.waitKey(5)

                    ######  裁减测试demo，裁减图像中间掩码区域生成点云  ########################
                    start_x, end_x = x1, x2
                    start_y, end_y = y1, y2      

                    points = []
                    # height, width, channels = cv_color_image.shape
                    for v in range(start_y, end_y):
                        for u in range(start_x, end_x):
                            depth = cv_depth_image[v, u]
                            if depth > 0:  # 简单的深度滤波，移除深度值为0的点
                                # 这里的内参需要根据实际相机调整
                                z = depth * 0.001  # scale depth to meters
                                x = (u - 425.98785400390625) * z / 425.2796325683594
                                y = (v - 241.7391357421875) * z / 425.2796325683594
                                b, g, r = cv_color_image[v, u].astype(np.uint8)
                                # print("BGR values:", b, g, r)  # 直接打印看是否有异常
                                rgb = struct.pack('BBBB', b, g, r, 255)  # 封装BGR到一个uint32中
                                rgb = struct.unpack('I', rgb)[0]
                                points.append([x, y, z, rgb])

                    # Create PointCloud2 message
                    header = Header(frame_id='camera_link', stamp=self.get_clock().now().to_msg())
                    fields = [PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                            PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1)]
                    point_cloud_msg = pc2.create_cloud(header, fields, points)
                    if(cls_idx == 0):
                        self.upteeth_publisher.publish(point_cloud_msg)
                        print("upteeth Point Cloud published")
                    elif(cls_idx == 1):
                        self.lowteeth_publisher.publish(point_cloud_msg)
                        print("lowteeth Point Cloud published")
                    
                    
        else:
            print("No objects detected")






        

def main(args=None):
    rclpy.init(args=args)
    image_subscriber = ImageSubscriber()
    rclpy.spin(image_subscriber)
    cv2.destroyAllWindows()
    image_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
