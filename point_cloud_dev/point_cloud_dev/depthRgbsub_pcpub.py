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
from tf2_ros import StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped
from scipy.interpolate import splprep, splev


class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        self.bridge = CvBridge()
        self.color_sub = message_filters.Subscriber(self, Image, '/camera/camera/color/image_rect_raw')
        self.depth_sub = message_filters.Subscriber(self, Image, '/camera/camera/aligned_depth_to_color/image_raw')
        self.lowfront_publisher = self.create_publisher(PointCloud2, 'lowfront_point_cloud', 10)
        # self.lowteeth_publisher = self.create_publisher(PointCloud2, 'lowteeth_point_cloud', 10)
        self.roi_publisher = self.create_publisher(PointCloud2, 'roi_point_cloud', 10)
        self.processed_image_publisher = self.create_publisher(Image, 'processed_image', 10)
        # 该同步器会在一个队列中存储最多 10 个消息，并且它会容忍消息时间戳之间最多相差 0.05 秒的偏差。
        self.ts = message_filters.ApproximateTimeSynchronizer([self.color_sub, self.depth_sub], 10, 0.05)
        # 当同步器检测到两个消息源有符合时间同步条件的消息时，该函数被调用。
        self.ts.registerCallback(self.callback)

        # 相机内参
        self.fx = 641.9315185546875
        self.fy = 641.9315185546875
        self.cx = 643.0005493164062
        self.cy = 362.68548583984375

        self.latest_color_image = None
        self.latest_depth_image = None
        self.image_index = 0
        self.image_folder = "src/markless-calibration/image"  # 路径需要根据你的文件系统进行修改

##################  采集RGB和深度图并保存,用于yolo训练  ####################################################################
        # self.timer = self.create_timer(2.0, self.save_images)

##################   yolo集成，用于加载训练好的模型 ########################################################################################
        self.model = YOLO("bestone.pt") #yolov8在本地训练的实例分割模型
        
    def save_images(self):
        if self.latest_color_image is not None and self.latest_depth_image is not None:
            current_time = datetime.now().strftime("%m%d_%H%M%S")
            color_filename = os.path.join(self.image_folder, f"color_{current_time}.png")
            depth_filename = os.path.join(self.image_folder, f"depth_{current_time}.png")

            # 保存彩色图像
            cv2.imwrite(color_filename, self.latest_color_image)
            self.get_logger().info(f'rgb Images saved: {color_filename}')

            # 保存深度图像，先进行归一化处理
            # depth_normalized = cv2.normalize(self.latest_depth_image, None, 0, 255, cv2.NORM_MINMAX)
            # depth_normalized = np.uint8(depth_normalized)
            # cv2.imwrite(depth_filename, depth_normalized)
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
            # 将彩色图像消息转换为 OpenCV 格式的图像，颜色格式为 BGR
        self.latest_color_image = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
        # 将深度图像消息转换为 OpenCV 格式的图像，数据保持原样（passthrough）
        self.latest_depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")

######### 显示接收到的rgb和depth图像 ##################################
        # 归一化深度图像以增强显示效果
        cv_depth_normalized = cv2.normalize(self.latest_depth_image, None, 0, 255, cv2.NORM_MINMAX)
        cv_depth_normalized = np.uint8(cv_depth_normalized)  # 转换为8位图像
        # 显示处理后的RGB和深度图像
        # cv2.imshow("Color Image",  self.latest_color_image)
        # cv2.imshow("Depth Image", cv_depth_normalized)
        # if cv2.waitKey(10) & 0xFF == ord('q'):
        #     cv2.destroyAllWindows()
############################################################### yolo检测demo ################3#########################################
        # 使用模型对最新的彩色图像进行预测，得到检测结果
        yolo_results = self.model.predict(self.latest_color_image)
        # 如果不想进行预测，可以将结果设为 None
        # yolo_results = None
        if yolo_results:
            for res in yolo_results:
                # 对每一个结果进行进一步处理
                self.process_oneres(res)
                    
        else:
            print("No objects detected")

    def process_oneres(self, r):
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
            # 结果是原图中只有与掩码白色区域相对应的部分被保留，其余部分因为与黑色（0）的与操作而变为黑色。
            isolated = cv2.bitwise_and(mask3ch, img)
            #  Bounding box coordinates
            x1, y1, x2, y2 = c.boxes.xyxy.cpu().numpy().squeeze().astype(np.int32)
            print(f"{cls_idx}_{label}: {x1, y1, x2, y2}")
            # 得到感兴趣区域
            iso_crop = isolated[y1:y2, x1:x2]

            # cv2.namedWindow(f"{cls_idx}_{label}", cv2.WINDOW_NORMAL)
            # cv2.imshow(f"{cls_idx}_{label}", iso_crop)
            # cv2.waitKey(5)

            ############################################## 牙齿轮廓提取 #####################################################
            contours, large_contours, interpolated_contours = self.edge_extration(x1, y1, x2, y2, b_mask, iso_crop)
            
            # 选择面积最大的轮廓绘制
            if contours:
                # 根据面积排序轮廓
                contours = sorted(contours, key=cv2.contourArea, reverse=True)
                # contours = contours[:len(large_contours)] # 选择前 n个 根据实际情况调整
                contours = contours[:2] # 选择前 n个 根据实际情况调整
                #绘制前n个最大轮廓
                mask = np.zeros_like(iso_crop)
                cv2.drawContours(mask, contours, -1, (0, 255, 0), 1)

                overlaid_image = cv2.addWeighted(iso_crop, 0.7, mask, 0.3, 0)

                # cv2.namedWindow(f"{cls_idx}_{label} Overlaid", cv2.WINDOW_NORMAL)
                # cv2.imshow(f"{cls_idx}_{label}mask", mask)
                # cv2.imshow(f"{cls_idx}_{label} Overlaid", overlaid_image)
                # cv2.waitKey(5)
                self.publish_processed_image(overlaid_image)

            ###############################  深度图边缘转点云 ##################################################
            start_x, end_x = x1, x2
            start_y, end_y = y1, y2      

            points_edge = []
            points_roi = []


            # for v in range(start_y, end_y):
            #     for u in range(start_x, end_x):
            #         depth = cv_depth_image[v, u]
            for contour in large_contours:
                for point in contour:
                    u = point[0][0] + start_x
                    v = point[0][1] + start_y
                    depth = self.latest_depth_image[v, u]
                    if depth > 0:  # 简单的深度滤波，移除深度值为0的点
                        # 这里的内参需要根据实际相机调整
                        z = depth * 0.001  # scale depth to meters
                        x = (u - self.cx) * z / self.fx
                        y = (v - self.cy) * z / self.fy
                        b, g, r = self.latest_color_image[v, u].astype(np.uint8)
                        # print("BGR values:", b, g, r)  # 直接打印看是否有异常
                        rgb = struct.pack('BBBB', b, g, r, 255)  # 封装BGR到一个uint32中
                        rgb = struct.unpack('I', rgb)[0]
                        points_edge.append([x, y, z, rgb])
            
            self.create_pointcloud2_msg(points_edge, cls_idx)
            
            ############################## 裁剪感兴趣区域点云，用于可视化验证 ###############################################
            for v in range(start_y, end_y):
                for u in range(start_x, end_x):
                    depth = self.latest_depth_image[v, u]
                    if depth > 0:  # 简单的深度滤波，移除深度值为0的点
                        # 这里的内参需要根据实际相机调整
                        z = depth * 0.001  # scale depth to meters
                        x = (u - self.cx) * z / self.fx
                        y = (v - self.cy) * z / self.fy
                        b, g, r = self.latest_color_image[v, u].astype(np.uint8)
                        # print("BGR values:", b, g, r)  # 直接打印看是否有异常
                        rgb = struct.pack('BBBB', b, g, r, 255)  # 封装BGR到一个uint32中
                        rgb = struct.unpack('I', rgb)[0]
                        points_roi.append([x, y, z, rgb])
            val_idx = 7
                        
            
            self.create_pointcloud2_msg(points_roi, val_idx)
    def edge_extration(self, x1, y1, x2, y2, b_mask, iso_crop):
        ############################################## 牙齿轮廓提取 #####################################################
        # 转换为灰度图
        gray_image = cv2.cvtColor(iso_crop, cv2.COLOR_BGR2GRAY)
        # 应用高斯模糊，用于降低图像噪声
        gray_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
        # 腐蚀操作
        kernel = np.ones((3,3), np.uint8)
        gray_image = cv2.erode(gray_image, kernel, iterations=1)
        # 使用 Otsu 的方法自动确定二值化阈值
        otsu_thresh, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        threshold1 = 0.5 * otsu_thresh  # 计算 Canny 边缘检测的低阈值
        threshold2 = otsu_thresh        # 计算 Canny 边缘检测的高阈值
        print(f"low threshold: {threshold1}, high threshold: {threshold2}")   # 打印阈值，用于调试

        # 应用 Canny 边缘检测
        edges = cv2.Canny(gray_image, threshold1, threshold2)
        cropped_mask = b_mask[y1:y2, x1:x2]  # 从黑色掩码中裁剪与感兴趣区域相对应的部分
        erode_mask = cv2.erode(cropped_mask, kernel, iterations=2)  # 对裁剪的掩码进行进一步腐蚀
        edges = cv2.bitwise_and(edges, edges, mask=erode_mask)  # 使用掩码过滤边缘，保留主要部分

        # 查找边缘的轮廓，只检索最外层轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # 过滤轮廓，只保留面积大于设定阈值的轮廓
        min_area = 3  # 设置最小面积阈值
        large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        print(f"Number of main contours: {len(large_contours)}")  # 打印过滤后的主要轮廓数量

        interpolated_contours = []
        # 打印每个主要轮廓中点的数量
        for i, contour in enumerate(large_contours):
            num_points = len(contour)
            # 轮廓插值
            # interpolated_contour = self.interpolate_contour(contour)
            # num_points = len(interpolated_contour)  # 获取轮廓中点的数量
            # interpolated_contours.append(interpolated_contour.reshape(-1, 1, 2)) #保证了每个轮廓都是由形状为 (n, 1, 2) 的数组组成，其中 n 是点的数量，符合 OpenCV 轮廓处理函数的要求
            print(f"Contour {i} has {num_points} points:")
            
        return contours, large_contours, interpolated_contours  # 返回所有轮廓和过滤后的主要轮廓
    
    def interpolate_contour(self, contour, num_points=400):
        # 将轮廓转换为 numpy 数组并确保其形状正确
        contour = np.array(contour, dtype=np.float32).squeeze()
        if contour.ndim == 1:
            contour = contour.reshape(-1, 2)  # 保证轮廓是二维的

        # 闭合轮廓，确保开始点和结束点相同
        if not np.all(contour[0] == contour[-1]):
            contour = np.vstack([contour, contour[0]])

        # 参数化轮廓，准备样条插值
        tck, u = splprep(contour.T, s=0, per=True)  # 参数 s 控制平滑度，per=True 确保是周期函数（闭合轮廓）

        # 生成均匀分布的新参数值
        unew = np.linspace(0, 1, num_points)

        # 使用生成的参数值进行样条插值
        out = splev(unew, tck)

        # 将输出转换为正确的形状
        interpolated_contour = np.column_stack(out).astype(np.int32)
        return interpolated_contour

    def interpolate_contour_uniform(self, contour, num_points=400):
        # 创建一个用于插值的空列表
        interpolated_contour = []
        # 将轮廓转换为numpy数组
        contour = np.array(contour).squeeze()
        # 获取轮廓上的点数
        contour_length = contour.shape[0]
        # 计算等间隔的插值点
        for i in range(contour_length):
            start_point = contour[i]
            end_point = contour[(i+1) % contour_length]  # 确保轮廓闭合
            for t in np.linspace(0, 1, num_points // contour_length):
                interpolated_point = (1 - t) * start_point + t * end_point
                interpolated_contour.append(interpolated_point)
        return np.array(interpolated_contour, dtype=np.int32)
    def publish_processed_image(self, image):
        image_msg = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")
        self.processed_image_publisher.publish(image_msg)

    def create_pointcloud2_msg(self, points, idx):
        header = Header(frame_id='camera_infra1_optical_frame', stamp=self.get_clock().now().to_msg())
        fields = [PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1)]
        point_cloud_msg = pc2.create_cloud(header, fields, points)
        if(idx == 0):
            self.lowfront_publisher.publish(point_cloud_msg)
            print("lowfront Point Cloud published")
        elif(idx == 1):
            # self.lowteeth_publisher.publish(point_cloud_msg)
            # print("lowteeth Point Cloud published")
            pass
        elif(idx == 7):
            self.roi_publisher.publish(point_cloud_msg)
            print("roi Point Cloud published")


def main(args=None):
    rclpy.init(args=args)
    image_subscriber = ImageSubscriber()
    rclpy.spin(image_subscriber)
    cv2.destroyAllWindows()
    image_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
