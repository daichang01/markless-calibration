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
        self.lowon_publisher = self.create_publisher(PointCloud2, 'lowfront_point_cloud', 10)
        self.lowoff_publisher = self.create_publisher(PointCloud2, 'lowoff_point_cloud', 10)
        self.roi_publisher = self.create_publisher(PointCloud2, 'roi_point_cloud', 10)
        self.processed_image_publisher = self.create_publisher(Image, 'processed_image', 10)
        self.combined_publisher = self.create_publisher(PointCloud2, 'combined_point_cloud', 10)  # 新增的发布器
        self.ts = message_filters.ApproximateTimeSynchronizer([self.color_sub, self.depth_sub], 10, 0.05)
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

        self.model = YOLO("/home/daichang/Desktop/teeth_ws/src/markless-calibration/seg_pt/best0729.pt") #yolov8在本地训练的实例分割模型

        self.points_combined = []  # 存储合并后的点云

    def callback(self, color_msg, depth_msg):
        self.latest_color_image = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
        self.latest_depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")

        cv_depth_normalized = cv2.normalize(self.latest_depth_image, None, 0, 255, cv2.NORM_MINMAX)
        cv_depth_normalized = np.uint8(cv_depth_normalized)

        cv2.imshow("Color Image",  self.latest_color_image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            cv2.destroyAllWindows()

        yolo_results = self.model.predict(self.latest_color_image)
        if yolo_results:
            for res in yolo_results:
                self.process_oneres(res)
            self.publish_combined_pointcloud()  # 发布合并后的点云
            self.points_combined = []  # 清空合并点云数据
        else:
            print("No objects detected")

    def process_oneres(self, r):
        img = np.copy(r.orig_img)
        img_name = Path(r.path).stem

        for ci, c in enumerate(r):
            cls_idx = int(c.boxes.cls[0])
            label = c.names[cls_idx]

            b_mask = np.zeros(img.shape[:2], np.uint8)
            contour = c.masks.xy.pop()
            contour = contour.astype(np.int32)
            contour = contour.reshape(-1, 1, 2)
            _ = cv2.drawContours(b_mask, [contour], -1, (255, 255, 255), cv2.FILLED)
            mask3ch = cv2.cvtColor(b_mask, cv2.COLOR_GRAY2BGR)
            isolated = cv2.bitwise_and(mask3ch, img)
            x1, y1, x2, y2 = c.boxes.xyxy.cpu().numpy().squeeze().astype(np.int32)

            cv2.namedWindow(f"{cls_idx}_{label}", cv2.WINDOW_NORMAL)
            cv2.imshow(f"{cls_idx}_{label}", isolated[y1:y2, x1:x2])
            cv2.waitKey(5)

            contours, large_contours, interpolated_contours = self.edge_extration(x1, y1, x2, y2, b_mask, isolated[y1:y2, x1:x2])
            if contours:
                contours = sorted(contours, key=cv2.contourArea, reverse=True)
                contours = contours[:4]
                mask = np.zeros_like(isolated[y1:y2, x1:x2])
                cv2.drawContours(mask, contours, -1, (0, 255, 0), 1)
                overlaid_image = cv2.addWeighted(isolated[y1:y2, x1:x2], 0.7, mask, 0.3, 0)
                cv2.namedWindow(f"{cls_idx}_{label} Overlaid", cv2.WINDOW_NORMAL)
                cv2.imshow(f"{cls_idx}_{label} Overlaid", overlaid_image)
                cv2.waitKey(5)

            start_x, end_x = x1, x2
            start_y, end_y = y1, y2

            points_edge = []
            points_roi = []

            for contour in large_contours:
                for point in contour:
                    u = point[0][0] + start_x
                    v = point[0][1] + start_y
                    depth = self.latest_depth_image[v, u]
                    if depth > 0:
                        z = depth * 0.001
                        x = (u - self.cx) * z / self.fx
                        y = (v - self.cy) * z / self.fy
                        b, g, r = self.latest_color_image[v, u].astype(np.uint8)
                        rgb = struct.pack('BBBB', b, g, r, 255)
                        rgb = struct.unpack('I', rgb)[0]
                        points_edge.append([x, y, z, rgb])
                        self.points_combined.append([x, y, z, rgb])  # 添加到合并的点云

            self.create_pointcloud2_msg(points_edge, cls_idx)

            for v in range(start_y, end_y):
                for u in range(start_x, end_x):
                    depth = self.latest_depth_image[v, u]
                    if depth > 0:
                        z = depth * 0.001
                        x = (u - self.cx) * z / self.fx
                        y = (v - self.cy) * z / self.fy
                        b, g, r = self.latest_color_image[v, u].astype(np.uint8)
                        rgb = struct.pack('BBBB', b, g, r, 255)
                        rgb = struct.unpack('I', rgb)[0]
                        points_roi.append([x, y, z, rgb])
            val_idx = 7
            self.create_pointcloud2_msg(points_roi, val_idx)

    def publish_combined_pointcloud(self):
        header = Header(frame_id='camera_infra1_optical_frame', stamp=self.get_clock().now().to_msg())
        fields = [PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                  PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                  PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                  PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1)]
        point_cloud_msg = pc2.create_cloud(header, fields, self.points_combined)
        self.combined_publisher.publish(point_cloud_msg)
        print("Combined Point Cloud published")

    def edge_extration(self, x1, y1, x2, y2, b_mask, iso_crop):
        gray_image = cv2.cvtColor(iso_crop, cv2.COLOR_BGR2GRAY)
        gray_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
        kernel = np.ones((3,3), np.uint8)
        gray_image = cv2.erode(gray_image, kernel, iterations=1)
        otsu_thresh, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        threshold1 = 0.5 * otsu_thresh
        threshold2 = otsu_thresh
        print(f"low threshold: {threshold1}, high threshold: {threshold2}")

        edges = cv2.Canny(gray_image, threshold1 = 20, threshold2 = 90)
        cropped_mask = b_mask[y1:y2, x1:x2]
        erode_mask = cv2.erode(cropped_mask, kernel, iterations=2)
        edges = cv2.bitwise_and(edges, edges, mask=erode_mask)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        min_area = 3
        large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        print(f"Number of main contours: {len(large_contours)}")

        interpolated_contours = []
        for i, contour in enumerate(large_contours):
            num_points = len(contour)
            print(f"Contour {i} has {num_points} points:")

        return contours, large_contours, interpolated_contours

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
        if idx == 0:
            self.lowon_publisher.publish(point_cloud_msg)
            print("lowon Cloud published")
        elif idx == 1:
            self.lowoff_publisher.publish(point_cloud_msg)
            print("lowoff Cloud published")
        elif idx == 7:
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
