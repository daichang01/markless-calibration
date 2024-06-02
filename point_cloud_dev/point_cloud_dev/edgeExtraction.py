import cv2
import numpy as np

# 读取图像
image = cv2.imread('/home/daichang/Desktop/ros2_ws/src/markless-calibration/image/2024-06-02_11-02.png')
if image is None:
    print("Image not found")
else:
    # 转换为灰度图像
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 应用高斯模糊
    gray_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # 腐蚀操作
    kernel = np.ones((3,3), np.uint8)
    gray_image = cv2.erode(gray_image, kernel, iterations=1)
    
    # 使用 Otsu 的方法自动确定阈值
    otsu_thresh, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold1 = 0.5 * otsu_thresh
    threshold2 = otsu_thresh
    print(threshold1, threshold2)

    # 应用 Canny 边缘检测
    edges = cv2.Canny(gray_image, threshold1, threshold2)
    # 查找边缘的轮廓，只检索最外层轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 过滤轮廓
    min_area = 10  # 设置最小面积阈值
    large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    print(f"Number of main contours: {len(large_contours)}")  # 打印主要轮廓的数量
    # 创建用于绘制轮廓的空白图像
    contour_image = np.zeros_like(image)
    
     # 选择面积最大的轮廓绘制
    if contours:
        # 根据面积排序轮廓
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        contours = contours[:len(large_contours)] # 选择前 1 个 根据实际情况调整
        # 绘制最大的轮廓
        cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 1)

            # 打印每个轮廓中的所有点
        for i, contour in enumerate(large_contours):
            print(f"Contour {i}:")
            for point in contour:
                x, y = point.ravel()  # 转换点为 x, y 坐标
                # print(f"({x}, {y})")  # 打印坐标

    overlaid_image = cv2.addWeighted(image, 0.7, contour_image, 0.3, 0)
    
    
    # 显示原始图像和边缘叠加图像
    cv2.imshow("Original Image", image)
    cv2.imshow("processed Image", gray_image)
    cv2.imshow("Edge Image", edges)
    cv2.imshow("Contour Image", contour_image)
    cv2.namedWindow("Overlaid Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Overlaid Image", overlaid_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
