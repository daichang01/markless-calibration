from .utils import *
from .pca_registration import PCARegistration
from .icp_registration import ICPRegistration



class PointCloudRegistration(Node):
    def __init__(self, threshold=0.001):
        super().__init__('transform_pcd_publisher')
        self.threshold = threshold
        self.pub_ori = self.create_publisher(PointCloud2, '/ori_pcd_topic', 10)
        self.pub_target = self.create_publisher(PointCloud2, '/target_pcd_topic', 10)
        self.pub_trans = self.create_publisher(PointCloud2, '/trans_pcd_topic', 10)
        self.pub_point = self.create_publisher(PointCloud2, '/trans_pcd_point', 10)
        # self.timer = self.create_timer(1, self.timer_callback)
        # 只有上边缘
        self.lowfront_sub = self.create_subscription(PointCloud2, '/lowfront_point_cloud', self.lowfront_callback, 10)
        # 用于pca校正的三个牙齿边缘
        self.lowpca_sub = self.create_subscription(PointCloud2, '/lowfront_pca_adjust', self.lowpca_callback, 10)
        # 利用上边缘和牙龈
        # self.combined_sub = self.create_subscription(PointCloud2, '/combined_point_cloud', self.lowfront_callback, 10)
        
        # 待配准边缘

        # self.source_path = "/home/daichang/Desktop/teeth_ws/src/markless-calibration/wait_to_reg/0807model/Vertices6.txt"
        self.source_path = "/home/daichang/Desktop/teeth_ws/src/markless-calibration/wait_to_reg/0807model/up6.txt"
        # self.source_path = "/home/daichang/Desktop/teeth_ws/src/markless-calibration/wait_to_reg/0807model/up6down2 - Cloud.txt"
        # self.source_path = "/home/daichang/Desktop/teeth_ws/src/markless-calibration/wait_to_reg/0807model/updown5.txt"
        
        self.source_path2 = "/home/daichang/Desktop/teeth_ws/src/markless-calibration/wait_to_reg/0807model/up3.txt"
        # 口扫点云验证
        self.valsource_path = "/home/daichang/Desktop/teeth_ws/src/markless-calibration/wait_to_reg/0807model/teethrealon-down.txt"

        # self.valsource_path = "/home/daichang/Desktop/teeth_ws/src/markless-calibration/wait_to_reg/0807model/halfval.txt"
        
        # self.valsource_path = "/home/daichang/Desktop/teeth_ws/src/markless-calibration/wait_to_reg/0807model/teethreal_downsample.txt"

        #待验证标记点
        self.valpoints_path = "/home/daichang/Desktop/teeth_ws/src/markless-calibration/wait_to_reg/points/lowfront_points.txt"
        
        # 图像坐标系中的点 (x, y, z)
        self.point_img = np.array([-0.090951, -0.027712, 0.294036, 1.0]).reshape(4, 1)

        self.rvizsource = load_point_cloud(self.valsource_path)

        # 创建配准对象
        self.pca_registrator = PCARegistration()
        self.icp_registrator = ICPRegistration()


################################################  Registration Pipeline #############################################
    def lowfront_callback(self, msg):
        self.target = pointcloud2_to_open3d(msg)
        if self.target is None or len(self.target.points) == 0:
            self.get_logger().info("Received empty target point cloud, skipping registration")
            return
        self.get_logger().info(f"Received new target point cloud with {len(self.target.points)} points)")
        # 去除离群值
        original_num_points = len(self.target.points)
        self.target, ind = self.target.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.5)
        filtered_num_points = len(self.target.points)
        num_outliers = original_num_points - filtered_num_points
        self.get_logger().info(f"Removed {num_outliers} outliers")
        self.publish_point_cloud(self.pub_target, self.target)

        # open3d格式
        self.source = load_point_cloud(self.source_path)
        self.rvizpcd = load_point_cloud(self.valsource_path)
        self.pointsval=  load_point_cloud(self.valpoints_path)
        self.source2 = load_point_cloud(self.source_path2)

        # 将目标点云保存为TXT文件
        # save_point_cloud_to_txt("/home/daichang/Desktop/teeth_ws/src/markless-calibration/wait_to_reg/preprocess/precloud", self.target)
        # 可视化预处理后的点云
        # visualize_initial_point_clouds(self.source,  self.target, window_name='preprocessed')
    ####################  pca粗配准  ##########
        start_time_pca = time.time()
        # 带调整主轴方向的pca
        #法一：根据重叠率调整主轴方向
        coarse_result = self.pca_registrator.pca_adjust_calibration(self.source, self.target,self.source2,self.target_pca_copy)
        if coarse_result is None:
            print("No valid transformation found, skipping further processing")
            return
        coarse_transformation, transformed_source_cloud, rmse, overlap_ratio = coarse_result
        
        # 法二：原始pca
        # coarse_transformation, transformed_source_cloud = self.pca_registrator.pca_calibration(self.source, self.target)
        end_time_pca = time.time()

        pca_time = end_time_pca - start_time_pca
        print("PCA粗配准后的变换矩阵：")
        print(f"{coarse_transformation}")
        print("PCA粗配准后的评估结果：")
        print(f"pca RMSE: {rmse}")
        print(f"pca overlap_ratio: {overlap_ratio}")

        # print(f"Best Axis Flip Combination: {best_frequent_combination}")
        print(f"pca粗配准共计耗时: {pca_time} 秒")

        # 将目标点云保存为TXT文件
        # save_point_cloud_to_txt("/home/daichang/Desktop/teeth_ws/src/markless-calibration/wait_to_reg/coarse/coarse", transformed_source_cloud)        
        # 可视化粗配准后的点云
        # visualize_initial_point_clouds(self.source,  self.target, window_name='coarse_registration')


    ####################  icp精配准  ##################
        print("11111111111111111111111111111")
        start_time_icp_tra = time.time()
        #法一：传统icp
        fine_transformation, fine_transformed_source, overlap_ratio, rmse, num_valid_pairs = self.icp_registrator.icp_fine_registration \
            (transformed_source_cloud, self.target)

        end_time_icp_tra = time.time()
        icp_time_tra = end_time_icp_tra - start_time_icp_tra
        # print(f"icp 精配准后的变换矩阵：{fine_transformation}")
        print(f"icp RMSE: {rmse}")
        print(f"icp overlap_ratio: {overlap_ratio}")
        print(f"icp精配准耗时: {icp_time_tra} 秒,有效点对数量: {num_valid_pairs}")


        print("22222222222222222222222222222")
        start_time_icp = time.time()
        #法一：传统icp
        # fine_transformation, fine_transformed_source, overlap_ratio, mse, num_valid_pairs = self.icp_registrator.icp_fine_registration \
        #     (transformed_source_cloud, self.target)
        #法二：曲线icp
        # fine_transformation, overlap_ratio, mse, num_valid_pairs = self.icp_registrator.icp_fine_registration_curve \
        #     (transformed_source_cloud, self.target)
        #法三：软分配
        fine_transformation, fine_transformed_source, overlap_ratio, rmse, num_valid_pairs = self.icp_registrator.icp_fine_registration_with_soft_assignments \
            (transformed_source_cloud, self.target)

        end_time_icp = time.time()
        icp_time = end_time_icp - start_time_icp
        # print(f"icp soft精配准后的变换矩阵：{fine_transformation}")
        print(f"icp soft RMSE: {rmse}")
        print(f"icp soft overlap_ratio: {overlap_ratio}")
        print(f"icp soft 精配准耗时: {icp_time} 秒,有效点对数量: {num_valid_pairs}")

        print("333333333333333333333333333333")
        start_time_cauchy = time.time()
        #法一：传统icp
        # fine_transformation, fine_transformed_source, overlap_ratio, mse, num_valid_pairs = self.icp_registrator.icp_fine_registration \
        #     (transformed_source_cloud, self.target)
        #法二：曲线icp
        # fine_transformation, overlap_ratio, mse, num_valid_pairs = self.icp_registrator.icp_fine_registration_curve \
        #     (transformed_source_cloud, self.target)
        #法三：new
        fine_transformation, fine_transformed_source, overlap_ratio, rmse, num_valid_pairs = self.icp_registrator.icp_fine_registration_with_adaptive_weights \
            (transformed_source_cloud, self.target)

        end_time_cauchy = time.time()
        icp_time = end_time_cauchy - start_time_cauchy
        # print(f"icp soft精配准后的变换矩阵：{fine_transformation}")
        print(f"icp cauchy RMSE: {rmse}")
        print(f"icp cauchy overlap_ratio: {overlap_ratio}")
        print(f"icp cauchy 精配准耗时: {icp_time} 秒,有效点对数量: {num_valid_pairs}")






        combined_transformation = np.dot(fine_transformation, coarse_transformation) 
        # combined_transformation = np.dot(curve_fine_transformation, coarse_transformation)
        print(f"总变换矩阵:{combined_transformation}")


        # 将目标点云保存为TXT文件
        # save_point_cloud_to_txt("/home/daichang/Desktop/teeth_ws/src/markless-calibration/wait_to_reg/fine/fine", fine_transformed_source)        
        # 可视化精配准后的点云
        # visualize_initial_point_clouds(self.source,  self.target, window_name='fine_registration')


        self.rvizpcd.transform(combined_transformation) #粗配准 + 精配准
        self.pointsval.transform(combined_transformation)
        # 计算变换后的点在相机坐标系中的位置
        point_cam = np.dot(combined_transformation, self.point_img)
        # 提取变换后的点 (x', y', z')
        x_prime, y_prime, z_prime = point_cam[:3, 0]

        # 打印结果
        print(f"Point1 in camera coor dinate system: ({x_prime}, {y_prime}, {z_prime})")
        # self.rvizpcd.transform(coarse_transformation) # 只进行粗配准
        self.publish_point_cloud(self.pub_trans, self.rvizpcd)
        self.publish_point_cloud(self.pub_point, self.pointsval)
        print("publish trans scan point cloud !")

    def lowpca_callback(self, msg):
        self.target_pca = pointcloud2_to_open3d(msg)
        if self.target_pca is None or len(self.target_pca.points) == 0:
            self.get_logger().info("Received empty target_pca  point cloud, skipping registration")
            return
        self.get_logger().info(f"Received new target point cloud with {len(self.target_pca.points)} points)")
        # 去除离群值
        original_num_points = len(self.target_pca.points)
        self.target_pca, ind = self.target_pca.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.5)
        filtered_num_points = len(self.target_pca.points)
        num_outliers = original_num_points - filtered_num_points
        self.get_logger().info(f"pca_target Removed {num_outliers} outliers")
        self.target_pca_copy = self.target_pca
        
    def timer_callback(self):
        self.publish_point_cloud(self.pub_ori, self.rvizsource)
        # self.publish_point_cloud(self.pub_trans, self.rvizpcd)
        print("publish ori scan point cloud !")

    def publish_point_cloud(self, pub, point_cloud):
        pc2_msg = convert_to_pointcloud2(point_cloud)
        pub.publish(pc2_msg)

    

    

def main(args=None):
    rclpy.init(args=args)
    processor = PointCloudRegistration()
    rclpy.spin(processor)
    processor.destroy_node()
    rclpy.shutdown()
    
if __name__ == "__main__":
    main()




