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
        # 用于pca校正的左侧三个牙齿边缘
        self.lowpca_sub = self.create_subscription(PointCloud2, '/lowfront_pca_adjust', self.lowpca_callback, 10)

        # 用于pca校正的右侧三个牙齿边缘
        self.lowpca_sub2 = self.create_subscription(PointCloud2, '/lowfront_pca_adjust2', self.lowpca_callback2, 10)
        # 利用上边缘和牙龈（不好用）
        # self.combined_sub = self.create_subscription(PointCloud2, '/combined_point_cloud', self.lowfront_callback, 10)
        
        # 待配准边缘

        # self.source_path = "/home/daichang/Desktop/teeth_ws/src/markless-calibration/wait_to_reg/0807model/Vertices6.txt"
        
        self.source_path = "/home/daichang/Desktop/teeth_ws/src/markless-calibration/wait_to_reg/0807model/up6.txt"  #整体
        self.source_path2 = "/home/daichang/Desktop/teeth_ws/src/markless-calibration/wait_to_reg/0807model/up3.txt" #左侧
        self.source_path3 = "/home/daichang/Desktop/teeth_ws/src/markless-calibration/wait_to_reg/0807model/up3onright.txt" #右侧
        
        # 口扫点云验证
        self.valsource_path = "/home/daichang/Desktop/teeth_ws/src/markless-calibration/wait_to_reg/0807model/teethrealon-down.txt"

        # self.valsource_path = "/home/daichang/Desktop/teeth_ws/src/markless-calibration/wait_to_reg/0807model/halfval.txt"
        
        # self.valsource_path = "/home/daichang/Desktop/teeth_ws/src/markless-calibration/wait_to_reg/0807model/teethreal_downsample.txt"

        #待验证标记点
        self.valpoints_path = "/home/daichang/Desktop/teeth_ws/src/markless-calibration/wait_to_reg/points/lowfront_points.txt"
        
        # 图像坐标系中的点 (x, y, z)
        self.point_img = np.array([-0.090951, -0.027712, 0.294036, 1.0]).reshape(4, 1)

        self.rvizsource = load_point_cloud(self.valsource_path)

        # 创建配准器
        self.pca_registrator = PCARegistration()
        self.icp_registrator = ICPRegistration()

        self.target_pca_copy = None  
        self.target_pca_copy2 = None


################################################  Registration Pipeline #############################################
    def lowfront_callback(self, msg):
        self.get_logger().info(f"############################## Registration start ######################################3")
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
        self.sourceright = load_point_cloud(self.source_path3)

        # 将目标点云保存为TXT文件
        # save_point_cloud_to_txt("/home/daichang/Desktop/teeth_ws/src/markless-calibration/wait_to_reg/preprocess/precloud", self.target)
        # 可视化预处理后的点云
        # visualize_initial_point_clouds(self.source,  self.target, window_name='preprocessed')
    ####################  pca粗配准  ##########
        start_time_pca = time.time()
        # save_point_cloud_to_txt("/home/daichang/Desktop/teeth_ws/src/markless-calibration/wait_to_reg/coarse/time", self.source) 
        # ######################带调整主轴方向的pca##################
        # 术前：self.source 整体整体牙齿轮廓点云， self.source2 为左侧牙齿轮廓点云   self.sourceright为右侧牙齿轮廓点云
        # 术中：self.target 整体牙齿轮廓点云， self.target_pca_copy为左侧牙齿轮廓点云 self.target_pca_copy2为右侧牙齿轮廓点云
        # coarse_result = self.pca_registrator.pca_adjust_calibration(self.source, self.target,self.source2,self.target_pca_copy)
        coarse_result = self.pca_registrator.pca_double_adjust(self.source, self.target,self.source2,self.target_pca_copy,self.sourceright,self.target_pca_copy2)
        
        if coarse_result is None:
            self.get_logger().info(f"No valid transformation found, skipping further processing")
            return
        # coarse_transformation, transformed_source_cloud, rmse, overlap_ratio = coarse_result
        coarse_transformation, transformed_source_cloud, rmse = coarse_result

        
        end_time_pca = time.time()

        pca_time = end_time_pca - start_time_pca
        # print("PCA粗配准后的变换矩阵：")
        # print(f"{coarse_transformation}")
        self.get_logger().info(f"PCA粗配准后的评估结果：")
        self.get_logger().info(f"pca RMSE: {rmse}")
        # self.get_logger().info(f"pca overlap_ratio: {overlap_ratio}")

        self.get_logger().info(f"pca粗配准共计耗时: {pca_time} 秒")

        # 将目标点云保存为TXT文件
        # save_point_cloud_to_txt("/home/daichang/Desktop/teeth_ws/src/markless-calibration/wait_to_reg/coarse/", transformed_source_cloud) 
        # save_point_cloud_to_txt("/home/daichang/Desktop/teeth_ws/src/markless-calibration/wait_to_reg/coarse/", self.source) 
        # save_point_cloud_to_txt("/home/daichang/Desktop/teeth_ws/src/markless-calibration/wait_to_reg/coarse/ori", self.target)        

        # 可视化粗配准后的点云
        # visualize_initial_point_clouds(self.source,  self.target, window_name='coarse_registration')



    ####################  icp精配准  ##################
        start_time_icp_tra = time.time()
        #法一：传统icp
        fine_transformation, fine_transformed_source, overlap_ratio, rmse, num_valid_pairs = self.icp_registrator.icp_fine_registration \
            (transformed_source_cloud, self.target)

        end_time_icp_tra = time.time()
        icp_time_tra = end_time_icp_tra - start_time_icp_tra
        # print(f"icp 精配准后的变换矩阵：{fine_transformation}")
        self.get_logger().info(f"传统icp粗配准后的评估结果：")
        self.get_logger().info(f"icp RMSE: {rmse}")
        self.get_logger().info(f"icp overlap_ratio: {overlap_ratio}")
        self.get_logger().info(f"icp精配准耗时: {icp_time_tra} 秒,有效点对数量: {num_valid_pairs}")


        start_time_icp = time.time()
        #法二：软分配
        fine_transformation, fine_transformed_source, overlap_ratio, rmse, num_valid_pairs = self.icp_registrator.icp_fine_registration_with_soft_assignments \
            (transformed_source_cloud, self.target)

        end_time_icp = time.time()
        icp_time = end_time_icp - start_time_icp
        # print(f"icp soft精配准后的变换矩阵：{fine_transformation}")
        self.get_logger().info(f"icp soft精配准后的评估结果：")
        self.get_logger().info(f"icp soft RMSE: {rmse}")
        self.get_logger().info(f"icp soft overlap_ratio: {overlap_ratio}")
        self.get_logger().info(f"icp soft 精配准耗时: {icp_time} 秒,有效点对数量: {num_valid_pairs}")

        start_time_cauchy = time.time()
        #法三：使用自适应权重和鲁棒损失函数的 ICP 精配准方法。
        fine_transformation, fine_transformed_source, overlap_ratio, rmse, num_valid_pairs = self.icp_registrator.icp_fine_registration_with_adaptive_weights \
            (transformed_source_cloud, self.target)

        end_time_cauchy = time.time()
        icp_time = end_time_cauchy - start_time_cauchy
        # print(f"icp soft精配准后的变换矩阵：{fine_transformation}")
        self.get_logger().info(f"icp cauchy精配准后的评估结果：")
        self.get_logger().info(f"icp cauchy RMSE: {rmse}")
        self.get_logger().info(f"icp cauchy overlap_ratio: {overlap_ratio}")
        self.get_logger().info(f"icp cauchy 精配准耗时: {icp_time} 秒,有效点对数量: {num_valid_pairs}")




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
        # 将接收到的点云消息转换为open3d格式
        self.target_pca = pointcloud2_to_open3d(msg)
        # 如果转换后的点云为空，则跳过注册
        if self.target_pca is None or len(self.target_pca.points) == 0:
            self.get_logger().info("Received empty target_pca  point cloud, skipping registration")
            return
        # 打印接收到的点云信息
        self.get_logger().info(f"Received new left target point cloud with {len(self.target_pca.points)} points)")
        # 去除离群值
        original_num_points = len(self.target_pca.points)
        # 使用open3d的remove_statistical_outlier函数去除离群值
        self.target_pca, ind = self.target_pca.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.5)
        filtered_num_points = len(self.target_pca.points)
        # 计算去除的离群值数量
        num_outliers = original_num_points - filtered_num_points
        # 打印去除的离群值数量
        self.get_logger().info(f"pca_target Removed {num_outliers} outliers")
        self.target_pca_copy = self.target_pca
    
    # 右边3颗牙齿回调函数
    def lowpca_callback2(self, msg):
        # 将接收到的点云消息转换为open3d格式
        self.target_pca2 = pointcloud2_to_open3d(msg)
        # 如果转换后的点云为空，则跳过注册
        if self.target_pca2 is None or len(self.target_pca2.points) == 0:
            self.get_logger().info("Received empty target_pca  point cloud, skipping registration")
            return
        # 打印接收到的点云信息
        self.get_logger().info(f"Received new right target point cloud with {len(self.target_pca2.points)} points)")
        # 去除离群值
        original_num_points = len(self.target_pca2.points)
        # 使用open3d的remove_statistical_outlier函数去除离群值
        self.target_pca2, ind = self.target_pca2.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.5)
        filtered_num_points = len(self.target_pca2.points)
        # 计算去除的离群值数量
        num_outliers = original_num_points - filtered_num_points
        # 打印去除的离群值数量
        self.get_logger().info(f"pca_target2 Removed {num_outliers} outliers")
        self.target_pca_copy2 = self.target_pca2
        
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




