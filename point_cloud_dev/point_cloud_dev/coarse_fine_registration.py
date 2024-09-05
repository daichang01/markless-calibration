from .utils import *
from scipy.spatial import cKDTree


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
        # self.valsource_path = "/home/daichang/Desktop/teeth_ws/src/markless-calibration/wait_to_reg/0807model/frontval_downsample.txt"

        self.valsource_path = "/home/daichang/Desktop/teeth_ws/src/markless-calibration/wait_to_reg/0807model/halfval.txt"
        
        # self.valsource_path = "/home/daichang/Desktop/teeth_ws/src/markless-calibration/wait_to_reg/0807model/teethreal_downsample.txt"

        #待验证标记点
        self.valpoints_path = "/home/daichang/Desktop/teeth_ws/src/markless-calibration/wait_to_reg/points/lowfront_points.txt"
        

        
        # 图像坐标系中的点 (x, y, z)
        self.point_img = np.array([-0.090951, -0.027712, 0.294036, 1.0]).reshape(4, 1)

        

        self.rvizsource = load_point_cloud(self.valsource_path)

        # 创建配准对象
        self.pca_registrator = PCARegistration()
        # self.fpfh_registrator = FPFHRegistration()
        self.icp_registrator = ICPRegistration()
        self.curve_icp_registrator = CurveICP()  

        self.best_combination = None
        self.computed_combination = True
        self.kalman_filter = KalmanFilter(state_dim=16, measurement_dim=16)
################################################ lowfront  registration pipeline #############################################
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

        self.source = load_point_cloud(self.source_path)
        self.rvizpcd = load_point_cloud(self.valsource_path)
        self.pointsval=  load_point_cloud(self.valpoints_path)
        self.source2 = load_point_cloud(self.source_path2)
        
        # 可视化预处理后的点云
        # visualize_initial_point_clouds(self.source,  self.target, window_name='preprocessed')
    ####################  pca粗配准  ##########################################################
        start_time_pca = time.time()
        # 带调整主轴方向的pca
        #法一：根据重叠率调整主轴方向
        coarse_transformation, transformed_source_cloud, best_mse, best_frequent_combination = self.pca_registrator.pca_adjust_calibration_nofpfh(self.source, self.target,self.source2,self.target_pca_copy)
        #法二： 只调整两个主轴
        # coarse_transformation, transformed_source_cloud, best_mse= self.pca_registrator.pca_adjust_calibration_dot_product(self.source, self.target)
        
        # 法三 原始pca
        # coarse_transformation, transformed_source_cloud = self.pca_registrator.pca_calibration(self.source, self.target)
        end_time_pca = time.time()

        pca_time = end_time_pca - start_time_pca
        print("PCA粗配准后的变换矩阵：")
        print(f"{coarse_transformation}")
        print("PCA粗配准后的评估结果：")
        # print(f"Best Axis Flip Combination: {best_frequent_combination}")
        print(f"pca粗配准共计耗时: {pca_time} 秒")

        ####################  曲线ICP精配准  ##########################################################

        # start_time_icp = time.time()
        # fine_transformation, fitness, inlier_rmse, num_valid_pairs = self.curve_icp_registrator.icp_fine_registration \
        #     (transformed_source_cloud, self.target)
        # end_time_icp = time.time()
        # icp_time = end_time_icp - start_time_icp
        # print("曲线ICP精配准后的变换矩阵：")
        # print(f"{fine_transformation}")
        # print("曲线ICP精配准后的评估结果：")
        # print(f"RMSE: {inlier_rmse}")
        # print(f"Fitness: {fitness}")
        # print(f"曲线ICP精配准耗时: {icp_time} 秒")

    ####################  icp精配准  ##########################################################

        start_time_icp = time.time()
        fine_transformation, fitness, inlier_rmse = self.icp_registrator.icp_fine_registration \
            (transformed_source_cloud, self.target, self.threshold)
        end_time_icp = time.time()
        icp_time = end_time_icp - start_time_icp
        print("精配准后的变换矩阵：")
        print(f"{fine_transformation}")
        print("精配准后的评估结果：")
        print(f"RMSE: {inlier_rmse}")
        print(f"Fitness: {fitness}")
        print(f"icp精配准耗时: {icp_time} 秒")

    #############粗配准 + 精配准  ##########################################################    
        combined_transformation = np.dot(fine_transformation, coarse_transformation) 
        print(f"总变换矩阵:{combined_transformation}")

        # 使用卡尔曼滤波进行平滑（未采用）
        # combined_transformation_flat = combined_transformation.flatten()
        # self.kalman_filter.update(combined_transformation_flat)
        # smoothed_transformation_flat = self.kalman_filter.get_state().reshape((4, 4))

        # self.rvizpcd.transform(smoothed_transformation_flat) #粗配准 + 精配准 + 卡尔曼滤波
        self.rvizpcd.transform(combined_transformation) #粗配准 + 精配准
        self.pointsval.transform(combined_transformation)
        # 计算变换后的点在相机坐标系中的位置
        point_cam = np.dot(combined_transformation, self.point_img)
        # 提取变换后的点 (x', y', z')
        x_prime, y_prime, z_prime = point_cam[:3, 0]

        # 打印结果
        print(f"Point1 in camera coordinate system: ({x_prime}, {y_prime}, {z_prime})")
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

    

class PCARegistration:
    def compute_pca(self, points):
        # 计算点集的中心点。这里使用np.mean计算所有点的平均值，axis=0确保按列求平均（即对每个维度求平均）。
        centroid = np.mean(points, axis=0)
        
        # 中心化点云：将每个点的坐标减去中心点的坐标，使得新的点云集中在原点附近。
        centered_points = points - centroid
        
        # 计算中心化后点云的协方差矩阵。np.cov用于计算协方差矩阵，参数.T表示转置，因为np.cov默认是按行处理的。
        cov_matrix = np.cov(centered_points.T)
        
        # 使用np.linalg.eigh计算协方差矩阵的特征值和特征向量。eigh是专为对称或厄米特矩阵设计的，更稳定。
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # 对特征值进行降序排序，获取排序后的索引。np.argsort对特征值数组进行排序，默认升序，[::-1]实现降序。
        idx = np.argsort(eigenvalues)[::-1]
        
        # 重排特征向量，使其与特征值的降序对应。这确保了第一个特征向量对应最大的特征值。
        eigenvectors = eigenvectors[:, idx]

        # 确保特征向量的方向一致性，例如，保持右手法则
        if np.linalg.det(eigenvectors) < 0:
            eigenvectors[:, 2] = -eigenvectors[:, 2]

            # 打印出最大的三个特征值，和相应的特征向量
        print("length of eigenvectors:", eigenvectors.shape[1])
        print("Top 3 eigenvalues:", eigenvalues[:3])
        print("Corresponding eigenvectors:\n", eigenvectors[:, :3])
        
        # 返回排序后的特征向量和中心点。特征向量的每一列都是一个主成分方向。
        return eigenvectors, centroid


    def transform_points(self, points, R, t):
        transformed_points = np.dot(points, R.T) + t
        return transformed_points

    def calculate_mse(self, source_points, target_points):
            # 使用 float64 确保高精度计算
        source_points = np.asarray(source_points, dtype=np.float64)
        target_points = np.asarray(target_points, dtype=np.float64)
        # 创建目标点云的KD树
        tree = cKDTree(target_points)
        # 查询源点云中每个点在目标点云中的最近邻点
        distances, indices = tree.query(source_points, k=1)
        # 找到每个源点云点对应的最近的目标点云点
        nearest_target_points = target_points[indices]
        # 计算源点云和最近的目标点云点之间的均方误差 (MSE)
        return np.mean((source_points - nearest_target_points)**2)

    def calculate_overlap_ratio(self, source_points, target_points, threshold=0.001):
        # 创建目标点云的KD树
        tree = cKDTree(target_points)
        # 查询源点云中每个点在目标点云中的最近邻点的距离
        distances, _ = tree.query(source_points, k=1)
        # 计算源点云中距离目标点云最近点距离小于阈值的点的数量
        overlap_count = np.sum(distances < threshold)
        # 计算重叠率，即重叠点的数量除以源点云的总点数
        return overlap_count / len(source_points)


    
    def pca_adjust_calibration_nofpfh(self, source_cloud, target_cloud, source_pca, target_pca):
        # 将源点云和目标点云的点转换为NumPy数组
        source_points = np.asarray(source_cloud.points)
        target_points = np.asarray(target_cloud.points)
        source_pca_points = np.asarray(source_pca.points)
        target_pca_points = np.asarray(target_pca.points)
        
        # 计算源点云和目标点云的PCA特征向量和质心
        source_eigenvectors, source_centroid = self.compute_pca(source_points)
        target_eigenvectors, target_centroid = self.compute_pca(target_points)

        # 初始化结果列表
        initial_results = []

        # 遍历所有 8 种可能的主轴方向组合 
        for i in range(8):
            signs = [(-1 if i & (1 << bit) else 1) for bit in range(3)]  # 生成一个包含3个元素的列表，分别为-1或1
            adjusted_source_eigenvectors = source_eigenvectors * signs  # 调整源点云的特征向量方向

            # 计算旋转矩阵和平移向量
            R = np.dot(target_eigenvectors, adjusted_source_eigenvectors.T)
            t = target_centroid - np.dot(R, source_centroid)
            
            # 将源点云的点进行变换
            transformed_source_points = self.transform_points(source_pca_points, R, t)
            # 计算均方误差 (MSE)
            mse = self.calculate_mse(transformed_source_points, target_pca_points)
            # 计算重叠率
            # overlap_ratio = self.calculate_overlap_ratio(transformed_source_points, target_points)
            # 改为用pca_target作校正
            overlap_ratio = self.calculate_overlap_ratio(transformed_source_points, target_pca_points)
            print(f"combine: {i}, mse: {mse}, overlap: {overlap_ratio}")

            # 添加到结果列表中
            initial_results.append((mse, overlap_ratio, R, t, tuple(signs), i))

        # 筛选出MSE最小和重叠率最大的结果
        min_mse_result = min(initial_results, key=lambda x: x[0])
        max_overlap_result = max(initial_results, key=lambda x: x[1])
        
        # 确保筛选出的结果是同一个
        if min_mse_result == max_overlap_result:
            best_result = min_mse_result
        else:
            # 如果不是同一个，选择重叠率最大的那个
            # best_result = max_overlap_result
            # 如果不是同一个，选择均方误差最小的那个
            best_result = min_mse_result

        mse, overlap_ratio, R, t, signs, i = best_result
        print(f"select: {i}, mse: {mse}, overlap: {overlap_ratio},signs: {signs}")
        coarse_transformation = np.eye(4)
        coarse_transformation[:3, :3] = R
        coarse_transformation[:3, 3] = t
        source_cloud.transform(coarse_transformation)
        return coarse_transformation, source_cloud, mse, signs
    
    def pca_adjust_calibration_dot_product(self, source_cloud, target_cloud):
        # 将源点云和目标点云的点转换为NumPy数组
        source_points = np.asarray(source_cloud.points)
        target_points = np.asarray(target_cloud.points)
        
        # 计算源点云和目标点云的PCA特征向量和质心
        source_eigenvectors, source_centroid = self.compute_pca(source_points) #先计算术前规划的点云pca主轴
        target_eigenvectors, target_centroid = self.compute_pca(target_points)

        # 调整源点云的第一和第二主方向使其与目标点云一致
        if np.dot(source_eigenvectors[:, 0], target_eigenvectors[:, 0]) < 0:
            source_eigenvectors[:, 0] = -source_eigenvectors[:, 0]
        if np.dot(source_eigenvectors[:, 1], target_eigenvectors[:, 1]) < 0:
            source_eigenvectors[:, 1] = -source_eigenvectors[:, 1]

        # 用调整后的第一和第二主方向计算第三主方向
        source_eigenvectors[:, 2] = np.cross(source_eigenvectors[:, 0], source_eigenvectors[:, 1])
        target_eigenvectors[:, 2] = np.cross(target_eigenvectors[:, 0], target_eigenvectors[:, 1])

        # 计算旋转矩阵和平移向量
        R = np.dot(target_eigenvectors, source_eigenvectors.T)
        t = target_centroid - np.dot(R, source_centroid)
        
        # 将源点云的点进行变换
        transformed_source_points = self.transform_points(source_points, R, t)
        
        # 计算均方误差 (MSE)
        mse = self.calculate_mse(transformed_source_points, target_points)
        # 计算重叠率
        overlap_ratio = self.calculate_overlap_ratio(transformed_source_points, target_points)
        print(f"mse: {mse}, overlap: {overlap_ratio}")

        # 创建并应用粗配准变换矩阵
        coarse_transformation = np.eye(4)
        coarse_transformation[:3, :3] = R
        coarse_transformation[:3, 3] = t
        source_cloud.transform(coarse_transformation)

        # 可视化粗配准
        # visualize_initial_point_clouds(source_cloud, target_cloud, "coarse_registration")
        
        # 返回配准结果
        return coarse_transformation, source_cloud, mse
    


    #原始pca
    def pca_calibration(self, source_cloud, target_cloud):
        # 将源点云和目标点云的点转换为NumPy数组
        source_points = np.asarray(source_cloud.points)
        target_points = np.asarray(target_cloud.points)

        # 计算源点云和目标点云的PCA特征向量和质心
        # PCA可以找出数据的主要变化方向，质心是所有点的均值，用于数据的归一化处理
        source_eigenvectors, source_centroid = self.compute_pca(source_points)
        target_eigenvectors, target_centroid = self.compute_pca(target_points)


        # 计算旋转矩阵和平移向量
        # 旋转矩阵R是通过将目标点云的特征向量与源点云的特征向量的转置相乘得到的
        # 这样可以将源点云旋转至与目标点云的主方向一致
        R = np.dot(target_eigenvectors, source_eigenvectors.T)
        # 平移向量t是通过目标点云的质心减去旋转后源点云的质心得到的
        # 这样可以将源点云平移至与目标点云的质心一致
        t = target_centroid - np.dot(R, source_centroid)

        coarse_transformation = np.eye(4)
        coarse_transformation[:3, :3] = R
        coarse_transformation[:3, 3] = t
        source_cloud.transform(coarse_transformation)
         # 可视化粗配准
        # visualize_initial_point_clouds(source_cloud, target_cloud, "ori_coarse_registration")
        return coarse_transformation, source_cloud
    


    
    



class CurveICP:
    def __init__(self, threshold=0.001, angle_threshold=np.pi / 6):
        self.threshold = threshold  # 设置距离阈值
        self.angle_threshold = angle_threshold  # 设置角度阈值

    def icp_fine_registration(self, source, target):
        source_points = np.asarray(source.points)  # 转换源点云为NumPy数组
        target_points = np.asarray(target.points)  # 转换目标点云为NumPy数组
        source_tangents = self.compute_tangents(source_points)  # 计算源点云的切线
        target_tangents = self.compute_tangents(target_points)  # 计算目标点云的切线

        prev_error = float('inf')  # 初始化前一轮的误差为无穷大
        for i in range(50):  # 进行50次迭代
            tree = cKDTree(target_points)  # 构建目标点云的KD树
            distances, indices = tree.query(source_points, k=1)  # 查找每个源点最近的目标点
            closest_points = target_points[indices]  # 找到最近的目标点
            closest_tangents = target_tangents[indices]  # 找到最近的目标点的切线

            valid_pairs = self.filter_pairs_by_tangent(source_points, source_tangents, closest_points, closest_tangents)  # 过滤掉不满足角度约束的点对
            # print(f"Iteration {i + 1}: Number of valid pairs = {len(valid_pairs)}")  # 打印有效点对的数量
            if len(valid_pairs) == 0:
                break  # 如果没有有效的点对，终止迭代

            source_valid = np.array([p[0] for p in valid_pairs])  # 获取有效的源点
            target_valid = np.array([p[1] for p in valid_pairs])  # 获取有效的目标点

            R, t = self.compute_transformation(source_valid, target_valid)  # 计算变换矩阵R和平移向量t
            source_points = np.dot(source_points, R.T) + t  # 应用变换矩阵和平移向量到源点云
            source_tangents = self.compute_tangents(source_points)  # 重新计算变换后的源点云的切线

            error = np.mean(np.linalg.norm(source_valid - target_valid, axis=1))  # 计算当前轮次的误差
            # print(f"Iteration {i + 1}: Error = {error}")
            if np.abs(prev_error - error) < 1e-6:
                break  # 如果误差变化很小，终止迭代
            prev_error = error  # 更新前一轮的误差

        transformation = np.eye(4)  # 初始化4x4的变换矩阵为单位矩阵
        transformation[:3, :3] = R  # 将旋转矩阵R赋值到变换矩阵的左上角3x3部分
        transformation[:3, 3] = t  # 将平移向量t赋值到变换矩阵的第4列前三行
        # 计算RMSE
        inlier_rmse = np.sqrt(np.mean((source_valid - target_valid) ** 2))
        # 计算Fitness
        inliers = distances < self.threshold
        fitness = np.sum(inliers) / len(source_points)
        return transformation, fitness, inlier_rmse, len(valid_pairs)  # 返回变换矩阵，最终误差和有效点对的数量

    def compute_tangents(self, points):
        tangents = []
        for i in range(1, len(points) - 1):
            tangent = (points[i + 1] - points[i - 1]) / 2  # 计算切线
            tangent /= np.linalg.norm(tangent)  # 归一化切线向量
            tangents.append(tangent)
        tangents = [tangents[0]] + tangents + [tangents[-1]]  # 补充第一个和最后一个切线向量
        return np.array(tangents)  # 返回切线向量数组

    def filter_pairs_by_tangent(self, source_points, source_tangents, target_points, target_tangents):
        valid_pairs = []
        for s_point, s_tangent, t_point, t_tangent in zip(source_points, source_tangents, target_points, target_tangents):
            angle = np.arccos(np.clip(np.dot(s_tangent, t_tangent), -1.0, 1.0))  # 计算两个切线向量之间的夹角
            if angle < self.angle_threshold:  # 如果夹角小于角度阈值，认为是有效点对
                valid_pairs.append((s_point, t_point))
        return valid_pairs  # 返回有效点对

    def compute_transformation(self, source, target):
        source_centroid = np.mean(source, axis=0)  # 计算源点云质心
        target_centroid = np.mean(target, axis=0)  # 计算目标点云质心
        H = (source - source_centroid).T @ (target - target_centroid)  # 计算协方差矩阵
        U, S, Vt = np.linalg.svd(H)  # 进行SVD分解
        R = Vt.T @ U.T  # 计算旋转矩阵R
        if np.linalg.det(R) < 0:  # 如果旋转矩阵的行列式为负
            Vt[2, :] *= -1  # 调整Vt
            R = Vt.T @ U.T  # 重新计算旋转矩阵R
        t = target_centroid - R @ source_centroid  # 计算平移向量t
        return R, t  # 返回旋转矩阵R和平移向量t

class ICPRegistration:
    def icp_fine_registration(self, source, target, threshold=0.02):
        
        trans_init = np.eye(4)
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source, target, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())
        transformation_icp = reg_p2p.transformation

        # 评估精配准结果
        fitness, inlier_rmse = evaluate_registration(source, target, transformation_icp, threshold)

        source.transform(transformation_icp)
        # visualize_initial_point_clouds(source, target, "icp_registration")
       
        return transformation_icp, fitness, inlier_rmse
 
class KalmanFilter:
    def __init__(self, state_dim, measurement_dim):
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim
        self.A = np.eye(state_dim)
        self.H = np.eye(state_dim)
        self.Q = np.eye(state_dim) * 0.01
        self.R = np.eye(state_dim) * 0.1
        self.P = np.eye(state_dim)
        self.x = np.zeros(state_dim)

    def update(self, z):
        # Prediction step
        x_pred = np.dot(self.A, self.x)
        P_pred = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q

        # Update step
        y = z - np.dot(self.H, x_pred)
        S = np.dot(np.dot(self.H, P_pred), self.H.T) + self.R
        K = np.dot(np.dot(P_pred, self.H.T), np.linalg.inv(S))
        self.x = x_pred + np.dot(K, y)
        self.P = P_pred - np.dot(np.dot(K, self.H), P_pred)

    def get_state(self):
        return self.x 
    


  
# class FPFHRegistration_old:
#     def compute_fpfh_feature(self, point_cloud, threshold):
#         radius_normal = threshold * 10
#         point_cloud.estimate_normals(
#             o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
#         radius_feature = threshold * 20
#         fpfh = o3d.pipelines.registration.compute_fpfh_feature(
#             point_cloud,
#             o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
#         return fpfh

#     def execute_global_registration(self, source, target, source_fpfh, target_fpfh, threshold):
#         distance_threshold = threshold
#         print(":: RANSAC registration on point clouds.")
#         result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
#             source, target, source_fpfh, target_fpfh, True,
#             distance_threshold,
#             o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
#             3, [
#                 o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
#                 o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
#             ], o3d.pipelines.registration.RANSACConvergenceCriteria(800000, 1000))
#         return result

#     def fpfh_ransac_coarse_registration(self, source, target, threshold):
#         start_time_ransac = time.time()
#         source_fpfh = self.compute_fpfh_feature(source, threshold)
#         target_fpfh = self.compute_fpfh_feature(target, threshold)
        
#         result_ransac = self.execute_global_registration(source, target, source_fpfh, target_fpfh, threshold)
        
#         # 应用 RANSAC 结果变换到原始点云
#         fitness, inlier_rmse = evaluate_registration(source, target, result_ransac.transformation, threshold)

#         end_time_ransac = time.time()
#         ransac_time = end_time_ransac - start_time_ransac
#         source.transform(result_ransac.transformation)
#         print("粗配准后的变换矩阵：")
#         print(f"{result_ransac.transformation}")
#         print("ransac粗配准后的评估结果：")
#         print(f"RMSE: {inlier_rmse}")
#         print(f"Fitness: {fitness}")
#         print(f"ransac粗配准耗时: {ransac_time} 秒")
#         return  result_ransac.transformation, source

def main(args=None):
    rclpy.init(args=args)
    processor = PointCloudRegistration()
    rclpy.spin(processor)
    processor.destroy_node()
    rclpy.shutdown()
    
if __name__ == "__main__":
    main()




