from .utils import *
from scipy.spatial import cKDTree


class PointCloudRegistration(Node):
    def __init__(self, threshold=0.001):
        super().__init__('transform_pcd_publisher')
        self.threshold = threshold
        self.pub_ori = self.create_publisher(PointCloud2, '/ori_pcd_topic', 10)
        self.pub_target = self.create_publisher(PointCloud2, '/target_pcd_topic', 10)
        self.pub_trans = self.create_publisher(PointCloud2, '/trans_pcd_topic', 10)
        self.timer = self.create_timer(1, self.timer_callback)
        self.lowfront_sub = self.create_subscription(PointCloud2, '/lowfront_point_cloud', self.lowfront_callback, 10)
        self.source_path = "/home/daichang/Desktop/ros2_ws/src/markless-calibration/pcd/wait-to-reg/lowfrontscan.txt"
        self.valsource_path = "/home/daichang/Desktop/ros2_ws/src/markless-calibration/pcd/wait-to-reg/pcavalue.txt"
        self.rvizsource = load_point_cloud(self.valsource_path)

        # 创建配准对象
        self.pca_registrator = PCARegistration()
        # self.fpfh_registrator = FPFHRegistration()
        self.icp_registrator = ICPRegistration()
        self.curve_icp_registrator = CurveICP()  

        self.best_combination = None
        self.computed_combination = True
        self.kalman_filter = KalmanFilter(state_dim=16, measurement_dim=16)
################################################ lowfront pipeline #############################################
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
        # 对去除离群值后的点云进行插值
        num_interpolated_points = 400 # 可以根据需要调整插值后的点数量
        if len(self.target.points) < num_interpolated_points:  # 确保需要增加点数时才进行插值
            try:
                self.target = spline_interpolation(self.target, num_interpolated_points)
            except ValueError as e:
                self.get_logger().info(f"Failed to perform spline interpolation: {e}")
                return
        else:
            self.get_logger().info("No interpolation needed, point count exceeds required for interpolation.")
        # visualize_initial_point_clouds(self.source,  self.target, window_name='preprocessed')
    ####################  pca粗配准  ##########################################################
        start_time_pca = time.time()
        # 带调整主轴方向的pca
        coarse_transformation, transformed_source_cloud, best_mse, best_frequent_combination = self.pca_registrator.pca_adjust_calibration(self.source, self.target)
        # 不带主轴方向调整的pca
        # coarse_transformation, transformed_source_cloud = self.pca_registrator.pca_calibration(self.source, self.target)
        end_time_pca = time.time()

        pca_time = end_time_pca - start_time_pca
        print("PCA粗配准后的变换矩阵：")
        print(f"{coarse_transformation}")
        print("PCA粗配准后的评估结果：")
        # print(f"Best Axis Flip Combination: {best_frequent_combination}")
        print(f"pca粗配准共计耗时: {pca_time} 秒")

        ####################  曲线ICP精配准  ##########################################################

        start_time_icp = time.time()
        fine_transformation, fitness, inlier_rmse, num_valid_pairs = self.curve_icp_registrator.icp_fine_registration \
            (transformed_source_cloud, self.target)
        end_time_icp = time.time()
        icp_time = end_time_icp - start_time_icp
        print("曲线ICP精配准后的变换矩阵：")
        print(f"{fine_transformation}")
        print("曲线ICP精配准后的评估结果：")
        print(f"RMSE: {inlier_rmse}")
        print(f"Fitness: {fitness}")
        print(f"曲线ICP精配准耗时: {icp_time} 秒")

    ####################  icp精配准  ##########################################################

        # start_time_icp = time.time()
        # fine_transformation, fitness, inlier_rmse = self.icp_registrator.icp_fine_registration \
        #     (transformed_source_cloud, self.target, self.threshold)
        # end_time_icp = time.time()
        # icp_time = end_time_icp - start_time_icp
        # print("精配准后的变换矩阵：")
        # print(f"{fine_transformation}")
        # print("精配准后的评估结果：")
        # print(f"RMSE: {inlier_rmse}")
        # print(f"Fitness: {fitness}")
        # print(f"icp精配准耗时: {icp_time} 秒")
#################### ransac粗配准  + icp精配准 #######################################################
        # coarse_transformation, transformed_source_cloud = self.fpfh_registrator.fpfh_ransac_coarse_registration(self.source, self.target,self.threshold)
        # fine_transformation = self.icp_registrator.icp_fine_registration(transformed_source_cloud, self.target ,self.threshold) 
        # combined_transformation = np.dot(fine_transformation, coarse_transformation)
    #############粗配准 + 精配准  ##########################################################    
        combined_transformation = np.dot(fine_transformation, coarse_transformation) 
        print(f"总变换矩阵:{combined_transformation}")

        # 使用卡尔曼滤波进行平滑
        combined_transformation_flat = combined_transformation.flatten()
        self.kalman_filter.update(combined_transformation_flat)
        smoothed_transformation_flat = self.kalman_filter.get_state().reshape((4, 4))

        # self.rvizpcd.transform(smoothed_transformation_flat) #粗配准 + 精配准 + 卡尔曼滤波
        self.rvizpcd.transform(combined_transformation) #粗配准 + 精配准
        # self.rvizpcd.transform(coarse_transformation) # 只进行粗配准
        self.publish_point_cloud(self.pub_trans, self.rvizpcd)
        print("publish trans scan point cloud !")
        
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
        
        # 返回排序后的特征向量和中心点。特征向量的每一列都是一个主成分方向。
        return eigenvectors, centroid


    def transform_points(self, points, R, t):
        transformed_points = np.dot(points, R.T) + t
        return transformed_points

    def calculate_mse(self, source_points, target_points):
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

    def compute_fpfh_feature(self, points):
        # 创建一个Open3D点云对象
        pcd = o3d.geometry.PointCloud()
        
        # 将输入的点坐标转换为Open3D的点云格式
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # 估计点云的法线
        # 参数search_param用于指定KD树搜索的参数
        # radius: 搜索半径，单位为米。这里设置为0.003米。
        # max_nn: 搜索的最大邻居数。这里设置为30。
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.003, max_nn=30))
        
        # 计算FPFH特征
        # 参数search_param用于指定KD树搜索的参数
        # radius: 搜索半径，单位为米。这里设置为0.005米。
        # max_nn: 搜索的最大邻居数。这里设置为50。
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd,
            o3d.geometry.KDTreeSearchParamHybrid(radius=0.005, max_nn=50)
        )
        
        # 返回计算得到的FPFH特征
        return fpfh
    
    def match_fpfh(self, source_points, target_points,source_fpfh, target_fpfh):
        # 创建一个Open3D点云对象，用于存储源点云
        source_pcd = o3d.geometry.PointCloud()
        target_pcd = o3d.geometry.PointCloud()
        
        # 将源和目标FPFH特征的数据转置后赋值给源点云对象
        source_pcd.points = o3d.utility.Vector3dVector(source_points)
        target_pcd.points = o3d.utility.Vector3dVector(target_points)
        # 设置距离阈值，单位为米。用于特征匹配时的最大对应距离。
        distance_threshold = 0.002
        # 使用RANSAC基于FPFH特征进行点云配准
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_pcd, target_pcd, source_fpfh, target_fpfh,
            mutual_filter=True,
            max_correspondence_distance=distance_threshold,  # 特征匹配的最大对应距离
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),  # 使用点到点的转换估计方法
            ransac_n=3,  # RANSAC算法中使用的样本点数
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),  # 基于边长的对应关系检查器
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)  # 基于距离的对应关系检查器
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(40000, 500)  # RANSAC收敛准则
        )
        
        # 返回匹配的fitness作为指标
        return result.fitness
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
        return coarse_transformation, source_cloud
    
    def pca_adjust_calibration(self, source_cloud, target_cloud):
        # 将源点云和目标点云的点转换为NumPy数组
        source_points = np.asarray(source_cloud.points)
        target_points = np.asarray(target_cloud.points)
        
        # 计算源点云和目标点云的PCA特征向量和质心
        source_eigenvectors, source_centroid = self.compute_pca(source_points)
        target_eigenvectors, target_centroid = self.compute_pca(target_points)
        # 计算源点云和目标点云的FPFH特征
        source_fpfh = self.compute_fpfh_feature(source_points)
        target_fpfh = self.compute_fpfh_feature(target_points)

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
            transformed_source_points = self.transform_points(source_points, R, t)
            # 计算均方误差 (MSE)
            mse = self.calculate_mse(transformed_source_points, target_points)
            # 计算重叠率
            overlap_ratio = self.calculate_overlap_ratio(transformed_source_points, target_points)

            # print(f"mse: {mse}, overlap: {overlap_ratio}")
            initial_results.append((mse, overlap_ratio, R, t, tuple(signs)))

        # 筛选出MSE最小或重叠率最大的结果
        min_mse = min(result[0] for result in initial_results)
        max_overlap_ratio = max(result[1] for result in initial_results)

        filtered_results = [result for result in initial_results if result[0] == min_mse or result[1] == max_overlap_ratio]
        
        # 精细评估：基于FPFH匹配的fitness
        best_result = None
        best_fpfh_fitness = -1
        i = 0
        for mse, overlap_ratio, R, t, signs in filtered_results:
            # 将源点云的点进行变换
            transformed_source_points = self.transform_points(source_points, R, t)
            # 计算变换后的源点云的FPFH特征
            transformed_source_fpfh = self.compute_fpfh_feature(transformed_source_points)
            # 计算FPFH匹配的fitness
            fpfh_fitness = self.match_fpfh(transformed_source_points, target_points, transformed_source_fpfh, target_fpfh)
            print(f"{i}:FPFH fitness: {fpfh_fitness}")
            i += 1
            # 如果当前的FPFH匹配fitness更好，则更新最佳结果
            if fpfh_fitness > best_fpfh_fitness:
                best_fpfh_fitness = fpfh_fitness
                best_result = (mse, overlap_ratio, R, t, signs)


        mse, overlap_ratio, R, t, signs = best_result
        coarse_transformation = np.eye(4)
        coarse_transformation[:3, :3] = R
        coarse_transformation[:3, 3] = t
        source_cloud.transform(coarse_transformation)
        return coarse_transformation, source_cloud, mse, signs




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





