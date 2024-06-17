from .utils import *

 
class PointCloudRegistration(Node):
    def __init__(self, threshold=0.001):
        super().__init__('transform_pcd_publisher')
        self.threshold = threshold
        self.pub_ori = self.create_publisher(PointCloud2, '/ori_pcd_topic', 10)
        self.pub_trans = self.create_publisher(PointCloud2, '/trans_pcd_topic', 10)
        self.timer = self.create_timer(0.5, self.timer_callback)
        self.lowfront_sub = self.create_subscription(PointCloud2, '/lowfront_point_cloud', self.lowfront_callback, 10)
        self.source_path = "/home/daichang/Desktop/ros2_ws/src/markless-calibration/pcd/wait-to-reg/lowfrontscan.txt"
        self.valsource_path = "/home/daichang/Desktop/ros2_ws/src/markless-calibration/pcd/wait-to-reg/newteeth_scantoval.txt"
        self.rvizsource = load_point_cloud(self.valsource_path)
        # print("测试1")
        # self.source_path = "/home/daichang/Desktop/ros2_ws/src/markless-calibration/pcd/wait-to-reg/lowfrontscan.txt"
        # self.target_path = "/home/daichang/Desktop/ros2_ws/src/markless-calibration/pcd/auto-seg/lowteeth/lowsegfilter0608_230559.txt"
        # self.valsource_path = "/home/daichang/Desktop/ros2_ws/src/markless-calibration/pcd/wait-to-reg/newteeth_scantoval.txt"
        # self.valtarget_path = "/home/daichang/Desktop/ros2_ws/src/markless-calibration/pcd/auto-seg/valteeth/roifilter0608_230559.txt"

        # self.source = load_point_cloud(self.source_path)
        # self.target = load_point_cloud(self.target_path)
        # self.valsource = load_point_cloud(self.valsource_path)
        # self.valtarget = load_point_cloud(self.valtarget_path)

        # visualize_initial_point_clouds(self.source, self.target, window_name='原始点云')

        # # s, t, vs, vt = pca_coarse_registration(source, target, valsource, valtarget, threshold) # pca粗配准
        # s, t, vs, vt,coarse_transformation = self.fpfh_ransac_coarse_registration(self.source, self.target, self.valsource, self.valtarget, threshold) # ransac粗配准
        # fine_transformation = self.icp_fine_registration(s, t ,vs,  vt, threshold) # 精配准

        # print("测试2")
        # pc1 = load_point_cloud("/home/daichang/Desktop/ros2_ws/src/markless-calibration/pcd/wait-to-reg/edge_teeth.txt")
        # pc2 = load_point_cloud("/home/daichang/Desktop/ros2_ws/src/markless-calibration/pcd/wait-to-reg/teethtrue-seg.txt")
        # pc3 = load_point_cloud("/home/daichang/Desktop/ros2_ws/src/markless-calibration/pcd/wait-to-reg/teethscan.txt")
        # pc4 = load_point_cloud("/home/daichang/Desktop/ros2_ws/src/markless-calibration/pcd/wait-to-reg/teethtrue.txt")
        # visualize_initial_point_clouds(pc1, pc2, window_name='原始点云')
        # # s, t, vs, vt = pca_coarse_registration(pc1, pc2, pc3, pc4, threshold) # pca粗配准
        # s, t, vs, vt = fpfh_ransac_coarse_registration(pc1, pc2, pc3, pc4, threshold) # ransac粗配准
        # icp_fine_registration(source = s, target=t ,valsource=vs, valtarget= vt, threshold=threshold) # 精配准

        # print("rviz可视化测试")
        # self.combined_transformation = np.dot(fine_transformation, coarse_transformation)
        # self.rvizsource = load_point_cloud(self.valsource_path)
        # self.rviztransform = load_point_cloud(self.valsource_path)
        # self.rviztransform.transform(self.combined_transformation)
    
    ################################################ lowfront 配准 #############################################
    def lowfront_callback(self, msg):
        self.target = pointcloud2_to_open3d(msg)
        if self.target is None or len(self.target.points) == 0:
            self.get_logger().info("Received empty target point cloud, skipping registration")
            return
        self.get_logger().info(f"Received new target point cloud with {len(self.target.points)} points)")

        # 去除离群值
        self.target, ind = self.target.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.5)
        num_outliers = np.sum(np.logical_not(ind))
        self.get_logger().info(f"Removed {num_outliers} outliers")

        # 进行配准
        self.source = load_point_cloud(self.source_path)
        self.rviztransform = load_point_cloud(self.valsource_path)
    # pca粗配准 + icp精配准
        s, t, vs, vt, coarse_transformation_source, coarse_transformation_target,translation_vector = self.pca_coarse_registration(self.source, self.target, None, None, self.threshold)

        # 可视化粗配准结果
        visualize_initial_point_clouds(s, t, window_name='粗配准结果')

        # 将目标点云的变换反向应用到源点云上。
        # inverse_transformation_target = np.linalg.inv(coarse_transformation_target)
            # 手动计算目标变换的逆
        R = coarse_transformation_target[:3, :3]
        t = coarse_transformation_target[:3, 3]
        inverse_R = R.T
        inverse_t = -np.dot(inverse_R, t)
        inverse_transformation_target = np.eye(4)
        inverse_transformation_target[:3, :3] = inverse_R
        inverse_transformation_target[:3, 3] = inverse_t
        # 包含平移矢量的变换矩阵
        translation_matrix = np.eye(4)
        translation_matrix[:3, 3] = translation_vector
        combined_coarse_transformation = np.dot(translation_matrix, np.dot(inverse_transformation_target, coarse_transformation_source))
        # 应用最终的变换矩阵到源点云
        self.source.transform(combined_coarse_transformation)
        # 可视化粗配准结果
        visualize_initial_point_clouds(self.source, self.target, window_name='针对source粗配准结果')
        fine_transformation = self.icp_fine_registration(self.source, self.target ,vs,  vt, self.threshold)

        self.combined_transformation = np.dot(fine_transformation, combined_coarse_transformation)
    # ransac粗配准  + icp精配准
        # s, t, vs, vt,coarse_transformation = self.fpfh_ransac_coarse_registration(self.source, self.target, None, None, self.threshold) # ransac粗配准
        # icp精配准
        # fine_transformation = self.icp_fine_registration(s, t ,vs,  vt, self.threshold) 
        # self.combined_transformation = np.dot(fine_transformation, coarse_transformation)
        
        self.rviztransform.transform(self.combined_transformation) #粗配准 + 精配准
        # self.rviztransform.transform(coarse_transformation) # 只进行粗配准



    ############################################## FPFH + RANSAC 粗配准 ##############################################
    def compute_fpfh_feature(self, point_cloud, threshold):
        radius_normal = threshold * 10
        point_cloud.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
        radius_feature = threshold * 20
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            point_cloud,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        return fpfh

    def execute_global_registration(self, source, target, source_fpfh, target_fpfh, threshold):
        distance_threshold = threshold
        print(":: RANSAC registration on point clouds.")
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source, target, source_fpfh, target_fpfh, True,
            distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            3, [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
            ], o3d.pipelines.registration.RANSACConvergenceCriteria(40000, 500))

        return result

    def fpfh_ransac_coarse_registration(self, source, target, valsource, valtarget, threshold):
        start_time_ransac= time.time()
        source_fpfh = self.compute_fpfh_feature(source, threshold)
        target_fpfh = self.compute_fpfh_feature(target, threshold)
        
        result_ransac = self.execute_global_registration(source, target, source_fpfh, target_fpfh, threshold)
        
        # 应用 RANSAC 结果变换到原始点云
        fitness, inlier_rmse = evaluate_registration(source, target, result_ransac.transformation, threshold)

        end_time_ransac = time.time()
        ransac_time = end_time_ransac - start_time_ransac
        source.transform(result_ransac.transformation)
        print("粗配准后的变换矩阵：")
        print(f"{result_ransac.transformation}")
        print("ransac粗配准后的评估结果：")
        print(f"RMSE: {inlier_rmse}")
        print(f"Fitness: {fitness}")
        print(f"ransac粗配准耗时: {ransac_time} 秒")
        # visualize_initial_point_clouds(source, target, window_name='ransac粗配准结果')
        # valsource.transform(result_ransac.transformation)
        # visualize_initial_point_clouds(valsource, valtarget, window_name='验证ransac粗配准结果')
        return source, target, valsource, valtarget, result_ransac.transformation

    ############################################################# pca粗配准(备选)  ##################################################
    def apply_pca(self, point_cloud):
        # 计算点云的主成分
        mean = np.mean(point_cloud.points, axis=0)
        point_cloud_centered = point_cloud.points - mean
        u, s, vh = np.linalg.svd(point_cloud_centered, full_matrices=False)
        transformation = np.eye(4)
        transformation[:3, :3] = vh
        transformation[:3, 3] = mean
        return transformation

    def align_point_cloud(self, point_cloud, transformation):
        # 使用 PCA 的结果对点云进行变换
        points = np.asarray(point_cloud.points)
        # 将点扩展为齐次坐标
        points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
        # points_transformed = np.dot(points - np.mean(points, axis=0), transformation.T)
        # 应用变换
        points_transformed = np.dot(points_homogeneous, transformation.T)
        # 只保留变换后的 x, y, z 坐标
        points_transformed = points_transformed[:, :3]
        point_cloud.points = o3d.utility.Vector3dVector(points_transformed)
        return point_cloud


    def correct_pca_orientation(self, transformation):
        # transformation 是一个4x4的变换矩阵。transformation[:3, 0] 表示提取该矩阵的第一列的前三个元素（即主轴的方向向量）。在PCA中，第一列通常代表了数据的主轴方向。
        primary_axis = transformation[:3, 0]
        #  检查 primary_axis 的Z分量（primary_axis[2]）是否为负。
        if primary_axis[2] < 0:  # 如果Z分量为负，将整个主轴向量取反。这样可以确保主轴的Z方向为正，避免因为对称性或其他原因导致的主轴方向翻转问题。
            transformation[:3, 0] = -primary_axis
        return transformation

    def compute_centroid(self, point_cloud):
        #计算质心
        points = np.asarray(point_cloud.points)
        centroid = np.mean(points, axis=0)
        return centroid

    def translate_point_cloud(self, point_cloud, vector):
        #平移向量
        translated_points = np.asarray(point_cloud.points) + vector
        point_cloud.points = o3d.utility.Vector3dVector(translated_points)
        return point_cloud
    def pca_coarse_registration(self, source, target, valsource, valtarget, threshold):
        # 应用 PCA 进行配准
        start_time_pca= time.time()
        transformation_source = self.apply_pca(source)
        # transformation_source = self.correct_pca_orientation(transformation_source) #主轴校正
        transformation_target = self.apply_pca(target)
        # transformation_target = self.correct_pca_orientation(transformation_target) #主轴校正
        source_aligned = self.align_point_cloud(source, transformation_source)
        target_aligned = self.align_point_cloud(target, transformation_target)
        #计算平移向量
        centroid_source = self.compute_centroid(source_aligned)
        centroid_target = self.compute_centroid(target_aligned)
        translation_vector = centroid_target - centroid_source
        source_aligned = self.translate_point_cloud(source_aligned, translation_vector)

        end_time_pca = time.time()
        pca_time = end_time_pca - start_time_pca

        print(f"PCA source 结果：\n{transformation_source}")
        print(f"PCA target 结果：\n{transformation_target}")
        print(f"平移向量：\n{translation_vector}")
        print(f"pca粗配准耗时: {pca_time} 秒")


        # 评估粗配准结果
        fitness, inlier_rmse = evaluate_registration(source_aligned, target_aligned, np.eye(4), threshold)
        print("粗配准后的评估结果：")
        print(f"RMSE: {inlier_rmse}")
        print(f"Fitness: {fitness}")
        # visualize_initial_point_clouds(source_aligned, target_aligned, window_name='粗配准结果')
        valsource_aligned, valtarget_aligned = None, None
        # valsource_aligned = self.align_point_cloud(valsource, transformation_source)
        # valsource_aligned = self.translate_point_cloud(valsource, translation_vector)
        # valtarget_aligned = self.align_point_cloud(valtarget, transformation_target)
        # visualize_initial_point_clouds(valsource_aligned, valtarget_aligned, window_name='验证粗配准结果')
        return source_aligned, target_aligned, valsource_aligned, valtarget_aligned, transformation_source, transformation_target, translation_vector


    ############################################ icp精配准 ###################################################################
    def icp_fine_registration(self, source, target, valsource, valtarget, threshold = 0.02):
        start_time_icp = time.time()
        trans_init = np.eye(4)
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source, target, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())
        transformation_icp =  reg_p2p.transformation

        print("精配准后的变换矩阵：")
        print(f"{transformation_icp}")

        # 评估精配准结果
        fitness, inlier_rmse = evaluate_registration(source, target, transformation_icp, threshold)
        end_time_icp = time.time()
        icp_time = end_time_icp - start_time_icp
        source.transform(transformation_icp)
        print("精配准后的评估结果：")
        print(f"RMSE: {inlier_rmse}")
        print(f"Fitness: {fitness}")
        print(f"icp精配准耗时: {icp_time} 秒")

        # visualize_initial_point_clouds(source, target, window_name='icp精配准结果')
        # valsource.transform(transformation_icp)
        # visualize_initial_point_clouds(valsource, valtarget, window_name='验证icp精配准结果')
        return transformation_icp
    
    def timer_callback(self):
        self.publish_point_cloud(self.pub_ori, self.rvizsource)
        self.publish_point_cloud(self.pub_trans, self.rviztransform)
        print("publish ori and transformed point cloud!")

    def publish_point_cloud(self, pub, point_cloud):
        pc2_msg = convert_to_pointcloud2(point_cloud)
        pub.publish(pc2_msg)


############################################ start #########################################################
def main(args=None):
    rclpy.init(args=args)
    processor = PointCloudRegistration()
    rclpy.spin(processor)
    processor.destroy_node()
    rclpy.shutdown()
    


if __name__ == "__main__":
    main()





