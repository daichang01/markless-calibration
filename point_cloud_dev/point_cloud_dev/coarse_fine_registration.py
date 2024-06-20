from .utils import *
from scipy.spatial import cKDTree
from collections import deque, Counter

class FPFHRegistration:
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
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
            ], o3d.pipelines.registration.RANSACConvergenceCriteria(800000, 1000))
        return result

    def fpfh_ransac_coarse_registration(self, source, target, threshold):
        start_time_ransac = time.time()
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
        return  result_ransac.transformation, source

class PCARegistration:
    def compute_pca(self, points):
        centroid = np.mean(points, axis=0)
        centered_points = points - centroid
        cov_matrix = np.cov(centered_points.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]
        return eigenvectors, centroid

    def transform_points(self, points, R, t):
        transformed_points = np.dot(points, R.T) + t
        return transformed_points

    def calculate_mse(self, source_points, target_points):
        # 使用KD树找到最近邻点
        tree = cKDTree(target_points)
        distances, indices = tree.query(source_points, k=1)
        nearest_target_points = target_points[indices]
        return np.mean((source_points - nearest_target_points)**2)

    def find_best_orientation(self, source_cloud, target_cloud):
        source_points = np.asarray(source_cloud.points)
        target_points = np.asarray(target_cloud.points)
        source_eigenvectors, source_centroid = self.compute_pca(source_points)
        target_eigenvectors, target_centroid = self.compute_pca(target_points)

        best_mse = float('inf')
        best_transformation = None
        best_combination = None

        # 遍历所有 8 种可能的主轴方向组合
        for i in range(8):
            signs = [(-1 if i & (1 << bit) else 1) for bit in range(3)]
            adjusted_source_eigenvectors = source_eigenvectors * signs

            R = np.dot(target_eigenvectors, adjusted_source_eigenvectors.T)
            t = target_centroid - np.dot(R, source_centroid)
            transformed_source_points = self.transform_points(source_points, R, t)
            # 计算转换后的源点云和目标点云之间的均方误差 (MSE)。
            mse = self.calculate_mse(transformed_source_points, target_points)

            if mse < best_mse:
                best_mse = mse
                best_transformation = R, t
                best_combination = tuple(signs)
            
            # 生成 4x4 变换矩阵
            coarse_transformation = np.eye(4)
            coarse_transformation[:3, :3] = best_transformation[0]
            coarse_transformation[:3, 3] = best_transformation[1]
        
        source_cloud.transform(coarse_transformation)

        
        return coarse_transformation, source_cloud, best_mse, best_combination
    

    
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
        self.valsource_path = "/home/daichang/Desktop/ros2_ws/src/markless-calibration/pcd/wait-to-reg/newteeth_scantoval.txt"
        self.rvizsource = load_point_cloud(self.valsource_path)

        # 创建配准对象
        self.pca_registrator = PCARegistration()
        self.fpfh_registrator = FPFHRegistration()
        self.icp_registrator = ICPRegistration()

        self.combination_queue = deque(maxlen=30)
        self.most_frequent_combination = None
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
        num_interpolated_points = 400 # 你可以根据需要调整插值后的点数量
        if len(self.target.points) > 0:
            self.target = linear_interpolation(self.target, num_interpolated_points)     
        else:
            self.get_logger().info("Target point cloud is empty after outlier removal, skipping interpolation")
            return  
        # self.source = linear_interpolation(self.source, num_interpolated_points)
        # visualize_initial_point_clouds(self.source,  self.target, window_name='preprocessed')
    ####################  pca粗配准 + icp精配准 ##########################################################
        start_time_pca = time.time()
        coarse_transformation, transformed_source_cloud, best_mse, best_frequent_combination = \
            self.pca_registrator.find_best_orientation(self.source, self.target)
        # 更新组合队列
        self.combination_queue.append(best_frequent_combination)
        # 计算出现次数最多的组合
        if len(self.combination_queue) == self.combination_queue.maxlen:
            combination_counter = Counter(self.combination_queue)
            self.most_frequent_combination = combination_counter.most_common(1)[0][0]
            self.get_logger().info(f"Most Frequent Combination: {self.most_frequent_combination}")
        else:
            print("Combination Queue is not full yet.")
        
        if best_frequent_combination != self.most_frequent_combination:
            self.get_logger().info(f"Skipping callback due to combination mismatch: {best_frequent_combination}")
            return
       
        # _, transformed_source_cloud, _, combination = self.pca_registrator.find_best_orientation(self.source, self.target)
        
        end_time_pca = time.time()

        pca_time = end_time_pca - start_time_pca
        print("PCA粗配准后的变换矩阵：")
        print(f"{coarse_transformation}")
        print("PCA粗配准后的评估结果：")
        print(f"Best MSE: {best_mse}")
        print(f"Best Axis Flip Combination: {best_frequent_combination}")
        print(f"pca粗配准共计耗时: {pca_time} 秒")

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
        
        
        combined_transformation = np.dot(fine_transformation, coarse_transformation) 
        self.rvizpcd.transform(combined_transformation) #粗配准 + 精配准
        # self.rvizpcd.transform(coarse_transformation) # 只进行粗配准
        self.publish_point_cloud(self.pub_trans, self.rvizpcd)
        print("publish trans scan point cloud !")
#################### ransac粗配准  + icp精配准 #######################################################
    # coarse_transformation, transformed_source_cloud = self.fpfh_registrator.fpfh_ransac_coarse_registration(self.source, self.target,self.threshold)
    # fine_transformation = self.icp_registrator.icp_fine_registration(transformed_source_cloud, self.target ,self.threshold) 
    # combined_transformation = np.dot(fine_transformation, coarse_transformation)
############################# 最终可视化 ################################   
        


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





