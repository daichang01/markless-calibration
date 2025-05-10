from .utils import *
from pyquaternion import Quaternion

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
        # print("length of eigenvectors:", eigenvectors.shape[1])
        # print("Top 3 eigenvalues:", eigenvalues[:3])
        # print("Corresponding eigenvectors:\n", eigenvectors[:, :3])
        
        # 返回排序后的特征向量和中心点。特征向量的每一列都是一个主成分方向。
        return eigenvectors, centroid

    # source_cloud和target_cloud是整个上边缘，source_pac和target_pca是上边缘一半
    def pca_double_adjust(self, source_cloud, target_cloud, source_pca, target_pca,source_pca2, target_pca2):
        if target_pca is None or target_pca2 is None:
            print("target_pca is None")
            return None
        # 将源点云和目标点云的点转换为NumPy数组
        source_points = np.asarray(source_cloud.points)
        target_points = np.asarray(target_cloud.points)
        source_left_points = np.asarray(source_pca.points)
        target_left_points = np.asarray(target_pca.points)
        source_right_points = np.asarray(source_pca2.points)
        target_right_points = np.asarray(target_pca2.points)

    

        source_left_eigenvectors, source_left_centroid = compute_pca(source_left_points)
        target_left_eigenvectors, target_left_centroid = compute_pca(target_left_points)
        source_right_eigenvectors, source_right_centroid = compute_pca(source_right_points)
        target_right_eigenvectors, target_right_centroid = compute_pca(target_right_points)
        

        # 生成所有可能的符号组合 (2^3=8种可能性)
        sign_combinations = [(1,1,1), (1,1,-1), (1,-1,1), (1,-1,-1),
                        (-1,1,1), (-1,1,-1), (-1,-1,1), (-1,-1,-1)]
        
        min_rmse = float('inf')
        best_R, best_t = None, None

    
        for signs in sign_combinations:
            print(f"\n尝试符号组合: {signs}")
            # 根据符号组合调整源点云的特征向量方向
            adjusted_target_left = target_left_eigenvectors * signs
            adjusted_target_right = target_right_eigenvectors * signs

            R_left = adjusted_target_left @ source_left_eigenvectors.T
            R_right = adjusted_target_right @ source_right_eigenvectors.T

            if not is_matrix_sane(R_left) or not is_matrix_sane(R_right):
                print("矩阵数值异常，跳过")
                continue


            R_left = ensure_rotation_matrix(R_left)
            R_right = ensure_rotation_matrix(R_right)


            if not is_valid_rotation_matrix(R_left) or not is_valid_rotation_matrix(R_right):
                print(f"跳过无效旋转矩阵的符号组合: {signs}")
                continue  # 直接跳过当前组合

            t_left = target_left_centroid - R_left @ source_left_centroid
            t_right = target_right_centroid - R_right @ source_right_centroid

            
            # ---- 平均变换（基于四元数球面平均）----------------
            # 将旋转矩阵转换为四元数
            try:
                q_left = Quaternion(matrix=R_left)
                q_right = Quaternion(matrix=R_right)
            except Exception as e:
                print(f"Skipping signs {signs} due to invalid rotation: {e}")
                continue
            q_avg = average_quaternions(q_left, q_right)
            R_avg = q_avg.rotation_matrix


            # 平均位移
            t_avg = (t_left + t_right) * 0.5

            # ---- 评估当前符号组合的配准质量 ---------------------
            transformed_points = (R_avg @ source_points.T).T + t_avg
            # current_rmse = np.sqrt(np.mean(np.sum((transformed_points - target_points)**2, axis=1)))
            current_rmse = calculate_rmse_with_matching(transformed_points, target_points)


            if current_rmse < min_rmse:
                min_rmse = current_rmse
                best_R, best_t = R_avg, t_avg
        
        coarse_transformation = np.eye(4)
        coarse_transformation[:3, :3] = best_R
        coarse_transformation[:3, 3] = best_t
        source_cloud.transform(coarse_transformation)
        transformed_whole_source = np.asarray(source_cloud.points)

        rmse = calculate_rmse(transformed_whole_source, target_points)

        return coarse_transformation,source_cloud, rmse


    def pca_double_adjust_old(self,source_cloud, target_cloud, source_pca, target_pca,source_pca2, target_pca2):
        if target_pca is None or target_pca2 is None:
            print("target_pca is None")
            return None
        # 将源点云和目标点云的点转换为NumPy数组
        source_points = np.asarray(source_cloud.points)
        target_points = np.asarray(target_cloud.points)
        source_pca_points = np.asarray(source_pca.points)
        target_pca_points = np.asarray(target_pca.points)
        source_pca2_points = np.asarray(source_pca2.points)
        target_pca2_points = np.asarray(target_pca2.points)
        
        # 计算源点云和目标点云的PCA特征向量和质心
        source_eigenvectors, source_centroid = self.compute_pca(source_points)
        target_eigenvectors, target_centroid = self.compute_pca(target_points)

        # 初始化结果列表
        initial_results = []

        # 遍历所有 8 种可能的主轴方向组合
        for i in range(8):
            signs = [(-1 if i & (1 << bit) else 1) for bit in range(3)]  # 生成一个包含3个元素的列表，分别为-1或1
            adjusted_source_eigenvectors = source_eigenvectors * signs  # 调整源点云的特征向量方向
            # 计算旋转矩阵和平移向量（重要）
            R = np.dot(target_eigenvectors, adjusted_source_eigenvectors.T) 
            t = target_centroid - np.dot(R, source_centroid)
            transformed_left_source = transform_points(source_pca_points, R, t)
            transformed_right_source = transform_points(source_pca2_points, R, t)

            rmse_left = calculate_rmse(transformed_left_source, target_pca_points)
            rmse_right = calculate_rmse(transformed_right_source, target_pca2_points)
            # 先尝试不计算重叠率
            initial_results.append((rmse_left, rmse_right, R, t, tuple(signs), i))
        min_rmse_left = min(initial_results, key=lambda x: x[0])
        min_rmse_right = min(initial_results, key=lambda x: x[1])
        # 确保筛选出的结果是同一个
        sigma = 1
        if min_rmse_left == min_rmse_right and min_rmse_left[0] < sigma:
            best_result = min_rmse_left
        else:
            return None
        rmse_left, rmse_right, R, t, signs, i = best_result
        coarse_transformation = np.eye(4)
        coarse_transformation[:3, :3] = R
        coarse_transformation[:3, 3] = t
        source_cloud.transform(coarse_transformation)
        transformed_whole_source = np.asarray(source_cloud.points)

        rmse = calculate_rmse(transformed_whole_source, target_points)
            
        return coarse_transformation, source_cloud, rmse

            




    def pca_adjust_calibration(self, source_cloud, target_cloud, source_pca, target_pca):
        if target_pca is None:
            print("target_pca is None")
            return None
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

            # 计算旋转矩阵和平移向量（重要）
            R = np.dot(target_eigenvectors, adjusted_source_eigenvectors.T)
            t = target_centroid - np.dot(R, source_centroid)
            
            # 将源点云的点进行变换
            transformed_source_points = transform_points(source_pca_points, R, t)
            # 计算均方误差 (MSE)
            rmse = calculate_rmse(transformed_source_points, target_pca_points)
            # 计算重叠率
            # overlap_ratio = self.calculate_overlap_ratio(transformed_source_points, target_points)
            overlap_ratio = calculate_overlap_ratio(transformed_source_points, target_pca_points)
            # print(f"combine: {i}, mse: {rmse}, overlap: {overlap_ratio}")

            # 添加到结果列表中
            initial_results.append((rmse, overlap_ratio, R, t, tuple(signs), i))

        # 筛选出MSE最小和重叠率最大的结果
        min_mse_result = min(initial_results, key=lambda x: x[0])
        max_overlap_result = max(initial_results, key=lambda x: x[1])
        
        # 确保筛选出的结果是同一个
        if min_mse_result == max_overlap_result:
            best_result = min_mse_result
        else:
            return None

        mse, overlap_ratio, R, t, signs, i = best_result
        # print(f"select: {i}, mse: {mse}, overlap: {overlap_ratio},signs: {signs}")
        coarse_transformation = np.eye(4)
        coarse_transformation[:3, :3] = R
        coarse_transformation[:3, 3] = t
        source_cloud.transform(coarse_transformation)
        transformed_source_points = np.asarray(source_cloud.points)

        rmse = calculate_rmse(transformed_source_points, target_points)
            
        overlap_ratio = calculate_overlap_ratio(transformed_source_points, target_points)
        return coarse_transformation, source_cloud, rmse, overlap_ratio
    
    
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
    
    