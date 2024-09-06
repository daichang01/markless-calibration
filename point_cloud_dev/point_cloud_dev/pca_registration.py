from .utils import *

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

    
    def pca_adjust_calibration(self, source_cloud, target_cloud, source_pca, target_pca):
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
            transformed_source_points = transform_points(source_pca_points, R, t)
            # 计算均方误差 (MSE)
            mse = calculate_mse(transformed_source_points, target_pca_points)
            # 计算重叠率
            # overlap_ratio = self.calculate_overlap_ratio(transformed_source_points, target_points)
            overlap_ratio = calculate_overlap_ratio(transformed_source_points, target_pca_points)
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
            # best_result = min_mse_result
            return None

        mse, overlap_ratio, R, t, signs, i = best_result
        print(f"select: {i}, mse: {mse}, overlap: {overlap_ratio},signs: {signs}")
        coarse_transformation = np.eye(4)
        coarse_transformation[:3, :3] = R
        coarse_transformation[:3, 3] = t
        source_cloud.transform(coarse_transformation)
        transformed_source_points = np.asarray(source_cloud.points)

        mse = calculate_mse(transformed_source_points, target_points)
            # 计算重叠率
            # overlap_ratio = self.calculate_overlap_ratio(transformed_source_points, target_points)
        overlap_ratio = calculate_overlap_ratio(transformed_source_points, target_points)
        return coarse_transformation, source_cloud, mse, overlap_ratio
    
    
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
    