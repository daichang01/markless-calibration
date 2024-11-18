from .utils import *

class ICPRegistration:

    def __init__(self, threshold=0.001, angle_threshold=np.pi / 4, sigma = 0.0005, icp_threshold = 1e-7):
        self.threshold = threshold  # 设置距离阈值
        self.angle_threshold = angle_threshold  # 设置角度阈值
        self.valid_pairs = []
        self.sigma = sigma
        self.icp_threshold = icp_threshold

    # def icp_fine_registration_open3d(self, source, target, threshold=0.02):
    #     trans_init = np.eye(4)
    #     reg_p2p = o3d.pipelines.registration.registration_icp(
    #         source, target, threshold, trans_init,
    #         o3d.pipelines.registration.TransformationEstimationPointToPoint())
    #     transformation_icp = reg_p2p.transformation

    #     source.transform(transformation_icp)
    #     transformed_source_points = np.asarray(source.points)
    #     target_points = np.asarray(target.points)

    #     # 评估精配准结果
    #     # fitness, inlier_rmse = evaluate_registration(source, target, transformation_icp, threshold)
    #     mse = calculate_mse(transformed_source_points, target_points)
    #     overlap_ratio = calculate_overlap_ratio(transformed_source_points, target_points)
       
    #     return transformation_icp, source, overlap_ratio, mse
    
    def icp_fine_registration(self, source, target):
        source_points = np.asarray(source.points)  # 转换源点云为NumPy数组
        target_points = np.asarray(target.points)  # 转换目标点云为NumPy数组
        print(f"source_points: {len(source_points)}, target_points: {len(target_points)}")


        prev_error = float('inf')  # 初始化前一轮的误差为无穷大
        for i in range(50):  # 进行50次迭代
            
            tree = cKDTree(source_points)  # 构建点云的KD树
            distances, indices = tree.query(target_points, k=1)  # 查找每个源点最近的目标点
            # print(f"length of indices: {len(indices)}")

            valid_pairs = []

            for t_idx, s_idx in enumerate(indices):
                s_point = source_points[s_idx]
                t_point = target_points[t_idx]

                valid_pairs.append((s_point, t_point))


            if len(valid_pairs) == 0:
                break  # 如果没有有效的点对，终止迭代

            source_valid = np.array([p[0] for p in valid_pairs])  # 获取有效的源点
            target_valid = np.array([p[1] for p in valid_pairs])  # 获取有效的目标点

            R, t = self.compute_transformation(source_valid, target_valid)  # 计算变换矩阵R和平移向量t
            source_points = np.dot(source_points, R.T) + t  # 应用变换矩阵和平移向量到源点云
            
            error = np.mean(np.linalg.norm(source_valid - target_valid, axis=1))  # 计算当前轮次的误差
            # print(f"Iteration {i + 1}: Error = {error}")
            if np.abs(prev_error - error) < self.icp_threshold:
                print(f"Converged after {i + 1} iterations.")
                break  # 如果误差变化很小，终止迭代
            prev_error = error  # 更新前一轮的误差

        transformation = np.eye(4)  # 初始化4x4的变换矩阵为单位矩阵
        transformation[:3, :3] = R  # 将旋转矩阵R赋值到变换矩阵的左上角3x3部分
        transformation[:3, 3] = t  # 将平移向量t赋值到变换矩阵的第4列前三行

        # 将源点云的点进行变换
        transformed_source_points = transform_points(source_points, R, t)
        # 计算均方误差 (MSE)
        rmse = calculate_rmse(transformed_source_points, target_points)
        # 计算重叠率
        overlap_ratio = calculate_overlap_ratio(transformed_source_points, target_points)

        transformed_source_pcd = o3d.geometry.PointCloud()
        transformed_source_pcd.points = o3d.utility.Vector3dVector(transformed_source_points)

        return transformation, transformed_source_pcd, overlap_ratio, rmse, len(valid_pairs)  # 返回变换矩阵，最终误差和有效点对的数量


    def icp_fine_registration_with_soft_assignments(self, source, target):
        source_points = np.asarray(source.points)  # 转换源点云为NumPy数组
        target_points = np.asarray(target.points)  # 转换目标点云为NumPy数组
        print(f"source_points: {len(source_points)}, target_points: {len(target_points)}")

        prev_error = float('inf')  # 初始化前一轮的误差为无穷大

        for i in range(50):  # 进行50次迭代
            tree = cKDTree(source_points)  # 构建点云的KD树
            distances, indices = tree.query(target_points, k=1)  # 查找每个目标点最近的源点
            # print(f"distances: {distances}")

            # 计算基于距离的权重（软指派）
            # weights = np.exp(-distances**2 / (2 * self.sigma**2))  # 距离越大，权重越小
            weights = np.exp(-distances**2 / (2 * 0.001**2))  # 距离越大，权重越小


            valid_pairs = []
            # for idx, (s_idx, t_idx, weight) in enumerate(zip(indices, range(len(target_points)), weights)):
            for t_idx in range(len(target_points)):
                s_idx = indices[t_idx]
                weight = weights[t_idx]
                s_point = source_points[s_idx]
                t_point = target_points[t_idx]
                # print(f"i: {i} t_idx: {t_idx} weight: {weight}")

                valid_pairs.append((s_point, t_point, weight))

            if len(valid_pairs) == 0:
                break  # 如果没有有效的点对，终止迭代

            # 提取有效点对的源点、目标点和权重
            source_valid = np.array([p[0] for p in valid_pairs])  # 获取有效的源点
            target_valid = np.array([p[1] for p in valid_pairs])  # 获取有效的目标点
            weights_valid = np.array([p[2] for p in valid_pairs])  # 获取有效的权重

            # 计算加权的变换矩阵 R 和平移向量 t
            R, t = self.weighted_compute_transformation(source_valid, target_valid, weights_valid)

            # 应用变换矩阵和平移向量到源点云
            source_points = np.dot(source_points, R.T) + t

            # 计算加权误差
            error = np.average(np.linalg.norm(source_valid - target_valid, axis=1), weights=weights_valid)
            
            # print(f"Iteration {i + 1}: Error = {error}")
            if np.abs(prev_error - error) < self.icp_threshold:
                print(f"Converged after {i + 1} iterations.")
                break  # 如果误差变化很小，终止迭代
            prev_error = error  # 更新前一轮的误差

        # 构建变换矩阵
        transformation = np.eye(4)
        transformation[:3, :3] = R
        transformation[:3, 3] = t

        # 将源点云应用变换
        transformed_source_points = transform_points(source_points, R, t)
        
        # 计算均方误差 (MSE)
        rmse = calculate_rmse(transformed_source_points, target_points)
        
        # 计算重叠率
        overlap_ratio = calculate_overlap_ratio(transformed_source_points, target_points)

        # 将源点云转换为Open3D点云格式
        transformed_source_pcd = o3d.geometry.PointCloud()
        transformed_source_pcd.points = o3d.utility.Vector3dVector(transformed_source_points)

        return transformation, transformed_source_pcd, overlap_ratio, rmse, len(valid_pairs)


    def compute_point_density(self, points, k=10):
        """
        计算每个点的局部密度，使用k近邻距离的平均值作为密度估计。
        """
        tree = cKDTree(points)
        densities = []
        for point in points:
            distances, _ = tree.query(point, k=k+1)  # 包含自身点，需加1
            avg_distance = np.mean(distances[1:])  # 排除自身距离（为0）
            density = 1.0 / (avg_distance + 1e-8)  # 避免除以零
            densities.append(density)
        densities = np.array(densities)
        # 将密度归一化
        densities /= np.max(densities)
        return densities

    def icp_fine_registration_with_adaptive_weights(self, source, target, k=10):
        """
        使用自适应权重和鲁棒损失函数的 ICP 精配准方法。
        """
        source_points = np.asarray(source.points)
        target_points = np.asarray(target.points)
        print(f"source_points: {len(source_points)}, target_points: {len(target_points)}")

        # 计算源点云和目标点云的密度权重
        print("Computing point densities for source and target point clouds...")
        source_densities = self.compute_point_density(source_points, k=k)
        target_densities = self.compute_point_density(target_points, k=k)

        # 对密度进行调整，高密度区域的权重降低，低密度区域的权重提高
        # 使用反比关系
        source_density_weights = 1.0 / (source_densities + 1e-8)
        target_density_weights = 1.0 / (target_densities + 1e-8)

        # 将权重归一化
        source_density_weights /= np.max(source_density_weights)
        target_density_weights /= np.max(target_density_weights)

        prev_error = float('inf')  # 初始化前一轮的误差为无穷大

        for iteration in range(50):  # 进行50次迭代
            # 构建 KD 树
            source_tree = cKDTree(source_points)
            distances, indices = source_tree.query(target_points, k=1)

            valid_pairs = []
            for t_idx in range(len(target_points)):
                s_idx = indices[t_idx]
                s_point = source_points[s_idx]
                t_point = target_points[t_idx]

                # 计算残差
                residual = np.linalg.norm(s_point - t_point)
                # print(f"residual: {residual}")

                # 使用 Geman-McClure 损失函数计算权重
                # w = 1 / (1 + (residual / sigma) ** 2)
                sigma = self.threshold  # 可以根据实际情况调整
                weight = 1 / (1 + (residual / sigma) ** 2)

                # 结合密度权重
                density_weight = source_density_weights[s_idx] * target_density_weights[t_idx]
                combined_weight = weight * density_weight

                valid_pairs.append((s_point, t_point, combined_weight))

            if len(valid_pairs) == 0:
                print("No valid pairs found. Exiting ICP loop.")
                break  # 如果没有有效的点对，终止迭代

            # 提取有效点对的源点、目标点和权重
            source_valid = np.array([p[0] for p in valid_pairs])
            target_valid = np.array([p[1] for p in valid_pairs])
            weights_valid = np.array([p[2] for p in valid_pairs])

            # 归一化权重
            weights_valid /= np.sum(weights_valid)

            # 计算加权的变换矩阵 R 和平移向量 t
            R, t = self.weighted_compute_transformation(source_valid, target_valid, weights_valid)

            # 应用变换矩阵和平移向量到源点云
            source_points = np.dot(source_points, R.T) + t

            # 计算加权误差
            residuals = np.linalg.norm(source_valid - target_valid, axis=1)
            geman_mcclure_losses = residuals ** 2 / (sigma ** 2 + residuals ** 2)
            error = np.sum(geman_mcclure_losses * weights_valid)

            print(f"Iteration {iteration + 1}: Error = {error}")
            if np.abs(prev_error - error) < self.icp_threshold:
                print(f"Converged after {iteration + 1} iterations.")
                break  # 如果误差变化很小，终止迭代
            prev_error = error  # 更新前一轮的误差

        # 构建变换矩阵
        transformation = np.eye(4)
        transformation[:3, :3] = R
        transformation[:3, 3] = t

        # 将源点云转换为 Open3D 点云格式
        transformed_source_pcd = o3d.geometry.PointCloud()
        transformed_source_pcd.points = o3d.utility.Vector3dVector(source_points)

        # 计算均方误差 (RMSE)
        rmse = calculate_rmse(np.asarray(transformed_source_pcd.points), target_points)

        # 计算重叠率
        overlap_ratio = calculate_overlap_ratio(np.asarray(transformed_source_pcd.points), target_points)

        return transformation, transformed_source_pcd, overlap_ratio, rmse, len(valid_pairs)


    def icp_fine_registration_curve(self, source, target):
        converged = False
        source_points = np.asarray(source.points)  # 转换源点云为NumPy数组
        target_points = np.asarray(target.points)  # 转换目标点云为NumPy数组
        print(f"source_points: {len(source_points)}, target_points: {len(target_points)}")
        source_tangents = self.compute_tangents(source_points)  # 计算源点云的切线
        target_tangents = self.compute_tangents(target_points)  # 计算目标点云的切线

        prev_error = float('inf')  # 初始化前一轮的误差为无穷大
        for i in range(50):  # 进行50次迭代
            source_tangents = self.compute_tangents(source_points)  # 更新切线
            target_tangents = self.compute_tangents(target_points)  # 更新切线
            
            tree = cKDTree(source_points)  # 构建目标点云的KD树
            distances, indices = tree.query(target_points, k=1)  # 查找每个源点最近的目标点

            # 创建一个标记数组，确保每个target_point最多被配对一次
            used = np.zeros(len(target_points), dtype=bool)
            valid_pairs = []
            # closest_points = target_points[indices]  # 找到最近的目标点
            # closest_tangents = target_tangents[indices]  # 找到最近的目标点的切线

            for t_idx, s_idx in enumerate(indices):
                if used[t_idx]:
                    continue  # 如果target_point已被配对，跳过该对
                s_point = source_points[s_idx]
                t_point = target_points[t_idx]
                s_tangent = source_tangents[s_idx]
                t_tangent = target_tangents[t_idx]
                
                angle = np.arccos(np.clip(np.dot(s_tangent, t_tangent), -1.0, 1.0))  # 计算两个切线向量之间的夹角
                if angle < self.angle_threshold:  # 如果夹角小于角度阈值，认为是有效点对
                    valid_pairs.append((s_point, t_point))
                    used[t_idx] = True  # 标记target_point为已使用

            # self.valid_pairs = self.filter_pairs_by_tangent(source_points, source_tangents, closest_points, closest_tangents)  # 过滤掉不满足角度约束的点对
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
            if np.abs(prev_error - error) < self.icp_threshold:
                print(f" times: {i+ 1} Converged!")
                converged = True
                break  # 如果误差变化很小，终止迭代
            prev_error = error  # 更新前一轮的误差
        if converged is False:
            print("Not converged!")

        transformation = np.eye(4)  # 初始化4x4的变换矩阵为单位矩阵
        transformation[:3, :3] = R  # 将旋转矩阵R赋值到变换矩阵的左上角3x3部分
        transformation[:3, 3] = t  # 将平移向量t赋值到变换矩阵的第4列前三行

        # 将源点云的点进行变换
        transformed_source_points = transform_points(source_points, R, t)
        # 计算均方误差 (MSE)
        rmse = calculate_rmse(transformed_source_points, target_points)
        # 计算重叠率
        overlap_ratio = calculate_overlap_ratio(transformed_source_points, target_points)

        # 将 transformed_source_points 转换回 Open3D 点云格式
        transformed_source_pcd = o3d.geometry.PointCloud()
        transformed_source_pcd.points = o3d.utility.Vector3dVector(transformed_source_points)

        return transformation, transformed_source_pcd, overlap_ratio, rmse, len(valid_pairs)  # 返回变换矩阵，最终误差和有效点对的数量
    
    
    def compute_tangents(self, points):
        tangents = []
        for i in range(1, len(points) - 1):
            tangent = (points[i + 1] - points[i - 1]) / 2  # 计算切线
            tangent /= np.linalg.norm(tangent)  # 归一化切线向量
            tangents.append(tangent)
        tangents = [tangents[0]] + tangents + [tangents[-1]]  # 补充第一个和最后一个切线向量
        return np.array(tangents)  # 返回切线向量数组


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
    
    
    def weighted_compute_transformation(self, source, target, weights):
        # 计算加权均值
        source_mean = np.average(source, axis=0, weights=weights)
        target_mean = np.average(target, axis=0, weights=weights)

        # 去中心化
        source_centered = source - source_mean
        target_centered = target - target_mean

        # 计算加权协方差矩阵
        W = np.diag(weights)
        H = source_centered.T @ W @ target_centered

        # 使用SVD求解R
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # 确保R是一个有效的旋转矩阵
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # 计算平移向量t
        t = target_mean - R @ source_mean

        return R, t

