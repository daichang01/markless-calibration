from .utils import *

class ICPRegistration:

    def __init__(self, threshold=0.001, angle_threshold=np.pi / 6):
        self.threshold = threshold  # 设置距离阈值
        self.angle_threshold = angle_threshold  # 设置角度阈值
    def icp_fine_registration(self, source, target, threshold=0.02):
        trans_init = np.eye(4)
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source, target, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())
        transformation_icp = reg_p2p.transformation

        source.transform(transformation_icp)
        transformed_source_points = np.asarray(source.points)
        target_points = np.asarray(target.points)

        # 评估精配准结果
        # fitness, inlier_rmse = evaluate_registration(source, target, transformation_icp, threshold)
        mse = calculate_mse(transformed_source_points, target_points)
        overlap_ratio = calculate_overlap_ratio(transformed_source_points, target_points)


        # visualize_initial_point_clouds(transformed_source_points, target, "icp_registration")
       
        return transformation_icp, overlap_ratio, mse
    
    def icp_fine_registration_curve(self, source, target):
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
 

