import open3d as o3d
import numpy as np

def load_point_cloud(file_path):
    data = np.loadtxt(file_path)
    points = data[:, :3]
    colors = data[:, 3:6] / 255.0 if data.shape[1] > 3 else None
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        point_cloud.colors = o3d.utility.Vector3dVector(colors)
    return point_cloud

def apply_pca(point_cloud):
    # 计算点云的主成分
    mean = np.mean(point_cloud.points, axis=0)
    point_cloud_centered = point_cloud.points - mean
    u, s, vh = np.linalg.svd(point_cloud_centered, full_matrices=False)
    transformation = np.eye(4)
    transformation[:3, :3] = vh
    transformation[:3, 3] = mean
    return transformation

def align_point_cloud(point_cloud, transformation):
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

def compute_centroid(point_cloud):
    #计算质心
    points = np.asarray(point_cloud.points)
    centroid = np.mean(points, axis=0)
    return centroid

def translate_point_cloud(point_cloud, vector):
    #平移向量
    translated_points = np.asarray(point_cloud.points) + vector
    point_cloud.points = o3d.utility.Vector3dVector(translated_points)
    return point_cloud

def fine_registration(source, target, threshold):
    trans_init = np.eye(4)
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    return reg_p2p.transformation

def evaluate_registration(source, target, transformation, threshold):
    evaluation = o3d.pipelines.registration.evaluate_registration(
        source, target, threshold, transformation)
    return evaluation

def create_local_axis(point_cloud, size=0.01):
    centroid = np.mean(np.asarray(point_cloud.points), axis=0)
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=centroid)
    return axis


# 加载点云
pc1 = load_point_cloud("/home/daichang/Desktop/ros2_ws/src/markless-calibration/pcd/wait-to-reg/edge_teeth.txt")
pc2 = load_point_cloud("/home/daichang/Desktop/ros2_ws/src/markless-calibration/pcd/wait-to-reg/teethtrue-seg.txt")

pc3 = load_point_cloud("/home/daichang/Desktop/ros2_ws/src/markless-calibration/pcd/wait-to-reg/teethscan.txt")
pc4 = load_point_cloud("/home/daichang/Desktop/ros2_ws/src/markless-calibration/pcd/wait-to-reg/teethtrue.txt")

###################################################################### 原始点云 ##################################################################################
pc1.paint_uniform_color([1, 0, 0])  # 红色表示源点云
pc2.paint_uniform_color([0, 1, 0])  # 绿色表示目标点云
# 添加坐标轴
axis_pc = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01)
axis_pc1 = create_local_axis(pc1)
axis_pc2 = create_local_axis(pc2)


# o3d.visualization.draw_geometries([pc1, pc2])
vis = o3d.visualization.Visualizer()
vis.create_window(window_name='原始点云', width=800, height=600)
vis.add_geometry(pc1)
vis.add_geometry(pc2)
# 添加坐标轴
vis.add_geometry(axis_pc)
vis.add_geometry(axis_pc1)
vis.add_geometry(axis_pc2)
vis.run()
vis.destroy_window()

############################################################# 粗配准  ##################################################

# 应用 PCA 进行配准
transformation1 = apply_pca(pc1)
transformation2 = apply_pca(pc2)
pc1_aligned = align_point_cloud(pc1, transformation1)
pc2_aligned = align_point_cloud(pc2, transformation2)
#计算平移向量
centroid_pc1 = compute_centroid(pc1_aligned)
centroid_pc2 = compute_centroid(pc2_aligned)
translation_vector = centroid_pc2 - centroid_pc1
pc1_aligned = translate_point_cloud(pc1_aligned, translation_vector)




print("原始点云的 PCA 结果：")
print(f"PCA1 结果：\n{transformation1}")
print(f"PCA2 结果：\n{transformation2}")
print(f"平移向量：\n{translation_vector}")


# 评估粗配准结果
threshold = 0.02  # 设置一个合适的阈值
evaluation_coarse = evaluate_registration(pc1_aligned, pc2_aligned, np.eye(4), threshold)
print("粗配准后的评估结果：")
print(f"RMSE: {evaluation_coarse.inlier_rmse}")
print(f"Fitness: {evaluation_coarse.fitness}")

# 为粗配准后的点云添加局部坐标轴
axis_pc1_aligned = create_local_axis(pc1_aligned)
axis_pc2_aligned = create_local_axis(pc2_aligned)


# 可视化粗配准结果
pc1_aligned.paint_uniform_color([1, 0, 0])  # 红色表示源点云
pc2_aligned.paint_uniform_color([0, 1, 0])  # 绿色表示目标点云


vis = o3d.visualization.Visualizer()
vis.create_window(window_name='粗配准结果', width=800, height=600)
vis.add_geometry(pc1_aligned)
vis.add_geometry(pc2_aligned)

vis.add_geometry(axis_pc)
vis.add_geometry(axis_pc1_aligned)
vis.add_geometry(axis_pc2_aligned)
vis.run()
vis.destroy_window()

############################################################## 精配准 ###################################################3

# 使用ICP进行精细配准
transformation_icp = fine_registration(pc1_aligned, pc2_aligned, threshold)
pc1_aligned.transform(transformation_icp)

print("精配准后的变换矩阵：")
print(f"{transformation_icp}")

# 评估精配准结果
evaluation_fine = evaluate_registration(pc1_aligned, pc2_aligned, transformation_icp, threshold)
print("精配准后的评估结果：")
print(f"RMSE: {evaluation_fine.inlier_rmse}")
print(f"Fitness: {evaluation_fine.fitness}")

# 为精配准后的点云添加局部坐标轴
axis_pc1_aligned_icp = create_local_axis(pc1_aligned)
axis_pc2_aligned_icp = create_local_axis(pc2_aligned)

# 可视化配准结果
# o3d.visualization.draw_geometries([pc1_aligned, pc2_aligned])
vis = o3d.visualization.Visualizer()
vis.create_window(window_name='精配准结果', width=800, height=600)
vis.add_geometry(pc1_aligned)
vis.add_geometry(pc2_aligned)
vis.add_geometry(axis_pc)
vis.add_geometry(axis_pc1_aligned_icp)
vis.add_geometry(axis_pc2_aligned_icp)
vis.run()
vis.destroy_window()

#######################################################验证######################################################
pc3_aligned = align_point_cloud(pc3, transformation1)
pc3_aligned = translate_point_cloud(pc3_aligned, translation_vector)
pc4_aligned = align_point_cloud(pc4, transformation2)
axis_pc3_aligned = create_local_axis(pc3_aligned)
axis_pc4_aligned = create_local_axis(pc4_aligned)

# 可视化粗配准结果
pc3_aligned.paint_uniform_color([1, 0, 0])  # 红色表示源点云
pc4_aligned.paint_uniform_color([0, 1, 0])  # 绿色表示目标点云
vis = o3d.visualization.Visualizer()
vis.create_window(window_name='验证粗配准结果', width=800, height=600)
vis.add_geometry(pc3_aligned)
vis.add_geometry(pc4_aligned)
vis.add_geometry(axis_pc)
vis.add_geometry(axis_pc3_aligned)
vis.add_geometry(axis_pc4_aligned)
vis.run()
vis.destroy_window()

pc3_aligned.transform(transformation_icp)
# 为精配准后的点云添加局部坐标轴
axis_pc3_aligned_icp = create_local_axis(pc3_aligned)
axis_pc4_aligned_icp = create_local_axis(pc4_aligned)

# 可视化配准结果
# o3d.visualization.draw_geometries([pc1_aligned, pc2_aligned])
vis = o3d.visualization.Visualizer()
vis.create_window(window_name='验证精配准结果', width=800, height=600)
vis.add_geometry(pc3_aligned)
vis.add_geometry(pc4_aligned)
vis.add_geometry(axis_pc)
vis.add_geometry(axis_pc3_aligned_icp)
vis.add_geometry(axis_pc4_aligned_icp)
vis.run()
vis.destroy_window()

