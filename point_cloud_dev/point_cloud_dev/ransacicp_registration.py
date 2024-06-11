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

def compute_fpfh_feature(point_cloud):
    radius_normal = 0.02  # 法线估计的半径
    radius_feature = 0.1  # FPFH特征的半径
    point_cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        point_cloud,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )
    return fpfh

def execute_global_registration(source, target, source_fpfh, target_fpfh, distance_threshold):
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source, target, source_fpfh, target_fpfh,
        mutual_filter=True,  # 使用互相关滤波
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                  o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
    )
    return result.transformation

def fine_registration(source, target, threshold):
    trans_init = np.eye(4)
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    return reg_p2p.transformation

def evaluate_registration(source, target, transformation, threshold):
    evaluation = o3d.pipelines.registration.evaluate_registration(
        source, target, threshold, transformation)
    
    source_transformed = source.transform(transformation)
    distances = source_transformed.compute_point_cloud_distance(target)
    distances = np.asarray(distances)

    rmse = np.sqrt(np.mean(distances ** 2))
    mae = np.mean(np.abs(distances))

    return evaluation, rmse, mae

def create_local_axis(point_cloud, size=0.01):
    centroid = np.mean(np.asarray(point_cloud.points), axis=0)
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=centroid)
    return axis

def align_point_cloud(point_cloud, transformation):
    points = np.asarray(point_cloud.points)
    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
    points_transformed = np.dot(points_homogeneous, transformation.T)
    points_transformed = points_transformed[:, :3]
    point_cloud.points = o3d.utility.Vector3dVector(points_transformed)
    return point_cloud

# 加载点云
pc1 = load_point_cloud("/home/daichang/Desktop/ros2_ws/src/markless-calibration/pcd/wait-to-reg/lowfrontscan.txt")
pc2 = load_point_cloud("/home/daichang/Desktop/ros2_ws/src/markless-calibration/pcd/auto-seg/lowteeth/lowsegfilter0608_230559.txt")

pc3 = load_point_cloud("/home/daichang/Desktop/ros2_ws/src/markless-calibration/pcd/wait-to-reg/newteeth_scantoval.txt")
pc4 = load_point_cloud("/home/daichang/Desktop/ros2_ws/src/markless-calibration/pcd/auto-seg/valteeth/roifilter0608_230559.txt")

###################################################################### 原始点云 ##################################################################################
pc1.paint_uniform_color([1, 0, 0])  # 红色表示源点云
pc2.paint_uniform_color([0, 1, 0])  # 绿色表示目标点云
# 添加坐标轴
axis_pc = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01)
axis_pc1 = create_local_axis(pc1)
axis_pc2 = create_local_axis(pc2)

vis = o3d.visualization.Visualizer()
vis.create_window(window_name='原始点云', width=800, height=600)
vis.add_geometry(pc1)
vis.add_geometry(pc2)
vis.add_geometry(axis_pc)
vis.add_geometry(axis_pc1)
vis.add_geometry(axis_pc2)
vis.run()
vis.destroy_window()

############################################################# 粗配准  ##################################################

# 计算FPFH特征
source_fpfh = compute_fpfh_feature(pc1)
target_fpfh = compute_fpfh_feature(pc2)

# 基于RANSAC的粗配准
distance_threshold = 0.05  # 根据数据集特性调整
transformation_ransac = execute_global_registration(pc1, pc2, source_fpfh, target_fpfh, distance_threshold)
pc1_aligned = align_point_cloud(pc1, transformation_ransac)

print("RANSAC配准后的变换矩阵：")
print(f"{transformation_ransac}")

# 评估粗配准结果
threshold = 0.02  # 设置一个合适的阈值
evaluation_coarse, rmse_coarse, mae_coarse = evaluate_registration(pc1_aligned, pc2, np.eye(4), threshold)
print("粗配准后的评估结果：")
print(f"RMSE: {evaluation_coarse.inlier_rmse}, RMSE (Custom): {rmse_coarse}")
print(f"MAE (Custom): {mae_coarse}")
# print(f"Fitness: {evaluation_coarse.fitness}")

# 为粗配准后的点云添加局部坐标轴
axis_pc1_aligned = create_local_axis(pc1_aligned)
axis_pc2_aligned = create_local_axis(pc2)

# 可视化粗配准结果
pc1_aligned.paint_uniform_color([1, 0, 0])  # 红色表示源点云
pc2.paint_uniform_color([0, 1, 0])  # 绿色表示目标点云

vis = o3d.visualization.Visualizer()
vis.create_window(window_name='粗配准结果', width=800, height=600)
vis.add_geometry(pc1_aligned)
vis.add_geometry(pc2)
vis.add_geometry(axis_pc)
vis.add_geometry(axis_pc1_aligned)
vis.add_geometry(axis_pc2_aligned)
vis.run()
vis.destroy_window()

############################################################## 精配准 ###################################################

# 使用ICP进行精细配准
transformation_icp = fine_registration(pc1_aligned, pc2, threshold)
pc1_aligned.transform(transformation_icp)

print("精配准后的变换矩阵：")
print(f"{transformation_icp}")

# 评估精配准结果
evaluation_fine, rmse_fine, mae_fine = evaluate_registration(pc1_aligned, pc2, transformation_icp, threshold)
print("精配准后的评估结果：")
print(f"RMSE: {evaluation_fine.inlier_rmse}, RMSE (Custom): {rmse_fine}")
print(f"MAE (Custom): {mae_fine}")
# print(f"Fitness: {evaluation_fine.fitness}")

# 为精配准后的点云添加局部坐标轴
axis_pc1_aligned_icp = create_local_axis(pc1_aligned)
axis_pc2_aligned_icp = create_local_axis(pc2)

# 可视化精配准结果
vis = o3d.visualization.Visualizer()
vis.create_window(window_name='精配准结果', width=800, height=600)
vis.add_geometry(pc1_aligned)
vis.add_geometry(pc2)
vis.add_geometry(axis_pc)
vis.add_geometry(axis_pc1_aligned_icp)
vis.add_geometry(axis_pc2_aligned_icp)
vis.run()
vis.destroy_window()

####################################################### 验证 ######################################################
pc3_aligned = align_point_cloud(pc3, transformation_ransac)
pc4_aligned = pc4  # 目标点云不变
axis_pc3_aligned = create_local_axis(pc3_aligned)
axis_pc4_aligned = create_local_axis(pc4_aligned)

# 可视化粗配准结果
pc3_aligned.paint_uniform_color([0,1, 0])  # 红色表示源点云
# pc4_aligned.paint_uniform_color([0, 1, 0])  # 绿色表示目标点云
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

# 可视化精配准结果
vis = o3d.visualization.Visualizer()
vis.create_window(window_name='验证精配准结果', width=800, height=600)
vis.add_geometry(pc3_aligned)
vis.add_geometry(pc4_aligned)
vis.add_geometry(axis_pc)
vis.add_geometry(axis_pc3_aligned_icp)
vis.add_geometry(axis_pc4_aligned_icp)
vis.run()
vis.destroy_window()
