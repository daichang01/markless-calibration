import open3d as o3d
import numpy as np
import time
 ############################################## utils ##############################################
def visualize_initial_point_clouds(pc1, pc2, window_name='untitle', width=800, height=600):
    # Set colors for point clouds
    pc1.paint_uniform_color([0, 1, 0])  # Green color for the second point cloud
    # pc2.paint_uniform_color([0, 1, 0])  # 保留原始rgb

    # Create coordinate frames
    axis_pc = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01)
    axis_pc1 = create_local_axis(pc1)
    axis_pc2 = create_local_axis(pc2)

    # Setup the visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=width, height=height)
    vis.add_geometry(pc1)
    vis.add_geometry(pc2)
    vis.add_geometry(axis_pc)
    vis.add_geometry(axis_pc1)
    vis.add_geometry(axis_pc2)
    
    # Run the visualizer
    vis.run()
    vis.destroy_window()

def create_local_axis(point_cloud, size=0.01):
    centroid = np.mean(np.asarray(point_cloud.points), axis=0)
    return o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=centroid)
def load_point_cloud(file_path):
    data = np.loadtxt(file_path)
    points = data[:, :3]
    colors = data[:, 3:6] / 255.0 if data.shape[1] > 3 else None
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        point_cloud.colors = o3d.utility.Vector3dVector(colors)
    return point_cloud


def evaluate_registration(source, target, transformation, threshold):
    evaluation = o3d.pipelines.registration.evaluate_registration(
        source, target,max_correspondence_distance= threshold, transformation= transformation)
    fitness = evaluation.fitness # 重叠区域（内部对应数/源中的点数）。越高越好。
    inlier_rmse = evaluation.inlier_rmse # 所有内部对应关系的 RMSE。越低越好。
    return fitness, inlier_rmse

############################################## FPFH + RANSAC 粗配准 ##############################################
def compute_fpfh_feature(point_cloud, threshold):
    print(":: Estimate normal.")
    radius_normal = threshold * 10
    point_cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    print(":: Compute FPFH feature.")
    radius_feature = threshold * 20
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        point_cloud,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return fpfh

def execute_global_registration(source, target, source_fpfh, target_fpfh, threshold):
    distance_threshold = threshold
    print(":: RANSAC registration on point clouds.")
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source, target, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(20000, 500))
    return result

def fpfh_ransac_coarse_registration(source, target, valsource, valtarget, threshold):
    start_time_ransac= time.time()
    source_fpfh = compute_fpfh_feature(source, threshold)
    target_fpfh = compute_fpfh_feature(target, threshold)
    
    result_ransac = execute_global_registration(source, target, source_fpfh, target_fpfh, threshold)
    
    # 应用 RANSAC 结果变换到原始点云
    fitness, inlier_rmse = evaluate_registration(source, target, result_ransac.transformation, threshold)

    end_time_ransac = time.time()
    ransac_time = end_time_ransac - start_time_ransac
    source.transform(result_ransac.transformation)
    
    print("ransac粗配准后的评估结果：")
    print(f"RMSE: {inlier_rmse}")
    print(f"Fitness: {fitness}")
    print(f"ransac粗配准耗时: {ransac_time} 秒")
    visualize_initial_point_clouds(source, target, window_name='ransac粗配准结果')
    
    valsource.transform(result_ransac.transformation)
    visualize_initial_point_clouds(valsource, valtarget, window_name='验证ransac粗配准结果')
    return source, target, valsource, valtarget

############################################################# pca粗配准  ##################################################
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


def correct_pca_orientation(transformation):
    # transformation 是一个4x4的变换矩阵。transformation[:3, 0] 表示提取该矩阵的第一列的前三个元素（即主轴的方向向量）。在PCA中，第一列通常代表了数据的主轴方向。
    primary_axis = transformation[:3, 0]
    #  检查 primary_axis 的Z分量（primary_axis[2]）是否为负。
    if primary_axis[2] < 0:  # 如果Z分量为负，将整个主轴向量取反。这样可以确保主轴的Z方向为正，避免因为对称性或其他原因导致的主轴方向翻转问题。
        transformation[:3, 0] = -primary_axis
    return transformation

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
def pca_coarse_registration(source, target, valsource, valtarget, threshold):
    # 应用 PCA 进行配准
    start_time_pca= time.time()
    transformation1 = apply_pca(source)
    # transformation1 = correct_pca_orientation(transformation1) #主轴校正
    transformation2 = apply_pca(target)
    # transformation2 = correct_pca_orientation(transformation2) #主轴校正
    source_aligned = align_point_cloud(source, transformation1)
    target_aligned = align_point_cloud(target, transformation2)
    #计算平移向量
    centroid_source = compute_centroid(source_aligned)
    centroid_target = compute_centroid(target_aligned)
    translation_vector = centroid_target - centroid_source
    source_aligned = translate_point_cloud(source_aligned, translation_vector)

    end_time_pca = time.time()
    pca_time = end_time_pca - start_time_pca

    print(f"PCA1 结果：\n{transformation1}")
    print(f"PCA2 结果：\n{transformation2}")
    print(f"平移向量：\n{translation_vector}")
    print(f"pca粗配准耗时: {pca_time} 秒")


    # 评估粗配准结果
    fitness, inlier_rmse = evaluate_registration(source_aligned, target_aligned, np.eye(4), threshold)
    print("粗配准后的评估结果：")
    print(f"RMSE: {inlier_rmse}")
    print(f"Fitness: {fitness}")
    visualize_initial_point_clouds(source_aligned, target_aligned, window_name='粗配准结果')

    valsource_aligned = align_point_cloud(valsource, transformation1)
    valsource_aligned = translate_point_cloud(valsource, translation_vector)
    valtarget_aligned = align_point_cloud(valtarget, transformation2)
    visualize_initial_point_clouds(valsource_aligned, valtarget_aligned, window_name='验证粗配准结果')
    return source_aligned, target_aligned, valsource_aligned, valtarget_aligned


############################################ icp精配准 ###################################################################
def icp_fine_registration(source, target, valsource, valtarget, threshold = 0.02):
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

    visualize_initial_point_clouds(source, target, window_name='icp精配准结果')
    valsource.transform(transformation_icp)
    visualize_initial_point_clouds(valsource, valtarget, window_name='验证icp精配准结果')


############################################ start #########################################################
def main():
    threshold = 0.001
    print("测试1")
    source = load_point_cloud("/home/daichang/Desktop/ros2_ws/src/markless-calibration/pcd/wait-to-reg/lowfrontscan.txt")
    target = load_point_cloud("/home/daichang/Desktop/ros2_ws/src/markless-calibration/pcd/auto-seg/lowteeth/lowsegfilter0608_230559.txt")
    valsource = load_point_cloud("/home/daichang/Desktop/ros2_ws/src/markless-calibration/pcd/wait-to-reg/newteeth_scantoval.txt")
    valtarget = load_point_cloud("/home/daichang/Desktop/ros2_ws/src/markless-calibration/pcd/auto-seg/valteeth/roifilter0608_230559.txt")
    visualize_initial_point_clouds(source, target, window_name='原始点云')
    # s, t, vs, vt = pca_coarse_registration(source, target, valsource, valtarget, threshold) # pca粗配准
    s, t, vs, vt = fpfh_ransac_coarse_registration(source, target, valsource, valtarget, threshold) # ransac粗配准
    icp_fine_registration(source = s, target=t ,valsource=vs, valtarget= vt, threshold=threshold) # 精配准

    # print("测试2")
    # pc1 = load_point_cloud("/home/daichang/Desktop/ros2_ws/src/markless-calibration/pcd/wait-to-reg/edge_teeth.txt")
    # pc2 = load_point_cloud("/home/daichang/Desktop/ros2_ws/src/markless-calibration/pcd/wait-to-reg/teethtrue-seg.txt")
    # pc3 = load_point_cloud("/home/daichang/Desktop/ros2_ws/src/markless-calibration/pcd/wait-to-reg/teethscan.txt")
    # pc4 = load_point_cloud("/home/daichang/Desktop/ros2_ws/src/markless-calibration/pcd/wait-to-reg/teethtrue.txt")
    # visualize_initial_point_clouds(pc1, pc2, window_name='原始点云')
    # # s, t, vs, vt = pca_coarse_registration(pc1, pc2, pc3, pc4, threshold) # pca粗配准
    # s, t, vs, vt = fpfh_ransac_coarse_registration(pc1, pc2, pc3, pc4, threshold) # ransac粗配准
    # icp_fine_registration(source = s, target=t ,valsource=vs, valtarget= vt, threshold=threshold) # 精配准

if __name__ == "__main__":
    main()





