import open3d as o3d
import numpy as np
import time

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

def compute_fpfh_feature(point_cloud, radius_normal, radius_feature):
    point_cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )

    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        point_cloud,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )

    return fpfh

def extract_keypoints_by_feature(feature, threshold=0.01):
    # feature.data 是一个 numpy 数组，代表所有点的特征描述子。 .T 是转置操作，转换特征矩阵的行列。这样做的目的是将每个点的特征向量作为矩阵的行。
    feature_values = np.asarray(feature.data.T) 
    # 算每个点的特征向量的范数（即特征向量的长度或大小）。 np.linalg.norm 函数计算矩阵沿指定轴的范数。这里 axis=1 指的是沿行方向计算每行的范数。
    feature_norms = np.linalg.norm(feature_values, axis=1) 
    # 筛选出特征向量范数大于阈值的点的索引。 np.where 函数返回满足条件的元素的索引，这里返回的是范数大于 threshold 的点的索引。
    keypoints_indices = np.where(feature_norms > threshold)[0]
    return keypoints_indices

def apply_pca(point_cloud):
    mean = np.mean(point_cloud.points, axis=0)
    point_cloud_centered = point_cloud.points - mean
    u, s, vh = np.linalg.svd(point_cloud_centered, full_matrices=False)
    transformation = np.eye(4)
    transformation[:3, :3] = vh
    transformation[:3, 3] = mean
    return transformation

def align_point_cloud(point_cloud, transformation):
    points = np.asarray(point_cloud.points)
    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
    points_transformed = np.dot(points_homogeneous, transformation.T)
    points_transformed = points_transformed[:, :3]
    point_cloud.points = o3d.utility.Vector3dVector(points_transformed)
    return point_cloud

def compute_centroid(point_cloud):
    points = np.asarray(point_cloud.points)
    centroid = np.mean(points, axis=0)
    return centroid

def translate_point_cloud(point_cloud, vector):
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


def correct_pca_orientation(transformation):
    # transformation 是一个4x4的变换矩阵。transformation[:3, 0] 表示提取该矩阵的第一列的前三个元素（即主轴的方向向量）。在PCA中，第一列通常代表了数据的主轴方向。
    primary_axis = transformation[:3, 0]
    #  检查 primary_axis 的Z分量（primary_axis[2]）是否为负。
    if primary_axis[2] < 0:  # 如果Z分量为负，将整个主轴向量取反。这样可以确保主轴的Z方向为正，避免因为对称性或其他原因导致的主轴方向翻转问题。
        transformation[:3, 0] = -primary_axis
    return transformation



####################################################### main #########################################################
def main():
    # 加载点云
    pc1 = load_point_cloud("/home/daichang/Desktop/ros2_ws/src/markless-calibration/pcd/wait-to-reg/lowfrontscan.txt")
    pc2 = load_point_cloud("/home/daichang/Desktop/ros2_ws/src/markless-calibration/pcd/auto-seg/lowteeth/lowsegfilter0608_230559.txt")

    pc3 = load_point_cloud("/home/daichang/Desktop/ros2_ws/src/markless-calibration/pcd/wait-to-reg/newteeth_scantoval.txt")
    pc4 = load_point_cloud("/home/daichang/Desktop/ros2_ws/src/markless-calibration/pcd/auto-seg/valteeth/roifilter0608_230559.txt")

   

    visualize_initial_point_clouds(pc1, pc2, window_name='原始点云')



    ############################################################# 粗配准  ##################################################
     # 设置法线和特征计算半径
    radius_normal = 0.005  # 根据点云的实际尺寸进行调整
    radius_feature = 0.01  # 根据点云的实际尺寸进行调整

    # 计算FPFH特征
    fpfh1 = compute_fpfh_feature(pc1, radius_normal, radius_feature)
    fpfh2 = compute_fpfh_feature(pc2, radius_normal, radius_feature)

    # 提取FPFH特征明显的点
    fpfh_keypoints1 = extract_keypoints_by_feature(fpfh1, threshold=0.01)
    fpfh_keypoints2 = extract_keypoints_by_feature(fpfh2, threshold=0.01)
    pc1_fpfh_keypoints = pc1.select_by_index(fpfh_keypoints1)
    pc2_fpfh_keypoints = pc2.select_by_index(fpfh_keypoints2)

    # 应用PCA进行配准
    start_time_pca = time.time()
    transformation1 = apply_pca(pc1_fpfh_keypoints)
    transformation1 = correct_pca_orientation(transformation1) #主轴校正
    transformation2 = apply_pca(pc2_fpfh_keypoints)
    transformation2 = correct_pca_orientation(transformation2) #主轴校正
    pc1_aligned = align_point_cloud(pc1, transformation1)
    pc2_aligned = align_point_cloud(pc2, transformation2)
    centroid_pc1 = compute_centroid(pc1_aligned)
    centroid_pc2 = compute_centroid(pc2_aligned)
    translation_vector = centroid_pc2 - centroid_pc1
    pc1_aligned = translate_point_cloud(pc1_aligned, translation_vector)
    end_time_pca = time.time()
    pca_time = end_time_pca - start_time_pca

    print("原始点云的PCA结果：")
    print(f"PCA1结果：\n{transformation1}")
    print(f"PCA2结果：\n{transformation2}")
    print(f"平移向量：\n{translation_vector}")
    print(f"PCA粗配准耗时: {pca_time}秒")

    # 评估粗配准结果
    threshold = 0.02  # 设置一个合适的阈值
    evaluation_coarse = evaluate_registration(pc1_aligned, pc2_aligned, np.eye(4), threshold)
    print("粗配准后的评估结果：")
    print(f"RMSE: {evaluation_coarse.inlier_rmse}")
    print(f"Fitness: {evaluation_coarse.fitness}")

    visualize_initial_point_clouds(pc1_aligned, pc2_aligned, window_name='粗配准结果')


    ############################################################## 精配准 ###################################################3

    # 使用ICP进行精细配准
    start_time_icp = time.time()
    transformation_icp = fine_registration(pc1_aligned, pc2_aligned, threshold)
    pc1_aligned.transform(transformation_icp)
    end_time_icp = time.time()
    icp_time = end_time_icp - start_time_icp

    print("精配准后的变换矩阵：")
    print(f"{transformation_icp}")
    
    # 评估精配准结果
    evaluation_fine = evaluate_registration(pc1_aligned, pc2_aligned, transformation_icp, threshold)
    print("精配准后的评估结果：")
    print(f"RMSE: {evaluation_fine.inlier_rmse}")
    print(f"Fitness: {evaluation_fine.fitness}")
    print(f"icp精配准耗时: {icp_time}秒")

    visualize_initial_point_clouds(pc1_aligned, pc2_aligned, window_name='精配准结果')


    ####################################################### 验证粗配准 ######################################################
    pc3_aligned = align_point_cloud(pc3, transformation1)
    pc3_aligned = translate_point_cloud(pc3_aligned, translation_vector)
    pc4_aligned = align_point_cloud(pc4, transformation2)
    visualize_initial_point_clouds(pc3_aligned, pc4_aligned, window_name='验证粗配准结果')

    ################################################ 验证精配准 #############################################################

    pc3_aligned.transform(transformation_icp)
    visualize_initial_point_clouds(pc3_aligned, pc4_aligned, window_name='验证精配准结果')


if __name__ == '__main__':
    main()
