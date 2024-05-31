import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
import matplotlib.pyplot as plt

def read_point_cloud(file_path):
    data = np.loadtxt(file_path)
    points = data[:, :3]
    colors = data[:, 3:6] / 255.0 if data.shape[1] > 3 else None
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        point_cloud.colors = o3d.utility.Vector3dVector(colors)
    return point_cloud

def rotate_point_cloud(point_cloud, angle=np.pi):
    # 创建旋转矩阵（绕Z轴旋转180度）
    R = np.array([[np.cos(angle), -np.sin(angle), 0],
                  [np.sin(angle), np.cos(angle), 0],
                  [0, 0, 1]])
    # 应用旋转变换
    point_cloud.rotate(R, center=(0, 0, 0))
    return point_cloud

def find_correspondences(source_points, target_points):
    kd_tree = KDTree(target_points)
    distances, indices = kd_tree.query(source_points)
    return indices

def compute_optimal_transform(source_points, target_points):
    source_centroid = np.mean(source_points, axis=0)
    target_centroid = np.mean(target_points, axis=0)

    source_centered = source_points - source_centroid
    target_centered = target_points - target_centroid

    H = source_centered.T @ target_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = target_centroid - R @ source_centroid
    return R, t

def icp_point_to_point(source_cloud, target_cloud, max_iterations=100, tolerance=1e-6):
    source_points = np.asarray(source_cloud.points)
    target_points = np.asarray(target_cloud.points)

    transform = np.eye(4)

    for i in range(max_iterations):
        correspondences = find_correspondences(source_points, target_points)
        matched_target_points = target_points[correspondences]

        R, t = compute_optimal_transform(source_points, matched_target_points)

        new_transform = np.eye(4)
        new_transform[:3, :3] = R
        new_transform[:3, 3] = t
        transform = new_transform @ transform

        source_points = (R @ source_points.T + t[:, np.newaxis]).T

        if np.linalg.norm(new_transform - np.eye(4)) < tolerance:
            break
    # transform = np.loadtxt('transform_matrix.txt')

    source_cloud.transform(transform)
    return transform, source_cloud

def compute_rmse(source_cloud, target_cloud):
    source_points = np.asarray(source_cloud.points)
    target_points = np.asarray(target_cloud.points)
    
    correspondences = find_correspondences(source_points, target_points)
    matched_target_points = target_points[correspondences]
    
    distances = np.linalg.norm(source_points - matched_target_points, axis=1)
    rmse = np.sqrt(np.mean(distances ** 2))
    return rmse

def plot_error_histogram(source_cloud, target_cloud):
    source_points = np.asarray(source_cloud.points)
    target_points = np.asarray(target_cloud.points)
    
    correspondences = find_correspondences(source_points, target_points)
    matched_target_points = target_points[correspondences]
    
    distances = np.linalg.norm(source_points - matched_target_points, axis=1)
    
    plt.hist(distances, bins=50)
    plt.title('Registration Error Histogram')
    plt.xlabel('Error (distance)')
    plt.ylabel('Frequency')
    plt.show()

def compute_overlap(source_cloud, target_cloud, threshold=0.01):
    source_points = np.asarray(source_cloud.points)
    target_points = np.asarray(target_cloud.points)
    
    correspondences = find_correspondences(source_points, target_points)
    matched_target_points = target_points[correspondences]
    
    distances = np.linalg.norm(source_points - matched_target_points, axis=1)
    overlap = np.sum(distances < threshold) / len(source_points)
    return overlap

def feature_based_registration(source_cloud, target_cloud):
    # 下采样点云以加速特征计算，并使点云密度相似
    source_down = source_cloud.voxel_down_sample(voxel_size=0.05)
    target_down = target_cloud.voxel_down_sample(voxel_size=0.05)

    # 计算法线
    source_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=50))
    target_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=50))

    # 计算FPFH特征
    source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        source_down, o3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=100))
    target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        target_down, o3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=100))

    # 使用RANSAC进行全局配准
    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        0.075,  # 增大匹配距离阈值
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        4,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(0.075)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))

    return result_ransac.transformation

# 粗略对齐函数
def rough_align(source_cloud, target_cloud):
    # 使用边界框进行粗略对齐
    source_bbox = source_cloud.get_axis_aligned_bounding_box()
    target_bbox = target_cloud.get_axis_aligned_bounding_box()

    source_center = source_bbox.get_center()
    target_center = target_bbox.get_center()

    initial_translation = target_center - source_center

    initial_transform = np.eye(4)
    initial_transform[:3, 3] = initial_translation

    source_cloud.transform(initial_transform)
    return initial_transform

def main():
    # 读取点云数据
    source_path ="/home/daichang/Desktop/ros2_ws/src/markless-calibration/pcd/wait-to-reg/edge_teeth.txt"  # 源点云文件路径
    target_path ="/home/daichang/Desktop/ros2_ws/src/markless-calibration/pcd/wait-to-reg/teethtrue-seg.txt"  # 目标点云文件路径
  
    source_cloud = read_point_cloud(source_path)
    target_cloud = read_point_cloud(target_path)

    # 旋转目标点云180度
    # target_cloud = rotate_point_cloud(target_cloud, angle=np.pi)

    # 创建并添加坐标系
    source_coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01, origin=source_cloud.get_center())
    target_coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01, origin=target_cloud.get_center())
    all_coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.02, origin=[0, 0, 0])

    # 创建可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # 添加几何对象
    vis.add_geometry(source_cloud)
    vis.add_geometry(target_cloud)
    vis.add_geometry(source_coordinate_frame)  # 添加源点云坐标系
    vis.add_geometry(target_coordinate_frame)  # 添加目标点云坐标系
    vis.add_geometry(all_coordinate_frame)  # 添加目标点云坐标系


    # 开始可视化
    vis.run()
    vis.destroy_window()

    # 粗略对齐
    rough_initial_transform = rough_align(source_cloud, target_cloud)

    # 粗配准
    initial_transform = feature_based_registration(source_cloud, target_cloud)
    source_cloud.transform(initial_transform)

    # 进行点对点ICP精配准
    transform, aligned_cloud = icp_point_to_point(source_cloud, target_cloud)



    # 创建并添加配准后的坐标系
    aligned_source_coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01, origin=aligned_cloud.get_center())
    aligned_target_coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01, origin=target_cloud.get_center())
    all_coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.02, origin=[0, 0, 0])

    # 打印最终的变换矩阵
    print("Final Transformation Matrix:")
    print(transform)

    # 评估配准结果
    rmse = compute_rmse(aligned_cloud, target_cloud)
    print("RMSE after alignment:", rmse)

    overlap = compute_overlap(aligned_cloud, target_cloud)
    print("Overlap ratio after alignment:", overlap)
    plot_error_histogram(aligned_cloud, target_cloud)

    # 显示配准后的点云和坐标轴
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    aligned_cloud.paint_uniform_color([1, 0, 0])  # 红色表示源点云
    target_cloud.paint_uniform_color([0, 1, 0])  # 绿色表示目标点云

    # 添加几何对象
    vis.add_geometry(aligned_cloud)
    vis.add_geometry(target_cloud)
    vis.add_geometry(aligned_source_coordinate_frame)  # 配准后的源点云坐标系
    vis.add_geometry(aligned_target_coordinate_frame)  # 配准后的目标点云坐标系
    vis.add_geometry(all_coordinate_frame)  # 添加目标点云坐标系


    # 开始可视化
    vis.run()
    vis.destroy_window()


if __name__ == '__main__':
    main()