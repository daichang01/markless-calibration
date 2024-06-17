import open3d as o3d
import numpy as np

def mean_filter_point_cloud(point_cloud, nb_neighbors=20):
    """
    对点云应用均值滤波。
    
    Args:
    - point_cloud: 输入的点云数据（open3d.geometry.PointCloud）
    - nb_neighbors: 每个点考虑的邻居数
    
    Returns:
    - filtered_point_cloud: 经过均值滤波处理后的点云（open3d.geometry.PointCloud）
    """
    # 计算每个点的邻域
    kdtree = o3d.geometry.KDTreeFlann(point_cloud)
    points = np.asarray(point_cloud.points)
    filtered_points = np.zeros_like(points)

    for i in range(points.shape[0]):
        [_, idx, _] = kdtree.search_knn_vector_3d(point_cloud.points[i], nb_neighbors)
        neighbor_points = points[idx, :]
        filtered_points[i, :] = np.mean(neighbor_points, axis=0)

    filtered_point_cloud = o3d.geometry.PointCloud()
    filtered_point_cloud.points = o3d.utility.Vector3dVector(filtered_points)
    return filtered_point_cloud

def main():
    # 加载点云
    point_cloud = o3d.io.read_point_cloud("point_cloud.pcd")

    # 应用均值滤波
    filtered_point_cloud = mean_filter_point_cloud(point_cloud, nb_neighbors=20)

    # 可视化
    o3d.visualization.draw_geometries([point_cloud], window_name='Original Point Cloud')
    o3d.visualization.draw_geometries([filtered_point_cloud], window_name='Filtered Point Cloud')

if __name__ == "__main__":
    main()
