import open3d as o3d
import numpy as np
import time
import rclpy
import struct
from rclpy.node import Node
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as pc2
from scipy.interpolate import interp1d, splprep, splev

############################################## utils ##############################################
def visualize_initial_point_clouds(pc1, pc2, window_name='untitle', width=1000, height=800):
    # Set colors for point clouds
    pc1.paint_uniform_color([1, 0, 0])  # red color for the first point cloud
    pc2.paint_uniform_color([0, 1, 0])  # green

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

def pointcloud2_to_open3d(pointcloud2_msg):
    points_list = list(pc2.read_points(pointcloud2_msg, field_names=("x", "y", "z"), skip_nans=True))
    if not points_list:
        return None
    # 直接提取 x, y, z 数据
    points = np.array([[p[0], p[1], p[2]] for p in points_list], dtype=np.float32)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    return point_cloud

def evaluate_registration(source, target, transformation, threshold):
    evaluation = o3d.pipelines.registration.evaluate_registration(
        source, target,max_correspondence_distance= threshold, transformation= transformation)
    fitness = evaluation.fitness # 重叠区域（内部对应数/源中的点数）。越高越好。
    inlier_rmse = evaluation.inlier_rmse # 所有内部对应关系的 RMSE。越低越好。
    return fitness, inlier_rmse

def convert_to_pointcloud2(point_cloud):
    points = np.asarray(point_cloud.points)
    if point_cloud.colors:
        colors = (np.asarray(point_cloud.colors) * 255).astype(np.uint8)
    else:
        colors = np.zeros((points.shape[0], 3), dtype=np.uint8)

    header = Header()
    header.stamp = rclpy.time.Time().to_msg()
    header.frame_id = 'camera_infra1_optical_frame'

    fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name='r', offset=12, datatype=PointField.UINT8, count=1),
        PointField(name='g', offset=13, datatype=PointField.UINT8, count=1),
        PointField(name='b', offset=14, datatype=PointField.UINT8, count=1),
    ]

    cloud_data = []
    for i in range(points.shape[0]):
        x, y, z = points[i]
        r, g, b = colors[i]
        cloud_data.append(struct.pack('fffBBB', x, y, z, r, g, b))

    cloud_data = b''.join(cloud_data)
    return PointCloud2(header=header, height=1, width=points.shape[0], fields=fields, is_bigendian=False, point_step=15, row_step=15 * points.shape[0], data=cloud_data, is_dense=True)


def linear_interpolation(pcd, num_points):
    """
    对点云进行线性插值

    Args:
        pcd (open3d.geometry.PointCloud): 输入点云
        num_points (int): 插值后的点云包含的点的数量

    Returns:
        open3d.geometry.PointCloud: 插值后的点云
    """
    # 提取点云坐标
    points = np.asarray(pcd.points)
    
    # 确保点云按某个维度排序，例如按x坐标
    sorted_indices = np.argsort(points[:, 0])
    points = points[sorted_indices]

    # 原始点的x坐标
    x_original = points[:, 0]
    
    # 创建新的x坐标，均匀分布在原始x坐标范围内
    x_new = np.linspace(x_original.min(), x_original.max(), num_points)
    
    # 对每个维度进行线性插值
    interpolated_points = []
    for i in range(points.shape[1]):
        f = interp1d(x_original, points[:, i], kind='linear')
        interpolated_points.append(f(x_new))
    
    interpolated_points = np.stack(interpolated_points, axis=-1)
    
    # 创建新的点云
    interpolated_pcd = o3d.geometry.PointCloud()
    interpolated_pcd.points = o3d.utility.Vector3dVector(interpolated_points)
    
    return interpolated_pcd



def spline_interpolation(pcd, num_points):
    """
    对点云进行样条插值

    Args:
        pcd (open3d.geometry.PointCloud): 输入点云
        num_points (int): 插值后的点云包含的点的数量

    Returns:
        open3d.geometry.PointCloud: 插值后的点云
    """
    points = np.asarray(pcd.points)
    
    if len(points) < 2:
        raise ValueError("点云中点的数量太少，无法进行插值")

    # 去除重复点
    points = np.unique(points, axis=0)
    
    # 确保点云按某个维度排序，例如按x坐标
    sorted_indices = np.argsort(points[:, 0])
    points = points[sorted_indices]

    # 原始点的x坐标
    x_original = points[:, 0]

    # 创建新的x坐标，均匀分布在原始x坐标范围内
    x_new = np.linspace(x_original.min(), x_original.max(), num_points)

    try:
        # 对每个维度进行样条插值
        tck, u = splprep([points[:, 0], points[:, 1], points[:, 2]], s=0)
        new_points = splev(np.linspace(0, 1, num_points), tck)
    except Exception as e:
        raise ValueError(f"样条插值失败: {e}")

    interpolated_points = np.vstack(new_points).T

    # 创建新的点云
    interpolated_pcd = o3d.geometry.PointCloud()
    interpolated_pcd.points = o3d.utility.Vector3dVector(interpolated_points)

    return interpolated_pcd

