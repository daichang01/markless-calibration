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
from scipy.spatial import cKDTree
import pyrealsense2 as rs
from datetime import datetime
import os


############################################## utils ##############################################
def getintrinsic():
    # 设置和获取内参
    pipeline = rs.pipeline()
    config = rs.config()
    # config.enable_device('你的设备ID')  # 如有必要
    # config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
    # config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    # 启动管道并获取内参
    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    # 获取内参
    intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
    fx = intrinsics.fx  # x轴焦距
    fy = intrinsics.fy  # y轴焦距
    cx = intrinsics.ppx  # x轴光学中心
    cy = intrinsics.ppy  # y轴光学中心
    print(f"fx: {fx}, fy: {fy}, cx: {cx}, cy: {cy}")

    # 在点云生成中使用这些内参
    # x = (u - cx) * z / fx
    # y = (v - cy) * z / fy


def visualize_initial_point_clouds(pc1, pc2, window_name='untitle', width=1000, height=800):
    # Set colors for point clouds
    pc1.paint_uniform_color([1, 0, 0])  # red color for the first point cloud
    pc2.paint_uniform_color([0, 1, 0])  # green

    # Create coordinate frames
    axis_pc = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.001)
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
    # 从文件加载点云数据。假设数据格式为：X Y Z R G B
    data = np.loadtxt(file_path)
    
    # 从前三列提取坐标（X, Y, Z）
    points = data[:, :3]
    
    # 如果存在，从第四到第六列提取颜色数据（R, G, B），并将颜色标准化到 [0, 1] 范围
    colors = data[:, 3:6] / 255.0 if data.shape[1] > 3 else None
    
    # 创建一个空的Open3D点云对象
    point_cloud = o3d.geometry.PointCloud()
    
    # 将提取的坐标点赋值给点云对象的点属性
    point_cloud.points = o3d.utility.Vector3dVector(points)
    
    # 如果存在颜色信息，也将颜色信息赋值给点云对象的颜色属性
    if colors is not None:
        point_cloud.colors = o3d.utility.Vector3dVector(colors)
    
    # 返回装载好的点云对象
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


def calculate_rmse(source_points, target_points):
        # 使用 float64 确保高精度计算
    source_points = np.asarray(source_points, dtype=np.float64)
    target_points = np.asarray(target_points, dtype=np.float64)
    # 创建目标点云的KD树
    tree = cKDTree(target_points)
    # 查询源点云中每个点在目标点云中的最近邻点
    distances, indices = tree.query(source_points, k=1)
    # 找到每个源点云点对应的最近的目标点云点
    nearest_target_points = target_points[indices]
    # 计算源点云和最近的目标点云点之间的均方误差 (MSE)
    mse = np.mean((source_points - nearest_target_points)**2)
    rmse = np.sqrt(mse)
    return rmse * 1000

def calculate_overlap_ratio(source_points, target_points, threshold=0.001):
    # 创建目标点云的KD树
    tree = cKDTree(target_points)
    # 查询源点云中每个点在目标点云中的最近邻点的距离
    distances, _ = tree.query(source_points, k=1)
    # 计算源点云中距离目标点云最近点距离小于阈值的点的数量
    overlap_count = np.sum(distances < threshold)
    # 计算重叠率，即重叠点的数量除以源点云的总点数
    return overlap_count / len(source_points)
def evaluate_registration(source, target, transformation, threshold= 0.02):
    # https://www.open3d.org/docs/latest/python_api/open3d.pipelines.registration.RegistrationResult.html
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

def transform_points(points, R, t):
    transformed_points = np.dot(points, R.T) + t
    return transformed_points

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

def save_point_cloud_to_txt(filepath, point_cloud):
    current_time = datetime.now().strftime("%m%d_%H%M%S")
    save_path= f"{filepath}_{current_time}.txt"

    # # 检查路径中的目录是否存在，不存在则创建
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    # 将点云保存为TXT文件格式
    with open(save_path, 'w') as f:
        for point in point_cloud.points:
            f.write(f"{point[0]} {point[1]} {point[2]}\n")
    print(f"Point cloud saved to {save_path}")


def main():
    getintrinsic()

if __name__ == '__main__':
    main()