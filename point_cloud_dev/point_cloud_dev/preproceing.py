import numpy as np
import open3d as o3d
import os


def read_point_cloud_with_scalar(file_path):
    # 读取点云数据和标量字段
    data = np.loadtxt(file_path)
    points = data[:, 0:3]
    colors = data[:, 3:6] / 255.0  # 假设颜色值需要从0-255转换到0-1
    scalars = data[:, -1]  # 假设标量字段在最后一列

    # 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd, scalars

def remove_noise(pcd):
    # 创建一个去噪对象
    sor = o3d.geometry.PointCloud.remove_statistical_outlier(pcd, nb_neighbors=20, std_ratio=2.0)
    # 应用去噪
    filtered_pcd, _ = sor
    return filtered_pcd

def uniform_downsample(pcd, every_k_points):
    """均匀下采样 每k个点选择一个点"""
    # 从点云中获取所有点
    all_points = np.asarray(pcd.points)
    all_colors = np.asarray(pcd.colors) if pcd.has_colors() else None

    # 每k个点选择一个点
    indices = np.arange(0, len(all_points), every_k_points)
    downsampled_points = all_points[indices, :]
    
    # 创建新的点云对象
    down_pcd = o3d.geometry.PointCloud()
    down_pcd.points = o3d.utility.Vector3dVector(downsampled_points)
    
    # 如果点云中有颜色信息，同样进行下采样
    if all_colors is not None:
        downsampled_colors = all_colors[indices, :]
        down_pcd.colors = o3d.utility.Vector3dVector(downsampled_colors)

    return down_pcd, indices



def compute_normals(pcd):
    # 计算点云的法线
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    # 确保法线的方向是正确的,可省略
    pcd.orient_normals_towards_camera_location(pcd.get_center())

#保存新格式的点云数据
def save_point_cloud_with_normals_and_scalar(pcd, scalars, file_path):
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    # 假设标量字段已经是正确的格式，直接合并
    data_to_save = np.hstack((points, normals, scalars.reshape(-1, 1)))
    # np.savetxt(file_path, data_to_save, fmt='%f', header='X Y Z Nx Ny Nz Scalar', comments='')
    np.savetxt(file_path, data_to_save, fmt='%f', comments='')

def process_folder(input_folder, output_folder):
    # 获取输入文件夹中所有文件的列表
    files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
    # 对文件列表进行排序，确保按字母顺序处理，这有助于保持处理的一致性和可追溯性
    files.sort()
    # 遍历文件列表，处理每个文件
    for index, filename in enumerate(files):
        print(f"Processing {filename}...")  # 打印正在处理的文件名，提供用户反馈
        input_path = os.path.join(input_folder, filename)  # 拼接完整的输入文件路径
        output_path = os.path.join(output_folder, f"normals-{index + 1}.txt")  # 拼接完整的输出文件路径
        # 读取点云数据和相关标量值
        pcd, scalars = read_point_cloud_with_scalar(input_path)    
        # 对点云数据应用去噪处理
        # pcd = remove_noise(pcd)
        # 应用均匀下采样
        pcd, selected_indices = uniform_downsample(pcd, every_k_points=10)
        updated_scalars = scalars[selected_indices]  # 更新标量字段
        # 计算点云的表面法线，这对许多点云处理任务是必需的，如表面重建和特征提取
        compute_normals(pcd)
        # 将处理后的点云数据及其法线和标量值保存到文件中
        save_point_cloud_with_normals_and_scalar(pcd, updated_scalars, output_path)


def main():
    input_folder = "/home/daichang/Desktop/ros2_ws/src/markless-calibration/pcd/03handle_seg"
    output_folder = "/home/daichang/Desktop/ros2_ws/src/markless-calibration/pcd/04label_normals"
    process_folder(input_folder, output_folder)

if __name__ == '__main__':
    main()
