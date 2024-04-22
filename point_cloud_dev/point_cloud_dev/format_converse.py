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


def compute_normals(pcd):
    # 计算点云的法线
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    # 确保法线的方向是正确的，这一步可根据需要省略
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
    files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
    files.sort()  # Ensure files are processed in alphabetical order
    for index, filename in enumerate(files):
        print(f"Processing {filename}...")
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, f"normals-{index + 1}.txt")
        pcd, scalars = read_point_cloud_with_scalar(input_path)
        compute_normals(pcd)
        save_point_cloud_with_normals_and_scalar(pcd, scalars, output_path)

def main():
    input_folder = "/home/daichang/Desktop/ros2_ws/src/markless-calibration/pcd/label_data"
    output_folder = "/home/daichang/Desktop/ros2_ws/src/markless-calibration/pcd/label_normals"
    process_folder(input_folder, output_folder)

if __name__ == '__main__':
    main()
