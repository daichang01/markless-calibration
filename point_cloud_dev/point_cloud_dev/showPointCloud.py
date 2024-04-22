import numpy as np
import open3d as o3d

def read_xyzrgb_point_cloud(file_path):
    # 加载点云文件，假定文件格式为：X Y Z R G B，其中RGB是0-255范围
    data = np.loadtxt(file_path)
    points = data[:, 0:3]  # XYZ坐标
    colors = data[:, 3:6] / 255.0  # 将RGB从0-255缩放到0-1

    # 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd

def visualize_point_cloud(pcd):
    # 可视化点云
    o3d.visualization.draw_geometries([pcd])

def main():
    file_path = '/home/daichang/Desktop/ros2_ws/src/markless-calibration/pcd/origin_data/ori1.txt'  
    pcd = read_xyzrgb_point_cloud(file_path)
    visualize_point_cloud(pcd)

if __name__ == '__main__':
    main()
