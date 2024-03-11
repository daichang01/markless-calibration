import open3d as o3d

def showPCD(pcd_path):
    # 加载PLY文件
    point_cloud = o3d.io.read_point_cloud(pcd_path)

    # 显示点云
    # 使用draw_geometries来显示点云，点的大小默认是不可调的，因为它依赖于OpenGL的点渲染
    # o3d.visualization.draw_geometries([point_cloud])

    # 创建可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # 将点云添加到可视化窗口
    vis.add_geometry(point_cloud)

    # 获取渲染选项
    opt = vis.get_render_option()
    opt.point_size = 1  # 设置点的大小

    # 运行可视化
    vis.run()
    vis.destroy_window()

if __name__ == '__main__':
    showPCD("src/markless-calibration/pcd/output_point_cloud.ply")