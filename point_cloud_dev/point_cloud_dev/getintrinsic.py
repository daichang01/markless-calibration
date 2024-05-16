import pyrealsense2 as rs

# 设置和获取内参
pipeline = rs.pipeline()
config = rs.config()
# config.enable_device('你的设备ID')  # 如有必要
config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)

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
