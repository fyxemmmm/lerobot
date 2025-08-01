import rerun as rr
import h5py
import numpy as np
import cv2

with h5py.File("proprio_stats.h5", "r") as f:
    timestamps = f['timestamp'][:] # <class 'numpy.ndarray'> [1742793443099683000 1742793443133031000 1742793443166378000...]
    num_timesteps = len(timestamps)
    print(f"时间戳数量: {num_timesteps}")
    
    # 计算数据时间范围
    start_time = timestamps[0] / 1e9  # 转换为秒
    end_time = timestamps[-1] / 1e9
    data_duration = end_time - start_time
    print(f"数据时间范围: {data_duration:.2f} 秒")
    
    rr.init("zhiyuan_robot_data", spawn=True)
    
    # 打开视频文件
    video_path = "cameras/hand_left_color.mp4"
    cap = cv2.VideoCapture(video_path)
    
    video_frames = []  # 初始化空的视频帧列表
    
    if not cap.isOpened():
        print(f"错误：无法打开视频文件 {video_path}")
        print("将只渲染数据，不包含视频")
    else:
        # 获取视频信息
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = video_frame_count / video_fps if video_fps > 0 else 0
        
        print(f"视频信息:")
        print(f"  帧率: {video_fps} FPS")
        print(f"  总帧数: {video_frame_count}")
        print(f"  时长: {video_duration:.2f} 秒")
        
        # 预加载所有视频帧（可选，用于更好的性能）
        video_frames = []
        print("正在预加载视频帧...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # 转换BGR到RGB（OpenCV默认BGR，rerun期望RGB）
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_frames.append(frame_rgb)
        
        cap.release()
        print(f"视频帧加载完成: {len(video_frames)} 帧")
    
    # 定义要渲染的数据结构
    data_groups = ['action', 'state']
    component_types = ['effector', 'end', 'head', 'joint', 'robot', 'waist']
    
    # 收集所有position数据和对应的时间索引
    position_data = {}
    
    for group in data_groups:
        if group in f:
            position_data[group] = {}
            for component in component_types:
                # if group == 'state' and component == 'effector':
                #     continue
                if component in f[group] and 'position' in f[group][component]:
                    data = f[group][component]['position'][:]
                    print(f"检查数据: {group}/{component}/position, 形状: {data.shape}")
                    
                    # 验证数据不为空
                    if data.size > 0:
                        # 获取时间索引（如果有的话）
                        time_indices = None
                        if 'index' in f[group][component]:
                            time_indices = f[group][component]['index'][:]
                            print(f"  找到时间索引: 长度{len(time_indices)}, 范围{time_indices[0]}-{time_indices[-1]}")
                        
                        position_data[group][component] = {
                            'data': data,
                            'time_indices': time_indices
                        }
                        print(f"✓ 加载数据: {group}/{component}/position, 形状: {data.shape}")
                    else:
                        print(f"✗ 跳过数据: {group}/{component}/position, 形状: {data.shape} (空数组)")
    
    # 按时间步记录所有数据
    for t in range(num_timesteps):
        # 将纳秒时间戳转换为秒（除以1e9）
        timestamp_seconds = timestamps[t] / 1e9
        rr.set_time("real_time", timestamp=timestamp_seconds)
        
        # 渲染对应的视频帧（如果视频已加载）
        if len(video_frames) > 0:
            # 计算当前时间步对应的视频帧索引，使视频从第0帧平铺到最后一帧
            frame_idx = int(t * len(video_frames) / num_timesteps)
            # 防止越界
            frame_idx = min(frame_idx, len(video_frames) - 1)
            frame = video_frames[frame_idx]
            rr.log("camera/hand_left", rr.Image(frame))
        
        # 遍历所有数据组和组件
        for group in position_data:
            for component in position_data[group]:
                component_info = position_data[group][component]
                data = component_info['data']
                time_indices = component_info['time_indices']
                
                # 确定当前时间步对应的数据索引
                data_index = None
                
                if time_indices is not None:
                    # 有时间索引，需要检查当前时间步是否在索引中
                    if t in time_indices:
                        # 找到时间索引t在time_indices中的位置
                        data_index = np.where(time_indices == t)[0][0]
                else:
                    # 没有时间索引，直接使用时间步作为数据索引（如果在范围内）
                    if t < len(data):
                        data_index = t
                
                # 如果找到了有效的数据索引，记录数据
                if data_index is not None:
                    # 根据数据维度决定如何记录
                    if len(data.shape) == 2:  # (timesteps, joints/dimensions)
                        num_elements = data.shape[1]
                        for j in range(num_elements):
                            rr.log(f"{group}/{component}/element_{j}", rr.Scalars(data[data_index, j]))
                    elif len(data.shape) == 1:  # (timesteps,) - 单个值
                        rr.log(f"{group}/{component}/value", rr.Scalars(data[data_index]))
                    else:  # 其他维度，展平处理
                        flat_data = data[data_index].flatten()
                        for j, val in enumerate(flat_data):
                            rr.log(f"{group}/{component}/dim_{j}", rr.Scalars(val))


print("数据渲染完成！在 Rerun 查看器中查看结果")