import rerun as rr
import h5py
import numpy as np

with h5py.File("proprio_stats.h5", "r") as f:
    timestamps = f['timestamp'][:] # <class 'numpy.ndarray'> [1742793443099683000 1742793443133031000 1742793443166378000...]
    num_timesteps = len(timestamps)
    print(f"时间戳数量: {num_timesteps}")
    
    # 初始化 Rerun
    rr.init("zhiyuan_robot_data", spawn=True)
    
    # 定义要渲染的数据结构
    data_groups = ['action', 'state']
    component_types = ['effector', 'end', 'head', 'joint', 'robot', 'waist']
    
    # 收集所有position数据和对应的时间索引
    position_data = {}
    
    for group in data_groups:
        if group in f:
            position_data[group] = {}
            for component in component_types:
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