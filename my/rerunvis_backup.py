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
    
    # 收集所有position数据
    position_data = {}
    
    for group in data_groups:
        if group in f:
            position_data[group] = {}
            for component in component_types:
                if component in f[group] and 'position' in f[group][component]:
                    data = f[group][component]['position'][:]
                    print(f"检查数据: {group}/{component}/position, 形状: {data.shape}")
                    
                    # 验证数据不为空且第一维与时间戳匹配
                    if data.size > 0 and len(data) == num_timesteps:
                        position_data[group][component] = data
                        print(f"✓ 加载数据: {group}/{component}/position, 形状: {data.shape}")
                    else:
                        print(f"✗ 跳过数据: {group}/{component}/position, 形状: {data.shape} (空数组或长度不匹配)")
    
    # 按时间步记录所有数据
    for t in range(num_timesteps):
        # 将纳秒时间戳转换为秒（除以1e9）
        timestamp_seconds = timestamps[t] / 1e9
        rr.set_time("real_time", timestamp=timestamp_seconds)
        
        # 遍历所有数据组和组件
        for group in position_data:
            if group == 'action':
                continue
            for component in position_data[group]:
                data = position_data[group][component]
                
                # 根据数据维度决定如何记录
                if len(data.shape) == 2:  # (timesteps, joints/dimensions)
                    num_elements = data.shape[1]
                    for j in range(num_elements):
                        rr.log(f"{group}/{component}/element_{j}", rr.Scalars(data[t, j]))
                elif len(data.shape) == 1:  # (timesteps,) - 单个值
                    rr.log(f"{group}/{component}/value", rr.Scalars(data[t]))
                else:  # 其他维度，展平处理
                    flat_data = data[t].flatten()
                    for j, val in enumerate(flat_data):
                        rr.log(f"{group}/{component}/dim_{j}", rr.Scalars(val))


print("数据渲染完成！在 Rerun 查看器中查看结果")