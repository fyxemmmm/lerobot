import rerun as rr
import h5py
import numpy as np

with h5py.File("proprio_stats.h5", "r") as f:
    timestamps = f['timestamp'][:] # <class 'numpy.ndarray'> [1742793443099683000 1742793443133031000 1742793443166378000...]
    print(len(timestamps))
    # exit()
   
    # 检查并加载关节数据
    if 'position' in f['state']['joint']:
        joint_positions = f['state']['joint']['position'][:]
        num_timesteps, num_joints = joint_positions.shape
        print(f"关节数据形状: {num_timesteps} 时间步, {num_joints} 个关节")

        # 初始化 Rerun
        rr.init("robot_joint_data", spawn=True)
        
        # 按时间步记录每个关节的位置
        for t in range(num_timesteps):
            # 将纳秒时间戳转换为秒（除以1e9）
            timestamp_seconds = timestamps[t] / 1e9
            rr.set_time("real_time", timestamp=timestamp_seconds)
            for j in range(num_joints):
                # 使用新的API：Scalars替代废弃的Scalar
                rr.log(f"joints/joint_{j}/position", rr.Scalars(joint_positions[t, j]))

# 在 Rerun 查看器中查看结果