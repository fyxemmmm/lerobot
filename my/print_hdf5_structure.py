import h5py
import numpy as np

def print_hdf5_structure(item, indent=0):
    """递归打印HDF5文件的结构"""
    prefix = "  " * indent
    
    if isinstance(item, h5py.Group):
        print(f"{prefix}📁 Group: {item.name}")
        for key in item.keys():
            print(f"{prefix}├── {key}")
            print_hdf5_structure(item[key], indent + 1)
    elif isinstance(item, h5py.Dataset):
        print(f"{prefix}📄 Dataset: {item.name}")
        print(f"{prefix}    Shape: {item.shape}")
        print(f"{prefix}    Type: {item.dtype}")
        print(f"{prefix}    Size: {item.size}")
        
        # 如果数据集不太大，显示一些统计信息
        if item.size > 0 and item.size < 10000:
            try:
                data = item[:]
                if np.issubdtype(item.dtype, np.number):
                    print(f"{prefix}    Range: [{np.min(data):.6f}, {np.max(data):.6f}]")
                    if data.size <= 10:
                        print(f"{prefix}    Data: {data}")
                    else:
                        print(f"{prefix}    First 3: {data.flat[:3]}")
                        print(f"{prefix}    Last 3: {data.flat[-3:]}")
            except Exception as e:
                print(f"{prefix}    Error reading data: {e}")

def print_full_hdf5_tree(filename):
    """打印完整的HDF5文件树结构"""
    print(f"🗂️  HDF5 文件结构: {filename}")
    print("=" * 60)
    
    with h5py.File(filename, "r") as f:
        print(f"Root group keys: {list(f.keys())}")
        print("\n详细结构:")
        print_hdf5_structure(f)
        
        print("\n" + "=" * 60)
        print("📊 数据摘要:")
        
        # 打印一些关键统计信息
        if 'timestamp' in f:
            timestamps = f['timestamp'][:]
            print(f"⏰ 时间戳: {len(timestamps)} 个时间点")
            print(f"   时间范围: {timestamps[0]} 到 {timestamps[-1]}")
            
        if 'state' in f and 'joint' in f['state'] and 'position' in f['state']['joint']:
            joint_pos = f['state']['joint']['position']
            print(f"🤖 关节位置: {joint_pos.shape[0]} 时间步, {joint_pos.shape[1]} 个关节")
            
        if 'action' in f and 'joint' in f['action'] and 'position' in f['action']['joint']:
            action_pos = f['action']['joint']['position']
            print(f"🎯 动作指令: {action_pos.shape[0]} 时间步, {action_pos.shape[1]} 个关节")

if __name__ == "__main__":
    print_full_hdf5_tree("proprio_stats.h5")