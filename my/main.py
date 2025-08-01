import h5py

with h5py.File("proprio_stats.h5", "r") as f:
    # action动作  state状态
    print(f.keys()) # 结果 <KeysViewHDF5 ['action', 'state', 'timestamp']>
    print(f["state"].keys()) # 结果 <KeysViewHDF5 ['effector', 'end', 'head', 'joint', 'robot', 'waist']>
    print(f['action'].keys()) # 结果 <KeysViewHDF5 ['effector', 'end', 'head', 'joint', 'robot', 'waist']>
    print(f['action']['effector']['position'].shape) # (1378, 2)


    print(f['timestamp']) # 结果 <HDF5 dataset "timestamp": shape (1378,), type "<i8">
    print(f['state']['effector'].keys()) # 结果 <KeysViewHDF5 ['force', 'position']>
    print(f['state']['joint'].keys()) # 结果 <KeysViewHDF5 ['current_value', 'effort', 'position', 'velocity']>
    print(f['state']['joint']['position'].shape) #  形状: (1378, 14)

    # 如果有 'position' 数据集，打印其形状和前几个值
    if 'position' in f['state']['joint']:
        joint_data = f['state']['joint']['position'][:]
        print("state/joint/position 形状:", joint_data.shape)
        print("前5个时间步的数据:", joint_data[:5])

